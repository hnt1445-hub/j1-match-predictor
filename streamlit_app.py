import streamlit as st
import pandas as pd
import numpy as np

from collections import defaultdict, deque
from math import exp, factorial

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor


# =========================================================
# 基本設定
# =========================================================

st.set_page_config(
    page_title="J1 Goal Predictor G5",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ J1 Goal Predictor - G5")

st.write(
    "G5 = G2 + 対戦相手の強さを考慮して更新する "
    "動的Attack / Defense Rating"
)


# =========================================================
# データ
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("data/JPN.csv")

    df = df[
        df["League"] == "J1 League"
    ].copy()

    df["Season"] = df["Season"].astype(str)

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Date",
            "Home",
            "Away",
            "HG",
            "AG"
        ]
    ).copy()

    df["OriginalOrder"] = range(len(df))

    return (
        df
        .sort_values(
            ["Date", "OriginalOrder"]
        )
        .reset_index(drop=True)
    )


matches = load_data()


# =========================================================
# 補助関数
# =========================================================

def average(values):

    if len(values) == 0:
        return np.nan

    return sum(values) / len(values)


# =========================================================
# 特徴量作成
# =========================================================

@st.cache_data
def build_features(matches):

    # -------------------------
    # Elo
    # -------------------------

    elo = defaultdict(
        lambda: 1500.0
    )

    # -------------------------
    # 最近の成績
    # -------------------------

    recent_points_5 = defaultdict(
        lambda: deque(maxlen=5)
    )

    recent_gf_10 = defaultdict(
        lambda: deque(maxlen=10)
    )

    recent_ga_10 = defaultdict(
        lambda: deque(maxlen=10)
    )

    home_gf_10 = defaultdict(
        lambda: deque(maxlen=10)
    )

    home_ga_10 = defaultdict(
        lambda: deque(maxlen=10)
    )

    away_gf_10 = defaultdict(
        lambda: deque(maxlen=10)
    )

    away_ga_10 = defaultdict(
        lambda: deque(maxlen=10)
    )

    # -------------------------
    # G5
    #
    # 1.0 = リーグ平均
    #
    # Attack > 1
    #   得点力が平均より高い
    #
    # Defense < 1
    #   失点しにくい
    # -------------------------

    attack = defaultdict(
        lambda: 1.0
    )

    defense = defaultdict(
        lambda: 1.0
    )

    # Rating更新速度
    LEARNING_RATE = 0.08

    # Rating暴走防止
    MIN_RATING = 0.50
    MAX_RATING = 1.80

    K_FACTOR = 20

    rows = []

    for _, match in matches.iterrows():

        home = match["Home"]
        away = match["Away"]

        hg = float(match["HG"])
        ag = float(match["AG"])

        # =============================================
        # 試合前Rating
        # =============================================

        home_elo = elo[home]
        away_elo = elo[away]

        home_attack = attack[home]
        home_defense = defense[home]

        away_attack = attack[away]
        away_defense = defense[away]

        home_form5 = sum(
            recent_points_5[home]
        )

        away_form5 = sum(
            recent_points_5[away]
        )

        rows.append(
            {
                "Season": str(match["Season"]),
                "Date": match["Date"],

                "Home": home,
                "Away": away,

                "HG": int(hg),
                "AG": int(ag),

                # Elo
                "Home_Elo": home_elo,
                "Away_Elo": away_elo,
                "EloDiff":
                    home_elo - away_elo,

                # Form
                "Home_Form5":
                    home_form5,

                "Away_Form5":
                    away_form5,

                "FormDiff":
                    home_form5
                    - away_form5,

                # 最近10試合
                "Home_Recent10_GF":
                    average(
                        recent_gf_10[home]
                    ),

                "Home_Recent10_GA":
                    average(
                        recent_ga_10[home]
                    ),

                "Away_Recent10_GF":
                    average(
                        recent_gf_10[away]
                    ),

                "Away_Recent10_GA":
                    average(
                        recent_ga_10[away]
                    ),

                # Home / Away別
                "Home_Home10_GF":
                    average(
                        home_gf_10[home]
                    ),

                "Home_Home10_GA":
                    average(
                        home_ga_10[home]
                    ),

                "Away_Away10_GF":
                    average(
                        away_gf_10[away]
                    ),

                "Away_Away10_GA":
                    average(
                        away_ga_10[away]
                    ),

                # G5
                "Home_AttackRating":
                    home_attack,

                "Home_DefenseRating":
                    home_defense,

                "Away_AttackRating":
                    away_attack,

                "Away_DefenseRating":
                    away_defense,

                "AttackAdvantage":
                    home_attack
                    - away_attack,

                "DefenseAdvantage":
                    away_defense
                    - home_defense
            }
        )

        # =============================================
        # 試合後更新
        #
        # ここより上だけが予測に使われるので、
        # 当該試合の結果漏洩はありません。
        # =============================================

        # ---------------------------------------------
        # 勝点
        # ---------------------------------------------

        if hg > ag:

            hp = 3
            ap = 0
            home_actual = 1.0

        elif hg < ag:

            hp = 0
            ap = 3
            home_actual = 0.0

        else:

            hp = 1
            ap = 1
            home_actual = 0.5

        recent_points_5[
            home
        ].append(hp)

        recent_points_5[
            away
        ].append(ap)

        # ---------------------------------------------
        # 得失点履歴
        # ---------------------------------------------

        recent_gf_10[
            home
        ].append(hg)

        recent_ga_10[
            home
        ].append(ag)

        recent_gf_10[
            away
        ].append(ag)

        recent_ga_10[
            away
        ].append(hg)

        home_gf_10[
            home
        ].append(hg)

        home_ga_10[
            home
        ].append(ag)

        away_gf_10[
            away
        ].append(ag)

        away_ga_10[
            away
        ].append(hg)

        # =============================================
        # G5 Attack / Defense Rating更新
        #
        # 予想得点の基準：
        #
        # Home expected
        # = Home Attack × Away Defense
        #
        # Away expected
        # = Away Attack × Home Defense
        #
        # Defenseが大きいほど失点しやすい
        # =============================================

        expected_home_goals = (
            home_attack
            *
            away_defense
        )

        expected_away_goals = (
            away_attack
            *
            home_defense
        )

        # ---------------------------------------------
        # 実際 / 期待 の比率
        #
        # 0点でも極端にならないよう +0.5
        # ---------------------------------------------

        home_ratio = (
            (hg + 0.5)
            /
            (expected_home_goals + 0.5)
        )

        away_ratio = (
            (ag + 0.5)
            /
            (expected_away_goals + 0.5)
        )

        # ---------------------------------------------
        # Attack更新
        # ---------------------------------------------

        new_home_attack = (
            home_attack
            *
            (
                home_ratio
                **
                LEARNING_RATE
            )
        )

        new_away_attack = (
            away_attack
            *
            (
                away_ratio
                **
                LEARNING_RATE
            )
        )

        # ---------------------------------------------
        # Defense更新
        #
        # 相手に予想以上に点を取られた
        # → Defense値が上昇
        # → 守備が弱い方向
        # ---------------------------------------------

        new_home_defense = (
            home_defense
            *
            (
                away_ratio
                **
                LEARNING_RATE
            )
        )

        new_away_defense = (
            away_defense
            *
            (
                home_ratio
                **
                LEARNING_RATE
            )
        )

        attack[home] = np.clip(
            new_home_attack,
            MIN_RATING,
            MAX_RATING
        )

        attack[away] = np.clip(
            new_away_attack,
            MIN_RATING,
            MAX_RATING
        )

        defense[home] = np.clip(
            new_home_defense,
            MIN_RATING,
            MAX_RATING
        )

        defense[away] = np.clip(
            new_away_defense,
            MIN_RATING,
            MAX_RATING
        )

        # =============================================
        # Elo更新
        # =============================================

        expected_home = (
            1
            /
            (
                1
                +
                10
                **
                (
                    (
                        away_elo
                        - home_elo
                    )
                    / 400
                )
            )
        )

        elo[home] = (
            home_elo
            +
            K_FACTOR
            *
            (
                home_actual
                - expected_home
            )
        )

        elo[away] = (
            away_elo
            +
            K_FACTOR
            *
            (
                (1.0 - home_actual)
                -
                (1.0 - expected_home)
            )
        )

    return pd.DataFrame(rows)


data = build_features(matches)


# =========================================================
# 結果
# =========================================================

def get_result(row):

    if row["HG"] > row["AG"]:
        return "H"

    if row["HG"] < row["AG"]:
        return "A"

    return "D"


data["Result"] = data.apply(
    get_result,
    axis=1
)


# =========================================================
# G2特徴量
# =========================================================

G2_NUMERIC = [

    "Home_Elo",
    "Away_Elo",
    "EloDiff",

    "Home_Form5",
    "Away_Form5",
    "FormDiff",

    "Home_Recent10_GF",
    "Home_Recent10_GA",

    "Away_Recent10_GF",
    "Away_Recent10_GA",

    "Home_Home10_GF",
    "Home_Home10_GA",

    "Away_Away10_GF",
    "Away_Away10_GA"
]


# =========================================================
# G5特徴量
# =========================================================

G5_NUMERIC = (
    G2_NUMERIC
    +
    [
        "Home_AttackRating",
        "Home_DefenseRating",

        "Away_AttackRating",
        "Away_DefenseRating",

        "AttackAdvantage",
        "DefenseAdvantage"
    ]
)


CATEGORICAL = [
    "Home",
    "Away"
]


# =========================================================
# モデル作成
# =========================================================

def make_model(
    numeric_features
):

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_features
            ),
            (
                "team",
                categorical_transformer,
                CATEGORICAL
            )
        ]
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "poisson",
                PoissonRegressor(
                    alpha=0.1,
                    max_iter=1000
                )
            )
        ]
    )


# =========================================================
# Poisson
# =========================================================

def poisson_probability(
    goals,
    expected_goals
):

    return (
        exp(-expected_goals)
        *
        expected_goals ** goals
        /
        factorial(goals)
    )


def match_probabilities(
    home_lambda,
    away_lambda,
    max_goals=10
):

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    for hg in range(
        max_goals + 1
    ):

        ph = poisson_probability(
            hg,
            home_lambda
        )

        for ag in range(
            max_goals + 1
        ):

            pa = poisson_probability(
                ag,
                away_lambda
            )

            p = ph * pa

            if hg > ag:
                home_win += p

            elif hg == ag:
                draw += p

            else:
                away_win += p

    total = (
        home_win
        +
        draw
        +
        away_win
    )

    return (
        home_win / total,
        draw / total,
        away_win / total
    )


# =========================================================
# モデル予測
# =========================================================

def run_goal_model(
    train_df,
    test_df,
    numeric_features
):

    all_features = (
        numeric_features
        +
        CATEGORICAL
    )

    home_model = make_model(
        numeric_features
    )

    away_model = make_model(
        numeric_features
    )

    home_model.fit(
        train_df[
            all_features
        ],
        train_df["HG"]
    )

    away_model.fit(
        train_df[
            all_features
        ],
        train_df["AG"]
    )

    pred_hg = home_model.predict(
        test_df[
            all_features
        ]
    )

    pred_ag = away_model.predict(
        test_df[
            all_features
        ]
    )

    pred_hg = np.clip(
        pred_hg,
        0.05,
        5.0
    )

    pred_ag = np.clip(
        pred_ag,
        0.05,
        5.0
    )

    probabilities = []

    for hg, ag in zip(
        pred_hg,
        pred_ag
    ):

        probabilities.append(
            match_probabilities(
                hg,
                ag
            )
        )

    return (
        pred_hg,
        pred_ag,
        np.array(
            probabilities
        )
    )


# =========================================================
# 評価
# =========================================================

LABELS = np.array(
    ["H", "D", "A"]
)


def evaluate(
    actual,
    probabilities
):

    actual = np.array(
        actual
    )

    prediction = LABELS[
        np.argmax(
            probabilities,
            axis=1
        )
    ]

    correct = np.sum(
        prediction == actual
    )

    accuracy = np.mean(
        prediction == actual
    )

    actual_index = np.array(
        [
            0 if x == "H"
            else 1 if x == "D"
            else 2
            for x in actual
        ]
    )

    true_probability = (
        probabilities[
            np.arange(
                len(actual)
            ),
            actual_index
        ]
    )

    log_loss = -np.mean(
        np.log(
            np.clip(
                true_probability,
                1e-15,
                1.0
            )
        )
    )

    onehot = np.zeros(
        (
            len(actual),
            3
        )
    )

    onehot[
        np.arange(
            len(actual)
        ),
        actual_index
    ] = 1.0

    brier = np.mean(
        np.sum(
            (
                probabilities
                -
                onehot
            )
            ** 2,
            axis=1
        )
    )

    draw_mask = (
        actual == "D"
    )

    if draw_mask.sum() > 0:

        draw_recall = np.mean(
            prediction[
                draw_mask
            ]
            == "D"
        )

    else:

        draw_recall = np.nan

    return {
        "Correct":
            int(correct),

        "Accuracy":
            accuracy,

        "LogLoss":
            log_loss,

        "Brier":
            brier,

        "DrawRecall":
            draw_recall
    }


# =========================================================
# 2025
# =========================================================

train = (
    data[
        data["Season"]
        != "2025"
    ]
    .copy()
)


test = (
    data[
        data["Season"]
        == "2025"
    ]
    .copy()
    .reset_index(drop=True)
)


# =========================================================
# G2を同条件で再計算
# =========================================================

g2_hg, g2_ag, g2_probs = (
    run_goal_model(
        train,
        test,
        G2_NUMERIC
    )
)


# =========================================================
# G5
# =========================================================

g5_hg, g5_ag, g5_probs = (
    run_goal_model(
        train,
        test,
        G5_NUMERIC
    )
)


g2_metrics = evaluate(
    test["Result"],
    g2_probs
)


g5_metrics = evaluate(
    test["Result"],
    g5_probs
)


# =========================================================
# 保存
# =========================================================

test["G2_H"] = g2_probs[:, 0]
test["G2_D"] = g2_probs[:, 1]
test["G2_A"] = g2_probs[:, 2]

test["G5_H"] = g5_probs[:, 0]
test["G5_D"] = g5_probs[:, 1]
test["G5_A"] = g5_probs[:, 2]

test["G5_HG"] = g5_hg
test["G5_AG"] = g5_ag

test["G5_Prediction"] = (
    LABELS[
        np.argmax(
            g5_probs,
            axis=1
        )
    ]
)


# =========================================================
# メイン
# =========================================================

st.header(
    "🏆 G5：2025年 全380試合"
)


c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "正解率",
    f"{g5_metrics['Accuracy']:.1%}"
)

c2.metric(
    "Log Loss",
    f"{g5_metrics['LogLoss']:.4f}"
)

c3.metric(
    "Brier Score",
    f"{g5_metrics['Brier']:.4f}"
)

c4.metric(
    "Draw Recall",
    f"{g5_metrics['DrawRecall']:.1%}"
)


st.write(
    "正解数:",
    g5_metrics["Correct"],
    "/",
    len(test)
)


# =========================================================
# G2 vs G5
# =========================================================

st.subheader(
    "🆚 G2 vs G5"
)


comparison = pd.DataFrame(
    {
        "モデル": [
            "G2",
            "G5"
        ],

        "Accuracy": [
            g2_metrics[
                "Accuracy"
            ] * 100,

            g5_metrics[
                "Accuracy"
            ] * 100
        ],

        "Log Loss": [
            g2_metrics[
                "LogLoss"
            ],

            g5_metrics[
                "LogLoss"
            ]
        ],

        "Brier": [
            g2_metrics[
                "Brier"
            ],

            g5_metrics[
                "Brier"
            ]
        ],

        "Draw Recall": [
            g2_metrics[
                "DrawRecall"
            ] * 100,

            g5_metrics[
                "DrawRecall"
            ] * 100
        ]
    }
)


comparison["Accuracy"] = (
    comparison[
        "Accuracy"
    ].round(1)
)


comparison["Log Loss"] = (
    comparison[
        "Log Loss"
    ].round(4)
)


comparison["Brier"] = (
    comparison[
        "Brier"
    ].round(4)
)


comparison["Draw Recall"] = (
    comparison[
        "Draw Recall"
    ].round(1)
)


st.dataframe(
    comparison,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 得点平均
# =========================================================

st.subheader(
    "⚽ 実得点 vs G2 vs G5"
)


goal_comparison = pd.DataFrame(
    {
        "項目": [
            "Home平均得点",
            "Away平均得点"
        ],

        "実際": [
            test[
                "HG"
            ].mean(),

            test[
                "AG"
            ].mean()
        ],

        "G2": [
            g2_hg.mean(),
            g2_ag.mean()
        ],

        "G5": [
            g5_hg.mean(),
            g5_ag.mean()
        ]
    }
)


for col in [
    "実際",
    "G2",
    "G5"
]:

    goal_comparison[col] = (
        goal_comparison[col]
        .round(3)
    )


st.dataframe(
    goal_comparison,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 平均H/D/A
# =========================================================

st.subheader(
    "🎯 実際率 vs G2 vs G5"
)


probability_comparison = pd.DataFrame(
    {
        "結果": [
            "Home",
            "Draw",
            "Away"
        ],

        "実際率": [
            (
                test["Result"]
                == "H"
            ).mean(),

            (
                test["Result"]
                == "D"
            ).mean(),

            (
                test["Result"]
                == "A"
            ).mean()
        ],

        "G2平均確率": [
            test[
                "G2_H"
            ].mean(),

            test[
                "G2_D"
            ].mean(),

            test[
                "G2_A"
            ].mean()
        ],

        "G5平均確率": [
            test[
                "G5_H"
            ].mean(),

            test[
                "G5_D"
            ].mean(),

            test[
                "G5_A"
            ].mean()
        ]
    }
)


for col in [
    "実際率",
    "G2平均確率",
    "G5平均確率"
]:

    probability_comparison[
        col
    ] = (
        probability_comparison[
            col
        ]
        * 100
    ).round(1)


st.dataframe(
    probability_comparison,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# G5 Rating確認
# =========================================================

st.header(
    "💪 G5 Attack / Defense Rating"
)


rating_columns = [
    "Date",
    "Home",
    "Away",

    "Home_AttackRating",
    "Home_DefenseRating",

    "Away_AttackRating",
    "Away_DefenseRating"
]


st.dataframe(
    test[
        rating_columns
    ],
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 個別試合
# =========================================================

st.header(
    "🔍 個別試合"
)


match_options = {}


for index, row in test.iterrows():

    label = (
        row["Date"].strftime(
            "%Y-%m-%d"
        )
        +
        " | "
        +
        row["Home"]
        +
        " vs "
        +
        row["Away"]
    )

    match_options[label] = index


selected_label = st.selectbox(
    "試合を選択",
    list(
        match_options.keys()
    )
)


selected_index = (
    match_options[
        selected_label
    ]
)


selected = test.loc[
    selected_index
]


st.subheader(
    f"{selected['Home']} "
    f"vs "
    f"{selected['Away']}"
)


# =========================================================
# Rating
# =========================================================

r1, r2 = st.columns(2)


with r1:

    st.write(
        f"### {selected['Home']}"
    )

    st.write(
        "Attack:",
        f"{selected['Home_AttackRating']:.3f}"
    )

    st.write(
        "Defense:",
        f"{selected['Home_DefenseRating']:.3f}"
    )


with r2:

    st.write(
        f"### {selected['Away']}"
    )

    st.write(
        "Attack:",
        f"{selected['Away_AttackRating']:.3f}"
    )

    st.write(
        "Defense:",
        f"{selected['Away_DefenseRating']:.3f}"
    )


st.write(
    "G5予想得点：",
    f"{selected['G5_HG']:.2f}",
    "-",
    f"{selected['G5_AG']:.2f}"
)


# =========================================================
# G2 vs G5 個別確率
# =========================================================

individual = pd.DataFrame(
    {
        "結果": [
            "Home",
            "Draw",
            "Away"
        ],

        "G2": [
            selected["G2_H"],
            selected["G2_D"],
            selected["G2_A"]
        ],

        "G5": [
            selected["G5_H"],
            selected["G5_D"],
            selected["G5_A"]
        ]
    }
)


individual["G2"] = (
    individual["G2"]
    * 100
).round(1)


individual["G5"] = (
    individual["G5"]
    * 100
).round(1)


st.dataframe(
    individual,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# ランダム予想
# =========================================================

sample_key = (
    "g5_sample_"
    +
    str(selected_index)
)


if st.button(
    "🎲 G5確率から1回予想"
):

    sampled = np.random.choice(
        ["H", "D", "A"],
        p=[
            selected["G5_H"],
            selected["G5_D"],
            selected["G5_A"]
        ]
    )

    st.session_state[
        sample_key
    ] = sampled


if sample_key in st.session_state:

    sampled = (
        st.session_state[
            sample_key
        ]
    )

    if sampled == "H":

        result_text = (
            f"🏠 {selected['Home']} 勝ち"
        )

    elif sampled == "D":

        result_text = (
            "🤝 引き分け"
        )

    else:

        result_text = (
            f"✈️ {selected['Away']} 勝ち"
        )

    st.success(
        "今回の予想："
        +
        result_text
    )


st.write(
    "実際：",
    f"{selected['Home']} "
    f"{int(selected['HG'])}"
    " - "
    f"{int(selected['AG'])} "
    f"{selected['Away']}"
)


st.info(
    "Attack / Defense Ratingは各試合の前の値だけを"
    "特徴量として使用し、その試合終了後に更新しています。"
)
