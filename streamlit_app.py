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
    page_title="J1 Goal Predictor G2",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ J1 Goal Predictor - G2")

st.write(
    "G2ではチーム名そのものも学習し、"
    "各クラブ固有の攻撃・守備傾向を予想得点へ反映します。"
)


# =========================================================
# データ読み込み
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

    df = (
        df
        .sort_values(
            ["Date", "OriginalOrder"]
        )
        .reset_index(drop=True)
    )

    return df


matches = load_data()


# =========================================================
# 補助関数
# =========================================================

def average(values):

    if len(values) == 0:
        return np.nan

    return sum(values) / len(values)


# =========================================================
# 試合前チーム能力作成
# =========================================================

@st.cache_data
def build_features(matches):

    elo = defaultdict(
        lambda: 1500.0
    )

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

    rows = []

    K_FACTOR = 20

    for _, match in matches.iterrows():

        home = match["Home"]
        away = match["Away"]

        hg = float(match["HG"])
        ag = float(match["AG"])

        home_elo = elo[home]
        away_elo = elo[away]

        home_form5 = sum(
            recent_points_5[home]
        )

        away_form5 = sum(
            recent_points_5[away]
        )

        home_recent_gf = average(
            recent_gf_10[home]
        )

        home_recent_ga = average(
            recent_ga_10[home]
        )

        away_recent_gf = average(
            recent_gf_10[away]
        )

        away_recent_ga = average(
            recent_ga_10[away]
        )

        home_home_gf = average(
            home_gf_10[home]
        )

        home_home_ga = average(
            home_ga_10[home]
        )

        away_away_gf = average(
            away_gf_10[away]
        )

        away_away_ga = average(
            away_ga_10[away]
        )

        rows.append(
            {
                "Season": str(
                    match["Season"]
                ),

                "Date": match["Date"],

                "Home": home,
                "Away": away,

                "HG": int(hg),
                "AG": int(ag),

                "Home_Elo": home_elo,
                "Away_Elo": away_elo,

                "EloDiff":
                    home_elo
                    - away_elo,

                "Home_Form5":
                    home_form5,

                "Away_Form5":
                    away_form5,

                "FormDiff":
                    home_form5
                    - away_form5,

                "Home_Recent10_GF":
                    home_recent_gf,

                "Home_Recent10_GA":
                    home_recent_ga,

                "Away_Recent10_GF":
                    away_recent_gf,

                "Away_Recent10_GA":
                    away_recent_ga,

                "Home_Home10_GF":
                    home_home_gf,

                "Home_Home10_GA":
                    home_home_ga,

                "Away_Away10_GF":
                    away_away_gf,

                "Away_Away10_GA":
                    away_away_ga
            }
        )

        # =================================================
        # ここから試合後更新
        # =================================================

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

        expected_home = (
            1
            /
            (
                1
                +
                10 ** (
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
# 特徴量
# =========================================================

NUMERIC_FEATURES = [

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


CATEGORICAL_FEATURES = [
    "Home",
    "Away"
]


ALL_FEATURES = (
    NUMERIC_FEATURES
    +
    CATEGORICAL_FEATURES
)


# =========================================================
# 2024以前 = 学習
# 2025 = 比較用ベンチマーク
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
# 前処理
# =========================================================

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
            NUMERIC_FEATURES
        ),
        (
            "team",
            categorical_transformer,
            CATEGORICAL_FEATURES
        )
    ]
)


# =========================================================
# モデル作成関数
# =========================================================

def make_poisson_model():

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
# Home得点モデル
# =========================================================

home_model = make_poisson_model()

home_model.fit(
    train[ALL_FEATURES],
    train["HG"]
)


# =========================================================
# Away得点モデル
# =========================================================

away_model = make_poisson_model()

away_model.fit(
    train[ALL_FEATURES],
    train["AG"]
)


# =========================================================
# 2025予想得点
# =========================================================

test["Pred_HG"] = (
    home_model.predict(
        test[ALL_FEATURES]
    )
)


test["Pred_AG"] = (
    away_model.predict(
        test[ALL_FEATURES]
    )
)


test["Pred_HG"] = (
    test["Pred_HG"]
    .clip(0.05, 5.0)
)

test["Pred_AG"] = (
    test["Pred_AG"]
    .clip(0.05, 5.0)
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


def calculate_match_probabilities(
    home_expected,
    away_expected,
    max_goals=10
):

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    score_probabilities = []

    for home_goals in range(
        max_goals + 1
    ):

        ph = poisson_probability(
            home_goals,
            home_expected
        )

        for away_goals in range(
            max_goals + 1
        ):

            pa = poisson_probability(
                away_goals,
                away_expected
            )

            probability = ph * pa

            score_probabilities.append(
                (
                    home_goals,
                    away_goals,
                    probability
                )
            )

            if home_goals > away_goals:

                home_win += probability

            elif home_goals == away_goals:

                draw += probability

            else:

                away_win += probability

    total = (
        home_win
        + draw
        + away_win
    )

    home_win /= total
    draw /= total
    away_win /= total

    return (
        home_win,
        draw,
        away_win,
        score_probabilities
    )


# =========================================================
# 全試合 H/D/A
# =========================================================

prob_h_list = []
prob_d_list = []
prob_a_list = []

prediction_list = []


for _, row in test.iterrows():

    ph, pd_, pa, _ = (
        calculate_match_probabilities(
            row["Pred_HG"],
            row["Pred_AG"]
        )
    )

    prob_h_list.append(ph)
    prob_d_list.append(pd_)
    prob_a_list.append(pa)

    probabilities = {
        "H": ph,
        "D": pd_,
        "A": pa
    }

    prediction_list.append(
        max(
            probabilities,
            key=probabilities.get
        )
    )


test["Prob_H"] = prob_h_list
test["Prob_D"] = prob_d_list
test["Prob_A"] = prob_a_list

test["Prediction"] = (
    prediction_list
)


# =========================================================
# 実際の結果
# =========================================================

def actual_result(row):

    if row["HG"] > row["AG"]:
        return "H"

    if row["HG"] < row["AG"]:
        return "A"

    return "D"


test["Result"] = test.apply(
    actual_result,
    axis=1
)


# =========================================================
# 評価
# =========================================================

accuracy = (
    test["Prediction"]
    ==
    test["Result"]
).mean()


correct_count = (
    test["Prediction"]
    ==
    test["Result"]
).sum()


actual_probability = np.where(
    test["Result"] == "H",
    test["Prob_H"],
    np.where(
        test["Result"] == "D",
        test["Prob_D"],
        test["Prob_A"]
    )
)


actual_probability = np.clip(
    actual_probability,
    1e-15,
    1.0
)


log_loss = -np.mean(
    np.log(
        actual_probability
    )
)


actual_h = (
    test["Result"] == "H"
).astype(int)

actual_d = (
    test["Result"] == "D"
).astype(int)

actual_a = (
    test["Result"] == "A"
).astype(int)


brier = np.mean(

    (
        test["Prob_H"]
        - actual_h
    ) ** 2

    +

    (
        test["Prob_D"]
        - actual_d
    ) ** 2

    +

    (
        test["Prob_A"]
        - actual_a
    ) ** 2
)


actual_draws = (
    test["Result"] == "D"
).sum()


correct_draws = (
    (
        test["Result"] == "D"
    )
    &
    (
        test["Prediction"] == "D"
    )
).sum()


draw_recall = (
    correct_draws
    /
    actual_draws
)


# =========================================================
# Draw分析
# =========================================================

max_draw_probability = (
    test["Prob_D"].max()
)


draw_30_count = (
    test["Prob_D"]
    >= 0.30
).sum()


draw_25_count = (
    test["Prob_D"]
    >= 0.25
).sum()


draw_top_count = (
    test["Prediction"]
    == "D"
).sum()


# =========================================================
# メイン結果
# =========================================================

st.header(
    "🏆 G2：2025年 全380試合"
)


c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "正解率",
    f"{accuracy:.1%}"
)

c2.metric(
    "Log Loss",
    f"{log_loss:.4f}"
)

c3.metric(
    "Brier Score",
    f"{brier:.4f}"
)

c4.metric(
    "引き分けRecall",
    f"{draw_recall:.1%}"
)


st.write(
    "正解数:",
    int(correct_count),
    "/",
    len(test)
)


# =========================================================
# G1 / M8比較
# =========================================================

st.subheader(
    "🆚 モデル比較"
)


comparison = pd.DataFrame(
    {
        "モデル": [
            "M8",
            "G1",
            "G2"
        ],

        "正解率": [
            48.4,
            47.1,
            round(
                accuracy * 100,
                1
            )
        ],

        "Log Loss": [
            1.0422,
            1.0467,
            round(
                log_loss,
                4
            )
        ],

        "Brier Score": [
            0.6279,
            0.6297,
            round(
                brier,
                4
            )
        ],

        "Draw Recall": [
            0.0,
            0.0,
            round(
                draw_recall * 100,
                1
            )
        ]
    }
)


st.dataframe(
    comparison,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 得点予測チェック
# =========================================================

st.subheader(
    "⚽ 得点予測チェック"
)


goal_check = pd.DataFrame(
    {
        "項目": [
            "Home平均得点",
            "Away平均得点"
        ],

        "実際": [
            test["HG"].mean(),
            test["AG"].mean()
        ],

        "予測": [
            test["Pred_HG"].mean(),
            test["Pred_AG"].mean()
        ]
    }
)


goal_check["実際"] = (
    goal_check["実際"]
    .round(3)
)

goal_check["予測"] = (
    goal_check["予測"]
    .round(3)
)


st.dataframe(
    goal_check,
    hide_index=True
)


# =========================================================
# 実際率 vs 平均予測確率
# =========================================================

st.subheader(
    "🎯 実際率 vs 平均予測確率"
)


probability_check = pd.DataFrame(
    {
        "結果": [
            "Home",
            "Draw",
            "Away"
        ],

        "実際率": [
            (
                test["Result"] == "H"
            ).mean(),

            (
                test["Result"] == "D"
            ).mean(),

            (
                test["Result"] == "A"
            ).mean()
        ],

        "平均予測確率": [
            test["Prob_H"].mean(),
            test["Prob_D"].mean(),
            test["Prob_A"].mean()
        ]
    }
)


probability_check[
    "実際率"
] = (
    probability_check[
        "実際率"
    ]
    * 100
).round(1)


probability_check[
    "平均予測確率"
] = (
    probability_check[
        "平均予測確率"
    ]
    * 100
).round(1)


st.dataframe(
    probability_check,
    hide_index=True
)


# =========================================================
# Draw詳細
# =========================================================

st.subheader(
    "🤝 Draw確率の分析"
)


d1, d2, d3, d4 = (
    st.columns(4)
)


d1.metric(
    "最大Draw確率",
    f"{max_draw_probability:.1%}"
)

d2.metric(
    "Draw 30%以上",
    f"{int(draw_30_count)}試合"
)

d3.metric(
    "Draw 25%以上",
    f"{int(draw_25_count)}試合"
)

d4.metric(
    "Drawが確率1位",
    f"{int(draw_top_count)}試合"
)


st.caption(
    "Draw Recallが0%でも、Drawに十分な確率を"
    "割り当てている可能性があります。"
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
# 予想得点
# =========================================================

g1, g2 = st.columns(2)


g1.metric(
    f"{selected['Home']} 予想得点",
    f"{selected['Pred_HG']:.2f}"
)

g2.metric(
    f"{selected['Away']} 予想得点",
    f"{selected['Pred_AG']:.2f}"
)


# =========================================================
# 確率
# =========================================================

p1, p2, p3 = st.columns(3)


p1.metric(
    "🏠 Home",
    f"{selected['Prob_H']:.1%}"
)

p2.metric(
    "🤝 Draw",
    f"{selected['Prob_D']:.1%}"
)

p3.metric(
    "✈️ Away",
    f"{selected['Prob_A']:.1%}"
)


st.write(
    "最大確率による予想：",
    selected["Prediction"]
)


# =========================================================
# ランダム予想
#
# Streamlitの再描画で勝手に変わらないよう、
# ボタンを押したときだけsession_stateへ保存
# =========================================================

st.subheader(
    "🎲 確率サンプリング"
)


sample_key = (
    "sample_"
    +
    str(selected_index)
)


if st.button(
    "この確率から1回予想する"
):

    sampled_result = (
        np.random.choice(
            ["H", "D", "A"],
            p=[
                selected["Prob_H"],
                selected["Prob_D"],
                selected["Prob_A"]
            ]
        )
    )

    st.session_state[
        sample_key
    ] = sampled_result


if sample_key in st.session_state:

    sampled = (
        st.session_state[
            sample_key
        ]
    )

    if sampled == "H":

        sampled_text = (
            f"🏠 {selected['Home']} 勝ち"
        )

    elif sampled == "D":

        sampled_text = "🤝 引き分け"

    else:

        sampled_text = (
            f"✈️ {selected['Away']} 勝ち"
        )

    st.success(
        "今回のランダム予想："
        +
        sampled_text
    )


# =========================================================
# 10回試行
# =========================================================

multi_key = (
    "multi_"
    +
    str(selected_index)
)


if st.button(
    "10回シミュレーション"
):

    simulations = (
        np.random.choice(
            ["H", "D", "A"],
            size=10,
            p=[
                selected["Prob_H"],
                selected["Prob_D"],
                selected["Prob_A"]
            ]
        )
    )

    st.session_state[
        multi_key
    ] = simulations.tolist()


if multi_key in st.session_state:

    simulations = (
        st.session_state[
            multi_key
        ]
    )

    st.write(
        "10回の結果：",
        " → ".join(
            simulations
        )
    )

    simulation_summary = pd.DataFrame(
        {
            "結果": [
                "H",
                "D",
                "A"
            ],

            "回数": [
                simulations.count("H"),
                simulations.count("D"),
                simulations.count("A")
            ]
        }
    )

    st.dataframe(
        simulation_summary,
        hide_index=True
    )


# =========================================================
# スコア確率
# =========================================================

_, _, _, score_probs = (
    calculate_match_probabilities(
        selected["Pred_HG"],
        selected["Pred_AG"]
    )
)


score_table = pd.DataFrame(
    score_probs,
    columns=[
        "HomeGoals",
        "AwayGoals",
        "Probability"
    ]
)


score_table = (
    score_table
    .sort_values(
        "Probability",
        ascending=False
    )
    .head(10)
    .copy()
)


score_table["Score"] = (
    score_table[
        "HomeGoals"
    ].astype(str)
    +
    " - "
    +
    score_table[
        "AwayGoals"
    ].astype(str)
)


score_table["確率"] = (
    score_table[
        "Probability"
    ]
    * 100
).round(1)


st.subheader(
    "⚽ スコア確率 TOP10"
)


st.dataframe(
    score_table[
        [
            "Score",
            "確率"
        ]
    ],
    hide_index=True
)


# =========================================================
# 実際の結果
# =========================================================

st.write(
    "実際：",
    f"{selected['Home']} "
    f"{int(selected['HG'])}"
    " - "
    f"{int(selected['AG'])} "
    f"{selected['Away']}"
)


st.info(
    "モデル比較では最大確率のH/D/Aを使用します。"
    "ランダム予想は完成アプリで予想セットを生成するための"
    "別機能として扱います。"
)
