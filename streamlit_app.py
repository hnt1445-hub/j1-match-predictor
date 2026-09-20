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
    page_title="J1 Goal Predictor G5.1",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ J1 Goal Predictor - G5.1")

st.write(
    "Attack / Defense Rating の更新速度を2024年だけで選び、"
    "その値を固定して2025年380試合を評価します。"
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
# 特徴量作成
#
# learning_rate を外から変更できるようにした
# =========================================================

@st.cache_data
def build_features(
    matches,
    learning_rate
):

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

    attack = defaultdict(
        lambda: 1.0
    )

    defense = defaultdict(
        lambda: 1.0
    )

    MIN_RATING = 0.50
    MAX_RATING = 1.80

    K_FACTOR = 20

    rows = []

    for _, match in matches.iterrows():

        home = match["Home"]
        away = match["Away"]

        hg = float(match["HG"])
        ag = float(match["AG"])

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
                "Season":
                    str(match["Season"]),

                "Date":
                    match["Date"],

                "Home":
                    home,

                "Away":
                    away,

                "HG":
                    int(hg),

                "AG":
                    int(ag),

                "Home_Elo":
                    home_elo,

                "Away_Elo":
                    away_elo,

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

        # =================================================
        # Attack / Defense Rating更新
        # =================================================

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

        home_ratio = (
            (hg + 0.5)
            /
            (
                expected_home_goals
                + 0.5
            )
        )

        away_ratio = (
            (ag + 0.5)
            /
            (
                expected_away_goals
                + 0.5
            )
        )

        attack[home] = np.clip(
            home_attack
            *
            (
                home_ratio
                **
                learning_rate
            ),
            MIN_RATING,
            MAX_RATING
        )

        attack[away] = np.clip(
            away_attack
            *
            (
                away_ratio
                **
                learning_rate
            ),
            MIN_RATING,
            MAX_RATING
        )

        defense[home] = np.clip(
            home_defense
            *
            (
                away_ratio
                **
                learning_rate
            ),
            MIN_RATING,
            MAX_RATING
        )

        defense[away] = np.clip(
            away_defense
            *
            (
                home_ratio
                **
                learning_rate
            ),
            MIN_RATING,
            MAX_RATING
        )

        # =================================================
        # Elo更新
        # =================================================

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

    result = pd.DataFrame(rows)

    result["Result"] = np.where(
        result["HG"] > result["AG"],
        "H",
        np.where(
            result["HG"] < result["AG"],
            "A",
            "D"
        )
    )

    return result


# =========================================================
# 特徴量
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
# Poissonモデル
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
# H/D/A確率
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
    away_lambda
):

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    for hg in range(11):

        ph = poisson_probability(
            hg,
            home_lambda
        )

        for ag in range(11):

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
        + draw
        + away_win
    )

    return (
        home_win / total,
        draw / total,
        away_win / total
    )


# =========================================================
# 学習・予測
# =========================================================

def run_model(
    data,
    train_seasons_excluded,
    test_season,
    numeric_features
):

    train = data[
        ~data["Season"].isin(
            train_seasons_excluded
        )
    ].copy()

    target = data[
        data["Season"]
        == test_season
    ].copy().reset_index(
        drop=True
    )

    features = (
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
        train[features],
        train["HG"]
    )

    away_model.fit(
        train[features],
        train["AG"]
    )

    home_lambda = np.clip(
        home_model.predict(
            target[features]
        ),
        0.05,
        5.0
    )

    away_lambda = np.clip(
        away_model.predict(
            target[features]
        ),
        0.05,
        5.0
    )

    probabilities = []

    for h, a in zip(
        home_lambda,
        away_lambda
    ):

        probabilities.append(
            match_probabilities(
                h,
                a
            )
        )

    return (
        target,
        home_lambda,
        away_lambda,
        np.array(probabilities)
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

    actual = np.array(actual)

    prediction = LABELS[
        np.argmax(
            probabilities,
            axis=1
        )
    ]

    correct = int(
        np.sum(
            prediction == actual
        )
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
                - onehot
            ) ** 2,
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
            ] == "D"
        )

    else:

        draw_recall = np.nan

    return {
        "Correct": correct,
        "Accuracy": accuracy,
        "LogLoss": log_loss,
        "Brier": brier,
        "DrawRecall": draw_recall
    }


# =========================================================
# STEP 1
# 2024年だけで更新速度を選択
# =========================================================

st.header(
    "🔧 2024年で更新速度を選択"
)


LEARNING_RATE_CANDIDATES = [
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.16,
    0.20
]


validation_rows = []


with st.spinner(
    "2024年で更新速度を比較中..."
):

    for learning_rate in (
        LEARNING_RATE_CANDIDATES
    ):

        candidate_data = (
            build_features(
                matches,
                learning_rate
            )
        )

        (
            validation_target,
            _,
            _,
            validation_probs
        ) = run_model(
            candidate_data,

            # 2024・2025を学習から除外
            [
                "2024",
                "2025"
            ],

            "2024",

            G5_NUMERIC
        )

        metrics = evaluate(
            validation_target[
                "Result"
            ],
            validation_probs
        )

        validation_rows.append(
            {
                "Learning Rate":
                    learning_rate,

                "Accuracy":
                    metrics[
                        "Accuracy"
                    ],

                "Log Loss":
                    metrics[
                        "LogLoss"
                    ],

                "Brier":
                    metrics[
                        "Brier"
                    ]
            }
        )


validation_table = pd.DataFrame(
    validation_rows
)


# Log Loss最小を採用
best_index = (
    validation_table[
        "Log Loss"
    ].idxmin()
)


BEST_RATE = float(
    validation_table.loc[
        best_index,
        "Learning Rate"
    ]
)


display_validation = (
    validation_table.copy()
)


display_validation[
    "Accuracy"
] = (
    display_validation[
        "Accuracy"
    ]
    * 100
).round(1)


display_validation[
    "Log Loss"
] = (
    display_validation[
        "Log Loss"
    ].round(4)
)


display_validation[
    "Brier"
] = (
    display_validation[
        "Brier"
    ].round(4)
)


st.metric(
    "2024年で選ばれた更新速度",
    f"{BEST_RATE:.2f}"
)


st.dataframe(
    display_validation,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# STEP 2
# 2025年を評価
# =========================================================

st.header(
    "🏆 2025年380試合"
)


# -------------------------
# 基準用データ
# 0.08
# -------------------------

data_008 = build_features(
    matches,
    0.08
)


# -------------------------
# 選択されたデータ
# -------------------------

data_best = build_features(
    matches,
    BEST_RATE
)


# =========================================================
# G2
#
# Ratingを使わないので0.08データでOK
# =========================================================

(
    test_g2,
    g2_hg,
    g2_ag,
    g2_probs
) = run_model(
    data_008,
    ["2025"],
    "2025",
    G2_NUMERIC
)


g2_metrics = evaluate(
    test_g2["Result"],
    g2_probs
)


# =========================================================
# G5 固定0.08
# =========================================================

(
    test_g5,
    g5_hg,
    g5_ag,
    g5_probs
) = run_model(
    data_008,
    ["2025"],
    "2025",
    G5_NUMERIC
)


g5_metrics = evaluate(
    test_g5["Result"],
    g5_probs
)


# =========================================================
# G5.1
# 2024で選んだ更新速度
# =========================================================

(
    test_best,
    best_hg,
    best_ag,
    best_probs
) = run_model(
    data_best,
    ["2025"],
    "2025",
    G5_NUMERIC
)


best_metrics = evaluate(
    test_best["Result"],
    best_probs
)


# =========================================================
# 比較
# =========================================================

comparison = pd.DataFrame(
    {
        "モデル": [
            "G2",
            "G5 (0.08)",
            f"G5.1 ({BEST_RATE:.2f})"
        ],

        "Accuracy": [
            g2_metrics[
                "Accuracy"
            ] * 100,

            g5_metrics[
                "Accuracy"
            ] * 100,

            best_metrics[
                "Accuracy"
            ] * 100
        ],

        "Log Loss": [
            g2_metrics[
                "LogLoss"
            ],

            g5_metrics[
                "LogLoss"
            ],

            best_metrics[
                "LogLoss"
            ]
        ],

        "Brier": [
            g2_metrics[
                "Brier"
            ],

            g5_metrics[
                "Brier"
            ],

            best_metrics[
                "Brier"
            ]
        ],

        "Draw Recall": [
            g2_metrics[
                "DrawRecall"
            ] * 100,

            g5_metrics[
                "DrawRecall"
            ] * 100,

            best_metrics[
                "DrawRecall"
            ] * 100
        ],

        "Correct": [
            g2_metrics[
                "Correct"
            ],

            g5_metrics[
                "Correct"
            ],

            best_metrics[
                "Correct"
            ]
        ]
    }
)


comparison[
    "Accuracy"
] = (
    comparison[
        "Accuracy"
    ].round(1)
)


comparison[
    "Log Loss"
] = (
    comparison[
        "Log Loss"
    ].round(4)
)


comparison[
    "Brier"
] = (
    comparison[
        "Brier"
    ].round(4)
)


comparison[
    "Draw Recall"
] = (
    comparison[
        "Draw Recall"
    ].round(1)
)


st.subheader(
    "🆚 G2 vs G5 vs G5.1"
)


st.dataframe(
    comparison,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 得点比較
# =========================================================

st.subheader(
    "⚽ 平均得点"
)


goal_table = pd.DataFrame(
    {
        "項目": [
            "Home平均得点",
            "Away平均得点"
        ],

        "実際": [
            test_best[
                "HG"
            ].mean(),

            test_best[
                "AG"
            ].mean()
        ],

        "G2": [
            g2_hg.mean(),
            g2_ag.mean()
        ],

        "G5 0.08": [
            g5_hg.mean(),
            g5_ag.mean()
        ],

        "G5.1": [
            best_hg.mean(),
            best_ag.mean()
        ]
    }
)


for col in [
    "実際",
    "G2",
    "G5 0.08",
    "G5.1"
]:

    goal_table[col] = (
        goal_table[col]
        .round(3)
    )


st.dataframe(
    goal_table,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# H/D/A平均確率
# =========================================================

st.subheader(
    "🎯 H / D / A 平均確率"
)


probability_table = pd.DataFrame(
    {
        "結果": [
            "Home",
            "Draw",
            "Away"
        ],

        "実際率": [
            (
                test_best[
                    "Result"
                ] == "H"
            ).mean(),

            (
                test_best[
                    "Result"
                ] == "D"
            ).mean(),

            (
                test_best[
                    "Result"
                ] == "A"
            ).mean()
        ],

        "G2": [
            g2_probs[
                :,
                0
            ].mean(),

            g2_probs[
                :,
                1
            ].mean(),

            g2_probs[
                :,
                2
            ].mean()
        ],

        "G5 0.08": [
            g5_probs[
                :,
                0
            ].mean(),

            g5_probs[
                :,
                1
            ].mean(),

            g5_probs[
                :,
                2
            ].mean()
        ],

        "G5.1": [
            best_probs[
                :,
                0
            ].mean(),

            best_probs[
                :,
                1
            ].mean(),

            best_probs[
                :,
                2
            ].mean()
        ]
    }
)


for col in [
    "実際率",
    "G2",
    "G5 0.08",
    "G5.1"
]:

    probability_table[col] = (
        probability_table[
            col
        ]
        * 100
    ).round(1)


st.dataframe(
    probability_table,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 選ばれた更新速度のRatingを見る
# =========================================================

st.subheader(
    "💪 G5.1 Rating例"
)


rating_view = test_best[
    [
        "Date",
        "Home",
        "Away",
        "Home_AttackRating",
        "Home_DefenseRating",
        "Away_AttackRating",
        "Away_DefenseRating"
    ]
].copy()


st.dataframe(
    rating_view,
    hide_index=True,
    use_container_width=True
)


st.info(
    "更新速度は2024年のLog Lossだけで選択しています。"
    "2025年の結果は更新速度の選択には使用していません。"
)
