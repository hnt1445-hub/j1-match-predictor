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
    page_title="J1 Goal Predictor G4",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ J1 Goal Predictor - G4")

st.write(
    "G4 = G2のチーム固有Poissonモデル "
    "+ Dixon-Coles低スコア補正"
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
# 試合前特徴量
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

        rows.append(
            {
                "Season": str(match["Season"]),
                "Date": match["Date"],

                "Home": home,
                "Away": away,

                "HG": int(hg),
                "AG": int(ag),

                "Home_Elo": home_elo,
                "Away_Elo": away_elo,
                "EloDiff": home_elo - away_elo,

                "Home_Form5": home_form5,
                "Away_Form5": away_form5,
                "FormDiff":
                    home_form5 - away_form5,

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
                    )
            }
        )

        # ================================================
        # 試合後更新
        # ================================================

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
# 実際の結果
# =========================================================

def actual_result(row):

    if row["HG"] > row["AG"]:
        return "H"

    if row["HG"] < row["AG"]:
        return "A"

    return "D"


data["Result"] = data.apply(
    actual_result,
    axis=1
)


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
# Poissonモデル
# =========================================================

def make_goal_model():

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
# λを予測
# =========================================================

def predict_lambdas(
    train_df,
    target_df
):

    home_model = make_goal_model()
    away_model = make_goal_model()

    home_model.fit(
        train_df[ALL_FEATURES],
        train_df["HG"]
    )

    away_model.fit(
        train_df[ALL_FEATURES],
        train_df["AG"]
    )

    home_lambda = (
        home_model.predict(
            target_df[ALL_FEATURES]
        )
    )

    away_lambda = (
        away_model.predict(
            target_df[ALL_FEATURES]
        )
    )

    home_lambda = np.clip(
        home_lambda,
        0.05,
        5.0
    )

    away_lambda = np.clip(
        away_lambda,
        0.05,
        5.0
    )

    return (
        home_lambda,
        away_lambda
    )


# =========================================================
# Poisson確率
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


# =========================================================
# Dixon-Coles補正
#
# 0-0
# 0-1
# 1-0
# 1-1
#
# だけを補正
# =========================================================

def dc_tau(
    home_goals,
    away_goals,
    home_lambda,
    away_lambda,
    rho
):

    if (
        home_goals == 0
        and away_goals == 0
    ):

        return (
            1
            -
            home_lambda
            *
            away_lambda
            *
            rho
        )

    if (
        home_goals == 0
        and away_goals == 1
    ):

        return (
            1
            +
            home_lambda
            *
            rho
        )

    if (
        home_goals == 1
        and away_goals == 0
    ):

        return (
            1
            +
            away_lambda
            *
            rho
        )

    if (
        home_goals == 1
        and away_goals == 1
    ):

        return (
            1
            -
            rho
        )

    return 1.0


# =========================================================
# H/D/A確率
# =========================================================

def match_probabilities(
    home_lambda,
    away_lambda,
    rho=0.0,
    max_goals=10
):

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    score_rows = []

    for hg in range(
        max_goals + 1
    ):

        home_p = (
            poisson_probability(
                hg,
                home_lambda
            )
        )

        for ag in range(
            max_goals + 1
        ):

            away_p = (
                poisson_probability(
                    ag,
                    away_lambda
                )
            )

            tau = dc_tau(
                hg,
                ag,
                home_lambda,
                away_lambda,
                rho
            )

            probability = (
                home_p
                *
                away_p
                *
                tau
            )

            # 不正な負確率を防ぐ
            probability = max(
                probability,
                0.0
            )

            score_rows.append(
                (
                    hg,
                    ag,
                    probability
                )
            )

            if hg > ag:

                home_win += probability

            elif hg == ag:

                draw += probability

            else:

                away_win += probability

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
        away_win / total,
        score_rows
    )


# =========================================================
# λ → 全試合確率
# =========================================================

def make_probability_array(
    home_lambdas,
    away_lambdas,
    rho
):

    probabilities = []

    for home_lambda, away_lambda in zip(
        home_lambdas,
        away_lambdas
    ):

        ph, pd_, pa, _ = (
            match_probabilities(
                home_lambda,
                away_lambda,
                rho
            )
        )

        probabilities.append(
            [
                ph,
                pd_,
                pa
            ]
        )

    return np.array(
        probabilities
    )


# =========================================================
# 評価
# =========================================================

LABELS = np.array(
    ["H", "D", "A"]
)


def evaluate(
    actual_results,
    probabilities
):

    actual_results = np.array(
        actual_results
    )

    predictions = LABELS[
        np.argmax(
            probabilities,
            axis=1
        )
    ]

    accuracy = np.mean(
        predictions
        ==
        actual_results
    )

    correct = np.sum(
        predictions
        ==
        actual_results
    )

    actual_indices = np.array(
        [
            0 if result == "H"
            else 1 if result == "D"
            else 2
            for result
            in actual_results
        ]
    )

    true_probabilities = (
        probabilities[
            np.arange(
                len(probabilities)
            ),
            actual_indices
        ]
    )

    true_probabilities = np.clip(
        true_probabilities,
        1e-15,
        1.0
    )

    log_loss = -np.mean(
        np.log(
            true_probabilities
        )
    )

    actual_onehot = np.zeros(
        (
            len(probabilities),
            3
        )
    )

    actual_onehot[
        np.arange(
            len(probabilities)
        ),
        actual_indices
    ] = 1.0

    brier = np.mean(
        np.sum(
            (
                probabilities
                -
                actual_onehot
            ) ** 2,
            axis=1
        )
    )

    draw_mask = (
        actual_results == "D"
    )

    if draw_mask.sum() > 0:

        draw_recall = np.mean(
            predictions[
                draw_mask
            ]
            ==
            "D"
        )

    else:

        draw_recall = np.nan

    return {
        "Accuracy": accuracy,
        "Correct": int(correct),
        "LogLoss": log_loss,
        "Brier": brier,
        "DrawRecall": draw_recall
    }


# =========================================================
# STEP 1
# 2023以前 → 2024
# =========================================================

train_pre_2024 = (
    data[
        ~data["Season"].isin(
            ["2024", "2025"]
        )
    ]
    .copy()
)


validation_2024 = (
    data[
        data["Season"] == "2024"
    ]
    .copy()
    .reset_index(drop=True)
)


val_home_lambda, val_away_lambda = (
    predict_lambdas(
        train_pre_2024,
        validation_2024
    )
)


# =========================================================
# 2024だけでρを選択
#
# Log Loss最小を採用
# =========================================================

rho_candidates = np.round(
    np.arange(
        -0.20,
        0.201,
        0.01
    ),
    2
)


rho_results = []


for rho in rho_candidates:

    probs = make_probability_array(
        val_home_lambda,
        val_away_lambda,
        rho
    )

    metrics = evaluate(
        validation_2024[
            "Result"
        ],
        probs
    )

    rho_results.append(
        {
            "rho": rho,
            "Accuracy":
                metrics["Accuracy"],

            "LogLoss":
                metrics["LogLoss"],

            "Brier":
                metrics["Brier"]
        }
    )


rho_table = pd.DataFrame(
    rho_results
)


best_row = (
    rho_table
    .sort_values(
        "LogLoss"
    )
    .iloc[0]
)


BEST_RHO = float(
    best_row["rho"]
)


# =========================================================
# STEP 2
# 2024以前全部 → 2025
# =========================================================

train_pre_2025 = (
    data[
        data["Season"]
        != "2025"
    ]
    .copy()
)


test_2025 = (
    data[
        data["Season"]
        == "2025"
    ]
    .copy()
    .reset_index(drop=True)
)


home_lambda_2025, away_lambda_2025 = (
    predict_lambdas(
        train_pre_2025,
        test_2025
    )
)


# =========================================================
# G2
# rho = 0
# =========================================================

g2_probs = make_probability_array(
    home_lambda_2025,
    away_lambda_2025,
    0.0
)


# =========================================================
# G4
# 2024で選んだrho
# =========================================================

g4_probs = make_probability_array(
    home_lambda_2025,
    away_lambda_2025,
    BEST_RHO
)


g2_metrics = evaluate(
    test_2025["Result"],
    g2_probs
)


g4_metrics = evaluate(
    test_2025["Result"],
    g4_probs
)


# =========================================================
# DataFrame保存
# =========================================================

test_2025[
    "Pred_HG"
] = home_lambda_2025

test_2025[
    "Pred_AG"
] = away_lambda_2025


test_2025[
    "G2_H"
] = g2_probs[:, 0]

test_2025[
    "G2_D"
] = g2_probs[:, 1]

test_2025[
    "G2_A"
] = g2_probs[:, 2]


test_2025[
    "Prob_H"
] = g4_probs[:, 0]

test_2025[
    "Prob_D"
] = g4_probs[:, 1]

test_2025[
    "Prob_A"
] = g4_probs[:, 2]


test_2025[
    "Prediction"
] = LABELS[
    np.argmax(
        g4_probs,
        axis=1
    )
]


# =========================================================
# 画面
# =========================================================

st.header(
    "🔧 2024年で選んだDixon-Coles補正"
)


st.metric(
    "採用ρ（rho）",
    f"{BEST_RHO:.2f}"
)


st.caption(
    "ρは2024年のLog Lossが最小になる値を採用。"
    "2025年の結果はρ選択に使用していません。"
)


# 上位候補も確認
best_rhos = (
    rho_table
    .sort_values(
        "LogLoss"
    )
    .head(10)
    .copy()
)


best_rhos[
    "Accuracy"
] = (
    best_rhos[
        "Accuracy"
    ]
    * 100
).round(1)


best_rhos[
    "LogLoss"
] = (
    best_rhos[
        "LogLoss"
    ]
    .round(4)
)


best_rhos[
    "Brier"
] = (
    best_rhos[
        "Brier"
    ]
    .round(4)
)


st.dataframe(
    best_rhos,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 2025 G4
# =========================================================

st.header(
    "🏆 G4：2025年 全380試合"
)


c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "正解率",
    f"{g4_metrics['Accuracy']:.1%}"
)


c2.metric(
    "Log Loss",
    f"{g4_metrics['LogLoss']:.4f}"
)


c3.metric(
    "Brier Score",
    f"{g4_metrics['Brier']:.4f}"
)


c4.metric(
    "Draw Recall",
    f"{g4_metrics['DrawRecall']:.1%}"
)


st.write(
    "正解数:",
    g4_metrics["Correct"],
    "/",
    len(test_2025)
)


# =========================================================
# モデル比較
# =========================================================

st.subheader(
    "🆚 G2 vs G4"
)


comparison = pd.DataFrame(
    {
        "モデル": [
            "G2",
            "G4"
        ],

        "Accuracy": [
            g2_metrics[
                "Accuracy"
            ] * 100,

            g4_metrics[
                "Accuracy"
            ] * 100
        ],

        "Log Loss": [
            g2_metrics[
                "LogLoss"
            ],

            g4_metrics[
                "LogLoss"
            ]
        ],

        "Brier": [
            g2_metrics[
                "Brier"
            ],

            g4_metrics[
                "Brier"
            ]
        ],

        "Draw Recall": [
            g2_metrics[
                "DrawRecall"
            ] * 100,

            g4_metrics[
                "DrawRecall"
            ] * 100
        ]
    }
)


comparison[
    "Accuracy"
] = (
    comparison[
        "Accuracy"
    ]
    .round(1)
)


comparison[
    "Log Loss"
] = (
    comparison[
        "Log Loss"
    ]
    .round(4)
)


comparison[
    "Brier"
] = (
    comparison[
        "Brier"
    ]
    .round(4)
)


comparison[
    "Draw Recall"
] = (
    comparison[
        "Draw Recall"
    ]
    .round(1)
)


st.dataframe(
    comparison,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 平均確率
# =========================================================

st.subheader(
    "🎯 実際率 vs G2 vs G4"
)


actual_h = (
    test_2025[
        "Result"
    ] == "H"
).mean()


actual_d = (
    test_2025[
        "Result"
    ] == "D"
).mean()


actual_a = (
    test_2025[
        "Result"
    ] == "A"
).mean()


probability_comparison = pd.DataFrame(
    {
        "結果": [
            "Home",
            "Draw",
            "Away"
        ],

        "実際率": [
            actual_h,
            actual_d,
            actual_a
        ],

        "G2平均確率": [
            test_2025[
                "G2_H"
            ].mean(),

            test_2025[
                "G2_D"
            ].mean(),

            test_2025[
                "G2_A"
            ].mean()
        ],

        "G4平均確率": [
            test_2025[
                "Prob_H"
            ].mean(),

            test_2025[
                "Prob_D"
            ].mean(),

            test_2025[
                "Prob_A"
            ].mean()
        ]
    }
)


for column in [
    "実際率",
    "G2平均確率",
    "G4平均確率"
]:

    probability_comparison[
        column
    ] = (
        probability_comparison[
            column
        ]
        * 100
    ).round(1)


st.dataframe(
    probability_comparison,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# Draw分析
# =========================================================

st.subheader(
    "🤝 G4 Draw分析"
)


max_draw = (
    test_2025[
        "Prob_D"
    ].max()
)


draw_30 = (
    test_2025[
        "Prob_D"
    ]
    >= 0.30
).sum()


draw_25 = (
    test_2025[
        "Prob_D"
    ]
    >= 0.25
).sum()


draw_top = (
    test_2025[
        "Prediction"
    ]
    == "D"
).sum()


d1, d2, d3, d4 = st.columns(4)


d1.metric(
    "最大Draw確率",
    f"{max_draw:.1%}"
)


d2.metric(
    "Draw 30%以上",
    f"{int(draw_30)}試合"
)


d3.metric(
    "Draw 25%以上",
    f"{int(draw_25)}試合"
)


d4.metric(
    "Drawが確率1位",
    f"{int(draw_top)}試合"
)


# =========================================================
# Draw確率帯
# =========================================================

st.subheader(
    "📏 G4 Draw確率帯"
)


bins = [
    0.00,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
    1.01
]


band_labels = [
    "0-20%",
    "20-25%",
    "25-30%",
    "30-35%",
    "35-40%",
    "40-50%",
    "50%+"
]


draw_temp = (
    test_2025.copy()
)


draw_temp[
    "Band"
] = pd.cut(
    draw_temp[
        "Prob_D"
    ],
    bins=bins,
    labels=band_labels,
    right=False
)


draw_temp[
    "ActualDraw"
] = (
    draw_temp[
        "Result"
    ] == "D"
).astype(int)


draw_table = (
    draw_temp
    .groupby(
        "Band",
        observed=True
    )
    .agg(
        試合数=(
            "ActualDraw",
            "size"
        ),

        平均Draw予測=(
            "Prob_D",
            "mean"
        ),

        実際Draw率=(
            "ActualDraw",
            "mean"
        )
    )
    .reset_index()
)


draw_table[
    "平均Draw予測"
] = (
    draw_table[
        "平均Draw予測"
    ]
    * 100
).round(1)


draw_table[
    "実際Draw率"
] = (
    draw_table[
        "実際Draw率"
    ]
    * 100
).round(1)


st.dataframe(
    draw_table,
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


for index, row in (
    test_2025.iterrows()
):

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

    match_options[
        label
    ] = index


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


selected = (
    test_2025.loc[
        selected_index
    ]
)


st.subheader(
    f"{selected['Home']} "
    f"vs "
    f"{selected['Away']}"
)


st.write(
    "予想得点：",
    f"{selected['Pred_HG']:.2f}",
    "-",
    f"{selected['Pred_AG']:.2f}"
)


individual = pd.DataFrame(
    {
        "結果": [
            "Home",
            "Draw",
            "Away"
        ],

        "G2": [
            selected[
                "G2_H"
            ],

            selected[
                "G2_D"
            ],

            selected[
                "G2_A"
            ]
        ],

        "G4": [
            selected[
                "Prob_H"
            ],

            selected[
                "Prob_D"
            ],

            selected[
                "Prob_A"
            ]
        ]
    }
)


individual["G2"] = (
    individual["G2"]
    * 100
).round(1)


individual["G4"] = (
    individual["G4"]
    * 100
).round(1)


st.dataframe(
    individual,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# G4ランダム予想
# =========================================================

sample_key = (
    "g4_sample_"
    +
    str(selected_index)
)


if st.button(
    "🎲 G4確率から予想"
):

    sampled = np.random.choice(
        ["H", "D", "A"],
        p=[
            selected[
                "Prob_H"
            ],
            selected[
                "Prob_D"
            ],
            selected[
                "Prob_A"
            ]
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
            f"🏠 "
            f"{selected['Home']} 勝ち"
        )

    elif sampled == "D":

        result_text = (
            "🤝 引き分け"
        )

    else:

        result_text = (
            f"✈️ "
            f"{selected['Away']} 勝ち"
        )

    st.success(
        "今回の予想："
        +
        result_text
    )


# =========================================================
# スコアTOP10
# =========================================================

_, _, _, score_rows = (
    match_probabilities(
        selected[
            "Pred_HG"
        ],

        selected[
            "Pred_AG"
        ],

        BEST_RHO
    )
)


score_table = pd.DataFrame(
    score_rows,
    columns=[
        "HG",
        "AG",
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
        "HG"
    ].astype(str)
    +
    " - "
    +
    score_table[
        "AG"
    ].astype(str)
)


score_table["確率"] = (
    score_table[
        "Probability"
    ]
    * 100
).round(1)


st.subheader(
    "⚽ G4 スコア確率 TOP10"
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


st.write(
    "実際：",
    f"{selected['Home']} "
    f"{int(selected['HG'])}"
    " - "
    f"{int(selected['AG'])} "
    f"{selected['Away']}"
)


st.info(
    "G4のρは2024年だけで選択しています。"
    "2025年の結果はρの選択には使っていません。"
)
