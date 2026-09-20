import streamlit as st
import pandas as pd
import numpy as np

from collections import defaultdict, deque
from math import exp, factorial

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, LogisticRegression


# =========================================================
# 基本設定
# =========================================================

st.set_page_config(
    page_title="J1 Goal Predictor G3",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ J1 Goal Predictor - G3")

st.write(
    "G3ではG2型の得点モデルに加えて、"
    "2024年だけを使ってH / D / A確率を校正します。"
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
# 試合前チーム能力
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
                "Season": str(match["Season"]),
                "Date": match["Date"],

                "Home": home,
                "Away": away,

                "HG": int(hg),
                "AG": int(ag),

                "Home_Elo": home_elo,
                "Away_Elo": away_elo,

                "EloDiff":
                    home_elo - away_elo,

                "Home_Form5":
                    home_form5,

                "Away_Form5":
                    away_form5,

                "FormDiff":
                    home_form5 - away_form5,

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
        # 試合終了後に更新
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

        recent_points_5[home].append(hp)
        recent_points_5[away].append(ap)

        recent_gf_10[home].append(hg)
        recent_ga_10[home].append(ag)

        recent_gf_10[away].append(ag)
        recent_ga_10[away].append(hg)

        home_gf_10[home].append(hg)
        home_ga_10[home].append(ag)

        away_gf_10[away].append(ag)
        away_ga_10[away].append(hg)

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
# 得点モデル作成
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

    model = Pipeline(
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

    return model


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


def calculate_match_probabilities(
    home_expected,
    away_expected,
    max_goals=10
):

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

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

    return (
        home_win / total,
        draw / total,
        away_win / total
    )


# =========================================================
# 得点モデルからH/D/A確率を作る関数
# =========================================================

def predict_goal_probabilities(
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

    pred_hg = home_model.predict(
        target_df[ALL_FEATURES]
    )

    pred_ag = away_model.predict(
        target_df[ALL_FEATURES]
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

    for home_xg, away_xg in zip(
        pred_hg,
        pred_ag
    ):

        ph, pd_, pa = (
            calculate_match_probabilities(
                home_xg,
                away_xg
            )
        )

        probabilities.append(
            [
                ph,
                pd_,
                pa
            ]
        )

    return (
        pred_hg,
        pred_ag,
        np.array(probabilities)
    )


# =========================================================
# 実際の結果
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
# STEP 1
# 2023以前 → 2024を予測
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


_, _, probs_2024 = (
    predict_goal_probabilities(
        train_pre_2024,
        validation_2024
    )
)


# =========================================================
# 2024確率を校正モデルへ
#
# log(p)を使います。
# =========================================================

EPSILON = 1e-8


calibration_X = np.log(
    np.clip(
        probs_2024,
        EPSILON,
        1.0
    )
)


calibration_y = (
    validation_2024[
        "Result"
    ]
)


calibrator = LogisticRegression(
    max_iter=2000,
    C=1.0
)


calibrator.fit(
    calibration_X,
    calibration_y
)


# =========================================================
# STEP 2
# 2024以前全部 → 2025得点予測
# =========================================================

train_pre_2025 = (
    data[
        data["Season"] != "2025"
    ]
    .copy()
)


test_2025 = (
    data[
        data["Season"] == "2025"
    ]
    .copy()
    .reset_index(drop=True)
)


pred_hg_2025, pred_ag_2025, raw_probs = (
    predict_goal_probabilities(
        train_pre_2025,
        test_2025
    )
)


# =========================================================
# 2024で学習した校正を2025へ
# =========================================================

calibration_test_X = np.log(
    np.clip(
        raw_probs,
        EPSILON,
        1.0
    )
)


calibrated_raw = (
    calibrator.predict_proba(
        calibration_test_X
    )
)


# sklearnのclasses_順を確認して並び替える
class_index = {
    label: index
    for index, label
    in enumerate(
        calibrator.classes_
    )
}


calibrated_probs = np.column_stack(
    [
        calibrated_raw[
            :,
            class_index["H"]
        ],

        calibrated_raw[
            :,
            class_index["D"]
        ],

        calibrated_raw[
            :,
            class_index["A"]
        ]
    ]
)


# =========================================================
# DataFrameへ保存
# =========================================================

test_2025["Pred_HG"] = pred_hg_2025
test_2025["Pred_AG"] = pred_ag_2025


# G2相当の補正前確率
test_2025["Raw_H"] = raw_probs[:, 0]
test_2025["Raw_D"] = raw_probs[:, 1]
test_2025["Raw_A"] = raw_probs[:, 2]


# G3校正後
test_2025["Prob_H"] = calibrated_probs[:, 0]
test_2025["Prob_D"] = calibrated_probs[:, 1]
test_2025["Prob_A"] = calibrated_probs[:, 2]


# =========================================================
# 最大確率予想
# =========================================================

labels = np.array(
    ["H", "D", "A"]
)


test_2025["Prediction"] = (
    labels[
        np.argmax(
            calibrated_probs,
            axis=1
        )
    ]
)


# =========================================================
# 評価関数
# =========================================================

def evaluate_probabilities(
    df,
    h_col,
    d_col,
    a_col
):

    probabilities = (
        df[
            [
                h_col,
                d_col,
                a_col
            ]
        ]
        .to_numpy()
    )

    predictions = (
        labels[
            np.argmax(
                probabilities,
                axis=1
            )
        ]
    )

    actual = (
        df["Result"]
        .to_numpy()
    )

    accuracy = np.mean(
        predictions == actual
    )

    correct = np.sum(
        predictions == actual
    )

    actual_indices = np.array(
        [
            0 if result == "H"
            else 1 if result == "D"
            else 2
            for result in actual
        ]
    )

    true_probs = (
        probabilities[
            np.arange(
                len(df)
            ),
            actual_indices
        ]
    )

    true_probs = np.clip(
        true_probs,
        1e-15,
        1.0
    )

    log_loss = -np.mean(
        np.log(
            true_probs
        )
    )

    actual_onehot = np.zeros(
        (
            len(df),
            3
        )
    )

    actual_onehot[
        np.arange(
            len(df)
        ),
        actual_indices
    ] = 1

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
        actual == "D"
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


raw_metrics = (
    evaluate_probabilities(
        test_2025,
        "Raw_H",
        "Raw_D",
        "Raw_A"
    )
)


g3_metrics = (
    evaluate_probabilities(
        test_2025,
        "Prob_H",
        "Prob_D",
        "Prob_A"
    )
)


# =========================================================
# メイン結果
# =========================================================

st.header(
    "🏆 G3：2025年 全380試合"
)


c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "正解率",
    f"{g3_metrics['Accuracy']:.1%}"
)

c2.metric(
    "Log Loss",
    f"{g3_metrics['LogLoss']:.4f}"
)

c3.metric(
    "Brier Score",
    f"{g3_metrics['Brier']:.4f}"
)

c4.metric(
    "Draw Recall",
    f"{g3_metrics['DrawRecall']:.1%}"
)


st.write(
    "正解数:",
    g3_metrics["Correct"],
    "/",
    len(test_2025)
)


# =========================================================
# 比較
# =========================================================

st.subheader(
    "🆚 モデル比較"
)


comparison = pd.DataFrame(
    {
        "モデル": [
            "M8",
            "G1",
            "G2",
            "G3"
        ],

        "Accuracy": [
            48.4,
            47.1,
            round(
                raw_metrics[
                    "Accuracy"
                ] * 100,
                1
            ),
            round(
                g3_metrics[
                    "Accuracy"
                ] * 100,
                1
            )
        ],

        "Log Loss": [
            1.0422,
            1.0467,
            round(
                raw_metrics[
                    "LogLoss"
                ],
                4
            ),
            round(
                g3_metrics[
                    "LogLoss"
                ],
                4
            )
        ],

        "Brier": [
            0.6279,
            0.6297,
            round(
                raw_metrics[
                    "Brier"
                ],
                4
            ),
            round(
                g3_metrics[
                    "Brier"
                ],
                4
            )
        ],

        "Draw Recall": [
            0.0,
            0.0,
            round(
                raw_metrics[
                    "DrawRecall"
                ] * 100,
                1
            ),
            round(
                g3_metrics[
                    "DrawRecall"
                ] * 100,
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


st.caption(
    "G2行は、このG3コード内で再計算した"
    "校正前の2025確率です。"
)


# =========================================================
# 平均確率
# =========================================================

st.subheader(
    "🎯 実際率 vs G2確率 vs G3確率"
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
                test_2025[
                    "Result"
                ] == "H"
            ).mean(),

            (
                test_2025[
                    "Result"
                ] == "D"
            ).mean(),

            (
                test_2025[
                    "Result"
                ] == "A"
            ).mean()
        ],

        "G2平均確率": [
            test_2025[
                "Raw_H"
            ].mean(),

            test_2025[
                "Raw_D"
            ].mean(),

            test_2025[
                "Raw_A"
            ].mean()
        ],

        "G3平均確率": [
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
    "G3平均確率"
]:

    probability_check[
        column
    ] = (
        probability_check[
            column
        ]
        * 100
    ).round(1)


st.dataframe(
    probability_check,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# Draw分析
# =========================================================

st.subheader(
    "🤝 G3 Draw分析"
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
# 確率帯ごとの校正
# =========================================================

st.header(
    "📏 確率の校正チェック"
)


def calibration_table(
    df,
    probability_column,
    target_result
):

    bins = [
        0.00,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
        0.60,
        1.01
    ]

    labels_bins = [
        "0-20%",
        "20-25%",
        "25-30%",
        "30-35%",
        "35-40%",
        "40-50%",
        "50-60%",
        "60%+"
    ]

    temp = df.copy()

    temp["Band"] = pd.cut(
        temp[
            probability_column
        ],
        bins=bins,
        labels=labels_bins,
        right=False
    )

    temp["Actual"] = (
        temp["Result"]
        ==
        target_result
    ).astype(int)

    result = (
        temp
        .groupby(
            "Band",
            observed=True
        )
        .agg(
            試合数=(
                "Actual",
                "size"
            ),

            平均予測確率=(
                probability_column,
                "mean"
            ),

            実際発生率=(
                "Actual",
                "mean"
            )
        )
        .reset_index()
    )

    result[
        "平均予測確率"
    ] = (
        result[
            "平均予測確率"
        ]
        * 100
    ).round(1)

    result[
        "実際発生率"
    ] = (
        result[
            "実際発生率"
        ]
        * 100
    ).round(1)

    return result


st.subheader(
    "🤝 Draw"
)


draw_calibration = (
    calibration_table(
        test_2025,
        "Prob_D",
        "D"
    )
)


st.dataframe(
    draw_calibration,
    hide_index=True,
    use_container_width=True
)


st.subheader(
    "🏠 Home"
)


home_calibration = (
    calibration_table(
        test_2025,
        "Prob_H",
        "H"
    )
)


st.dataframe(
    home_calibration,
    hide_index=True,
    use_container_width=True
)


st.subheader(
    "✈️ Away"
)


away_calibration = (
    calibration_table(
        test_2025,
        "Prob_A",
        "A"
    )
)


st.dataframe(
    away_calibration,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 個別試合
# =========================================================

st.header(
    "🔍 個別試合 + ランダム予想"
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


# =========================================================
# G2 → G3比較
# =========================================================

individual_probability = pd.DataFrame(
    {
        "結果": [
            "Home",
            "Draw",
            "Away"
        ],

        "G2": [
            selected["Raw_H"],
            selected["Raw_D"],
            selected["Raw_A"]
        ],

        "G3": [
            selected["Prob_H"],
            selected["Prob_D"],
            selected["Prob_A"]
        ]
    }
)


individual_probability[
    "G2"
] = (
    individual_probability[
        "G2"
    ]
    * 100
).round(1)


individual_probability[
    "G3"
] = (
    individual_probability[
        "G3"
    ]
    * 100
).round(1)


st.dataframe(
    individual_probability,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# G3ランダムサンプリング
# =========================================================

sample_key = (
    "g3_sample_"
    +
    str(selected_index)
)


if st.button(
    "🎲 G3確率から1回予想"
):

    sampled = np.random.choice(
        ["H", "D", "A"],
        p=[
            selected["Prob_H"],
            selected["Prob_D"],
            selected["Prob_A"]
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

        text = (
            f"🏠 {selected['Home']} 勝ち"
        )

    elif sampled == "D":

        text = "🤝 引き分け"

    else:

        text = (
            f"✈️ {selected['Away']} 勝ち"
        )

    st.success(
        "今回の予想："
        +
        text
    )


# =========================================================
# 1000回シミュレーション
# =========================================================

if st.button(
    "🎰 1000回シミュレーション"
):

    simulations = np.random.choice(
        ["H", "D", "A"],
        size=1000,
        p=[
            selected["Prob_H"],
            selected["Prob_D"],
            selected["Prob_A"]
        ]
    )

    simulation_result = (
        pd.Series(
            simulations
        )
        .value_counts()
        .reindex(
            ["H", "D", "A"],
            fill_value=0
        )
    )

    simulation_table = pd.DataFrame(
        {
            "結果": [
                "H",
                "D",
                "A"
            ],

            "回数": [
                int(
                    simulation_result[
                        "H"
                    ]
                ),

                int(
                    simulation_result[
                        "D"
                    ]
                ),

                int(
                    simulation_result[
                        "A"
                    ]
                )
            ]
        }
    )

    st.dataframe(
        simulation_table,
        hide_index=True
    )


# =========================================================
# 実際
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
    "G3の確率補正は2024年だけから学習しています。"
    "2025年の実際の結果を使って確率を合わせてはいません。"
)
