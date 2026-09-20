import streamlit as st
import pandas as pd
import numpy as np

from collections import defaultdict, deque
from math import exp, factorial

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor


# =========================================================
# 基本設定
# =========================================================

st.set_page_config(
    page_title="J1 Goal Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ J1 Goal Predictor - G1")

st.write(
    "チーム能力から両チームの予想得点を計算し、"
    "Poisson分布からHome / Draw / Away確率を求めます。"
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

    df["Season"] = (
        df["Season"]
        .astype(str)
    )

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

    df["OriginalOrder"] = range(
        len(df)
    )

    df = (
        df
        .sort_values(
            [
                "Date",
                "OriginalOrder"
            ]
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

    return (
        sum(values)
        /
        len(values)
    )


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

        hg = float(
            match["HG"]
        )

        ag = float(
            match["AG"]
        )


        # =================================================
        # 試合前情報
        # =================================================

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


        rows.append({

            "Season":
                str(
                    match["Season"]
                ),

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
                -
                away_elo,

            "Home_Form5":
                home_form5,

            "Away_Form5":
                away_form5,

            "FormDiff":
                home_form5
                -
                away_form5,

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
        })


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


        # -------------------------------------------------
        # Form
        # -------------------------------------------------

        recent_points_5[
            home
        ].append(hp)

        recent_points_5[
            away
        ].append(ap)


        # -------------------------------------------------
        # 直近10
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Home限定
        # -------------------------------------------------

        home_gf_10[
            home
        ].append(hg)

        home_ga_10[
            home
        ].append(ag)


        # -------------------------------------------------
        # Away限定
        # -------------------------------------------------

        away_gf_10[
            away
        ].append(ag)

        away_ga_10[
            away
        ].append(hg)


        # -------------------------------------------------
        # Elo
        # -------------------------------------------------

        expected_home = (
            1
            /
            (
                1
                +
                10 ** (
                    (
                        away_elo
                        -
                        home_elo
                    )
                    /
                    400
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
                -
                expected_home
            )
        )


        elo[away] = (
            away_elo
            +
            K_FACTOR
            *
            (
                (
                    1.0
                    -
                    home_actual
                )
                -
                (
                    1.0
                    -
                    expected_home
                )
            )
        )


    return pd.DataFrame(
        rows
    )


data = build_features(
    matches
)


# =========================================================
# G1特徴量
# =========================================================

FEATURES = [

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
# 欠損値処理
#
# 初登場チームなどは過去履歴がありません。
# 学習データの中央値で補います。
# =========================================================

def prepare_features(
    train_df,
    test_df
):

    X_train = (
        train_df[
            FEATURES
        ].copy()
    )

    X_test = (
        test_df[
            FEATURES
        ].copy()
    )


    medians = (
        X_train.median()
    )


    X_train = (
        X_train.fillna(
            medians
        )
    )

    X_test = (
        X_test.fillna(
            medians
        )
    )


    return (
        X_train,
        X_test
    )


# =========================================================
# 学習データ
#
# 2024年以前だけ
# =========================================================

train = (
    data[
        data["Season"]
        !=
        "2025"
    ]
    .copy()
)


# =========================================================
# テストデータ
#
# 2025年全380試合
# =========================================================

test = (
    data[
        data["Season"]
        ==
        "2025"
    ]
    .copy()
    .reset_index(drop=True)
)


X_train, X_test = (
    prepare_features(
        train,
        test
    )
)


# =========================================================
# Home得点モデル
# =========================================================

home_model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "poisson",
        PoissonRegressor(
            alpha=0.1,
            max_iter=1000
        )
    )
])


home_model.fit(
    X_train,
    train["HG"]
)


# =========================================================
# Away得点モデル
# =========================================================

away_model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "poisson",
        PoissonRegressor(
            alpha=0.1,
            max_iter=1000
        )
    )
])


away_model.fit(
    X_train,
    train["AG"]
)


# =========================================================
# 2025年予想得点
# =========================================================

test[
    "Pred_HG"
] = home_model.predict(
    X_test
)

test[
    "Pred_AG"
] = away_model.predict(
    X_test
)


# 異常な値を念のため制限
test[
    "Pred_HG"
] = test[
    "Pred_HG"
].clip(
    0.05,
    5.0
)

test[
    "Pred_AG"
] = test[
    "Pred_AG"
].clip(
    0.05,
    5.0
)


# =========================================================
# Poisson確率
# =========================================================

def poisson_probability(
    goals,
    expected_goals
):

    return (
        exp(
            -expected_goals
        )
        *
        (
            expected_goals
            **
            goals
        )
        /
        factorial(
            goals
        )
    )


# =========================================================
# H / D / A確率
# =========================================================

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

        home_probability = (
            poisson_probability(
                home_goals,
                home_expected
            )
        )


        for away_goals in range(
            max_goals + 1
        ):

            away_probability = (
                poisson_probability(
                    away_goals,
                    away_expected
                )
            )


            probability = (
                home_probability
                *
                away_probability
            )


            score_probabilities.append(
                (
                    home_goals,
                    away_goals,
                    probability
                )
            )


            if (
                home_goals
                >
                away_goals
            ):

                home_win += (
                    probability
                )

            elif (
                home_goals
                ==
                away_goals
            ):

                draw += (
                    probability
                )

            else:

                away_win += (
                    probability
                )


    total = (
        home_win
        +
        draw
        +
        away_win
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
# 全380試合を確率化
# =========================================================

prob_h_list = []
prob_d_list = []
prob_a_list = []

prediction_list = []


for _, row in test.iterrows():

    prob_h, prob_d, prob_a, _ = (
        calculate_match_probabilities(

            row[
                "Pred_HG"
            ],

            row[
                "Pred_AG"
            ]
        )
    )


    prob_h_list.append(
        prob_h
    )

    prob_d_list.append(
        prob_d
    )

    prob_a_list.append(
        prob_a
    )


    probabilities = {

        "H":
            prob_h,

        "D":
            prob_d,

        "A":
            prob_a
    }


    prediction_list.append(
        max(
            probabilities,
            key=probabilities.get
        )
    )


test[
    "Prob_H"
] = prob_h_list

test[
    "Prob_D"
] = prob_d_list

test[
    "Prob_A"
] = prob_a_list

test[
    "Prediction"
] = prediction_list


# =========================================================
# 実際のH/D/A
# =========================================================

def actual_result(row):

    if (
        row["HG"]
        >
        row["AG"]
    ):

        return "H"

    elif (
        row["HG"]
        <
        row["AG"]
    ):

        return "A"

    return "D"


test[
    "Result"
] = test.apply(
    actual_result,
    axis=1
)


# =========================================================
# 評価
# =========================================================

accuracy = (
    test[
        "Prediction"
    ]
    ==
    test[
        "Result"
    ]
).mean()


actual_probability = np.where(

    test[
        "Result"
    ]
    ==
    "H",

    test[
        "Prob_H"
    ],

    np.where(

        test[
            "Result"
        ]
        ==
        "D",

        test[
            "Prob_D"
        ],

        test[
            "Prob_A"
        ]
    )
)


actual_probability = np.clip(
    actual_probability,
    1e-15,
    1.0
)


log_loss = (
    -np.mean(
        np.log(
            actual_probability
        )
    )
)


actual_h = (
    test["Result"]
    ==
    "H"
).astype(int)

actual_d = (
    test["Result"]
    ==
    "D"
).astype(int)

actual_a = (
    test["Result"]
    ==
    "A"
).astype(int)


brier = np.mean(

    (
        test[
            "Prob_H"
        ]
        -
        actual_h
    ) ** 2

    +

    (
        test[
            "Prob_D"
        ]
        -
        actual_d
    ) ** 2

    +

    (
        test[
            "Prob_A"
        ]
        -
        actual_a
    ) ** 2
)


actual_draws = (
    test[
        "Result"
    ]
    ==
    "D"
).sum()


correct_draws = (

    (
        test[
            "Result"
        ]
        ==
        "D"
    )

    &

    (
        test[
            "Prediction"
        ]
        ==
        "D"
    )

).sum()


draw_recall = (
    correct_draws
    /
    actual_draws
)


correct_count = (
    test[
        "Prediction"
    ]
    ==
    test[
        "Result"
    ]
).sum()


# =========================================================
# 画面
# =========================================================

st.header(
    "🏆 G1：2025年 全380試合"
)


c1, c2, c3, c4 = (
    st.columns(4)
)


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
    int(
        correct_count
    ),
    "/",
    len(test)
)


# =========================================================
# 得点モデルチェック
# =========================================================

st.subheader(
    "⚽ 得点予測チェック"
)


goal_check = pd.DataFrame({

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

    "予測": [

        test[
            "Pred_HG"
        ].mean(),

        test[
            "Pred_AG"
        ].mean()
    ]
})


goal_check[
    "実際"
] = goal_check[
    "実際"
].round(3)

goal_check[
    "予測"
] = goal_check[
    "予測"
].round(3)


st.dataframe(
    goal_check,
    hide_index=True
)


# =========================================================
# H/D/A 内訳
# =========================================================

st.subheader(
    "📊 H / D / A 内訳"
)


breakdown = pd.DataFrame({

    "結果": [
        "H",
        "D",
        "A"
    ],

    "実際": [

        int(
            (
                test[
                    "Result"
                ]
                ==
                "H"
            ).sum()
        ),

        int(
            (
                test[
                    "Result"
                ]
                ==
                "D"
            ).sum()
        ),

        int(
            (
                test[
                    "Result"
                ]
                ==
                "A"
            ).sum()
        )
    ],

    "予測": [

        int(
            (
                test[
                    "Prediction"
                ]
                ==
                "H"
            ).sum()
        ),

        int(
            (
                test[
                    "Prediction"
                ]
                ==
                "D"
            ).sum()
        ),

        int(
            (
                test[
                    "Prediction"
                ]
                ==
                "A"
            ).sum()
        )
    ]
})


st.dataframe(
    breakdown,
    hide_index=True
)


# =========================================================
# 平均H/D/A確率
# =========================================================

st.subheader(
    "🎯 平均予測確率"
)


probability_summary = pd.DataFrame({

    "結果": [
        "Home",
        "Draw",
        "Away"
    ],

    "平均予測確率": [

        test[
            "Prob_H"
        ].mean(),

        test[
            "Prob_D"
        ].mean(),

        test[
            "Prob_A"
        ].mean()
    ]
})


probability_summary[
    "平均予測確率"
] = (
    probability_summary[
        "平均予測確率"
    ]
    *
    100
).round(1)


st.dataframe(
    probability_summary,
    hide_index=True
)


# =========================================================
# M8との比較
# =========================================================

st.subheader(
    "🆚 M8との比較"
)


comparison = pd.DataFrame({

    "モデル": [
        "M8",
        "G1"
    ],

    "正解率": [
        48.4,
        round(
            accuracy
            *
            100,
            1
        )
    ],

    "Log Loss": [
        1.0422,
        round(
            log_loss,
            4
        )
    ],

    "Brier Score": [
        0.6279,
        round(
            brier,
            4
        )
    ],

    "Draw Recall": [
        0.0,
        round(
            draw_recall
            *
            100,
            1
        )
    ]
})


st.dataframe(
    comparison,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 個別試合を見る
# =========================================================

st.header(
    "🔍 個別試合"
)


match_labels = []

for index, row in test.iterrows():

    label = (
        row[
            "Date"
        ].strftime(
            "%Y-%m-%d"
        )
        +
        " | "
        +
        row[
            "Home"
        ]
        +
        " vs "
        +
        row[
            "Away"
        ]
    )

    match_labels.append(
        (
            label,
            index
        )
    )


selected_label = st.selectbox(

    "試合を選択",

    [
        item[0]
        for item
        in match_labels
    ]
)


selected_index = dict(
    match_labels
)[
    selected_label
]


selected = (
    test.loc[
        selected_index
    ]
)


st.subheader(
    f"{selected['Home']} vs "
    f"{selected['Away']}"
)


# =========================================================
# 予想得点
# =========================================================

c1, c2 = st.columns(2)


c1.metric(
    f"{selected['Home']} 予想得点",
    f"{selected['Pred_HG']:.2f}"
)


c2.metric(
    f"{selected['Away']} 予想得点",
    f"{selected['Pred_AG']:.2f}"
)


# =========================================================
# H/D/A
# =========================================================

c1, c2, c3 = st.columns(3)


c1.metric(
    "🏠 Home",
    f"{selected['Prob_H']:.1%}"
)

c2.metric(
    "🤝 Draw",
    f"{selected['Prob_D']:.1%}"
)

c3.metric(
    "✈️ Away",
    f"{selected['Prob_A']:.1%}"
)


# =========================================================
# スコア確率
# =========================================================

_, _, _, score_probs = (
    calculate_match_probabilities(

        selected[
            "Pred_HG"
        ],

        selected[
            "Pred_AG"
        ]
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


score_table[
    "Score"
] = (

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


score_table[
    "確率"
] = (
    score_table[
        "Probability"
    ]
    *
    100
).round(1)


st.subheader(
    "🎲 最も起こりやすいスコア TOP10"
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
    "G1は2024年以前のデータだけで学習し、"
    "2025年380試合を学習に使わず予測しています。"
)
