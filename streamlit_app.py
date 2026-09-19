import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss


# ==================================================
# 基本設定
# ==================================================

st.set_page_config(
    page_title="J1 Match Predictor",
    page_icon="⚽",
)

st.title("⚽ J1 Match Predictor")
st.write(
    "2025年J1リーグの試合データを使って予測モデルを比較します。"
)


# ==================================================
# データ読み込み
# ==================================================

df = pd.read_csv("data/JPN.csv")

df_2025 = df[
    (df["Season"] == "2025") &
    (df["League"] == "J1 League")
].copy()

df_2025["Date"] = pd.to_datetime(
    df_2025["Date"],
    dayfirst=True
)

df_2025 = (
    df_2025
    .sort_values("Date")
    .reset_index(drop=True)
)


# ==================================================
# 手作りモデル共通関数
# ==================================================

def strength_to_probabilities(
    strength,
    prefix
):

    home_score = np.exp(
        strength / 10
    )

    away_score = np.exp(
        -strength / 10
    )

    draw_score = 1.0

    total = (
        home_score +
        draw_score +
        away_score
    )

    return pd.Series({
        f"{prefix}_Prob_H":
            home_score / total,

        f"{prefix}_Prob_D":
            draw_score / total,

        f"{prefix}_Prob_A":
            away_score / total
    })


# ==================================================
# M1〜M6
# ==================================================

def m1_probabilities(
    points_diff
):

    return strength_to_probabilities(
        points_diff,
        "M1"
    )


def m2_probabilities(
    points_diff
):

    strength = (
        points_diff +
        2.0
    )

    return strength_to_probabilities(
        strength,
        "M2"
    )


def m3_probabilities(
    points_diff,
    form_diff
):

    strength = (
        points_diff +
        2.0 +
        form_diff * 0.5
    )

    return strength_to_probabilities(
        strength,
        "M3"
    )


def m4_probabilities(
    points_diff,
    form_diff,
    goal_diff_diff
):

    strength = (
        points_diff +
        2.0 +
        form_diff * 0.5 +
        goal_diff_diff * 0.2
    )

    return strength_to_probabilities(
        strength,
        "M4"
    )


def m5_probabilities(
    points_diff,
    form_diff,
    goal_diff_diff,
    elo_diff
):

    strength = (
        points_diff +
        2.0 +
        form_diff * 0.5 +
        goal_diff_diff * 0.2 +
        elo_diff * 0.02
    )

    return strength_to_probabilities(
        strength,
        "M5"
    )


def m6_probabilities(
    points_diff,
    form_diff,
    goal_diff_diff,
    elo_diff,
    attack_defense_diff
):

    strength = (
        points_diff +
        2.0 +
        form_diff * 0.5 +
        goal_diff_diff * 0.2 +
        elo_diff * 0.02 +
        attack_defense_diff * 2.0
    )

    return strength_to_probabilities(
        strength,
        "M6"
    )


# ==================================================
# 試合前特徴量を作成
# ==================================================

points = {}
recent_points = {}

goals_for = {}
goals_against = {}

matches_played = {}

elo = {}

K_FACTOR = 20

model_rows = []


for _, match in df_2025.iterrows():

    home = match["Home"]
    away = match["Away"]

    hg = match["HG"]
    ag = match["AG"]

    # ----------------------------------------------
    # 試合前の勝点
    # ----------------------------------------------

    home_points = points.get(
        home,
        0
    )

    away_points = points.get(
        away,
        0
    )


    # ----------------------------------------------
    # 直近5試合
    # ----------------------------------------------

    home_recent = recent_points.get(
        home,
        []
    )

    away_recent = recent_points.get(
        away,
        []
    )

    home_form = sum(
        home_recent[-5:]
    )

    away_form = sum(
        away_recent[-5:]
    )


    # ----------------------------------------------
    # 得点・失点
    # ----------------------------------------------

    home_gf = goals_for.get(
        home,
        0
    )

    home_ga = goals_against.get(
        home,
        0
    )

    away_gf = goals_for.get(
        away,
        0
    )

    away_ga = goals_against.get(
        away,
        0
    )


    # ----------------------------------------------
    # 試合数
    # ----------------------------------------------

    home_matches = matches_played.get(
        home,
        0
    )

    away_matches = matches_played.get(
        away,
        0
    )


    # ----------------------------------------------
    # 得失点差
    # ----------------------------------------------

    home_goal_diff = (
        home_gf -
        home_ga
    )

    away_goal_diff = (
        away_gf -
        away_ga
    )


    # ----------------------------------------------
    # 攻撃・守備
    # ----------------------------------------------

    if home_matches > 0:

        home_attack = (
            home_gf /
            home_matches
        )

        home_defense = (
            home_ga /
            home_matches
        )

    else:

        home_attack = 0.0
        home_defense = 0.0


    if away_matches > 0:

        away_attack = (
            away_gf /
            away_matches
        )

        away_defense = (
            away_ga /
            away_matches
        )

    else:

        away_attack = 0.0
        away_defense = 0.0


    home_attack_edge = (
        home_attack -
        away_defense
    )

    away_attack_edge = (
        away_attack -
        home_defense
    )

    attack_defense_diff = (
        home_attack_edge -
        away_attack_edge
    )


    # ----------------------------------------------
    # Elo
    # ----------------------------------------------

    home_elo = elo.get(
        home,
        1500.0
    )

    away_elo = elo.get(
        away,
        1500.0
    )


    # ----------------------------------------------
    # 試合前情報を保存
    # ----------------------------------------------

    model_rows.append({

        "Date":
            match["Date"],

        "Home":
            home,

        "Away":
            away,

        "HomePointsBefore":
            home_points,

        "AwayPointsBefore":
            away_points,

        "PointsDiff":
            home_points -
            away_points,

        "HomeForm5":
            home_form,

        "AwayForm5":
            away_form,

        "HomeGoalDiffBefore":
            home_goal_diff,

        "AwayGoalDiffBefore":
            away_goal_diff,

        "HomeEloBefore":
            home_elo,

        "AwayEloBefore":
            away_elo,

        "HomeAttackBefore":
            home_attack,

        "AwayAttackBefore":
            away_attack,

        "HomeDefenseBefore":
            home_defense,

        "AwayDefenseBefore":
            away_defense,

        "AttackDefenseDiff":
            attack_defense_diff
    })


    # ==================================================
    # ここから試合終了後の更新
    # ==================================================

    if hg > ag:

        home_match_points = 3
        away_match_points = 0

        home_actual = 1.0
        away_actual = 0.0

    elif hg < ag:

        home_match_points = 0
        away_match_points = 3

        home_actual = 0.0
        away_actual = 1.0

    else:

        home_match_points = 1
        away_match_points = 1

        home_actual = 0.5
        away_actual = 0.5


    # 勝点
    points[home] = (
        home_points +
        home_match_points
    )

    points[away] = (
        away_points +
        away_match_points
    )


    # 直近成績
    recent_points.setdefault(
        home,
        []
    ).append(
        home_match_points
    )

    recent_points.setdefault(
        away,
        []
    ).append(
        away_match_points
    )


    # 得点・失点
    goals_for[home] = (
        home_gf +
        hg
    )

    goals_against[home] = (
        home_ga +
        ag
    )

    goals_for[away] = (
        away_gf +
        ag
    )

    goals_against[away] = (
        away_ga +
        hg
    )


    # 試合数
    matches_played[home] = (
        home_matches + 1
    )

    matches_played[away] = (
        away_matches + 1
    )


    # Elo
    home_expected = (
        1 /
        (
            1 +
            10 ** (
                (
                    away_elo -
                    home_elo
                ) /
                400
            )
        )
    )

    away_expected = (
        1 -
        home_expected
    )

    elo[home] = (
        home_elo +
        K_FACTOR *
        (
            home_actual -
            home_expected
        )
    )

    elo[away] = (
        away_elo +
        K_FACTOR *
        (
            away_actual -
            away_expected
        )
    )


# ==================================================
# DataFrame
# ==================================================

model_data = pd.DataFrame(
    model_rows
)


model_data["FormDiff"] = (
    model_data["HomeForm5"] -
    model_data["AwayForm5"]
)


model_data["GoalDiffDiff"] = (
    model_data[
        "HomeGoalDiffBefore"
    ] -
    model_data[
        "AwayGoalDiffBefore"
    ]
)


model_data["EloDiff"] = (
    model_data[
        "HomeEloBefore"
    ] -
    model_data[
        "AwayEloBefore"
    ]
)


# ホーム開催を示す特徴量
# 全行ホームチーム視点なので1
model_data["HomeFlag"] = 1.0


# ==================================================
# 実際の結果
# ==================================================

model_data["Result"] = [
    "H" if hg > ag else
    "A" if hg < ag else
    "D"

    for hg, ag in zip(
        df_2025["HG"],
        df_2025["AG"]
    )
]


actual_h = (
    model_data["Result"] == "H"
).astype(int)

actual_d = (
    model_data["Result"] == "D"
).astype(int)

actual_a = (
    model_data["Result"] == "A"
).astype(int)


# ==================================================
# 評価関数
# ==================================================

def calculate_metrics(
    data,
    prediction_column,
    prob_h,
    prob_d,
    prob_a
):

    accuracy = (
        data[prediction_column] ==
        data["Result"]
    ).mean()

    actual_prob = np.where(
        data["Result"] == "H",
        data[prob_h],
        np.where(
            data["Result"] == "D",
            data[prob_d],
            data[prob_a]
        )
    )

    actual_prob = np.clip(
        actual_prob,
        1e-15,
        1.0
    )

    model_log_loss = -np.mean(
        np.log(actual_prob)
    )

    brier = np.mean(
        (
            data[prob_h] -
            (
                data["Result"] == "H"
            ).astype(int)
        ) ** 2
        +
        (
            data[prob_d] -
            (
                data["Result"] == "D"
            ).astype(int)
        ) ** 2
        +
        (
            data[prob_a] -
            (
                data["Result"] == "A"
            ).astype(int)
        ) ** 2
    )

    return (
        accuracy,
        model_log_loss,
        brier
    )


def make_prediction(
    data,
    columns
):

    return (
        data[columns]
        .idxmax(axis=1)
        .map({
            columns[0]: "H",
            columns[1]: "D",
            columns[2]: "A"
        })
    )


# ==================================================
# M1
# ==================================================

m1_probs = model_data[
    "PointsDiff"
].apply(
    m1_probabilities
)

model_data = pd.concat(
    [model_data, m1_probs],
    axis=1
)

model_data["M1_Prediction"] = (
    model_data[
        "PointsDiff"
    ].apply(
        lambda x:
        "H" if x > 0 else
        "A" if x < 0 else
        "D"
    )
)


# ==================================================
# M2
# ==================================================

m2_probs = model_data[
    "PointsDiff"
].apply(
    m2_probabilities
)

model_data = pd.concat(
    [model_data, m2_probs],
    axis=1
)

model_data["M2_Prediction"] = (
    make_prediction(
        model_data,
        [
            "M2_Prob_H",
            "M2_Prob_D",
            "M2_Prob_A"
        ]
    )
)


# ==================================================
# M3
# ==================================================

m3_probs = model_data.apply(
    lambda row:
    m3_probabilities(
        row["PointsDiff"],
        row["FormDiff"]
    ),
    axis=1
)

model_data = pd.concat(
    [model_data, m3_probs],
    axis=1
)

model_data["M3_Prediction"] = (
    make_prediction(
        model_data,
        [
            "M3_Prob_H",
            "M3_Prob_D",
            "M3_Prob_A"
        ]
    )
)


# ==================================================
# M4
# ==================================================

m4_probs = model_data.apply(
    lambda row:
    m4_probabilities(
        row["PointsDiff"],
        row["FormDiff"],
        row["GoalDiffDiff"]
    ),
    axis=1
)

model_data = pd.concat(
    [model_data, m4_probs],
    axis=1
)

model_data["M4_Prediction"] = (
    make_prediction(
        model_data,
        [
            "M4_Prob_H",
            "M4_Prob_D",
            "M4_Prob_A"
        ]
    )
)


# ==================================================
# M5
# ==================================================

m5_probs = model_data.apply(
    lambda row:
    m5_probabilities(
        row["PointsDiff"],
        row["FormDiff"],
        row["GoalDiffDiff"],
        row["EloDiff"]
    ),
    axis=1
)

model_data = pd.concat(
    [model_data, m5_probs],
    axis=1
)

model_data["M5_Prediction"] = (
    make_prediction(
        model_data,
        [
            "M5_Prob_H",
            "M5_Prob_D",
            "M5_Prob_A"
        ]
    )
)


# ==================================================
# M6
# ==================================================

m6_probs = model_data.apply(
    lambda row:
    m6_probabilities(
        row["PointsDiff"],
        row["FormDiff"],
        row["GoalDiffDiff"],
        row["EloDiff"],
        row["AttackDefenseDiff"]
    ),
    axis=1
)

model_data = pd.concat(
    [model_data, m6_probs],
    axis=1
)

model_data["M6_Prediction"] = (
    make_prediction(
        model_data,
        [
            "M6_Prob_H",
            "M6_Prob_D",
            "M6_Prob_A"
        ]
    )
)


# ==================================================
# M1〜M6の380試合評価
# ==================================================

manual_metrics = {}


for model in [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6"
]:

    manual_metrics[model] = (
        calculate_metrics(
            model_data,
            f"{model}_Prediction",
            f"{model}_Prob_H",
            f"{model}_Prob_D",
            f"{model}_Prob_A"
        )
    )


# ==================================================
# M7
# ロジスティック回帰
# ウォークフォワード検証
# ==================================================

feature_columns = [
    "PointsDiff",
    "FormDiff",
    "GoalDiffDiff",
    "EloDiff",
    "AttackDefenseDiff"
]


INITIAL_TRAIN_SIZE = 100


m7_predictions = []

m7_prob_h = []
m7_prob_d = []
m7_prob_a = []

m7_indexes = []


# ----------------------------------------------
# 100試合を学習した後、
# 101試合目を予測
#
# 次に101試合を学習して
# 102試合目を予測
#
# という処理を繰り返す
# ----------------------------------------------

for test_index in range(
    INITIAL_TRAIN_SIZE,
    len(model_data)
):

    train_data = (
        model_data
        .iloc[:test_index]
        .copy()
    )

    test_data = (
        model_data
        .iloc[[test_index]]
        .copy()
    )


    X_train = (
        train_data[
            feature_columns
        ]
    )

    y_train = (
        train_data["Result"]
    )

    X_test = (
        test_data[
            feature_columns
        ]
    )


    # ------------------------------------------
    # 標準化＋ロジスティック回帰
    # ------------------------------------------

    m7_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ])


    m7_model.fit(
        X_train,
        y_train
    )


    probabilities = (
        m7_model.predict_proba(
            X_test
        )[0]
    )

    classes = (
        m7_model
        .named_steps["model"]
        .classes_
    )


    probability_map = dict(
        zip(
            classes,
            probabilities
        )
    )


    prob_h = (
        probability_map.get(
            "H",
            0.0
        )
    )

    prob_d = (
        probability_map.get(
            "D",
            0.0
        )
    )

    prob_a = (
        probability_map.get(
            "A",
            0.0
        )
    )


    prediction = max(
        {
            "H": prob_h,
            "D": prob_d,
            "A": prob_a
        },
        key={
            "H": prob_h,
            "D": prob_d,
            "A": prob_a
        }.get
    )


    m7_indexes.append(
        test_index
    )

    m7_predictions.append(
        prediction
    )

    m7_prob_h.append(
        prob_h
    )

    m7_prob_d.append(
        prob_d
    )

    m7_prob_a.append(
        prob_a
    )


# ==================================================
# M7評価用データ
# ==================================================

m7_data = (
    model_data
    .iloc[m7_indexes]
    .copy()
)


m7_data[
    "M7_Prediction"
] = m7_predictions

m7_data[
    "M7_Prob_H"
] = m7_prob_h

m7_data[
    "M7_Prob_D"
] = m7_prob_d

m7_data[
    "M7_Prob_A"
] = m7_prob_a


m7_accuracy, m7_log_loss, m7_brier = (
    calculate_metrics(
        m7_data,
        "M7_Prediction",
        "M7_Prob_H",
        "M7_Prob_D",
        "M7_Prob_A"
    )
)


# ==================================================
# M7 引き分けRecall
# ==================================================

m7_actual_draws = (
    m7_data["Result"] == "D"
).sum()


m7_correct_draws = (
    (
        m7_data["Result"] == "D"
    )
    &
    (
        m7_data[
            "M7_Prediction"
        ] == "D"
    )
).sum()


if m7_actual_draws > 0:

    m7_draw_recall = (
        m7_correct_draws /
        m7_actual_draws
    )

else:

    m7_draw_recall = 0.0


# ==================================================
# 公平比較
# M2もM7と同じ280試合だけで評価
# ==================================================

m2_same_period = (
    model_data
    .iloc[
        INITIAL_TRAIN_SIZE:
    ]
    .copy()
)


(
    m2_same_accuracy,
    m2_same_log_loss,
    m2_same_brier
) = calculate_metrics(
    m2_same_period,
    "M2_Prediction",
    "M2_Prob_H",
    "M2_Prob_D",
    "M2_Prob_A"
)


m2_same_actual_draws = (
    m2_same_period[
        "Result"
    ] == "D"
).sum()


m2_same_correct_draws = (
    (
        m2_same_period[
            "Result"
        ] == "D"
    )
    &
    (
        m2_same_period[
            "M2_Prediction"
        ] == "D"
    )
).sum()


if m2_same_actual_draws > 0:

    m2_same_draw_recall = (
        m2_same_correct_draws /
        m2_same_actual_draws
    )

else:

    m2_same_draw_recall = 0.0


# ==================================================
# 表示
# ==================================================

st.header(
    "📊 M1〜M6 手作りモデル"
)


comparison_rows = []


model_names = {

    "M1":
        "勝点差",

    "M2":
        "＋ホーム補正",

    "M3":
        "＋直近5試合",

    "M4":
        "＋得失点差",

    "M5":
        "＋Elo",

    "M6":
        "＋攻撃力・守備力"
}


for model in [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6"
]:

    accuracy = (
        manual_metrics[
            model
        ][0]
    )

    model_log_loss = (
        manual_metrics[
            model
        ][1]
    )

    brier = (
        manual_metrics[
            model
        ][2]
    )


    comparison_rows.append({

        "モデル":
            model,

        "内容":
            model_names[model],

        "評価試合数":
            len(model_data),

        "正解率":
            round(
                accuracy * 100,
                1
            ),

        "Log Loss":
            round(
                model_log_loss,
                4
            ),

        "Brier Score":
            round(
                brier,
                4
            )
    })


comparison = pd.DataFrame(
    comparison_rows
)


st.dataframe(
    comparison,
    hide_index=True
)


# ==================================================
# M7表示
# ==================================================

st.header(
    "🤖 M7：ロジスティック回帰"
)


st.write(
    "方式：過去の試合だけで学習し、次の試合を予測"
)

st.write(
    "初期学習試合数:",
    INITIAL_TRAIN_SIZE
)

st.write(
    "評価対象試合数:",
    len(m7_data)
)


st.metric(
    "M7 正解率",
    f"{m7_accuracy:.1%}"
)

st.metric(
    "M7 Log Loss",
    f"{m7_log_loss:.4f}"
)

st.metric(
    "M7 Brier Score",
    f"{m7_brier:.4f}"
)

st.metric(
    "M7 引き分けRecall",
    f"{m7_draw_recall:.1%}"
)


# ==================================================
# M2 vs M7
# 同じ試合だけで比較
# ==================================================

st.header(
    "⚖️ M2 vs M7 公平比較"
)


fair_comparison = pd.DataFrame({

    "モデル": [
        "M2",
        "M7"
    ],

    "評価試合数": [
        len(m2_same_period),
        len(m7_data)
    ],

    "正解率": [
        round(
            m2_same_accuracy * 100,
            1
        ),
        round(
            m7_accuracy * 100,
            1
        )
    ],

    "Log Loss": [
        round(
            m2_same_log_loss,
            4
        ),
        round(
            m7_log_loss,
            4
        )
    ],

    "Brier Score": [
        round(
            m2_same_brier,
            4
        ),
        round(
            m7_brier,
            4
        )
    ],

    "引き分けRecall": [
        round(
            m2_same_draw_recall *
            100,
            1
        ),
        round(
            m7_draw_recall *
            100,
            1
        )
    ]
})


st.dataframe(
    fair_comparison,
    hide_index=True
)


# ==================================================
# M7予測内訳
# ==================================================

st.header(
    "🔍 M7 予測内訳"
)


m7_prediction_counts = pd.DataFrame({

    "結果": [
        "H",
        "D",
        "A"
    ],

    "実際": [
        int(
            (
                m7_data[
                    "Result"
                ] == "H"
            ).sum()
        ),

        int(
            (
                m7_data[
                    "Result"
                ] == "D"
            ).sum()
        ),

        int(
            (
                m7_data[
                    "Result"
                ] == "A"
            ).sum()
        )
    ],

    "M7予測": [
        int(
            (
                m7_data[
                    "M7_Prediction"
                ] == "H"
            ).sum()
        ),

        int(
            (
                m7_data[
                    "M7_Prediction"
                ] == "D"
            ).sum()
        ),

        int(
            (
                m7_data[
                    "M7_Prediction"
                ] == "A"
            ).sum()
        )
    ]
})


st.dataframe(
    m7_prediction_counts,
    hide_index=True
)


# ==================================================
# 最近20試合のM7予測
# ==================================================

st.header(
    "🧪 M7 最近20試合"
)


recent_m7 = (
    m7_data[
        [
            "Date",
            "Home",
            "Away",
            "Result",
            "M7_Prediction",
            "M7_Prob_H",
            "M7_Prob_D",
            "M7_Prob_A"
        ]
    ]
    .tail(20)
    .copy()
)


recent_m7[
    "M7_Prob_H"
] = (
    recent_m7[
        "M7_Prob_H"
    ] * 100
).round(1)


recent_m7[
    "M7_Prob_D"
] = (
    recent_m7[
        "M7_Prob_D"
    ] * 100
).round(1)


recent_m7[
    "M7_Prob_A"
] = (
    recent_m7[
        "M7_Prob_A"
    ] * 100
).round(1)


st.dataframe(
    recent_m7,
    hide_index=True
)


# ==================================================
# 試合予測画面
# ==================================================

st.header(
    "⚽ 試合予測"
)


teams = [
    "鹿島アントラーズ",
    "浦和レッズ",
    "柏レイソル",
    "FC東京",
    "東京ヴェルディ",
    "FC町田ゼルビア",
    "川崎フロンターレ",
    "横浜F・マリノス",
    "横浜FC",
    "湘南ベルマーレ",
    "アルビレックス新潟",
    "清水エスパルス",
    "名古屋グランパス",
    "京都サンガF.C.",
    "ガンバ大阪",
    "セレッソ大阪",
    "ヴィッセル神戸",
    "ファジアーノ岡山",
    "サンフレッチェ広島",
    "アビスパ福岡",
]


home_team = st.selectbox(
    "🏠 ホームチーム",
    teams
)

away_team = st.selectbox(
    "✈️ アウェイチーム",
    teams,
    index=1
)


if st.button(
    "試合を予測する"
):

    if (
        home_team ==
        away_team
    ):

        st.error(
            "ホームとアウェイには別のチームを選んでください。"
        )

    else:

        st.success(
            f"{home_team} vs {away_team}"
        )

        st.info(
            "実戦用M7予測は次のステップで接続します。"
        )
