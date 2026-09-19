import streamlit as st
import pandas as pd
import numpy as np


# =========================
# 基本設定
# =========================

st.set_page_config(
    page_title="J1 Match Predictor",
    page_icon="⚽",
)

st.title("⚽ J1 Match Predictor")
st.write("J1リーグの試合結果を予測するアプリです。")


# =========================
# データ読み込み
# =========================

df = pd.read_csv("data/JPN.csv")

# 2025年J1だけを取り出す
df_2025 = df[
    (df["Season"] == "2025") &
    (df["League"] == "J1 League")
].copy()

# 日付順に並べる
df_2025["Date"] = pd.to_datetime(
    df_2025["Date"],
    dayfirst=True
)

df_2025 = df_2025.sort_values("Date").reset_index(drop=True)


# =========================
# M1〜M4の確率計算関数
# =========================

# M1：勝点差のみ
def m1_probabilities(points_diff):

    home_score = np.exp(points_diff / 10)
    away_score = np.exp(-points_diff / 10)
    draw_score = 1.0

    total = home_score + draw_score + away_score

    return pd.Series({
        "Prob_H": home_score / total,
        "Prob_D": draw_score / total,
        "Prob_A": away_score / total
    })


# M2：勝点差＋ホーム補正
def m2_probabilities(points_diff):

    home_advantage = 2.0

    strength = points_diff + home_advantage

    home_score = np.exp(strength / 10)
    away_score = np.exp(-strength / 10)
    draw_score = 1.0

    total = home_score + draw_score + away_score

    return pd.Series({
        "M2_Prob_H": home_score / total,
        "M2_Prob_D": draw_score / total,
        "M2_Prob_A": away_score / total
    })


# M3：勝点差＋ホーム補正＋直近5試合
def m3_probabilities(points_diff, form_diff):

    home_advantage = 2.0
    form_weight = 0.5

    strength = (
        points_diff
        + home_advantage
        + form_diff * form_weight
    )

    home_score = np.exp(strength / 10)
    away_score = np.exp(-strength / 10)
    draw_score = 1.0

    total = home_score + draw_score + away_score

    return pd.Series({
        "M3_Prob_H": home_score / total,
        "M3_Prob_D": draw_score / total,
        "M3_Prob_A": away_score / total
    })


# M4：勝点差＋ホーム補正＋直近5試合＋得失点差
def m4_probabilities(
    points_diff,
    form_diff,
    goal_diff_diff
):

    home_advantage = 2.0
    form_weight = 0.5
    goal_diff_weight = 0.2

    strength = (
        points_diff
        + home_advantage
        + form_diff * form_weight
        + goal_diff_diff * goal_diff_weight
    )

    home_score = np.exp(strength / 10)
    away_score = np.exp(-strength / 10)
    draw_score = 1.0

    total = home_score + draw_score + away_score

    return pd.Series({
        "M4_Prob_H": home_score / total,
        "M4_Prob_D": draw_score / total,
        "M4_Prob_A": away_score / total
    })


# =========================
# 試合前データを作る
# =========================

points = {}
recent_points = {}

goals_for = {}
goals_against = {}

model_rows = []


for _, match in df_2025.iterrows():

    home = match["Home"]
    away = match["Away"]

    # -------------------------
    # 試合前の累積勝点
    # -------------------------

    home_points = points.get(home, 0)
    away_points = points.get(away, 0)


    # -------------------------
    # 試合前の直近5試合
    # -------------------------

    home_recent = recent_points.get(home, [])
    away_recent = recent_points.get(away, [])

    home_form = sum(home_recent[-5:])
    away_form = sum(away_recent[-5:])


    # -------------------------
    # 試合前の得失点差
    # -------------------------

    home_goals_for = goals_for.get(home, 0)
    home_goals_against = goals_against.get(home, 0)

    away_goals_for = goals_for.get(away, 0)
    away_goals_against = goals_against.get(away, 0)

    home_goal_diff = (
        home_goals_for -
        home_goals_against
    )

    away_goal_diff = (
        away_goals_for -
        away_goals_against
    )


    # -------------------------
    # 試合前情報を保存
    # -------------------------

    model_rows.append({
        "Home": home,
        "Away": away,

        "HomePointsBefore": home_points,
        "AwayPointsBefore": away_points,

        "PointsDiff":
            home_points - away_points,

        "HomeForm5": home_form,
        "AwayForm5": away_form,

        "HomeGoalDiffBefore":
            home_goal_diff,

        "AwayGoalDiffBefore":
            away_goal_diff
    })


    # -------------------------
    # この試合の勝点
    # -------------------------

    if match["HG"] > match["AG"]:

        home_match_points = 3
        away_match_points = 0

    elif match["HG"] < match["AG"]:

        home_match_points = 0
        away_match_points = 3

    else:

        home_match_points = 1
        away_match_points = 1


    # -------------------------
    # 試合終了後に累積勝点を更新
    # -------------------------

    points[home] = (
        home_points +
        home_match_points
    )

    points[away] = (
        away_points +
        away_match_points
    )


    # -------------------------
    # 直近成績を更新
    # -------------------------

    recent_points.setdefault(
        home, []
    ).append(
        home_match_points
    )

    recent_points.setdefault(
        away, []
    ).append(
        away_match_points
    )


    # -------------------------
    # 得点・失点を更新
    # ※試合前情報を保存した後に更新する
    # -------------------------

    goals_for[home] = (
        home_goals_for +
        match["HG"]
    )

    goals_against[home] = (
        home_goals_against +
        match["AG"]
    )

    goals_for[away] = (
        away_goals_for +
        match["AG"]
    )

    goals_against[away] = (
        away_goals_against +
        match["HG"]
    )


# =========================
# DataFrame作成
# =========================

model_data = pd.DataFrame(
    model_rows
)


# 直近5試合の差
model_data["FormDiff"] = (
    model_data["HomeForm5"] -
    model_data["AwayForm5"]
)


# 得失点差の差
model_data["GoalDiffDiff"] = (
    model_data["HomeGoalDiffBefore"] -
    model_data["AwayGoalDiffBefore"]
)


# =========================
# 実際の試合結果
# =========================

model_data["Result"] = [
    "H" if hg > ag else
    "A" if hg < ag else
    "D"

    for hg, ag in zip(
        df_2025["HG"],
        df_2025["AG"]
    )
]


# =========================
# 実際の結果を0/1に変換
# =========================

actual_h = (
    model_data["Result"] == "H"
).astype(int)

actual_d = (
    model_data["Result"] == "D"
).astype(int)

actual_a = (
    model_data["Result"] == "A"
).astype(int)


# =========================
# 共通の評価関数
# =========================

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

    log_loss = -np.mean(
        np.log(actual_prob)
    )

    brier = np.mean(
        (data[prob_h] - actual_h) ** 2 +
        (data[prob_d] - actual_d) ** 2 +
        (data[prob_a] - actual_a) ** 2
    )

    return accuracy, log_loss, brier


# =========================
# M1
# =========================

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
    model_data["PointsDiff"].apply(
        lambda x:
        "H" if x > 0 else
        "A" if x < 0 else
        "D"
    )
)


m1_accuracy, m1_log_loss, m1_brier = (
    calculate_metrics(
        model_data,
        "M1_Prediction",
        "Prob_H",
        "Prob_D",
        "Prob_A"
    )
)


# =========================
# M2
# =========================

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
    model_data[
        [
            "M2_Prob_H",
            "M2_Prob_D",
            "M2_Prob_A"
        ]
    ]
    .idxmax(axis=1)
    .map({
        "M2_Prob_H": "H",
        "M2_Prob_D": "D",
        "M2_Prob_A": "A"
    })
)


m2_accuracy, m2_log_loss, m2_brier = (
    calculate_metrics(
        model_data,
        "M2_Prediction",
        "M2_Prob_H",
        "M2_Prob_D",
        "M2_Prob_A"
    )
)


# =========================
# M3
# =========================

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
    model_data[
        [
            "M3_Prob_H",
            "M3_Prob_D",
            "M3_Prob_A"
        ]
    ]
    .idxmax(axis=1)
    .map({
        "M3_Prob_H": "H",
        "M3_Prob_D": "D",
        "M3_Prob_A": "A"
    })
)


m3_accuracy, m3_log_loss, m3_brier = (
    calculate_metrics(
        model_data,
        "M3_Prediction",
        "M3_Prob_H",
        "M3_Prob_D",
        "M3_Prob_A"
    )
)


# =========================
# M4
# =========================

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
    model_data[
        [
            "M4_Prob_H",
            "M4_Prob_D",
            "M4_Prob_A"
        ]
    ]
    .idxmax(axis=1)
    .map({
        "M4_Prob_H": "H",
        "M4_Prob_D": "D",
        "M4_Prob_A": "A"
    })
)


m4_accuracy, m4_log_loss, m4_brier = (
    calculate_metrics(
        model_data,
        "M4_Prediction",
        "M4_Prob_H",
        "M4_Prob_D",
        "M4_Prob_A"
    )
)


# =========================
# モデル評価
# =========================

st.header("📊 モデル評価")


st.subheader("M1：勝点差")
st.write(
    "正解率:",
    f"{m1_accuracy:.1%}"
)
st.write(
    "Log Loss:",
    round(m1_log_loss, 4)
)
st.write(
    "Brier Score:",
    round(m1_brier, 4)
)


st.subheader(
    "M2：勝点差＋ホーム補正"
)
st.write(
    "正解率:",
    f"{m2_accuracy:.1%}"
)
st.write(
    "Log Loss:",
    round(m2_log_loss, 4)
)
st.write(
    "Brier Score:",
    round(m2_brier, 4)
)


st.subheader(
    "M3：勝点差＋ホーム補正＋直近5試合"
)
st.write(
    "正解率:",
    f"{m3_accuracy:.1%}"
)
st.write(
    "Log Loss:",
    round(m3_log_loss, 4)
)
st.write(
    "Brier Score:",
    round(m3_brier, 4)
)


st.subheader(
    "M4：M3＋得失点差"
)
st.write(
    "正解率:",
    f"{m4_accuracy:.1%}"
)
st.write(
    "Log Loss:",
    round(m4_log_loss, 4)
)
st.write(
    "Brier Score:",
    round(m4_brier, 4)
)


# =========================
# モデル比較表
# =========================

st.subheader("📋 モデル比較")


comparison = pd.DataFrame({
    "モデル": [
        "M1",
        "M2",
        "M3",
        "M4"
    ],

    "内容": [
        "勝点差",
        "勝点差＋ホーム補正",
        "M2＋直近5試合",
        "M3＋得失点差"
    ],

    "正解率": [
        m1_accuracy,
        m2_accuracy,
        m3_accuracy,
        m4_accuracy
    ],

    "Log Loss": [
        m1_log_loss,
        m2_log_loss,
        m3_log_loss,
        m4_log_loss
    ],

    "Brier Score": [
        m1_brier,
        m2_brier,
        m3_brier,
        m4_brier
    ]
})


comparison["正解率"] = (
    comparison["正解率"] * 100
).round(1)


comparison["Log Loss"] = (
    comparison["Log Loss"]
    .round(4)
)


comparison["Brier Score"] = (
    comparison["Brier Score"]
    .round(4)
)


st.dataframe(
    comparison,
    hide_index=True
)


# =========================
# M1予測結果
# =========================

st.subheader("M1 予測結果")


st.write(
    pd.crosstab(
        model_data["Result"],
        model_data["M1_Prediction"],
        rownames=["実際"],
        colnames=["予測"]
    )
)


# =========================
# 試合予測画面
# =========================

st.header("⚽ 試合予測")


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


if st.button("試合を予測する"):

    if home_team == away_team:

        st.error(
            "ホームとアウェイには別のチームを選んでください。"
        )

    else:

        st.success(
            f"{home_team} vs {away_team}"
        )

        st.info(
            "予測モデルは次のステップで追加します。"
        )
