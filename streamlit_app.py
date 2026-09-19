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
# M1・M2・M3の確率計算関数
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


# =========================
# 試合前の勝点・直近5試合を作る
# =========================

points = {}
recent_points = {}
model_rows = []

for _, match in df_2025.iterrows():

    home = match["Home"]
    away = match["Away"]

    # この試合より前の勝点
    home_points = points.get(home, 0)
    away_points = points.get(away, 0)

    # この試合より前の直近成績
    home_recent = recent_points.get(home, [])
    away_recent = recent_points.get(away, [])

    # 直近5試合の獲得勝点
    home_form = sum(home_recent[-5:])
    away_form = sum(away_recent[-5:])

    # 試合前情報を保存
    model_rows.append({
        "Home": home,
        "Away": away,
        "HomePointsBefore": home_points,
        "AwayPointsBefore": away_points,
        "PointsDiff": home_points - away_points,
        "HomeForm5": home_form,
        "AwayForm5": away_form
    })

    # =========================
    # この試合終了後の勝点を更新
    # =========================

    if match["HG"] > match["AG"]:

        home_match_points = 3
        away_match_points = 0

    elif match["HG"] < match["AG"]:

        home_match_points = 0
        away_match_points = 3

    else:

        home_match_points = 1
        away_match_points = 1

    # 累積勝点を更新
    points[home] = home_points + home_match_points
    points[away] = away_points + away_match_points

    # 直近成績に今回の試合を追加
    recent_points.setdefault(home, []).append(
        home_match_points
    )

    recent_points.setdefault(away, []).append(
        away_match_points
    )


# DataFrameにする
m1 = pd.DataFrame(model_rows)


# =========================
# 直近5試合の差
# =========================

m1["FormDiff"] = (
    m1["HomeForm5"] -
    m1["AwayForm5"]
)


# =========================
# 実際の試合結果
# =========================

m1["Result"] = [
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
    m1["Result"] == "H"
).astype(int)

actual_d = (
    m1["Result"] == "D"
).astype(int)

actual_a = (
    m1["Result"] == "A"
).astype(int)


# =========================
# M1
# =========================

m1_probs = m1["PointsDiff"].apply(
    m1_probabilities
)

m1 = pd.concat(
    [m1, m1_probs],
    axis=1
)


# M1の予測
m1["Prediction"] = m1["PointsDiff"].apply(
    lambda x:
    "H" if x > 0 else
    "A" if x < 0 else
    "D"
)


# M1 Accuracy
m1_accuracy = (
    m1["Prediction"] ==
    m1["Result"]
).mean()


# M1 Log Loss
m1_actual_prob = np.where(
    m1["Result"] == "H",
    m1["Prob_H"],
    np.where(
        m1["Result"] == "D",
        m1["Prob_D"],
        m1["Prob_A"]
    )
)

m1_log_loss = -np.mean(
    np.log(m1_actual_prob)
)


# M1 Brier Score
m1_brier = np.mean(
    (m1["Prob_H"] - actual_h) ** 2 +
    (m1["Prob_D"] - actual_d) ** 2 +
    (m1["Prob_A"] - actual_a) ** 2
)


# =========================
# M2
# =========================

m2_probs = m1["PointsDiff"].apply(
    m2_probabilities
)

m1 = pd.concat(
    [m1, m2_probs],
    axis=1
)


# M2の予測
m1["M2_Prediction"] = m1[
    [
        "M2_Prob_H",
        "M2_Prob_D",
        "M2_Prob_A"
    ]
].idxmax(axis=1).map({
    "M2_Prob_H": "H",
    "M2_Prob_D": "D",
    "M2_Prob_A": "A"
})


# M2 Accuracy
m2_accuracy = (
    m1["M2_Prediction"] ==
    m1["Result"]
).mean()


# M2 Log Loss
m2_actual_prob = np.where(
    m1["Result"] == "H",
    m1["M2_Prob_H"],
    np.where(
        m1["Result"] == "D",
        m1["M2_Prob_D"],
        m1["M2_Prob_A"]
    )
)

m2_log_loss = -np.mean(
    np.log(m2_actual_prob)
)


# M2 Brier Score
m2_brier = np.mean(
    (m1["M2_Prob_H"] - actual_h) ** 2 +
    (m1["M2_Prob_D"] - actual_d) ** 2 +
    (m1["M2_Prob_A"] - actual_a) ** 2
)


# =========================
# M3
# =========================

m3_probs = m1.apply(
    lambda row: m3_probabilities(
        row["PointsDiff"],
        row["FormDiff"]
    ),
    axis=1
)

m1 = pd.concat(
    [m1, m3_probs],
    axis=1
)


# M3の予測
m1["M3_Prediction"] = m1[
    [
        "M3_Prob_H",
        "M3_Prob_D",
        "M3_Prob_A"
    ]
].idxmax(axis=1).map({
    "M3_Prob_H": "H",
    "M3_Prob_D": "D",
    "M3_Prob_A": "A"
})


# M3 Accuracy
m3_accuracy = (
    m1["M3_Prediction"] ==
    m1["Result"]
).mean()


# M3 Log Loss
m3_actual_prob = np.where(
    m1["Result"] == "H",
    m1["M3_Prob_H"],
    np.where(
        m1["Result"] == "D",
        m1["M3_Prob_D"],
        m1["M3_Prob_A"]
    )
)

m3_log_loss = -np.mean(
    np.log(m3_actual_prob)
)


# M3 Brier Score
m3_brier = np.mean(
    (m1["M3_Prob_H"] - actual_h) ** 2 +
    (m1["M3_Prob_D"] - actual_d) ** 2 +
    (m1["M3_Prob_A"] - actual_a) ** 2
)


# =========================
# モデル評価
# =========================

st.header("📊 モデル評価")


# M1
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


# M2
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


# M3
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


# =========================
# 比較表
# =========================

st.subheader("📋 モデル比較")

comparison = pd.DataFrame({
    "モデル": [
        "M1",
        "M2",
        "M3"
    ],
    "正解率": [
        m1_accuracy,
        m2_accuracy,
        m3_accuracy
    ],
    "Log Loss": [
        m1_log_loss,
        m2_log_loss,
        m3_log_loss
    ],
    "Brier Score": [
        m1_brier,
        m2_brier,
        m3_brier
    ]
})

comparison["正解率"] = (
    comparison["正解率"] * 100
).round(1)

comparison["Log Loss"] = (
    comparison["Log Loss"].round(4)
)

comparison["Brier Score"] = (
    comparison["Brier Score"].round(4)
)

st.dataframe(
    comparison,
    hide_index=True
)


# =========================
# M1の予測結果表
# =========================

st.subheader("M1 予測結果")

st.write(
    pd.crosstab(
        m1["Result"],
        m1["Prediction"],
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
