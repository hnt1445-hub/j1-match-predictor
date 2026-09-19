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

df_2025 = df_2025.sort_values("Date")


# =========================
# 各試合の直前時点の勝点を計算
# =========================

points = {}
recent_points = {}
m1_rows = []

for _, match in df_2025.iterrows():

    home = match["Home"]
    away = match["Away"]

    home_points = points.get(home, 0)
    away_points = points.get(away, 0)
    
    home_recent = recent_points.get(home, [])
    away_recent = recent_points.get(away, [])

    home_form = sum(home_recent[-5:])
    away_form = sum(away_recent[-5:])

    m1_rows.append({
        "Home": home,
        "Away": away,
        "HomePointsBefore": home_points,
        "AwayPointsBefore": away_points,
        "PointsDiff": home_points - away_points,
        "HomeForm5": home_form,
        "AwayForm5": away_form
    })

    # この試合終了後の勝点を更新
    if match["HG"] > match["AG"]:
        points[home] = home_points + 3
        points[away] = away_points

    elif match["HG"] < match["AG"]:
        points[home] = home_points
        points[away] = away_points + 3

    else:
        points[home] = home_points + 1
        points[away] = away_points + 1


m1 = pd.DataFrame(m1_rows)


# =========================
# 実際の試合結果
# =========================

m1["Result"] = [
    "H" if hg > ag else "A" if hg < ag else "D"
    for hg, ag in zip(df_2025["HG"], df_2025["AG"])
]


# =========================
# M1
# 勝点差のみ
# =========================

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


m1_probs = m1["PointsDiff"].apply(m1_probabilities)

m1 = pd.concat(
    [m1, m1_probs],
    axis=1
)


# M1の予測
m1["Prediction"] = m1["PointsDiff"].apply(
    lambda x: "H" if x > 0 else "A" if x < 0 else "D"
)


# M1 Accuracy
m1_accuracy = (
    m1["Prediction"] == m1["Result"]
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


# 実際の結果を0/1に変換
actual_h = (
    m1["Result"] == "H"
).astype(int)

actual_d = (
    m1["Result"] == "D"
).astype(int)

actual_a = (
    m1["Result"] == "A"
).astype(int)


# M1 Brier Score
m1_brier = np.mean(
    (m1["Prob_H"] - actual_h) ** 2 +
    (m1["Prob_D"] - actual_d) ** 2 +
    (m1["Prob_A"] - actual_a) ** 2
)


# =========================
# M2
# 勝点差＋ホーム補正
# =========================

def m2_probabilities(points_diff):

    home_advantage = 2.0

    home_score = np.exp(
        (points_diff + home_advantage) / 10
    )

    away_score = np.exp(
        -(points_diff + home_advantage) / 10
    )

    draw_score = 1.0

    total = (
        home_score +
        draw_score +
        away_score
    )

    return pd.Series({
        "M2_Prob_H": home_score / total,
        "M2_Prob_D": draw_score / total,
        "M2_Prob_A": away_score / total
    })


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
# モデル評価を表示
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


st.subheader("M2：勝点差＋ホーム補正")

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
