import streamlit as st
import pandas as pd
import numpy as np
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
# M1用：各試合の直前時点の勝点を計算
points = {}
m1_rows = []

for _, match in df_2025.iterrows():
    home = match["Home"]
    away = match["Away"]

    home_points = points.get(home, 0)
    away_points = points.get(away, 0)

    m1_rows.append({
        "Home": home,
        "Away": away,
        "HomePointsBefore": home_points,
        "AwayPointsBefore": away_points,
        "PointsDiff": home_points - away_points
    })

    if match["HG"] > match["AG"]:
        points[home] = home_points + 3
        points[away] = away_points
    elif match["HG"] < match["AG"]:
        points[home] = home_points
        points[away] = away_points + 3
    else:
        points[home] = home_points + 1
        points[away] = away_points + 1
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
m1 = pd.DataFrame(m1_rows)
m1["Result"] = [
    "H" if hg > ag else "A" if hg < ag else "D"
    for hg, ag in zip(df_2025["HG"], df_2025["AG"])
]

st.write("実際の結果:")
st.write(m1["Result"].value_counts())
st.write("M1の試合数:", len(m1))
st.dataframe(m1.head(10))
m1["Prediction"] = m1["PointsDiff"].apply(
    lambda x: "H" if x > 0 else "A" if x < 0 else "D"
)

m1_accuracy = (m1["Prediction"] == m1["Result"]).mean()

st.write("M1 正解率:", f"{m1_accuracy:.1%}")
st.write("M1 予測結果:")
st.write(pd.crosstab(
    m1["Result"],
    m1["Prediction"],
    rownames=["実際"],
    colnames=["予測"]
))

st.title("⚽ J1 Match Predictor")
st.write("J1リーグの試合結果を予測するアプリです。")

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

home_team = st.selectbox("🏠 ホームチーム", teams)

away_team = st.selectbox(
    "✈️ アウェイチーム",
    teams,
    index=1,
)

if st.button("試合を予測する"):
    if home_team == away_team:
        st.error("ホームとアウェイには別のチームを選んでください。")
    else:
        st.success(f"{home_team} vs {away_team}")
        st.info("予測モデルは次のステップで追加します。")
