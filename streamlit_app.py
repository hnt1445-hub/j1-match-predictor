import streamlit as st
import pandas as pd
df = pd.read_csv("data/JPN.csv")
st.write(df[df["Season"] == 2025].head())
st.write("2025年の試合数:", len(df[df["Season"] == 2025]))

st.set_page_config(
    page_title="J1 Match Predictor",
    page_icon="⚽",
)

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
