import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Jリーグ公式データ取得テスト",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Jリーグ公式データ取得テスト")

URL = (
    "https://data.j-league.or.jp/SFMS01/search"
    "?competition_frame_ids=1"
    "&competition_ids=725"
    "&competition_years=2026"
)

st.write("Jリーグ公式Data Siteから2026/27 J1を読み込みます。")

try:
    tables = pd.read_html(URL)

    st.success(
        f"取得成功：{len(tables)}個の表を取得しました。"
    )

    st.subheader("取得した表")

    for i, table in enumerate(tables):
        st.write(f"表 {i}")
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True
        )

except Exception as e:
    st.error("取得に失敗しました。")
    st.exception(e)
