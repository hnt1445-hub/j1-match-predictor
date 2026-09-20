import streamlit as st
import pandas as pd
from collections import defaultdict, deque


# =========================================================
# 基本設定
# =========================================================

st.set_page_config(
    page_title="J1 Team Strength",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ J1 チーム能力マスター")

st.write(
    "各試合の直前時点で利用できた情報だけを使って、"
    "チーム能力を作成します。"
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

    # 元の並びを保存
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
        return None

    return (
        sum(values)
        /
        len(values)
    )


def recent_points(values):

    if len(values) == 0:
        return 0

    return sum(values)


# =========================================================
# チーム能力作成
#
# 重要：
#
# 1. 試合前の能力を保存
# 2. その後で試合結果を反映
#
# この順番なので未来情報は入りません。
# =========================================================

@st.cache_data
def build_team_strength(matches):

    # -----------------------------------------------------
    # Elo
    # -----------------------------------------------------

    elo = defaultdict(
        lambda: 1500.0
    )


    # -----------------------------------------------------
    # 全会場：直近データ
    # -----------------------------------------------------

    recent_points_5 = defaultdict(
        lambda: deque(
            maxlen=5
        )
    )

    recent_gf_10 = defaultdict(
        lambda: deque(
            maxlen=10
        )
    )

    recent_ga_10 = defaultdict(
        lambda: deque(
            maxlen=10
        )
    )


    # -----------------------------------------------------
    # ホーム限定
    # -----------------------------------------------------

    home_gf_10 = defaultdict(
        lambda: deque(
            maxlen=10
        )
    )

    home_ga_10 = defaultdict(
        lambda: deque(
            maxlen=10
        )
    )


    # -----------------------------------------------------
    # アウェイ限定
    # -----------------------------------------------------

    away_gf_10 = defaultdict(
        lambda: deque(
            maxlen=10
        )
    )

    away_ga_10 = defaultdict(
        lambda: deque(
            maxlen=10
        )
    )


    rows = []

    K_FACTOR = 20


    # =====================================================
    # 全試合を古い順に処理
    # =====================================================

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
        # 試合前データ
        # =================================================

        home_elo = elo[home]
        away_elo = elo[away]


        # -------------------------------------------------
        # 直近5試合勝点
        # -------------------------------------------------

        home_form5 = recent_points(
            recent_points_5[home]
        )

        away_form5 = recent_points(
            recent_points_5[away]
        )


        # -------------------------------------------------
        # 直近10試合
        # -------------------------------------------------

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


        # -------------------------------------------------
        # ホーム限定
        # -------------------------------------------------

        home_home_gf = average(
            home_gf_10[home]
        )

        home_home_ga = average(
            home_ga_10[home]
        )


        # -------------------------------------------------
        # アウェイ限定
        # -------------------------------------------------

        away_away_gf = average(
            away_gf_10[away]
        )

        away_away_ga = average(
            away_ga_10[away]
        )


        # -------------------------------------------------
        # 過去試合数
        #
        # 昇格クラブなどのデータ量確認用
        # -------------------------------------------------

        home_history_count = len(
            recent_gf_10[home]
        )

        away_history_count = len(
            recent_gf_10[away]
        )

        home_home_count = len(
            home_gf_10[home]
        )

        away_away_count = len(
            away_gf_10[away]
        )


        # =================================================
        # 試合前能力を保存
        # =================================================

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

            # Elo
            "Home_Elo":
                home_elo,

            "Away_Elo":
                away_elo,

            # 直近5試合
            "Home_Form5":
                home_form5,

            "Away_Form5":
                away_form5,

            # 全会場 直近10
            "Home_Recent10_GF":
                home_recent_gf,

            "Home_Recent10_GA":
                home_recent_ga,

            "Away_Recent10_GF":
                away_recent_gf,

            "Away_Recent10_GA":
                away_recent_ga,

            # ホーム限定
            "Home_Home10_GF":
                home_home_gf,

            "Home_Home10_GA":
                home_home_ga,

            # アウェイ限定
            "Away_Away10_GF":
                away_away_gf,

            "Away_Away10_GA":
                away_away_ga,

            # データ量
            "Home_History":
                home_history_count,

            "Away_History":
                away_history_count,

            "Home_HomeHistory":
                home_home_count,

            "Away_AwayHistory":
                away_away_count
        })


        # =================================================
        # ここから試合終了後
        # =================================================

        if hg > ag:

            home_points = 3
            away_points = 0

            home_actual = 1.0

        elif hg < ag:

            home_points = 0
            away_points = 3

            home_actual = 0.0

        else:

            home_points = 1
            away_points = 1

            home_actual = 0.5


        # -------------------------------------------------
        # 直近5試合勝点 更新
        # -------------------------------------------------

        recent_points_5[
            home
        ].append(
            home_points
        )

        recent_points_5[
            away
        ].append(
            away_points
        )


        # -------------------------------------------------
        # 全会場 直近10 更新
        # -------------------------------------------------

        recent_gf_10[
            home
        ].append(
            hg
        )

        recent_ga_10[
            home
        ].append(
            ag
        )

        recent_gf_10[
            away
        ].append(
            ag
        )

        recent_ga_10[
            away
        ].append(
            hg
        )


        # -------------------------------------------------
        # Home限定 更新
        # -------------------------------------------------

        home_gf_10[
            home
        ].append(
            hg
        )

        home_ga_10[
            home
        ].append(
            ag
        )


        # -------------------------------------------------
        # Away限定 更新
        # -------------------------------------------------

        away_gf_10[
            away
        ].append(
            ag
        )

        away_ga_10[
            away
        ].append(
            hg
        )


        # -------------------------------------------------
        # Elo更新
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


strength = build_team_strength(
    matches
)


# =========================================================
# 基本チェック
# =========================================================

st.header(
    "📚 データ確認"
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "J1総試合数",
    len(strength)
)

c2.metric(
    "シーズン数",
    strength[
        "Season"
    ].nunique()
)

c3.metric(
    "2025年試合数",
    (
        strength[
            "Season"
        ]
        ==
        "2025"
    ).sum()
)


# =========================================================
# 2025年データ
# =========================================================

season_2025 = (
    strength[
        strength["Season"]
        ==
        "2025"
    ]
    .copy()
    .reset_index(drop=True)
)


st.header(
    "🔍 2025年の試合前能力を確認"
)


# =========================================================
# 日付選択
# =========================================================

available_dates = (
    season_2025[
        "Date"
    ]
    .dt.date
    .unique()
)


selected_date = st.selectbox(
    "日付を選択",
    available_dates
)


date_matches = (
    season_2025[
        season_2025[
            "Date"
        ].dt.date
        ==
        selected_date
    ]
    .copy()
)


# =========================================================
# 試合選択
# =========================================================

match_options = {}

for index, row in date_matches.iterrows():

    label = (
        f"{row['Home']} vs "
        f"{row['Away']}"
    )

    match_options[
        label
    ] = index


selected_match_label = (
    st.selectbox(
        "試合を選択",
        list(
            match_options.keys()
        )
    )
)


selected_index = (
    match_options[
        selected_match_label
    ]
)


selected = (
    season_2025.loc[
        selected_index
    ]
)


# =========================================================
# 対戦表示
# =========================================================

st.subheader(
    f"{selected['Home']}  vs  "
    f"{selected['Away']}"
)

st.caption(
    "以下はすべて、この試合が始まる前のデータです。"
)


# =========================================================
# Home
# =========================================================

home_col, away_col = (
    st.columns(2)
)


with home_col:

    st.markdown(
        f"### 🏠 {selected['Home']}"
    )

    st.metric(
        "Elo",
        f"{selected['Home_Elo']:.1f}"
    )

    st.metric(
        "直近5試合 勝点",
        int(
            selected[
                "Home_Form5"
            ]
        )
    )

    if pd.notna(
        selected[
            "Home_Recent10_GF"
        ]
    ):

        st.metric(
            "直近10試合 平均得点",
            f"{selected['Home_Recent10_GF']:.2f}"
        )

        st.metric(
            "直近10試合 平均失点",
            f"{selected['Home_Recent10_GA']:.2f}"
        )

    else:

        st.write(
            "直近10試合：データなし"
        )


    if pd.notna(
        selected[
            "Home_Home10_GF"
        ]
    ):

        st.metric(
            "Home直近10試合 平均得点",
            f"{selected['Home_Home10_GF']:.2f}"
        )

        st.metric(
            "Home直近10試合 平均失点",
            f"{selected['Home_Home10_GA']:.2f}"
        )

    else:

        st.write(
            "Home履歴：データなし"
        )


    st.caption(
        "J1直近履歴："
        f"{int(selected['Home_History'])}試合"
    )


# =========================================================
# Away
# =========================================================

with away_col:

    st.markdown(
        f"### ✈️ {selected['Away']}"
    )

    st.metric(
        "Elo",
        f"{selected['Away_Elo']:.1f}"
    )

    st.metric(
        "直近5試合 勝点",
        int(
            selected[
                "Away_Form5"
            ]
        )
    )

    if pd.notna(
        selected[
            "Away_Recent10_GF"
        ]
    ):

        st.metric(
            "直近10試合 平均得点",
            f"{selected['Away_Recent10_GF']:.2f}"
        )

        st.metric(
            "直近10試合 平均失点",
            f"{selected['Away_Recent10_GA']:.2f}"
        )

    else:

        st.write(
            "直近10試合：データなし"
        )


    if pd.notna(
        selected[
            "Away_Away10_GF"
        ]
    ):

        st.metric(
            "Away直近10試合 平均得点",
            f"{selected['Away_Away10_GF']:.2f}"
        )

        st.metric(
            "Away直近10試合 平均失点",
            f"{selected['Away_Away10_GA']:.2f}"
        )

    else:

        st.write(
            "Away履歴：データなし"
        )


    st.caption(
        "J1直近履歴："
        f"{int(selected['Away_History'])}試合"
    )


# =========================================================
# 実際の結果
# =========================================================

st.divider()

st.subheader(
    "⚽ 実際の結果"
)

st.write(
    f"{selected['Home']} "
    f"{int(selected['HG'])}"
    " - "
    f"{int(selected['AG'])} "
    f"{selected['Away']}"
)


# =========================================================
# 2025開幕戦チェック
# =========================================================

st.divider()

st.header(
    "🌱 2025年開幕時点チェック"
)

first_date = (
    season_2025[
        "Date"
    ].min()
)

opening_matches = (
    season_2025[
        season_2025[
            "Date"
        ]
        ==
        first_date
    ]
    .copy()
)


opening_view = opening_matches[
    [
        "Home",
        "Away",
        "Home_Elo",
        "Away_Elo",
        "Home_Form5",
        "Away_Form5",
        "Home_Recent10_GF",
        "Home_Recent10_GA",
        "Away_Recent10_GF",
        "Away_Recent10_GA"
    ]
].copy()


for column in [
    "Home_Elo",
    "Away_Elo",
    "Home_Recent10_GF",
    "Home_Recent10_GA",
    "Away_Recent10_GF",
    "Away_Recent10_GA"
]:

    opening_view[column] = (
        opening_view[column]
        .round(2)
    )


st.dataframe(
    opening_view,
    hide_index=True,
    use_container_width=True
)


st.info(
    "2025年開幕時点でも直近5試合・直近10試合・Eloが"
    "入っていれば、過去シーズンの情報を正常に"
    "引き継げています。"
)
