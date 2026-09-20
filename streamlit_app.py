import streamlit as st
import pandas as pd
import numpy as np
import re

from collections import defaultdict, deque
from math import exp, factorial

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor


# =========================================================
# 基本設定
# =========================================================

st.set_page_config(
    page_title="J1 Matchday Predictor",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ J1 Matchday Predictor")

st.write(
    "Jリーグ公式データから最新結果と未消化試合を取得し、"
    "G5.1で予測します。"
)

st.caption(
    "G5.1 / Attack・Defense Rating 学習率 = 0.06"
)

LEARNING_RATE = 0.06
K_FACTOR = 20

JLEAGUE_URL = (
    "https://data.j-league.or.jp/SFMS01/search"
    "?competition_frame_ids=1"
    "&competition_ids=725"
    "&competition_years=2026"
)


# =========================================================
# Jリーグ公式 → JPN.csv チーム名
# =========================================================

TEAM_MAP = {
    "鹿島": "Kashima Antlers",
    "水戸": "Mito",
    "浦和": "Urawa Reds",
    "千葉": "Chiba",
    "柏": "Kashiwa Reysol",
    "FC東京": "FC Tokyo",
    "ＦＣ東京": "FC Tokyo",
    "東京Ｖ": "Verdy",
    "東京V": "Verdy",
    "町田": "Machida",
    "川崎Ｆ": "Kawasaki Frontale",
    "川崎F": "Kawasaki Frontale",
    "横浜FM": "Yokohama F. Marinos",
    "横浜ＦＭ": "Yokohama F. Marinos",
    "清水": "Shimizu S-Pulse",
    "名古屋": "Nagoya Grampus",
    "京都": "Kyoto",
    "Ｇ大阪": "Gamba Osaka",
    "G大阪": "Gamba Osaka",
    "Ｃ大阪": "Cerezo Osaka",
    "C大阪": "Cerezo Osaka",
    "神戸": "Vissel Kobe",
    "岡山": "Okayama",
    "広島": "Sanfrecce Hiroshima",
    "福岡": "Avispa Fukuoka",
    "長崎": "V-Varen Nagasaki",
}


# =========================================================
# 画面表示用チーム名（モデル内部名とは分離）
# =========================================================

DEFAULT_DISPLAY_NAMES = {
    "Kashima Antlers": "鹿島",
    "Mito": "水戸",
    "Urawa Reds": "浦和",
    "Chiba": "千葉",
    "Kashiwa Reysol": "柏",
    "FC Tokyo": "FC東京",
    "Verdy": "東京V",
    "Machida": "町田",
    "Kawasaki Frontale": "川崎F",
    "Yokohama F. Marinos": "横浜FM",
    "Shimizu S-Pulse": "清水",
    "Nagoya Grampus": "名古屋",
    "Kyoto": "京都",
    "Gamba Osaka": "G大阪",
    "Cerezo Osaka": "C大阪",
    "Vissel Kobe": "神戸",
    "Okayama": "岡山",
    "Sanfrecce Hiroshima": "広島",
    "Avispa Fukuoka": "福岡",
    "V-Varen Nagasaki": "長崎",
}


def load_display_name_file():

    try:
        df = pd.read_csv(
            "data/team_display_names.csv"
        )
    except Exception:
        return {}

    if not {
        "internal_name",
        "display_name",
    }.issubset(df.columns):
        return {}

    result = {}

    for _, row in df.iterrows():
        internal = str(
            row["internal_name"]
        ).strip()
        display = str(
            row["display_name"]
        ).strip()

        if (
            internal
            and display
            and display.lower() != "nan"
        ):
            result[internal] = display

    return result


# =========================================================
# モデル特徴量
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
    "Away_Away10_GA",
    "Home_AttackRating",
    "Home_DefenseRating",
    "Away_AttackRating",
    "Away_DefenseRating",
    "AttackAdvantage",
    "DefenseAdvantage",
]

CATEGORICAL_FEATURES = [
    "Home",
    "Away",
]

ALL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# =========================================================
# 過去データ
# =========================================================

@st.cache_data
def load_historical():

    df = pd.read_csv(
        "data/JPN.csv"
    )

    df = df[
        df["League"] == "J1 League"
    ].copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce",
    )

    df["HG"] = pd.to_numeric(
        df["HG"],
        errors="coerce",
    )

    df["AG"] = pd.to_numeric(
        df["AG"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Date",
            "Home",
            "Away",
            "HG",
            "AG",
        ]
    ).copy()

    return pd.DataFrame({
        "Date": df["Date"],
        "Home": (
            df["Home"]
            .astype(str)
            .str.strip()
        ),
        "Away": (
            df["Away"]
            .astype(str)
            .str.strip()
        ),
        "HG": df["HG"],
        "AG": df["AG"],
        "Source": "JPN.csv",
    })


# =========================================================
# Jリーグ公式データ
# =========================================================

def parse_official_date(value):

    text = str(value).strip()

    # 26/09/20(日) → 26/09/20
    text = text.split("(")[0]

    return pd.to_datetime(
        text,
        format="%y/%m/%d",
        errors="coerce",
    )


def clean_score(value):

    text = str(value).strip()

    return (
        text
        .replace("－", "-")
        .replace("−", "-")
        .replace("―", "-")
        .replace("ー", "-")
    )


def extract_round(value):

    text = str(value)

    match = re.search(
        r"第\s*(\d+)\s*節",
        text,
    )

    if match:

        return int(
            match.group(1)
        )

    return np.nan


@st.cache_data(ttl=1800)
def load_jleague_official():

    tables = pd.read_html(
        JLEAGUE_URL
    )

    if len(tables) == 0:

        raise ValueError(
            "公式サイトから表を取得できませんでした。"
        )

    official = tables[0].copy()

    required = [
        "節",
        "試合日",
        "ホーム",
        "スコア",
        "アウェイ",
    ]

    missing = [
        column
        for column in required
        if column not in official.columns
    ]

    if missing:

        raise ValueError(
            "必要な列がありません："
            + "、".join(missing)
        )

    games = official[
        required
    ].copy()

    games["Round"] = (
        games["節"]
        .apply(extract_round)
    )

    games["Date"] = (
        games["試合日"]
        .apply(parse_official_date)
    )

    games["OfficialHome"] = (
        games["ホーム"]
        .astype(str)
        .str.strip()
    )

    games["OfficialAway"] = (
        games["アウェイ"]
        .astype(str)
        .str.strip()
    )

    games["Score"] = (
        games["スコア"]
        .apply(clean_score)
    )

    games["Home"] = (
        games["OfficialHome"]
        .map(TEAM_MAP)
    )

    games["Away"] = (
        games["OfficialAway"]
        .map(TEAM_MAP)
    )

    score_parts = (
        games["Score"]
        .str.extract(
            r"^\s*(\d+)\s*-\s*(\d+)\s*$"
        )
    )

    games["HG"] = pd.to_numeric(
        score_parts[0],
        errors="coerce",
    )

    games["AG"] = pd.to_numeric(
        score_parts[1],
        errors="coerce",
    )

    games = games.dropna(
        subset=["Date"]
    ).copy()

    return games


# =========================================================
# 公式データ取得
# =========================================================

official_error = None

try:

    official_games = (
        load_jleague_official()
    )

except Exception as e:

    official_error = str(e)

    official_games = pd.DataFrame()


historical = load_historical()

known_historical_teams = (
    set(historical["Home"])
    |
    set(historical["Away"])
)


# =========================================================
# チーム名チェック
# =========================================================

unmapped_names = []
mapped_not_in_history = []

if len(official_games) > 0:

    official_names = sorted(
        set(
            official_games[
                "OfficialHome"
            ]
        )
        |
        set(
            official_games[
                "OfficialAway"
            ]
        )
    )

    unmapped_names = [
        team
        for team in official_names
        if team not in TEAM_MAP
    ]

    mapped_names = {
        TEAM_MAP[team]
        for team in official_names
        if team in TEAM_MAP
    }

    mapped_not_in_history = sorted(
        mapped_names
        - known_historical_teams
    )


# =========================================================
# 表示名設定
# =========================================================

all_internal_teams = sorted(
    known_historical_teams
    | set(TEAM_MAP.values())
)

file_display_names = (
    load_display_name_file()
)

base_display_names = {
    team: file_display_names.get(
        team,
        DEFAULT_DISPLAY_NAMES.get(
            team,
            team,
        ),
    )
    for team in all_internal_teams
}

if "team_display_names" not in st.session_state:
    st.session_state[
        "team_display_names"
    ] = base_display_names.copy()
else:
    for team, name in base_display_names.items():
        st.session_state[
            "team_display_names"
        ].setdefault(team, name)


def display_team(team):

    if pd.isna(team):
        return "-"

    team = str(team)

    return st.session_state[
        "team_display_names"
    ].get(
        team,
        DEFAULT_DISPLAY_NAMES.get(
            team,
            team,
        ),
    )


# =========================================================
# 公式の終了済み試合
# =========================================================

if len(official_games) > 0:

    official_completed = (
        official_games[
            official_games["HG"].notna()
            &
            official_games["AG"].notna()
            &
            official_games["Home"].notna()
            &
            official_games["Away"].notna()
        ]
        .copy()
    )

else:

    official_completed = (
        pd.DataFrame()
    )


if len(official_completed) > 0:

    current_results = pd.DataFrame({
        "Date":
            official_completed["Date"],

        "Home":
            official_completed["Home"],

        "Away":
            official_completed["Away"],

        "HG":
            official_completed["HG"],

        "AG":
            official_completed["AG"],

        "Source":
            "J.League Official",
    })

else:

    current_results = pd.DataFrame(
        columns=[
            "Date",
            "Home",
            "Away",
            "HG",
            "AG",
            "Source",
        ]
    )


# =========================================================
# 過去 + 公式最新結果
# =========================================================

matches = pd.concat(
    [
        historical,
        current_results,
    ],
    ignore_index=True,
)

matches["Date"] = pd.to_datetime(
    matches["Date"],
    errors="coerce",
)

matches["HG"] = pd.to_numeric(
    matches["HG"],
    errors="coerce",
)

matches["AG"] = pd.to_numeric(
    matches["AG"],
    errors="coerce",
)

matches = matches.dropna(
    subset=[
        "Date",
        "Home",
        "Away",
        "HG",
        "AG",
    ]
).copy()

matches = (
    matches
    .drop_duplicates(
        subset=[
            "Date",
            "Home",
            "Away",
        ],
        keep="last",
    )
    .sort_values(
        [
            "Date",
            "Home",
            "Away",
        ]
    )
    .reset_index(drop=True)
)


# =========================================================
# 次の未消化試合
# =========================================================

next_round = None
next_games = pd.DataFrame()

if len(official_games) > 0:

    today = (
        pd.Timestamp.now()
        .normalize()
    )

    future_games = official_games[
        official_games[
            "Score"
        ].str.lower().eq("vs")
        &
        (
            official_games["Date"]
            >= today
        )
        &
        official_games["Round"].notna()
        &
        official_games["Home"].notna()
        &
        official_games["Away"].notna()
    ].copy()

    if len(future_games) > 0:

        first_future_date = (
            future_games["Date"].min()
        )

        candidate_rounds = (
            future_games[
                future_games["Date"]
                == first_future_date
            ]["Round"]
            .dropna()
            .astype(int)
            .tolist()
        )

        if len(candidate_rounds) > 0:

            next_round = min(
                candidate_rounds
            )

            next_games = (
                future_games[
                    future_games["Round"]
                    == next_round
                ]
                .sort_values(
                    [
                        "Date",
                        "Home",
                    ]
                )
                .reset_index(drop=True)
            )


# =========================================================
# CSVフォールバック
# =========================================================

if len(next_games) == 0:

    try:

        fallback = pd.read_csv(
            "data/next_matches.csv"
        )

        fallback["Date"] = (
            pd.to_datetime(
                fallback["date"],
                errors="coerce",
            )
        )

        fallback["Home"] = (
            fallback["home_team"]
            .astype(str)
            .str.strip()
        )

        fallback["Away"] = (
            fallback["away_team"]
            .astype(str)
            .str.strip()
        )

        fallback = fallback.dropna(
            subset=[
                "Date",
                "Home",
                "Away",
            ]
        )

        next_games = fallback[
            [
                "Date",
                "Home",
                "Away",
            ]
        ].copy()

    except Exception:

        next_games = pd.DataFrame(
            columns=[
                "Date",
                "Home",
                "Away",
            ]
        )


# =========================================================
# 平均
# =========================================================

def average(values):

    if len(values) == 0:

        return np.nan

    return float(
        np.mean(values)
    )


# =========================================================
# 時系列特徴量
# =========================================================

@st.cache_data
def build_history(matches):

    elo = defaultdict(
        lambda: 1500.0
    )

    attack = defaultdict(
        lambda: 1.0
    )

    defense = defaultdict(
        lambda: 1.0
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

    for _, match in matches.iterrows():

        home = match["Home"]
        away = match["Away"]

        hg = float(
            match["HG"]
        )

        ag = float(
            match["AG"]
        )

        home_elo = elo[home]
        away_elo = elo[away]

        home_attack = attack[home]
        away_attack = attack[away]

        home_defense = defense[home]
        away_defense = defense[away]

        home_form = sum(
            recent_points_5[home]
        )

        away_form = sum(
            recent_points_5[away]
        )

        rows.append({
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
                - away_elo,

            "Home_Form5":
                home_form,

            "Away_Form5":
                away_form,

            "FormDiff":
                home_form
                - away_form,

            "Home_Recent10_GF":
                average(
                    recent_gf_10[home]
                ),

            "Home_Recent10_GA":
                average(
                    recent_ga_10[home]
                ),

            "Away_Recent10_GF":
                average(
                    recent_gf_10[away]
                ),

            "Away_Recent10_GA":
                average(
                    recent_ga_10[away]
                ),

            "Home_Home10_GF":
                average(
                    home_gf_10[home]
                ),

            "Home_Home10_GA":
                average(
                    home_ga_10[home]
                ),

            "Away_Away10_GF":
                average(
                    away_gf_10[away]
                ),

            "Away_Away10_GA":
                average(
                    away_ga_10[away]
                ),

            "Home_AttackRating":
                home_attack,

            "Home_DefenseRating":
                home_defense,

            "Away_AttackRating":
                away_attack,

            "Away_DefenseRating":
                away_defense,

            "AttackAdvantage":
                home_attack
                - away_attack,

            "DefenseAdvantage":
                away_defense
                - home_defense,
        })

        # 勝点
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

        # 得失点
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

        home_gf_10[
            home
        ].append(hg)

        home_ga_10[
            home
        ].append(ag)

        away_gf_10[
            away
        ].append(ag)

        away_ga_10[
            away
        ].append(hg)

        # Attack / Defense
        expected_home_goals = (
            home_attack
            * away_defense
        )

        expected_away_goals = (
            away_attack
            * home_defense
        )

        home_ratio = (
            (hg + 0.5)
            /
            (
                expected_home_goals
                + 0.5
            )
        )

        away_ratio = (
            (ag + 0.5)
            /
            (
                expected_away_goals
                + 0.5
            )
        )

        attack[home] = float(
            np.clip(
                home_attack
                * (
                    home_ratio
                    ** LEARNING_RATE
                ),
                0.50,
                1.80,
            )
        )

        attack[away] = float(
            np.clip(
                away_attack
                * (
                    away_ratio
                    ** LEARNING_RATE
                ),
                0.50,
                1.80,
            )
        )

        defense[home] = float(
            np.clip(
                home_defense
                * (
                    away_ratio
                    ** LEARNING_RATE
                ),
                0.50,
                1.80,
            )
        )

        defense[away] = float(
            np.clip(
                away_defense
                * (
                    home_ratio
                    ** LEARNING_RATE
                ),
                0.50,
                1.80,
            )
        )

        # Elo
        expected_home = (
            1.0
            /
            (
                1.0
                +
                10.0
                ** (
                    (
                        away_elo
                        - home_elo
                    )
                    / 400.0
                )
            )
        )

        expected_away = (
            1.0
            - expected_home
        )

        away_actual = (
            1.0
            - home_actual
        )

        elo[home] = (
            home_elo
            +
            K_FACTOR
            * (
                home_actual
                - expected_home
            )
        )

        elo[away] = (
            away_elo
            +
            K_FACTOR
            * (
                away_actual
                - expected_away
            )
        )

    history = pd.DataFrame(
        rows
    )

    state = {
        "elo":
            dict(elo),

        "attack":
            dict(attack),

        "defense":
            dict(defense),

        "recent_points_5":
            {
                k: list(v)
                for k, v
                in recent_points_5.items()
            },

        "recent_gf_10":
            {
                k: list(v)
                for k, v
                in recent_gf_10.items()
            },

        "recent_ga_10":
            {
                k: list(v)
                for k, v
                in recent_ga_10.items()
            },

        "home_gf_10":
            {
                k: list(v)
                for k, v
                in home_gf_10.items()
            },

        "home_ga_10":
            {
                k: list(v)
                for k, v
                in home_ga_10.items()
            },

        "away_gf_10":
            {
                k: list(v)
                for k, v
                in away_gf_10.items()
            },

        "away_ga_10":
            {
                k: list(v)
                for k, v
                in away_ga_10.items()
            },
    }

    return history, state


history, current_state = (
    build_history(matches)
)


# =========================================================
# モデル
# =========================================================

def make_model():

    numeric = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            )
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric,
                NUMERIC_FEATURES,
            ),
            (
                "team",
                categorical,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessing,
            ),
            (
                "poisson",
                PoissonRegressor(
                    alpha=0.1,
                    max_iter=1000,
                ),
            ),
        ]
    )


@st.cache_resource
def train_models(history):

    home_model = make_model()
    away_model = make_model()

    home_model.fit(
        history[
            ALL_FEATURES
        ],
        history["HG"],
    )

    away_model.fit(
        history[
            ALL_FEATURES
        ],
        history["AG"],
    )

    return (
        home_model,
        away_model,
    )


home_model, away_model = (
    train_models(history)
)


# =========================================================
# 現在のチーム状態
# =========================================================

def state_average(
    dictionary,
    team,
):

    values = dictionary.get(
        team,
        [],
    )

    if len(values) == 0:

        return np.nan

    return float(
        np.mean(values)
    )


def future_features(
    home,
    away,
):

    home_elo = (
        current_state[
            "elo"
        ].get(
            home,
            1500.0,
        )
    )

    away_elo = (
        current_state[
            "elo"
        ].get(
            away,
            1500.0,
        )
    )

    home_attack = (
        current_state[
            "attack"
        ].get(
            home,
            1.0,
        )
    )

    away_attack = (
        current_state[
            "attack"
        ].get(
            away,
            1.0,
        )
    )

    home_defense = (
        current_state[
            "defense"
        ].get(
            home,
            1.0,
        )
    )

    away_defense = (
        current_state[
            "defense"
        ].get(
            away,
            1.0,
        )
    )

    home_form = sum(
        current_state[
            "recent_points_5"
        ].get(
            home,
            [],
        )
    )

    away_form = sum(
        current_state[
            "recent_points_5"
        ].get(
            away,
            [],
        )
    )

    row = {
        "Home":
            home,

        "Away":
            away,

        "Home_Elo":
            home_elo,

        "Away_Elo":
            away_elo,

        "EloDiff":
            home_elo
            - away_elo,

        "Home_Form5":
            home_form,

        "Away_Form5":
            away_form,

        "FormDiff":
            home_form
            - away_form,

        "Home_Recent10_GF":
            state_average(
                current_state[
                    "recent_gf_10"
                ],
                home,
            ),

        "Home_Recent10_GA":
            state_average(
                current_state[
                    "recent_ga_10"
                ],
                home,
            ),

        "Away_Recent10_GF":
            state_average(
                current_state[
                    "recent_gf_10"
                ],
                away,
            ),

        "Away_Recent10_GA":
            state_average(
                current_state[
                    "recent_ga_10"
                ],
                away,
            ),

        "Home_Home10_GF":
            state_average(
                current_state[
                    "home_gf_10"
                ],
                home,
            ),

        "Home_Home10_GA":
            state_average(
                current_state[
                    "home_ga_10"
                ],
                home,
            ),

        "Away_Away10_GF":
            state_average(
                current_state[
                    "away_gf_10"
                ],
                away,
            ),

        "Away_Away10_GA":
            state_average(
                current_state[
                    "away_ga_10"
                ],
                away,
            ),

        "Home_AttackRating":
            home_attack,

        "Home_DefenseRating":
            home_defense,

        "Away_AttackRating":
            away_attack,

        "Away_DefenseRating":
            away_defense,

        "AttackAdvantage":
            home_attack
            - away_attack,

        "DefenseAdvantage":
            away_defense
            - home_defense,
    }

    return pd.DataFrame(
        [row]
    )


# =========================================================
# Poisson → H/D/A
# =========================================================

def poisson_probability(
    goals,
    expected,
):

    return (
        exp(-expected)
        * expected ** goals
        / factorial(goals)
    )


def result_probabilities(
    home_lambda,
    away_lambda,
):

    h = 0.0
    d = 0.0
    a = 0.0

    for hg in range(11):

        ph = poisson_probability(
            hg,
            home_lambda,
        )

        for ag in range(11):

            probability = (
                ph
                * poisson_probability(
                    ag,
                    away_lambda,
                )
            )

            if hg > ag:

                h += probability

            elif hg == ag:

                d += probability

            else:

                a += probability

    total = (
        h + d + a
    )

    return np.array(
        [
            h / total,
            d / total,
            a / total,
        ]
    )


def predict_match(
    home,
    away,
):

    x = future_features(
        home,
        away,
    )

    home_lambda = float(
        home_model.predict(
            x[
                ALL_FEATURES
            ]
        )[0]
    )

    away_lambda = float(
        away_model.predict(
            x[
                ALL_FEATURES
            ]
        )[0]
    )

    home_lambda = float(
        np.clip(
            home_lambda,
            0.05,
            5.0,
        )
    )

    away_lambda = float(
        np.clip(
            away_lambda,
            0.05,
            5.0,
        )
    )

    probabilities = (
        result_probabilities(
            home_lambda,
            away_lambda,
        )
    )

    return (
        home_lambda,
        away_lambda,
        probabilities,
    )


# =========================================================
# 現在のJ1順位表を公式終了済み試合から計算
# =========================================================

def build_current_standings(completed):

    teams = sorted(
        set(completed["Home"].dropna())
        | set(completed["Away"].dropna())
    )

    stats = {
        team: {
            "Team": team,
            "P": 0,
            "W": 0,
            "D": 0,
            "L": 0,
            "GF": 0,
            "GA": 0,
            "Pts": 0,
        }
        for team in teams
    }

    for _, game in completed.iterrows():
        home = game["Home"]
        away = game["Away"]
        hg = int(game["HG"])
        ag = int(game["AG"])

        if home not in stats or away not in stats:
            continue

        stats[home]["P"] += 1
        stats[away]["P"] += 1
        stats[home]["GF"] += hg
        stats[home]["GA"] += ag
        stats[away]["GF"] += ag
        stats[away]["GA"] += hg

        if hg > ag:
            stats[home]["W"] += 1
            stats[away]["L"] += 1
            stats[home]["Pts"] += 3
        elif hg < ag:
            stats[away]["W"] += 1
            stats[home]["L"] += 1
            stats[away]["Pts"] += 3
        else:
            stats[home]["D"] += 1
            stats[away]["D"] += 1
            stats[home]["Pts"] += 1
            stats[away]["Pts"] += 1

    table = pd.DataFrame(
        list(stats.values())
    )

    if len(table) == 0:
        return table

    table["GD"] = (
        table["GF"] - table["GA"]
    )

    table = (
        table
        .sort_values(
            ["Pts", "GD", "GF", "Team"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )

    table.insert(
        0,
        "Rank",
        range(1, len(table) + 1),
    )

    return table



current_standings = (
    build_current_standings(
        official_completed
    )
)

rank_map = (
    dict(
        zip(
            current_standings["Team"],
            current_standings["Rank"],
        )
    )
    if len(current_standings) > 0
    else {}
)

points_map = (
    dict(
        zip(
            current_standings["Team"],
            current_standings["Pts"],
        )
    )
    if len(current_standings) > 0
    else {}
)


# =========================================================
# 表示名設定UI
# =========================================================

with st.expander(
    "⚙️ チーム表示名を変更する"
):

    st.write(
        "画面に表示するクラブ名だけを変更できます。"
        "モデル内部のチーム名・予測・保存済み記録には影響しません。"
    )

    current_season_teams = sorted(
        set(official_games.get(
            "Home", pd.Series(dtype=object)
        ).dropna())
        | set(official_games.get(
            "Away", pd.Series(dtype=object)
        ).dropna())
    )

    settings_teams = (
        current_season_teams
        if current_season_teams
        else all_internal_teams
    )

    editor_source = pd.DataFrame({
        "内部名": settings_teams,
        "表示名": [
            display_team(team)
            for team in settings_teams
        ],
    })

    edited_names = st.data_editor(
        editor_source,
        hide_index=True,
        use_container_width=True,
        disabled=["内部名"],
        key="display_name_editor",
    )

    if st.button(
        "表示名をこの画面に反映",
        use_container_width=True,
    ):
        for _, row in edited_names.iterrows():
            internal = str(
                row["内部名"]
            ).strip()
            display = str(
                row["表示名"]
            ).strip()
            if internal and display:
                st.session_state[
                    "team_display_names"
                ][internal] = display

        st.rerun()

    export_names = pd.DataFrame({
        "internal_name": all_internal_teams,
        "display_name": [
            st.session_state[
                "team_display_names"
            ].get(
                team,
                DEFAULT_DISPLAY_NAMES.get(
                    team, team
                ),
            )
            for team in all_internal_teams
        ],
    })

    st.download_button(
        "⬇️ 表示名設定CSVをダウンロード",
        data=export_names.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="team_display_names.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "次回以降も同じ表示名を使う場合は、"
        "ダウンロードしたCSVをGitHubの "
        "data/team_display_names.csv に置いてください。"
        "新しいクラブも一覧に自動追加されます。"
    )


# =========================================================
# データ状態
# =========================================================

st.header(
    "📡 自動データ更新"
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "過去データ",
    f"{len(historical)}試合",
)

c2.metric(
    "公式追加結果",
    f"{len(current_results)}試合",
)

c3.metric(
    "モデル学習試合",
    f"{len(matches)}試合",
)


if official_error:

    st.warning(
        "Jリーグ公式データの取得に失敗したため、"
        "CSVの次節カードを使用しています。"
    )

    st.code(
        official_error
    )

else:

    st.success(
        "Jリーグ公式データを取得できました。"
    )


if unmapped_names:

    st.error(
        "変換表に登録されていない公式チーム名があります："
        + "、".join(unmapped_names)
    )


if mapped_not_in_history:

    st.warning(
        "JPN.csvに同じ名前が見つからないチーム："
        + "、".join(mapped_not_in_history)
    )


# =========================================================
# 現在順位
# =========================================================

st.header(
    "📊 現在のJ1順位"
)

if len(current_standings) == 0:
    st.info(
        "今季の終了済み公式試合がまだないため、順位を計算できません。"
    )
else:
    standings_display = (
        current_standings.copy()
    )
    standings_display["Team"] = (
        standings_display["Team"]
        .map(display_team)
    )
    standings_display = standings_display.rename(
        columns={
            "Rank": "順位",
            "Team": "クラブ",
            "P": "試合",
            "W": "勝",
            "D": "分",
            "L": "敗",
            "GF": "得点",
            "GA": "失点",
            "GD": "得失点差",
            "Pts": "勝点",
        }
    )

    st.dataframe(
        standings_display[
            [
                "順位", "クラブ", "試合", "勝点",
                "勝", "分", "敗",
                "得点", "失点", "得失点差",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Jリーグ公式サイトから取得した今季の終了済み試合を使い、"
        "勝点 → 得失点差 → 得点の順でアプリ内計算しています。"
    )


# =========================================================
# 次節
# =========================================================

st.header(
    "🗓️ 自動取得した次節"
)


if len(next_games) == 0:

    st.error(
        "予測対象の試合を取得できませんでした。"
    )

    st.stop()


if next_round is not None:

    st.subheader(
        f"第{int(next_round)}節"
    )


next_display = (
    next_games[
        [
            "Date",
            "Home",
            "Away",
        ]
    ]
    .copy()
)

next_display["Date"] = (
    next_display["Date"]
    .dt.strftime(
        "%Y-%m-%d"
    )
)

next_display["Home"] = (
    next_display["Home"].map(
        display_team
    )
)
next_display["Away"] = (
    next_display["Away"].map(
        display_team
    )
)
next_display = next_display.rename(
    columns={
        "Date": "日付",
        "Home": "ホーム",
        "Away": "アウェイ",
    }
)

st.dataframe(
    next_display,
    hide_index=True,
    use_container_width=True,
)


# =========================================================
# 全試合予測
# =========================================================

prediction_rows = []


for i, game in (
    next_games.iterrows()
):

    home = game["Home"]
    away = game["Away"]

    (
        home_lambda,
        away_lambda,
        probs,
    ) = predict_match(
        home,
        away,
    )

    labels = [
        "H",
        "D",
        "A",
    ]

    top = labels[
        int(
            np.argmax(probs)
        )
    ]

    prediction_rows.append({
        "No":
            i + 1,

        "Date":
            game["Date"],

        "Home":
            home,

        "Away":
            away,

        "PredHG":
            home_lambda,

        "PredAG":
            away_lambda,

        "H":
            probs[0],

        "D":
            probs[1],

        "A":
            probs[2],

        "Top":
            top,
    })


prediction_df = pd.DataFrame(
    prediction_rows
)


# =========================================================
# 試合を一意に識別するキー
# =========================================================

def fixture_key(
    date,
    home,
    away,
):

    if isinstance(
        date,
        pd.Timestamp,
    ):

        date_text = (
            date.strftime(
                "%Y-%m-%d"
            )
        )

    else:

        date_text = str(date)

    return (
        f"{date_text}"
        f"|{home}"
        f"|{away}"
    )


current_fixture_keys = [
    fixture_key(
        row["Date"],
        row["Home"],
        row["Away"],
    )
    for _, row
    in prediction_df.iterrows()
]


# =========================================================
# 本命の強さ
# =========================================================

def confidence_label(probability):

    percent = float(probability) * 100

    if percent >= 60:
        return "かなり優勢"

    elif percent >= 50:
        return "優勢"

    elif percent >= 43:
        return "やや優勢"

    return "ほぼ互角"


# =========================================================
# 予測表示（スマホ向けカードUI）
# =========================================================

st.header(
    "🔮 G5.1 自動予測"
)

round_text = (
    f"第{int(next_round)}節"
    if next_round is not None
    else "次の予測対象"
)

summary_c1, summary_c2 = st.columns(2)
summary_c1.metric(
    "予測対象",
    round_text,
)
summary_c2.metric(
    "試合数",
    f"{len(prediction_df)}試合",
)

st.caption(
    "H＝ホーム勝ち / D＝引き分け / A＝アウェイ勝ち。"
    "本命は3つの中で最も確率が高い結果です。"
)

for _, row in prediction_df.iterrows():

    top_probability = max(
        float(row["H"]),
        float(row["D"]),
        float(row["A"]),
    )

    strength = confidence_label(
        top_probability
    )

    home_display = display_team(
        row["Home"]
    )
    away_display = display_team(
        row["Away"]
    )

    home_rank = rank_map.get(
        row["Home"]
    )
    away_rank = rank_map.get(
        row["Away"]
    )

    home_label = (
        f"{home_display}（{home_rank}位）"
        if home_rank is not None
        else home_display
    )
    away_label = (
        f"{away_display}（{away_rank}位）"
        if away_rank is not None
        else away_display
    )

    top_text = {
        "H": f"{home_display} 勝ち",
        "D": "引き分け",
        "A": f"{away_display} 勝ち",
    }[row["Top"]]

    with st.container(border=True):

        st.caption(
            row["Date"].strftime(
                "%Y-%m-%d"
            )
        )

        st.subheader(
            f"{home_label}  vs  {away_label}"
        )

        st.markdown(
            f"### ⭐ 本命：{top_text}  "
            f"{top_probability * 100:.1f}%"
        )

        st.caption(
            f"本命の強さ：{strength}"
        )

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "🏠 H",
            f"{row['H'] * 100:.1f}%",
        )

        p2.metric(
            "🤝 D",
            f"{row['D'] * 100:.1f}%",
        )

        p3.metric(
            "✈️ A",
            f"{row['A'] * 100:.1f}%",
        )

        st.progress(
            float(top_probability),
            text=(
                f"本命確率 {top_probability * 100:.1f}%"
            ),
        )

        st.caption(
            "予想得点："
            f"{home_display} {row['PredHG']:.2f} - "
            f"{row['PredAG']:.2f} {away_display}"
            " ｜ 勝点："
            f"{points_map.get(row['Home'], '-')} - "
            f"{points_map.get(row['Away'], '-')}"
        )


with st.expander(
    "📋 予測を表でも見る"
):

    display = prediction_df.copy()

    display["本命の強さ"] = display.apply(
        lambda row: confidence_label(
            max(
                row["H"],
                row["D"],
                row["A"],
            )
        ),
        axis=1,
    )

    display["Home"] = display["Home"].map(
        display_team
    )
    display["Away"] = display["Away"].map(
        display_team
    )

    display["Date"] = (
        display["Date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    display["PredHG"] = (
        display["PredHG"]
        .round(2)
    )

    display["PredAG"] = (
        display["PredAG"]
        .round(2)
    )

    for column in [
        "H",
        "D",
        "A",
    ]:

        display[column] = (
            display[column]
            * 100
        ).round(1)

    display = display.rename(
        columns={
            "Date": "日付",
            "PredHG": "予想HG",
            "PredAG": "予想AG",
            "H": "H %",
            "D": "D %",
            "A": "A %",
            "Top": "本命",
        }
    )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# 一括抽選
# =========================================================

st.header(
    "🎲 全試合を一括抽選"
)

st.write(
    "各試合のH/D/A確率に従って、"
    "今回の予想を1セット生成します。"
)

if st.button(
    "🎲 今回の予想を生成",
    type="primary",
    use_container_width=True,
):

    sample_map = {}

    for _, row in (
        prediction_df.iterrows()
    ):

        probabilities = np.array([
            float(row["H"]),
            float(row["D"]),
            float(row["A"]),
        ])

        probabilities = (
            probabilities
            / probabilities.sum()
        )

        sample = str(
            np.random.choice(
                [
                    "H",
                    "D",
                    "A",
                ],
                p=probabilities,
            )
        )

        key = fixture_key(
            row["Date"],
            row["Home"],
            row["Away"],
        )

        sample_map[key] = sample

    st.session_state[
        "sample_map"
    ] = sample_map


sample_map = (
    st.session_state.get(
        "sample_map",
        {},
    )
)

has_current_samples = all(
    key in sample_map
    for key in current_fixture_keys
)


if has_current_samples:

    final_rows = []
    samples = []

    st.subheader(
        "🎯 今回生成された予想"
    )

    for _, row in (
        prediction_df.iterrows()
    ):

        key = fixture_key(
            row["Date"],
            row["Home"],
            row["Away"],
        )

        sample = sample_map[key]
        samples.append(sample)

        if sample == "H":
            result_text = (
                f"{display_team(row['Home'])} 勝ち"
            )
        elif sample == "D":
            result_text = "引き分け"
        else:
            result_text = (
                f"{display_team(row['Away'])} 勝ち"
            )

        top_probability = max(
            float(row["H"]),
            float(row["D"]),
            float(row["A"]),
        )

        strength = confidence_label(
            top_probability
        )

        with st.container(border=True):

            st.subheader(
                f"{display_team(row['Home'])}  vs  {display_team(row['Away'])}"
            )

            st.markdown(
                f"## 🎲 {result_text}"
            )

            st.caption(
                "今回の抽選："
                f"{sample} ｜ "
                f"本命：{row['Top']} "
                f"({top_probability * 100:.1f}%・{strength})"
            )

            p1, p2, p3 = st.columns(3)
            p1.metric(
                "H",
                f"{row['H'] * 100:.1f}%",
            )
            p2.metric(
                "D",
                f"{row['D'] * 100:.1f}%",
            )
            p3.metric(
                "A",
                f"{row['A'] * 100:.1f}%",
            )

        final_rows.append({
            "No": int(row["No"]),
            "試合": (
                f"{display_team(row['Home'])} vs {display_team(row['Away'])}"
            ),
            "H %": round(
                row["H"] * 100,
                1,
            ),
            "D %": round(
                row["D"] * 100,
                1,
            ),
            "A %": round(
                row["A"] * 100,
                1,
            ),
            "本命": row["Top"],
            "本命の強さ": strength,
            "今回の予想": sample,
            "予想内容": result_text,
        })

    final_df = pd.DataFrame(
        final_rows
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🏠 H",
        samples.count("H"),
    )

    c2.metric(
        "🤝 D",
        samples.count("D"),
    )

    c3.metric(
        "✈️ A",
        samples.count("A"),
    )

    with st.expander(
        "📋 抽選結果を表でも見る"
    ):
        st.dataframe(
            final_df,
            hide_index=True,
            use_container_width=True,
        )

    st.success(
        "この抽選結果はCSV保存用にも保持されています。"
    )

    st.caption(
        "もう一度ボタンを押すと、"
        "同じ確率から新しい予想セットを生成します。"
    )

else:

    st.info(
        "まだ今回の抽選結果はありません。"
        "「🎲 今回の予想を生成」を押してください。"
    )


# =========================================================
# 自動取得詳細
# =========================================================

with st.expander(
    "🔧 自動取得の詳細を見る"
):

    st.write(
        f"公式終了済み試合："
        f"{len(current_results)}試合"
    )

    st.write(
        f"最新学習結果日："
        f"{matches['Date'].max().strftime('%Y-%m-%d')}"
    )

    if next_round is not None:

        st.write(
            f"予測対象："
            f"第{int(next_round)}節"
        )

    st.write(
        f"予測対象試合数："
        f"{len(next_games)}試合"
    )


st.info(
    "公式サイト側で終了済み試合がまだ「vs」のままの場合、"
    "過去日付なら予測対象から除外します。"
)


# =========================================================
# ここから予測記録
# =========================================================

st.divider()

st.header(
    "📒 予測記録・実戦検証"
)

st.write(
    "試合前に出した確率予測と抽選結果を保存し、"
    "試合終了後に公式結果と照合します。"
)


PREDICTION_COLUMNS = [
    "prediction_time",
    "date",
    "home",
    "away",
    "pred_hg",
    "pred_ag",
    "p_h",
    "p_d",
    "p_a",
    "top",
    "sampled",
]


# =========================================================
# 保存済み予測
# =========================================================

def load_saved_predictions():

    try:

        saved = pd.read_csv(
            "data/predictions.csv"
        )

    except Exception:

        saved = pd.DataFrame(
            columns=PREDICTION_COLUMNS
        )

    for column in (
        PREDICTION_COLUMNS
    ):

        if column not in saved.columns:

            saved[column] = np.nan

    saved = saved[
        PREDICTION_COLUMNS
    ].copy()

    saved["date"] = (
        pd.to_datetime(
            saved["date"],
            errors="coerce",
        )
    )

    saved["prediction_time"] = (
        pd.to_datetime(
            saved["prediction_time"],
            errors="coerce",
        )
    )

    for column in [
        "pred_hg",
        "pred_ag",
        "p_h",
        "p_d",
        "p_a",
    ]:

        saved[column] = (
            pd.to_numeric(
                saved[column],
                errors="coerce",
            )
        )

    saved["home"] = (
        saved["home"]
        .astype(str)
        .str.strip()
    )

    saved["away"] = (
        saved["away"]
        .astype(str)
        .str.strip()
    )

    saved["top"] = (
        saved["top"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # NaNを文字列 "nan" にしない
    saved["sampled"] = (
        saved["sampled"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return saved


saved_predictions = (
    load_saved_predictions()
)


# =========================================================
# 現在の予測をCSV形式にする
# =========================================================

def make_current_prediction_export():

    rows = []

    # -----------------------------------------------------
    # session_stateのsample_mapを直接読む
    # -----------------------------------------------------

    current_sample_map = (
        st.session_state.get(
            "sample_map",
            {},
        )
    )

    prediction_time = (
        pd.Timestamp.now()
        .strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    for _, row in (
        prediction_df.iterrows()
    ):

        key = fixture_key(
            row["Date"],
            row["Home"],
            row["Away"],
        )

        sampled = (
            current_sample_map.get(
                key,
                "",
            )
        )

        rows.append({
            "prediction_time":
                prediction_time,

            "date":
                row["Date"].strftime(
                    "%Y-%m-%d"
                ),

            "home":
                row["Home"],

            "away":
                row["Away"],

            "pred_hg":
                round(
                    float(
                        row["PredHG"]
                    ),
                    4,
                ),

            "pred_ag":
                round(
                    float(
                        row["PredAG"]
                    ),
                    4,
                ),

            "p_h":
                round(
                    float(
                        row["H"]
                    ),
                    6,
                ),

            "p_d":
                round(
                    float(
                        row["D"]
                    ),
                    6,
                ),

            "p_a":
                round(
                    float(
                        row["A"]
                    ),
                    6,
                ),

            "top":
                row["Top"],

            "sampled":
                sampled,
        })

    return pd.DataFrame(
        rows,
        columns=PREDICTION_COLUMNS,
    )


current_export = (
    make_current_prediction_export()
)


# =========================================================
# 抽選結果がCSVへ入るか画面でも確認
# =========================================================

st.subheader(
    "🎲 CSV保存前チェック"
)


check_rows = []

for _, row in (
    current_export.iterrows()
):

    check_rows.append({
        "試合":
            (
                f"{display_team(row['home'])} "
                f"vs "
                f"{display_team(row['away'])}"
            ),

        "本命":
            row["top"],

        "保存される抽選":
            (
                row["sampled"]
                if row["sampled"]
                else "未抽選"
            ),
    })


st.dataframe(
    pd.DataFrame(
        check_rows
    ),
    hide_index=True,
    use_container_width=True,
)


if (
    len(current_export) > 0
    and
    current_export[
        "sampled"
    ].isin(
        [
            "H",
            "D",
            "A",
        ]
    ).all()
):

    st.success(
        "✅ 全試合の抽選結果がCSVに保存されます。"
    )

else:

    st.warning(
        "⚠️ 抽選結果が未保存の試合があります。"
        "上の「🎲 今回の予想を生成」を押してから"
        "CSVをダウンロードしてください。"
    )


# =========================================================
# 過去記録 + 今回
# =========================================================

def build_merged_export(
    saved,
    current,
):

    saved_for_merge = (
        saved.copy()
    )

    current_for_merge = (
        current.copy()
    )

    if len(saved_for_merge) > 0:

        saved_for_merge[
            "date"
        ] = (
            saved_for_merge[
                "date"
            ]
            .dt.strftime(
                "%Y-%m-%d"
            )
        )

        saved_for_merge[
            "prediction_time"
        ] = (
            saved_for_merge[
                "prediction_time"
            ]
            .dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

    combined = pd.concat(
        [
            saved_for_merge,
            current_for_merge,
        ],
        ignore_index=True,
    )

    combined[
        "_prediction_sort"
    ] = pd.to_datetime(
        combined[
            "prediction_time"
        ],
        errors="coerce",
    )

    combined = (
        combined
        .sort_values(
            "_prediction_sort"
        )
        .drop_duplicates(
            subset=[
                "date",
                "home",
                "away",
            ],
            keep="first",
        )
        .drop(
            columns=[
                "_prediction_sort"
            ]
        )
        .reset_index(drop=True)
    )

    return combined[
        PREDICTION_COLUMNS
    ]


# =========================================================
# 重要：
# 既存行のsampledが空で、
# 今回の同じ試合にsampledがある場合は補完する
# =========================================================

def merge_with_sample_fix(
    saved,
    current,
):

    merged = build_merged_export(
        saved,
        current,
    )

    # 現在の抽選結果辞書
    current_samples = {}

    for _, row in (
        current.iterrows()
    ):

        key = (
            str(row["date"]),
            str(row["home"]),
            str(row["away"]),
        )

        sample = str(
            row["sampled"]
        ).strip()

        if sample in [
            "H",
            "D",
            "A",
        ]:

            current_samples[
                key
            ] = sample

    # 既存の最初の予測値は変えない。
    # sampledだけ空欄なら今回の抽選で補完する。
    for index, row in (
        merged.iterrows()
    ):

        existing_sample = str(
            row["sampled"]
        ).strip()

        if (
            existing_sample == ""
            or existing_sample.lower()
            == "nan"
        ):

            key = (
                str(row["date"]),
                str(row["home"]),
                str(row["away"]),
            )

            if key in current_samples:

                merged.at[
                    index,
                    "sampled"
                ] = (
                    current_samples[
                        key
                    ]
                )

    return merged


merged_export = (
    merge_with_sample_fix(
        saved_predictions,
        current_export,
    )
)


# =========================================================
# CSVダウンロード
# =========================================================

st.subheader(
    "💾 今回の予測を記録"
)


csv_bytes = (
    merged_export
    .to_csv(
        index=False
    )
    .encode(
        "utf-8-sig"
    )
)


st.download_button(
    label=(
        "⬇️ predictions.csv をダウンロード"
    ),
    data=csv_bytes,
    file_name="predictions.csv",
    mime="text/csv",
    use_container_width=True,
)


st.caption(
    "ダウンロードした predictions.csv を、"
    "GitHub の data/predictions.csv と置き換えて"
    "Commitしてください。"
)


# =========================================================
# 保存済み予測一覧
# =========================================================

st.subheader(
    "📚 GitHubに保存済みの予測"
)


if len(saved_predictions) == 0:

    st.info(
        "まだGitHubに保存済みの予測はありません。"
    )

else:

    saved_display = (
        saved_predictions.copy()
    )

    saved_display[
        "prediction_time"
    ] = (
        saved_display[
            "prediction_time"
        ]
        .dt.strftime(
            "%Y-%m-%d %H:%M"
        )
    )

    saved_display[
        "date"
    ] = (
        saved_display[
            "date"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    for column in [
        "p_h",
        "p_d",
        "p_a",
    ]:

        saved_display[
            column
        ] = (
            saved_display[
                column
            ]
            * 100
        ).round(1)

    saved_display = (
        saved_display.rename(
            columns={
                "prediction_time":
                    "予測日時",

                "date":
                    "試合日",

                "home":
                    "Home",

                "away":
                    "Away",

                "pred_hg":
                    "予想HG",

                "pred_ag":
                    "予想AG",

                "p_h":
                    "H %",

                "p_d":
                    "D %",

                "p_a":
                    "A %",

                "top":
                    "本命",

                "sampled":
                    "抽選",
            }
        )
    )

    if "Home" in saved_display.columns:
        saved_display["Home"] = saved_display["Home"].map(display_team)
    if "Away" in saved_display.columns:
        saved_display["Away"] = saved_display["Away"].map(display_team)

    st.dataframe(
        saved_display,
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# 実際の結果
# =========================================================

def actual_result(
    hg,
    ag,
):

    if hg > ag:

        return "H"

    elif hg < ag:

        return "A"

    return "D"


actual_matches = (
    matches[
        [
            "Date",
            "Home",
            "Away",
            "HG",
            "AG",
        ]
    ]
    .copy()
)

actual_matches[
    "Actual"
] = actual_matches.apply(
    lambda row:
        actual_result(
            row["HG"],
            row["AG"],
        ),
    axis=1,
)


# =========================================================
# 保存予測と実結果を照合
# =========================================================

evaluation = (
    saved_predictions.copy()
)


if len(evaluation) > 0:

    evaluation = (
        evaluation.merge(
            actual_matches,
            left_on=[
                "date",
                "home",
                "away",
            ],
            right_on=[
                "Date",
                "Home",
                "Away",
            ],
            how="left",
        )
    )

    completed_predictions = (
        evaluation[
            evaluation[
                "Actual"
            ].notna()
        ]
        .copy()
    )

else:

    completed_predictions = (
        pd.DataFrame()
    )


# =========================================================
# Log Loss / Brier
# =========================================================

def match_log_loss(row):

    epsilon = 1e-15

    if row["Actual"] == "H":

        probability = row["p_h"]

    elif row["Actual"] == "D":

        probability = row["p_d"]

    else:

        probability = row["p_a"]

    probability = float(
        np.clip(
            probability,
            epsilon,
            1.0 - epsilon,
        )
    )

    return -np.log(
        probability
    )


def match_brier(row):

    actual_h = (
        1.0
        if row["Actual"] == "H"
        else 0.0
    )

    actual_d = (
        1.0
        if row["Actual"] == "D"
        else 0.0
    )

    actual_a = (
        1.0
        if row["Actual"] == "A"
        else 0.0
    )

    return (
        (
            row["p_h"]
            - actual_h
        ) ** 2
        +
        (
            row["p_d"]
            - actual_d
        ) ** 2
        +
        (
            row["p_a"]
            - actual_a
        ) ** 2
    )


# =========================================================
# 実戦成績
# =========================================================

st.header(
    "📊 G5.1 実戦成績"
)


if len(saved_predictions) == 0:

    st.info(
        "まだ正式な予測記録がありません。"
    )


elif len(completed_predictions) == 0:

    st.info(
        "保存済み予測はありますが、"
        "まだ公式結果と一致する終了済み試合がありません。"
    )


else:

    completed_predictions[
        "TopCorrect"
    ] = (
        completed_predictions[
            "top"
        ]
        ==
        completed_predictions[
            "Actual"
        ]
    )

    completed_predictions[
        "SampleCorrect"
    ] = (
        completed_predictions[
            "sampled"
        ]
        ==
        completed_predictions[
            "Actual"
        ]
    )

    completed_predictions[
        "LogLoss"
    ] = (
        completed_predictions.apply(
            match_log_loss,
            axis=1,
        )
    )

    completed_predictions[
        "Brier"
    ] = (
        completed_predictions.apply(
            match_brier,
            axis=1,
        )
    )

    n_completed = len(
        completed_predictions
    )

    accuracy = (
        completed_predictions[
            "TopCorrect"
        ].mean()
    )

    log_loss_value = (
        completed_predictions[
            "LogLoss"
        ].mean()
    )

    brier_value = (
        completed_predictions[
            "Brier"
        ].mean()
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "採点済み",
        f"{n_completed}試合",
    )

    c2.metric(
        "本命 Accuracy",
        f"{accuracy * 100:.1f}%",
    )

    c3.metric(
        "Log Loss",
        f"{log_loss_value:.4f}",
    )

    c4.metric(
        "Brier Score",
        f"{brier_value:.4f}",
    )


    sampled_completed = (
        completed_predictions[
            completed_predictions[
                "sampled"
            ].isin(
                [
                    "H",
                    "D",
                    "A",
                ]
            )
        ]
        .copy()
    )

    if len(sampled_completed) > 0:

        sampled_accuracy = (
            sampled_completed[
                "SampleCorrect"
            ].mean()
        )

        st.metric(
            "🎲 抽選予想 Accuracy",
            (
                f"{sampled_accuracy * 100:.1f}% "
                f"({len(sampled_completed)}試合)"
            ),
        )


    # =====================================================
    # H/D/A別
    # =====================================================

    st.subheader(
        "H / D / A 別の成績"
    )

    result_rows = []

    for result in [
        "H",
        "D",
        "A",
    ]:

        actual_subset = (
            completed_predictions[
                completed_predictions[
                    "Actual"
                ]
                == result
            ]
        )

        actual_count = len(
            actual_subset
        )

        if actual_count > 0:

            recall = (
                (
                    actual_subset[
                        "top"
                    ]
                    == result
                )
                .mean()
            )

        else:

            recall = np.nan

        predicted_count = int(
            (
                completed_predictions[
                    "top"
                ]
                == result
            ).sum()
        )

        result_rows.append({
            "結果":
                result,

            "実際の試合数":
                actual_count,

            "本命にした回数":
                predicted_count,

            "Recall":
                (
                    f"{recall * 100:.1f}%"
                    if not np.isnan(
                        recall
                    )
                    else "-"
                ),
        })


    st.dataframe(
        pd.DataFrame(
            result_rows
        ),
        hide_index=True,
        use_container_width=True,
    )


    # =====================================================
    # 1試合ずつ
    # =====================================================

    st.subheader(
        "📝 1試合ずつの採点"
    )

    detail = (
        completed_predictions[
            [
                "date",
                "home",
                "away",
                "p_h",
                "p_d",
                "p_a",
                "top",
                "sampled",
                "HG",
                "AG",
                "Actual",
                "TopCorrect",
                "LogLoss",
                "Brier",
            ]
        ]
        .copy()
    )

    detail[
        "date"
    ] = (
        detail[
            "date"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    for column in [
        "p_h",
        "p_d",
        "p_a",
    ]:

        detail[
            column
        ] = (
            detail[
                column
            ]
            * 100
        ).round(1)

    detail[
        "LogLoss"
    ] = (
        detail[
            "LogLoss"
        ].round(4)
    )

    detail[
        "Brier"
    ] = (
        detail[
            "Brier"
        ].round(4)
    )

    detail[
        "TopCorrect"
    ] = (
        detail[
            "TopCorrect"
        ]
        .map({
            True: "○",
            False: "×",
        })
    )

    detail = (
        detail.rename(
            columns={
                "date":
                    "試合日",

                "home":
                    "Home",

                "away":
                    "Away",

                "p_h":
                    "H %",

                "p_d":
                    "D %",

                "p_a":
                    "A %",

                "top":
                    "本命",

                "sampled":
                    "抽選",

                "HG":
                    "実HG",

                "AG":
                    "実AG",

                "Actual":
                    "実結果",

                "TopCorrect":
                    "本命的中",

                "LogLoss":
                    "Log Loss",

                "Brier":
                    "Brier",
            }
        )
    )

    if "Home" in detail.columns:
        detail["Home"] = detail["Home"].map(display_team)
    if "Away" in detail.columns:
        detail["Away"] = detail["Away"].map(display_team)

    st.dataframe(
        detail,
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# 説明
# =========================================================

with st.expander(
    "📖 実戦成績の見方"
):

    st.markdown(
        """
**本命 Accuracy**

H / D / A の中で最も高確率だった結果が
実際に当たった割合です。

**Log Loss**

実際に起きた結果へ適切な確率を付けられたかを見る指標です。
**小さいほど良い**です。

**Brier Score**

H / D / A の確率全体の精度を見る指標です。
これも**小さいほど良い**です。

**抽選 Accuracy**

H / D / A確率から実際にランダム生成した予想が
当たった割合です。

モデルそのものの比較では、
主に **Log Loss と Brier Score** を見ます。
"""
    )
