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
    layout="wide"
)

st.title("⚽ J1 Matchday Predictor")

st.write(
    "Jリーグ公式データから最新結果と次節を取得し、"
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
# 公式 → モデル側 チーム名
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
# 特徴量
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
# JPN.csv
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
        errors="coerce"
    )

    df["HG"] = pd.to_numeric(
        df["HG"],
        errors="coerce"
    )

    df["AG"] = pd.to_numeric(
        df["AG"],
        errors="coerce"
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

    result = pd.DataFrame({
        "Date": df["Date"],
        "Home": df["Home"].astype(str).str.strip(),
        "Away": df["Away"].astype(str).str.strip(),
        "HG": df["HG"],
        "AG": df["AG"],
        "Source": "JPN.csv",
    })

    return result


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
        errors="coerce"
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
        text
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
        errors="coerce"
    )

    games["AG"] = pd.to_numeric(
        score_parts[1],
        errors="coerce"
    )

    games = games.dropna(
        subset=["Date"]
    ).copy()

    return games


# =========================================================
# 公式取得
# =========================================================

official_error = None

try:

    official_games = (
        load_jleague_official()
    )

except Exception as e:

    official_error = str(e)

    official_games = pd.DataFrame()


# =========================================================
# 過去データ
# =========================================================

historical = load_historical()

known_historical_teams = (
    set(historical["Home"])
    |
    set(historical["Away"])
)


# =========================================================
# 公式チーム名変換チェック
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
# 過去＋現在を結合
# =========================================================

matches = pd.concat(
    [
        historical,
        current_results,
    ],
    ignore_index=True
)

matches["Date"] = pd.to_datetime(
    matches["Date"],
    errors="coerce"
)

matches["HG"] = pd.to_numeric(
    matches["HG"],
    errors="coerce"
)

matches["AG"] = pd.to_numeric(
    matches["AG"],
    errors="coerce"
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
        keep="last"
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
# 予測対象となる次の節
# =========================================================

next_round = None

next_games = pd.DataFrame()

if len(official_games) > 0:

    today = pd.Timestamp.now().normalize()

    # スコアが vs で、
    # 今日以降の試合だけ候補にする
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

        # 最も近い未来試合の日付
        first_future_date = (
            future_games["Date"].min()
        )

        # その日に開催される節番号
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

            # 同じ節の今日以降の未消化試合を全部取得
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
# 公式取得に失敗した場合
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
                errors="coerce"
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
            "Date": match["Date"],
            "Home": home,
            "Away": away,
            "HG": int(hg),
            "AG": int(ag),

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
                    recent_gf_10[
                        home
                    ]
                ),

            "Home_Recent10_GA":
                average(
                    recent_ga_10[
                        home
                    ]
                ),

            "Away_Recent10_GF":
                average(
                    recent_gf_10[
                        away
                    ]
                ),

            "Away_Recent10_GA":
                average(
                    recent_ga_10[
                        away
                    ]
                ),

            "Home_Home10_GF":
                average(
                    home_gf_10[
                        home
                    ]
                ),

            "Home_Home10_GA":
                average(
                    home_ga_10[
                        home
                    ]
                ),

            "Away_Away10_GF":
                average(
                    away_gf_10[
                        away
                    ]
                ),

            "Away_Away10_GA":
                average(
                    away_ga_10[
                        away
                    ]
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

        # ---------------------------------------------
        # 勝点
        # ---------------------------------------------

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

        # ---------------------------------------------
        # 得失点
        # ---------------------------------------------

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

        # ---------------------------------------------
        # Attack / Defense
        # ---------------------------------------------

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

        # ---------------------------------------------
        # Elo
        # ---------------------------------------------

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

        away_actual = (
            1.0
            - home_actual
        )

        expected_away = (
            1.0
            - expected_home
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
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
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
# 最新状態から未来特徴量
# =========================================================

def state_average(
    dictionary,
    team
):

    values = dictionary.get(
        team,
        []
    )

    if len(values) == 0:

        return np.nan

    return float(
        np.mean(values)
    )


def future_features(
    home,
    away
):

    home_elo = (
        current_state[
            "elo"
        ].get(
            home,
            1500.0
        )
    )

    away_elo = (
        current_state[
            "elo"
        ].get(
            away,
            1500.0
        )
    )

    home_attack = (
        current_state[
            "attack"
        ].get(
            home,
            1.0
        )
    )

    away_attack = (
        current_state[
            "attack"
        ].get(
            away,
            1.0
        )
    )

    home_defense = (
        current_state[
            "defense"
        ].get(
            home,
            1.0
        )
    )

    away_defense = (
        current_state[
            "defense"
        ].get(
            away,
            1.0
        )
    )

    home_form = sum(
        current_state[
            "recent_points_5"
        ].get(
            home,
            []
        )
    )

    away_form = sum(
        current_state[
            "recent_points_5"
        ].get(
            away,
            []
        )
    )

    row = {
        "Home": home,
        "Away": away,

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
# Poisson H/D/A
# =========================================================

def poisson_probability(
    goals,
    expected
):

    return (
        exp(-expected)
        * expected ** goals
        / factorial(goals)
    )


def result_probabilities(
    home_lambda,
    away_lambda
):

    h = 0.0
    d = 0.0
    a = 0.0

    for hg in range(11):

        ph = poisson_probability(
            hg,
            home_lambda
        )

        for ag in range(11):

            probability = (
                ph
                * poisson_probability(
                    ag,
                    away_lambda
                )
            )

            if hg > ag:

                h += probability

            elif hg == ag:

                d += probability

            else:

                a += probability

    total = h + d + a

    return np.array(
        [
            h / total,
            d / total,
            a / total,
        ]
    )


def predict_match(
    home,
    away
):

    x = future_features(
        home,
        away
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
            5.0
        )
    )

    away_lambda = float(
        np.clip(
            away_lambda,
            0.05,
            5.0
        )
    )

    probabilities = (
        result_probabilities(
            home_lambda,
            away_lambda
        )
    )

    return (
        home_lambda,
        away_lambda,
        probabilities,
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
    f"{len(historical)}試合"
)

c2.metric(
    "公式追加結果",
    f"{len(current_results)}試合"
)

c3.metric(
    "モデル学習試合",
    f"{len(matches)}試合"
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


# =========================================================
# 名前変換チェック
# =========================================================

if unmapped_names:

    st.error(
        "変換表に登録されていない公式チーム名があります："
        + "、".join(unmapped_names)
    )


if mapped_not_in_history:

    st.warning(
        "JPN.csvに同じ名前が見つからないチーム："
        + "、".join(mapped_not_in_history)
        + "。昇格クラブまたは表記差の可能性があります。"
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


next_display = next_games[
    [
        "Date",
        "Home",
        "Away",
    ]
].copy()

next_display["Date"] = (
    next_display["Date"]
    .dt.strftime(
        "%Y-%m-%d"
    )
)

st.dataframe(
    next_display,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 予測
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
        away
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
        "No": i + 1,
        "Date": game["Date"],
        "Home": home,
        "Away": away,
        "PredHG": home_lambda,
        "PredAG": away_lambda,
        "H": probs[0],
        "D": probs[1],
        "A": probs[2],
        "Top": top,
    })


prediction_df = pd.DataFrame(
    prediction_rows
)


st.header(
    "🔮 G5.1 自動予測"
)


display = prediction_df.copy()

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
    use_container_width=True
)


# =========================================================
# 一括抽選
# =========================================================

st.header(
    "🎲 全試合を一括抽選"
)


fixture_signature = [
    (
        row["Home"],
        row["Away"],
    )
    for _, row
    in prediction_df.iterrows()
]


if st.button(
    "🎲 今回の予想を生成",
    type="primary",
    use_container_width=True,
):

    samples = []

    for _, row in (
        prediction_df.iterrows()
    ):

        probabilities = np.array([
            row["H"],
            row["D"],
            row["A"],
        ])

        probabilities = (
            probabilities
            / probabilities.sum()
        )

        sample = np.random.choice(
            [
                "H",
                "D",
                "A",
            ],
            p=probabilities,
        )

        samples.append(
            sample
        )

    st.session_state[
        "auto_samples"
    ] = samples

    st.session_state[
        "auto_signature"
    ] = fixture_signature


if (
    st.session_state.get(
        "auto_signature"
    )
    == fixture_signature
    and
    "auto_samples"
    in st.session_state
):

    samples = (
        st.session_state[
            "auto_samples"
        ]
    )

    final_rows = []

    for (
        (_, row),
        sample,
    ) in zip(
        prediction_df.iterrows(),
        samples,
    ):

        if sample == "H":

            result_text = (
                f"{row['Home']} 勝ち"
            )

        elif sample == "D":

            result_text = (
                "引き分け"
            )

        else:

            result_text = (
                f"{row['Away']} 勝ち"
            )

        final_rows.append({
            "No":
                int(row["No"]),

            "試合":
                (
                    f"{row['Home']} "
                    f"vs "
                    f"{row['Away']}"
                ),

            "H %":
                round(
                    row["H"]
                    * 100,
                    1
                ),

            "D %":
                round(
                    row["D"]
                    * 100,
                    1
                ),

            "A %":
                round(
                    row["A"]
                    * 100,
                    1
                ),

            "本命":
                row["Top"],

            "今回の予想":
                sample,

            "予想内容":
                result_text,
        })

    final_df = pd.DataFrame(
        final_rows
    )

    st.subheader(
        "🎯 今回生成された予想"
    )

    st.dataframe(
        final_df,
        hide_index=True,
        use_container_width=True
    )

    h_count = samples.count("H")
    d_count = samples.count("D")
    a_count = samples.count("A")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🏠 H",
        h_count
    )

    c2.metric(
        "🤝 D",
        d_count
    )

    c3.metric(
        "✈️ A",
        a_count
    )

    st.caption(
        "もう一度ボタンを押すと、"
        "同じH/D/A確率から別の予想セットを生成します。"
    )


# =========================================================
# 詳細
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
            f"予測対象：第{int(next_round)}節"
        )

    st.write(
        f"予測対象試合数："
        f"{len(next_games)}試合"
    )

    if mapped_not_in_history:

        st.write(
            "JPN.csvで完全一致しなかったチーム："
        )

        st.write(
            mapped_not_in_history
        )


st.info(
    "公式サイト側で終了済み試合がまだ「vs」のままの場合、"
    "その試合は結果として取り込まず、"
    "過去日付なら予測対象にも入れない設計です。"
)
