import streamlit as st
import pandas as pd
import numpy as np

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
    "過去データ＋2026年の最新結果を使って、"
    "G5.1で次節を予測します。"
)

st.caption(
    "G5.1 / Attack・Defense Rating 学習率 = 0.06"
)

LEARNING_RATE = 0.06
K_FACTOR = 20


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
    "DefenseAdvantage"
]

CATEGORICAL_FEATURES = [
    "Home",
    "Away"
]

ALL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# =========================================================
# JPN.csv
# 過去データ読み込み
# =========================================================

@st.cache_data
def load_historical_results():

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

    df["Season"] = (
        df["Season"]
        .astype(str)
        .str.strip()
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

    result = pd.DataFrame(
        {
            "Date":
                df["Date"],

            "Home":
                df["Home"]
                .astype(str)
                .str.strip(),

            "Away":
                df["Away"]
                .astype(str)
                .str.strip(),

            "HG":
                df["HG"],

            "AG":
                df["AG"],

            "Season":
                df["Season"],

            "Source":
                "JPN.csv"
        }
    )

    return result


# =========================================================
# j1_2026_results.csv
# =========================================================

@st.cache_data
def load_2026_results():

    columns = [
        "Date",
        "Home",
        "Away",
        "HG",
        "AG",
        "Season",
        "Source"
    ]

    try:

        raw = pd.read_csv(
            "data/j1_2026_results.csv"
        )

    except (
        FileNotFoundError,
        pd.errors.EmptyDataError
    ):

        return pd.DataFrame(
            columns=columns
        )


    required = {
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals"
    }


    raw.columns = [
        str(column).strip()
        for column in raw.columns
    ]


    if not required.issubset(
        set(raw.columns)
    ):

        return pd.DataFrame(
            columns=columns
        )


    # ヘッダーだけの場合
    if len(raw) == 0:

        return pd.DataFrame(
            columns=columns
        )


    raw["date"] = pd.to_datetime(
        raw["date"],
        errors="coerce"
    )

    raw["home_goals"] = pd.to_numeric(
        raw["home_goals"],
        errors="coerce"
    )

    raw["away_goals"] = pd.to_numeric(
        raw["away_goals"],
        errors="coerce"
    )


    raw = raw.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals"
        ]
    ).copy()


    if len(raw) == 0:

        return pd.DataFrame(
            columns=columns
        )


    result = pd.DataFrame(
        {
            "Date":
                raw["date"],

            "Home":
                raw["home_team"]
                .astype(str)
                .str.strip(),

            "Away":
                raw["away_team"]
                .astype(str)
                .str.strip(),

            "HG":
                raw["home_goals"],

            "AG":
                raw["away_goals"],

            "Season":
                "2026",

            "Source":
                "j1_2026_results.csv"
        }
    )


    return result


# =========================================================
# 過去＋2026を結合
# =========================================================

historical_matches = (
    load_historical_results()
)

matches_2026 = (
    load_2026_results()
)


matches = pd.concat(
    [
        historical_matches,
        matches_2026
    ],
    ignore_index=True
)


# =========================================================
# データ掃除
# =========================================================

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

matches["Home"] = (
    matches["Home"]
    .astype(str)
    .str.strip()
)

matches["Away"] = (
    matches["Away"]
    .astype(str)
    .str.strip()
)


matches = matches.dropna(
    subset=[
        "Date",
        "Home",
        "Away",
        "HG",
        "AG"
    ]
).copy()


# =========================================================
# 重複試合を除外
#
# 同じ日付・Home・Awayなら1試合として扱う
#
# 2026ファイルを後から追加した場合は
# 後にあるデータを優先
# =========================================================

matches = (
    matches
    .drop_duplicates(
        subset=[
            "Date",
            "Home",
            "Away"
        ],
        keep="last"
    )
    .sort_values(
        [
            "Date",
            "Home",
            "Away"
        ]
    )
    .reset_index(
        drop=True
    )
)


if len(matches) == 0:

    st.error(
        "終了済みJ1試合を読み込めませんでした。"
    )

    st.stop()


# =========================================================
# next_matches.csv
# =========================================================

@st.cache_data
def load_next_matches():

    columns = [
        "date",
        "home_team",
        "away_team"
    ]

    try:

        df = pd.read_csv(
            "data/next_matches.csv"
        )

    except (
        FileNotFoundError,
        pd.errors.EmptyDataError
    ):

        return pd.DataFrame(
            columns=columns
        )


    df.columns = [
        str(column).strip()
        for column in df.columns
    ]


    required = {
        "date",
        "home_team",
        "away_team"
    }


    if not required.issubset(
        set(df.columns)
    ):

        return pd.DataFrame(
            columns=columns
        )


    df = df[
        [
            "date",
            "home_team",
            "away_team"
        ]
    ].copy()


    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )


    df["home_team"] = (
        df["home_team"]
        .astype(str)
        .str.strip()
    )


    df["away_team"] = (
        df["away_team"]
        .astype(str)
        .str.strip()
    )


    df = df.dropna(
        subset=[
            "date"
        ]
    ).copy()


    df = df[
        (df["home_team"] != "")
        &
        (df["away_team"] != "")
        &
        (df["home_team"] != "nan")
        &
        (df["away_team"] != "nan")
    ].copy()


    df = (
        df
        .drop_duplicates(
            subset=[
                "date",
                "home_team",
                "away_team"
            ]
        )
        .sort_values(
            [
                "date",
                "home_team"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    return df


next_matches = load_next_matches()


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
# 全履歴から特徴量＋最新状態
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

    MIN_RATING = 0.50
    MAX_RATING = 1.80


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
        # 試合前状態
        # =================================================

        home_elo = elo[home]
        away_elo = elo[away]

        home_attack = attack[home]
        away_attack = attack[away]

        home_defense = defense[home]
        away_defense = defense[away]

        home_form = sum(
            recent_points_5[
                home
            ]
        )

        away_form = sum(
            recent_points_5[
                away
            ]
        )


        # =================================================
        # 学習用特徴量
        # =================================================

        rows.append(
            {
                "Season":
                    str(
                        match[
                            "Season"
                        ]
                    ),

                "Date":
                    match[
                        "Date"
                    ],

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
                    - home_defense
            }
        )


        # =================================================
        # 試合結果
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


        # =================================================
        # 直近データ更新
        # =================================================

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


        # =================================================
        # Attack / Defense更新
        # =================================================

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


        attack[
            home
        ] = np.clip(
            home_attack
            * (
                home_ratio
                ** LEARNING_RATE
            ),
            MIN_RATING,
            MAX_RATING
        )


        attack[
            away
        ] = np.clip(
            away_attack
            * (
                away_ratio
                ** LEARNING_RATE
            ),
            MIN_RATING,
            MAX_RATING
        )


        defense[
            home
        ] = np.clip(
            home_defense
            * (
                away_ratio
                ** LEARNING_RATE
            ),
            MIN_RATING,
            MAX_RATING
        )


        defense[
            away
        ] = np.clip(
            away_defense
            * (
                home_ratio
                ** LEARNING_RATE
            ),
            MIN_RATING,
            MAX_RATING
        )


        # =================================================
        # Elo更新
        # =================================================

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


        elo[
            home
        ] = (
            home_elo
            +
            K_FACTOR
            * (
                home_actual
                - expected_home
            )
        )


        elo[
            away
        ] = (
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
            dict(
                elo
            ),

        "attack":
            dict(
                attack
            ),

        "defense":
            dict(
                defense
            ),

        "recent_points_5":
            {
                key: list(value)
                for key, value
                in recent_points_5.items()
            },

        "recent_gf_10":
            {
                key: list(value)
                for key, value
                in recent_gf_10.items()
            },

        "recent_ga_10":
            {
                key: list(value)
                for key, value
                in recent_ga_10.items()
            },

        "home_gf_10":
            {
                key: list(value)
                for key, value
                in home_gf_10.items()
            },

        "home_ga_10":
            {
                key: list(value)
                for key, value
                in home_ga_10.items()
            },

        "away_gf_10":
            {
                key: list(value)
                for key, value
                in away_gf_10.items()
            },

        "away_ga_10":
            {
                key: list(value)
                for key, value
                in away_ga_10.items()
            }
    }


    return (
        history,
        state
    )


history, current_state = (
    build_history(
        matches
    )
)


# =========================================================
# モデル作成
# =========================================================

def make_model():

    numeric_transformer = Pipeline(
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
            )
        ]
    )


    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )


    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                NUMERIC_FEATURES
            ),

            (
                "team",
                categorical_transformer,
                CATEGORICAL_FEATURES
            )
        ]
    )


    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),

            (
                "poisson",
                PoissonRegressor(
                    alpha=0.1,
                    max_iter=1000
                )
            )
        ]
    )


    return model


# =========================================================
# モデル学習
# =========================================================

@st.cache_resource
def train_models(history):

    home_model = make_model()
    away_model = make_model()


    home_model.fit(
        history[
            ALL_FEATURES
        ],
        history[
            "HG"
        ]
    )


    away_model.fit(
        history[
            ALL_FEATURES
        ],
        history[
            "AG"
        ]
    )


    return (
        home_model,
        away_model
    )


home_model, away_model = (
    train_models(
        history
    )
)


# =========================================================
# 状態平均
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


# =========================================================
# 未来試合特徴量
# =========================================================

def make_future_match(
    home,
    away,
    state
):

    home_elo = (
        state[
            "elo"
        ].get(
            home,
            1500.0
        )
    )


    away_elo = (
        state[
            "elo"
        ].get(
            away,
            1500.0
        )
    )


    home_attack = (
        state[
            "attack"
        ].get(
            home,
            1.0
        )
    )


    away_attack = (
        state[
            "attack"
        ].get(
            away,
            1.0
        )
    )


    home_defense = (
        state[
            "defense"
        ].get(
            home,
            1.0
        )
    )


    away_defense = (
        state[
            "defense"
        ].get(
            away,
            1.0
        )
    )


    home_form = sum(
        state[
            "recent_points_5"
        ].get(
            home,
            []
        )
    )


    away_form = sum(
        state[
            "recent_points_5"
        ].get(
            away,
            []
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
                state[
                    "recent_gf_10"
                ],
                home
            ),

        "Home_Recent10_GA":
            state_average(
                state[
                    "recent_ga_10"
                ],
                home
            ),

        "Away_Recent10_GF":
            state_average(
                state[
                    "recent_gf_10"
                ],
                away
            ),

        "Away_Recent10_GA":
            state_average(
                state[
                    "recent_ga_10"
                ],
                away
            ),

        "Home_Home10_GF":
            state_average(
                state[
                    "home_gf_10"
                ],
                home
            ),

        "Home_Home10_GA":
            state_average(
                state[
                    "home_ga_10"
                ],
                home
            ),

        "Away_Away10_GF":
            state_average(
                state[
                    "away_gf_10"
                ],
                away
            ),

        "Away_Away10_GA":
            state_average(
                state[
                    "away_ga_10"
                ],
                away
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
            - home_defense
    }


    return pd.DataFrame(
        [row]
    )


# =========================================================
# Poisson
# =========================================================

def poisson_probability(
    goals,
    expected_goals
):

    return (
        exp(
            -expected_goals
        )
        *
        expected_goals ** goals
        /
        factorial(
            goals
        )
    )


# =========================================================
# H/D/A
# =========================================================

def calculate_probabilities(
    home_lambda,
    away_lambda
):

    home_win = 0.0
    draw = 0.0
    away_win = 0.0


    for hg in range(11):

        ph = poisson_probability(
            hg,
            home_lambda
        )


        for ag in range(11):

            pa = poisson_probability(
                ag,
                away_lambda
            )


            p = (
                ph
                * pa
            )


            if hg > ag:

                home_win += p

            elif hg == ag:

                draw += p

            else:

                away_win += p


    total = (
        home_win
        + draw
        + away_win
    )


    return np.array(
        [
            home_win / total,
            draw / total,
            away_win / total
        ]
    )


# =========================================================
# 1試合予測
# =========================================================

def predict_match(
    home,
    away
):

    future = make_future_match(
        home,
        away,
        current_state
    )


    home_lambda = float(
        home_model.predict(
            future[
                ALL_FEATURES
            ]
        )[0]
    )


    away_lambda = float(
        away_model.predict(
            future[
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
        calculate_probabilities(
            home_lambda,
            away_lambda
        )
    )


    return (
        home_lambda,
        away_lambda,
        probabilities
    )


# =========================================================
# データ状態表示
# =========================================================

st.header(
    "📅 現在のデータ状態"
)


latest_date = (
    matches[
        "Date"
    ].max()
)


historical_count = len(
    historical_matches
)


results_2026_count = len(
    matches_2026
)


col1, col2, col3, col4 = (
    st.columns(4)
)


col1.metric(
    "最新結果日",
    latest_date.strftime(
        "%Y-%m-%d"
    )
)


col2.metric(
    "過去データ",
    historical_count
)


col3.metric(
    "2026追加結果",
    results_2026_count
)


col4.metric(
    "合計学習試合",
    len(matches)
)


# =========================================================
# 2026結果確認
# =========================================================

with st.expander(
    "📘 2026年の追加結果を確認"
):

    if len(
        matches_2026
    ) == 0:

        st.info(
            "j1_2026_results.csv は現在ヘッダーのみです。"
            "これは正常です。"
        )

    else:

        show_2026 = (
            matches_2026.copy()
        )


        show_2026[
            "Date"
        ] = (
            show_2026[
                "Date"
            ].dt.strftime(
                "%Y-%m-%d"
            )
        )


        st.dataframe(
            show_2026[
                [
                    "Date",
                    "Home",
                    "Away",
                    "HG",
                    "AG"
                ]
            ],
            hide_index=True,
            use_container_width=True
        )


# =========================================================
# 次節
# =========================================================

st.header(
    "🗓️ 次節カード"
)


if len(
    next_matches
) == 0:

    st.error(
        "next_matches.csvに試合がありません。"
    )

    st.stop()


fixture_display = (
    next_matches.copy()
)


fixture_display[
    "date"
] = (
    fixture_display[
        "date"
    ].dt.strftime(
        "%Y-%m-%d"
    )
)


fixture_display = (
    fixture_display.rename(
        columns={
            "date":
                "日付",

            "home_team":
                "Home",

            "away_team":
                "Away"
        }
    )
)


st.dataframe(
    fixture_display,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 未知チーム
# =========================================================

known_teams = (
    set(
        matches[
            "Home"
        ]
    )
    |
    set(
        matches[
            "Away"
        ]
    )
)


future_teams = (
    set(
        next_matches[
            "home_team"
        ]
    )
    |
    set(
        next_matches[
            "away_team"
        ]
    )
)


unknown_teams = sorted(
    future_teams
    - known_teams
)


if len(
    unknown_teams
) > 0:

    st.warning(
        "過去データがないチーム："
        + "、".join(
            unknown_teams
        )
        + "。初期値を使って予測します。"
    )


# =========================================================
# 全試合予測
# =========================================================

prediction_rows = []


for i, fixture in (
    next_matches.iterrows()
):

    home = fixture[
        "home_team"
    ]

    away = fixture[
        "away_team"
    ]


    (
        home_lambda,
        away_lambda,
        probabilities
    ) = predict_match(
        home,
        away
    )


    h = float(
        probabilities[0]
    )

    d = float(
        probabilities[1]
    )

    a = float(
        probabilities[2]
    )


    top_result = [
        "H",
        "D",
        "A"
    ][
        int(
            np.argmax(
                probabilities
            )
        )
    ]


    prediction_rows.append(
        {
            "No":
                i + 1,

            "Date":
                fixture[
                    "date"
                ],

            "Home":
                home,

            "Away":
                away,

            "PredHG":
                home_lambda,

            "PredAG":
                away_lambda,

            "H":
                h,

            "D":
                d,

            "A":
                a,

            "Top":
                top_result
        }
    )


prediction_df = pd.DataFrame(
    prediction_rows
)


# =========================================================
# 予測結果
# =========================================================

st.header(
    "🔮 G5.1 次節予測"
)


display_df = (
    prediction_df.copy()
)


display_df[
    "Date"
] = (
    display_df[
        "Date"
    ].dt.strftime(
        "%Y-%m-%d"
    )
)


display_df[
    "PredHG"
] = (
    display_df[
        "PredHG"
    ].round(2)
)


display_df[
    "PredAG"
] = (
    display_df[
        "PredAG"
    ].round(2)
)


for column in [
    "H",
    "D",
    "A"
]:

    display_df[
        column
    ] = (
        display_df[
            column
        ]
        * 100
    ).round(1)


display_df = (
    display_df.rename(
        columns={
            "Date":
                "日付",

            "PredHG":
                "予想HG",

            "PredAG":
                "予想AG",

            "H":
                "H %",

            "D":
                "D %",

            "A":
                "A %",

            "Top":
                "本命"
        }
    )
)


st.dataframe(
    display_df,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 本命一覧
# =========================================================

st.subheader(
    "📊 モデル本命"
)


favorite_rows = []


for _, row in (
    prediction_df.iterrows()
):

    if row["Top"] == "H":

        text = (
            f"{row['Home']} 勝ち"
        )

    elif row["Top"] == "D":

        text = (
            "引き分け"
        )

    else:

        text = (
            f"{row['Away']} 勝ち"
        )


    probability = max(
        row["H"],
        row["D"],
        row["A"]
    )


    favorite_rows.append(
        {
            "No":
                int(
                    row["No"]
                ),

            "試合":
                (
                    f"{row['Home']} "
                    f"vs "
                    f"{row['Away']}"
                ),

            "本命":
                text,

            "確率":
                f"{probability:.1%}"
        }
    )


st.dataframe(
    pd.DataFrame(
        favorite_rows
    ),
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 一括抽選
# =========================================================

st.header(
    "🎲 次節を一括抽選"
)


st.write(
    "各試合のH/D/A確率に従って、"
    "次節全体を1回シミュレーションします。"
)


fixture_signature = [
    (
        row["Home"],
        row["Away"]
    )
    for _, row
    in prediction_df.iterrows()
]


if st.button(
    "🎲 全試合を一括抽選",
    type="primary",
    use_container_width=True
):

    samples = []


    for _, row in (
        prediction_df.iterrows()
    ):

        probabilities = np.array(
            [
                row["H"],
                row["D"],
                row["A"]
            ],
            dtype=float
        )


        probabilities = (
            probabilities
            / probabilities.sum()
        )


        sample = np.random.choice(
            [
                "H",
                "D",
                "A"
            ],
            p=probabilities
        )


        samples.append(
            sample
        )


    st.session_state[
        "samples"
    ] = samples


    st.session_state[
        "fixture_signature"
    ] = fixture_signature


# =========================================================
# 抽選結果
# =========================================================

if (
    st.session_state.get(
        "fixture_signature"
    )
    == fixture_signature
    and
    "samples"
    in st.session_state
):

    samples = (
        st.session_state[
            "samples"
        ]
    )


    final_rows = []


    for (
        (_, row),
        sample
    ) in zip(
        prediction_df.iterrows(),
        samples
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


        final_rows.append(
            {
                "No":
                    int(
                        row["No"]
                    ),

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
                    row[
                        "Top"
                    ],

                "今回の予想":
                    sample,

                "予想内容":
                    result_text
            }
        )


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


    c1, c2, c3 = (
        st.columns(3)
    )


    c1.metric(
        "🏠 H",
        samples.count(
            "H"
        )
    )


    c2.metric(
        "🤝 D",
        samples.count(
            "D"
        )
    )


    c3.metric(
        "✈️ A",
        samples.count(
            "A"
        )
    )


    st.caption(
        "もう一度ボタンを押すと、"
        "同じ確率から別の予想セットになります。"
    )


# =========================================================
# チーム状態
# =========================================================

with st.expander(
    "💪 現在のチーム状態を見る"
):

    teams_in_fixtures = sorted(
        future_teams
    )


    state_rows = []


    for team in teams_in_fixtures:

        state_rows.append(
            {
                "Team":
                    team,

                "Elo":
                    round(
                        current_state[
                            "elo"
                        ].get(
                            team,
                            1500.0
                        ),
                        1
                    ),

                "Attack":
                    round(
                        current_state[
                            "attack"
                        ].get(
                            team,
                            1.0
                        ),
                        3
                    ),

                "Defense":
                    round(
                        current_state[
                            "defense"
                        ].get(
                            team,
                            1.0
                        ),
                        3
                    ),

                "直近5試合勝点":
                    sum(
                        current_state[
                            "recent_points_5"
                        ].get(
                            team,
                            []
                        )
                    )
            }
        )


    st.dataframe(
        pd.DataFrame(
            state_rows
        ),
        hide_index=True,
        use_container_width=True
    )


# =========================================================
# 説明
# =========================================================

with st.expander(
    "ℹ️ 現在のデータの流れ"
):

    st.write(
        """
### ① JPN.csv

過去のJ1終了済み試合です。

↓

### ② j1_2026_results.csv

2026年の終了済み試合を追加します。

↓

### ③ 自動結合

2つの結果データを日付順に並べます。

↓

### ④ G5.1

Elo、直近成績、Attack Rating、Defense Ratingを
試合ごとに更新します。

↓

### ⑤ next_matches.csv

まだ結果が出ていない次節カードです。

↓

### ⑥ 予測

予想得点とH/D/A確率を計算します。

↓

### ⑦ 一括抽選

H/D/A確率に従って次節全体を1回シミュレーションします。
"""
    )


# =========================================================
# 注意
# =========================================================

if len(
    matches_2026
) == 0:

    st.info(
        "現在、j1_2026_results.csvには"
        "2026年の試合結果がまだ入っていません。"
        "そのためチーム状態はJPN.csvの"
        "最新結果時点です。"
    )

else:

    st.success(
        f"2026年の終了済み試合 "
        f"{len(matches_2026)}試合を"
        "チーム状態とモデル学習に反映しています。"
    )
