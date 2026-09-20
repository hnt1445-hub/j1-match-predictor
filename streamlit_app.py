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
    "G5.1を使って、次節の複数試合をまとめて予測します。"
)

st.caption(
    "モデル：G5.1 / Attack・Defense Rating更新速度 = 0.06"
)

LEARNING_RATE = 0.06
K_FACTOR = 20


# =========================================================
# 使用特徴量
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
# データ読み込み
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/JPN.csv"
    )

    df = df[
        df["League"] == "J1 League"
    ].copy()

    df["Season"] = (
        df["Season"]
        .astype(str)
        .str.strip()
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
        .reset_index(
            drop=True
        )
    )

    return df


matches = load_data()


if len(matches) == 0:

    st.error(
        "J1の試合データが見つかりません。"
    )

    st.stop()


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
# 履歴特徴量 + 最新チーム状態
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
        # 学習用の試合前特徴量
        # =================================================

        rows.append(
            {
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
        # 試合後更新
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
        # 最近成績
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
        # Attack / Defense Rating
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
        # Elo
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


        away_actual = (
            1.0
            - home_actual
        )

        expected_away = (
            1.0
            - expected_home
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
# モデル
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


    return Pipeline(
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
# 最新状態の平均
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
# 未来試合の特徴量
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

            probability = (
                ph * pa
            )


            if hg > ag:

                home_win += (
                    probability
                )

            elif hg == ag:

                draw += (
                    probability
                )

            else:

                away_win += (
                    probability
                )


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
# 1試合を予測
# =========================================================

def predict_match(
    home,
    away
):

    future_match = (
        make_future_match(
            home,
            away,
            current_state
        )
    )


    home_lambda = float(
        home_model.predict(
            future_match[
                ALL_FEATURES
            ]
        )[0]
    )


    away_lambda = float(
        away_model.predict(
            future_match[
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
# 最新シーズンのチーム
# Seasonを整数変換しない
# =========================================================

latest_season = str(
    matches.iloc[-1][
        "Season"
    ]
)


latest_season_matches = (
    matches[
        matches[
            "Season"
        ].astype(str)
        == latest_season
    ]
    .copy()
)


teams = sorted(
    set(
        latest_season_matches[
            "Home"
        ]
    )
    |
    set(
        latest_season_matches[
            "Away"
        ]
    )
)


# フォールバック
if len(teams) < 2:

    teams = sorted(
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


# =========================================================
# 使用データ
# =========================================================

st.header(
    "📅 モデルの状態"
)


latest_date = (
    matches[
        "Date"
    ].max()
)


c1, c2, c3 = st.columns(3)


c1.metric(
    "最新収録試合",
    latest_date.strftime(
        "%Y-%m-%d"
    )
)


c2.metric(
    "収録J1試合数",
    len(matches)
)


c3.metric(
    "選択可能チーム",
    len(teams)
)


st.caption(
    f"チーム候補に使用しているシーズン：{latest_season}"
)


# =========================================================
# 試合数
# =========================================================

st.header(
    "📝 次節カードを設定"
)


number_of_matches = st.number_input(
    "試合数",
    min_value=1,
    max_value=10,
    value=min(
        10,
        max(
            1,
            len(teams) // 2
        )
    ),
    step=1
)


number_of_matches = int(
    number_of_matches
)


# =========================================================
# 試合選択
# =========================================================

selected_matches = []


for i in range(
    number_of_matches
):

    st.markdown(
        f"**第 {i + 1} 試合**"
    )


    col_home, col_away = (
        st.columns(2)
    )


    default_home_index = (
        (i * 2)
        % len(teams)
    )


    default_away_index = (
        (i * 2 + 1)
        % len(teams)
    )


    with col_home:

        home = st.selectbox(
            "Home",
            teams,
            index=default_home_index,
            key=f"home_{i}"
        )


    away_options = [
        team
        for team in teams
        if team != home
    ]


    default_away_team = (
        teams[
            default_away_index
        ]
    )


    if (
        default_away_team
        in away_options
    ):

        away_index = (
            away_options.index(
                default_away_team
            )
        )

    else:

        away_index = 0


    with col_away:

        away = st.selectbox(
            "Away",
            away_options,
            index=away_index,
            key=f"away_{i}"
        )


    selected_matches.append(
        (
            home,
            away
        )
    )


# =========================================================
# 重複チーム確認
# =========================================================

all_selected_teams = []


for home, away in selected_matches:

    all_selected_teams.append(
        home
    )

    all_selected_teams.append(
        away
    )


duplicate_teams = sorted(
    {
        team
        for team
        in all_selected_teams
        if all_selected_teams.count(
            team
        ) > 1
    }
)


if len(
    duplicate_teams
) > 0:

    st.warning(
        "同じ節に同じチームが複数回選ばれています："
        + "、".join(
            duplicate_teams
        )
    )


# =========================================================
# 全試合予測
# =========================================================

st.header(
    "🔮 次節予測"
)


prediction_rows = []


for i, (
    home,
    away
) in enumerate(
    selected_matches
):

    (
        home_lambda,
        away_lambda,
        probabilities
    ) = predict_match(
        home,
        away
    )


    home_probability = float(
        probabilities[0]
    )

    draw_probability = float(
        probabilities[1]
    )

    away_probability = float(
        probabilities[2]
    )


    top_index = int(
        np.argmax(
            probabilities
        )
    )


    top_result = [
        "H",
        "D",
        "A"
    ][
        top_index
    ]


    prediction_rows.append(
        {
            "No":
                i + 1,

            "Home":
                home,

            "Away":
                away,

            "予想HG":
                home_lambda,

            "予想AG":
                away_lambda,

            "H":
                home_probability,

            "D":
                draw_probability,

            "A":
                away_probability,

            "本命":
                top_result
        }
    )


prediction_df = pd.DataFrame(
    prediction_rows
)


# =========================================================
# 表示用
# =========================================================

display_df = (
    prediction_df.copy()
)


display_df[
    "予想HG"
] = (
    display_df[
        "予想HG"
    ].round(2)
)


display_df[
    "予想AG"
] = (
    display_df[
        "予想AG"
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
            "H":
                "H %",
            "D":
                "D %",
            "A":
                "A %"
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

    result = row[
        "本命"
    ]


    if result == "H":

        text = (
            f"{row['Home']} 勝ち"
        )

    elif result == "D":

        text = (
            "引き分け"
        )

    else:

        text = (
            f"{row['Away']} 勝ち"
        )


    favorite_rows.append(
        {
            "No":
                int(
                    row["No"]
                ),

            "試合":
                (
                    f"{row['Home']}"
                    f" vs "
                    f"{row['Away']}"
                ),

            "本命":
                text,

            "記号":
                result
        }
    )


favorite_df = pd.DataFrame(
    favorite_rows
)


st.dataframe(
    favorite_df,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 一括ランダム抽選
# =========================================================

st.header(
    "🎲 1節まるごとランダム予想"
)


st.write(
    "各試合のH/D/A確率に従って、"
    "全試合を1回ずつ抽選します。"
)


if st.button(
    "🎲 全試合を一括抽選",
    type="primary",
    use_container_width=True
):

    sampled_results = []


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


        sampled = np.random.choice(
            [
                "H",
                "D",
                "A"
            ],
            p=probabilities
        )


        sampled_results.append(
            sampled
        )


    st.session_state[
        "matchday_samples"
    ] = sampled_results


    st.session_state[
        "matchday_signature"
    ] = [
        (
            row["Home"],
            row["Away"]
        )
        for _, row
        in prediction_df.iterrows()
    ]


# =========================================================
# カード変更確認
# =========================================================

current_signature = [
    (
        row["Home"],
        row["Away"]
    )
    for _, row
    in prediction_df.iterrows()
]


stored_signature = (
    st.session_state.get(
        "matchday_signature"
    )
)


if (
    stored_signature
    == current_signature
    and
    "matchday_samples"
    in st.session_state
):

    sampled_results = (
        st.session_state[
            "matchday_samples"
        ]
    )


    final_rows = []


    for (
        (_, row),
        sampled
    ) in zip(
        prediction_df.iterrows(),
        sampled_results
    ):

        if sampled == "H":

            sampled_text = (
                f"{row['Home']} 勝ち"
            )

        elif sampled == "D":

            sampled_text = (
                "引き分け"
            )

        else:

            sampled_text = (
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
                        f"{row['Home']}"
                        f" vs "
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
                        "本命"
                    ],

                "今回の予想":
                    sampled,

                "予想内容":
                    sampled_text
            }
        )


    final_df = pd.DataFrame(
        final_rows
    )


    st.subheader(
        "🎯 今回生成された1節"
    )


    st.dataframe(
        final_df,
        hide_index=True,
        use_container_width=True
    )


    # =====================================================
    # H/D/A個数
    # =====================================================

    home_count = (
        sampled_results.count(
            "H"
        )
    )

    draw_count = (
        sampled_results.count(
            "D"
        )
    )

    away_count = (
        sampled_results.count(
            "A"
        )
    )


    count1, count2, count3 = (
        st.columns(3)
    )


    count1.metric(
        "🏠 Home予想",
        home_count
    )


    count2.metric(
        "🤝 Draw予想",
        draw_count
    )


    count3.metric(
        "✈️ Away予想",
        away_count
    )


    st.caption(
        "もう一度「全試合を一括抽選」を押すと、"
        "同じ確率から別の予想セットを生成します。"
    )


# =========================================================
# 予測の読み方
# =========================================================

with st.expander(
    "ℹ️ この予想の読み方"
):

    st.write(
        """
**H** = Home勝ち  
**D** = 引き分け  
**A** = Away勝ち

「本命」は3つの中で最も確率が高い結果です。

「今回の予想」は本命をそのまま採用するのではなく、
G5.1が計算したH/D/A確率に従ってランダム抽選しています。

例えば、

H = 45%  
D = 28%  
A = 27%

なら、Homeが本命ですが、
毎回必ずHomeになるわけではありません。

これによりモデルが持っている不確実性を残したまま、
1つの予想セットを生成できます。
"""
    )


# =========================================================
# 注意
# =========================================================

st.warning(
    "現在はJPN.csvに収録されている最新の終了済み試合までを"
    "使っています。次の段階では、実際の次節カードを"
    "自動で取得できる仕組みを追加します。"
)
