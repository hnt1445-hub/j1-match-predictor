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
    page_title="J1 Future Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ J1 Future Match Predictor")

st.write(
    "G5.1を使って、まだ結果の出ていない試合を予測します。"
)

LEARNING_RATE = 0.06
K_FACTOR = 20

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

    df = pd.read_csv("data/JPN.csv")

    df = df[
        df["League"] == "J1 League"
    ].copy()

    df["Season"] = df["Season"].astype(str)

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

    df["OriginalOrder"] = range(len(df))

    df = (
        df
        .sort_values(
            ["Date", "OriginalOrder"]
        )
        .reset_index(drop=True)
    )

    return df


matches = load_data()


# =========================================================
# 履歴平均
# =========================================================

def average(values):

    if len(values) == 0:
        return np.nan

    return sum(values) / len(values)


# =========================================================
# 過去試合から
# 「試合前特徴量」と「最新チーム状態」を同時に作る
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

        hg = float(match["HG"])
        ag = float(match["AG"])

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

        # ---------------------------------------------
        # 試合前情報を保存
        # ---------------------------------------------

        rows.append(
            {
                "Season":
                    str(match["Season"]),

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
                    home_elo - away_elo,

                "Home_Form5":
                    home_form,

                "Away_Form5":
                    away_form,

                "FormDiff":
                    home_form - away_form,

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
                    home_attack - away_attack,

                "DefenseAdvantage":
                    away_defense - home_defense
            }
        )

        # ---------------------------------------------
        # 試合後更新
        # ---------------------------------------------

        if hg > ag:

            hp = 3
            ap = 0
            home_actual = 1.0

        elif hg < ag:

            hp = 0
            ap = 3
            home_actual = 0.0

        else:

            hp = 1
            ap = 1
            home_actual = 0.5

        recent_points_5[home].append(hp)
        recent_points_5[away].append(ap)

        recent_gf_10[home].append(hg)
        recent_ga_10[home].append(ag)

        recent_gf_10[away].append(ag)
        recent_ga_10[away].append(hg)

        home_gf_10[home].append(hg)
        home_ga_10[home].append(ag)

        away_gf_10[away].append(ag)
        away_ga_10[away].append(hg)

        # ---------------------------------------------
        # Attack / Defense更新
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
            (expected_home_goals + 0.5)
        )

        away_ratio = (
            (ag + 0.5)
            /
            (expected_away_goals + 0.5)
        )

        attack[home] = np.clip(
            home_attack
            * (
                home_ratio
                ** LEARNING_RATE
            ),
            MIN_RATING,
            MAX_RATING
        )

        attack[away] = np.clip(
            away_attack
            * (
                away_ratio
                ** LEARNING_RATE
            ),
            MIN_RATING,
            MAX_RATING
        )

        defense[home] = np.clip(
            home_defense
            * (
                away_ratio
                ** LEARNING_RATE
            ),
            MIN_RATING,
            MAX_RATING
        )

        defense[away] = np.clip(
            away_defense
            * (
                home_ratio
                ** LEARNING_RATE
            ),
            MIN_RATING,
            MAX_RATING
        )

        # ---------------------------------------------
        # Elo更新
        # ---------------------------------------------

        expected_home = (
            1
            /
            (
                1
                +
                10
                ** (
                    (
                        away_elo
                        - home_elo
                    )
                    / 400
                )
            )
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
                (1.0 - home_actual)
                -
                (1.0 - expected_home)
            )
        )

    history = pd.DataFrame(rows)

    # ---------------------------------------------
    # 最終状態を普通のdictへ変換
    # ---------------------------------------------

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
            }
    }

    return history, state


history, current_state = (
    build_history(matches)
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
        history[ALL_FEATURES],
        history["HG"]
    )

    away_model.fit(
        history[ALL_FEATURES],
        history["AG"]
    )

    return home_model, away_model


home_model, away_model = (
    train_models(history)
)


# =========================================================
# Future特徴量
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

    return np.mean(values)


def make_future_match(
    home,
    away,
    state
):

    home_elo = (
        state["elo"]
        .get(home, 1500.0)
    )

    away_elo = (
        state["elo"]
        .get(away, 1500.0)
    )

    home_attack = (
        state["attack"]
        .get(home, 1.0)
    )

    away_attack = (
        state["attack"]
        .get(away, 1.0)
    )

    home_defense = (
        state["defense"]
        .get(home, 1.0)
    )

    away_defense = (
        state["defense"]
        .get(away, 1.0)
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
            home_elo - away_elo,

        "Home_Form5":
            home_form,

        "Away_Form5":
            away_form,

        "FormDiff":
            home_form - away_form,

        "Home_Recent10_GF":
            state_average(
                state["recent_gf_10"],
                home
            ),

        "Home_Recent10_GA":
            state_average(
                state["recent_ga_10"],
                home
            ),

        "Away_Recent10_GF":
            state_average(
                state["recent_gf_10"],
                away
            ),

        "Away_Recent10_GA":
            state_average(
                state["recent_ga_10"],
                away
            ),

        "Home_Home10_GF":
            state_average(
                state["home_gf_10"],
                home
            ),

        "Home_Home10_GA":
            state_average(
                state["home_ga_10"],
                home
            ),

        "Away_Away10_GF":
            state_average(
                state["away_gf_10"],
                away
            ),

        "Away_Away10_GA":
            state_average(
                state["away_ga_10"],
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
            home_attack - away_attack,

        "DefenseAdvantage":
            away_defense - home_defense
    }

    return pd.DataFrame([row])


# =========================================================
# Poisson → H/D/A
# =========================================================

def poisson_probability(
    goals,
    expected_goals
):

    return (
        exp(-expected_goals)
        *
        expected_goals ** goals
        /
        factorial(goals)
    )


def calculate_probabilities(
    home_lambda,
    away_lambda
):

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    score_rows = []

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

            probability = ph * pa

            score_rows.append(
                {
                    "Score":
                        f"{hg} - {ag}",

                    "Probability":
                        probability
                }
            )

            if hg > ag:
                home_win += probability

            elif hg == ag:
                draw += probability

            else:
                away_win += probability

    total = (
        home_win
        + draw
        + away_win
    )

    probabilities = np.array(
        [
            home_win / total,
            draw / total,
            away_win / total
        ]
    )

    return probabilities, score_rows


# =========================================================
# データ状態
# =========================================================

st.header("📅 使用しているデータ")

latest_date = matches["Date"].max()

st.write(
    "最新の収録試合日：",
    latest_date.strftime("%Y-%m-%d")
)

st.write(
    "J1収録試合数：",
    len(matches)
)

st.caption(
    "この日までの終了済み試合を使って、"
    "その次に行われる試合を予測する状態です。"
)


# =========================================================
# チーム候補
# =========================================================

# 最新シーズンに登場したクラブを優先
latest_season = (
    matches["Season"]
    .astype(int)
    .max()
)

latest_season_teams = sorted(
    set(
        matches.loc[
            matches["Season"].astype(int)
            == latest_season,
            "Home"
        ]
    )
    |
    set(
        matches.loc[
            matches["Season"].astype(int)
            == latest_season,
            "Away"
        ]
    )
)


# =========================================================
# 未来試合
# =========================================================

st.header("🔮 未来試合を予測")


col1, col2 = st.columns(2)


with col1:

    home_team = st.selectbox(
        "ホームチーム",
        latest_season_teams,
        index=0
    )


away_candidates = [
    team
    for team
    in latest_season_teams
    if team != home_team
]


with col2:

    away_team = st.selectbox(
        "アウェイチーム",
        away_candidates,
        index=0
    )


future_match = make_future_match(
    home_team,
    away_team,
    current_state
)


# =========================================================
# 予測
# =========================================================

pred_home_goals = float(
    home_model.predict(
        future_match[
            ALL_FEATURES
        ]
    )[0]
)

pred_away_goals = float(
    away_model.predict(
        future_match[
            ALL_FEATURES
        ]
    )[0]
)


pred_home_goals = float(
    np.clip(
        pred_home_goals,
        0.05,
        5.0
    )
)

pred_away_goals = float(
    np.clip(
        pred_away_goals,
        0.05,
        5.0
    )
)


probabilities, score_rows = (
    calculate_probabilities(
        pred_home_goals,
        pred_away_goals
    )
)


home_probability = probabilities[0]
draw_probability = probabilities[1]
away_probability = probabilities[2]


# =========================================================
# 表示
# =========================================================

st.subheader(
    f"{home_team} vs {away_team}"
)


g1, g2 = st.columns(2)


g1.metric(
    f"{home_team} 予想得点",
    f"{pred_home_goals:.2f}"
)


g2.metric(
    f"{away_team} 予想得点",
    f"{pred_away_goals:.2f}"
)


st.subheader("🎯 H / D / A 確率")


p1, p2, p3 = st.columns(3)


p1.metric(
    "🏠 Home",
    f"{home_probability:.1%}"
)


p2.metric(
    "🤝 Draw",
    f"{draw_probability:.1%}"
)


p3.metric(
    "✈️ Away",
    f"{away_probability:.1%}"
)


# =========================================================
# モデルが最も高く評価した結果
# =========================================================

labels = np.array(
    ["H", "D", "A"]
)

top_result = labels[
    np.argmax(probabilities)
]


if top_result == "H":

    top_text = (
        f"{home_team} 勝ち"
    )

elif top_result == "D":

    top_text = "引き分け"

else:

    top_text = (
        f"{away_team} 勝ち"
    )


st.info(
    "📊 最も確率が高い結果："
    + top_text
)


# =========================================================
# ランダム抽選
# =========================================================

st.subheader("🎲 確率に従って1回予想")


match_key = (
    "future_sample_"
    + home_team
    + "_"
    + away_team
)


if st.button(
    "🎲 予想を抽選"
):

    sampled = np.random.choice(
        ["H", "D", "A"],
        p=probabilities
    )

    st.session_state[
        match_key
    ] = sampled


if match_key in st.session_state:

    sampled = st.session_state[
        match_key
    ]

    if sampled == "H":

        sampled_text = (
            f"🏠 {home_team} 勝ち"
        )

    elif sampled == "D":

        sampled_text = (
            "🤝 引き分け"
        )

    else:

        sampled_text = (
            f"✈️ {away_team} 勝ち"
        )

    st.success(
        "今回の予想："
        + sampled_text
    )


# =========================================================
# スコア確率
# =========================================================

st.subheader("⚽ スコア確率 TOP10")


score_table = pd.DataFrame(
    score_rows
)


score_table = (
    score_table
    .sort_values(
        "Probability",
        ascending=False
    )
    .head(10)
    .copy()
)


score_table["確率"] = (
    score_table["Probability"]
    * 100
).round(1)


st.dataframe(
    score_table[
        [
            "Score",
            "確率"
        ]
    ],
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 現在のチーム状態
# =========================================================

st.subheader("💪 現在のチーム状態")


state_table = pd.DataFrame(
    {
        "項目": [
            "Elo",
            "Attack Rating",
            "Defense Rating",
            "直近5試合 勝点"
        ],

        home_team: [
            current_state[
                "elo"
            ].get(
                home_team,
                1500
            ),

            current_state[
                "attack"
            ].get(
                home_team,
                1.0
            ),

            current_state[
                "defense"
            ].get(
                home_team,
                1.0
            ),

            sum(
                current_state[
                    "recent_points_5"
                ].get(
                    home_team,
                    []
                )
            )
        ],

        away_team: [
            current_state[
                "elo"
            ].get(
                away_team,
                1500
            ),

            current_state[
                "attack"
            ].get(
                away_team,
                1.0
            ),

            current_state[
                "defense"
            ].get(
                away_team,
                1.0
            ),

            sum(
                current_state[
                    "recent_points_5"
                ].get(
                    away_team,
                    []
                )
            )
        ]
    }
)


state_table[
    home_team
] = (
    state_table[
        home_team
    ].round(3)
)


state_table[
    away_team
] = (
    state_table[
        away_team
    ].round(3)
)


st.dataframe(
    state_table,
    hide_index=True,
    use_container_width=True
)


st.caption(
    "Defense Ratingは値が小さいほど守備が強い設計です。"
)


st.warning(
    "現在はJPN.csvに収録されている最新試合までの状態を使用しています。"
    "実運用では、新しい試合結果を追加するたびにチーム状態も更新します。"
)
