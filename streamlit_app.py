import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# =========================================================
# 基本設定
# =========================================================

st.set_page_config(
    page_title="J1 Match Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ J1 Match Predictor")
st.write(
    "過去シーズンを学習し、2025年J1全380試合を"
    "未来情報を使わずバックテストします。"
)


# =========================================================
# データ読み込み
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("data/JPN.csv")

    j1 = df[
        df["League"] == "J1 League"
    ].copy()

    j1["Season"] = (
        j1["Season"]
        .astype(str)
    )

    j1["Date"] = pd.to_datetime(
        j1["Date"],
        dayfirst=True,
        errors="coerce"
    )

    j1 = j1.dropna(
        subset=[
            "Date",
            "Home",
            "Away",
            "HG",
            "AG"
        ]
    )

    j1 = (
        j1
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return j1


j1 = load_data()


# =========================================================
# 特徴量作成
# =========================================================

@st.cache_data
def build_features(j1):

    elo = {}

    season_points = {}
    season_recent = {}

    season_goals_for = {}
    season_goals_against = {}

    long_goals_for = {}
    long_goals_against = {}
    long_matches = {}

    rows = []

    current_season = None

    K_FACTOR = 20


    for _, match in j1.iterrows():

        season = str(
            match["Season"]
        )

        home = match["Home"]
        away = match["Away"]

        hg = float(
            match["HG"]
        )

        ag = float(
            match["AG"]
        )


        # -------------------------------------------------
        # 新シーズン
        # -------------------------------------------------

        if season != current_season:

            season_points = {}
            season_recent = {}

            season_goals_for = {}
            season_goals_against = {}

            current_season = season


        # -------------------------------------------------
        # 勝点
        # -------------------------------------------------

        home_points = (
            season_points.get(
                home,
                0
            )
        )

        away_points = (
            season_points.get(
                away,
                0
            )
        )

        points_diff = (
            home_points -
            away_points
        )


        # -------------------------------------------------
        # 直近5試合
        # -------------------------------------------------

        home_recent = (
            season_recent.get(
                home,
                []
            )
        )

        away_recent = (
            season_recent.get(
                away,
                []
            )
        )

        home_form = sum(
            home_recent[-5:]
        )

        away_form = sum(
            away_recent[-5:]
        )

        form_diff = (
            home_form -
            away_form
        )


        # -------------------------------------------------
        # シーズン得失点差
        # -------------------------------------------------

        home_gf = (
            season_goals_for.get(
                home,
                0
            )
        )

        home_ga = (
            season_goals_against.get(
                home,
                0
            )
        )

        away_gf = (
            season_goals_for.get(
                away,
                0
            )
        )

        away_ga = (
            season_goals_against.get(
                away,
                0
            )
        )

        goal_diff_diff = (
            (home_gf - home_ga)
            -
            (away_gf - away_ga)
        )


        # -------------------------------------------------
        # Elo
        # -------------------------------------------------

        home_elo = (
            elo.get(
                home,
                1500.0
            )
        )

        away_elo = (
            elo.get(
                away,
                1500.0
            )
        )

        elo_diff = (
            home_elo -
            away_elo
        )


        # -------------------------------------------------
        # 長期攻撃・守備
        # -------------------------------------------------

        home_matches = (
            long_matches.get(
                home,
                0
            )
        )

        away_matches = (
            long_matches.get(
                away,
                0
            )
        )


        if home_matches > 0:

            home_attack = (
                long_goals_for.get(
                    home,
                    0
                )
                /
                home_matches
            )

            home_concede = (
                long_goals_against.get(
                    home,
                    0
                )
                /
                home_matches
            )

        else:

            home_attack = 1.0
            home_concede = 1.0


        if away_matches > 0:

            away_attack = (
                long_goals_for.get(
                    away,
                    0
                )
                /
                away_matches
            )

            away_concede = (
                long_goals_against.get(
                    away,
                    0
                )
                /
                away_matches
            )

        else:

            away_attack = 1.0
            away_concede = 1.0


        # 相手が失点しやすいほど得点しやすい方向
        home_scoring_signal = (
            home_attack +
            away_concede
        )

        away_scoring_signal = (
            away_attack +
            home_concede
        )

        attack_defense_diff = (
            home_scoring_signal -
            away_scoring_signal
        )


        # -------------------------------------------------
        # 結果
        # -------------------------------------------------

        if hg > ag:

            result = "H"

        elif hg < ag:

            result = "A"

        else:

            result = "D"


        # -------------------------------------------------
        # 試合前データ保存
        # -------------------------------------------------

        rows.append({

            "Season":
                season,

            "Date":
                match["Date"],

            "Home":
                home,

            "Away":
                away,

            "Result":
                result,

            "PointsDiff":
                points_diff,

            "FormDiff":
                form_diff,

            "GoalDiffDiff":
                goal_diff_diff,

            "EloDiff":
                elo_diff,

            "AttackDefenseDiff":
                attack_defense_diff
        })


        # =================================================
        # ここから試合後更新
        # =================================================

        if result == "H":

            hp = 3
            ap = 0

            home_actual = 1.0
            away_actual = 0.0

        elif result == "A":

            hp = 0
            ap = 3

            home_actual = 0.0
            away_actual = 1.0

        else:

            hp = 1
            ap = 1

            home_actual = 0.5
            away_actual = 0.5


        # 勝点
        season_points[home] = (
            home_points + hp
        )

        season_points[away] = (
            away_points + ap
        )


        # 直近成績
        season_recent.setdefault(
            home,
            []
        ).append(hp)

        season_recent.setdefault(
            away,
            []
        ).append(ap)


        # シーズン得失点
        season_goals_for[home] = (
            home_gf + hg
        )

        season_goals_against[home] = (
            home_ga + ag
        )

        season_goals_for[away] = (
            away_gf + ag
        )

        season_goals_against[away] = (
            away_ga + hg
        )


        # 長期得失点
        long_goals_for[home] = (
            long_goals_for.get(
                home,
                0
            )
            + hg
        )

        long_goals_against[home] = (
            long_goals_against.get(
                home,
                0
            )
            + ag
        )

        long_goals_for[away] = (
            long_goals_for.get(
                away,
                0
            )
            + ag
        )

        long_goals_against[away] = (
            long_goals_against.get(
                away,
                0
            )
            + hg
        )

        long_matches[home] = (
            home_matches + 1
        )

        long_matches[away] = (
            away_matches + 1
        )


        # -------------------------------------------------
        # Elo更新
        # -------------------------------------------------

        home_expected = (
            1
            /
            (
                1
                +
                10 **
                (
                    (
                        away_elo -
                        home_elo
                    )
                    /
                    400
                )
            )
        )

        away_expected = (
            1 -
            home_expected
        )

        elo[home] = (
            home_elo
            +
            K_FACTOR
            *
            (
                home_actual -
                home_expected
            )
        )

        elo[away] = (
            away_elo
            +
            K_FACTOR
            *
            (
                away_actual -
                away_expected
            )
        )


    return pd.DataFrame(rows)


data = build_features(j1)


# =========================================================
# 特徴量
# =========================================================

FEATURES = [

    "PointsDiff",

    "FormDiff",

    "GoalDiffDiff",

    "EloDiff",

    "AttackDefenseDiff"
]


# =========================================================
# 軽量ウォークフォワード
#
# 20試合ごとに再学習
# =========================================================

@st.cache_data
def run_light_walk_forward(
    full_data,
    target_season,
    window_size,
    retrain_every=20
):

    target_indexes = (
        full_data.index[
            full_data["Season"]
            ==
            target_season
        ].tolist()
    )


    predictions = []

    model = None


    for position, test_index in enumerate(
        target_indexes
    ):

        # -------------------------------------------------
        # 最初の試合 または 20試合ごとに再学習
        # -------------------------------------------------

        if (
            model is None
            or
            position % retrain_every == 0
        ):

            train_data = (
                full_data.iloc[
                    :test_index
                ].copy()
            )


            if window_size is not None:

                train_data = (
                    train_data.tail(
                        window_size
                    )
                )


            X_train = (
                train_data[
                    FEATURES
                ]
            )

            y_train = (
                train_data[
                    "Result"
                ]
            )


            model = Pipeline([

                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42
                    )
                )
            ])


            model.fit(
                X_train,
                y_train
            )


        # -------------------------------------------------
        # 1試合予測
        # -------------------------------------------------

        X_test = (
            full_data.loc[
                [test_index],
                FEATURES
            ]
        )


        probabilities = (
            model.predict_proba(
                X_test
            )[0]
        )


        classes = (
            model
            .named_steps[
                "model"
            ]
            .classes_
        )


        prob_map = dict(
            zip(
                classes,
                probabilities
            )
        )


        prob_h = (
            prob_map.get(
                "H",
                0.0
            )
        )

        prob_d = (
            prob_map.get(
                "D",
                0.0
            )
        )

        prob_a = (
            prob_map.get(
                "A",
                0.0
            )
        )


        prediction = max(

            {
                "H": prob_h,
                "D": prob_d,
                "A": prob_a
            },

            key={
                "H": prob_h,
                "D": prob_d,
                "A": prob_a
            }.get
        )


        row = (
            full_data.loc[
                test_index
            ]
        )


        predictions.append({

            "Date":
                row["Date"],

            "Home":
                row["Home"],

            "Away":
                row["Away"],

            "Result":
                row["Result"],

            "Prediction":
                prediction,

            "Prob_H":
                prob_h,

            "Prob_D":
                prob_d,

            "Prob_A":
                prob_a
        })


    return pd.DataFrame(
        predictions
    )


# =========================================================
# 評価
# =========================================================

def evaluate(result_df):

    accuracy = (
        result_df["Prediction"]
        ==
        result_df["Result"]
    ).mean()


    actual_probability = np.where(

        result_df["Result"] == "H",

        result_df["Prob_H"],

        np.where(

            result_df["Result"] == "D",

            result_df["Prob_D"],

            result_df["Prob_A"]
        )
    )


    actual_probability = np.clip(
        actual_probability,
        1e-15,
        1.0
    )


    logloss = (
        -np.mean(
            np.log(
                actual_probability
            )
        )
    )


    actual_h = (
        result_df["Result"] == "H"
    ).astype(int)

    actual_d = (
        result_df["Result"] == "D"
    ).astype(int)

    actual_a = (
        result_df["Result"] == "A"
    ).astype(int)


    brier = np.mean(

        (
            result_df["Prob_H"]
            -
            actual_h
        ) ** 2

        +

        (
            result_df["Prob_D"]
            -
            actual_d
        ) ** 2

        +

        (
            result_df["Prob_A"]
            -
            actual_a
        ) ** 2
    )


    actual_draws = (
        result_df["Result"]
        ==
        "D"
    ).sum()


    correct_draws = (

        (
            result_df["Result"]
            ==
            "D"
        )

        &

        (
            result_df["Prediction"]
            ==
            "D"
        )

    ).sum()


    if actual_draws > 0:

        draw_recall = (
            correct_draws
            /
            actual_draws
        )

    else:

        draw_recall = 0.0


    correct = (
        result_df["Prediction"]
        ==
        result_df["Result"]
    ).sum()


    return {

        "Accuracy":
            accuracy,

        "LogLoss":
            logloss,

        "Brier":
            brier,

        "DrawRecall":
            draw_recall,

        "Correct":
            int(correct)
    }


# =========================================================
# データ情報
# =========================================================

st.header(
    "📚 データ"
)


c1, c2, c3 = st.columns(3)

c1.metric(
    "J1総試合数",
    len(data)
)

c2.metric(
    "シーズン数",
    data["Season"].nunique()
)

c3.metric(
    "2025年",
    (
        data["Season"] == "2025"
    ).sum()
)


# =========================================================
# STEP 1
# 2024年で学習範囲を選ぶ
# =========================================================

st.header(
    "🧪 STEP 1：2024年で学習範囲を選択"
)

st.caption(
    "500 / 1000 / 2000 / ALL を比較します。"
)


WINDOWS = {

    "500":
        500,

    "1000":
        1000,

    "2000":
        2000,

    "ALL":
        None
}


summary_rows = []


with st.spinner(
    "2024年を計算中..."
):

    for name, window in WINDOWS.items():

        result = (
            run_light_walk_forward(
                data,
                "2024",
                window,
                20
            )
        )

        metrics = evaluate(
            result
        )


        summary_rows.append({

            "学習範囲":
                name,

            "試合数":
                len(result),

            "正解率":
                round(
                    metrics[
                        "Accuracy"
                    ]
                    * 100,
                    1
                ),

            "Log Loss":
                round(
                    metrics[
                        "LogLoss"
                    ],
                    4
                ),

            "Brier Score":
                round(
                    metrics[
                        "Brier"
                    ],
                    4
                ),

            "引き分けRecall":
                round(
                    metrics[
                        "DrawRecall"
                    ]
                    * 100,
                    1
                )
        })


summary_2024 = pd.DataFrame(
    summary_rows
)


st.dataframe(
    summary_2024,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# Log Lossで選択
# =========================================================

best_row = (
    summary_2024
    .sort_values(
        "Log Loss"
    )
    .iloc[0]
)


best_name = str(
    best_row[
        "学習範囲"
    ]
)


best_window = (
    WINDOWS[
        best_name
    ]
)


st.success(
    "2024年の結果だけで選択："
    f"過去 {best_name} 試合"
)


# =========================================================
# STEP 2
# 2025年全380試合
# =========================================================

st.header(
    "🏆 STEP 2：2025年 全380試合バックテスト"
)

st.write(
    "2024年で選んだ学習範囲を固定して、"
    "2025年を評価します。"
)


with st.spinner(
    "2025年を計算中..."
):

    result_2025 = (
        run_light_walk_forward(
            data,
            "2025",
            best_window,
            20
        )
    )


metrics_2025 = evaluate(
    result_2025
)


# =========================================================
# 結果表示
# =========================================================

c1, c2, c3, c4 = (
    st.columns(4)
)


c1.metric(
    "正解率",
    f"{metrics_2025['Accuracy']:.1%}"
)


c2.metric(
    "Log Loss",
    f"{metrics_2025['LogLoss']:.4f}"
)


c3.metric(
    "Brier Score",
    f"{metrics_2025['Brier']:.4f}"
)


c4.metric(
    "引き分けRecall",
    f"{metrics_2025['DrawRecall']:.1%}"
)


st.write(
    "正解数:",
    metrics_2025[
        "Correct"
    ],
    "/",
    len(result_2025)
)


# =========================================================
# 実際 vs 予測
# =========================================================

st.subheader(
    "📊 H / D / A 内訳"
)


breakdown = pd.DataFrame({

    "結果": [
        "H",
        "D",
        "A"
    ],

    "実際": [

        int(
            (
                result_2025[
                    "Result"
                ] == "H"
            ).sum()
        ),

        int(
            (
                result_2025[
                    "Result"
                ] == "D"
            ).sum()
        ),

        int(
            (
                result_2025[
                    "Result"
                ] == "A"
            ).sum()
        )
    ],

    "予測": [

        int(
            (
                result_2025[
                    "Prediction"
                ] == "H"
            ).sum()
        ),

        int(
            (
                result_2025[
                    "Prediction"
                ] == "D"
            ).sum()
        ),

        int(
            (
                result_2025[
                    "Prediction"
                ] == "A"
            ).sum()
        )
    ]
})


st.dataframe(
    breakdown,
    hide_index=True
)


# =========================================================
# 最近20試合
# =========================================================

st.subheader(
    "🔍 2025年 最後の20試合"
)


recent = (
    result_2025
    .tail(20)
    .copy()
)


recent["Home勝率%"] = (
    recent["Prob_H"]
    * 100
).round(1)

recent["引分率%"] = (
    recent["Prob_D"]
    * 100
).round(1)

recent["Away勝率%"] = (
    recent["Prob_A"]
    * 100
).round(1)


st.dataframe(

    recent[
        [
            "Date",
            "Home",
            "Away",
            "Result",
            "Prediction",
            "Home勝率%",
            "引分率%",
            "Away勝率%"
        ]
    ],

    hide_index=True,
    use_container_width=True
)


# =========================================================
# 説明
# =========================================================

st.info(
    "この軽量版は20試合ごとにモデルを再学習します。"
    "予測対象より未来の試合結果は学習に使用していません。"
    "2025年開幕戦は2024年以前のデータだけで予測します。"
)
