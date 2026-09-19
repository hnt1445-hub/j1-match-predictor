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
    "時系列順にバックテストします。"
)


# =========================================================
# データ読み込み
# =========================================================

df = pd.read_csv("data/JPN.csv")

j1 = df[
    df["League"] == "J1 League"
].copy()

j1["Season"] = j1["Season"].astype(str)

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
).copy()

j1 = j1.sort_values(
    ["Date"]
).reset_index(drop=True)


# =========================================================
# 基本情報
# =========================================================

st.header("📚 使用データ")

col1, col2, col3 = st.columns(3)

col1.metric(
    "J1総試合数",
    len(j1)
)

col2.metric(
    "シーズン数",
    j1["Season"].nunique()
)

col3.metric(
    "2025年試合数",
    (
        j1["Season"] == "2025"
    ).sum()
)


# =========================================================
# 特徴量作成
#
# 全シーズンを古い順に処理する
#
# 重要：
# 特徴量を記録してから試合結果を反映する
# =========================================================

elo = {}

season_points = {}
season_recent = {}

long_goals_for = {}
long_goals_against = {}
long_matches = {}

season_goals_for = {}
season_goals_against = {}
season_matches = {}

feature_rows = []

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


    # -----------------------------------------------------
    # 新しいシーズンになったら
    # シーズン内の数字だけリセット
    #
    # Eloと長期攻撃・守備情報は引き継ぐ
    # -----------------------------------------------------

    if season != current_season:

        season_points = {}
        season_recent = {}

        season_goals_for = {}
        season_goals_against = {}
        season_matches = {}

        current_season = season


    # -----------------------------------------------------
    # 試合前：シーズン勝点
    # -----------------------------------------------------

    home_points = season_points.get(
        home,
        0
    )

    away_points = season_points.get(
        away,
        0
    )

    points_diff = (
        home_points -
        away_points
    )


    # -----------------------------------------------------
    # 試合前：直近5試合
    # -----------------------------------------------------

    home_recent = season_recent.get(
        home,
        []
    )

    away_recent = season_recent.get(
        away,
        []
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


    # -----------------------------------------------------
    # 試合前：シーズン得失点差
    # -----------------------------------------------------

    home_season_gf = (
        season_goals_for.get(
            home,
            0
        )
    )

    home_season_ga = (
        season_goals_against.get(
            home,
            0
        )
    )

    away_season_gf = (
        season_goals_for.get(
            away,
            0
        )
    )

    away_season_ga = (
        season_goals_against.get(
            away,
            0
        )
    )

    home_goal_diff = (
        home_season_gf -
        home_season_ga
    )

    away_goal_diff = (
        away_season_gf -
        away_season_ga
    )

    goal_diff_diff = (
        home_goal_diff -
        away_goal_diff
    )


    # -----------------------------------------------------
    # 試合前：Elo
    #
    # シーズンをまたいで引き継ぐ
    # -----------------------------------------------------

    home_elo = elo.get(
        home,
        1500.0
    )

    away_elo = elo.get(
        away,
        1500.0
    )

    elo_diff = (
        home_elo -
        away_elo
    )


    # -----------------------------------------------------
    # 試合前：長期攻撃・守備
    #
    # 過去シーズンも含めて蓄積
    # -----------------------------------------------------

    home_long_matches = (
        long_matches.get(
            home,
            0
        )
    )

    away_long_matches = (
        long_matches.get(
            away,
            0
        )
    )


    if home_long_matches > 0:

        home_attack = (
            long_goals_for.get(
                home,
                0
            )
            /
            home_long_matches
        )

        home_concede = (
            long_goals_against.get(
                home,
                0
            )
            /
            home_long_matches
        )

    else:

        home_attack = 1.0
        home_concede = 1.0


    if away_long_matches > 0:

        away_attack = (
            long_goals_for.get(
                away,
                0
            )
            /
            away_long_matches
        )

        away_concede = (
            long_goals_against.get(
                away,
                0
            )
            /
            away_long_matches
        )

    else:

        away_attack = 1.0
        away_concede = 1.0


    # -----------------------------------------------------
    # 攻撃・守備の組み合わせ
    #
    # ホームの攻撃力が高い
    # ＋
    # アウェイの平均失点が多い
    #
    # → ホーム側に有利
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # 実際の結果
    # -----------------------------------------------------

    if hg > ag:

        result = "H"

    elif hg < ag:

        result = "A"

    else:

        result = "D"


    # -----------------------------------------------------
    # 試合前情報を保存
    # -----------------------------------------------------

    feature_rows.append({

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
            attack_defense_diff,

        "HomeElo":
            home_elo,

        "AwayElo":
            away_elo,

        "HomeAttack":
            home_attack,

        "AwayAttack":
            away_attack,

        "HomeConcede":
            home_concede,

        "AwayConcede":
            away_concede
    })


    # =====================================================
    # ここから下は試合終了後
    #
    # ここで初めて今回の結果を反映する
    # =====================================================


    # -----------------------------------------------------
    # 勝点
    # -----------------------------------------------------

    if result == "H":

        home_match_points = 3
        away_match_points = 0

        home_actual = 1.0
        away_actual = 0.0

    elif result == "A":

        home_match_points = 0
        away_match_points = 3

        home_actual = 0.0
        away_actual = 1.0

    else:

        home_match_points = 1
        away_match_points = 1

        home_actual = 0.5
        away_actual = 0.5


    season_points[home] = (
        home_points +
        home_match_points
    )

    season_points[away] = (
        away_points +
        away_match_points
    )


    # -----------------------------------------------------
    # 直近5試合
    # -----------------------------------------------------

    season_recent.setdefault(
        home,
        []
    ).append(
        home_match_points
    )

    season_recent.setdefault(
        away,
        []
    ).append(
        away_match_points
    )


    # -----------------------------------------------------
    # シーズン得失点
    # -----------------------------------------------------

    season_goals_for[home] = (
        home_season_gf +
        hg
    )

    season_goals_against[home] = (
        home_season_ga +
        ag
    )

    season_goals_for[away] = (
        away_season_gf +
        ag
    )

    season_goals_against[away] = (
        away_season_ga +
        hg
    )

    season_matches[home] = (
        season_matches.get(
            home,
            0
        ) + 1
    )

    season_matches[away] = (
        season_matches.get(
            away,
            0
        ) + 1
    )


    # -----------------------------------------------------
    # 長期攻撃・守備
    # -----------------------------------------------------

    long_goals_for[home] = (
        long_goals_for.get(
            home,
            0
        ) +
        hg
    )

    long_goals_against[home] = (
        long_goals_against.get(
            home,
            0
        ) +
        ag
    )

    long_goals_for[away] = (
        long_goals_for.get(
            away,
            0
        ) +
        ag
    )

    long_goals_against[away] = (
        long_goals_against.get(
            away,
            0
        ) +
        hg
    )

    long_matches[home] = (
        home_long_matches +
        1
    )

    long_matches[away] = (
        away_long_matches +
        1
    )


    # -----------------------------------------------------
    # Elo更新
    # -----------------------------------------------------

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


# =========================================================
# 特徴量DataFrame
# =========================================================

data = pd.DataFrame(
    feature_rows
)

data = data.sort_values(
    "Date"
).reset_index(drop=True)


# =========================================================
# M8で使用する特徴量
# =========================================================

FEATURES = [

    "PointsDiff",

    "FormDiff",

    "GoalDiffDiff",

    "EloDiff",

    "AttackDefenseDiff"
]


# =========================================================
# 評価関数
# =========================================================

def evaluate_predictions(
    result_df
):

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


    model_log_loss = (
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


    return {

        "Accuracy":
            accuracy,

        "LogLoss":
            model_log_loss,

        "Brier":
            brier,

        "DrawRecall":
            draw_recall,

        "Correct":
            int(
                (
                    result_df["Prediction"]
                    ==
                    result_df["Result"]
                ).sum()
            )
    }


# =========================================================
# ウォークフォワード関数
#
# target_seasonを1試合ずつ予測
#
# window_size
# 500 / 1000 / 2000 / None(全部)
# =========================================================

def run_walk_forward(
    full_data,
    target_season,
    window_size=None
):

    target_indexes = (
        full_data.index[
            full_data["Season"]
            ==
            target_season
        ].tolist()
    )


    predictions = []


    for test_index in target_indexes:

        # ---------------------------------------------
        # この試合より過去だけ
        # ---------------------------------------------

        train_data = (
            full_data.iloc[
                :test_index
            ].copy()
        )


        # ---------------------------------------------
        # 直近N試合だけにする
        # ---------------------------------------------

        if (
            window_size
            is not None
        ):

            train_data = (
                train_data.tail(
                    window_size
                )
            )


        # ---------------------------------------------
        # 学習
        # ---------------------------------------------

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


        X_test = (
            full_data
            .loc[
                [test_index],
                FEATURES
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
                    max_iter=2000,
                    random_state=42
                )
            )
        ])


        model.fit(
            X_train,
            y_train
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


        probability_map = dict(
            zip(
                classes,
                probabilities
            )
        )


        prob_h = (
            probability_map.get(
                "H",
                0.0
            )
        )

        prob_d = (
            probability_map.get(
                "D",
                0.0
            )
        )

        prob_a = (
            probability_map.get(
                "A",
                0.0
            )
        )


        probability_dict = {

            "H":
                prob_h,

            "D":
                prob_d,

            "A":
                prob_a
        }


        prediction = max(
            probability_dict,
            key=probability_dict.get
        )


        row = (
            full_data
            .loc[test_index]
        )


        predictions.append({

            "Index":
                test_index,

            "Season":
                row["Season"],

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
# 2024年
#
# モデル選択用
# =========================================================

st.header(
    "🧪 2024年：学習期間の比較"
)

st.write(
    "2024年を仮想本番として、"
    "500・1000・2000・全過去データを比較します。"
)


window_settings = {

    "500試合":
        500,

    "1000試合":
        1000,

    "2000試合":
        2000,

    "ALL":
        None
}


results_2024 = {}

summary_2024 = []


with st.spinner(
    "2024年をバックテスト中..."
):

    for name, window in (
        window_settings.items()
    ):

        result = run_walk_forward(
            data,
            "2024",
            window
        )

        results_2024[name] = (
            result
        )

        metrics = (
            evaluate_predictions(
                result
            )
        )


        summary_2024.append({

            "学習範囲":
                name,

            "試合数":
                len(result),

            "正解数":
                metrics[
                    "Correct"
                ],

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


summary_2024_df = pd.DataFrame(
    summary_2024
)

st.dataframe(
    summary_2024_df,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 2024年でLog Lossが最小の方式
# =========================================================

best_row = (
    summary_2024_df
    .sort_values(
        "Log Loss"
    )
    .iloc[0]
)

best_name = (
    best_row[
        "学習範囲"
    ]
)

best_window = (
    window_settings[
        best_name
    ]
)


st.success(
    "2024年のLog Lossを基準に選んだ学習範囲："
    f"{best_name}"
)


# =========================================================
# 2025年
#
# 最終バックテスト
#
# ここでは2024年で選んだ方式を使用
# =========================================================

st.header(
    "🏆 2025年：最終バックテスト"
)

st.write(
    "2024年で選んだ学習範囲を固定して、"
    "2025年全380試合を予測します。"
)


with st.spinner(
    "2025年380試合を予測中..."
):

    final_2025 = (
        run_walk_forward(
            data,
            "2025",
            best_window
        )
    )


metrics_2025 = (
    evaluate_predictions(
        final_2025
    )
)


# =========================================================
# 2025年結果
# =========================================================

col1, col2, col3, col4 = (
    st.columns(4)
)


col1.metric(
    "正解率",
    f"{metrics_2025['Accuracy']:.1%}"
)

col2.metric(
    "Log Loss",
    f"{metrics_2025['LogLoss']:.4f}"
)

col3.metric(
    "Brier Score",
    f"{metrics_2025['Brier']:.4f}"
)

col4.metric(
    "引き分けRecall",
    f"{metrics_2025['DrawRecall']:.1%}"
)


st.write(
    "正解数:",
    metrics_2025[
        "Correct"
    ],
    "/",
    len(final_2025)
)


# =========================================================
# 2025年 H/D/A 内訳
# =========================================================

st.subheader(
    "📊 2025年 H / D / A 内訳"
)


result_breakdown = pd.DataFrame({

    "結果": [
        "H",
        "D",
        "A"
    ],

    "実際": [

        int(
            (
                final_2025[
                    "Result"
                ]
                ==
                "H"
            ).sum()
        ),

        int(
            (
                final_2025[
                    "Result"
                ]
                ==
                "D"
            ).sum()
        ),

        int(
            (
                final_2025[
                    "Result"
                ]
                ==
                "A"
            ).sum()
        )
    ],

    "予測": [

        int(
            (
                final_2025[
                    "Prediction"
                ]
                ==
                "H"
            ).sum()
        ),

        int(
            (
                final_2025[
                    "Prediction"
                ]
                ==
                "D"
            ).sum()
        ),

        int(
            (
                final_2025[
                    "Prediction"
                ]
                ==
                "A"
            ).sum()
        )
    ]
})


st.dataframe(
    result_breakdown,
    hide_index=True
)


# =========================================================
# 最近20試合
# =========================================================

st.subheader(
    "🔍 2025年 最後の20試合"
)


recent = (
    final_2025
    .tail(20)
    .copy()
)


recent[
    "Home勝率%"
] = (
    recent[
        "Prob_H"
    ]
    * 100
).round(1)

recent[
    "引分率%"
] = (
    recent[
        "Prob_D"
    ]
    * 100
).round(1)

recent[
    "Away勝率%"
] = (
    recent[
        "Prob_A"
    ]
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
# 参考：
# 2025年でも4方式を比較
#
# ただしモデル選択には使用しない
# =========================================================

st.header(
    "🔬 参考：2025年4方式比較"
)

st.caption(
    "ここは分析用です。"
    "2025年の結果を見て学習範囲を選び直すためには使いません。"
)


summary_2025_all = []


with st.spinner(
    "2025年の4方式を比較中..."
):

    for name, window in (
        window_settings.items()
    ):

        result = run_walk_forward(
            data,
            "2025",
            window
        )

        metrics = (
            evaluate_predictions(
                result
            )
        )


        summary_2025_all.append({

            "学習範囲":
                name,

            "試合数":
                len(result),

            "正解数":
                metrics[
                    "Correct"
                ],

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


summary_2025_all_df = (
    pd.DataFrame(
        summary_2025_all
    )
)


st.dataframe(
    summary_2025_all_df,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# 注意書き
# =========================================================

st.info(
    "M8では、予測対象の試合より後の結果を"
    "学習データに使用していません。"
    "2025年開幕戦も、2024年以前のデータだけで予測します。"
)
