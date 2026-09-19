import streamlit as st
import pandas as pd
import numpy as np


# ==================================================
# 基本設定
# ==================================================

st.set_page_config(
    page_title="J1 Match Predictor",
    page_icon="⚽",
)

st.title("⚽ J1 Match Predictor")
st.write("2025年J1リーグ380試合を使って予測モデルを比較します。")


# ==================================================
# データ読み込み
# ==================================================

df = pd.read_csv("data/JPN.csv")

df_2025 = df[
    (df["Season"] == "2025") &
    (df["League"] == "J1 League")
].copy()

df_2025["Date"] = pd.to_datetime(
    df_2025["Date"],
    dayfirst=True
)

df_2025 = (
    df_2025
    .sort_values("Date")
    .reset_index(drop=True)
)


# ==================================================
# 確率計算の共通部分
# ==================================================

def strength_to_probabilities(
    strength,
    prefix
):

    home_score = np.exp(
        strength / 10
    )

    away_score = np.exp(
        -strength / 10
    )

    draw_score = 1.0

    total = (
        home_score +
        draw_score +
        away_score
    )

    return pd.Series({
        f"{prefix}_Prob_H":
            home_score / total,

        f"{prefix}_Prob_D":
            draw_score / total,

        f"{prefix}_Prob_A":
            away_score / total
    })


# ==================================================
# M1〜M6
# ==================================================

# M1：勝点差
def m1_probabilities(
    points_diff
):

    strength = points_diff

    return strength_to_probabilities(
        strength,
        "M1"
    )


# M2：勝点差＋ホーム補正
def m2_probabilities(
    points_diff
):

    home_advantage = 2.0

    strength = (
        points_diff +
        home_advantage
    )

    return strength_to_probabilities(
        strength,
        "M2"
    )


# M3：M2＋直近5試合
def m3_probabilities(
    points_diff,
    form_diff
):

    home_advantage = 2.0

    # 仮の重み
    form_weight = 0.5

    strength = (
        points_diff +
        home_advantage +
        form_diff * form_weight
    )

    return strength_to_probabilities(
        strength,
        "M3"
    )


# M4：M3＋得失点差
def m4_probabilities(
    points_diff,
    form_diff,
    goal_diff_diff
):

    home_advantage = 2.0

    form_weight = 0.5
    goal_diff_weight = 0.2

    strength = (
        points_diff +
        home_advantage +
        form_diff * form_weight +
        goal_diff_diff * goal_diff_weight
    )

    return strength_to_probabilities(
        strength,
        "M4"
    )


# M5：M4＋Elo
def m5_probabilities(
    points_diff,
    form_diff,
    goal_diff_diff,
    elo_diff
):

    home_advantage = 2.0

    form_weight = 0.5
    goal_diff_weight = 0.2
    elo_weight = 0.02

    strength = (
        points_diff +
        home_advantage +
        form_diff * form_weight +
        goal_diff_diff * goal_diff_weight +
        elo_diff * elo_weight
    )

    return strength_to_probabilities(
        strength,
        "M5"
    )


# M6：M5＋攻撃力・守備力
def m6_probabilities(
    points_diff,
    form_diff,
    goal_diff_diff,
    elo_diff,
    attack_defense_diff
):

    home_advantage = 2.0

    # すべて現段階では仮の重み
    form_weight = 0.5
    goal_diff_weight = 0.2
    elo_weight = 0.02

    # 攻撃力・守備力用の仮の重み
    attack_defense_weight = 2.0

    strength = (
        points_diff +
        home_advantage +
        form_diff * form_weight +
        goal_diff_diff * goal_diff_weight +
        elo_diff * elo_weight +
        attack_defense_diff *
        attack_defense_weight
    )

    return strength_to_probabilities(
        strength,
        "M6"
    )


# ==================================================
# 試合前データ作成
# ==================================================

points = {}

recent_points = {}

goals_for = {}
goals_against = {}

matches_played = {}

elo = {}

K_FACTOR = 20

model_rows = []


for _, match in df_2025.iterrows():

    home = match["Home"]
    away = match["Away"]

    hg = match["HG"]
    ag = match["AG"]


    # ----------------------------------------------
    # 試合前の勝点
    # ----------------------------------------------

    home_points = points.get(
        home,
        0
    )

    away_points = points.get(
        away,
        0
    )


    # ----------------------------------------------
    # 試合前の直近5試合
    # ----------------------------------------------

    home_recent = recent_points.get(
        home,
        []
    )

    away_recent = recent_points.get(
        away,
        []
    )

    home_form = sum(
        home_recent[-5:]
    )

    away_form = sum(
        away_recent[-5:]
    )


    # ----------------------------------------------
    # 試合前の得点・失点
    # ----------------------------------------------

    home_goals_for = goals_for.get(
        home,
        0
    )

    home_goals_against = goals_against.get(
        home,
        0
    )

    away_goals_for = goals_for.get(
        away,
        0
    )

    away_goals_against = goals_against.get(
        away,
        0
    )


    # ----------------------------------------------
    # 試合前の試合数
    # ----------------------------------------------

    home_matches = matches_played.get(
        home,
        0
    )

    away_matches = matches_played.get(
        away,
        0
    )


    # ----------------------------------------------
    # 試合前の得失点差
    # ----------------------------------------------

    home_goal_diff = (
        home_goals_for -
        home_goals_against
    )

    away_goal_diff = (
        away_goals_for -
        away_goals_against
    )


    # ----------------------------------------------
    # 試合前の平均得点・平均失点
    # ----------------------------------------------

    if home_matches > 0:

        home_attack = (
            home_goals_for /
            home_matches
        )

        home_defense = (
            home_goals_against /
            home_matches
        )

    else:

        home_attack = 0.0
        home_defense = 0.0


    if away_matches > 0:

        away_attack = (
            away_goals_for /
            away_matches
        )

        away_defense = (
            away_goals_against /
            away_matches
        )

    else:

        away_attack = 0.0
        away_defense = 0.0


    # ----------------------------------------------
    # 攻撃力・守備力の差
    #
    # ホーム側：
    # 自分の平均得点 - 相手の平均失点
    #
    # アウェイ側：
    # 自分の平均得点 - 相手の平均失点
    #
    # 最後にホーム側－アウェイ側
    # ----------------------------------------------

    home_attack_edge = (
        home_attack -
        away_defense
    )

    away_attack_edge = (
        away_attack -
        home_defense
    )

    attack_defense_diff = (
        home_attack_edge -
        away_attack_edge
    )


    # ----------------------------------------------
    # 試合前のElo
    # ----------------------------------------------

    home_elo = elo.get(
        home,
        1500.0
    )

    away_elo = elo.get(
        away,
        1500.0
    )


    # ----------------------------------------------
    # 重要
    # 試合結果を反映する前に保存
    # ----------------------------------------------

    model_rows.append({

        "Date":
            match["Date"],

        "Home":
            home,

        "Away":
            away,

        "HomePointsBefore":
            home_points,

        "AwayPointsBefore":
            away_points,

        "PointsDiff":
            home_points -
            away_points,

        "HomeForm5":
            home_form,

        "AwayForm5":
            away_form,

        "HomeGoalDiffBefore":
            home_goal_diff,

        "AwayGoalDiffBefore":
            away_goal_diff,

        "HomeEloBefore":
            home_elo,

        "AwayEloBefore":
            away_elo,

        "HomeAttackBefore":
            home_attack,

        "AwayAttackBefore":
            away_attack,

        "HomeDefenseBefore":
            home_defense,

        "AwayDefenseBefore":
            away_defense,

        "AttackDefenseDiff":
            attack_defense_diff
    })


    # ==================================================
    # ここから下は試合終了後の更新
    # ==================================================

    if hg > ag:

        home_match_points = 3
        away_match_points = 0

        home_actual = 1.0
        away_actual = 0.0

    elif hg < ag:

        home_match_points = 0
        away_match_points = 3

        home_actual = 0.0
        away_actual = 1.0

    else:

        home_match_points = 1
        away_match_points = 1

        home_actual = 0.5
        away_actual = 0.5


    # ----------------------------------------------
    # 勝点更新
    # ----------------------------------------------

    points[home] = (
        home_points +
        home_match_points
    )

    points[away] = (
        away_points +
        away_match_points
    )


    # ----------------------------------------------
    # 直近成績更新
    # ----------------------------------------------

    recent_points.setdefault(
        home,
        []
    ).append(
        home_match_points
    )

    recent_points.setdefault(
        away,
        []
    ).append(
        away_match_points
    )


    # ----------------------------------------------
    # 得点・失点更新
    # ----------------------------------------------

    goals_for[home] = (
        home_goals_for +
        hg
    )

    goals_against[home] = (
        home_goals_against +
        ag
    )

    goals_for[away] = (
        away_goals_for +
        ag
    )

    goals_against[away] = (
        away_goals_against +
        hg
    )


    # ----------------------------------------------
    # 試合数更新
    # ----------------------------------------------

    matches_played[home] = (
        home_matches + 1
    )

    matches_played[away] = (
        away_matches + 1
    )


    # ----------------------------------------------
    # Elo更新
    # ----------------------------------------------

    home_expected = (
        1 /
        (
            1 +
            10 ** (
                (
                    away_elo -
                    home_elo
                ) /
                400
            )
        )
    )

    away_expected = (
        1 -
        home_expected
    )

    elo[home] = (
        home_elo +
        K_FACTOR *
        (
            home_actual -
            home_expected
        )
    )

    elo[away] = (
        away_elo +
        K_FACTOR *
        (
            away_actual -
            away_expected
        )
    )


# ==================================================
# DataFrame作成
# ==================================================

model_data = pd.DataFrame(
    model_rows
)


# 直近5試合の差
model_data["FormDiff"] = (
    model_data["HomeForm5"] -
    model_data["AwayForm5"]
)


# 得失点差の差
model_data["GoalDiffDiff"] = (
    model_data[
        "HomeGoalDiffBefore"
    ] -
    model_data[
        "AwayGoalDiffBefore"
    ]
)


# Elo差
model_data["EloDiff"] = (
    model_data["HomeEloBefore"] -
    model_data["AwayEloBefore"]
)


# ==================================================
# 実際の結果
# ==================================================

model_data["Result"] = [
    "H" if hg > ag else
    "A" if hg < ag else
    "D"

    for hg, ag in zip(
        df_2025["HG"],
        df_2025["AG"]
    )
]


actual_h = (
    model_data["Result"] == "H"
).astype(int)

actual_d = (
    model_data["Result"] == "D"
).astype(int)

actual_a = (
    model_data["Result"] == "A"
).astype(int)


# ==================================================
# 評価関数
# ==================================================

def calculate_metrics(
    data,
    prediction_column,
    prob_h,
    prob_d,
    prob_a
):

    accuracy = (
        data[prediction_column] ==
        data["Result"]
    ).mean()

    actual_prob = np.where(
        data["Result"] == "H",
        data[prob_h],
        np.where(
            data["Result"] == "D",
            data[prob_d],
            data[prob_a]
        )
    )

    # 念のため0を避ける
    actual_prob = np.clip(
        actual_prob,
        1e-15,
        1.0
    )

    log_loss = -np.mean(
        np.log(actual_prob)
    )

    brier = np.mean(
        (data[prob_h] - actual_h) ** 2 +
        (data[prob_d] - actual_d) ** 2 +
        (data[prob_a] - actual_a) ** 2
    )

    return (
        accuracy,
        log_loss,
        brier
    )


# ==================================================
# 最大確率からH/D/Aを選ぶ
# ==================================================

def make_prediction(
    data,
    columns
):

    return (
        data[columns]
        .idxmax(axis=1)
        .map({
            columns[0]: "H",
            columns[1]: "D",
            columns[2]: "A"
        })
    )


# ==================================================
# M1
# ==================================================

m1_probs = model_data[
    "PointsDiff"
].apply(
    m1_probabilities
)

model_data = pd.concat(
    [
        model_data,
        m1_probs
    ],
    axis=1
)


# M1は従来方式を維持
model_data["M1_Prediction"] = (
    model_data[
        "PointsDiff"
    ].apply(
        lambda x:
        "H" if x > 0 else
        "A" if x < 0 else
        "D"
    )
)


m1_accuracy, m1_log_loss, m1_brier = (
    calculate_metrics(
        model_data,
        "M1_Prediction",
        "M1_Prob_H",
        "M1_Prob_D",
        "M1_Prob_A"
    )
)


# ==================================================
# M2
# ==================================================

m2_probs = model_data[
    "PointsDiff"
].apply(
    m2_probabilities
)

model_data = pd.concat(
    [
        model_data,
        m2_probs
    ],
    axis=1
)

model_data["M2_Prediction"] = (
    make_prediction(
        model_data,
        [
            "M2_Prob_H",
            "M2_Prob_D",
            "M2_Prob_A"
        ]
    )
)

m2_accuracy, m2_log_loss, m2_brier = (
    calculate_metrics(
        model_data,
        "M2_Prediction",
        "M2_Prob_H",
        "M2_Prob_D",
        "M2_Prob_A"
    )
)


# ==================================================
# M3
# ==================================================

m3_probs = model_data.apply(
    lambda row:
    m3_probabilities(
        row["PointsDiff"],
        row["FormDiff"]
    ),
    axis=1
)

model_data = pd.concat(
    [
        model_data,
        m3_probs
    ],
    axis=1
)

model_data["M3_Prediction"] = (
    make_prediction(
        model_data,
        [
            "M3_Prob_H",
            "M3_Prob_D",
            "M3_Prob_A"
        ]
    )
)

m3_accuracy, m3_log_loss, m3_brier = (
    calculate_metrics(
        model_data,
        "M3_Prediction",
        "M3_Prob_H",
        "M3_Prob_D",
        "M3_Prob_A"
    )
)


# ==================================================
# M4
# ==================================================

m4_probs = model_data.apply(
    lambda row:
    m4_probabilities(
        row["PointsDiff"],
        row["FormDiff"],
        row["GoalDiffDiff"]
    ),
    axis=1
)

model_data = pd.concat(
    [
        model_data,
        m4_probs
    ],
    axis=1
)

model_data["M4_Prediction"] = (
    make_prediction(
        model_data,
        [
            "M4_Prob_H",
            "M4_Prob_D",
            "M4_Prob_A"
        ]
    )
)

m4_accuracy, m4_log_loss, m4_brier = (
    calculate_metrics(
        model_data,
        "M4_Prediction",
        "M4_Prob_H",
        "M4_Prob_D",
        "M4_Prob_A"
    )
)


# ==================================================
# M5
# ==================================================

m5_probs = model_data.apply(
    lambda row:
    m5_probabilities(
        row["PointsDiff"],
        row["FormDiff"],
        row["GoalDiffDiff"],
        row["EloDiff"]
    ),
    axis=1
)

model_data = pd.concat(
    [
        model_data,
        m5_probs
    ],
    axis=1
)

model_data["M5_Prediction"] = (
    make_prediction(
        model_data,
        [
            "M5_Prob_H",
            "M5_Prob_D",
            "M5_Prob_A"
        ]
    )
)

m5_accuracy, m5_log_loss, m5_brier = (
    calculate_metrics(
        model_data,
        "M5_Prediction",
        "M5_Prob_H",
        "M5_Prob_D",
        "M5_Prob_A"
    )
)


# ==================================================
# M6
# ==================================================

m6_probs = model_data.apply(
    lambda row:
    m6_probabilities(
        row["PointsDiff"],
        row["FormDiff"],
        row["GoalDiffDiff"],
        row["EloDiff"],
        row["AttackDefenseDiff"]
    ),
    axis=1
)

model_data = pd.concat(
    [
        model_data,
        m6_probs
    ],
    axis=1
)

model_data["M6_Prediction"] = (
    make_prediction(
        model_data,
        [
            "M6_Prob_H",
            "M6_Prob_D",
            "M6_Prob_A"
        ]
    )
)

m6_accuracy, m6_log_loss, m6_brier = (
    calculate_metrics(
        model_data,
        "M6_Prediction",
        "M6_Prob_H",
        "M6_Prob_D",
        "M6_Prob_A"
    )
)


# ==================================================
# モデル評価
# ==================================================

st.header("📊 モデル評価")


metrics = [
    (
        "M1：勝点差",
        m1_accuracy,
        m1_log_loss,
        m1_brier
    ),

    (
        "M2：勝点差＋ホーム補正",
        m2_accuracy,
        m2_log_loss,
        m2_brier
    ),

    (
        "M3：M2＋直近5試合",
        m3_accuracy,
        m3_log_loss,
        m3_brier
    ),

    (
        "M4：M3＋得失点差",
        m4_accuracy,
        m4_log_loss,
        m4_brier
    ),

    (
        "M5：M4＋Elo",
        m5_accuracy,
        m5_log_loss,
        m5_brier
    ),

    (
        "M6：M5＋攻撃力・守備力",
        m6_accuracy,
        m6_log_loss,
        m6_brier
    )
]


for (
    title,
    accuracy,
    log_loss,
    brier
) in metrics:

    st.subheader(title)

    st.write(
        "正解率:",
        f"{accuracy:.1%}"
    )

    st.write(
        "Log Loss:",
        round(log_loss, 4)
    )

    st.write(
        "Brier Score:",
        round(brier, 4)
    )


# ==================================================
# 比較表
# ==================================================

st.header("📋 モデル比較")


comparison = pd.DataFrame({

    "モデル": [
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6"
    ],

    "内容": [
        "勝点差",
        "＋ホーム補正",
        "＋直近5試合",
        "＋得失点差",
        "＋Elo",
        "＋攻撃力・守備力"
    ],

    "正解率": [
        m1_accuracy,
        m2_accuracy,
        m3_accuracy,
        m4_accuracy,
        m5_accuracy,
        m6_accuracy
    ],

    "Log Loss": [
        m1_log_loss,
        m2_log_loss,
        m3_log_loss,
        m4_log_loss,
        m5_log_loss,
        m6_log_loss
    ],

    "Brier Score": [
        m1_brier,
        m2_brier,
        m3_brier,
        m4_brier,
        m5_brier,
        m6_brier
    ]
})


comparison["正解率"] = (
    comparison["正解率"] *
    100
).round(1)

comparison["Log Loss"] = (
    comparison[
        "Log Loss"
    ].round(4)
)

comparison["Brier Score"] = (
    comparison[
        "Brier Score"
    ].round(4)
)


st.dataframe(
    comparison,
    hide_index=True
)


# ==================================================
# 検証
# ==================================================

st.header("🔍 M1〜M6 検証")


# ==================================================
# ① 正解数と予測内訳
# ==================================================

verification_rows = []


for model in [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6"
]:

    prediction_col = (
        f"{model}_Prediction"
    )

    predictions = (
        model_data[
            prediction_col
        ]
    )

    correct_count = (
        predictions ==
        model_data["Result"]
    ).sum()

    verification_rows.append({

        "モデル":
            model,

        "正解数":
            int(correct_count),

        "全試合":
            len(model_data),

        "正解率":
            round(
                correct_count /
                len(model_data) *
                100,
                2
            ),

        "H予測":
            int(
                (
                    predictions == "H"
                ).sum()
            ),

        "D予測":
            int(
                (
                    predictions == "D"
                ).sum()
            ),

        "A予測":
            int(
                (
                    predictions == "A"
                ).sum()
            )
    })


verification = pd.DataFrame(
    verification_rows
)


st.subheader(
    "① 正解数と予測内訳"
)

st.dataframe(
    verification,
    hide_index=True
)


# ==================================================
# ② 前モデルから予測が変わった試合
# ==================================================

change_rows = []


model_pairs = [
    ("M1", "M2"),
    ("M2", "M3"),
    ("M3", "M4"),
    ("M4", "M5"),
    ("M5", "M6")
]


for (
    old_model,
    new_model
) in model_pairs:

    old_col = (
        f"{old_model}_Prediction"
    )

    new_col = (
        f"{new_model}_Prediction"
    )

    changed = (
        model_data[old_col] !=
        model_data[new_col]
    )

    became_correct = (
        changed &
        (
            model_data[new_col] ==
            model_data["Result"]
        ) &
        (
            model_data[old_col] !=
            model_data["Result"]
        )
    ).sum()

    became_wrong = (
        changed &
        (
            model_data[new_col] !=
            model_data["Result"]
        ) &
        (
            model_data[old_col] ==
            model_data["Result"]
        )
    ).sum()

    change_rows.append({

        "比較":
            f"{old_model} → {new_model}",

        "予測が変わった試合":
            int(
                changed.sum()
            ),

        "変更で正解になった":
            int(
                became_correct
            ),

        "変更で不正解になった":
            int(
                became_wrong
            ),

        "正解数の差":
            int(
                became_correct -
                became_wrong
            )
    })


changes = pd.DataFrame(
    change_rows
)


st.subheader(
    "② 前モデルから予測が変わった試合"
)

st.dataframe(
    changes,
    hide_index=True
)


# ==================================================
# ③ 実際の結果
# ==================================================

actual_counts = (
    model_data["Result"]
    .value_counts()
)


actual_table = pd.DataFrame({

    "結果": [
        "H（ホーム勝ち）",
        "D（引き分け）",
        "A（アウェイ勝ち）"
    ],

    "試合数": [
        int(
            actual_counts.get(
                "H",
                0
            )
        ),

        int(
            actual_counts.get(
                "D",
                0
            )
        ),

        int(
            actual_counts.get(
                "A",
                0
            )
        )
    ]
})


st.subheader(
    "③ 実際の2025年J1結果"
)

st.dataframe(
    actual_table,
    hide_index=True
)


# ==================================================
# ④ 引き分けRecall
# ==================================================

draw_rows = []


actual_draws = (
    model_data["Result"] == "D"
).sum()


for model in [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6"
]:

    prediction_col = (
        f"{model}_Prediction"
    )

    draw_predictions = (
        model_data[
            prediction_col
        ] == "D"
    )

    correct_draws = (
        (
            model_data[
                "Result"
            ] == "D"
        ) &
        draw_predictions
    ).sum()

    if actual_draws > 0:

        draw_recall = (
            correct_draws /
            actual_draws *
            100
        )

    else:

        draw_recall = 0


    draw_rows.append({

        "モデル":
            model,

        "実際の引き分け":
            int(actual_draws),

        "引き分け予測数":
            int(
                draw_predictions.sum()
            ),

        "的中した引き分け":
            int(correct_draws),

        "引き分けRecall":
            round(
                draw_recall,
                1
            )
    })


draw_verification = pd.DataFrame(
    draw_rows
)


st.subheader(
    "④ 引き分け予測の確認"
)

st.dataframe(
    draw_verification,
    hide_index=True
)


# ==================================================
# ⑤ 予測一致チェック
# ==================================================

st.subheader(
    "⑤ 前モデルとの予測一致"
)


for (
    old_model,
    new_model
) in model_pairs:

    same_count = (
        model_data[
            f"{old_model}_Prediction"
        ] ==
        model_data[
            f"{new_model}_Prediction"
        ]
    ).sum()

    st.write(
        f"{old_model} と {new_model} が同じ予測:",
        f"{same_count} / {len(model_data)}"
    )


# ==================================================
# ⑥ M6特徴量の確認
# ==================================================

st.subheader(
    "⑥ M6 攻撃力・守備力データ確認"
)


st.dataframe(
    model_data[
        [
            "Date",
            "Home",
            "Away",
            "HomeAttackBefore",
            "HomeDefenseBefore",
            "AwayAttackBefore",
            "AwayDefenseBefore",
            "AttackDefenseDiff"
        ]
    ].tail(20),
    hide_index=True
)


# ==================================================
# 試合予測画面
# ==================================================

st.header("⚽ 試合予測")


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


home_team = st.selectbox(
    "🏠 ホームチーム",
    teams
)

away_team = st.selectbox(
    "✈️ アウェイチーム",
    teams,
    index=1
)


if st.button(
    "試合を予測する"
):

    if (
        home_team ==
        away_team
    ):

        st.error(
            "ホームとアウェイには別のチームを選んでください。"
        )

    else:

        st.success(
            f"{home_team} vs {away_team}"
        )

        st.info(
            "実際の試合予測機能は、学習モデル完成後に接続します。"
        )
