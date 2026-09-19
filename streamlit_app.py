import streamlit as st
import pandas as pd
import numpy as np


# =========================
# 基本設定
# =========================

st.set_page_config(
    page_title="J1 Match Predictor",
    page_icon="⚽",
)

st.title("⚽ J1 Match Predictor")
st.write("J1リーグの試合結果を予測するアプリです。")


# =========================
# データ読み込み
# =========================

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


# =========================
# 確率計算関数
# =========================

# M1：勝点差
def m1_probabilities(points_diff):

    home_score = np.exp(points_diff / 10)
    away_score = np.exp(-points_diff / 10)
    draw_score = 1.0

    total = home_score + draw_score + away_score

    return pd.Series({
        "M1_Prob_H": home_score / total,
        "M1_Prob_D": draw_score / total,
        "M1_Prob_A": away_score / total
    })


# M2：勝点差＋ホーム補正
def m2_probabilities(points_diff):

    home_advantage = 2.0

    strength = (
        points_diff +
        home_advantage
    )

    home_score = np.exp(strength / 10)
    away_score = np.exp(-strength / 10)
    draw_score = 1.0

    total = home_score + draw_score + away_score

    return pd.Series({
        "M2_Prob_H": home_score / total,
        "M2_Prob_D": draw_score / total,
        "M2_Prob_A": away_score / total
    })


# M3：M2＋直近5試合
def m3_probabilities(
    points_diff,
    form_diff
):

    home_advantage = 2.0
    form_weight = 0.5

    strength = (
        points_diff +
        home_advantage +
        form_diff * form_weight
    )

    home_score = np.exp(strength / 10)
    away_score = np.exp(-strength / 10)
    draw_score = 1.0

    total = home_score + draw_score + away_score

    return pd.Series({
        "M3_Prob_H": home_score / total,
        "M3_Prob_D": draw_score / total,
        "M3_Prob_A": away_score / total
    })


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

    home_score = np.exp(strength / 10)
    away_score = np.exp(-strength / 10)
    draw_score = 1.0

    total = home_score + draw_score + away_score

    return pd.Series({
        "M4_Prob_H": home_score / total,
        "M4_Prob_D": draw_score / total,
        "M4_Prob_A": away_score / total
    })


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

    # Elo差100を勝点差約2相当として扱う仮設定
    elo_weight = 0.02

    strength = (
        points_diff +
        home_advantage +
        form_diff * form_weight +
        goal_diff_diff * goal_diff_weight +
        elo_diff * elo_weight
    )

    home_score = np.exp(strength / 10)
    away_score = np.exp(-strength / 10)
    draw_score = 1.0

    total = home_score + draw_score + away_score

    return pd.Series({
        "M5_Prob_H": home_score / total,
        "M5_Prob_D": draw_score / total,
        "M5_Prob_A": away_score / total
    })


# =========================
# 試合前データ作成
# =========================

points = {}
recent_points = {}

goals_for = {}
goals_against = {}

# 全チームのEloは1500から開始
elo = {}

# Eloの変動幅
K_FACTOR = 20

model_rows = []


for _, match in df_2025.iterrows():

    home = match["Home"]
    away = match["Away"]

    # -------------------------
    # 試合前の勝点
    # -------------------------

    home_points = points.get(home, 0)
    away_points = points.get(away, 0)


    # -------------------------
    # 試合前の直近5試合
    # -------------------------

    home_recent = recent_points.get(
        home, []
    )

    away_recent = recent_points.get(
        away, []
    )

    home_form = sum(
        home_recent[-5:]
    )

    away_form = sum(
        away_recent[-5:]
    )


    # -------------------------
    # 試合前の得失点差
    # -------------------------

    home_goals_for = goals_for.get(
        home, 0
    )

    home_goals_against = goals_against.get(
        home, 0
    )

    away_goals_for = goals_for.get(
        away, 0
    )

    away_goals_against = goals_against.get(
        away, 0
    )

    home_goal_diff = (
        home_goals_for -
        home_goals_against
    )

    away_goal_diff = (
        away_goals_for -
        away_goals_against
    )


    # -------------------------
    # 試合前のElo
    # -------------------------

    home_elo = elo.get(
        home, 1500.0
    )

    away_elo = elo.get(
        away, 1500.0
    )


    # -------------------------
    # 試合前情報を保存
    # -------------------------

    model_rows.append({
        "Home": home,
        "Away": away,

        "HomePointsBefore":
            home_points,

        "AwayPointsBefore":
            away_points,

        "PointsDiff":
            home_points - away_points,

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
            away_elo
    })


    # =========================
    # 実際の試合結果
    # =========================

    if match["HG"] > match["AG"]:

        home_match_points = 3
        away_match_points = 0

        home_actual = 1.0
        away_actual = 0.0

    elif match["HG"] < match["AG"]:

        home_match_points = 0
        away_match_points = 3

        home_actual = 0.0
        away_actual = 1.0

    else:

        home_match_points = 1
        away_match_points = 1

        home_actual = 0.5
        away_actual = 0.5


    # =========================
    # 試合終了後に勝点更新
    # =========================

    points[home] = (
        home_points +
        home_match_points
    )

    points[away] = (
        away_points +
        away_match_points
    )


    # =========================
    # 直近成績更新
    # =========================

    recent_points.setdefault(
        home, []
    ).append(
        home_match_points
    )

    recent_points.setdefault(
        away, []
    ).append(
        away_match_points
    )


    # =========================
    # 得点・失点更新
    # =========================

    goals_for[home] = (
        home_goals_for +
        match["HG"]
    )

    goals_against[home] = (
        home_goals_against +
        match["AG"]
    )

    goals_for[away] = (
        away_goals_for +
        match["AG"]
    )

    goals_against[away] = (
        away_goals_against +
        match["HG"]
    )


    # =========================
    # Elo更新
    # =========================

    # ホームチームの期待勝率
    home_expected = (
        1 /
        (
            1 +
            10 ** (
                (away_elo - home_elo) /
                400
            )
        )
    )

    # アウェイチームの期待勝率
    away_expected = (
        1 - home_expected
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


# =========================
# DataFrame作成
# =========================

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
    model_data["HomeGoalDiffBefore"] -
    model_data["AwayGoalDiffBefore"]
)


# Elo差
model_data["EloDiff"] = (
    model_data["HomeEloBefore"] -
    model_data["AwayEloBefore"]
)


# =========================
# 実際の試合結果
# =========================

model_data["Result"] = [
    "H" if hg > ag else
    "A" if hg < ag else
    "D"

    for hg, ag in zip(
        df_2025["HG"],
        df_2025["AG"]
    )
]


# =========================
# 実際の結果を0/1に変換
# =========================

actual_h = (
    model_data["Result"] == "H"
).astype(int)

actual_d = (
    model_data["Result"] == "D"
).astype(int)

actual_a = (
    model_data["Result"] == "A"
).astype(int)


# =========================
# 評価関数
# =========================

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


# =========================
# 予測結果を作る共通関数
# =========================

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


# =========================
# M1
# =========================

m1_probs = model_data[
    "PointsDiff"
].apply(
    m1_probabilities
)

model_data = pd.concat(
    [model_data, m1_probs],
    axis=1
)

model_data["M1_Prediction"] = (
    model_data["PointsDiff"].apply(
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


# =========================
# M2
# =========================

m2_probs = model_data[
    "PointsDiff"
].apply(
    m2_probabilities
)

model_data = pd.concat(
    [model_data, m2_probs],
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


# =========================
# M3
# =========================

m3_probs = model_data.apply(
    lambda row:
    m3_probabilities(
        row["PointsDiff"],
        row["FormDiff"]
    ),
    axis=1
)

model_data = pd.concat(
    [model_data, m3_probs],
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


# =========================
# M4
# =========================

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
    [model_data, m4_probs],
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


# =========================
# M5
# =========================

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
    [model_data, m5_probs],
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


# =========================
# モデル評価
# =========================

st.header("📊 モデル評価")


st.subheader("M1：勝点差")
st.write(
    "正解率:",
    f"{m1_accuracy:.1%}"
)
st.write(
    "Log Loss:",
    round(m1_log_loss, 4)
)
st.write(
    "Brier Score:",
    round(m1_brier, 4)
)


st.subheader(
    "M2：勝点差＋ホーム補正"
)
st.write(
    "正解率:",
    f"{m2_accuracy:.1%}"
)
st.write(
    "Log Loss:",
    round(m2_log_loss, 4)
)
st.write(
    "Brier Score:",
    round(m2_brier, 4)
)


st.subheader(
    "M3：M2＋直近5試合"
)
st.write(
    "正解率:",
    f"{m3_accuracy:.1%}"
)
st.write(
    "Log Loss:",
    round(m3_log_loss, 4)
)
st.write(
    "Brier Score:",
    round(m3_brier, 4)
)


st.subheader(
    "M4：M3＋得失点差"
)
st.write(
    "正解率:",
    f"{m4_accuracy:.1%}"
)
st.write(
    "Log Loss:",
    round(m4_log_loss, 4)
)
st.write(
    "Brier Score:",
    round(m4_brier, 4)
)


st.subheader(
    "M5：M4＋Elo"
)
st.write(
    "正解率:",
    f"{m5_accuracy:.1%}"
)
st.write(
    "Log Loss:",
    round(m5_log_loss, 4)
)
st.write(
    "Brier Score:",
    round(m5_brier, 4)
)


# =========================
# モデル比較表
# =========================

st.subheader("📋 モデル比較")

comparison = pd.DataFrame({

    "モデル": [
        "M1",
        "M2",
        "M3",
        "M4",
        "M5"
    ],

    "内容": [
        "勝点差",
        "勝点差＋ホーム補正",
        "M2＋直近5試合",
        "M3＋得失点差",
        "M4＋Elo"
    ],

    "正解率": [
        m1_accuracy,
        m2_accuracy,
        m3_accuracy,
        m4_accuracy,
        m5_accuracy
    ],

    "Log Loss": [
        m1_log_loss,
        m2_log_loss,
        m3_log_loss,
        m4_log_loss,
        m5_log_loss
    ],

    "Brier Score": [
        m1_brier,
        m2_brier,
        m3_brier,
        m4_brier,
        m5_brier
    ]
})


comparison["正解率"] = (
    comparison["正解率"] * 100
).round(1)

comparison["Log Loss"] = (
    comparison["Log Loss"]
    .round(4)
)

comparison["Brier Score"] = (
    comparison["Brier Score"]
    .round(4)
)


st.dataframe(
    comparison,
    hide_index=True
)


# =========================
# 試合予測画面
# =========================

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

# =========================
# 🔍 M1〜M5 検証
# =========================

st.header("🔍 M1〜M5 検証")


# -------------------------
# 正解数を確認
# -------------------------

verification_rows = []

for model in [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5"
]:

    prediction_col = (
        f"{model}_Prediction"
    )

    predictions = (
        model_data[prediction_col]
    )

    correct_count = (
        predictions ==
        model_data["Result"]
    ).sum()

    home_count = (
        predictions == "H"
    ).sum()

    draw_count = (
        predictions == "D"
    ).sum()

    away_count = (
        predictions == "A"
    ).sum()

    verification_rows.append({
        "モデル": model,
        "正解数": int(correct_count),
        "全試合": len(model_data),
        "正解率": (
            correct_count /
            len(model_data) *
            100
        ),
        "H予測": int(home_count),
        "D予測": int(draw_count),
        "A予測": int(away_count)
    })


verification = pd.DataFrame(
    verification_rows
)

verification["正解率"] = (
    verification["正解率"]
    .round(2)
)


st.subheader(
    "① 正解数と予測内訳"
)

st.dataframe(
    verification,
    hide_index=True
)


# =========================
# 前モデルから何試合変わったか
# =========================

change_rows = []

model_pairs = [
    ("M1", "M2"),
    ("M2", "M3"),
    ("M3", "M4"),
    ("M4", "M5")
]


for old_model, new_model in model_pairs:

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

    changed_count = changed.sum()

    # 変更によって正解になった試合
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

    # 変更によって不正解になった試合
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
            int(changed_count),

        "変更で正解になった":
            int(became_correct),

        "変更で不正解になった":
            int(became_wrong),

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


# =========================
# 実際の結果の内訳
# =========================

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
        int(actual_counts.get("H", 0)),
        int(actual_counts.get("D", 0)),
        int(actual_counts.get("A", 0))
    ]
})


st.subheader(
    "③ 実際の2025年J1結果"
)

st.dataframe(
    actual_table,
    hide_index=True
)


# =========================
# 引き分けをどれだけ当てたか
# =========================

draw_rows = []

actual_draws = (
    model_data["Result"] == "D"
).sum()


for model in [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5"
]:

    prediction_col = (
        f"{model}_Prediction"
    )

    correct_draws = (
        (
            model_data["Result"] == "D"
        ) &
        (
            model_data[prediction_col] == "D"
        )
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
        "モデル": model,

        "実際の引き分け":
            int(actual_draws),

        "引き分け予測数":
            int(
                (
                    model_data[
                        prediction_col
                    ] == "D"
                ).sum()
            ),

        "的中した引き分け":
            int(correct_draws),

        "引き分けRecall":
            round(draw_recall, 1)
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


# =========================
# M2〜M5が完全一致しているか
# =========================

st.subheader(
    "⑤ M2〜M5の予測一致チェック"
)


m2_m3_same = (
    model_data["M2_Prediction"] ==
    model_data["M3_Prediction"]
).sum()

m3_m4_same = (
    model_data["M3_Prediction"] ==
    model_data["M4_Prediction"]
).sum()

m4_m5_same = (
    model_data["M4_Prediction"] ==
    model_data["M5_Prediction"]
).sum()


st.write(
    "M2とM3が同じ予測:",
    f"{m2_m3_same} / {len(model_data)}"
)

st.write(
    "M3とM4が同じ予測:",
    f"{m3_m4_same} / {len(model_data)}"
)

st.write(
    "M4とM5が同じ予測:",
    f"{m4_m5_same} / {len(model_data)}"
)

if st.button("試合を予測する"):

    if home_team == away_team:

        st.error(
            "ホームとアウェイには別のチームを選んでください。"
        )

    else:

        st.success(
            f"{home_team} vs {away_team}"
        )

        st.info(
            "予測モデルは次のステップで追加します。"
        )
