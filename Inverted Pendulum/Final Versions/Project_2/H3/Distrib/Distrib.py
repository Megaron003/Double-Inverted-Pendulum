# file: loeo_xgboost_markov.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

from scipy import stats
from scipy.stats import gaussian_kde

# =============================================================================
# CONFIG
# =============================================================================

CSV = (r"C:\\Users\\Guilherme\\Mestrado\\Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Data Processed\\pendulum_dataset_tidy_with_acceleration.csv")

TARGET = "omega1"

ALPHA = 0.05

# =============================================================================
# LOAD
# =============================================================================

print("\nLoading dataset...")

df = pd.read_csv(CSV)

df["theta1"] = np.arctan2(
    df["sin_theta1"],
    df["cos_theta1"],
)

df["theta2"] = np.arctan2(
    df["sin_theta2"],
    df["cos_theta2"],
)

episodes = sorted(df["episode"].unique())

print(f"Episodes found: {episodes}")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_features(ep):

    X_base = ep[[
        "theta1",
        "theta2",
        "omega1",
        "omega2",
        "tau1_dynamics",
        "tau2_dynamics",
    ]].values

    X_extra = np.column_stack([

        np.sin(ep["theta1"]),
        np.cos(ep["theta1"]),

        np.sin(ep["theta2"]),
        np.cos(ep["theta2"]),

        ep["theta1"] * ep["omega1"],
        ep["theta2"] * ep["omega2"],

        ep["omega1"] ** 2,
        ep["omega2"] ** 2,

        np.sin(ep["theta1"] - ep["theta2"]),
        np.cos(ep["theta1"] - ep["theta2"]),
    ])

    X = np.hstack([
        X_base,
        X_extra,
    ])

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    return X

# =============================================================================
# BUILD EPISODE DATA
# =============================================================================

episode_data = {}

for ep_id in episodes:

    ep = (
        df[df["episode"] == ep_id]
        .copy()
        .reset_index(drop=True)
    )

    X = build_features(ep)

    y = np.roll(
        ep[TARGET].values,
        -1,
    )

    # remove last
    X = X[:-1]
    y = y[:-1]

    # lags
    X_t0 = X[1:]
    X_t1 = X[:-1]

    y_t = y[1:]

    # H0
    X_markov = X_t0

    # H1
    X_memory = np.hstack([
        X_t0,
        X_t1,
    ])

    episode_data[ep_id] = {
        "X0": X_markov,
        "X1": X_memory,
        "y": y_t,
    }

# =============================================================================
# MODEL
# =============================================================================

def make_model(seed=42):

    return XGBRegressor(

        n_estimators=120,
        max_depth=5,

        learning_rate=0.05,

        subsample=0.80,
        colsample_bytree=0.80,

        reg_alpha=0.25,
        reg_lambda=2.0,

        objective="reg:squarederror",

        tree_method="hist",

        random_state=seed,
        n_jobs=-1,
    )

# =============================================================================
# LEAVE-ONE-EPISODE-OUT
# =============================================================================

results = []

print("\nRunning LOEO inference...\n")

for test_ep in episodes:

    print(f"Testing episode {test_ep}")

    # -------------------------------------------------------------------------
    # split
    # -------------------------------------------------------------------------

    train_eps = [
        ep for ep in episodes
        if ep != test_ep
    ]

    # -------------------------------------------------------------------------
    # concatenate train
    # -------------------------------------------------------------------------

    X0_train = np.vstack([
        episode_data[e]["X0"]
        for e in train_eps
    ])

    X1_train = np.vstack([
        episode_data[e]["X1"]
        for e in train_eps
    ])

    y_train = np.concatenate([
        episode_data[e]["y"]
        for e in train_eps
    ])

    # -------------------------------------------------------------------------
    # test
    # -------------------------------------------------------------------------

    X0_test = episode_data[test_ep]["X0"]
    X1_test = episode_data[test_ep]["X1"]
    y_test  = episode_data[test_ep]["y"]

    # -------------------------------------------------------------------------
    # H0
    # -------------------------------------------------------------------------

    model0 = make_model(seed=test_ep)

    model0.fit(
        X0_train,
        y_train,
    )

    pred0 = model0.predict(
        X0_test
    )

    r2_0 = r2_score(
        y_test,
        pred0,
    )

    # -------------------------------------------------------------------------
    # H1
    # -------------------------------------------------------------------------

    model1 = make_model(seed=test_ep)

    model1.fit(
        X1_train,
        y_train,
    )

    pred1 = model1.predict(
        X1_test
    )

    r2_1 = r2_score(
        y_test,
        pred1,
    )

    # -------------------------------------------------------------------------
    # effect size
    # -------------------------------------------------------------------------

    delta_r2 = r2_1 - r2_0

    results.append({

        "episode": test_ep,

        "r2_h0": r2_0,
        "r2_h1": r2_1,

        "delta_r2": delta_r2,
    })

# =============================================================================
# RESULTS
# =============================================================================

res = pd.DataFrame(results)

print("\n================================================")
print("LEAVE-ONE-EPISODE-OUT RESULTS")
print("================================================")

print(res)

# =============================================================================
# GLOBAL INFERENCE
# =============================================================================

delta_vals = res["delta_r2"].values

mean_delta = delta_vals.mean()
std_delta  = delta_vals.std(ddof=1)

# one-sample t-test
t_stat, p_value = stats.ttest_1samp(
    delta_vals,
    0.0,
)

# effect size
cohen_d = mean_delta / std_delta

print("\n================================================")
print("GLOBAL STATISTICAL INFERENCE")
print("================================================")

print(f"Mean ΔR²     : {mean_delta:.10f}")
print(f"Std ΔR²      : {std_delta:.10f}")

print(f"T-statistic  : {t_stat:.6f}")
print(f"P-value      : {p_value:.10f}")

print(f"Cohen's d    : {cohen_d:.6f}")

# =============================================================================
# DECISION
# =============================================================================

if p_value < ALPHA:

    print("\nDecision: REJECT H0")

    print(
        "Past states provide statistically significant "
        "predictive information."
    )

else:

    print("\nDecision: FAIL TO REJECT H0")

    print(
        "Dynamics compatible with nonlinear "
        "Markovianity."
    )

# =============================================================================
# KDE
# =============================================================================

xgrid = np.linspace(
    delta_vals.min() * 0.9,
    delta_vals.max() * 1.1,
    1000,
)

kde = gaussian_kde(
    delta_vals,
    bw_method=0.25,
)

ygrid = kde(xgrid)

# =============================================================================
# FIGURE
# =============================================================================

plt.style.use("dark_background")

fig, ax = plt.subplots(
    figsize=(16, 8),
    facecolor="#0D1117",
)

# -----------------------------------------------------------------------------
# density
# -----------------------------------------------------------------------------

ax.plot(
    xgrid,
    ygrid,
    color="#58A6FF",
    linewidth=3,
    label=r"Empirical $P(\Delta R^2)$ across episodes",
)

ax.fill_between(
    xgrid,
    ygrid,
    color="#58A6FF",
    alpha=0.30,
)

# -----------------------------------------------------------------------------
# observations
# -----------------------------------------------------------------------------

ax.scatter(
    delta_vals,
    np.zeros_like(delta_vals),

    s=120,

    color="#FFD166",
    edgecolors="white",

    zorder=5,

    label="Episode observations",
)

# -----------------------------------------------------------------------------
# mean
# -----------------------------------------------------------------------------

ax.axvline(
    mean_delta,

    color="#FF5555",

    linestyle="--",
    linewidth=3,

    label=rf"Mean $\Delta R^2={mean_delta:.6f}$",
)

# -----------------------------------------------------------------------------
# zero reference
# -----------------------------------------------------------------------------

ax.axvline(
    0,

    color="white",

    linestyle=":",
    linewidth=2,

    alpha=0.7,
)

# -----------------------------------------------------------------------------
# labels
# -----------------------------------------------------------------------------

ax.set_title(
    (
        "Leave-One-Episode-Out Markov Inference\n"
        "Nonlinear XGBoost Dynamics"
    ),

    fontsize=24,
    fontweight="bold",
)

ax.set_xlabel(
    r"$\Delta R^2$",
    fontsize=16,
)

ax.set_ylabel(
    "Probability Density",
    fontsize=16,
)

ax.grid(
    alpha=0.25,
)

ax.legend(
    fontsize=13,
)

# -----------------------------------------------------------------------------
# text
# -----------------------------------------------------------------------------

txt = (

    f"Episodes = {len(episodes)}\n"

    f"Mean ΔR² = {mean_delta:.8f}\n"
    f"Std ΔR²  = {std_delta:.8f}\n"

    f"p-value  = {p_value:.8f}\n"
    f"Cohen d  = {cohen_d:.6f}"
)

ax.text(
    0.02,
    0.97,

    txt,

    transform=ax.transAxes,

    va="top",

    fontsize=12,

    bbox=dict(
        facecolor="black",
        alpha=0.8,
        edgecolor="white",
    ),
)

# =============================================================================
# SAVE
# =============================================================================

plt.tight_layout()

OUT = (r"C:\Users\Guilherme\Mestrado\Inverted_Pendulum\Double-Inverted-Pendulum\Inverted Pendulum\Final Versions\Project_2\H3\loeo_xgboost_markov.png")

plt.savefig(
    OUT,
    dpi=300,
    bbox_inches="tight",
)

print(f"\n[OK] Figure saved:\n{OUT}")

plt.close()

print("\nDone.")