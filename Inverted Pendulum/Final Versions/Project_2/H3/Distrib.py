# file: nonlinear_bootstrap_markov_rf.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from scipy.stats import gaussian_kde

# =============================================================================
# CONFIG
# =============================================================================

CSV = (
    r"D:\Trabalhos\Smart Agri\Double_Inverted_Pendulum"
    r"\Double-Inverted-Pendulum\Inverted Pendulum"
    r"\Final Versions\Data Processed"
    r"\pendulum_dataset_tidy_with_acceleration.csv"
)

N_BOOT = 1000
BLOCK_SIZE = 128
TEST_SIZE = 0.25
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

ep0 = (
    df[df["episode"] == 0]
    .copy()
    .reset_index(drop=True)
)

print(f"Samples: {len(ep0):,}")

# =============================================================================
# FEATURES
# =============================================================================

COLS = [
    "theta1",
    "theta2",
    "omega1",
    "omega2",
    "tau1_dynamics",
    "tau2_dynamics",
]

X_base = ep0[COLS].values

# non-linear enrichments
X_extra = np.column_stack([
    np.sin(ep0["theta1"]),
    np.cos(ep0["theta1"]),
    np.sin(ep0["theta2"]),
    np.cos(ep0["theta2"]),
    ep0["theta1"] * ep0["omega1"],
    ep0["theta2"] * ep0["omega2"],
])

X = np.hstack([
    X_base,
    X_extra,
])

scaler = StandardScaler()

X = scaler.fit_transform(X)

# target
y = np.roll(
    ep0["omega1"].values,
    -1,
)

X = X[:-1]
y = y[:-1]

# =============================================================================
# BUILD LAGS
# =============================================================================

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

# =============================================================================
# RANDOM FOREST COMPARISON
# =============================================================================

def compute_delta_r2(
    X0,
    X1,
    y,
    rng_seed=42,
):

    n = len(y)

    idx = np.arange(n)

    rng = np.random.default_rng(rng_seed)

    rng.shuffle(idx)

    split = int((1 - TEST_SIZE) * n)

    train_idx = idx[:split]
    test_idx = idx[split:]

    # ---------------------------------------------------------
    # H0 model
    # ---------------------------------------------------------

    rf0 = RandomForestRegressor(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=rng_seed,
    )

    rf0.fit(
        X0[train_idx],
        y[train_idx],
    )

    pred0 = rf0.predict(
        X0[test_idx]
    )

    r2_0 = r2_score(
        y[test_idx],
        pred0,
    )

    # ---------------------------------------------------------
    # H1 model
    # ---------------------------------------------------------

    rf1 = RandomForestRegressor(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=rng_seed,
    )

    rf1.fit(
        X1[train_idx],
        y[train_idx],
    )

    pred1 = rf1.predict(
        X1[test_idx]
    )

    r2_1 = r2_score(
        y[test_idx],
        pred1,
    )

    delta_r2 = r2_1 - r2_0

    return (
        delta_r2,
        r2_0,
        r2_1,
    )

# =============================================================================
# OBSERVED
# =============================================================================

print("\nComputing observed ΔR²...")

delta_obs, r2_obs_0, r2_obs_1 = compute_delta_r2(
    X_markov,
    X_memory,
    y_t,
)

print(f"\nObserved ΔR² = {delta_obs:.8f}")

# =============================================================================
# BLOCK BOOTSTRAP
# =============================================================================

print("\nRunning bootstrap...")

rng = np.random.default_rng(123)

n = len(y_t)

n_blocks = int(np.ceil(n / BLOCK_SIZE))

boot_delta = []

for b in range(N_BOOT):

    idx = []

    for _ in range(n_blocks):

        start = rng.integers(
            0,
            n - BLOCK_SIZE,
        )

        idx.extend(
            range(
                start,
                start + BLOCK_SIZE,
            )
        )

    idx = np.array(idx[:n])

    X0_boot = X_markov[idx]
    X1_boot = X_memory[idx]
    y_boot = y_t[idx]

    d_r2, _, _ = compute_delta_r2(
        X0_boot,
        X1_boot,
        y_boot,
        rng_seed=b,
    )

    boot_delta.append(d_r2)

    if (b + 1) % 50 == 0:
        print(f"{b+1}/{N_BOOT}")

boot_delta = np.array(boot_delta)

# =============================================================================
# EMPIRICAL TEST
# =============================================================================

p_empirical = np.mean(
    boot_delta >= delta_obs
)

crit = np.quantile(
    boot_delta,
    1 - ALPHA,
)

print("\n================================================")
print("NONLINEAR BOOTSTRAP RESULTS")
print("================================================")

print(f"R² Markov      : {r2_obs_0:.8f}")
print(f"R² Memory      : {r2_obs_1:.8f}")
print(f"Observed ΔR²   : {delta_obs:.8f}")
print(f"Critical ΔR²   : {crit:.8f}")
print(f"Empirical p    : {p_empirical:.8f}")

if delta_obs > crit:
    print("\nDecision: REJECT H0")
    print("Past states add predictive information.")
else:
    print("\nDecision: FAIL TO REJECT H0")
    print("Dynamics compatible with nonlinear Markovity.")

# =============================================================================
# KDE
# =============================================================================

xgrid = np.linspace(
    boot_delta.min(),
    boot_delta.max(),
    1000,
)

kde = gaussian_kde(
    boot_delta,
    bw_method=0.2,
)

ygrid = kde(xgrid)

# =============================================================================
# FIGURE
# =============================================================================

plt.style.use("dark_background")

fig, ax = plt.subplots(
    figsize=(15, 8),
    facecolor="#0D1117",
)

# -----------------------------------------------------------------------------
# bootstrap density
# -----------------------------------------------------------------------------

ax.plot(
    xgrid,
    ygrid,
    color="#58A6FF",
    linewidth=3,
    label=r"Bootstrap $\hat P(\Delta R^2 | H_0)$",
)

ax.fill_between(
    xgrid,
    ygrid,
    color="#58A6FF",
    alpha=0.30,
)

# -----------------------------------------------------------------------------
# observed
# -----------------------------------------------------------------------------

ax.axvline(
    delta_obs,
    color="#FF5555",
    linestyle="--",
    linewidth=3,
    label=rf"Observed $\Delta R^2={delta_obs:.5f}$",
)

# -----------------------------------------------------------------------------
# critical
# -----------------------------------------------------------------------------

ax.axvline(
    crit,
    color="#FFD166",
    linestyle=":",
    linewidth=3,
    label=rf"95% critical region",
)

# -----------------------------------------------------------------------------
# labels
# -----------------------------------------------------------------------------

ax.set_title(
    (
        "Nonlinear Bootstrap Test of Markovianity\n"
        "Random Forest Dynamics Model"
    ),
    fontsize=22,
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
    f"R² Markov = {r2_obs_0:.6f}\n"
    f"R² Memory = {r2_obs_1:.6f}\n"
    f"Observed ΔR² = {delta_obs:.8f}\n"
    f"Bootstrap samples = {N_BOOT}\n"
    f"Block size = {BLOCK_SIZE}"
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

OUT = (
    r"D:\Trabalhos\Smart Agri\Double_Inverted_Pendulum"
    r"\Double-Inverted-Pendulum\Inverted Pendulum"
    r"\Final Versions\Project_2\H3"
    r"\nonlinear_bootstrap_rf.png"
)

plt.savefig(
    OUT,
    dpi=300,
    bbox_inches="tight",
)

print(f"\n[OK] Figure saved:\n{OUT}")

plt.close()

print("\nDone.")