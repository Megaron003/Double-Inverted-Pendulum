# Hypothesis Testing for Neural Network Applicability on Double Inverted Pendulum Data

> **Context:** This document formalizes five statistical hypothesis tests applied to a dataset derived from the natural behavior of a **double inverted pendulum** (two-segment, simulation-generated). The goal is to assess whether the dataset is suitable for two downstream tasks: (1) **system identification** via a predictive neural network capable of inferring the system class from behavioral data alone, and (2) **neural network-based control** with improved learning curve convergence.

---

## Dataset Description

| Variable | Symbol | Description |
|---|---|---|
| Joint angle 1 | $\theta_1$ | Angle of first pendulum segment |
| Joint angle 2 | $\theta_2$ | Angle of second pendulum segment |
| Trigonometric encoding | $\sin\theta_1,\ \cos\theta_1,\ \sin\theta_2,\ \cos\theta_2$ | Circular-safe angular representation |
| Angular velocity | $\dot{\theta}$ | Time derivative of joint angles |
| Control input | $u$ | Applied force or torque |
| Linear acceleration | $\ddot{x}$ | Translational acceleration of the cart or base |

**Source:** Numerical simulation (e.g., OpenAI Gym `InvertedDoublePendulum-v4`, MuJoCo, or custom ODE integrator).

**System class:** Underactuated nonlinear mechanical system with chaotic dynamics for large angular displacements. Described by a 4-DoF Lagrangian with coupled, non-integrable equations of motion.

---

## Theoretical Motivation

The double inverted pendulum is governed by the Euler-Lagrange equations:

$$\frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \dot{q}_i}\right) - \frac{\partial \mathcal{L}}{\partial q_i} = \tau_i, \quad i = 1, 2$$

where $\mathcal{L} = T - V$ is the Lagrangian, $q = [\theta_1, \theta_2]^T$ are the generalized coordinates, and $\tau$ is the generalized force vector. The mass matrix $M(q)$ is configuration-dependent and the Coriolis/centrifugal matrix $C(q, \dot{q})$ introduces velocity-quadratic coupling terms, making the system inherently nonlinear.

For large displacements, the system exhibits **sensitive dependence on initial conditions** (positive maximal Lyapunov exponent $\lambda_{\max} > 0$), which has direct implications for dataset representativeness and model generalization.

---

## Hypothesis 1 — Stationarity and Ergodicity of the Dynamical Process

### Motivation

Neural network training implicitly assumes that the joint distribution $P(x_t, y_t)$ is stationary across the dataset. If the time series exhibits trends, seasonality, or non-constant variance (heteroscedasticity), the train/test split is statistically invalid and the learned model suffers from temporal data leakage.

### Formal Statement

$$H_0: \mathbb{E}[x_t] = \mu \text{ and } \mathrm{Var}[x_t] = \sigma^2, \quad \forall t \in [0, T]$$

$$H_1: \mu(t) \text{ and/or } \sigma^2(t) \text{ are time-varying}$$

### Statistical Tests

**Augmented Dickey-Fuller (ADF) test** — tests for a unit root (random walk) as evidence of non-stationarity:

$$\Delta x_t = \alpha + \beta t + \gamma x_{t-1} + \sum_{j=1}^{p} \delta_j \Delta x_{t-j} + \varepsilon_t$$

$H_0^{\text{ADF}}$: $\gamma = 0$ (unit root present). Reject if $p$-value $< 0.05$.

**KPSS test** — complementary to ADF; null hypothesis is *stationarity*:

$$x_t = \xi t + r_t + \varepsilon_t, \quad r_t = r_{t-1} + u_t$$

The KPSS statistic is:

$$\text{KPSS} = \frac{1}{n^2} \sum_{t=1}^{n} \frac{S_t^2}{\hat{\sigma}^2_\varepsilon}$$

where $S_t = \sum_{i=1}^t \hat{e}_i$ is the partial sum of residuals.

> A **conflicting result** (ADF rejects, KPSS does not reject, or vice versa) indicates trend-stationarity rather than strict stationarity — a common finding in controlled simulations near equilibrium.

**Rolling statistics** — visual and quantitative check:

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

def stationarity_report(series: pd.Series, variable_name: str, window: int = 50):
    # ADF test
    adf_result = adfuller(series.dropna(), autolag='AIC')
    print(f"[{variable_name}] ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")
    
    # KPSS test
    kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    print(f"[{variable_name}] KPSS Statistic: {kpss_result[0]:.4f}, p-value: {kpss_result[1]:.4f}")
    
    # Rolling mean and std
    rolling_mean = series.rolling(window).mean()
    rolling_std  = series.rolling(window).std()
    
    return {
        'adf_stat': adf_result[0], 'adf_pval': adf_result[1],
        'kpss_stat': kpss_result[0], 'kpss_pval': kpss_result[1],
        'rolling_mean': rolling_mean, 'rolling_std': rolling_std
    }
```

### Decision Criteria and Consequences

| Outcome | Interpretation | Action |
|---|---|---|
| $H_0$ not rejected (ADF + KPSS agree) | Series is stationary | Proceed with standard train/val/test split |
| $H_0$ rejected — trend present | Non-stationary | Apply first-difference $\Delta x_t$ or detrend |
| $H_0$ rejected — variance drift | Heteroscedastic | Apply rolling normalization or log-transform |

**Applicability scope:** Both Finalidades 1 and 2.

---

## Hypothesis 2 — Informational Sufficiency for System Identification

### Motivation

For Finalidade 1, the network must infer from raw behavioral data that the source system is a double inverted pendulum (as opposed to a simple pendulum, cart-pole, or other underactuated system). This requires that the dataset contain sufficient discriminative information — specifically, the **mutual information** between the feature set $X$ and the system class label $C$ must exceed a minimum threshold.

The double inverted pendulum's chaotic attractor in the $(\theta_1, \theta_2, \dot\theta_1, \dot\theta_2)$ space is geometrically distinct from other mechanical systems and constitutes the primary discriminative signature.

### Formal Statement

$$H_0: I(X;\, C) \geq I_{\min} \quad \text{(features are discriminative)}$$

$$H_1: I(X;\, C) < I_{\min} \quad \text{(features are insufficient for classification)}$$

where $I(X; C) = \sum_{x,c} P(x,c) \log \frac{P(x,c)}{P(x)P(c)}$ is the mutual information.

### Lyapunov Exponent as a System Fingerprint

The maximal Lyapunov exponent $\lambda_{\max}$ quantifies the exponential rate of divergence of nearby trajectories:

$$\lambda_{\max} = \lim_{t \to \infty} \frac{1}{t} \ln \frac{\|\delta Z(t)\|}{\|\delta Z(0)\|}$$

For the double inverted pendulum, $\lambda_{\max} > 0$ in chaotic regimes. This is a **system-level invariant** that a neural network can implicitly learn to recognize.

### Statistical Tests

**Mutual Information estimation** (continuous variables via $k$-NN estimator):

```python
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
import numpy as np

def mutual_info_analysis(X: np.ndarray, y: np.ndarray, feature_names: list):
    mi_scores = mutual_info_regression(X, y, random_state=42)
    for name, score in zip(feature_names, mi_scores):
        print(f"  MI({name}; target) = {score:.4f} nats")
    return mi_scores

def pca_variance_analysis(X: np.ndarray, threshold: float = 0.95):
    pca = PCA()
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumvar, threshold) + 1
    print(f"  Components for {threshold*100:.0f}% variance: {n_components} / {X.shape[1]}")
    return pca.explained_variance_ratio_, n_components
```

**Lyapunov exponent estimation** via Rosenstein algorithm:

```python
from nolds import lyap_r  # pip install nolds

def estimate_lyapunov(time_series: np.ndarray, emb_dim: int = 5, lag: int = 1):
    lmax = lyap_r(time_series, emb_dim=emb_dim, lag=lag, min_tsep=10)
    print(f"  Maximal Lyapunov exponent: λ_max = {lmax:.5f}")
    print(f"  System is {'chaotic' if lmax > 0 else 'non-chaotic'}")
    return lmax
```

### Decision Criteria and Consequences

| Outcome | Interpretation | Action |
|---|---|---|
| $I(X; C)$ large + $\lambda_{\max} > 0$ confirmed | Dataset is discriminative; identification feasible | Proceed to Finalidade 1 architecture design |
| Low MI, high PCA compression | Features are redundant or uninformative | Add derivative features $\ddot\theta$ or energy $E = T + V$ |
| $\lambda_{\max} \leq 0$ | Simulation constrained to near-equilibrium only | Expand trajectory space; include diverging trajectories |

**Applicability scope:** Finalidade 1 (system identification).

---

## Hypothesis 3 — Local Linearity vs. Global Nonlinearity of the Phase Space

### Motivation

This hypothesis formally justifies the choice of a nonlinear function approximator (neural network) over linear models (LTI systems, linear regression, ARMA). If the state-to-output mapping is globally linear, a neural network provides no benefit over a Kalman filter or linear MPC.

### Formal Statement

$$H_0: f(x_t) = Ax_t + b \quad \text{(linear model is sufficient)}$$

$$H_1: f(x_t) \text{ is nonlinear} \quad \text{(universal approximation required)}$$

### Theoretical Basis — Takens' Embedding Theorem

For a deterministic dynamical system with attractor of dimension $d$, a delay embedding of dimension $m \geq 2d + 1$ reconstructs a diffeomorphic copy of the attractor. The double inverted pendulum has state dimension 4 (or higher with the cart), implying embedding dimension $m \geq 9$.

If the reconstructed attractor is not topologically equivalent to a line or plane, the system is globally nonlinear — and $H_0$ is rejected.

### Statistical Tests

**BDS test** — tests for nonlinear dependence in residuals after linear filtering:

$$\text{BDS statistic} = \frac{\sqrt{n}\left[C_{m,\varepsilon} - C_{1,\varepsilon}^m\right]}{\sigma_{m,\varepsilon}} \xrightarrow{d} \mathcal{N}(0,1)$$

where $C_{m,\varepsilon}$ is the correlation integral at embedding dimension $m$ and radius $\varepsilon$.

```python
from statsmodels.stats.diagnostic import linear_reset
from arch.unitroot.cointegration import engle_granger
import statsmodels.api as sm

def nonlinearity_test(X: np.ndarray, y: np.ndarray):
    # Fit linear model and check residuals
    model = sm.OLS(y, sm.add_constant(X)).fit()
    reset = linear_reset(model, power=3, use_f=True)
    print(f"  RESET test F-statistic: {reset.fvalue:.4f}, p-value: {reset.pvalue:.4f}")
    print(f"  {'Nonlinearity detected' if reset.pvalue < 0.05 else 'Cannot reject linearity'}")
    return reset
```

**Correlation dimension** via Grassberger-Procaccia algorithm:

```python
from nolds import corr_dim

def correlation_dimension(time_series: np.ndarray, emb_dim: int = 7):
    cd = corr_dim(time_series, emb_dim=emb_dim)
    print(f"  Correlation dimension D₂ = {cd:.4f}")
    print(f"  Non-integer D₂ indicates fractal attractor (strongly nonlinear)")
    return cd
```

**Minimum embedding dimension** via false nearest neighbors (FNN):

```python
# Using nolds or custom implementation
from nolds import fnn  # if available, else implement manually

def false_nearest_neighbors(time_series: np.ndarray, max_dim: int = 10, lag: int = 1):
    fnn_rates = []
    for d in range(1, max_dim + 1):
        # FNN rate at dimension d
        # Rate drops to ~0 at the correct embedding dimension
        pass
    return fnn_rates
```

### Decision Criteria and Consequences

| Outcome | Interpretation | Action |
|---|---|---|
| RESET $p < 0.05$, $D_2$ non-integer | System is globally nonlinear | Justified use of deep neural networks |
| Non-integer $D_2 \approx 2$–$3$ | Low-dimensional chaotic attractor | Architecture: 3–5 hidden layers is sufficient |
| $D_2$ integer (e.g., $D_2 = 1$) | System is limit-cycle or equilibrium | Consider RNN/LSTM only if temporal memory needed |

**Applicability scope:** Both Finalidades 1 and 2.

---

## Hypothesis 4 — Phase Space Coverage and Dataset Representativeness

### Motivation

A neural network generalizes only to regions of state space encountered during training. For a chaotic system like the double inverted pendulum, the attractor may occupy a high-dimensional submanifold — if the dataset samples only trajectories near the upright equilibrium, the network will fail catastrophically when presented with large-amplitude or transient dynamics.

This hypothesis tests whether the empirical distribution $P_{\text{data}}$ adequately approximates the true invariant measure $P_{\text{real}}$ of the system.

### Formal Statement

$$H_0: D_{\mathrm{KL}}\!\left(P_{\text{data}} \,\|\, P_{\text{real}}\right) \leq \varepsilon \quad \text{(dataset is representative)}$$

$$H_1: D_{\mathrm{KL}}\!\left(P_{\text{data}} \,\|\, P_{\text{real}}\right) > \varepsilon \quad \text{(critical undersampling)}$$

where $D_{\mathrm{KL}}$ is the Kullback-Leibler divergence and $\varepsilon$ is a problem-specific tolerance.

### Statistical Tests

**Kernel Density Estimation (KDE) on the $(\theta_1, \theta_2)$ subspace:**

```python
from scipy.stats import gaussian_kde, ks_2samp
import numpy as np

def phase_space_coverage(theta1: np.ndarray, theta2: np.ndarray,
                          theta1_ref: np.ndarray = None, theta2_ref: np.ndarray = None):
    # KDE of observed data
    X = np.vstack([theta1, theta2])
    kde = gaussian_kde(X, bw_method='scott')
    
    # Evaluate over a grid
    g1 = np.linspace(theta1.min(), theta1.max(), 100)
    g2 = np.linspace(theta2.min(), theta2.max(), 100)
    G1, G2 = np.meshgrid(g1, g2)
    density = kde(np.vstack([G1.ravel(), G2.ravel()])).reshape(G1.shape)
    
    # KS test between datasets if reference available
    if theta1_ref is not None:
        ks_stat, ks_pval = ks_2samp(theta1, theta1_ref)
        print(f"  KS test θ₁: statistic={ks_stat:.4f}, p-value={ks_pval:.4f}")
    
    return density, kde
```

**Recurrence Plot analysis** — visualizes return statistics of the trajectory:

```python
def recurrence_plot(state_sequence: np.ndarray, threshold: float = 0.1):
    """
    Computes the recurrence matrix R[i,j] = Θ(ε - ||x_i - x_j||)
    High recurrence rate in all regions → good coverage
    Sparse diagonal structures → trajectory does not revisit state-space regions
    """
    n = len(state_sequence)
    dist_matrix = np.linalg.norm(
        state_sequence[:, None, :] - state_sequence[None, :, :], axis=-1
    )
    R = (dist_matrix < threshold).astype(float)
    recurrence_rate = R.sum() / (n * n)
    print(f"  Recurrence rate: {recurrence_rate:.4f}")
    print(f"  {'Good coverage' if recurrence_rate > 0.05 else 'Sparse coverage — consider augmentation'}")
    return R, recurrence_rate
```

**Approximate entropy (ApEn)** as a coverage regularity metric:

```python
from nolds import sampen  # Sample entropy

def approximate_entropy_analysis(time_series: np.ndarray, m: int = 2, r_factor: float = 0.2):
    r = r_factor * np.std(time_series)
    samp_en = sampen(time_series, emb_dim=m, tolerance=r)
    print(f"  Sample Entropy (SampEn): {samp_en:.4f}")
    print(f"  Higher values → more complex/irregular dynamics")
    return samp_en
```

### Decision Criteria and Consequences

| Outcome | Interpretation | Action |
|---|---|---|
| High KDE coverage, KS $p > 0.05$ | Distribution is representative | Proceed to model training |
| Sparse KDE near $\theta \approx \pm\pi$ | Large-angle dynamics undersampled | Run additional simulations from random ICs |
| Low recurrence rate | Trajectory poorly revisits state space | Augment with time-reversed or perturbed trajectories |
| Very high SampEn | Highly irregular — possible noise contamination | Apply median filtering or check simulator numerical precision |

**Applicability scope:** Both Finalidades 1 and 2.

---

## Hypothesis 5 — Observability and Absence of Critical Multicollinearity

### Motivation

The feature set includes both raw angles $(\theta_1, \theta_2)$ and their trigonometric representations $(\sin\theta_1, \cos\theta_1, \sin\theta_2, \cos\theta_2)$. By the Pythagorean identity, these satisfy:

$$\sin^2\theta_i + \cos^2\theta_i = 1, \quad i = 1, 2$$

This introduces **exact structural multicollinearity** in the original angle space. While the trigonometric encoding is theoretically preferable (it avoids angle wrapping discontinuities and provides a smooth manifold embedding), the joint inclusion of $\theta$ and $\{\sin\theta, \cos\theta\}$ may cause pathological conditioning in the input layer's weight matrix during gradient-based optimization.

### Formal Statement

$$H_0: \kappa(\mathbf{X}) \leq \kappa_{\max} \quad \text{(condition number within acceptable bounds)}$$

$$H_1: \kappa(\mathbf{X}) > \kappa_{\max} \quad \text{(critical multicollinearity — ill-conditioned input)}$$

where $\kappa(\mathbf{X}) = \sigma_{\max} / \sigma_{\min}$ is the ratio of largest to smallest singular values of the feature matrix $\mathbf{X}$.

### Mathematical Analysis of the Structural Redundancy

Let $\mathbf{x} = [\theta_1, \theta_2, \sin\theta_1, \cos\theta_1, \sin\theta_2, \cos\theta_2, \dot\theta, u, \ddot x]^T$.

The Jacobian of the constraint $g(\mathbf{x}) = \sin^2\theta_i + \cos^2\theta_i - 1 = 0$ has rank 2 over the full feature space, meaning the effective dimensionality of the input is at most $9 - 2 = 7$.

The **Variance Inflation Factor** for feature $j$ is:

$$\mathrm{VIF}_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the coefficient of determination of a regression of $x_j$ on all remaining features. $\mathrm{VIF}_j > 10$ is conventionally considered severe multicollinearity.

### Statistical Tests

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

def condition_number_analysis(X: np.ndarray):
    X_scaled = StandardScaler().fit_transform(X)
    singular_values = np.linalg.svd(X_scaled, compute_uv=False)
    kappa = singular_values.max() / singular_values.min()
    print(f"  Condition number κ(X) = {kappa:.2f}")
    if kappa < 30:
        print("  Well-conditioned (κ < 30)")
    elif kappa < 100:
        print("  Moderate multicollinearity (30 ≤ κ < 100)")
    else:
        print("  Severe multicollinearity (κ ≥ 100) — consider feature pruning")
    return kappa, singular_values

def vif_analysis(X: np.ndarray, feature_names: list):
    vif_results = {}
    for i, name in enumerate(feature_names):
        y_i = X[:, i]
        X_rest = np.delete(X, i, axis=1)
        model = sm.OLS(y_i, sm.add_constant(X_rest)).fit()
        r2 = model.rsquared
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        vif_results[name] = vif
        print(f"  VIF({name}) = {vif:.2f}")
    return vif_results

def correlation_matrix(X: np.ndarray, feature_names: list):
    corr = np.corrcoef(X.T)
    df_corr = pd.DataFrame(corr, index=feature_names, columns=feature_names)
    # Flag pairs with |r| > 0.95
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr[i, j]) > 0.95:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr[i, j]))
    return df_corr, high_corr_pairs
```

### Expected Findings and Recommendations

Given the structural constraint $\sin^2\theta + \cos^2\theta = 1$, the following VIF pattern is expected:

| Feature | Expected VIF | Interpretation |
|---|---|---|
| $\theta_1$ | High (> 10) | Redundant with $\sin\theta_1, \cos\theta_1$ |
| $\sin\theta_1$ | Moderate–High | Partially redundant with $\theta_1$ |
| $\cos\theta_1$ | Moderate–High | Partially redundant with $\theta_1$ |
| $\dot\theta$ | Low (< 5) | Independent dynamical information |
| $u$ | Low (< 5) | Control input — exogenous |
| $\ddot{x}$ | Moderate | Correlated with $u$ through dynamics |

### Recommended Feature Strategy

**Option A — Trigonometric-only (recommended for neural networks):**

$$\mathbf{x}_{\text{trig}} = [\sin\theta_1, \cos\theta_1, \sin\theta_2, \cos\theta_2, \dot\theta_1, \dot\theta_2, u, \ddot x]$$

Rationale: eliminates the angle wrapping discontinuity at $\pm\pi$, maps the state onto a smooth compact manifold $\mathbb{S}^1 \times \mathbb{S}^1$, and removes the redundant $\theta_i$ columns.

**Option B — PCA-whitened input:**

Apply PCA to the full feature matrix, retaining components with explained variance ratio $> 0.01$. This decorrelates the input and resolves multicollinearity at the cost of interpretability.

$$\mathbf{x}_{\text{PCA}} = \mathbf{W}^T (\mathbf{x} - \bar{\mathbf{x}}), \quad \mathbf{W} \in \mathbb{R}^{9 \times k}$$

### Decision Criteria and Consequences

| Outcome | Interpretation | Action |
|---|---|---|
| $\kappa < 30$, all VIF $< 10$ | Well-conditioned input | No feature engineering required |
| $\kappa \geq 100$, VIF$(\theta_i) \gg 10$ | Structural redundancy confirmed | Drop raw $\theta_i$; keep only trig encoding |
| VIF$(u)$ or VIF$(\ddot x) > 10$ | Control–dynamics coupling | Consider removing $\ddot x$ or computing residuals |

**Applicability scope:** Both Finalidades 1 and 2.

---

## Summary Table

| # | Hypothesis | Null ($H_0$) | Primary Test | Scope |
|---|---|---|---|---|
| H1 | Stationarity | $\mu, \sigma^2$ constant over time | ADF + KPSS | F1 + F2 |
| H2 | Informational sufficiency | $I(X; C) \geq I_{\min}$ | Mutual Information + Lyapunov | F1 |
| H3 | Nonlinearity | $f(x_t) = Ax_t + b$ is insufficient | BDS test + Correlation Dimension | F1 + F2 |
| H4 | Phase space coverage | $D_{\mathrm{KL}}(P_{\text{data}} \| P_{\text{real}}) \leq \varepsilon$ | KDE + Recurrence Plot | F1 + F2 |
| H5 | Feature observability | $\kappa(\mathbf{X}) \leq \kappa_{\max}$ | VIF + Condition Number | F1 + F2 |

**F1** = System identification (predictive classification network)  
**F2** = Neural network-based control

---

## Recommended Execution Pipeline

```python
import numpy as np
import pandas as pd

# --- Load dataset ---
# Expected columns: theta1, theta2, sin_t1, cos_t1, sin_t2, cos_t2, dtheta, u, ddx
df = pd.read_csv("double_pendulum_data.csv")

feature_names = ['theta1', 'theta2', 'sin_t1', 'cos_t1',
                 'sin_t2', 'cos_t2', 'dtheta', 'u', 'ddx']
X = df[feature_names].values

# --- H1: Stationarity ---
for col in feature_names:
    stationarity_report(df[col], col, window=100)

# --- H2: Informational sufficiency ---
# (Requires a target label y for MI; use dtheta or energy as proxy)
mi_scores = mutual_info_analysis(X, df['dtheta'].values, feature_names)
pca_var, n_comp = pca_variance_analysis(X)
lambda_max = estimate_lyapunov(df['theta1'].values, emb_dim=5, lag=1)

# --- H3: Nonlinearity ---
nonlinearity_test(X, df['dtheta'].values)
cd = correlation_dimension(df['theta1'].values, emb_dim=7)

# --- H4: Phase space coverage ---
density, kde = phase_space_coverage(df['theta1'].values, df['theta2'].values)
state_seq = df[['theta1', 'theta2', 'dtheta']].values
R, rr = recurrence_plot(state_seq, threshold=0.15)
samp_en = approximate_entropy_analysis(df['theta1'].values)

# --- H5: Multicollinearity ---
kappa, svs = condition_number_analysis(X)
vif_scores   = vif_analysis(X, feature_names)
corr_df, high_corr = correlation_matrix(X, feature_names)

print("\nHigh-correlation pairs (|r| > 0.95):")
for pair in high_corr:
    print(f"  {pair[0]} — {pair[1]}: r = {pair[2]:.4f}")
```

---

## Dependencies

```txt
numpy>=1.24
pandas>=2.0
scipy>=1.10
statsmodels>=0.14
scikit-learn>=1.3
nolds>=0.5.2        # Lyapunov exponent, correlation dimension, sample entropy
arch>=6.2           # BDS test (via arch.unitroot or statsmodels)
matplotlib>=3.7     # Visualization of KDE, recurrence plots, rolling stats
seaborn>=0.12       # Correlation heatmap
```

---

## References

1. Takens, F. (1981). *Detecting strange attractors in turbulence*. Lecture Notes in Mathematics, 898, 366–381.
2. Grassberger, P., & Procaccia, I. (1983). Characterization of strange attractors. *Physical Review Letters*, 50(5), 346.
3. Brock, W. A., Dechert, W. D., & Scheinkman, J. A. (1987). *A test for independence based on the correlation dimension*. SSRN Working Paper.
4. Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993). A practical method for calculating largest Lyapunov exponents from small data sets. *Physica D*, 65(1–2), 117–134.
5. Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *Journal of Econometrics*, 54(1–3), 159–178.
6. Kullback, S., & Leibler, R. A. (1951). On information and sufficiency. *Annals of Mathematical Statistics*, 22(1), 79–86.
7. Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. *Physics Reports*, 438(5–6), 237–329.
8. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.
