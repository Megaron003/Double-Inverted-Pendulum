import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import mutual_info_score

# =========================
# CONFIGURAÇÕES
# =========================
FILE_PATH = r"C:\Users\Guilherme\Mestrado\Invertede Pendulum\Inverted Pendulum\Final Versions\Data Processed\pendulum_dataset_tidy_non_ZMP_Velocities.csv"  # <-- ajuste aqui
m = 3
MAX_STEPS = 100
MIN_DISTANCE = 1e-10

# =========================
# FUNÇÕES AUXILIARES
# =========================
def mutual_information(x, lag, bins=50):
    x1 = x[:-lag]
    x2 = x[lag:]
    c_xy = np.histogram2d(x1, x2, bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)

def takens(series, tau, m):
    N = len(series)
    M = N - (m - 1) * tau
    return np.array([series[i:i + m * tau:tau] for i in range(M)])

# =========================
# LEITURA DO CSV
# =========================
data = []
with open(FILE_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# =========================
# AGRUPAR POR EPISÓDIO
# =========================
episodes = {}
for r in data:
    ep = int(r["episode"])
    if ep not in episodes:
        episodes[ep] = []
    episodes[ep].append(r)

print(f"Episódios encontrados: {len(episodes)}")

all_lambdas = []

# =========================
# LOOP POR EPISÓDIO
# =========================
for ep_id, ep_data in episodes.items():

    print(f"\n===== EPISÓDIO {ep_id} =====")

    time = np.array([float(r["time"]) for r in ep_data])
    sin1 = np.array([float(r["sin_theta1"]) for r in ep_data])
    cos1 = np.array([float(r["cos_theta1"]) for r in ep_data])
    omega1 = np.array([float(r["omega1"]) for r in ep_data])
    omega2 = np.array([float(r["omega2"]) for r in ep_data])

    # =========================
    # RECONSTRUÇÃO DO ÂNGULO
    # =========================
    theta1 = np.arctan2(sin1, cos1)
    theta1 = (theta1 - np.mean(theta1)) / np.std(theta1)

    # =========================
    # CÁLCULO DE τ (Mutual Information)
    # =========================
    lags = np.arange(1, len(theta1)//4)
    mi_vals = np.array([mutual_information(theta1, l) for l in lags])

    tau = None
    for i in range(1, len(mi_vals)-1):
        if mi_vals[i] < mi_vals[i-1] and mi_vals[i] < mi_vals[i+1]:
            tau = lags[i]
            break

    if tau is None:
        tau = lags[np.argmin(mi_vals)]

    print(f"τ = {tau}")

    # =========================
    # EMBEDDING
    # =========================
    states = takens(theta1, tau, m)

    # =========================
    # ROSENSTEIN
    # =========================
    log_div = np.zeros(MAX_STEPS)
    count = np.zeros(MAX_STEPS)

    for i in range(len(states) - MAX_STEPS):

        dists = np.linalg.norm(states - states[i], axis=1)
        dists[max(0, i-tau):i+tau] = np.inf

        j = np.argmin(dists)

        if dists[j] == np.inf:
            continue

        for k in range(MAX_STEPS):
            if i+k >= len(states) or j+k >= len(states):
                break

            dist = np.linalg.norm(states[i+k] - states[j+k])
            dist = max(dist, MIN_DISTANCE)

            log_div[k] += np.log(dist)
            count[k] += 1

    valid = count > 0
    avg_log = log_div[valid] / count[valid]

    # =========================
    # TEMPO REAL
    # =========================
    dt = np.mean(np.diff(time))
    t_vals = np.arange(len(avg_log)) * dt

    # =========================
    # REGIÃO LINEAR AUTOMÁTICA
    # =========================
    best_r2 = -np.inf
    best_coef = None
    best_w = 0

    for w in range(20, int(len(t_vals)*0.5)):

        x = t_vals[:w]
        y = avg_log[:w]

        coef = np.polyfit(x, y, 1)

        if coef[0] <= 0:
            continue

        fit = np.polyval(coef, x)

        ss_res = np.sum((y - fit)**2)
        ss_tot = np.sum((y - np.mean(y))**2)

        if ss_tot == 0:
            continue

        r2 = 1 - ss_res/ss_tot

        if r2 > best_r2:
            best_r2 = r2
            best_coef = coef
            best_w = w

    lambda_est = best_coef[0]
    all_lambdas.append(lambda_est)

    print(f"λ = {lambda_est:.6f} | R² = {best_r2:.4f}")

    # =========================
    # PLOT ln(d(t))
    # =========================
    plt.figure(figsize=(8,5))
    plt.plot(t_vals, avg_log, label="ln(d(t))")
    plt.plot(t_vals[:best_w],
             np.polyval(best_coef, t_vals[:best_w]),
             linestyle="--",
             label="Região linear")
    plt.title(f"ln(d(t)) - Episódio {ep_id}")
    plt.xlabel("Tempo (s)")
    plt.ylabel("ln(d)")
    plt.legend()
    plt.grid()
    plt.savefig(f"ln_dt_ep_{ep_id}.png", dpi=300)

    # =========================
    # FTLE
    # =========================
    window_size = 30
    ftle = []
    ftle_time = []

    for i in range(len(t_vals) - window_size):
        x = t_vals[i:i+window_size]
        y = avg_log[i:i+window_size]
        coef = np.polyfit(x, y, 1)
        ftle.append(coef[0])
        ftle_time.append(x[0])

    plt.figure(figsize=(8,5))
    plt.plot(ftle_time, ftle)
    plt.axhline(lambda_est, linestyle="--", label="λ global")
    plt.title(f"FTLE - Episódio {ep_id}")
    plt.xlabel("Tempo (s)")
    plt.ylabel("λ(t)")
    plt.legend()
    plt.grid()
    plt.savefig(f"ftle_ep_{ep_id}.png", dpi=300)

# =========================
# RESULTADO GLOBAL
# =========================
print("\n===== RESULTADO GLOBAL =====")

mean_lambda = np.mean(all_lambdas)
std_lambda = np.std(all_lambdas)

print(f"λ médio = {mean_lambda:.6f}")
print(f"desvio padrão = {std_lambda:.6f}")

if mean_lambda > 0:
    print("Sistema caótico (consistente entre episódios)")
else:
    print("Sistema não caótico")