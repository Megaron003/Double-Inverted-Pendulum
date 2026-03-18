import csv
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = r"C:\Users\Guilherme\Mestrado\Invertede Pendulum\Inverted Pendulum\Final Versions\Data Processed\pendulum_dataset_tidy_non_ZMP_Velocities.csv"

MAX_STEPS = 50
MIN_DISTANCE = 1e-10
MAX_LAG = 300  # limite físico para τ

# =========================
# LEITURA DOS DADOS
# =========================
data = []
with open(FILE_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

episodes = sorted(set(int(r["episode"]) for r in data))
print(f"Episódios encontrados: {len(episodes)}")

lambdas = []

# =========================
# LOOP POR EPISÓDIO
# =========================
for ep in episodes:

    print(f"\n===== EPISÓDIO {ep} =====")

    ep_data = [r for r in data if int(r["episode"]) == ep]

    time = np.array([float(r["time"]) for r in ep_data])
    sin1 = np.array([float(r["sin_theta1"]) for r in ep_data])
    cos1 = np.array([float(r["cos_theta1"]) for r in ep_data])
    sin2 = np.array([float(r["sin_theta2"]) for r in ep_data])
    cos2 = np.array([float(r["cos_theta2"]) for r in ep_data])
    omega1 = np.array([float(r["omega1"]) for r in ep_data])
    omega2 = np.array([float(r["omega2"]) for r in ep_data])

    # =========================
    # RECONSTRUÇÃO θ
    # =========================
    theta1 = np.arctan2(sin1, cos1)

    # normalização
    theta1 = (theta1 - np.mean(theta1)) / np.std(theta1)

    # =========================
    # INFORMAÇÃO MÚTUA
    # =========================
    def mutual_information(x, lag, bins=64):
        x1 = x[:-lag]
        x2 = x[lag:]

        H, _, _ = np.histogram2d(x1, x2, bins=bins)
        Pxy = H / np.sum(H)

        Px = np.sum(Pxy, axis=1)
        Py = np.sum(Pxy, axis=0)

        mi = 0
        for i in range(len(Px)):
            for j in range(len(Py)):
                if Pxy[i, j] > 0:
                    mi += Pxy[i, j] * np.log(Pxy[i, j] / (Px[i]*Py[j] + 1e-12))
        return mi

    lags = np.arange(1, min(len(theta1)//4, MAX_LAG))
    mi_vals = np.array([mutual_information(theta1, l) for l in lags])

    # τ = primeiro mínimo local
    tau = None
    for i in range(1, len(mi_vals)-1):
        if mi_vals[i] < mi_vals[i-1] and mi_vals[i] < mi_vals[i+1]:
            tau = lags[i]
            break

    if tau is None:
        tau = lags[np.argmin(mi_vals)]

    print(f"τ ótimo = {tau}")

    # =========================
    # ESTADOS COMPLETOS (ROBUSTO)
    # =========================
    states = np.column_stack((cos1, sin1, cos2, sin2, omega1, omega2))

    # =========================
    # ROSENSTEIN
    # =========================
    log_div = np.zeros(MAX_STEPS)
    count = np.zeros(MAX_STEPS)

    for i in range(len(states) - MAX_STEPS):

        dists = np.linalg.norm(states - states[i], axis=1)

        # Theiler window
        dists[max(0, i - tau):i + tau] = np.inf

        j = np.argmin(dists)

        if dists[j] == np.inf:
            continue

        for k in range(MAX_STEPS):
            if i + k >= len(states) or j + k >= len(states):
                break

            dist = np.linalg.norm(states[i+k] - states[j+k])
            dist = max(dist, MIN_DISTANCE)

            log_div[k] += np.log(dist)
            count[k] += 1

    valid = count > 0
    avg_log = log_div[valid] / count[valid]
    t_vals = np.arange(len(avg_log))

    # =========================
    # REGIÃO LINEAR AUTOMÁTICA
    # =========================
    best_r2 = -np.inf
    best_lambda = 0

    min_window = 5
    max_window = int(len(t_vals) * 0.5)

    for w in range(min_window, max_window):
        coef = np.polyfit(t_vals[:w], avg_log[:w], 1)
        fit = np.polyval(coef, t_vals[:w])

        ss_res = np.sum((avg_log[:w] - fit) ** 2)
        ss_tot = np.sum((avg_log[:w] - np.mean(avg_log[:w])) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf

        if r2 > best_r2:
            best_r2 = r2
            best_lambda = coef[0]
            best_w = w

    lambdas.append(best_lambda)

    print(f"λ = {best_lambda:.6f} | R² = {best_r2:.4f}")

    # =========================
    # PLOT
    # =========================
    plt.figure()
    plt.plot(t_vals, avg_log, label="ln(d(t))")
    plt.plot(t_vals[:best_w],
             np.polyval(np.polyfit(t_vals[:best_w], avg_log[:best_w], 1), t_vals[:best_w]),
             '--', label=f"λ={best_lambda:.4f}")
    plt.legend()
    plt.grid()
    plt.title(f"Lyapunov - Episódio {ep}")
    plt.savefig(f"lyapunov_ep{ep}.png", dpi=300)

# =========================
# RESULTADO GLOBAL
# =========================
print("\n===== RESULTADO GLOBAL =====")
print(f"λ médio = {np.mean(lambdas):.6f}")
print(f"desvio padrão = {np.std(lambdas):.6f}")