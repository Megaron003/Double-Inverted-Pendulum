import csv
import numpy as np
import matplotlib.pyplot as plt
import os

FILE_PATH = r"C:\Users\Guilherme\Mestrado\Invertede Pendulum\Inverted Pendulum\Final Versions\Data Processed\pendulum_dataset_tidy_non_ZMP_Velocities.csv"

k1 = 1.0
k2 = 1.0
MAX_STEPS = 50
MIN_DISTANCE = 1e-8

# =========================
# CRIAÇÃO DE PASTAS
# =========================
BASE_DIR = "results"

folders = {
    "lyapunov": "lyapunov",
    "ftle": "ftle",
    "ln_dt": "ln_dt",
    "mi": "mutual_information",
    "takens": "takens",
    "V": "lyapunov_function"
}

for f in folders.values():
    os.makedirs(os.path.join(BASE_DIR, f), exist_ok=True)

# =========================
# LEITURA DOS DADOS
# =========================
data = []
with open(FILE_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# =========================
# AGRUPAMENTO POR EPISÓDIO
# =========================
episodes = {}
for r in data:
    ep = int(r["episode"])
    episodes.setdefault(ep, []).append(r)

print(f"Episódios encontrados: {len(episodes)}")

all_lambdas = []

# =========================
# FUNÇÕES
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

def takens(x, tau, m):
    N = len(x)
    M = N - (m - 1) * tau
    return np.array([[x[i + j*tau] for j in range(m)] for i in range(M)])

# =========================
# LOOP PRINCIPAL
# =========================
for ep_id, ep_data in episodes.items():

    print(f"\n===== EPISÓDIO {ep_id} =====")

    time = np.array([float(r["time"]) for r in ep_data])
    sin1 = np.array([float(r["sin_theta1"]) for r in ep_data])
    cos1 = np.array([float(r["cos_theta1"]) for r in ep_data])
    sin2 = np.array([float(r["sin_theta2"]) for r in ep_data])
    cos2 = np.array([float(r["cos_theta2"]) for r in ep_data])
    omega1 = np.array([float(r["omega1"]) for r in ep_data])
    omega2 = np.array([float(r["omega2"]) for r in ep_data])

    # =========================
    # NORMALIZAÇÃO
    # =========================
    theta1 = np.arctan2(sin1, cos1)
    theta1 = (theta1 - np.mean(theta1)) / np.std(theta1)

    # =========================
    # FUNÇÃO DE LYAPUNOV
    # =========================
    V = 0.5*(omega1**2 + omega2**2) + k1*(1-cos1) + k2*(1-cos2)
    Vdot = np.diff(V) / np.diff(time)

    plt.figure()
    plt.plot(time, V)
    plt.xlabel("Tempo (s)")
    plt.ylabel(r"$V(t)$")
    plt.title(f"V(t) - Episódio {ep_id}")
    plt.grid()
    plt.savefig(f"{BASE_DIR}/{folders['V']}/V_ep{ep_id}.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(time[:-1], Vdot)
    plt.xlabel("Tempo (s)")
    plt.ylabel(r"$\dot{V}(t)$")
    plt.title(f"Vdot(t) - Episódio {ep_id}")
    plt.grid()
    plt.savefig(f"{BASE_DIR}/{folders['V']}/Vdot_ep{ep_id}.png", dpi=300)
    plt.close()

    # =========================
    # INFORMAÇÃO MÚTUA (τ CORRIGIDO)
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

    print(f"τ ótimo = {tau}")

    plt.figure()
    plt.plot(lags, mi_vals)
    plt.axvline(tau, linestyle='--')
    plt.xlabel(r"Atraso $\tau$")
    plt.ylabel("Informação Mútua")
    plt.title(f"MI - Episódio {ep_id}")
    plt.grid()
    plt.savefig(f"{BASE_DIR}/{folders['mi']}/mi_ep{ep_id}.png", dpi=300)
    plt.close()

    # =========================
    # TAKENS
    # =========================
    embedded = takens(theta1, tau, 3)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(embedded[:,0], embedded[:,1], embedded[:,2])
    ax.set_xlabel(r"$x(t)$")
    ax.set_ylabel(r"$x(t+\tau)$")
    ax.set_zlabel(r"$x(t+2\tau)$")
    plt.title(f"Atrator - Episódio {ep_id}")
    plt.savefig(f"{BASE_DIR}/{folders['takens']}/takens_ep{ep_id}.png", dpi=300)
    plt.close()

    # =========================
    # ESPAÇO DE ESTADOS
    # =========================
    states = np.array([
        [cos1[i], sin1[i], cos2[i], sin2[i], omega1[i], omega2[i]]
        for i in range(len(cos1))
    ])

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
    # FTLE
    # =========================
    t_vals = np.arange(len(avg_log))
    ftle = avg_log / (t_vals + 1e-8)

    plt.figure()
    plt.plot(t_vals[1:], ftle[1:])
    plt.xlabel(r"Passos $k$")
    plt.ylabel(r"$\lambda(k)$")
    plt.title(f"FTLE - Episódio {ep_id}")
    plt.grid()
    plt.savefig(f"{BASE_DIR}/{folders['ftle']}/ftle_ep{ep_id}.png", dpi=300)
    plt.close()

    # =========================
    # REGIÃO LINEAR
    # =========================
    max_r2 = -np.inf
    best_lambda = 0
    best_end = 10

    for end in range(10, int(len(t_vals)*0.3)):

        x = t_vals[:end]
        y = avg_log[:end]

        coef = np.polyfit(x, y, 1)
        y_pred = np.polyval(coef, x)

        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)

        r2 = 1 - ss_res/(ss_tot + 1e-12)

        if r2 > max_r2:
            max_r2 = r2
            best_lambda = coef[0]
            best_end = end

    lambda_est = best_lambda
    all_lambdas.append(lambda_est)

    print(f"λ = {lambda_est:.6f} | R² = {max_r2:.4f}")

    # =========================
    # LN(d(t))
    # =========================
    plt.figure()
    plt.plot(t_vals, avg_log, label="ln(d(t))")

    coef = np.polyfit(t_vals[:best_end], avg_log[:best_end], 1)

    plt.plot(t_vals, np.polyval(coef, t_vals), '--',
             label=f"λ={lambda_est:.6f}")

    plt.axvline(t_vals[best_end], linestyle='--',
                label='Região linear')

    plt.xlabel(r"Passos $k$")
    plt.ylabel(r"$\ln(d(k))$")
    plt.title(f"Lyapunov - Episódio {ep_id}")
    plt.legend()
    plt.grid()

    plt.savefig(f"{BASE_DIR}/{folders['ln_dt']}/ln_dt_ep{ep_id}.png", dpi=300)
    plt.close()

# =========================
# RESULTADO GLOBAL
# =========================
print("\n===== RESULTADO GLOBAL =====")

mean_lambda = np.mean(all_lambdas)
std_lambda = np.std(all_lambdas)

print(f"λ médio = {mean_lambda:.6f}")
print(f"desvio padrão = {std_lambda:.6f}")