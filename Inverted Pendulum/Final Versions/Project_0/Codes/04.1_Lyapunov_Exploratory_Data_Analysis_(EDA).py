import csv
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = r"C:\Users\Guilherme\Mestrado\Invertede Pendulum\Inverted Pendulum\Final Versions\Data Processed\pendulum_dataset_tidy_ZMP_Velocities.csv"

MAX_STEPS = 100
MIN_DISTANCE = 1e-8

# =========================
# LEITURA DOS DADOS
# =========================
data = []
with open(FILE_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# agrupar episódios
episodes = {}
for r in data:
    ep = int(r["episode"])
    if ep not in episodes:
        episodes[ep] = []
    episodes[ep].append(r)

print(f"Episódios encontrados: {len(episodes)}")

# =========================
# FUNÇÕES AUXILIARES
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
            if Pxy[i,j] > 0:
                mi += Pxy[i,j] * np.log(Pxy[i,j] / (Px[i]*Py[j] + 1e-12))
    return mi

def takens(x, tau, m):
    N = len(x)
    M = N - (m-1)*tau
    return np.array([[x[i + j*tau] for j in range(m)] for i in range(M)])

# =========================
# PROCESSAR TODOS EPISÓDIOS
# =========================
all_lambdas = []

for ep_id, ep_data in episodes.items():

    print(f"\n--- Episódio {ep_id} ---")

    time = np.array([float(r["time"]) for r in ep_data])
    sin1 = np.array([float(r["sin_theta1"]) for r in ep_data])
    cos1 = np.array([float(r["cos_theta1"]) for r in ep_data])
    sin2 = np.array([float(r["sin_theta2"]) for r in ep_data])
    cos2 = np.array([float(r["cos_theta2"]) for r in ep_data])
    omega1 = np.array([float(r["omega1"]) for r in ep_data])
    omega2 = np.array([float(r["omega2"]) for r in ep_data])

    # reconstrução angular
    theta1 = np.arctan2(sin1, cos1)
    theta1 = (theta1 - np.mean(theta1)) / np.std(theta1)

    # =========================
    # ESTIMAR τ (informação mútua)
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
    # TESTAR DIFERENTES m
    # =========================
    lambdas_m = []

    for m in [2, 3, 4, 5]:

        embedded = takens(theta1, tau, m)

        # espaço de estados completo
        states = np.array([
            [cos1[i], sin1[i], cos2[i], sin2[i], omega1[i], omega2[i]]
            for i in range(len(embedded))
        ])

        log_div = np.zeros(MAX_STEPS)
        count = np.zeros(MAX_STEPS)

        for i in range(len(states) - MAX_STEPS):

            dists = np.linalg.norm(states - states[i], axis=1)

            # Theiler window
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
        # DETECÇÃO REGIÃO LINEAR
        # =========================
        best_lambda = None
        best_r2 = -np.inf

        window_sizes = range(10, int(len(t_vals)*0.5))

        for w in window_sizes:
            x = t_vals[:w]
            y = avg_log[:w]

            coef = np.polyfit(x, y, 1)
            fit = np.polyval(coef, x)

            # R²
            ss_res = np.sum((y - fit)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0

            if r2 > best_r2:
                best_r2 = r2
                best_lambda = coef[0]

        print(f"m={m} | λ={best_lambda:.6f} | R²={best_r2:.4f}")
        lambdas_m.append(best_lambda)

    # =========================
    # CONVERGÊNCIA EM m
    # =========================
    lambda_final = np.mean(lambdas_m[-2:])  # últimos embeddings
    print(f"λ final (episódio {ep_id}) = {lambda_final:.6f}")

    all_lambdas.append(lambda_final)

# =========================
# RESULTADO GLOBAL
# =========================
print("\n===== RESULTADO FINAL =====")

if all_lambdas:
    mean_lambda = np.mean(all_lambdas)
    std_lambda = np.std(all_lambdas)

    print(f"λ médio = {mean_lambda:.6f}")
    print(f"desvio padrão = {std_lambda:.6f}")

    if mean_lambda > 0:
        print("Sistema possivelmente caótico (λ > 0)")
    else:
        print("Sistema não caótico")

else:
    print("Não foi possível calcular λ")