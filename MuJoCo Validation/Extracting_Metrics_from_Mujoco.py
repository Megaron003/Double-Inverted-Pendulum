import mujoco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────
XML_PATH    = r"C:\Users\sanch\Downloads\Projeto_0_IA376M\Double-Inverted-Pendulum\Inverted Pendulum\Final Versions\Models\double_inverted_pendulum.xml"
N_STEPS     = 5000
N_PERTURB   = 10                  # nº de perturbações para sensibilidade
DELTA_RANGE = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]  # magnitudes de perturbação

OUTPUT_DIR = Path(r"C:\Users\sanch\Downloads\Projeto_0_IA376M\Double-Inverted-Pendulum\MuJoco Validation\validation_output")
OUTPUT_DIR.mkdir(exist_ok=True)

model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

# ──────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ──────────────────────────────────────────────
def compute_energy(model, data):
    mujoco.mj_energyPos(model, data)
    mujoco.mj_energyVel(model, data)
    return float(data.energy[0]), float(data.energy[1])  # PE, KE

def run_simulation(model, data, qpos0, qvel0=None, n_steps=N_STEPS):
    """Roda simulação a partir de condição inicial e retorna DataFrame."""
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos0
    if qvel0 is not None:
        data.qvel[:] = qvel0

    records = []
    for step in range(n_steps):
        mujoco.mj_step(model, data)
        PE, KE = compute_energy(model, data)
        records.append({
            "step"           : step,
            "time"           : data.time,
            "qpos_0"         : data.qpos[0],
            "qpos_1"         : data.qpos[1],
            "qvel_0"         : data.qvel[0],
            "qvel_1"         : data.qvel[1],
            "qacc_norm"      : float(np.linalg.norm(data.qacc)),
            "kinetic_energy" : KE,
            "potential_energy": PE,
            "total_energy"   : KE + PE,
        })
    return pd.DataFrame(records)

# ──────────────────────────────────────────────
# 1. SIMULAÇÃO BASE
# ──────────────────────────────────────────────
print("▶ Rodando simulação base...")
qpos0_base = np.array([np.pi + 0.01, 0.01])  # próximo ao equilíbrio instável
df_base = run_simulation(model, data, qpos0_base)
df_base.to_csv(OUTPUT_DIR / "sim_base.csv", index=False)
print(f"  ✓ {N_STEPS} passos concluídos | dt={model.opt.timestep}s")

# ──────────────────────────────────────────────
# 2. ANÁLISE DE DISSIPAÇÃO DE ENERGIA
# ──────────────────────────────────────────────
print("\n▶ Analisando dissipação de energia...")

E      = df_base["total_energy"].values
dE     = np.diff(E)                          # variação por passo
dE_neg = dE[dE < 0]                          # deve ser sempre negativo (dissipação)
jumps  = np.abs(dE[np.abs(dE) > np.abs(dE).mean() + 3 * np.abs(dE).std()])

energy_report = {
    "E_inicial"                : float(E[0]),
    "E_final"                  : float(E[-1]),
    "dissipacao_total"         : float(E[0] - E[-1]),
    "dissipacao_pct"           : float((E[0] - E[-1]) / abs(E[0]) * 100) if E[0] != 0 else float("nan"),
    "dE_por_passo_media"       : float(np.mean(dE)),
    "dE_por_passo_std"         : float(np.std(dE)),
    "n_saltos_anomalos"        : int(len(jumps)),       # deve ser 0 idealmente
    "salto_maximo"             : float(jumps.max()) if len(jumps) > 0 else 0.0,
    "dissipacao_consistente"   : "SIM" if len(jumps) == 0 else f"NÃO ({len(jumps)} anomalias)",
}

print("\n=== ANÁLISE DE DISSIPAÇÃO ===")
for k, v in energy_report.items():
    print(f"  {k:35s}: {v}")

pd.DataFrame([energy_report]).to_csv(OUTPUT_DIR / "energy_report.csv", index=False)

# ──────────────────────────────────────────────
# 3. ANÁLISE DE SENSIBILIDADE A CI
# ──────────────────────────────────────────────
print("\n▶ Rodando análise de sensibilidade...")

sensitivity_records = []

for delta in DELTA_RANGE:
    divergences = []
    for _ in range(N_PERTURB):
        # Perturbação aleatória nas condições iniciais
        noise       = np.random.uniform(-delta, delta, size=model.nq)
        qpos_pert   = qpos0_base + noise
        df_pert     = run_simulation(model, data, qpos_pert, n_steps=N_STEPS)

        # Divergência média entre trajetórias (norma L2)
        diff_qpos0  = (df_pert["qpos_0"] - df_base["qpos_0"]).abs()
        diff_qpos1  = (df_pert["qpos_1"] - df_base["qpos_1"]).abs()
        divergence  = float(np.sqrt(diff_qpos0**2 + diff_qpos1**2).mean())
        divergences.append(divergence)

    sensitivity_records.append({
        "delta"              : delta,
        "divergencia_media"  : float(np.mean(divergences)),
        "divergencia_max"    : float(np.max(divergences)),
        "divergencia_std"    : float(np.std(divergences)),
        "razao_amplificacao" : float(np.mean(divergences) / delta),  # >1 = sistema instável/caótico
    })
    print(f"  δ={delta:.0e} → divergência média: {np.mean(divergences):.4e} | amplificação: {np.mean(divergences)/delta:.2f}x")

df_sensitivity = pd.DataFrame(sensitivity_records)
df_sensitivity.to_csv(OUTPUT_DIR / "sensitivity_report.csv", index=False)

# ──────────────────────────────────────────────
# 4. VISUALIZAÇÕES
# ──────────────────────────────────────────────
print("\n▶ Gerando gráficos...")

# --- Fig 1: Energia ao longo do tempo ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axes[0].plot(df_base["time"], df_base["total_energy"],     label="Total",    color="black")
axes[0].plot(df_base["time"], df_base["kinetic_energy"],   label="Cinética", color="blue",  alpha=0.7)
axes[0].plot(df_base["time"], df_base["potential_energy"], label="Potencial",color="red",   alpha=0.7)
axes[0].set_ylabel("Energia (J)")
axes[0].set_title("Dissipação de Energia — Sistema com Damping")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(df_base["time"][1:], np.diff(df_base["total_energy"].values), color="purple", linewidth=0.8)
axes[1].axhline(0, color="black", linestyle="--", linewidth=0.5)
axes[1].set_ylabel("ΔE por passo (J)")
axes[1].set_title("Variação de Energia por Passo — Suavidade da Dissipação")
axes[1].grid(True)

axes[2].plot(df_base["time"], df_base["qpos_0"], label="θ₁", color="blue")
axes[2].plot(df_base["time"], df_base["qpos_1"], label="θ₂", color="orange")
axes[2].set_ylabel("Posição (rad)")
axes[2].set_xlabel("Tempo (s)")
axes[2].set_title("Trajetória Angular")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "energy_analysis.png", dpi=150)
plt.close()
print("  ✓ energy_analysis.png")

# --- Fig 2: Sensibilidade ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].loglog(df_sensitivity["delta"], df_sensitivity["divergencia_media"],
               "o-", color="red", linewidth=2)
axes[0].loglog(df_sensitivity["delta"], df_sensitivity["delta"],
               "--", color="gray", label="y = δ (referência linear)")
axes[0].set_xlabel("Perturbação δ (rad)")
axes[0].set_ylabel("Divergência média da trajetória")
axes[0].set_title("Sensibilidade a Condições Iniciais")
axes[0].legend()
axes[0].grid(True, which="both", alpha=0.4)

axes[1].semilogx(df_sensitivity["delta"], df_sensitivity["razao_amplificacao"],
                 "s-", color="blue", linewidth=2)
axes[1].axhline(1, color="gray", linestyle="--", label="Amplificação = 1 (neutro)")
axes[1].set_xlabel("Perturbação δ (rad)")
axes[1].set_ylabel("Fator de amplificação")
axes[1].set_title("Amplificação do Erro — Estabilidade do Modelo")
axes[1].legend()
axes[1].grid(True, which="both", alpha=0.4)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sensitivity_analysis.png", dpi=150)
plt.close()
print("  ✓ sensitivity_analysis.png")

# ──────────────────────────────────────────────
# 5. RESUMO FINAL
# ──────────────────────────────────────────────
print("\n========================================")
print("      RESUMO DE VALIDAÇÃO")
print("========================================")
print(f"  Integrador          : {['Euler','RK4','implicit','implicitfast'][model.opt.integrator]}")
print(f"  Timestep            : {model.opt.timestep}s ({1/model.opt.timestep:.0f} Hz)")
print(f"  Freq. Nyquist       : {1/(2*model.opt.timestep):.0f} Hz")
print(f"  Dissipação total    : {energy_report['dissipacao_pct']:.2f}%")
print(f"  Anomalias de energia: {energy_report['n_saltos_anomalos']}")
print(f"  Dissipação consist. : {energy_report['dissipacao_consistente']}")
print(f"  Amplif. máx (δ=1e-6): {df_sensitivity[df_sensitivity['delta']==1e-6]['razao_amplificacao'].values[0]:.2f}x")
print(f"\n✓ Todos os arquivos salvos em: {OUTPUT_DIR.resolve()}")