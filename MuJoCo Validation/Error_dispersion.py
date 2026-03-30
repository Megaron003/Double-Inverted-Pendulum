import mujoco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

# ──────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────
XML_PATH   = r"C:\Users\sanch\Downloads\Projeto_0_IA376M\Double-Inverted-Pendulum\Inverted Pendulum\Final Versions\Models\double_inverted_pendulum.xml"  # ⚠️ ajuste aqui
N_STEPS    = 5000
OUTPUT_DIR = Path(r"C:\Users\sanch\Downloads\Projeto_0_IA376M\Double-Inverted-Pendulum\MuJoco Validation\validation_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Mapeamento de integradores MuJoCo
INTEGRATORS = {
    "Euler"        : mujoco.mjtIntegrator.mjINT_EULER,
    "RK4"          : mujoco.mjtIntegrator.mjINT_RK4,
    "Implicit"     : mujoco.mjtIntegrator.mjINT_IMPLICIT,
}

# Condição inicial — próxima ao equilíbrio instável
QPOS0 = np.array([np.pi + 0.01, 0.01])
QVEL0 = np.zeros(2)

# ──────────────────────────────────────────────
# FUNÇÃO: roda simulação com integrador definido
# ──────────────────────────────────────────────
def run_with_integrator(xml_path, integrator_enum, qpos0, qvel0, n_steps):
    model      = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.integrator = integrator_enum
    data       = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0

    records = []
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        records.append({
            "time"  : data.time,
            "qpos_0": data.qpos[0],
            "qpos_1": data.qpos[1],
            "qvel_0": data.qvel[0],
            "qvel_1": data.qvel[1],
        })

    return pd.DataFrame(records)

# ──────────────────────────────────────────────
# 1. RODA OS 3 INTEGRADORES
# ──────────────────────────────────────────────
print("▶ Rodando simulações...")
results = {}
for name, enum in INTEGRATORS.items():
    print(f"  → {name}...", end=" ")
    results[name] = run_with_integrator(XML_PATH, enum, QPOS0, QVEL0, N_STEPS)
    print("✓")

# ──────────────────────────────────────────────
# 2. CALCULA DIVERGÊNCIAS
# ──────────────────────────────────────────────
def compute_divergence(df_a, df_b):
    """Divergência L2 entre duas trajetórias."""
    dpos = np.sqrt((df_a["qpos_0"] - df_b["qpos_0"])**2 +
                   (df_a["qpos_1"] - df_b["qpos_1"])**2)
    dvel = np.sqrt((df_a["qvel_0"] - df_b["qvel_0"])**2 +
                   (df_a["qvel_1"] - df_b["qvel_1"])**2)
    return dpos, dvel

pairs = [
    ("RK4",      "Euler"),
    ("Implicit", "Euler"),
    ("RK4",      "Implicit"),
]

divergences = {}
print("\n▶ Calculando divergências...")
for a, b in pairs:
    dpos, dvel    = compute_divergence(results[a], results[b])
    label         = f"{a} vs {b}"
    divergences[label] = {"time": results[a]["time"], "dpos": dpos, "dvel": dvel}
    print(f"  {label:20s} → dpos_max: {dpos.max():.4e} | dvel_max: {dvel.max():.4e}")

# ──────────────────────────────────────────────
# 3. EXPORTA CSVs
# ──────────────────────────────────────────────
for label, div in divergences.items():
    fname = label.replace(" ", "_").replace("/", "_") + "_divergence.csv"
    pd.DataFrame({
        "time": div["time"],
        "divergence_qpos": div["dpos"],
        "divergence_qvel": div["dvel"],
    }).to_csv(OUTPUT_DIR / fname, index=False)

print("\n  ✓ CSVs exportados")

# ──────────────────────────────────────────────
# 4. GRÁFICOS DE DISPERSÃO DE ERRO
# ──────────────────────────────────────────────
COLORS = {
    "RK4 vs Euler"      : "#e63946",
    "Implicit vs Euler" : "#457b9d",
    "RK4 vs Implicit"   : "#2a9d8f",
}

time = results["Euler"]["time"]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Dispersão de Erro entre Integradores — Pêndulo Duplo Invertido", fontsize=14, fontweight="bold")

# ─── Linear: posição ───
ax = axes[0, 0]
for label, div in divergences.items():
    ax.scatter(div["time"], div["dpos"], s=1.5, alpha=0.5,
               color=COLORS[label], label=label)
ax.set_title("Divergência de Posição (escala linear)")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("||Δqpos|| (rad)")
ax.legend(markerscale=6)
ax.grid(True, alpha=0.3)

# ─── Log: posição ───
ax = axes[0, 1]
for label, div in divergences.items():
    ax.scatter(div["time"], div["dpos"] + 1e-16, s=1.5, alpha=0.5,
               color=COLORS[label], label=label)
ax.set_yscale("log")
ax.set_title("Divergência de Posição (escala log)")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("||Δqpos|| (rad)")
ax.legend(markerscale=6)
ax.grid(True, which="both", alpha=0.3)

# ─── Linear: velocidade ───
ax = axes[1, 0]
for label, div in divergences.items():
    ax.scatter(div["time"], div["dvel"], s=1.5, alpha=0.5,
               color=COLORS[label], label=label)
ax.set_title("Divergência de Velocidade (escala linear)")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("||Δqvel|| (rad/s)")
ax.legend(markerscale=6)
ax.grid(True, alpha=0.3)

# ─── Log: velocidade ───
ax = axes[1, 1]
for label, div in divergences.items():
    ax.scatter(div["time"], div["dvel"] + 1e-16, s=1.5, alpha=0.5,
               color=COLORS[label], label=label)
ax.set_yscale("log")
ax.set_title("Divergência de Velocidade (escala log)")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("||Δqvel|| (rad/s)")
ax.legend(markerscale=6)
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "integrator_divergence.png", dpi=150)
plt.close()
print("  ✓ integrator_divergence.png")

# ──────────────────────────────────────────────
# 5. GRÁFICO DE TRAJETÓRIAS (comparativo visual)
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Trajetórias por Integrador", fontsize=14, fontweight="bold")

INT_COLORS = {"Euler": "#e63946", "RK4": "#2a9d8f", "Implicit": "#457b9d"}

for col, var in enumerate(["qpos_0", "qpos_1"]):
    label_y = "θ₁ (rad)" if col == 0 else "θ₂ (rad)"
    for row, scale in enumerate(["linear", "log"]):
        ax = axes[row, col]
        for name, df in results.items():
            ax.plot(df["time"], df[var], linewidth=0.8,
                    color=INT_COLORS[name], label=name, alpha=0.85)
        if scale == "log":
            ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_title(f"{label_y} — escala {scale}")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel(label_y)
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "integrator_trajectories.png", dpi=150)
plt.close()
print("  ✓ integrator_trajectories.png")

# ──────────────────────────────────────────────
# 6. RESUMO ESTATÍSTICO
# ──────────────────────────────────────────────
print("\n========================================")
print("   RESUMO — DIVERGÊNCIA ENTRE INTEGRADORES")
print("========================================")
for label, div in divergences.items():
    print(f"\n  {label}")
    print(f"    dpos máx : {div['dpos'].max():.4e} rad")
    print(f"    dpos média: {div['dpos'].mean():.4e} rad")
    print(f"    dvel máx : {div['dvel'].max():.4e} rad/s")
    print(f"    dvel média: {div['dvel'].mean():.4e} rad/s")

print(f"\n✓ Arquivos salvos em: {OUTPUT_DIR.resolve()}")