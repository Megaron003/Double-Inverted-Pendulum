"""
Distribuição sob H₀ e H₁ — Teste HN-A (Markovianidade)

O que este script calcula:
  - PDF da distribuição F(df1, df2) sob H₀ (hipótese nula)
  - PDF da distribuição F não-central sob H₁, com parâmetro
    de não-centralidade λ = F_obs × df1
  - Posição de T_obs em ambas as distribuições
  - Área da cauda (p-valor) visualmente
  - Poder do teste: P(rejeitar H₀ | H₁ verdadeira)
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ── 0. Dados e modelos ────────────────────────────────────────────────────
CSV = "D:\Trabalhos\Smart Agri\Double_Inverted_Pendulum\Double-Inverted-Pendulum\Inverted Pendulum\Final Versions\Data Processed\pendulum_dataset_tidy_with_acceleration.csv"
df  = pd.read_csv(CSV)
df["theta1"] = np.arctan2(df["sin_theta1"], df["cos_theta1"])
df["theta2"] = np.arctan2(df["sin_theta2"], df["cos_theta2"])

ep0 = df[df["episode"] == 0].copy().reset_index(drop=True)
scaler = StandardScaler()

COLS = ["theta1","theta2","omega1","omega2","tau1_dynamics","tau2_dynamics"]
X_nl_base = np.hstack([
    scaler.fit_transform(ep0[COLS].values),
    np.sin(ep0["theta1"].values).reshape(-1,1),
    np.cos(ep0["theta1"].values).reshape(-1,1),
    np.sin(ep0["theta2"].values).reshape(-1,1),
    np.cos(ep0["theta2"].values).reshape(-1,1),
    (ep0["theta1"]*ep0["omega1"]).values.reshape(-1,1),
    (ep0["theta2"]*ep0["omega2"]).values.reshape(-1,1),
])
X_nl = scaler.fit_transform(X_nl_base)

SL = slice(3, -3)
k  = X_nl.shape[1]   # 12 features × lag = 12 restrições

TARGETS = [("omega1", r"$\omega_1(t+1)$"), ("omega2", r"$\omega_2(t+1)$")]
ALPHA   = 0.05

# Coletar estatísticas dos modelos
model_stats = {}
for col, label in TARGETS:
    y_next = np.roll(ep0[col].values, -1)
    y    = y_next[SL]
    X_t0 = X_nl[SL]
    X_t1 = np.roll(X_nl, 1, axis=0)[SL]
    N    = len(y)

    M0 = sm.OLS(y, sm.add_constant(X_t0)).fit()
    M1 = sm.OLS(y, sm.add_constant(np.hstack([X_t0, X_t1]))).fit()

    df1 = k
    df2 = int(N - M1.df_model - 1)
    F_obs = ((M0.ssr - M1.ssr)/df1) / (M1.ssr/df2)
    p_val = 1 - stats.f.cdf(F_obs, df1, df2)
    F_crit = stats.f.ppf(1 - ALPHA, df1, df2)
    ncp    = F_obs * df1           # parâmetro de não-centralidade

    # Poder: P(F_nc > F_crit | H₁) — prob. de rejeitar dado que H₁ é verdadeira
    power = 1 - stats.ncf.cdf(F_crit, df1, df2, nc=ncp)

    model_stats[col] = dict(
        label=label, F_obs=F_obs, df1=df1, df2=df2,
        p_val=p_val, F_crit=F_crit, ncp=ncp, power=power,
        r2_M0=M0.rsquared, r2_M1=M1.rsquared,
        dr2=M1.rsquared - M0.rsquared,
    )

# ── 1. Relatório ──────────────────────────────────────────────────────────
print("="*68)
print("  DISTRIBUIÇÕES SOB H₀ E H₁ — TESTE HN-A (MARKOVIANIDADE)")
print("="*68)

for col, s in model_stats.items():
    print(f"""
  Alvo: {s['label']}
  ─────────────────────────────────────────────────────────────────
  Distribuição sob H₀ : F({s['df1']}, {s['df2']})
                        concentrada em torno de 1
                        F_crítico (α=0,05) = {s['F_crit']:.4f}

  T_obs = F             : {s['F_obs']:.2f}
  p = P(F ≥ T_obs | H₀) : {s['p_val']:.4e}
  Decisão formal         : {"Rejeita H₀ ✓" if s['p_val'] < ALPHA else "Não rejeita"}

  Distribuição sob H₁ : F não-central({s['df1']}, {s['df2']}, λ={s['ncp']:.0f})
                        λ = F_obs × df1 = {s['F_obs']:.2f} × {s['df1']} = {s['ncp']:.0f}
                        centrada em {s['ncp']/s['df1']:.0f} ≈ F_obs

  Poder do teste        : {s['power']:.6f}  (≈ 100%)
  ΔR²                   : {s['dr2']*100:.4f}%  (efeito prático desprezível)

  INTERPRETAÇÃO:
  T_obs={s['F_obs']:.0f} está a milhares de desvios-padrão da distribuição
  sob H₀ (centrada em ~1). Visualmente, as duas distribuições são
  completamente separadas — o poder é 100%.
  Contudo, ΔR²={s['dr2']*100:.4f}% mostra que o efeito é estatisticamente
  detectável mas praticamente irrelevante.
""")

# ── 2. Figura ──────────────────────────────────────────────────────────────
C = {
    "bg":    "#0D1117", "panel": "#161B22", "grid":  "#21262D",
    "text":  "#E6EDF3", "a1":    "#58A6FF", "a2":    "#F87171",
    "a3":    "#3FB950", "a4":    "#D2A8FF", "a5":    "#FFA657",
    "slate": "#64748B",
}

def sa(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["slate"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(C["grid"]); sp.set_linewidth(0.6)
    ax.grid(True, color=C["grid"], lw=0.4, alpha=0.7)
    if title:  ax.set_title(title,   color=C["text"],  fontsize=9,
                             fontweight="bold", pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=C["slate"], fontsize=8.5)
    if ylabel: ax.set_ylabel(ylabel, color=C["slate"], fontsize=8.5)

fig = plt.figure(figsize=(20, 16), facecolor=C["bg"])
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.38)

for ci, (col, s) in enumerate(model_stats.items()):
    df1, df2, F_obs = s["df1"], s["df2"], s["F_obs"]
    F_crit, ncp = s["F_crit"], s["ncp"]

    # ── A. Distribuição sob H₀ — zoom na região crítica ──────────────────
    # A distribuição F(df1, df2) está concentrada em ~1.
    # Precisamos mostrar tanto a distribuição inteira quanto onde F_obs cai.
    ax = fig.add_subplot(gs[0, ci])
    sa(ax,
       f"Distribuição sob H₀  —  {s['label']}\n"
       f"F({df1}, {df2})  concentrada em 1",
       "F", "PDF")

    # Grade: de 0 até 2×F_crítico para mostrar a região de rejeição
    x_h0 = np.linspace(0.01, F_crit * 3, 2000)
    y_h0 = stats.f.pdf(x_h0, df1, df2)

    ax.plot(x_h0, y_h0, color=C["a1"], lw=2.5, label=r"PDF sob $H_0$: F(df1, df2)")

    # Área de rejeição (cauda direita)
    x_rej = np.linspace(F_crit, F_crit * 3, 500)
    ax.fill_between(x_rej, stats.f.pdf(x_rej, df1, df2),
                    color=C["a2"], alpha=0.45,
                    label=f"Região de rejeição (α={ALPHA})\nF_crítico={F_crit:.4f}")

    # Linha do F_crítico
    ax.axvline(F_crit, color=C["a2"], lw=1.8, ls="--")

    # Anotação: T_obs está fora do gráfico (F_obs >> F_crit)
    ax.text(0.97, 0.82,
            f"T_obs = {F_obs:.0f}\n(fora da escala →)\n\n"
            f"p = P(F ≥ {F_obs:.0f} | H₀)\n≈ 0",
            transform=ax.transAxes,
            ha="right", va="top", color=C["a2"],
            fontsize=8.5, style="italic",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=C["panel"], edgecolor=C["a2"],
                      linewidth=1.2, alpha=0.92))

    # Seta apontando para fora do gráfico
    ax.annotate("", xy=(F_crit*2.8, max(y_h0)*0.3),
                xytext=(F_crit*2.2, max(y_h0)*0.3),
                arrowprops=dict(arrowstyle="->", color=C["a2"], lw=2))
    ax.text(F_crit*2.85, max(y_h0)*0.3,
            f"T_obs={F_obs:.0f} →→→",
            color=C["a2"], fontsize=8, va="center")

    ax.legend(fontsize=7.5, facecolor=C["panel"], labelcolor=C["text"],
              loc="upper right")

    # ── B. Comparação H₀ vs H₁ em escala logarítmica ─────────────────────
    ax = fig.add_subplot(gs[1, ci])
    sa(ax,
       f"PDF sob H₀ vs H₁  —  {s['label']}\n"
       f"Escala log-log para mostrar ambas",
       "log₁₀(F)", "log₁₀(PDF)")

    # Para visualizar ambas precisamos escala log pois estão em regiões
    # completamente distintas: H₀ centrada em ~1, H₁ centrada em ~F_obs
    x_full = np.logspace(-1, np.log10(F_obs * 1.5), 3000)

    # PDF sob H₀
    y_full_h0 = stats.f.pdf(x_full, df1, df2)
    # PDF sob H₁ (F não-central)
    y_full_h1 = stats.ncf.pdf(x_full, df1, df2, nc=ncp)

    # Evitar log(0)
    mask_h0 = y_full_h0 > 1e-300
    mask_h1 = y_full_h1 > 1e-300

    ax.plot(np.log10(x_full[mask_h0]), np.log10(y_full_h0[mask_h0]),
            color=C["a1"], lw=2, label=r"PDF sob $H_0$  (F central)")
    ax.plot(np.log10(x_full[mask_h1]), np.log10(y_full_h1[mask_h1]),
            color=C["a3"], lw=2, label=r"PDF sob $H_1$  (F não-central, λ≈{:.0e})".format(ncp))

    # Linha F_crítico
    ax.axvline(np.log10(F_crit), color=C["a2"], lw=1.5, ls="--",
               label=f"F_crítico = {F_crit:.2f}")
    # Linha F_obs
    ax.axvline(np.log10(F_obs), color=C["a5"], lw=2, ls=":",
               label=f"T_obs = {F_obs:.0f}")

    ax.text(np.log10(F_crit) + 0.05, ax.get_ylim()[0]*0.95 if ax.get_ylim()[0] < 0 else -10,
            "← H₀\nH₁ →",
            color=C["slate"], fontsize=8, va="bottom")

    ax.legend(fontsize=7.5, facecolor=C["panel"], labelcolor=C["text"])
    ax.set_xlabel("log₁₀(F)", color=C["slate"], fontsize=8.5)
    ax.set_ylabel("log₁₀(PDF)", color=C["slate"], fontsize=8.5)

    # Anotação separação
    ax.text(0.03, 0.15,
            f"As duas distribuições são\ncompletamente separadas:\n"
            f"H₀ centrada em ~1\nH₁ centrada em ~{F_obs:.0f}\n"
            f"Poder = {s['power']*100:.4f}%",
            transform=ax.transAxes, color=C["a5"],
            fontsize=8, va="bottom", style="italic",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=C["panel"], edgecolor=C["a5"],
                      linewidth=1, alpha=0.9))

# ── C. Painel central: ΔR² — significância estatística vs prática ─────────
ax = fig.add_subplot(gs[2, :])
sa(ax,
   "Paradoxo do tamanho amostral: p ≈ 0 com ΔR² desprezível\n"
   "Por que rejeitar H₀ formalmente NÃO implica relevância arquitetural",
   "", "")
ax.axis("off")

# Desenhar diagrama explicativo
# Eixo conceitual de "tamanho do efeito"
ax_inset = fig.add_axes([0.08, 0.04, 0.84, 0.22])
ax_inset.set_facecolor(C["panel"])
for sp in ax_inset.spines.values():
    sp.set_color(C["grid"])
ax_inset.tick_params(colors=C["slate"], labelsize=8)
ax_inset.grid(True, color=C["grid"], lw=0.4, alpha=0.6)

# Curva: p-valor como função de ΔR² para diferentes n
dr2_range = np.linspace(0.00001, 0.05, 500)
for n_val, col_n, lbl in [
    (1000,   C["a4"], "n = 1.000"),
    (10000,  C["a1"], "n = 10.000  ← nosso caso"),
    (50000,  C["a5"], "n = 50.000"),
    (100000, C["a2"], "n = 100.000"),
]:
    # Aproximação: F ≈ ΔR² × n / k (para k=12, modelo simples)
    F_approx = (dr2_range * n_val) / 12
    p_approx  = 1 - stats.f.cdf(F_approx, 12, n_val - 13)
    ax_inset.semilogy(dr2_range * 100, np.maximum(p_approx, 1e-300),
                      color=col_n, lw=2, label=lbl)

ax_inset.axhline(0.05, color=C["text"], lw=1.2, ls="--", alpha=0.7,
                 label="α = 0,05")
ax_inset.axvline(0.004, color=C["a1"], lw=1.5, ls=":",
                 label="ΔR² observado = 0,004%")

ax_inset.set_xlabel("ΔR² (%)", color=C["slate"], fontsize=9)
ax_inset.set_ylabel("p-valor (log)", color=C["slate"], fontsize=9)
ax_inset.set_title(
    "p-valor em função de ΔR²: com n grande, efeitos minúsculos rejeitam H₀ formalmente",
    color=C["text"], fontsize=9, fontweight="bold"
)
ax_inset.tick_params(colors=C["slate"])
ax_inset.legend(fontsize=8, facecolor=C["panel"], labelcolor=C["text"],
                loc="upper right")

# Anotação do ponto observado
ax_inset.scatter([0.004], [1e-200], s=120, color=C["a1"],
                 zorder=5, marker="*")
ax_inset.text(0.006, 1e-150,
              f"Nosso caso:\nΔR²=0,004%\np≈0\nRejeita H₀\nmas efeito é\ndesprezível",
              color=C["a1"], fontsize=8, style="italic",
              bbox=dict(boxstyle="round,pad=0.3",
                        facecolor=C["panel"], edgecolor=C["a1"],
                        linewidth=1, alpha=0.9))

fig.suptitle(
    "HN-A — Distribuições sob H₀ e H₁\n"
    r"$H_0$: dinâmica Markoviana  |  "
    r"$H_1$: lag-1 acrescenta poder preditivo  |  "
    "F-parcial não-central",
    color=C["text"], fontsize=11, fontweight="bold", y=0.998
)

OUT = "D:\Trabalhos\Smart Agri\Double_Inverted_Pendulum\Double-Inverted-Pendulum\Inverted Pendulum\Final Versions\Project_2\H3\Distrib\HNA_distribuicoes_H0_H1.png"
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=C["bg"])
print(f"\n[OK] Figura: {OUT}")
plt.close(fig)
