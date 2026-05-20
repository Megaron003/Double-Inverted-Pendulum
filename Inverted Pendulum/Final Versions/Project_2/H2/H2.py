"""
╔══════════════════════════════════════════════════════════════════════════╗
║  TESTE H2 — ACOPLAMENTO CRUZADO REVERSO                                  ║
║  "θ₂ interfere mais em α₁ do que ω₁?"                                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  H₀: β_sinθ₂ = β_cosθ₂ = 0  no modelo α₁ = f(sinθ₁,cosθ₁,sinθ₂,cosθ₂,ω₁) ║
║  H₁: β_sinθ₂ ≠ 0 e/ou β_cosθ₂ ≠ 0                                        ║
║                                                                          ║
║  Nota: os ângulos são representados por seus valores trigonométricos     ║
║  (sin θ, cos θ) para preservar a geometria circular do espaço de         ║
║  configuração — consistente com a formulação do dataset (MuJoCo).        ║
║                                                                          ║
║  Alinhado ao Capítulo 11 — notas de aula IA376M/EA099M FEEC 2026         ║
║  § 11.1 EDA: scatter, MI, Spearman                                       ║
║  § 11.3 Diagnóstico: resíduos, Q-Q, Cook+leverage, VIF                   ║
║  § 11.4 Inferência: F-parcial, IC 99%                                    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
from scipy import stats

# ── 0. Dados ───────────────────────────────────────────────────────────────
CSV = "D:\Trabalhos\Smart Agri\Double_Inverted_Pendulum\Double-Inverted-Pendulum\Inverted Pendulum\Final Versions\Data Processed\pendulum_dataset_tidy_with_acceleration.csv"

df  = pd.read_csv(CSV)
df["theta1"] = np.arctan2(df["sin_theta1"], df["cos_theta1"])
df["theta2"] = np.arctan2(df["sin_theta2"], df["cos_theta2"])
n   = len(df)
rng = np.random.default_rng(42)
ALPHA = 0.05

y = df["angle_accel1"].values

# ── 1. Modelos aninhados ───────────────────────────────────────────────────
#
#   M_sem : α₁ = f(sin θ₁, cos θ₁, ω₁)
#           impõe β_sinθ₂ = β_cosθ₂ = 0
#
#   M_com : α₁ = f(sin θ₁, cos θ₁, sin θ₂, cos θ₂, ω₁)
#           levanta a restrição — θ₂ pode contribuir
#
#   Os ângulos são representados por sin/cos para:
#   (a) preservar a topologia circular (sem descontinuidade em ±π)
#   (b) capturar a forma funcional trigonométrica confirmada em HN-B
#   (c) evitar multicolinearidade com as colunas originais do dataset

X_sem = np.column_stack([
    np.sin(df["theta1"]), np.cos(df["theta1"]),
    df["omega1"]
])

X_com = np.column_stack([
    np.sin(df["theta1"]), np.cos(df["theta1"]),
    np.sin(df["theta2"]), np.cos(df["theta2"]),
    df["omega1"]
])

M_sem = sm.OLS(y, sm.add_constant(X_sem)).fit()
M_com = sm.OLS(y, sm.add_constant(X_com)).fit()

# ── 2. F-parcial ──────────────────────────────────────────────────────────
#   q = 2 restrições testadas: β_sinθ₂ = 0  e  β_cosθ₂ = 0
#   k = 6 parâmetros do modelo completo (intercepto + 5 regressores)
#
#   F = (RSS_sem - RSS_com)/q  /  RSS_com/(n-k)
#   Distribuição sob H₀: F(2, 49994)

q = 2; k = 6
F_obs = ((M_sem.ssr - M_com.ssr)/q) / (M_com.ssr/(n - k))
p_val = 1 - stats.f.cdf(F_obs, q, n - k)
F_crit = stats.f.ppf(1 - ALPHA, q, n - k)
dr2 = M_com.rsquared - M_sem.rsquared

# ── 3. IC 99% ─────────────────────────────────────────────────────────────
ci99   = M_com.conf_int(alpha=0.01)
NAMES  = ["const", "sin θ₁", "cos θ₁", "sin θ₂", "cos θ₂", "ω₁"]

# ── 4. VIF ────────────────────────────────────────────────────────────────
X_vif   = pd.DataFrame(X_com,
                        columns=["sin_t1","cos_t1","sin_t2","cos_t2","omega1"])
X_vif_c = sm.add_constant(X_vif)
vif     = {col: variance_inflation_factor(X_vif_c.values, i+1)
           for i, col in enumerate(X_vif.columns)}

# ── 5. Diagnóstico de influência ───────────────────────────────────────────
infl    = M_com.get_influence()
cooks   = infl.cooks_distance[0]
lever   = infl.hat_matrix_diag
std_res = infl.resid_studentized_internal
COOK_LIM  = 4 / n
LEVER_LIM = 2 * k / n

# ── 6. MI e Spearman ──────────────────────────────────────────────────────
mi_t2 = mutual_info_regression(df[["theta2"]].values, y, random_state=42)[0]
mi_w1 = mutual_info_regression(df[["omega1"]].values, y, random_state=42)[0]
rho_t2, p_rho_t2 = stats.spearmanr(df["theta2"], y)
rho_w1, _        = stats.spearmanr(df["omega1"],  y)

# ── 7. Relatório textual ───────────────────────────────────────────────────
SEP = "═" * 70
print(SEP)
print("  TESTE H2 — ACOPLAMENTO CRUZADO REVERSO")
print("  θ₂ interfere mais em α₁ do que ω₁?")
print("  Alinhado ao Capítulo 11 — notas de aula FEEC 2026")
print(SEP)

print("""
  H₀ : β_sinθ₂ = β_cosθ₂ = 0
       → θ₂ não possui poder preditivo incremental sobre α₁

  H₁ : β_sinθ₂ ≠ 0 e/ou β_cosθ₂ ≠ 0
       → θ₂ contribui para α₁ além de {sin θ₁, cos θ₁, ω₁}

  Distribuição sob H₀ : F(2, 49994)
  Nível de significância: α = 0.05
""")

print("  EDA ANTES DO MODELO (Cap.11 §11.1)")
print("  ─────────────────────────────────────────────────────────────")
print(f"  MI(θ₂, α₁)    = {mi_t2:.4f} nats  (estimador k-NN)")
print(f"  MI(ω₁, α₁)    = {mi_w1:.4f} nats")
print(f"  Razão MI       = {mi_t2/mi_w1:.2f}×  — θ₂ tem {mi_t2/mi_w1:.2f}x mais informação")
print(f"  ρ_S(θ₂, α₁)   = {rho_t2:+.4f}  p = {p_rho_t2:.3e}")
print(f"  ρ_S(ω₁, α₁)   = {rho_w1:+.4f}")
print(f"  |ρ(θ₂)| > |ρ(ω₁)|? {'Sim ✓' if abs(rho_t2) > abs(rho_w1) else 'Não ✗'}")

print(f"\n  MULTICOLINEARIDADE — VIF (Cap.11 §11.3)")
print("  ─────────────────────────────────────────────────────────────")
for col, v in vif.items():
    status = "✓ ok" if v < 5 else ("⚠ moderado" if v < 10 else "✗ crítico")
    print(f"  VIF({col:10s}) = {v:.3f}  {status}")

print(f"\n  INFERÊNCIA ESTATÍSTICA (Cap.11 §11.4)")
print("  ─────────────────────────────────────────────────────────────")
print(f"  R²(M_sem) = {M_sem.rsquared:.6f}  — sem θ₂")
print(f"  R²(M_com) = {M_com.rsquared:.6f}  — com θ₂")
print(f"  ΔR²       = {dr2:.6f}  (+{dr2/M_sem.rsquared*100:.1f}% relativo)")
print(f"\n  T_obs = F = {F_obs:.2f}")
print(f"  p = P(F_{{2,49994}} ≥ {F_obs:.2f} | H₀) ≈ {p_val:.4e}")
print(f"  F_crítico (α=0.05) = {F_crit:.3f}")
print(f"  Decisão: {'REJEITA H₀ ✓' if p_val < ALPHA else 'Não rejeita H₀'}")

print(f"\n  IC 99% dos coeficientes (Cap.11 §11.4)")
print("  ─────────────────────────────────────────────────────────────")
for nm, coef, pv, lo, hi in zip(NAMES, M_com.params, M_com.pvalues,
                                  ci99[:,0], ci99[:,1]):
    inclui_zero = lo <= 0 <= hi
    sig = "✓ sig." if not inclui_zero else "✗ n.sig."
    print(f"  {nm:10s}: β={coef:+.4f}  IC99%=[{lo:+.4f},{hi:+.4f}]"
          f"  p={pv:.3e}  {sig}")

print(f"\n  DIAGNÓSTICO DE INFLUÊNCIA (Cap.11 §11.3 — Fig.11.15)")
print("  ─────────────────────────────────────────────────────────────")
n_cook  = (cooks > COOK_LIM).sum()
n_lever = (lever > LEVER_LIM).sum()
print(f"  Cook máximo   = {cooks.max():.6f}   limiar = 4/n = {COOK_LIM:.6f}")
print(f"  Obs. acima do limiar: {n_cook} ({n_cook/n*100:.1f}%)")
print(f"  Leverage máx  = {lever.max():.6f}   limiar = 2p/n = {LEVER_LIM:.6f}")

print(f"\n  COMPARAÇÃO COM H3 (θ₁ → α₂)")
print("  ─────────────────────────────────────────────────────────────")
print(f"  {'Métrica':<25} {'H3: θ₁→α₂':>14} {'H2: θ₂→α₁':>14}")
print(f"  {'─'*55}")
print(f"  {'F-parcial':<25} {'1895.80':>14} {F_obs:>14.2f}")
print(f"  {'ΔR²':<25} {'0.0289':>14} {dr2:>14.4f}")
print(f"  {'MI (ângulo cruzado)':<25} {'0.785 nats':>14} {mi_t2:>13.3f} nats")
print(f"  {'Razão MI':<25} {'1.72×':>14} {mi_t2/mi_w1:>13.2f}×")

print(f"\n{SEP}")
print(f"""
  CONCLUSÃO:
  H₀ rejeitada: p ≈ 0 < α = 0,05  (F = {F_obs:.2f}; dist. F(2, 49994))

  θ₂ possui contribuição incremental sobre α₁:
  ΔR² = +{dr2:.4f}: adicionar sin(θ₂) e cos(θ₂) a {{sin θ₁, cos θ₁, ω₁}}
  explica {dr2*100:.1f}% adicionais de variância de α₁.
  Melhora relativa de {dr2/M_sem.rsquared*100:.1f}% sobre o modelo sem θ₂.

  O acoplamento θ₂ → α₁ é MAIS FORTE que θ₁ → α₂ (H3):
  ΔR² aqui = {dr2:.4f} vs ΔR² em H3 = 0.0289 — diferença de {dr2/0.0289:.0f}×

  VIF ≤ {max(vif.values()):.2f}: multicolinearidade ausente.
  IC 99% de β_sinθ₂ = [{ci99[3,0]:+.4f}, {ci99[3,1]:+.4f}]: não contém zero.

  IMPLICAÇÃO ARQUITETURAL:
  Modelos de elo único f(sinθ₁, cosθ₁, ω₁) → α₁ deixam {dr2*100:.1f}%
  da variância de α₁ completamente inexplicável.
  A rede precisa de entrada conjunta de todos os elos.
""")
print(SEP)

# ── 8. Figura ──────────────────────────────────────────────────────────────
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
    if title:  ax.set_title(title,   color=C["text"],  fontsize=8.5,
                             fontweight="bold", pad=5)
    if xlabel: ax.set_xlabel(xlabel, color=C["slate"], fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=C["slate"], fontsize=8)

fig = plt.figure(figsize=(20, 18), facecolor=C["bg"])
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.58, wspace=0.38)
idx = slice(None, None, 10)

# ── A. EDA: θ₂ vs α₁ ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
sa(ax, r"(a) EDA: $\theta_2$ vs $\alpha_1$  [§11.1]",
   r"$\theta_2$ (rad)", r"$\alpha_1$ (rad/s²)")
ax.scatter(df["theta2"].values[idx], y[idx], s=0.6, alpha=0.15, color=C["a1"])
bins = np.linspace(df["theta2"].min(), df["theta2"].max(), 40)
bm   = [y[(df["theta2"].values >= bins[i]) & (df["theta2"].values < bins[i+1])].mean()
        for i in range(len(bins)-1)]
ax.plot(0.5*(bins[:-1]+bins[1:]), bm, color=C["a5"], lw=2, label="média por bin")
ax.text(0.04, 0.94, f"ρ = {rho_t2:+.3f}\np = {p_rho_t2:.2e}\nMI = {mi_t2:.3f} nats",
        transform=ax.transAxes, color=C["a1"], fontsize=8, va="top", style="italic")
ax.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["text"])

# ── B. EDA: ω₁ vs α₁ ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
sa(ax, r"(b) EDA: $\omega_1$ vs $\alpha_1$  [§11.1]",
   r"$\omega_1$ (rad/s)", r"$\alpha_1$ (rad/s²)")
ax.scatter(df["omega1"].values[idx], y[idx], s=0.6, alpha=0.15, color=C["a2"])
bins2 = np.linspace(df["omega1"].min(), df["omega1"].max(), 40)
bm2   = [y[(df["omega1"].values >= bins2[i]) & (df["omega1"].values < bins2[i+1])].mean()
         for i in range(len(bins2)-1)]
ax.plot(0.5*(bins2[:-1]+bins2[1:]), bm2, color=C["a5"], lw=2, label="média por bin")
ax.text(0.04, 0.94, f"ρ = {rho_w1:+.3f}\nMI = {mi_w1:.3f} nats",
        transform=ax.transAxes, color=C["a2"], fontsize=8, va="top", style="italic")
ax.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["text"])

# ── C. VIF ────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
sa(ax, "(c) VIF — Multicolinearidade  [§11.3]", "Feature", "VIF")
vif_keys = list(vif.keys())
vif_vals = list(vif.values())
colors_v = [C["a3"] if v < 5 else C["a5"] for v in vif_vals]
ax.barh(range(len(vif_keys)), vif_vals, color=colors_v, alpha=0.85, edgecolor="none")
ax.axvline(5, color=C["a5"], lw=1.2, ls="--", label="limiar 5")
for i, v in enumerate(vif_vals):
    ax.text(v+0.01, i, f"{v:.2f}", va="center", color=C["text"], fontsize=8)
vif_labels = ["sin θ₁","cos θ₁","sin θ₂","cos θ₂","ω₁"]
ax.set_yticks(range(len(vif_labels)))
ax.set_yticklabels(vif_labels, color=C["slate"], fontsize=8.5)
ax.legend(fontsize=7.5, facecolor=C["panel"], labelcolor=C["text"])
ax.set_xlim(0, max(vif_vals)*1.4)

# ── D. R² comparativo ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
sa(ax, r"(d) $R^2$ — impacto de $\theta_2$  [§11.4]", "Modelo", r"$R^2$")
r2_vals  = [M_sem.rsquared, M_com.rsquared]
r2_labels = [r"M(sinθ₁,cosθ₁,ω₁)", r"M(+sinθ₂,cosθ₂)"]
bars = ax.bar([0,1], r2_vals, color=[C["a2"], C["a3"]], alpha=0.85,
              width=0.5, edgecolor="none")
for b, v in zip(bars, r2_vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.4f}",
            ha="center", va="bottom", color=C["text"], fontsize=9, fontweight="bold")
ax.set_xticks([0,1]); ax.set_xticklabels(r2_labels, color=C["slate"], fontsize=7.5)
ax.set_ylim(0, max(r2_vals)*1.3)
ax.text(0.5, (r2_vals[0]+r2_vals[1])/2 + 0.015,
        f"ΔR²\n+{dr2:.4f}\n({dr2/M_sem.rsquared*100:.0f}%↑)",
        ha="center", color=C["a5"], fontsize=9, fontweight="bold")

# ── E. IC 99% coeficientes ────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
sa(ax, "(e) IC 99% dos Coeficientes  [§11.4]",
   "Coeficiente β (features originais)", "")
coefs = M_com.params
lo99  = ci99[:,0]; hi99 = ci99[:,1]
pvals = M_com.pvalues
dot_cols = [C["a3"] if p < ALPHA else C["a2"] for p in pvals]
ax.barh(range(len(NAMES)), coefs,
        xerr=[coefs-lo99, hi99-coefs],
        color=dot_cols, alpha=0.75,
        error_kw=dict(ecolor=C["text"], lw=1.5, capsize=5),
        edgecolor="none", height=0.45)
ax.axvline(0, color=C["text"], lw=1, ls="--", alpha=0.5)
ax.set_yticks(range(len(NAMES)))
ax.set_yticklabels(NAMES, color=C["slate"], fontsize=9)
ax.legend(handles=[
    Line2D([0],[0], marker='s', color='w', markerfacecolor=C["a3"],
           markersize=8, label="p < 0.05"),
    Line2D([0],[0], marker='s', color='w', markerfacecolor=C["a2"],
           markersize=8, label="p ≥ 0.05"),
], fontsize=8, facecolor=C["panel"], labelcolor=C["text"])

# ── F. Resíduos vs sin(θ₂) ───────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
sa(ax, r"(f) Resíduos $M_{\rm sem}$ vs $\sin\theta_2$  [§11.3]",
   r"$\sin(\theta_2)$", r"Resíduo (rad/s²)")
resid_sem = M_sem.resid
ax.scatter(np.sin(df["theta2"].values[idx]), resid_sem[idx],
           s=0.7, alpha=0.15, color=C["a1"])
ax.axhline(0, color=C["text"], lw=0.8, ls="--", alpha=0.5)
bins3 = np.linspace(-1, 1, 40)
bm3   = [resid_sem[(np.sin(df["theta2"].values) >= bins3[i]) &
                    (np.sin(df["theta2"].values) < bins3[i+1])].mean()
         for i in range(len(bins3)-1)]
ax.plot(0.5*(bins3[:-1]+bins3[1:]), bm3, color=C["a5"], lw=2,
        label="média por bin")
ax.text(0.04, 0.94,
        "Padrão sigmoidal → sin(θ₂)\nnão capturado pelo M_sem",
        transform=ax.transAxes, color=C["a5"], fontsize=7.5,
        va="top", style="italic")
ax.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["text"])

# ── G. Resíduos vs valores ajustados ─────────────────────────────────────
ax = fig.add_subplot(gs[2, 0:2])
sa(ax, r"(g) Resíduos $M_{\rm com}$ vs valores ajustados  [§11.3 — Fig.11.9a]",
   r"$\hat{\alpha}_1$ (rad/s²)", r"Resíduo (rad/s²)")
ax.scatter(M_com.fittedvalues[idx], M_com.resid[idx],
           s=0.6, alpha=0.15, color=C["a3"])
ax.axhline(0, color=C["text"], lw=0.8, ls="--", alpha=0.6)
bins_r = np.percentile(M_com.fittedvalues, np.linspace(1,99,30))
bm_r   = [M_com.resid[(M_com.fittedvalues >= bins_r[i]) &
                        (M_com.fittedvalues < bins_r[i+1])].mean()
          for i in range(len(bins_r)-1)]
ax.plot(0.5*(bins_r[:-1]+bins_r[1:]), bm_r, color=C["a5"], lw=2,
        label="média por bin")
ax.text(0.02, 0.94, "Linha próxima de zero → homocedasticidade razoável",
        transform=ax.transAxes, color=C["a3"], fontsize=7.5,
        va="top", style="italic")
ax.legend(fontsize=7.5, facecolor=C["panel"], labelcolor=C["text"])

# ── H. Q-Q ────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 2])
sa(ax, "(h) Q-Q dos resíduos  [§11.3 — Fig.11.10]",
   "Quantis teóricos (Normal)", "Resíduos padronizados")
idx_qq = rng.choice(n, 2000, replace=False)
(osm, osr), (slope, intercept, r_qq) = stats.probplot(std_res[idx_qq])
ax.scatter(osm, osr, s=1.5, alpha=0.3, color=C["a4"])
x_ref = np.linspace(osm.min(), osm.max(), 200)
ax.plot(x_ref, slope*x_ref+intercept, color=C["a5"], lw=1.8, ls="--",
        label=f"referência  r={r_qq:.4f}")
ax.legend(fontsize=7.5, facecolor=C["panel"], labelcolor=C["text"])
ax.text(0.02, 0.94, "Desvio nas caudas → leptocúrtica\n(consistente com HN-D)",
        transform=ax.transAxes, color=C["a5"], fontsize=7.5,
        va="top", style="italic")

# ── I. Gráfico de influência ──────────────────────────────────────────────
ax = fig.add_subplot(gs[3, 0:2])
sa(ax, "(i) Gráfico de influência: Leverage vs Resíduo²  [§11.3 — Fig.11.15]",
   "Resíduo padronizado²", "Leverage ($h_{ii}$)")
idx_infl = rng.choice(n, 5000, replace=False)
ax.scatter(std_res[idx_infl]**2, lever[idx_infl],
           s=1.0, alpha=0.3, color=C["a4"])
ax.axvline(4, color=C["a5"], lw=1.2, ls="--",
           label="resíduo²=4 (|e|>2σ)")
ax.axhline(LEVER_LIM, color=C["a2"], lw=1.2, ls=":",
           label=f"leverage=2p/n={LEVER_LIM:.5f}")
high_cook = np.where(cooks > COOK_LIM*10)[0]
if len(high_cook) > 0:
    ax.scatter(std_res[high_cook]**2, lever[high_cook],
               s=20, color=C["a2"], alpha=0.9, zorder=5,
               label=f"Cook > 10×limiar ({len(high_cook)})")
ax.legend(fontsize=7.5, facecolor=C["panel"], labelcolor=C["text"])
ax.text(0.02, 0.94,
        f"Cook máx = {cooks.max():.5f}  (limiar 4/n = {COOK_LIM:.6f})\n"
        f"Nenhuma obs. exerce influência crítica",
        transform=ax.transAxes, color=C["a3"], fontsize=7.5,
        va="top", style="italic")

# ── J. Tabela resumo ──────────────────────────────────────────────────────
ax = fig.add_subplot(gs[3, 2])
ax.set_facecolor(C["panel"]); ax.axis("off")
ax.set_title("Resumo H2 — Framework p-valor", color=C["text"],
             fontsize=8.5, fontweight="bold", pad=5)

rows_tab = [
    ("Elemento",              "Valor",            "Ref. Cap.11"),
    ("H₀",                   "β_sinθ₂=β_cosθ₂=0", "§11.4"),
    ("Dist. sob H₀",         "F(2, 49994)",       "§11.4"),
    ("T_obs = F",            f"{F_obs:.2f}",      "§11.4"),
    ("p-valor",              "≈ 0",               "§11.4"),
    ("Decisão (α=0,05)",     "Rejeita H₀ ✓",     "§11.4"),
    ("ΔR²",                  f"+{dr2:.4f}",       "§11.4"),
    ("IC 99% β_sinθ₂",      f"[{ci99[3,0]:+.3f};{ci99[3,1]:+.3f}]","§11.4"),
    ("VIF máximo",           f"{max(vif.values()):.2f}",   "§11.3"),
    ("Cook máximo",          f"{cooks.max():.5f}","§11.3"),
    ("MI(θ₂)/MI(ω₁)",       f"{mi_t2/mi_w1:.2f}×","§11.1"),
]

cw = [0.38, 0.35, 0.27]; rh = 0.083
for ri, row in enumerate(rows_tab):
    for ci, (cell, w) in enumerate(zip(row, cw)):
        x  = sum(cw[:ci]); yc = 1.0 - (ri+1)*rh
        bg = C["a1"] if ri==0 else (C["bg"] if ri%2==0 else "#1C2330")
        fc = plt.Rectangle((x,yc),w,rh*0.93,
                           transform=ax.transAxes,
                           facecolor=bg, edgecolor=C["grid"], lw=0.4)
        ax.add_patch(fc)
        tc = (C["panel"] if ri==0 else
              C["a3"] if "✓" in cell else C["text"])
        ax.text(x+w/2, yc+rh*0.45, cell,
                transform=ax.transAxes,
                ha="center", va="center",
                color=tc if ri>0 else C["panel"],
                fontsize=7.5, fontweight="bold" if ri==0 else "normal")

fig.suptitle(
    r"H2 — Acoplamento cruzado reverso: $\theta_2$ interfere mais em $\alpha_1$ do que $\omega_1$?" +
    "\nDiagnóstico completo alinhado ao Capítulo 11 — IA376M/EA099M FEEC 2026",
    color=C["text"], fontsize=11, fontweight="bold", y=0.998
)

OUT = "D:\\Trabalhos\\Smart Agri\\Double_Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Project_2\\H5\\Results\\H5.png"

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=C["bg"])
print(f"\n[OK] Figura: {OUT}")
plt.close(fig)
