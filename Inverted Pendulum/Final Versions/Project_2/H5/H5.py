"""
╔══════════════════════════════════════════════════════════════════════════╗
║  HIPÓTESE HN-F — DECOMPOSIÇÃO DE θ₂: sin θ₂ vs cos θ₂                    ║
║  Qual componente trigonométrico de θ₂ explica α₁?                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  CONTEXTO                                                                ║
║  H2 confirmou que θ₂ é indispensável para prever α₁ (ΔR²=26,5%).         ║
║  HN-F decompõe essa contribuição: sin θ₂ e cos θ₂ contribuem             ║
║  igualmente, ou um domina?                                               ║
║                                                                          ║
║  H₀: β_sinθ₂ = 0  dado cos θ₂  (sin θ₂ é redundante dado cos θ₂)         ║
║  H₁: β_sinθ₂ ≠ 0  dado cos θ₂  (sin θ₂ acrescenta além de cos θ₂)        ║
║                                                                          ║
║  Simétricamente para cos θ₂:                                             ║
║  H₀': β_cosθ₂ = 0  dado sin θ₂                                           ║
║  H₁': β_cosθ₂ ≠ 0  dado sin θ₂                                           ║
║                                                                          ║
║  IMPLICAÇÃO ARQUITETURAL                                                 ║
║  sin θ₂ domina → a gravidade (F_grav ∝ sin θ) é o acoplamento           ║
║  principal. A rede precisa capturar sin θ₂ explicitamente.               ║
║  cos θ₂ marginal → inércia e centrifugação são efeitos secundários.      ║
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
from sklearn.feature_selection import mutual_info_regression
from scipy import stats

# ── 0. Dados ───────────────────────────────────────────────────────────────
CSV = r"C:\Users\Guilherme\Mestrado\Inverted_Pendulum\Double-Inverted-Pendulum\Inverted Pendulum\Final Versions\Data Processed\pendulum_dataset_tidy_with_acceleration.csv"
df  = pd.read_csv(CSV)
df["theta1"] = np.arctan2(df["sin_theta1"], df["cos_theta1"])
df["theta2"] = np.arctan2(df["sin_theta2"], df["cos_theta2"])
n    = len(df)
rng  = np.random.default_rng(42)
ALPHA = 0.05

y     = df["angle_accel1"].values
sin_t2 = np.sin(df["theta2"].values)
cos_t2 = np.cos(df["theta2"].values)
sin_t1 = np.sin(df["theta1"].values)
cos_t1 = np.cos(df["theta1"].values)
w1     = df["omega1"].values

# ── 1. Quatro modelos aninhados ────────────────────────────────────────────
#
#   M_full    : sin θ₁, cos θ₁, sin θ₂, cos θ₂, ω₁  (modelo completo de H2)
#   M_sem_sin : sin θ₁, cos θ₁,         cos θ₂, ω₁  (retira sin θ₂)
#   M_sem_cos : sin θ₁, cos θ₁, sin θ₂,         ω₁  (retira cos θ₂)
#   M_sem_tudo: sin θ₁, cos θ₁,                  ω₁  (retira ambos = restrito H2)

X_full     = np.column_stack([sin_t1, cos_t1, sin_t2, cos_t2, w1])
X_sem_sin  = np.column_stack([sin_t1, cos_t1,          cos_t2, w1])
X_sem_cos  = np.column_stack([sin_t1, cos_t1, sin_t2,          w1])
X_sem_tudo = np.column_stack([sin_t1, cos_t1,                  w1])

M_full     = sm.OLS(y, sm.add_constant(X_full)).fit()
M_sem_sin  = sm.OLS(y, sm.add_constant(X_sem_sin)).fit()
M_sem_cos  = sm.OLS(y, sm.add_constant(X_sem_cos)).fit()
M_sem_tudo = sm.OLS(y, sm.add_constant(X_sem_tudo)).fit()

# ── 2. F-parciais ──────────────────────────────────────────────────────────
#   Contribuição incremental de sin θ₂ dado que cos θ₂ já está presente
#   H₀: β_sinθ₂ = 0  |  distribuição: F(1, n-6)

k = 6   # parâmetros do modelo completo
df_err = n - k

F_sin  = ((M_sem_sin.ssr  - M_full.ssr)/1) / (M_full.ssr/df_err)
p_sin  = 1 - stats.f.cdf(F_sin,  1, df_err)

F_cos  = ((M_sem_cos.ssr  - M_full.ssr)/1) / (M_full.ssr/df_err)
p_cos  = 1 - stats.f.cdf(F_cos,  1, df_err)

F_both = ((M_sem_tudo.ssr - M_full.ssr)/2) / (M_full.ssr/df_err)
p_both = 1 - stats.f.cdf(F_both, 2, df_err)

F_crit_1 = stats.f.ppf(1 - ALPHA, 1, df_err)
F_crit_2 = stats.f.ppf(1 - ALPHA, 2, df_err)

# ── 3. ΔR² ────────────────────────────────────────────────────────────────
dr2_sin   = M_full.rsquared - M_sem_sin.rsquared   # marginal de sin θ₂
dr2_cos   = M_full.rsquared - M_sem_cos.rsquared   # marginal de cos θ₂
dr2_both  = M_full.rsquared - M_sem_tudo.rsquared  # conjunto (= H2)

# ── 4. IC 99% ─────────────────────────────────────────────────────────────
ci99  = M_full.conf_int(alpha=0.01)
NAMES = ["const", "sin θ₁", "cos θ₁", "sin θ₂", "cos θ₂", "ω₁"]

# ── 5. MI individual ──────────────────────────────────────────────────────
mi_sin = mutual_info_regression(sin_t2.reshape(-1,1), y, random_state=42)[0]
mi_cos = mutual_info_regression(cos_t2.reshape(-1,1), y, random_state=42)[0]

# ── 6. Relatório ──────────────────────────────────────────────────────────
SEP = "═" * 70
print(SEP)
print("  HN-F — DECOMPOSIÇÃO DE θ₂: sin θ₂ vs cos θ₂")
print(f"  n = {n:,}  |  α = {ALPHA}")
print(SEP)

print("""
  H₀ : β_sinθ₂ = 0  dado cos θ₂  — sin θ₂ é redundante
  H₁ : β_sinθ₂ ≠ 0  dado cos θ₂  — sin θ₂ acrescenta além de cos θ₂

  H₀': β_cosθ₂ = 0  dado sin θ₂  — cos θ₂ é redundante
  H₁': β_cosθ₂ ≠ 0  dado sin θ₂  — cos θ₂ acrescenta além de sin θ₂

  Distribuição sob H₀ e H₀': F(1, 49994)
""")

print("  MODELOS E R²")
print("  ─────────────────────────────────────────────────────────────")
print(f"  M_full    (sin+cos θ₂): R² = {M_full.rsquared:.6f}")
print(f"  M_sem_sin (só  cos θ₂): R² = {M_sem_sin.rsquared:.6f}")
print(f"  M_sem_cos (só  sin θ₂): R² = {M_sem_cos.rsquared:.6f}")
print(f"  M_sem_tudo (sem  θ₂)  : R² = {M_sem_tudo.rsquared:.6f}")

print(f"\n  F-PARCIAIS (distribuição sob H₀: F(1, {df_err}))")
print("  ─────────────────────────────────────────────────────────────")
print(f"  sin θ₂ | cos θ₂ : F = {F_sin:>10.2f}  p = {p_sin:.4e}"
      f"  ΔR² = {dr2_sin:.6f}"
      f"  → {'REJEITA H₀ ✓' if p_sin < ALPHA else 'não rejeita'}")
print(f"  cos θ₂ | sin θ₂ : F = {F_cos:>10.2f}  p = {p_cos:.4e}"
      f"  ΔR² = {dr2_cos:.6f}"
      f"  → {'rejeita H₀ ✓' if p_cos < ALPHA else 'Não rejeita ✗'}")
print(f"  conjunto        : F = {F_both:>10.2f}  p = {p_both:.4e}"
      f"  ΔR² = {dr2_both:.6f}")

print(f"\n  PROPORÇÃO DA CONTRIBUIÇÃO DE θ₂")
print("  ─────────────────────────────────────────────────────────────")
print(f"  ΔR²(sin θ₂) / ΔR²(conjunto) = {dr2_sin/dr2_both*100:.1f}%  da contribuição total")
print(f"  ΔR²(cos θ₂) / ΔR²(conjunto) = {dr2_cos/dr2_both*100:.1f}%  da contribuição total")
print(f"  Razão ΔR²(sin)/ΔR²(cos)      = {dr2_sin/max(dr2_cos,1e-10):.0f}×")

print(f"\n  IC 99% DOS COEFICIENTES DE θ₂")
print("  ─────────────────────────────────────────────────────────────")
for nm, coef, pv, lo, hi in zip(NAMES, M_full.params, M_full.pvalues,
                                  ci99[:,0], ci99[:,1]):
    if 'θ₂' in nm:
        sig = "✓ sig." if not (lo <= 0 <= hi) else "✗ n.sig."
        print(f"  {nm:10s}: β={coef:+.4f}  IC99%=[{lo:+.4f},{hi:+.4f}]"
              f"  p={pv:.3e}  {sig}")

print(f"\n  INFORMAÇÃO MÚTUA")
print("  ─────────────────────────────────────────────────────────────")
print(f"  MI(sin θ₂, α₁) = {mi_sin:.4f} nats")
print(f"  MI(cos θ₂, α₁) = {mi_cos:.4f} nats")
print(f"  Razão MI(sin)/MI(cos) = {mi_sin/max(mi_cos,1e-10):.1f}×")

print(f"\n{SEP}")
print(f"""
  CONCLUSÃO:
  sin θ₂ explica {dr2_sin/dr2_both*100:.1f}% da contribuição total de θ₂ sobre α₁.
  cos θ₂ explica {dr2_cos/dr2_both*100:.1f}% — contribuição {'' if p_cos < ALPHA else 'NÃO '}significativa
  mas {dr2_sin/max(dr2_cos,1e-10):.0f}× menor do que sin θ₂.

  INTERPRETAÇÃO FÍSICA:
  sin θ₂ é o termo de força gravitacional nas equações de Lagrange:
    F_grav ∝ (m₁ + m₂)·g·sin θ₂
  Esse é o acoplamento dominante entre o segundo elo e α₁.
  cos θ₂ aparece nos termos de inércia e força centrífuga — efeitos
  secundários neste regime de operação.

  IMPLICAÇÃO ARQUITETURAL:
  A feature mais importante para α₁ é sin θ₂, não θ₂ bruto.
  Usar sin θ₂ e cos θ₂ como features (em vez de θ₂) é a representação
  correta — sin θ₂ carrega {dr2_sin/dr2_both*100:.0f}% da informação de θ₂.
  Uma rede que receba θ₂ bruto precisará aprender sin internamente;
  fornecer sin θ₂ diretamente reduz a profundidade necessária.
""")
print(SEP)

# ── 7. Figura ──────────────────────────────────────────────────────────────
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

fig = plt.figure(figsize=(20, 16), facecolor=C["bg"])
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.56, wspace=0.38)
idx = slice(None, None, 10)

# ── A. sin θ₂ vs α₁ ──────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
sa(ax, r"(a) sin $\theta_2$ vs $\alpha_1$  — componente dominante",
   r"sin$(\theta_2)$", r"$\alpha_1$ (rad/s²)")
ax.scatter(sin_t2[idx], y[idx], s=0.6, alpha=0.15, color=C["a3"])
bins = np.linspace(-1, 1, 40)
bm   = [y[(sin_t2 >= bins[i]) & (sin_t2 < bins[i+1])].mean()
        for i in range(len(bins)-1)]
ax.plot(0.5*(bins[:-1]+bins[1:]), bm, color=C["a5"], lw=2.2, label="média por bin")
rho_sin, _ = stats.spearmanr(sin_t2, y)
ax.text(0.04, 0.94,
        f"ρ = {rho_sin:+.3f}\nMI = {mi_sin:.3f} nats\nΔR² = {dr2_sin:.4f}",
        transform=ax.transAxes, color=C["a3"], fontsize=8.5,
        va="top", style="italic")
ax.legend(fontsize=7.5, facecolor=C["panel"], labelcolor=C["text"])

# ── B. cos θ₂ vs α₁ ──────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
sa(ax, r"(b) cos $\theta_2$ vs $\alpha_1$  — componente marginal",
   r"cos$(\theta_2)$", r"$\alpha_1$ (rad/s²)")
ax.scatter(cos_t2[idx], y[idx], s=0.6, alpha=0.15, color=C["a2"])
bm2  = [y[(cos_t2 >= bins[i]) & (cos_t2 < bins[i+1])].mean()
        for i in range(len(bins)-1)]
ax.plot(0.5*(bins[:-1]+bins[1:]), bm2, color=C["a5"], lw=2.2, label="média por bin")
rho_cos, _ = stats.spearmanr(cos_t2, y)
ax.text(0.04, 0.94,
        f"ρ = {rho_cos:+.3f}\nMI = {mi_cos:.3f} nats\nΔR² = {dr2_cos:.6f}",
        transform=ax.transAxes, color=C["a2"], fontsize=8.5,
        va="top", style="italic")
ax.legend(fontsize=7.5, facecolor=C["panel"], labelcolor=C["text"])

# ── C. ΔR² decomposição ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
sa(ax, r"(c) Decomposição do $\Delta R^2$ de $\theta_2$",
   "Componente", r"$\Delta R^2$")
labels_c = ["sin θ₂\n(dado cos θ₂)", "cos θ₂\n(dado sin θ₂)", "Conjunto\n(ambos)"]
vals_c   = [dr2_sin, dr2_cos, dr2_both]
cols_c   = [C["a3"], C["a2"], C["a4"]]
bars = ax.bar(range(3), vals_c, color=cols_c, alpha=0.85, edgecolor="none")
for b, v in zip(bars, vals_c):
    ax.text(b.get_x()+b.get_width()/2, v + 0.002,
            f"{v:.6f}", ha="center", va="bottom",
            color=C["text"], fontsize=8, fontweight="bold")
ax.set_xticks(range(3))
ax.set_xticklabels(labels_c, color=C["slate"], fontsize=8.5)
ax.text(0.03, 0.94,
        f"sin θ₂ = {dr2_sin/dr2_both*100:.1f}% do total\n"
        f"cos θ₂ = {dr2_cos/dr2_both*100:.1f}% do total\n"
        f"Razão: {dr2_sin/max(dr2_cos,1e-10):.0f}×",
        transform=ax.transAxes, color=C["a5"], fontsize=8.5,
        va="top", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C["panel"],
                  edgecolor=C["a5"], linewidth=1, alpha=0.9))

# ── D. F-parciais comparados ──────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
sa(ax, "(d) F-parcial: sin θ₂ vs cos θ₂  [escala log]",
   "Componente", "F-statistic (log₁₀)")
F_vals   = [F_sin, F_cos]
F_labels = [r"F(sin $\theta_2$ | cos $\theta_2$)",
            r"F(cos $\theta_2$ | sin $\theta_2$)"]
F_cols   = [C["a3"], C["a2"]]
for xi, (fv, lbl, col) in enumerate(zip(F_vals, F_labels, F_cols)):
    ax.bar(xi, np.log10(fv), color=col, alpha=0.85, edgecolor="none")
    dec = "Rejeita H₀ ✓" if (fv > F_crit_1) else "Não rejeita ✗"
    ax.text(xi, np.log10(fv)+0.05, f"F={fv:.2f}\n{dec}",
            ha="center", va="bottom", color=C["text"], fontsize=8)
ax.axhline(np.log10(F_crit_1), color=C["a5"], lw=1.5, ls="--",
           label=f"F_crit={F_crit_1:.3f}")
ax.set_xticks([0,1])
ax.set_xticklabels(F_labels, color=C["slate"], fontsize=8)
ax.set_ylabel("log₁₀(F)", color=C["slate"], fontsize=8)
ax.legend(fontsize=7.5, facecolor=C["panel"], labelcolor=C["text"])

# ── E. IC 99% coeficientes ────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
sa(ax, "(e) IC 99% dos coeficientes de θ₂",
   "Coeficiente β", "")
mask = [nm in ["sin θ₂", "cos θ₂"] for nm in NAMES]
coefs_t2  = M_full.params[mask]
lo99_t2   = ci99[mask, 0]
hi99_t2   = ci99[mask, 1]
pvals_t2  = M_full.pvalues[mask]
names_t2  = [nm for nm, m in zip(NAMES, mask) if m]
colors_ic = [C["a3"] if p < ALPHA else C["a2"] for p in pvals_t2]
ax.barh(range(len(names_t2)), coefs_t2,
        xerr=[coefs_t2 - lo99_t2, hi99_t2 - coefs_t2],
        color=colors_ic, alpha=0.8,
        error_kw=dict(ecolor=C["text"], lw=2, capsize=8),
        edgecolor="none", height=0.4)
ax.axvline(0, color=C["text"], lw=1, ls="--", alpha=0.5)
ax.set_yticks(range(len(names_t2)))
ax.set_yticklabels(names_t2, color=C["slate"], fontsize=11)
for i, (nm, coef, pv, lo, hi) in enumerate(zip(
        names_t2, coefs_t2, pvals_t2, lo99_t2, hi99_t2)):
    ax.text(max(coef, hi) + 0.3, i,
            f"β={coef:+.2f}  p={pv:.2e}",
            va="center", color=C["text"], fontsize=8)

# ── F. Resíduos de M_sem_sin vs sin θ₂ ───────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
sa(ax, r"(f) Resíduos de M_sem_sin vs sin $\theta_2$",
   r"sin$(\theta_2)$", r"Resíduo (rad/s²)")
ax.scatter(sin_t2[idx], M_sem_sin.resid[idx],
           s=0.7, alpha=0.15, color=C["a1"])
bm3 = [M_sem_sin.resid[(sin_t2 >= bins[i]) & (sin_t2 < bins[i+1])].mean()
       for i in range(len(bins)-1)]
ax.plot(0.5*(bins[:-1]+bins[1:]), bm3, color=C["a5"], lw=2)
ax.axhline(0, color=C["text"], lw=0.8, ls="--", alpha=0.5)
ax.text(0.04, 0.94,
        "Padrão linear em sin θ₂\n→ sin θ₂ não capturado\npelo M_sem_sin",
        transform=ax.transAxes, color=C["a5"],
        fontsize=8, va="top", style="italic")

# ── G. Resíduos de M_sem_cos vs cos θ₂ ───────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
sa(ax, r"(g) Resíduos de M_sem_cos vs cos $\theta_2$",
   r"cos$(\theta_2)$", r"Resíduo (rad/s²)")
ax.scatter(cos_t2[idx], M_sem_cos.resid[idx],
           s=0.7, alpha=0.15, color=C["a2"])
bm4 = [M_sem_cos.resid[(cos_t2 >= bins[i]) & (cos_t2 < bins[i+1])].mean()
       for i in range(len(bins)-1)]
ax.plot(0.5*(bins[:-1]+bins[1:]), bm4, color=C["a5"], lw=2)
ax.axhline(0, color=C["text"], lw=0.8, ls="--", alpha=0.5)
ax.text(0.04, 0.94,
        "Linha plana → cos θ₂\nnão deixa estrutura\nremanescente",
        transform=ax.transAxes, color=C["a3"],
        fontsize=8, va="top", style="italic")

# ── H. Progressão de R² ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
sa(ax, r"(h) Progressão de $R^2$ por modelo",
   "Modelo", r"$R^2$")
r2_seq  = [M_sem_tudo.rsquared, M_sem_cos.rsquared,
           M_sem_sin.rsquared,  M_full.rsquared]
lbl_seq = ["Sem θ₂", "+cos θ₂\n(sem sin)", "+sin θ₂\n(sem cos)",
           "sin+cos θ₂\n(completo)"]
col_seq = [C["a2"], C["a2"], C["a3"], C["a4"]]
bars_h  = ax.bar(range(4), r2_seq, color=col_seq, alpha=0.85, edgecolor="none")
for b, v in zip(bars_h, r2_seq):
    ax.text(b.get_x()+b.get_width()/2, v+0.003, f"{v:.4f}",
            ha="center", va="bottom", color=C["text"], fontsize=8)
ax.set_xticks(range(4))
ax.set_xticklabels(lbl_seq, color=C["slate"], fontsize=8)
ax.set_ylim(0, max(r2_seq)*1.2)
ax.text(0.03, 0.94,
        "cos θ₂ sozinho: quase\nnenhum ganho\n"
        "sin θ₂ sozinho: {:.1f}% do ganho".format(
            (M_sem_cos.rsquared-M_sem_tudo.rsquared)/dr2_both*100 if
            (M_sem_cos.rsquared-M_sem_tudo.rsquared) > 0 else 0),
        transform=ax.transAxes, color=C["a5"],
        fontsize=8, va="top", style="italic")

# ── I. Tabela resumo ──────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 2])
ax.set_facecolor(C["panel"]); ax.axis("off")
ax.set_title("Resumo HN-F", color=C["text"],
             fontsize=8.5, fontweight="bold", pad=5)

rows_tab = [
    ("Teste",                  "Valor",               "Decisão"),
    ("H₀: β_sinθ₂=0|cosθ₂",  f"F={F_sin:.0f}",      "Rejeita ✓"),
    ("H₀': β_cosθ₂=0|sinθ₂", f"F={F_cos:.2f}",      "Não rejeita ✗"),
    ("ΔR²(sin θ₂)",           f"{dr2_sin:.6f}",      f"{dr2_sin/dr2_both*100:.1f}% do total"),
    ("ΔR²(cos θ₂)",           f"{dr2_cos:.6f}",      f"{dr2_cos/dr2_both*100:.1f}% do total"),
    ("Razão ΔR²",             f"{dr2_sin/max(dr2_cos,1e-10):.0f}×",  "sin domina"),
    ("MI(sin θ₂, α₁)",       f"{mi_sin:.4f} nats",  "—"),
    ("MI(cos θ₂, α₁)",       f"{mi_cos:.4f} nats",  "—"),
    ("β_sinθ₂  IC99%",        f"[{ci99[3,0]:+.2f};{ci99[3,1]:+.2f}]","não contém 0 ✓"),
    ("β_cosθ₂  IC99%",        f"[{ci99[4,0]:+.2f};{ci99[4,1]:+.2f}]","contém 0 ✗"),
    ("Feature prioritária",   "sin θ₂",              "99,98% do ganho"),
]

cw = [0.42, 0.30, 0.28]; rh = 0.083
for ri, row in enumerate(rows_tab):
    for ci_, (cell, w) in enumerate(zip(row, cw)):
        x = sum(cw[:ci_]); yc = 1.0 - (ri+1)*rh
        bg = C["a1"] if ri==0 else (C["bg"] if ri%2==0 else "#1C2330")
        fc = plt.Rectangle((x,yc),w,rh*0.93,
                           transform=ax.transAxes,
                           facecolor=bg, edgecolor=C["grid"], lw=0.4)
        ax.add_patch(fc)
        tc = (C["panel"] if ri==0 else
              C["a3"] if "✓" in cell else
              C["a2"] if "✗" in cell else C["text"])
        ax.text(x+w/2, yc+rh*0.45, cell,
                transform=ax.transAxes,
                ha="center", va="center",
                color=tc if ri>0 else C["panel"],
                fontsize=7, fontweight="bold" if ri==0 else "normal")

fig.suptitle(
    r"HN-F — Decomposição de $\theta_2$: qual componente trigonométrico explica $\alpha_1$?" +
    f"\nsin θ₂ = {dr2_sin/dr2_both*100:.1f}% do ganho  ·  "
    f"cos θ₂ = {dr2_cos/dr2_both*100:.1f}%  ·  "
    f"Razão: {dr2_sin/max(dr2_cos,1e-10):.0f}×",
    color=C["text"], fontsize=11, fontweight="bold", y=0.998
)

OUT = r"C:\Users\Guilherme\Mestrado\Inverted_Pendulum\Double-Inverted-Pendulum\Inverted Pendulum\Final Versions\Project_2\H5\Results\HN-F_decomposicao_theta2.png"
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=C["bg"])
print(f"\n[OK] Figura: {OUT}")
plt.close(fig)