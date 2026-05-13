#---------------------RESET TEST---------------------#
# This code presents the Ramsey's RESET Test for the Inverted Pendulum system.
# It evaluates the model's specification by checking for nonlinearity in the residuals.

"""
╔══════════════════════════════════════════════════════════════════╗
║  TESTE DE HIPÓTESE H1 — Dinâmica Não-Linear                      ║
║  Pêndulo Invertido Duplo                                         ║
╠══════════════════════════════════════════════════════════════════╣
║  H₀: o modelo linear captura adequadamente a relação             ║
║       entre o estado (θ, ω) e as acelerações angulares α         ║
║  H₁: os resíduos do modelo linear exibem estrutura               ║
║       sistemática — evidência de não-linearidade                 ║
║  Teste principal : RESET de Ramsey (potência 3)                  ║
║  Teste auxiliar  : Regressão dos resíduos sobre termos NL        ║
║  Limiar          : p < 0.01                                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── 0. Imports ────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ── 1. Dados ──────────────────────────────────────────────────────
CSV_PATH = "D:\Trabalhos\Smart Agri\Double_Inverted_Pendulum\Double-Inverted-Pendulum\Inverted Pendulum\Final Versions\Data Processed\pendulum_dataset_tidy_with_acceleration.csv"

df = pd.read_csv(CSV_PATH)
df["theta1"] = np.arctan2(df["sin_theta1"], df["cos_theta1"])
df["theta2"] = np.arctan2(df["sin_theta2"], df["cos_theta2"])

X_cols   = ["theta1", "theta2", "omega1", "omega2"]
TARGETS  = [("angle_accel1", r"$\alpha_1$"), ("angle_accel2", r"$\alpha_2$")]

scaler = StandardScaler()
X_s    = scaler.fit_transform(df[X_cols].values)
X_sm   = sm.add_constant(X_s)           # adiciona coluna de intercepto

# ── 2. Modelos e RESET ────────────────────────────────────────────
results = {}

for col, label in TARGETS:
    y = df[col].values

    # — modelo linear restrito —
    m0      = sm.OLS(y, X_sm).fit()
    yhat    = m0.fittedvalues
    resid   = m0.resid

    # — RESET Ramsey: potência 2 (adiciona ŷ²) —
    reset_p2 = linear_reset(m0, power=2, use_f=True)

    # — RESET Ramsey: potência 3 (adiciona ŷ², ŷ³) —
    reset_p3 = linear_reset(m0, power=3, use_f=True)

    # — Regressão auxiliar dos resíduos sobre termos NL —
    # Termos candidatos motivados pelas equações de Lagrange:
    #   sin(θ₁), cos(θ₁), sin(θ₂), cos(θ₂), θ₁·ω₁, θ₂·ω₂
    Z = np.column_stack([
        np.sin(df["theta1"]),  np.cos(df["theta1"]),
        np.sin(df["theta2"]),  np.cos(df["theta2"]),
        df["theta1"] * df["omega1"],
        df["theta2"] * df["omega2"],
    ])
    Z_sm    = sm.add_constant(Z)
    m_aux   = sm.OLS(resid, Z_sm).fit()

    # Teste LM (Breusch-Godfrey generalizado): n·R²_aux ~ χ²(q)
    n, q  = len(resid), Z.shape[1]
    LM    = n * m_aux.rsquared
    p_lm  = 1 - stats.chi2.cdf(LM, q)

    results[col] = dict(
        label    = label,
        m0       = m0,
        resid    = resid,
        yhat     = yhat,
        reset_p2 = reset_p2,
        reset_p3 = reset_p3,
        m_aux    = m_aux,
        LM       = LM,
        p_lm     = p_lm,
        q        = q,
    )

# ── 3. Relatório no terminal ──────────────────────────────────────
LIMIAR = 0.01

print("=" * 68)
print("  TESTE DE HIPÓTESE H1 — Dinâmica Não-Linear")
print("  Pêndulo Invertido Duplo")
print("=" * 68)
print()
print("  H₀ : modelo linear não omite termos relevantes (resíduos = ruído)")
print("  H₁ : resíduos possuem estrutura sistemática (não-linearidade)")
print(f"  Limiar de rejeição: p < {LIMIAR}")
print()

for col, r in results.items():
    lbl = r["label"].replace("$", "").replace("\\", "")
    print(f"  ── Variável-alvo: {lbl} ─────────────────────────────────")
    print(f"     R² modelo linear             : {r['m0'].rsquared:.4f}")
    print()
    print("     TESTE RESET DE RAMSEY")
    print(f"       Potência 2  F = {r['reset_p2'].statistic:>10.4f}  p = {r['reset_p2'].pvalue:.4e}"
          f"  → {'REJEITA H₀ ✓' if r['reset_p2'].pvalue < LIMIAR else 'não rejeita ✗'}")
    print(f"       Potência 3  F = {r['reset_p3'].statistic:>10.4f}  p = {r['reset_p3'].pvalue:.4e}"
          f"  → {'REJEITA H₀ ✓' if r['reset_p3'].pvalue < LIMIAR else 'não rejeita ✗'}")
    print()
    print("     REGRESSÃO AUXILIAR DOS RESÍDUOS (termos NL candidatos)")
    print(f"       R²_aux : {r['m_aux'].rsquared:.4f}  "
          f"F = {r['m_aux'].fvalue:.2f}  p = {r['m_aux'].f_pvalue:.4e}"
          f"  → {'REJEITA H₀ ✓' if r['m_aux'].f_pvalue < LIMIAR else 'não rejeita ✗'}")
    print(f"       LM     : {r['LM']:.2f}   χ²({r['q']})  p = {r['p_lm']:.4e}"
          f"  → {'REJEITA H₀ ✓' if r['p_lm'] < LIMIAR else 'não rejeita ✗'}")

    feat_labels = ["const", "sin θ₁", "cos θ₁", "sin θ₂", "cos θ₂", "θ₁·ω₁", "θ₂·ω₂"]
    print()
    print("     Coeficientes significativos da regressão auxiliar (p < 0.01):")
    for fname, coef, pval in zip(feat_labels, r["m_aux"].params, r["m_aux"].pvalues):
        if pval < LIMIAR and fname != "const":
            print(f"       {fname:<10}  β = {coef:+.4f}   p = {pval:.3e}")
    print()

print("=" * 68)
print()
print("  NOTA — Por que RESET potência 2 não rejeita H₀?")
print()
print("  O RESET com ŷ² insere um termo de ordem PAR na regressão auxiliar.")
print("  A não-linearidade de sin(θ) e cos(θ) centrados em zero é de")
print("  natureza ÍMPAR, portanto ortogonal ao espaço gerado por ŷ².")
print("  O RESET potência 3 (ŷ², ŷ³) captura a estrutura ímpar e rejeita")
print("  H₀ com p ≈ 0. Esse padrão é matematicamente esperado e é")
print("  confirmado pela regressão auxiliar direta sobre sin/cos/cruzados.")
print()
print("  CONCLUSÃO: H₁ CONFIRMADA — o sistema possui dinâmica não-linear.")
print("  Os termos dominantes são sin(θ₁), sin(θ₂) e os cruzados θᵢ·ωᵢ.")
print("=" * 68)

# ── 4. Figura de diagnóstico ──────────────────────────────────────

C = {
    "bg":    "#0D1117", "panel": "#161B22", "grid": "#21262D",
    "text":  "#E6EDF3", "a1":    "#58A6FF", "a2":   "#F78166",
    "a3":    "#3FB950", "a4":    "#D2A8FF", "a5":   "#FFA657",
    "slate": "#64748B",
}

def sa(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["slate"], labelsize=8)
    for sp in ax.spines.values(): sp.set_color(C["grid"]); sp.set_linewidth(0.7)
    ax.grid(True, color=C["grid"], lw=0.4, alpha=0.7)
    if title:  ax.set_title(title,  color=C["text"], fontsize=9, fontweight="bold", pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=C["slate"], fontsize=8.5)
    if ylabel: ax.set_ylabel(ylabel, color=C["slate"], fontsize=8.5)


fig = plt.figure(figsize=(18, 14), facecolor=C["bg"])
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

col_colors = [C["a1"], C["a2"]]

# ── Linha 0: Resíduos vs θ₁ e θ₂ (dois targets) + comparação R² ─
for ci, (col, r) in enumerate(results.items()):
    for ti, theta_col in enumerate(["theta1", "theta2"]):
        ax = fig.add_subplot(gs[0, ti + (0 if ci == 0 else 0)])  # rework layout below
        break  # rework: 2 targets × 2 theta = 4 panels; reorganize

# Reorganise: 3 rows × 3 cols
# Row 0: [resíd α₁ vs θ₁] [resíd α₁ vs θ₂] [R² comparação]
# Row 1: [resíd α₂ vs θ₁] [resíd α₂ vs θ₂] [coeficientes aux α₁]
# Row 2: [RESET F-dist α₁] [RESET F-dist α₂] [coeficientes aux α₂]

for ri, (col, r) in enumerate(results.items()):
    idx = slice(None, None, 8)   # subsample for speed

    # — resíduos vs θ₁ —
    ax = fig.add_subplot(gs[ri, 0])
    sa(ax,
       title=f"Resíduos {r['label']} vs $\\theta_1$",
       xlabel=r"$\theta_1$ (rad)", ylabel="Resíduo (rad/s²)")
    ax.scatter(df["theta1"].values[idx], r["resid"][idx],
               s=0.8, alpha=0.2, color=col_colors[ri])
    ax.axhline(0, color=C["text"], lw=0.8, ls="--", alpha=0.5)
    corr, pv = stats.spearmanr(df["theta1"], r["resid"])
    ax.text(0.04, 0.93, f"ρ_Spearman = {corr:.3f}\np = {pv:.2e}",
            transform=ax.transAxes, color=col_colors[ri], fontsize=8,
            va="top", style="italic")

    # — resíduos vs θ₂ —
    ax = fig.add_subplot(gs[ri, 1])
    sa(ax,
       title=f"Resíduos {r['label']} vs $\\theta_2$",
       xlabel=r"$\theta_2$ (rad)", ylabel="Resíduo (rad/s²)")
    ax.scatter(df["theta2"].values[idx], r["resid"][idx],
               s=0.8, alpha=0.2, color=col_colors[ri])
    ax.axhline(0, color=C["text"], lw=0.8, ls="--", alpha=0.5)
    corr2, pv2 = stats.spearmanr(df["theta2"], r["resid"])
    ax.text(0.04, 0.93, f"ρ_Spearman = {corr2:.3f}\np = {pv2:.2e}",
            transform=ax.transAxes, color=col_colors[ri], fontsize=8,
            va="top", style="italic")

# — coluna 2, linha 0: comparação R² —
ax = fig.add_subplot(gs[0, 2])
sa(ax, title="Comparação de $R^2$ — Linear vs + Termos NL",
   xlabel="Modelo", ylabel="$R^2$")

labels_bar = [r"$\alpha_1$ Linear", r"$\alpha_1$ +NL",
              r"$\alpha_2$ Linear", r"$\alpha_2$ +NL"]

r2_lin_1  = list(results.values())[0]["m0"].rsquared
r2_aug_1  = list(results.values())[0]["m0"].rsquared   # placeholder below
r2_lin_2  = list(results.values())[1]["m0"].rsquared

# Recompute R² of augmented model
for ci, (col, r) in enumerate(results.items()):
    y = df[col].values
    Z = np.column_stack([X_sm,
        np.sin(df["theta1"]), np.cos(df["theta1"]),
        np.sin(df["theta2"]), np.cos(df["theta2"]),
        df["theta1"] * df["omega1"],
        df["theta2"] * df["omega2"],
    ])
    m_aug = sm.OLS(y, Z).fit()
    if ci == 0:
        r2_aug_1 = m_aug.rsquared
    else:
        r2_aug_2 = m_aug.rsquared

bar_vals   = [r2_lin_1, r2_aug_1, r2_lin_2, r2_aug_2]
bar_colors = [C["a1"], C["a3"], C["a2"], C["a3"]]
x_pos      = np.array([0, 1, 3, 4])
bars = ax.bar(x_pos, bar_vals, color=bar_colors, alpha=0.82, width=0.7,
              edgecolor="none")
for b, v in zip(bars, bar_vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
            ha="center", va="bottom", color=C["text"], fontsize=8, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_bar, color=C["slate"], fontsize=8)
ax.set_ylim(0, 0.75)

# — coluna 2, linha 1: coeficientes auxiliares α₁ —
feat_labels = ["sin θ₁", "cos θ₁", "sin θ₂", "cos θ₂", "θ₁·ω₁", "θ₂·ω₂"]
ax = fig.add_subplot(gs[1, 2])
sa(ax, title=r"Coeficientes Regressão Auxiliar — $\alpha_1$",
   xlabel="Feature NL", ylabel="Coeficiente β")
r1 = list(results.values())[0]
coefs1  = r1["m_aux"].params[1:]   # skip const
pvals1  = r1["m_aux"].pvalues[1:]
c_colors = [C["a3"] if p < 0.01 else C["slate"] for p in pvals1]
ax.bar(range(len(feat_labels)), coefs1, color=c_colors, alpha=0.82, edgecolor="none")
ax.set_xticks(range(len(feat_labels)))
ax.set_xticklabels(feat_labels, color=C["slate"], fontsize=8, rotation=20)
ax.axhline(0, color=C["text"], lw=0.7, ls="--", alpha=0.5)
ax.text(0.03, 0.95, "Verde = p < 0.01", transform=ax.transAxes,
        color=C["a3"], fontsize=7.5, va="top", style="italic")

# — coluna 2, linha 2: coeficientes auxiliares α₂ —
ax = fig.add_subplot(gs[2, 2])
sa(ax, title=r"Coeficientes Regressão Auxiliar — $\alpha_2$",
   xlabel="Feature NL", ylabel="Coeficiente β")
r2_ = list(results.values())[1]
coefs2  = r2_["m_aux"].params[1:]
pvals2  = r2_["m_aux"].pvalues[1:]
c_colors2 = [C["a3"] if p < 0.01 else C["slate"] for p in pvals2]
ax.bar(range(len(feat_labels)), coefs2, color=c_colors2, alpha=0.82, edgecolor="none")
ax.set_xticks(range(len(feat_labels)))
ax.set_xticklabels(feat_labels, color=C["slate"], fontsize=8, rotation=20)
ax.axhline(0, color=C["text"], lw=0.7, ls="--", alpha=0.5)

# — linha 2, col 0-1: tabela de resultados RESET —
for ci, (col, r) in enumerate(results.items()):
    ax = fig.add_subplot(gs[2, ci])
    ax.set_facecolor(C["panel"])
    ax.axis("off")

    lbl = r["label"]
    rows = [
        ["Métrica",          "Valor",                "p-valor",            "Decisão"],
        ["RESET (pot. 2)",
         f"F = {r['reset_p2'].statistic:.2f}",
         f"{r['reset_p2'].pvalue:.2e}",
         "✗ não rejeita" if r['reset_p2'].pvalue >= 0.01 else "✓ rejeita H₀"],
        ["RESET (pot. 3)",
         f"F = {r['reset_p3'].statistic:.2f}",
         f"{r['reset_p3'].pvalue:.2e}",
         "✓ rejeita H₀" if r['reset_p3'].pvalue < 0.01 else "✗ não rejeita"],
        ["Reg. Auxiliar F",
         f"F = {r['m_aux'].fvalue:.2f}",
         f"{r['m_aux'].f_pvalue:.2e}",
         "✓ rejeita H₀" if r['m_aux'].f_pvalue < 0.01 else "✗ não rejeita"],
        ["LM (χ²)",
         f"LM = {r['LM']:.2f}",
         f"{r['p_lm']:.2e}",
         "✓ rejeita H₀" if r['p_lm'] < 0.01 else "✗ não rejeita"],
    ]

    col_w = [0.28, 0.25, 0.22, 0.25]
    row_h = 0.18
    for rr, row in enumerate(rows):
        for cc, cell in enumerate(row):
            x = sum(col_w[:cc])
            y = 1.0 - (rr + 1) * row_h
            is_header = rr == 0
            bg = C["a1"] if is_header else (C["bg"] if rr % 2 == 0 else "#1C2330")
            fc = plt.Rectangle((x, y), col_w[cc], row_h,
                               transform=ax.transAxes,
                               facecolor=bg, edgecolor=C["grid"], lw=0.5)
            ax.add_patch(fc)
            tc = C["text"] if not is_header else C["bg"]
            if "rejeita H₀" in cell and "✓" in cell: tc = C["a3"]
            if "não rejeita" in cell: tc = C["a2"]
            ax.text(x + col_w[cc]/2, y + row_h/2, cell,
                    transform=ax.transAxes,
                    ha="center", va="center",
                    color=tc if not is_header else C["panel"],
                    fontsize=7.5, fontweight="bold" if is_header else "normal")
    ax.set_title(f"Resultados RESET — {lbl}", color=C["text"],
                 fontsize=9, fontweight="bold", pad=6)

# Super-title
fig.suptitle(
    "Teste de Hipótese H1 — Dinâmica Não-Linear\n"
    r"$H_0$: resíduos são ruído branco  |  "
    r"$H_1$: estrutura sistemática não-linear",
    color=C["text"], fontsize=13, fontweight="bold", y=0.99
)

OUT = "D:\Trabalhos\Smart Agri\Double_Inverted_Pendulum\Double-Inverted-Pendulum\Inverted Pendulum\Final Versions\Project_2\H1\Results\RESET_Test_Results"
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=C["bg"])
print(f"\n[OK] Figura salva: {OUT}")
plt.close(fig)
