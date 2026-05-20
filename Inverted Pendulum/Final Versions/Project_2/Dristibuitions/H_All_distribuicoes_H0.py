"""
Distribuições sob H₀ — HN-B (Não-linearidade Trigonométrica)
Quatro testes × duas variáveis = 8 painéis de distribuição F
+ painel consolidado com curva p-valor vs ΔR²
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from scipy import stats

ALPHA = 0.05

# ── Parâmetros dos quatro testes × duas variáveis ──────────────────────────
TESTS = [
    # ── α₁ ─────────────────────────────────────────────────────────────────
    dict(
        var="α₁", test="RESET potência 2 (par)",
        h0=r"$H_0$: estrutura polinomial ausente nos resíduos",
        df1=1, df2=49994,
        F_obs=1.1464, ncp=1.1464, p_obs=0.2843,
        rejects=False,
        color="#3FB950",
        note="p=0,28 > α=0,05\nT_obs dentro da curva\n→ Estrutura PAR ausente ✓\nNão-linearidade NÃO\né polinomial",
        dr2=None,
    ),
    dict(
        var="α₁", test="RESET potência 3 (ímpar)",
        h0=r"$H_0$: estrutura trigonométrica ausente nos resíduos",
        df1=2, df2=49993,
        F_obs=724.14, ncp=1448.29, p_obs=0.0,
        rejects=True,
        color="#F87171",
        note="p ≈ 0 << α=0,05\nT_obs >> F_crit\n→ Estrutura ÍMPAR presente ✓\nsin/cos confirmado",
        dr2=9.0,
    ),
    dict(
        var="α₁", test="F(trig | poly)",
        h0=r"$H_0$: termos trig = 0 dado poly presente",
        df1=6, df2=49987,
        F_obs=834.10, ncp=5004.61, p_obs=0.0,
        rejects=True,
        color="#F87171",
        note="F=834 >> F_crit\nTrig acrescenta\nmuitíssimo além de poly\nR²_trig/R²_poly = 42×",
        dr2=None,
    ),
    dict(
        var="α₁", test="F(poly | trig)",
        h0=r"$H_0$: termos poly = 0 dado trig presente",
        df1=6, df2=49987,
        F_obs=27.14, ncp=162.83, p_obs=0.0,
        rejects=True,
        color="#FFA657",
        note="F=27 > F_crit\nPoly acrescenta\nmarginalmente\n31× menor que trig",
        dr2=None,
    ),
    # ── α₂ ─────────────────────────────────────────────────────────────────
    dict(
        var="α₂", test="RESET potência 2 (par)",
        h0=r"$H_0$: estrutura polinomial ausente nos resíduos",
        df1=1, df2=49994,
        F_obs=2.4333, ncp=2.4333, p_obs=0.1188,
        rejects=False,
        color="#3FB950",
        note="p=0,12 > α=0,05\nT_obs dentro da curva\n→ Estrutura PAR ausente ✓\nMesma assinatura de α₁",
        dr2=None,
    ),
    dict(
        var="α₂", test="RESET potência 3 (ímpar)",
        h0=r"$H_0$: estrutura trigonométrica ausente nos resíduos",
        df1=2, df2=49993,
        F_obs=4131.15, ncp=8262.31, p_obs=0.0,
        rejects=True,
        color="#F87171",
        note="F=4131 >> F_crit\np ≈ 0\nEfeito ainda maior\nque em α₁",
        dr2=10.3,
    ),
    dict(
        var="α₂", test="F(trig | poly)",
        h0=r"$H_0$: termos trig = 0 dado poly presente",
        df1=6, df2=49987,
        F_obs=974.34, ncp=5846.06, p_obs=0.0,
        rejects=True,
        color="#F87171",
        note="F=974 >> F_crit\nTrig domina\nR²_trig/R²_poly = 37×",
        dr2=None,
    ),
    dict(
        var="α₂", test="F(poly | trig)",
        h0=r"$H_0$: termos poly = 0 dado trig presente",
        df1=6, df2=49987,
        F_obs=39.49, ncp=236.95, p_obs=0.0,
        rejects=True,
        color="#FFA657",
        note="F=39 > F_crit\nPoly acrescenta\nmarginalmente\n25× menor que trig",
        dr2=None,
    ),
]

BG    = "#0D1117"; PANEL = "#161B22"; GRID = "#21262D"
TEXT  = "#E6EDF3"; SLATE = "#64748B"; BLUE = "#58A6FF"
RED   = "#F87171"; GREEN = "#3FB950"; AMBER = "#FFA657"

def sa(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=SLATE, labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_color(GRID); sp.set_linewidth(0.5)
    ax.grid(True, color=GRID, lw=0.35, alpha=0.6)
    if title:  ax.set_title(title,   color=TEXT,  fontsize=8,
                             fontweight="bold", pad=4)
    if xlabel: ax.set_xlabel(xlabel, color=SLATE, fontsize=7.5)
    if ylabel: ax.set_ylabel(ylabel, color=SLATE, fontsize=7.5)

def plot_h0(ax, t):
    df1, df2 = t["df1"], t["df2"]
    F_obs    = t["F_obs"]
    F_crit   = stats.f.ppf(1 - ALPHA, df1, df2)
    p_val    = t["p_obs"] if t["p_obs"] > 0 else 1 - stats.f.cdf(F_obs, df1, df2)

    # Eixo X
    x_max = F_obs * 1.3 if F_obs <= F_crit * 5 else F_crit * 4.5
    x   = np.linspace(1e-4, x_max, 2000)
    y_h0 = stats.f.pdf(x, df1, df2)

    sa(ax,
       f"{t['var']} — {t['test']}",
       f"F({df1}, {df2})",
       "Probabilidade de ocorrência sob H₀")

    # Curva H₀
    ax.plot(x, y_h0, color=BLUE, lw=2.2,
            path_effects=[pe.Stroke(linewidth=3.5, foreground=PANEL, alpha=0.5),
                           pe.Normal()],
            label=f"PDF sob H₀: F({df1},{df2})")

    # Área de rejeição
    x_rej = x[x >= F_crit]
    ax.fill_between(x_rej, stats.f.pdf(x_rej, df1, df2),
                    color=RED, alpha=0.28, label=f"Rejeição α={ALPHA}")

    # F_crítico
    ax.axvline(F_crit, color=AMBER, lw=1.6, ls=(0,(5,3)),
               label=f"F_crit={F_crit:.3f}")

    # T_obs
    col_obs = t["color"]
    if F_obs <= x_max:
        ax.axvline(F_obs, color=col_obs, lw=2, ls=":",
                   label=f"T_obs={F_obs:.2f}")
        ax.text(F_obs + x_max*0.02,
                stats.f.pdf(min(F_obs, F_crit*0.9), df1, df2) * 0.6,
                f"{F_obs:.2f}",
                color=col_obs, fontsize=7,
                path_effects=[pe.Stroke(linewidth=2, foreground=PANEL, alpha=0.8),
                               pe.Normal()])
    else:
        ax.annotate("", xy=(x_max*0.98, max(y_h0)*0.08),
                    xytext=(x_max*0.76, max(y_h0)*0.08),
                    arrowprops=dict(arrowstyle="->", color=col_obs, lw=2))
        ax.text(x_max*0.75, max(y_h0)*0.095,
                f"T_obs={F_obs:,.0f} →→",
                color=col_obs, fontsize=7.5, ha="right", fontweight="bold")
        ax.plot([], [], color=col_obs, lw=2, ls=":",
                label=f"T_obs={F_obs:,.0f} ({'rejeita ✓' if t['rejects'] else 'não rejeita'})")

    # Badge
    p_str = f"{p_val:.4f}" if p_val > 1e-4 else "≈ 0"
    badge = f"p = {p_str}\n{t['note']}"
    ax.text(0.97, 0.97, badge,
            transform=ax.transAxes, ha="right", va="top",
            color=col_obs, fontsize=7, style="italic", linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
                      edgecolor=col_obs, linewidth=1, alpha=0.93))

    ax.legend(fontsize=6.5, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, loc="upper center")
    ax.set_xlim(0, x_max); ax.set_ylim(bottom=0)

# ── Figura principal: 4×2 painéis de distribuição ─────────────────────────
fig = plt.figure(figsize=(20, 24), facecolor=BG)
gs  = gridspec.GridSpec(5, 4, figure=fig, hspace=0.60, wspace=0.38)

# Linha 0-3: os 8 testes (4 por variável, 2 colunas)
positions = [
    (0,0),(0,1),(0,2),(0,3),   # α₁: RESET2, RESET3, F(trig|poly), F(poly|trig)
    (1,0),(1,1),(1,2),(1,3),   # α₂: mesma ordem
]
for ti, (row, col) in enumerate(positions):
    ax = fig.add_subplot(gs[row, col])
    plot_h0(ax, TESTS[ti])

# ── Painel H₀ vs H₁ (log) para os dois RESET ──────────────────────────────
for pi, (ti, col_gs) in enumerate([(1, slice(0,2)), (5, slice(2,4))]):
    t = TESTS[ti]
    ax = fig.add_subplot(gs[2, col_gs])
    df1, df2, F_obs, ncp = t["df1"], t["df2"], t["F_obs"], t["ncp"]
    F_crit = stats.f.ppf(1-ALPHA, df1, df2)
    sa(ax, f"H₀ vs H₁ — RESET ímpar — {t['var']}\n(escala log)",
       "log₁₀(F)", "log₁₀(PDF)")
    x_log = np.logspace(-1, np.log10(F_obs*1.5), 3000)
    y0 = stats.f.pdf(x_log, df1, df2)
    y1 = stats.ncf.pdf(x_log, df1, df2, nc=ncp)
    m0 = y0>1e-300; m1 = y1>1e-300
    if m0.any(): ax.plot(np.log10(x_log[m0]), np.log10(y0[m0]),
                         color=BLUE, lw=2, label="PDF sob H₀")
    if m1.any(): ax.plot(np.log10(x_log[m1]), np.log10(y1[m1]),
                         color=GREEN, lw=2, label=f"PDF sob H₁ (λ={ncp:.0f})")
    ax.axvline(np.log10(F_crit), color=AMBER, lw=1.5, ls="--",
               label=f"F_crit={F_crit:.3f}")
    ax.axvline(np.log10(F_obs), color=RED, lw=2, ls=":",
               label=f"T_obs={F_obs:.0f}")
    power = 1 - stats.ncf.cdf(F_crit, df1, df2, nc=ncp)
    ax.text(0.03, 0.12,
            f"H₀ centrada em ~1\nH₁ centrada em ~{F_obs:.0f}\n"
            f"Poder = {power*100:.2f}%",
            transform=ax.transAxes, color=AMBER, fontsize=8,
            va="bottom", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL,
                      edgecolor=AMBER, linewidth=1, alpha=0.9))
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.set_xlabel("log₁₀(F)", color=SLATE, fontsize=8)

# ── Painel: o contraste RESET2 vs RESET3 (o diagnóstico central) ──────────
ax = fig.add_subplot(gs[3, :])
sa(ax,
   "O contraste diagnóstico central de HN-B: RESET par vs ímpar\n"
   "Por que p=2 não rejeita e p=3 rejeita — a assinatura matemática da não-linearidade trigonométrica",
   "F (escala log₁₀)", "Probabilidade de ocorrência sob H₀")

for ti_pair, col_p, lbl in [
    (0, GREEN,  r"RESET p=2 α₁  (F=1,15  NÃO rejeita ✓)"),
    (1, RED,    r"RESET p=3 α₁  (F=724   REJEITA ✓)"),
    (4, "#64D8A8", r"RESET p=2 α₂  (F=2,43  NÃO rejeita ✓)"),
    (5, "#F87171", r"RESET p=3 α₂  (F=4131  REJEITA ✓)"),
]:
    t = TESTS[ti_pair]
    df1, df2 = t["df1"], t["df2"]
    F_obs = t["F_obs"]
    x_grid = np.linspace(0.01, 6, 500)
    y_pdf  = stats.f.pdf(x_grid, df1, df2)
    ax.semilogy(x_grid, y_pdf, color=col_p, lw=2, label=lbl)
    F_crit = stats.f.ppf(1-ALPHA, df1, df2)
    if F_obs <= 6:
        ax.axvline(F_obs, color=col_p, lw=1.5, ls=":", alpha=0.8)
        ax.text(F_obs + 0.05, 0.8, f"T={F_obs:.2f}",
                color=col_p, fontsize=7.5, va="top")

ax.axvline(stats.f.ppf(1-ALPHA, 1, 49994), color=AMBER, lw=1.8, ls="--",
           label=f"F_crit≈{stats.f.ppf(1-ALPHA,1,49994):.2f} (df1=1)")
ax.axvline(stats.f.ppf(1-ALPHA, 2, 49993), color=AMBER, lw=1.4, ls=":",
           label=f"F_crit≈{stats.f.ppf(1-ALPHA,2,49993):.2f} (df1=2)")

ax.text(0.60, 0.85,
        "Interpretação:\nRESET p=2 (par): T_obs cai dentro da curva → não rejeita\n"
        "RESET p=3 (ímpar): T_obs >> F_crit → rejeita\n"
        "Isso é a assinatura de sin(θ) — função ímpar — nos resíduos",
        transform=ax.transAxes, color=AMBER, fontsize=9,
        va="top", style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
                  edgecolor=AMBER, linewidth=1.2, alpha=0.93))
ax.set_xlim(0, 6); ax.set_ylim(1e-4, 10)
ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID,
          labelcolor=TEXT, loc="upper right")

# ── Painel: p-valor vs ΔR² consolidado ────────────────────────────────────
ax = fig.add_subplot(gs[4, :])
sa(ax,
   "p-valor em função de ΔR² por n — contexto de HN-B",
   "ΔR² (%)", "p-valor (escala log)")

dr2_range = np.linspace(0.00001, 0.12, 800)
for n_val, col_n, lbl in [
    (1000,   "#C084FC", "n = 1.000"),
    (10000,  SLATE,     "n = 10.000"),
    (50000,  BLUE,      "n = 50.000  ← nosso caso"),
    (100000, RED,       "n = 100.000"),
]:
    F_approx = (dr2_range * n_val) / (2 * 0.7)
    p_approx  = 1 - stats.f.cdf(F_approx, 2, n_val - 5)
    ax.semilogy(dr2_range * 100, np.maximum(p_approx, 1e-300),
                color=col_n, lw=2, label=lbl)

ax.axhline(ALPHA, color=TEXT, lw=1.2, ls="--", alpha=0.7, label=f"α = {ALPHA}")

# Marcar os pontos dos testes RESET p=3
for F_val, nm, col_m, dr2_pct, lbl_m in [
    (724.14,  "α₁", RED,    9.0,  "RESET p=3 α₁\nΔR²≈9%"),
    (4131.15, "α₂", "#F87171", 10.3, "RESET p=3 α₂\nΔR²≈10%"),
]:
    p_mark = 1 - stats.f.cdf(F_val, 2, 49993)
    ax.scatter([dr2_pct], [max(p_mark, 1e-300)],
               s=150, color=col_m, zorder=6, marker="*")
    ax.annotate(lbl_m,
                xy=(dr2_pct, max(p_mark, 1e-300)),
                xytext=(dr2_pct + 0.5, max(p_mark, 1e-300) * 5),
                color=col_m, fontsize=8, style="italic",
                arrowprops=dict(arrowstyle="->", color=col_m, lw=1.2))

ax.legend(fontsize=8.5, facecolor=PANEL, edgecolor=GRID,
          labelcolor=TEXT, loc="upper right")
ax.set_xlim(0, 12)

fig.suptitle(
    "HN-B — Distribuições sob H₀: Não-linearidade Trigonométrica?\n"
    "Quatro testes × duas variáveis  ·  "
    "RESET par não rejeita + RESET ímpar rejeita = assinatura de sin(θ)  ·  α = 0,05",
    color=TEXT, fontsize=11, fontweight="bold", y=0.998
)

OUT = "/mnt/user-data/outputs/HNB_distribuicoes_H0.png"
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"[OK] {OUT}")
plt.close(fig)
