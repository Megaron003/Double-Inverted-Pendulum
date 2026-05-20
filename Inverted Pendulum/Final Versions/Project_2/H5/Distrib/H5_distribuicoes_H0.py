"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PROBABILIDADE DE OCORRÊNCIA SOB H₀ — HN-F                               ║
║  Decomposição de θ₂: sin θ₂ vs cos θ₂                                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  CONTEXTO METODOLÓGICO                                                   ║
║  Os dados foram extraídos de um simulador físico (MuJoCo), não de        ║
║  uma população real. Não há "população" no sentido clássico.             ║
║                                                                          ║
║  A distribuição de H₀ plotada é a distribuição AMOSTRAL da               ║
║  estatística F — derivada analiticamente, ela descreve:                  ║
║  "Se H₀ fosse verdadeira, com que frequência observaríamos               ║
║   cada valor de F em amostras de tamanho n=50.000 deste sistema?"        ║
║                                                                          ║
║  Isso é a melhor aproximação disponível da "distribuição populacional"   ║
║  da estatística sob H₀ para este sistema dinâmico específico.            ║
║                                                                          ║
║  Três testes de HN-F:                                                    ║
║  1. sin θ₂ | cos θ₂  →  F(1, 49994)  T_obs = 25.137   (p ≈ 0)            ║
║  2. cos θ₂ | sin θ₂  →  F(1, 49994)  T_obs =  3,96    (p = 0,046)        ║
║  3. conjunto          →  F(2, 49993)  T_obs = 12.573   (p ≈ 0)           ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from scipy import stats

# ── Parâmetros dos três testes ─────────────────────────────────────────────
TESTS = [
    dict(
        title     = r"Teste 1 — sin $\theta_2$ | cos $\theta_2$" + "\n" +
                    r"$H_0: \beta_{\sin\theta_2} = 0$ dado cos $\theta_2$ presente",
        df1=1, df2=49994,
        F_obs     = 25137.52,
        ncp       = 25137.52,
        dr2       = 0.26491153,
        alpha     = 0.05,
        rejects   = True,
        note      = "sin θ₂ explica 99,97%\nda contribuição total\nde θ₂ sobre α₁\nF_obs >> F_crit por\nefeito real (ΔR²=26,5%)",
        color_obs = "#F87171",
        percent   = 99.97,
    ),
    dict(
        title     = r"Teste 2 — cos $\theta_2$ | sin $\theta_2$" + "\n" +
                    r"$H_0: \beta_{\cos\theta_2} = 0$ dado sin $\theta_2$ presente",
        df1=1, df2=49994,
        F_obs     = 3.9645,
        ncp       = 3.9645,
        dr2       = 0.00004178,
        alpha     = 0.05,
        rejects   = True,
        note      = "cos θ₂ explica 0,02%\nΔR²=0,000042\nFormalmente rejeita\npor n=50.000\nEfeito desprezível\n(paradoxo amostral)",
        color_obs = "#FFA657",
        percent   = 0.02,
    ),
    dict(
        title     = r"Teste 3 — conjunto (sin + cos $\theta_2$)" + "\n" +
                    r"$H_0: \beta_{\sin\theta_2} = \beta_{\cos\theta_2} = 0$",
        df1=2, df2=49993,
        F_obs     = 12573.06,
        ncp       = 25146.13,
        dr2       = 0.26500227,
        alpha     = 0.05,
        rejects   = True,
        note      = "Teste conjunto de H2\nΔR²=26,5% total\nDominado pelo sin θ₂\ncos θ₂ contribui\nmarginalmente",
        color_obs = "#58A6FF",
        percent   = 100.0,
    ),
]

ALPHA = 0.05
N     = 50000

# ── Paleta ─────────────────────────────────────────────────────────────────
BG    = "#0D1117"
PANEL = "#161B22"
GRID  = "#21262D"
TEXT  = "#E6EDF3"
SLATE = "#64748B"
BLUE  = "#58A6FF"
RED   = "#F87171"
GREEN = "#3FB950"
AMBER = "#FFA657"

def sa(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=SLATE, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID); sp.set_linewidth(0.6)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.6)
    if title:  ax.set_title(title,   color=TEXT,  fontsize=8.5,
                             fontweight="bold", pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=SLATE, fontsize=8.5)
    if ylabel: ax.set_ylabel(ylabel, color=SLATE, fontsize=8.5)

# ── Figura ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22), facecolor=BG)
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.60, wspace=0.38)

for ti, t in enumerate(TESTS):
    df1, df2  = t["df1"], t["df2"]
    F_obs     = t["F_obs"]
    ncp       = t["ncp"]
    F_crit    = stats.f.ppf(1 - ALPHA, df1, df2)
    p_val     = 1 - stats.f.cdf(F_obs, df1, df2)

    # ── Painel A: PDF sob H₀, zoom na região de interesse ─────────────────
    ax = fig.add_subplot(gs[ti, 0])

    # Eixo X: se F_obs cabe, mostrar até F_obs×1.2; senão até F_crit×5
    if F_obs <= F_crit * 6:
        x_max = F_obs * 1.3
    else:
        x_max = F_crit * 5.0

    x  = np.linspace(1e-4, x_max, 2000)
    y0 = stats.f.pdf(x, df1, df2)

    sa(ax, t["title"],
       f"F({df1}, {df2})",
       "Probabilidade de ocorrência sob H₀")

    # Curva principal
    ax.plot(x, y0, color=BLUE, lw=2.5,
            path_effects=[pe.Stroke(linewidth=4, foreground=PANEL, alpha=0.5),
                           pe.Normal()],
            label=f"PDF sob H₀: F({df1}, {df2})")

    # Área de rejeição
    x_rej = x[x >= F_crit]
    y_rej = stats.f.pdf(x_rej, df1, df2)
    ax.fill_between(x_rej, y_rej, color=RED, alpha=0.30,
                    label=f"Rejeição α={ALPHA}")

    # F_crítico
    ax.axvline(F_crit, color=AMBER, lw=1.8, ls=(0,(5,3)), zorder=4,
               label=f"F_crit = {F_crit:.3f}")
    ax.text(F_crit + x_max*0.015,
            stats.f.pdf(F_crit, df1, df2)*0.5,
            f"F_crit\n{F_crit:.3f}",
            color=AMBER, fontsize=7.5, va="center",
            path_effects=[pe.Stroke(linewidth=2, foreground=PANEL, alpha=0.8),
                           pe.Normal()])

    # T_obs
    if F_obs <= x_max:
        ax.axvline(F_obs, color=t["color_obs"], lw=2.2, ls=":", zorder=5,
                   label=f"T_obs = {F_obs:.2f}")
        ax.text(F_obs + x_max*0.01,
                max(y0)*0.1,
                f"T_obs={F_obs:.2f}",
                color=t["color_obs"], fontsize=7.5,
                path_effects=[pe.Stroke(linewidth=2, foreground=PANEL, alpha=0.8),
                               pe.Normal()])
    else:
        ax.annotate("", xy=(x_max*0.98, max(y0)*0.06),
                    xytext=(x_max*0.78, max(y0)*0.06),
                    arrowprops=dict(arrowstyle="->",
                                    color=t["color_obs"], lw=2.5))
        F_str = f"{F_obs:,.0f}"
        ax.text(x_max*0.77, max(y0)*0.075,
                f"T_obs = {F_str} →→",
                color=t["color_obs"], fontsize=8.5, ha="right", fontweight="bold")
        ax.plot([], [], color=t["color_obs"], lw=2, ls=":",
                label=f"T_obs = {F_str}  ({'rejeita ✓' if t['rejects'] else 'não rejeita'})")

    # Badge com p-valor e ΔR²
    p_str  = "≈ 0" if p_val < 1e-10 else f"{p_val:.4f}"
    ax.text(0.97, 0.97,
            f"p = {p_str}\nα = {ALPHA}\nΔR² = {t['dr2']*100:.4f}%\n{t['note']}",
            transform=ax.transAxes, ha="right", va="top",
            color=t["color_obs"], fontsize=7.5, style="italic", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=PANEL,
                      edgecolor=t["color_obs"], linewidth=1.1, alpha=0.93))

    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, loc="upper right" if F_obs > x_max else "upper left")
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)

    # ── Painel B: H₀ vs H₁ em escala log ──────────────────────────────────
    ax = fig.add_subplot(gs[ti, 1])
    sa(ax,
       f"PDF sob H₀ vs H₁  —  Teste {ti+1}\nF central vs F não-central (λ={ncp:.1f})",
       "log₁₀(F)", "log₁₀(PDF)")

    x_log = np.logspace(-2, np.log10(max(F_obs*1.5, F_crit*10)), 3000)
    y_h0  = stats.f.pdf(x_log, df1, df2)
    y_h1  = stats.ncf.pdf(x_log, df1, df2, nc=ncp)

    m0 = y_h0 > 1e-300; m1 = y_h1 > 1e-300
    if m0.any():
        ax.plot(np.log10(x_log[m0]), np.log10(y_h0[m0]),
                color=BLUE, lw=2, label=r"PDF sob $H_0$ (F central)")
    if m1.any():
        ax.plot(np.log10(x_log[m1]), np.log10(y_h1[m1]),
                color=GREEN, lw=2, label=r"PDF sob $H_1$ (F não-central)")

    ax.axvline(np.log10(F_crit), color=AMBER, lw=1.5, ls="--",
               label=f"F_crit={F_crit:.3f}")
    ax.axvline(np.log10(F_obs),  color=t["color_obs"], lw=2, ls=":",
               label=f"T_obs={F_obs:.2f}")

    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT)
    ax.set_xlabel("log₁₀(F)", color=SLATE, fontsize=8.5)

    # Anotação: separação entre H₀ e H₁
    power = 1 - stats.ncf.cdf(F_crit, df1, df2, nc=ncp)
    ax.text(0.03, 0.12,
            f"H₀ centrada em ~1\n"
            f"H₁ centrada em ~{F_obs:.0f}\n"
            f"Poder = {power*100:.2f}%\n"
            f"ΔR² = {t['dr2']*100:.4f}%",
            transform=ax.transAxes, color=AMBER,
            fontsize=8, va="bottom", style="italic",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL,
                      edgecolor=AMBER, linewidth=1, alpha=0.9))

# ── Painel final: curva p-valor vs ΔR² para os três testes ────────────────
ax = fig.add_subplot(gs[3, :])
sa(ax,
   "Probabilidade de ocorrência sob H₀ em função de ΔR² — comparação dos três testes de HN-F\n"
   "Mostra por que sin θ₂ rejeita H₀ por efeito real e cos θ₂ apenas por n grande",
   "ΔR² (%)", "p-valor (escala log)")

dr2_range = np.linspace(0.000001, 0.35, 1000)

# Curvas por tamanho de amostra
for n_val, col_n, lbl in [
    (1000,   "#C084FC", "n = 1.000"),
    (10000,  "#64748B", "n = 10.000"),
    (50000,  BLUE,      "n = 50.000  ← nosso caso"),
    (100000, "#F87171", "n = 100.000"),
]:
    # F ≈ ΔR²·n / (q·(1-R²_completo)); usar q=1 e R²≈0.47
    F_approx = (dr2_range * n_val) / (1 * (1 - 0.473))
    p_approx  = 1 - stats.f.cdf(F_approx, 1, n_val - 6)
    ax.semilogy(dr2_range * 100, np.maximum(p_approx, 1e-300),
                color=col_n, lw=2, label=lbl)

ax.axhline(ALPHA, color=TEXT, lw=1.2, ls="--", alpha=0.7,
           label=f"α = {ALPHA}")

# Marcar os três testes
pontos = [
    (0.26491153, p_val:= 1-stats.f.cdf(25137.52,1,49994),
     "#3FB950",  "Teste 1: sin θ₂\nΔR²=26,5%\np≈0 por EFEITO REAL"),
    (0.00004178, p_val2:=1-stats.f.cdf(3.9645,1,49994),
     "#FFA657",  f"Teste 2: cos θ₂\nΔR²=0,004%\np={1-stats.f.cdf(3.9645,1,49994):.4f}\npor n grande"),
    (0.26500227, p_val3:=1-stats.f.cdf(12573.06,2,49993),
     BLUE,       "Teste 3: conjunto\nΔR²=26,5%\np≈0 por EFEITO REAL"),
]

for dr2, pv, col, lbl in pontos:
    ax.scatter([dr2*100], [max(pv, 1e-300)], s=150, color=col,
               zorder=6, marker="*")
    offset_x = 0.3 if dr2 < 0.001 else 1.0
    offset_y = 3 if pv > 1e-5 else 0.3
    ax.annotate(lbl,
                xy=(dr2*100, max(pv, 1e-300)),
                xytext=(dr2*100 + offset_x, max(pv, 1e-300) * offset_y),
                color=col, fontsize=8, style="italic",
                arrowprops=dict(arrowstyle="->", color=col, lw=1.2))

ax.legend(fontsize=8.5, facecolor=PANEL, edgecolor=GRID,
          labelcolor=TEXT, loc="upper right")
ax.set_xlim(0, 32)

# Anotação explicativa
ax.text(0.02, 0.08,
        "Teste 2 (cos θ₂): ΔR²=0,004% mas p<0,05 apenas porque n=50.000\n"
        "Teste 1 (sin θ₂): ΔR²=26,5% — p≈0 porque o efeito É grande, não por n",
        transform=ax.transAxes, color=AMBER, fontsize=9,
        va="bottom", style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
                  edgecolor=AMBER, linewidth=1.2, alpha=0.93))

# ── Nota metodológica ──────────────────────────────────────────────────────
fig.text(0.5, 0.002,
    "Nota metodológica: os dados provêm de um simulador físico (MuJoCo), não de uma população real.\n"
    "A distribuição de H₀ é a distribuição amostral analítica de F — descreve a probabilidade de ocorrência "
    "de cada valor de F em amostras de n=50.000 deste sistema dinâmico, assumindo H₀ verdadeira.",
    ha="center", va="bottom", color=SLATE, fontsize=8.5, style="italic")

fig.suptitle(
    "HN-F — Probabilidade de Ocorrência sob H₀ dado o Dataset do Pêndulo Invertido Duplo\n"
    r"sin $\theta_2$: efeito real (ΔR²=26,5%)  ·  "
    r"cos $\theta_2$: artefato amostral (ΔR²=0,004%)  ·  "
    "n = 50.000  ·  α = 0,05",
    color=TEXT, fontsize=11, fontweight="bold", y=0.998
)

OUT = "/mnt/user-data/outputs/HNF_distribuicoes_H0.png"
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"[OK] {OUT}")
plt.close(fig)
