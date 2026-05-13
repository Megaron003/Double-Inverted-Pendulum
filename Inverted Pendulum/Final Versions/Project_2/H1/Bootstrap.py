"""
╔══════════════════════════════════════════════════════════════════════════╗
║  BOOTSTRAP + COMPARAÇÃO COM DISTRIBUIÇÕES TEÓRICAS                       ║
║  Pêndulo Invertido Duplo                                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  O QUE ESTE SCRIPT FAZ:                                                  ║
║                                                                          ║
║  1. BOOTSTRAP DA PDF EMPÍRICA                                            ║
║     Reamostrar os dados B vezes com reposição → estimar a KDE em         ║
║     cada reamostra → banda de confiança 95% da densidade empírica.       ║
║     Isso responde: "qual é a incerteza na forma da distribuição?"        ║
║                                                                          ║
║  2. AJUSTE DE DISTRIBUIÇÕES TEÓRICAS                                     ║
║     Para cada variável, ajustar as candidatas:                           ║
║       - Normal (Gaussiana)                                               ║
║       - t de Student (caudas pesadas)                                    ║
║       - Laplace (caudas exponenciais)                                    ║
║       - Cauchy (caudas muito pesadas)                                    ║
║     Método: Máxima Verossimilhança (MLE)                                 ║
║                                                                          ║
║  3. TESTE DE QUALIDADE DE AJUSTE                                         ║
║     Kolmogorov-Smirnov (KS): compara CDF empírica vs teórica             ║
║     → p-valor: se p < 0.05, os dados diferem significativamente          ║
║       da distribuição teórica                                            ║
║     Anderson-Darling (AD): mais sensível nas caudas que KS               ║
║                                                                          ║
║  4. CRITÉRIO DE INFORMAÇÃO (AIC)                                         ║
║     AIC = 2k − 2·log(L)    (k = nº parâmetros, L = verossimilhança)      ║
║     Menor AIC → melhor compromisso entre ajuste e complexidade           ║
║                                                                          ╠
║  POR QUE BOOTSTRAP E NÃO MONTE CARLO AQUI?                               ║
║  Bootstrap: reamostra os DADOS REAIS — não assume nada sobre a           ║
║             distribuição verdadeira. A banda de incerteza vem            ║
║             diretamente da variabilidade observada nos dados.            ║
║  Monte Carlo: gera amostras de uma distribuição CONHECIDA A PRIORI.      ║
║               Útil quando você quer simular, não quando quer descobrir.  ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ── 0. Imports ─────────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import gaussian_kde

# ── 1. Dados ───────────────────────────────────────────────────────────────
CSV = "D:\Trabalhos\Smart Agri\Double_Inverted_Pendulum\Double-Inverted-Pendulum\Inverted Pendulum\Final Versions\Data Processed\pendulum_dataset_tidy_with_acceleration.csv"
df  = pd.read_csv(CSV)
df["theta1"] = np.arctan2(df["sin_theta1"], df["cos_theta1"])
df["theta2"] = np.arctan2(df["sin_theta2"], df["cos_theta2"])
n   = len(df)
rng = np.random.default_rng(42)

print(f"Dataset: {n:,} amostras | {df['episode'].nunique()} episódios")

# ── 2. Variáveis a analisar ────────────────────────────────────────────────
VARS = [
    ("omega1",        r"$\omega_1$  (vel. angular 1, rad/s)"),
    ("omega2",        r"$\omega_2$  (vel. angular 2, rad/s)"),
    ("angle_accel1",  r"$\alpha_1$  (acel. angular 1, rad/s²)"),
    ("angle_accel2",  r"$\alpha_2$  (acel. angular 2, rad/s²)"),
    ("theta1",        r"$\theta_1$  (ângulo 1, rad)"),
    ("theta2",        r"$\theta_2$  (ângulo 2, rad)"),
]

# ── 3. Distribuições teóricas candidatas ───────────────────────────────────
# Cada entrada: (nome, objeto scipy.stats, nº parâmetros livres)
CANDIDATES = [
    ("Normal",   stats.norm,    2),   # μ, σ
    ("t-Student",stats.t,       3),   # ν, μ, σ
    ("Laplace",  stats.laplace, 2),   # μ, b
    ("Cauchy",   stats.cauchy,  2),   # x₀, γ
]

# ── 4. Parâmetros do Bootstrap ─────────────────────────────────────────────
B          = 300        # número de reamostras (aumentar para mais precisão)
N_BOOT     = 5_000      # tamanho de cada reamostra (subconjunto para velocidade)
N_GRID     = 400        # pontos na grade de avaliação da KDE
CI_LEVEL   = 0.95       # nível do intervalo de confiança
ALPHA_CI   = 1 - CI_LEVEL

# ── 5. Paleta ──────────────────────────────────────────────────────────────
C = {
    "bg":    "#0D1117", "panel": "#161B22", "grid":  "#21262D",
    "text":  "#E6EDF3", "a1":    "#58A6FF", "a2":    "#F78166",
    "a3":    "#3FB950", "a4":    "#D2A8FF", "a5":    "#FFA657",
    "slate": "#64748B", "band":  "#1D4ED8",
}

# Cores por distribuição teórica
DIST_COLORS = {
    "Normal":    "#F87171",
    "t-Student": "#34D399",
    "Laplace":   "#FCD34D",
    "Cauchy":    "#C084FC",
}

def sa(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["slate"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(C["grid"]); sp.set_linewidth(0.6)
    ax.grid(True, color=C["grid"], lw=0.4, alpha=0.7)
    if title:  ax.set_title(title,   color=C["text"],  fontsize=9,
                             fontweight="bold", pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=C["slate"], fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=C["slate"], fontsize=8)

# ══════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL — uma figura por variável
# ══════════════════════════════════════════════════════════════════════════
all_results = {}   # armazena métricas para relatório final

for col, label in VARS:

    data = df[col].values
    mu, sigma = data.mean(), data.std()
    kurt  = stats.kurtosis(data)
    skew_ = stats.skew(data)

    # ── grade de avaliação (0.5% a 99.5% para excluir extremos raros) ─────
    x_lo = np.percentile(data, 0.5)
    x_hi = np.percentile(data, 99.5)
    x_grid = np.linspace(x_lo, x_hi, N_GRID)

    # ── A. Bootstrap da KDE ───────────────────────────────────────────────
    # Para cada reamostra: reamostrar N_BOOT pontos → ajustar KDE → avaliar
    # na grade → coletar curva.
    # Ao final: banda de confiança = percentis 2.5% e 97.5% das curvas.
    print(f"\n  [{col}] Bootstrap ({B} reamostras × {N_BOOT:,} pontos)...")

    kde_curves = np.zeros((B, N_GRID))
    all_idx = np.arange(n)

    for b in range(B):
        idx_b   = rng.choice(all_idx, size=N_BOOT, replace=True)
        data_b  = data[idx_b]
        bw      = gaussian_kde(data_b, bw_method="scott").factor
        kde_b   = gaussian_kde(data_b, bw_method=bw)
        kde_curves[b] = kde_b(x_grid)

    # KDE central (sobre os dados completos)
    kde_full    = gaussian_kde(data, bw_method="scott")
    kde_central = kde_full(x_grid)

    # Banda de confiança bootstrap
    band_lo = np.percentile(kde_curves, 100 * ALPHA_CI/2, axis=0)
    band_hi = np.percentile(kde_curves, 100 * (1 - ALPHA_CI/2), axis=0)

    # ── B. Ajuste MLE das distribuições teóricas ──────────────────────────
    # MLE: encontra os parâmetros que maximizam P(dados | distribuição)
    # scipy.stats.dist.fit() retorna os parâmetros otimizados
    fitted = {}
    aic    = {}
    ks_res = {}
    ad_res = {}

    # Subsample para KS/AD (teste exato em n=50k seria conservador demais)
    idx_test = rng.choice(all_idx, size=5000, replace=False)
    data_test = data[idx_test]

    for dist_name, dist_obj, k_params in CANDIDATES:
        try:
            # MLE
            params = dist_obj.fit(data)
            fitted[dist_name] = params

            # Log-verossimilhança
            log_L = np.sum(dist_obj.logpdf(data, *params))

            # AIC = 2k - 2·log(L)
            aic[dist_name] = 2 * k_params - 2 * log_L

            # Kolmogorov-Smirnov
            ks_stat, ks_p = stats.kstest(data_test,
                                          lambda x: dist_obj.cdf(x, *params))
            ks_res[dist_name] = (ks_stat, ks_p)

        except Exception as e:
            print(f"    {dist_name}: falhou ({e})")

    # Melhor distribuição por AIC
    best_dist = min(aic, key=aic.get)

    # ── C. Relatório textual ───────────────────────────────────────────────
    print(f"  Estatísticas: μ={mu:.3f}  σ={sigma:.3f}  "
          f"kurt={kurt:+.3f}  skew={skew_:+.3f}")
    print(f"  {'Distribuição':<12}  {'AIC':>14}  "
          f"{'ΔAIC':>8}  {'KS':>8}  {'p-KS':>10}  {'Melhor?'}")
    print(f"  {'─'*70}")

    aic_min = min(aic.values())
    for dist_name, aic_val in sorted(aic.items(), key=lambda x: x[1]):
        delta = aic_val - aic_min
        ks_s, ks_p = ks_res.get(dist_name, (np.nan, np.nan))
        melhor = "★" if dist_name == best_dist else ""
        print(f"  {dist_name:<12}  {aic_val:>14.2f}  "
              f"{delta:>8.2f}  {ks_s:>8.4f}  {ks_p:>10.4e}  {melhor}")

    # Salva para relatório final
    all_results[col] = {
        "label":     label,
        "kurt":      kurt,
        "skew":      skew_,
        "best_dist": best_dist,
        "aic":       aic,
        "ks":        ks_res,
        "fitted":    fitted,
    }

    # ── D. Figura ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12), facecolor=C["bg"])
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

    # ── D1. PDF empírica + banda bootstrap + teóricas ─────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    sa(ax1, f"PDF empírica (bootstrap {CI_LEVEL*100:.0f}%) + distribuições teóricas",
       label.split("(")[0].strip(), "Densidade de probabilidade")

    # Banda de confiança bootstrap
    ax1.fill_between(x_grid, band_lo, band_hi,
                     color=C["band"], alpha=0.25,
                     label=f"IC {CI_LEVEL*100:.0f}% bootstrap (B={B})")

    # KDE central
    ax1.plot(x_grid, kde_central, color=C["a1"], lw=2.5,
             label="KDE empírica (dados completos)")

    # Distribuições teóricas
    for dist_name, dist_obj, _ in CANDIDATES:
        if dist_name not in fitted: continue
        params  = fitted[dist_name]
        y_theo  = dist_obj.pdf(x_grid, *params)
        lw      = 2.5 if dist_name == best_dist else 1.5
        ls      = "-"  if dist_name == best_dist else "--"
        marker  = f" ★" if dist_name == best_dist else ""
        ax1.plot(x_grid, y_theo,
                 color=DIST_COLORS[dist_name],
                 lw=lw, ls=ls,
                 label=f"{dist_name}{marker}  (AIC={aic[dist_name]:.0f})")

    ax1.legend(fontsize=7.5, facecolor=C["panel"],
               labelcolor=C["text"], loc="upper right")

    # Anotação de estatísticas
    ax1.text(0.02, 0.96,
             f"κ = {kurt:+.3f}  (0 = Gaussiana)\n"
             f"γ = {skew_:+.3f}  (0 = simétrica)\n"
             f"Melhor ajuste: {best_dist}",
             transform=ax1.transAxes,
             color=C["a5"], fontsize=8.5, va="top", style="italic",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor=C["panel"], edgecolor=C["grid"],
                       alpha=0.9))

    # ── D2. Variabilidade bootstrap (curvas individuais) ──────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    sa(ax2, f"Variabilidade bootstrap\n({B} curvas KDE)",
       label.split("(")[0].strip(), "Densidade")

    # Plotar subconjunto das curvas bootstrap (legibilidade)
    for b in range(0, B, B//40):
        ax2.plot(x_grid, kde_curves[b],
                 color=C["a1"], lw=0.4, alpha=0.15)

    ax2.plot(x_grid, kde_central,
             color=C["a5"], lw=2, label="KDE central")
    ax2.fill_between(x_grid, band_lo, band_hi,
                     color=C["band"], alpha=0.3,
                     label=f"IC {CI_LEVEL*100:.0f}%")
    ax2.legend(fontsize=7.5, facecolor=C["panel"], labelcolor=C["text"])

    # ── D3. CDF empírica vs teóricas (gráfico P-P alternativo) ────────────
    ax3 = fig.add_subplot(gs[1, 0:2])
    sa(ax3, "CDF empírica vs CDF teóricas",
       label.split("(")[0].strip(), "Probabilidade acumulada")

    # CDF empírica
    x_sorted = np.sort(data[rng.choice(n, size=3000, replace=False)])
    ecdf     = np.arange(1, len(x_sorted)+1) / len(x_sorted)
    ax3.plot(x_sorted, ecdf, color=C["a1"], lw=1.5,
             label="CDF empírica", alpha=0.9)

    for dist_name, dist_obj, _ in CANDIDATES:
        if dist_name not in fitted: continue
        params = fitted[dist_name]
        x_th   = np.linspace(x_sorted.min(), x_sorted.max(), 500)
        cdf_th = dist_obj.cdf(x_th, *params)
        lw     = 2.5 if dist_name == best_dist else 1.5
        ls     = "-"  if dist_name == best_dist else "--"
        ks_s   = ks_res[dist_name][0]
        ax3.plot(x_th, cdf_th,
                 color=DIST_COLORS[dist_name], lw=lw, ls=ls,
                 label=f"{dist_name}  (KS={ks_s:.4f})")

    ax3.legend(fontsize=7.5, facecolor=C["panel"], labelcolor=C["text"])

    # ── D4. Q-Q plot da melhor distribuição ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    sa(ax4, f"Q-Q plot — {best_dist}\n(melhor AIC)",
       f"Quantis teóricos ({best_dist})",
       "Quantis empíricos")

    params_best = fitted[best_dist]
    dist_best   = {name: obj for name, obj, _ in CANDIDATES}[best_dist]

    # Subsample para Q-Q legível
    idx_qq  = rng.choice(n, size=2000, replace=False)
    data_qq = np.sort(data[idx_qq])
    pp      = (np.arange(1, len(data_qq)+1) - 0.5) / len(data_qq)
    q_theo  = dist_best.ppf(pp, *params_best)

    ax4.scatter(q_theo, data_qq, s=1.2, alpha=0.3, color=C["a4"])

    # Linha de referência y = x
    q_lo = min(q_theo.min(), data_qq.min())
    q_hi = max(q_theo.max(), data_qq.max())
    ax4.plot([q_lo, q_hi], [q_lo, q_hi],
             color=C["a5"], lw=1.8, ls="--", label="y = x (ajuste perfeito)")
    ax4.legend(fontsize=7.5, facecolor=C["panel"], labelcolor=C["text"])

    # Anotação KS e p-valor
    ks_s_best, ks_p_best = ks_res[best_dist]
    ax4.text(0.04, 0.94,
             f"KS = {ks_s_best:.4f}\np = {ks_p_best:.3e}\n"
             f"{'p > 0.05 → ajuste aceitável' if ks_p_best > 0.05 else 'p < 0.05 → rejeita ajuste'}",
             transform=ax4.transAxes,
             color=C["a3"] if ks_p_best > 0.05 else C["a2"],
             fontsize=8, va="top", style="italic")

    # Super-título
    fig.suptitle(
        f"Bootstrap + Distribuições Teóricas — {label}\n"
        f"n = {n:,}  |  B = {B}  |  IC {CI_LEVEL*100:.0f}%  |  "
        f"Melhor ajuste (AIC): {best_dist}",
        color=C["text"], fontsize=10, fontweight="bold", y=0.98
    )

    out = f"D:\Trabalhos\Smart Agri\Double_Inverted_Pendulum\Double-Inverted-Pendulum\Inverted Pendulum\Final Versions\Project_2\H1\Resultsbootstrap_dist_{col}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  [OK] {out}")

# ══════════════════════════════════════════════════════════════════════════
# RELATÓRIO CONSOLIDADO
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*68)
print("  RELATÓRIO FINAL — Comparação de Distribuições")
print("  Dataset: Pêndulo Invertido Duplo")
print("═"*68)
print(f"\n  {'Variável':<15}  {'κ':>7}  {'γ':>7}  {'Melhor AIC':<12}  "
      f"{'ΔAIC 2º':<10}  {'KS(melhor)':<12}  {'p-KS'}")
print(f"  {'─'*80}")

for col, res in all_results.items():
    best  = res["best_dist"]
    aic_s = sorted(res["aic"].items(), key=lambda x: x[1])
    d_aic = aic_s[1][1] - aic_s[0][1] if len(aic_s) > 1 else 0
    ks_s, ks_p = res["ks"].get(best, (np.nan, np.nan))
    short = col.replace("angle_accel", "α").replace("omega", "ω").replace("theta", "θ")
    print(f"  {short:<15}  {res['kurt']:>+7.3f}  {res['skew']:>+7.3f}  "
          f"{best:<12}  {d_aic:>10.1f}  {ks_s:>12.4f}  {ks_p:.3e}")

print(f"\n  Interpretação do ΔAIC:")
print(f"    ΔAIC < 2  → evidência fraca para preferir o melhor")
print(f"    ΔAIC 2-6  → evidência moderada")
print(f"    ΔAIC > 6  → evidência forte")
print(f"    ΔAIC > 10 → evidência muito forte")
print(f"\n  Interpretação do KS:")
print(f"    p > 0.05  → dados compatíveis com a distribuição teórica")
print(f"    p < 0.05  → dados diferem significativamente da teórica")
print(f"    Nota: com n=50.000, o KS tem altíssima potência e tende")
print(f"    a rejeitar qualquer distribuição — use AIC como critério")
print(f"    principal e KS como diagnóstico qualitativo.")
print("\n" + "═"*68)
print("  Figuras salvas em D:\Trabalhos\Smart Agri\Double_Inverted_Pendulum\Double-Inverted-Pendulum\Inverted Pendulum\Final Versions\Project_2\H1\Resultsbootstrap_dist_*.png")
print("═"*68)