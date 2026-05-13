"""
=============================================================================
Pêndulo Invertido Duplo — Análise Pré-Hipóteses
=============================================================================
Gera dois painéis gráficos que embasam as cinco hipóteses de teste:

  H1 — Dinâmica não-linear identificável via espaço de fase e resíduos
  H2 — Comportamento caótico via critério de Lyapunov e seção de Poincaré
  H3 — Subespaço de baixa dimensionalidade via PCA / autocorrelação
  H4 — Riqueza espectral suficiente para identificação do modelo
  H5 — Acoplamento causal τ → α, habilitando síntese de controlador

Uso:
    python pendulum_hypothesis_plots.py --csv <caminho_para_dataset.csv>

Saída:
    pendulum_hypothesis_analysis.png   (Figura 1 — visão geral)
    pendulum_hypothesis_math.png       (Figura 2 — embasamento matemático)
=============================================================================
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Paleta de cores (tema escuro GitHub-inspired)
# ---------------------------------------------------------------------------
C = {
    "bg":    "#0d1117",
    "panel": "#161b22",
    "grid":  "#21262d",
    "text":  "#e6edf3",
    "a1":    "#58a6ff",   # azul
    "a2":    "#f78166",   # coral
    "a3":    "#3fb950",   # verde
    "a4":    "#d2a8ff",   # lilás
    "a5":    "#ffa657",   # laranja
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def style_ax(ax, title: str = ""):
    """Aplica estilo escuro padronizado ao eixo."""
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["text"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(C["grid"])
    if title:
        ax.set_title(title, color=C["text"], fontsize=9, fontweight="bold", pad=6)
    ax.xaxis.label.set_color(C["text"])
    ax.yaxis.label.set_color(C["text"])
    ax.grid(True, color=C["grid"], linewidth=0.5, alpha=0.6)


def load_and_enrich(csv_path: str) -> pd.DataFrame:
    """Carrega o CSV e adiciona colunas derivadas."""
    df = pd.read_csv(csv_path)

    # Reconstrução dos ângulos a partir de sin/cos
    df["theta1"] = np.arctan2(df["sin_theta1"], df["cos_theta1"])
    df["theta2"] = np.arctan2(df["sin_theta2"], df["cos_theta2"])

    # Energia mecânica (proxy Lyapunov) — massas e comprimentos unitários
    g, L1, L2 = 9.81, 1.0, 1.0
    df["E_kin1"] = 0.5 * (L1 * df["omega1"]) ** 2
    df["E_kin2"] = 0.5 * (L2 * df["omega2"]) ** 2
    df["E_pot1"] = g * L1 * (1 - df["cos_theta1"])
    df["E_pot2"] = g * (L1 * (1 - df["cos_theta1"]) + L2 * (1 - df["cos_theta2"]))
    df["E_total"] = df["E_kin1"] + df["E_kin2"] + df["E_pot1"] + df["E_pot2"]

    return df


# ===========================================================================
# FIGURA 1 — Visão geral: espaço de fase, energia, ACF, FFT, Poincaré
# ===========================================================================

def plot_figure1(df: pd.DataFrame, out_path: str = "pendulum_hypothesis_analysis.png"):
    fig = plt.figure(figsize=(18, 14), facecolor=C["bg"])
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    episodes = df["episode"].unique()
    ep0 = df[df["episode"] == 0].copy()

    # ------------------------------------------------------------------
    # [0,0] H1 — Espaço de fase θ₁ × ω₁
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, "H1 — Espaço de Fase  θ₁ × ω₁")
    for ep in episodes:
        sub = df[df["episode"] == ep]
        ax.plot(sub["theta1"], sub["omega1"], lw=0.4, alpha=0.7, color=C["a1"])
    ax.set_xlabel("θ₁ (rad)")
    ax.set_ylabel("ω₁ (rad/s)")

    # ------------------------------------------------------------------
    # [0,1] H1 — Espaço de fase θ₂ × ω₂
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, "H1 — Espaço de Fase  θ₂ × ω₂")
    for ep in episodes:
        sub = df[df["episode"] == ep]
        ax.plot(sub["theta2"], sub["omega2"], lw=0.4, alpha=0.7, color=C["a2"])
    ax.set_xlabel("θ₂ (rad)")
    ax.set_ylabel("ω₂ (rad/s)")

    # ------------------------------------------------------------------
    # [0,2] H2 — Energia total (candidata V de Lyapunov)
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    style_ax(ax, "H2 — Energia Total  (proxy Lyapunov)")
    for ep in episodes:
        sub = df[df["episode"] == ep]
        ax.plot(sub["time"], sub["E_total"], lw=0.7, alpha=0.85, label=f"ep{ep}")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("E total (J)")
    ax.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["text"])

    # ------------------------------------------------------------------
    # [1,0] H3 — Autocorrelação ω₁
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax, "H3 — Autocorrelação  ω₁")
    signal = ep0["omega1"].values
    lags = np.arange(0, 300)
    acf = [
        np.corrcoef(signal[:-lag], signal[lag:])[0, 1] if lag > 0 else 1.0
        for lag in lags
    ]
    ax.plot(lags, acf, color=C["a3"], lw=1.2)
    ax.axhline(0, color=C["text"], lw=0.5, ls="--")
    ax.set_xlabel("Lag (amostras)")
    ax.set_ylabel("ACF")

    # ------------------------------------------------------------------
    # [1,1] H3 — Autocorrelação ω₂
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax, "H3 — Autocorrelação  ω₂")
    signal2 = ep0["omega2"].values
    acf2 = [
        np.corrcoef(signal2[:-lag], signal2[lag:])[0, 1] if lag > 0 else 1.0
        for lag in lags
    ]
    ax.plot(lags, acf2, color=C["a4"], lw=1.2)
    ax.axhline(0, color=C["text"], lw=0.5, ls="--")
    ax.set_xlabel("Lag (amostras)")
    ax.set_ylabel("ACF")

    # ------------------------------------------------------------------
    # [1,2] H4 — Espectro de frequência (FFT) ω₁, ω₂
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    style_ax(ax, "H4 — Espectro de Frequência  ω₁ e ω₂")
    dt = np.diff(ep0["time"].values).mean()
    N = len(ep0)
    freqs = fftfreq(N, dt)[: N // 2]
    fft1 = np.abs(fft(ep0["omega1"].values))[: N // 2]
    fft2 = np.abs(fft(ep0["omega2"].values))[: N // 2]
    ax.semilogy(freqs, fft1, color=C["a1"], lw=0.8, alpha=0.85, label="ω₁")
    ax.semilogy(freqs, fft2, color=C["a2"], lw=0.8, alpha=0.85, label="ω₂")
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("|FFT|  (log)")
    ax.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["text"])

    # ------------------------------------------------------------------
    # [2,0] H5 — Correlação cruzada θ₁ × θ₂
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    style_ax(ax, "H5 — Correlação Cruzada  θ₁ × θ₂")
    max_lag = 200
    cross = [
        pearsonr(ep0["theta1"].values[:-l], ep0["theta2"].values[l:])[0]
        if l > 0
        else pearsonr(ep0["theta1"].values, ep0["theta2"].values)[0]
        for l in range(max_lag)
    ]
    ax.plot(range(max_lag), cross, color=C["a3"], lw=1.2)
    ax.axhline(0, color=C["text"], lw=0.5, ls="--")
    ax.set_xlabel("Lag (amostras)")
    ax.set_ylabel("Pearson r")

    # ------------------------------------------------------------------
    # [2,1] H5 — Distribuição das acelerações angulares α₁ e α₂
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    style_ax(ax, "H5 — Distribuição  α₁  e  α₂")
    ax.hist(df["angle_accel1"], bins=80, color=C["a1"], alpha=0.6, density=True, label="α₁")
    ax.hist(df["angle_accel2"], bins=80, color=C["a2"], alpha=0.6, density=True, label="α₂")
    ax.set_xlabel("α (rad/s²)")
    ax.set_ylabel("Densidade")
    ax.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["text"])

    # ------------------------------------------------------------------
    # [2,2] H4 — Seção de Poincaré  (θ₂, ω₂)  quando ω₁ ≈ 0
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 2])
    style_ax(ax, "H4 — Seção de Poincaré  (ω₁ = 0)")
    for ep in episodes:
        sub = df[df["episode"] == ep].copy()
        signs = np.sign(sub["omega1"].values)
        crossings = np.where(np.diff(signs) != 0)[0]
        ax.scatter(
            sub["theta2"].values[crossings],
            sub["omega2"].values[crossings],
            s=3, alpha=0.6, label=f"ep{ep}",
        )
    ax.set_xlabel("θ₂ (rad)")
    ax.set_ylabel("ω₂ (rad/s)")
    ax.legend(fontsize=6, facecolor=C["panel"], labelcolor=C["text"], markerscale=3)

    fig.suptitle(
        "Análise Pré-Hipóteses — Pêndulo Invertido Duplo",
        color=C["text"], fontsize=14, fontweight="bold", y=0.98,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    print(f"[OK] Figura 1 salva em: {out_path}")
    plt.close(fig)


# ===========================================================================
# FIGURA 2 — Embasamento matemático: dV/dt, controlabilidade, linearidade
# ===========================================================================

def plot_figure2(df: pd.DataFrame, out_path: str = "pendulum_hypothesis_math.png"):
    fig = plt.figure(figsize=(18, 12), facecolor=C["bg"])
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    episodes = df["episode"].unique()
    ep0 = df[df["episode"] == 0].copy().reset_index(drop=True)

    # ------------------------------------------------------------------
    # [0,0] H2 — dV/dt por episódio (derivada da energia)
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, "H2 — dV/dt = Ė_total  (caos → não converge)")
    for ep in episodes:
        sub = df[df["episode"] == ep].copy()
        dE = np.gradient(sub["E_total"].values, sub["time"].values)
        ax.plot(sub["time"].values, dE, lw=0.5, alpha=0.7, label=f"ep{ep}")
    ax.axhline(0, color=C["text"], lw=0.8, ls="--")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("dV/dt  (W)")
    ax.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["text"])
    ax.text(0.05, 0.88, "Caos → dV/dt não converge a 0",
            transform=ax.transAxes, color=C["a2"], fontsize=7.5, style="italic")

    # ------------------------------------------------------------------
    # [0,1] H5/Controle — τ₁ × α₁  (correlação causal)
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, "H5 — τ₁ × α₁  (Controlabilidade)")
    ax.scatter(df["tau1_dynamics"], df["angle_accel1"], s=0.5, alpha=0.2, color=C["a1"])
    r1, _ = stats.pearsonr(df["tau1_dynamics"], df["angle_accel1"])
    ax.set_xlabel("τ₁  (torque dinâmica)")
    ax.set_ylabel("α₁  (rad/s²)")
    ax.text(0.05, 0.90, f"r = {r1:.3f}", transform=ax.transAxes,
            color=C["a3"], fontsize=10, fontweight="bold")

    # ------------------------------------------------------------------
    # [0,2] H5/Controle — τ₂ × α₂
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    style_ax(ax, "H5 — τ₂ × α₂  (Controlabilidade)")
    ax.scatter(df["tau2_dynamics"], df["angle_accel2"], s=0.5, alpha=0.2, color=C["a2"])
    r2, _ = stats.pearsonr(df["tau2_dynamics"], df["angle_accel2"])
    ax.set_xlabel("τ₂  (torque dinâmica)")
    ax.set_ylabel("α₂  (rad/s²)")
    ax.text(0.05, 0.90, f"r = {r2:.3f}", transform=ax.transAxes,
            color=C["a3"], fontsize=10, fontweight="bold")

    # ------------------------------------------------------------------
    # [1,0] H1 — Predição de ω₁(t+1) via integração de Euler (r²)
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax, "H1 — Predição Euler  ω₁(t+Δt) ≈ ω₁(t) + α₁·Δt")
    dt_mean = np.diff(ep0["time"].values).mean()
    o1_pred = ep0["omega1"].values[:-1] + ep0["angle_accel1"].values[:-1] * dt_mean
    o1_true = ep0["omega1"].values[1:]
    ax.scatter(o1_true, o1_pred, s=1, alpha=0.3, color=C["a4"])
    lo, hi = o1_true.min(), o1_true.max()
    ax.plot([lo, hi], [lo, hi], color=C["a5"], lw=1.5, ls="--", label="ideal (y=x)")
    r3, _ = stats.pearsonr(o1_true, o1_pred)
    ax.set_xlabel("ω₁ real")
    ax.set_ylabel("ω₁ predito (Euler 1ª ordem)")
    ax.text(0.05, 0.90, f"r² = {r3**2:.4f}", transform=ax.transAxes,
            color=C["a3"], fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["text"])

    # ------------------------------------------------------------------
    # [1,1] H3 — Autovalores da covariância do estado (dimensionalidade)
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax, "H3 — Autovalores da Covariância  (8 features)")
    feat_cols = [
        "sin_theta1", "cos_theta1", "sin_theta2", "cos_theta2",
        "omega1", "omega2", "angle_accel1", "angle_accel2",
    ]
    feat = df[feat_cols].values
    feat = (feat - feat.mean(0)) / feat.std(0)
    eigs = np.sort(np.linalg.eigvalsh(np.cov(feat.T)))[::-1]
    var_exp = np.cumsum(eigs) / eigs.sum() * 100
    ax.bar(range(1, 9), eigs, color=C["a1"], alpha=0.8, label="Autovalor")
    ax.set_xlabel("Componente")
    ax.set_ylabel("Autovalor")
    ax2 = ax.twinx()
    ax2.plot(range(1, 9), var_exp, "o-", color=C["a5"], lw=1.5, ms=5)
    ax2.set_ylabel("Variância acumulada (%)", color=C["a5"])
    ax2.tick_params(colors=C["a5"])
    ax.text(
        0.30, 0.88,
        f"{var_exp[3]:.1f}% em 4 componentes",
        transform=ax.transAxes, color=C["a5"], fontsize=8, fontweight="bold",
    )

    # ------------------------------------------------------------------
    # [1,2] H1 — Resíduos do ajuste linear  α₁ ~ θ₁  (não-linearidade)
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    style_ax(ax, "H1 — Resíduos: Ajuste Linear  α₁ ~ θ₁")
    x = df["theta1"].values
    y = df["angle_accel1"].values
    m, b = np.polyfit(x, y, 1)
    resid = y - (m * x + b)
    ax.scatter(x, resid, s=0.5, alpha=0.15, color=C["a2"])
    ax.axhline(0, color=C["text"], lw=0.8, ls="--")
    ax.set_xlabel("θ₁  (rad)")
    ax.set_ylabel("Resíduo  α₁  (rad/s²)")
    ax.text(0.05, 0.88, "Padrão estruturado → relação não-linear",
            transform=ax.transAxes, color=C["a3"], fontsize=8, style="italic")

    fig.suptitle(
        "Embasamento Matemático — Hipóteses de Identificação e Controle",
        color=C["text"], fontsize=13, fontweight="bold", y=0.98,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    print(f"[OK] Figura 2 salva em: {out_path}")
    plt.close(fig)


# ===========================================================================
# Entry-point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gera gráficos de análise pré-hipóteses para o pêndulo invertido duplo."
    )
    parser.add_argument(
        "--csv",
        default="D:\\Trabalhos\\Smart Agri\\Double_Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Data Processed\\pendulum_dataset_tidy_with_acceleration.csv",
        help="Caminho para o arquivo CSV do dataset.",
    )
    parser.add_argument("--out1", default="pendulum_hypothesis_analysis.png")
    parser.add_argument("--out2", default="pendulum_hypothesis_math.png")
    args = parser.parse_args()

    print(f"[...] Carregando dataset: {args.csv}")
    df = load_and_enrich(args.csv)
    print(f"[OK]  {len(df):,} amostras | {df['episode'].nunique()} episódios | {df.shape[1]} colunas")

    plot_figure1(df, out_path=args.out1)
    plot_figure2(df, out_path=args.out2)
    print("\nPronto! Execute com:")
    print(f"  python pendulum_hypothesis_plots.py --csv <dataset.csv>")


if __name__ == "__main__":
    main()
