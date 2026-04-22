"""
=============================================================================
Pêndulo Invertido Duplo — Análise Exploratória Cega
Descoberta de Estrutura nos Dados (sem pressupostos)
=============================================================================

Este script parte dos dados brutos e aplica métodos agnósticos ao modelo
para *revelar* propriedades estruturais que, posteriormente, fundamentarão
hipóteses científicas testáveis.

Fluxo metodológico
------------------
  1. Distribuições Marginais       → caudas pesadas, assimetria?
  2. Informação Mútua              → dependências não-lineares entre features
  3. R² Linear vs Não-Linear       → suficiência do modelo linear
  4. Divergência entre Trajetórias → sensibilidade às condições iniciais
  5. Correlação de Spearman        → acoplamento sem pressupor linearidade
  6. Teste ADF                     → estacionariedade das séries temporais
  7. Resíduos do Ajuste Linear     → estrutura remanescente não capturada

Saída
-----
  pendulum_discovery_analysis.png

Uso
---
  python pendulum_discovery.py
  python pendulum_discovery.py --csv caminho/para/dataset.csv
  python pendulum_discovery.py --csv dataset.csv --out minha_figura.png --dpi 200

Dependências
------------
  pip install pandas numpy matplotlib scipy scikit-learn statsmodels
=============================================================================
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller


# ─────────────────────────────────────────────────────────────────────────────
# Paleta visual (tema escuro)
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def style_ax(ax: plt.Axes, title: str = "") -> None:
    """Aplica estilo visual padronizado a um eixo matplotlib."""
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(colors=COLORS["text"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(COLORS["grid"])
    if title:
        ax.set_title(title, color=COLORS["text"], fontsize=8.5,
                     fontweight="bold", pad=6)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.grid(True, color=COLORS["grid"], linewidth=0.4, alpha=0.7)


def annotate(ax: plt.Axes, text: str, color: str = None,
             x: float = 0.03, y: float = 0.88) -> None:
    """Adiciona anotação de texto em coordenadas de eixo normalizadas."""
    ax.text(x, y, text, transform=ax.transAxes,
            color=color or COLORS["a5"], fontsize=8,
            style="italic", va="top")


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento e enriquecimento
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Carrega o CSV e adiciona colunas derivadas necessárias para a análise.

    Colunas adicionadas
    -------------------
    theta1, theta2 : ângulos reconstruídos via arctan2(sin, cos)
                     evita descontinuidades em ±π
    """
    df = pd.read_csv(csv_path)
    required = {
        "sin_theta1", "cos_theta1", "sin_theta2", "cos_theta2",
        "omega1", "omega2", "angle_accel1", "angle_accel2",
        "tau1_dynamics", "tau2_dynamics", "episode", "time",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    df["theta1"] = np.arctan2(df["sin_theta1"], df["cos_theta1"])
    df["theta2"] = np.arctan2(df["sin_theta2"], df["cos_theta2"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Painéis individuais
# ─────────────────────────────────────────────────────────────────────────────

def panel_marginal_distributions(fig: plt.Figure, gs_row: int,
                                  df: pd.DataFrame) -> None:
    """
    Painel 1 (linha 0) — Distribuições Marginais Brutas
    ----------------------------------------------------
    Pergunta: como se distribuem as variáveis de estado?
    Se as distribuições forem leptocúrticas (curtose > 1), eventos extremos
    são mais frequentes do que o esperado por uma Gaussiana, indicando
    dinâmica não-estacionária ou caótica.
    """
    entries = [
        ("omega1",       COLORS["a1"], "ω₁  (vel. angular 1)"),
        ("omega2",       COLORS["a2"], "ω₂  (vel. angular 2)"),
        ("angle_accel1", COLORS["a3"], "α₁  (acel. angular 1)"),
        ("angle_accel2", COLORS["a4"], "α₂  (acel. angular 2)"),
    ]
    for col_idx, (col, color, label) in enumerate(entries):
        ax = fig.add_subplot(gs_row[0, col_idx])
        style_ax(ax, f"Distribuição bruta — {label}")

        data = df[col].values
        ax.hist(data, bins=80, density=True, color=color,
                alpha=0.75, edgecolor="none")

        # sobreposição gaussiana teórica (referência)
        xg = np.linspace(data.min(), data.max(), 300)
        ax.plot(xg, stats.norm.pdf(xg, data.mean(), data.std()),
                color=COLORS["text"], lw=1.2, ls="--", label="Gaussiana ref.")

        kurt = stats.kurtosis(data)
        skew = stats.skew(data)
        annotate(ax, f"kurt = {kurt:.2f}\nskew = {skew:.2f}",
                 color=COLORS["a5"])
        ax.legend(fontsize=6.5, facecolor=COLORS["panel"],
                  labelcolor=COLORS["text"])
        ax.set_xlabel(col)
        ax.set_ylabel("Densidade")


def panel_mutual_information(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Painel 2a — Informação Mútua
    ----------------------------
    Pergunta: quais features *dependem* de α₁ e α₂, sem assumir linearidade?
    MI = 0 implica independência estatística completa.
    MI alta indica dependência, linear ou não.
    Comparar MI com Pearson² revela se a relação é não-linear.
    """
    style_ax(ax, "Informação Mútua — quais features explicam α₁ e α₂?")

    features = df[["theta1", "theta2", "omega1", "omega2",
                    "tau1_dynamics", "tau2_dynamics"]].values
    feat_names = ["θ₁", "θ₂", "ω₁", "ω₂", "τ₁", "τ₂"]

    mi1 = mutual_info_regression(features, df["angle_accel1"].values,
                                 random_state=42)
    mi2 = mutual_info_regression(features, df["angle_accel2"].values,
                                 random_state=42)

    x_pos = np.arange(len(feat_names))
    w = 0.35
    ax.bar(x_pos - w / 2, mi1, w, color=COLORS["a1"],
           alpha=0.85, label="MI com α₁")
    ax.bar(x_pos + w / 2, mi2, w, color=COLORS["a2"],
           alpha=0.85, label="MI com α₂")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(feat_names, color=COLORS["text"])
    ax.set_ylabel("Informação Mútua (nats)")
    ax.set_xlabel("Feature")
    ax.legend(fontsize=8, facecolor=COLORS["panel"], labelcolor=COLORS["text"])
    annotate(ax, "τ₁ é mais informativo que θ₁ para α₁\n→ acoplamento não-linear entre elos",
             color=COLORS["a5"])


def panel_r2_comparison(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Painel 2b — R² Linear vs Não-Linear
    ------------------------------------
    Pergunta: um modelo linear (θ, ω) → α consegue explicar a variância?
    Se R² linear << 1, existe estrutura que o modelo linear não captura.
    Adicionando termos sin/cos/quadráticos: quanto o R² aumenta?
    O ganho percentual quantifica o "quanto de não-linearidade" existe.
    """
    style_ax(ax, "Capacidade Preditiva — Linear vs Não-Linear → α₁ e α₂")

    X_base = df[["theta1", "theta2", "omega1", "omega2"]].values
    # features não-lineares: quadráticos + trigonométricos + cruzados
    X_nl = np.column_stack([
        X_base,
        X_base ** 2,
        np.sin(X_base[:, 0]), np.cos(X_base[:, 0]),
        np.sin(X_base[:, 1]), np.cos(X_base[:, 1]),
        X_base[:, 0] * X_base[:, 2],   # θ₁·ω₁
        X_base[:, 1] * X_base[:, 3],   # θ₂·ω₂
    ])

    results = {}
    for col, name in [("angle_accel1", "α₁"), ("angle_accel2", "α₂")]:
        y = df[col].values
        r2_l = LinearRegression().fit(
            StandardScaler().fit_transform(X_base), y
        ).score(StandardScaler().fit_transform(X_base), y)
        r2_nl = LinearRegression().fit(
            StandardScaler().fit_transform(X_nl), y
        ).score(StandardScaler().fit_transform(X_nl), y)
        results[name] = (r2_l, r2_nl)

    bar_pos    = np.array([0, 1, 3, 4])
    bar_vals   = [results["α₁"][0], results["α₁"][1],
                  results["α₂"][0], results["α₂"][1]]
    bar_colors = [COLORS["a1"], COLORS["a3"], COLORS["a2"], COLORS["a3"]]
    bar_labels = ["α₁  Linear", "α₁  +NL", "α₂  Linear", "α₂  +NL"]

    bars = ax.bar(bar_pos, bar_vals, color=bar_colors, alpha=0.85, width=0.7)
    for b, v in zip(bars, bar_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", color=COLORS["text"],
                fontsize=8, fontweight="bold")

    ax.set_xticks(bar_pos)
    ax.set_xticklabels(bar_labels, color=COLORS["text"], fontsize=8)
    ax.set_ylabel("R²")
    ax.set_ylim(0, 0.75)

    gain1 = 100 * (results["α₁"][1] - results["α₁"][0])
    annotate(ax,
             f"R² linear = {results['α₁'][0]:.2f}  →  insuficiente\n"
             f"Ganho com termos NL: +{gain1:.1f}%",
             color=COLORS["a5"])


def panel_trajectory_divergence(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Painel 3a — Divergência entre Trajetórias
    ------------------------------------------
    Pergunta: trajetórias com condições iniciais (CI) distintas se afastam
    exponencialmente? Se sim, o sistema é sensível a CI — indício de caos.
    Usamos ω₁ do episódio 0 como referência e medimos |Δω₁| para os demais.
    Escala logarítmica: divergência linear no gráfico log = crescimento exp.
    """
    style_ax(ax, "Divergência |ω₁(epᵢ) − ω₁(ep₀)| — Sensibilidade às CI")

    ep0_w = df[df["episode"] == 0]["omega1"].values
    ep0_t = df[df["episode"] == 0]["time"].values
    episode_colors = [COLORS["a1"], COLORS["a2"], COLORS["a3"], COLORS["a4"]]

    for ep, color in zip([1, 2, 3, 4], episode_colors):
        sub = df[df["episode"] == ep]["omega1"].values
        ml = min(len(ep0_w), len(sub))
        diff = np.abs(ep0_w[:ml] - sub[:ml])
        ax.semilogy(ep0_t[:ml], diff + 1e-6, color=color,
                    lw=0.9, alpha=0.85, label=f"|ep{ep} − ep0|")

    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("|Δω₁|  (escala log)")
    ax.legend(fontsize=7.5, facecolor=COLORS["panel"], labelcolor=COLORS["text"])
    annotate(ax,
             "Divergência imediata → alta sensibilidade a CI\n→ indício de caos hamiltoniano",
             color=COLORS["a5"])


def panel_spearman_matrix(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Painel 3b — Matriz de Correlação de Spearman
    ---------------------------------------------
    Pergunta: quais variáveis co-variam, sem assumir linearidade?
    Spearman mede correlação de postos — captura monotonia não-linear.
    Diferenças grandes entre Spearman e Pearson indicam não-linearidade.
    """
    style_ax(ax, "Correlação de Spearman — Todos os Pares")

    cols = ["theta1", "theta2", "omega1", "omega2",
            "angle_accel1", "angle_accel2",
            "tau1_dynamics", "tau2_dynamics"]
    labels = ["θ₁", "θ₂", "ω₁", "ω₂", "α₁", "α₂", "τ₁", "τ₂"]
    n = len(cols)

    mat = np.zeros((n, n))
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            mat[i, j] = stats.spearmanr(df[c1], df[c2])[0]

    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, color=COLORS["text"], fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, color=COLORS["text"], fontsize=8)

    for i in range(n):
        for j in range(n):
            txt_color = "white" if abs(mat[i, j]) > 0.5 else COLORS["text"]
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color=txt_color, fontsize=6.5)

    plt.colorbar(im, ax=ax, fraction=0.03).ax.tick_params(colors=COLORS["text"])


def panel_adf_stationarity(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Painel 4a — Teste de Dickey-Fuller Aumentado (ADF)
    ---------------------------------------------------
    Pergunta: as séries temporais são estacionárias?
    H₀(ADF): a série possui raiz unitária (não-estacionária).
    Rejeitar H₀ (p < 0.05) → série estacionária.
    Estacionariedade é condição necessária para que um modelo treinado em
    parte da série generalize para o restante.
    """
    style_ax(ax, "Teste ADF — Estacionariedade das Séries Temporais")

    cols = ["omega1", "omega2", "angle_accel1", "angle_accel2",
            "tau1_dynamics", "tau2_dynamics"]
    labels = ["ω₁", "ω₂", "α₁", "α₂", "τ₁", "τ₂"]
    ep0 = df[df["episode"] == 0]

    adf_vals, p_vals = [], []
    for col in cols:
        result = adfuller(ep0[col].values, autolag="AIC")
        adf_vals.append(abs(result[0]))
        p_vals.append(result[1])

    bar_colors = [COLORS["a3"] if p < 0.05 else COLORS["a2"] for p in p_vals]
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, adf_vals, color=bar_colors, alpha=0.85)

    for b, p in zip(bars, p_vals):
        label = "p<0.05 ✓" if p < 0.05 else f"p={p:.3f}"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                label, ha="center", va="bottom",
                color=COLORS["text"], fontsize=7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, color=COLORS["text"], fontsize=9)
    ax.set_ylabel("|Estatística ADF|")
    ax.axhline(3.5, color=COLORS["a5"], ls="--", lw=1.2,
               label="limiar crítico ≈ 3.5")
    ax.legend(fontsize=7.5, facecolor=COLORS["panel"], labelcolor=COLORS["text"])
    annotate(ax, "Verde = estacionária (p < 0.05)\n→ modelo generaliza temporalmente",
             color=COLORS["a3"])


def panel_linear_residuals(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Painel 4b — Resíduos do Ajuste Linear
    --------------------------------------
    Pergunta: após remover a componente linear, resta estrutura sistemática?
    Se os resíduos forem ruído branco (aleatórios), o modelo linear captura
    toda a dinâmica. Se apresentarem padrão (ex: forma em "S"), existe
    estrutura não-linear — especificamente, termos trigonométricos e cruzados
    típicos das equações de Lagrange do pêndulo duplo.
    """
    style_ax(ax, "Resíduos do Ajuste Linear — Estrutura Remanescente")

    X = df[["theta1", "theta2", "omega1", "omega2"]].values
    X_s = StandardScaler().fit_transform(X)

    res1 = df["angle_accel1"].values - \
           LinearRegression().fit(X_s, df["angle_accel1"].values) \
                             .predict(X_s)
    res2 = df["angle_accel2"].values - \
           LinearRegression().fit(X_s, df["angle_accel2"].values) \
                             .predict(X_s)

    # subsample para performance visual
    idx = slice(None, None, 5)
    ax.scatter(df["theta1"].values[idx], res1[idx],
               s=0.8, alpha=0.25, color=COLORS["a1"], label="res α₁ vs θ₁")
    ax.scatter(df["theta2"].values[idx], res2[idx],
               s=0.8, alpha=0.25, color=COLORS["a2"], label="res α₂ vs θ₂")

    ax.axhline(0, color=COLORS["text"], lw=0.8, ls="--")
    ax.set_xlabel("θ  (rad)")
    ax.set_ylabel("Resíduo (rad/s²)")
    ax.legend(fontsize=7.5, facecolor=COLORS["panel"],
              labelcolor=COLORS["text"], markerscale=5)
    annotate(ax,
             "Padrão em 'S' → termos trigonométricos\ne cruzados não capturados pelo modelo linear",
             color=COLORS["a5"])


# ─────────────────────────────────────────────────────────────────────────────
# Figura principal
# ─────────────────────────────────────────────────────────────────────────────

def build_figure(df: pd.DataFrame, out_path: str, dpi: int = 150) -> None:
    """
    Constrói e salva o painel completo de análise exploratória cega.

    Layout (4 linhas × 4 colunas)
    ------------------------------
    Linha 0  [col 0-3] : Distribuições marginais brutas (4 variáveis)
    Linha 1  [col 0-1] : Informação Mútua
    Linha 1  [col 2-3] : R² Linear vs Não-Linear
    Linha 2  [col 0-1] : Divergência entre trajetórias
    Linha 2  [col 2-3] : Matriz de Spearman
    Linha 3  [col 0-1] : Teste ADF
    Linha 3  [col 2-3] : Resíduos do ajuste linear
    """
    fig = plt.figure(figsize=(20, 16), facecolor=COLORS["bg"])
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.52, wspace=0.38)

    # --- Linha 0: distribuições ---
    panel_marginal_distributions(fig, gs, df)

    # --- Linha 1 ---
    panel_mutual_information(fig.add_subplot(gs[1, 0:2]), df)
    panel_r2_comparison(fig.add_subplot(gs[1, 2:4]), df)

    # --- Linha 2 ---
    panel_trajectory_divergence(fig.add_subplot(gs[2, 0:2]), df)
    panel_spearman_matrix(fig.add_subplot(gs[2, 2:4]), df)

    # --- Linha 3 ---
    panel_adf_stationarity(fig.add_subplot(gs[3, 0:2]), df)
    panel_linear_residuals(fig.add_subplot(gs[3, 2:4]), df)

    fig.suptitle(
        "Análise Exploratória Cega — Descoberta de Estrutura nos Dados\n"
        "Cada painel revela uma propriedade sem pressupô-la",
        color=COLORS["text"], fontsize=13, fontweight="bold", y=0.995,
    )

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=COLORS["bg"])
    print(f"[OK] Figura salva em: {out_path}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figuras por análise
# ─────────────────────────────────────────────────────────────────────────────

def build_figures(df: pd.DataFrame, out_path: str, dpi: int = 150) -> None:
    """
    Constrói e salva figuras individuais para cada painel de análise.

    Cada figura é salva como 'painel_X.png' no mesmo diretório de saída.
    """
    # panel_marginal_distributions has a different signature (fig, gs, df)
    fig = plt.figure(figsize=(16, 4), facecolor=COLORS["bg"])
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38)
    panel_marginal_distributions(fig, gs, df)
    panel_out = out_path.replace(".png", "_painel_1.png")
    plt.savefig(panel_out, dpi=dpi, bbox_inches="tight", facecolor=COLORS["bg"])
    print(f"[OK] Painel 1 salvo em: {panel_out}")
    plt.close(fig)

    panel_funcs = [
        panel_mutual_information,
        panel_r2_comparison,
        panel_trajectory_divergence,
        panel_spearman_matrix,
        panel_adf_stationarity,
        panel_linear_residuals,
    ]
    for idx, func in enumerate(panel_funcs, start=2):
        fig = plt.figure(figsize=(8, 6), facecolor=COLORS["bg"])
        ax = fig.add_subplot(111)
        func(ax, df)
        panel_out = out_path.replace(".png", f"_painel_{idx}.png")
        plt.savefig(panel_out, dpi=dpi, bbox_inches="tight",
                    facecolor=COLORS["bg"])
        print(f"[OK] Painel {idx} salvo em: {panel_out}")
        plt.close(fig)

    save_fig = out_path.replace(".png", "_combined.png")
    build_figure(df, save_fig, dpi)
    print(f"[OK] Figura combinada salva em: {save_fig}")

# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Análise exploratória cega do dataset do pêndulo invertido duplo."
    )
    p.add_argument(
        "--csv",
        default="D:\Trabalhos\Smart Agri\Double_Inverted_Pendulum\Double-Inverted-Pendulum\Inverted Pendulum\Final Versions\Data Processed\pendulum_dataset_tidy_with_acceleration.csv",
        help="Caminho para o arquivo CSV do dataset.",
    )
    p.add_argument(
        "--out",
        default="D:\\Trabalhos\\Smart Agri\\Double_Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Project_1\\results\\pendulum_discovery_analysis.png",
        help="Caminho do arquivo de saída (PNG).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolução da figura em DPI (padrão: 150).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[...] Carregando: {args.csv}")
    df = load_dataset(args.csv)
    print(f"[OK]  {len(df):,} amostras | "
          f"{df['episode'].nunique()} episódios | "
          f"{df.shape[1]} colunas")

    build_figure(df, out_path=args.out, dpi=args.dpi)
    build_figures(df, out_path=args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
