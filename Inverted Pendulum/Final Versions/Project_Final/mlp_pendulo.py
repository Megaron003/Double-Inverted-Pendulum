"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MLP com ativação tanh — Pêndulo Invertido Duplo                        ║
║  Predição de acelerações angulares [α₁, α₂]                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Justificativa arquitetural (testes de hipótese):                        ║
║  · Feedforward estático  ← H3  (ΔR²=0,004% com lag → sem memória)      ║
║  · Ativação tanh          ← H4  (RESET ímpar confirma estrutura sin/cos)║
║  · Entrada dim=8          ← H1+H2 (sin/cos θ, ω, τ — ambos os elos)    ║
║  · Perda Huber            ← H5  (t de Student ν≈3,15 → δ*≈37 e ≈86)   ║
║  · Split por episódios    ← HN-E (generalização entre cond. iniciais)   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# ── Reprodutibilidade ──────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")

# ── Caminhos ───────────────────────────────────────────────────────────────
CSV      = "C:\\Users\\Guilherme\\Mestrado\\Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Data Processed\\pendulum_dataset_tidy_with_acceleration.csv"
OUT_FIG  = "C:\\Users\\Guilherme\\Mestrado\\Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Project_Final\\Results\\MLP_Results\\mlp_tanh_diagnostico.png"
OUT_PT   = "C:\\Users\\Guilherme\\Mestrado\\Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Project_Final\\Results\\MLP_Results\\mlp_tanh_pendulo.pt"
CKPT     = "C:\\Users\\Guilherme\\Mestrado\\Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Project_Final\\Results\\MLP_Results\\best_model.pt"

# ── Colunas ────────────────────────────────────────────────────────────────
# sin/cos já presentes no dataset (HN-F: sin θ₂ = 99,97% do ganho de θ₂)
FEATURES = [
    "sin_theta1", "cos_theta1",
    "sin_theta2", "cos_theta2",
    "omega1",     "omega2",
    "tau1_dynamics", "tau2_dynamics",
]
TARGETS = ["angle_accel1", "angle_accel2"]

# Split por episódio — correto para simulação (H3/HN-E)
TRAIN_EPS = [0, 1, 2]
VAL_EPS   = [3]
TEST_EPS  = [4]

# ══════════════════════════════════════════════════════════════════════════
# 1. DADOS
# ══════════════════════════════════════════════════════════════════════════

class PendulumDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_splits(csv_path: str, batch_size: int):
    df = pd.read_csv(csv_path)
    print(f"\nDataset: {len(df):,} amostras · {df['episode'].nunique()} episódios")

    tr = df[df["episode"].isin(TRAIN_EPS)].reset_index(drop=True)
    va = df[df["episode"].isin(VAL_EPS)].reset_index(drop=True)
    te = df[df["episode"].isin(TEST_EPS)].reset_index(drop=True)
    print(f"  Treino (ep {TRAIN_EPS}): {len(tr):,}   "
          f"Val (ep {VAL_EPS}): {len(va):,}   "
          f"Teste (ep {TEST_EPS}): {len(te):,}")

    # Scaler fitado APENAS no treino
    sx = StandardScaler().fit(tr[FEATURES])
    sy = StandardScaler().fit(tr[TARGETS])

    def arrays(split_df):
        X = sx.transform(split_df[FEATURES].values).astype(np.float32)
        y = sy.transform(split_df[TARGETS].values).astype(np.float32)
        return X, y

    X_tr, y_tr = arrays(tr)
    X_va, y_va = arrays(va)
    X_te, y_te = arrays(te)

    # Loaders
    # IMPORTANTE: train_eval_loader usa shuffle=False — necessário para
    # alinhar predições com y_raw ao avaliar métricas no treino
    train_loader      = DataLoader(PendulumDataset(X_tr, y_tr),
                                    batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(PendulumDataset(X_tr, y_tr),
                                    batch_size=batch_size, shuffle=False)
    val_loader        = DataLoader(PendulumDataset(X_va, y_va),
                                    batch_size=batch_size, shuffle=False)
    test_loader       = DataLoader(PendulumDataset(X_te, y_te),
                                    batch_size=batch_size, shuffle=False)

    # Targets originais (sem normalização) para métricas em rad/s²
    y_tr_raw = tr[TARGETS].values.astype(np.float32)
    y_te_raw = te[TARGETS].values.astype(np.float32)

    return (train_loader, train_eval_loader, val_loader, test_loader,
            y_tr_raw, y_te_raw, sx, sy)


# ══════════════════════════════════════════════════════════════════════════
# 2. MODELO
# ══════════════════════════════════════════════════════════════════════════

class MLPtanh(nn.Module):
    """
    MLP feedforward com ativação tanh entre camadas ocultas.

    Arquitetura padrão para este problema:
      8 → 128 → 128 → 64 → 2   (26.050 parâmetros)

    Notas:
    · tanh captura sin(θ) via série de Taylor ímpar (H4)
    · Sem ativação na saída — α ∈ ℝ não está restrita a (−1,+1)
    · Xavier uniform na inicialização — recomendado para tanh
    """

    def __init__(self, dim_in: int = 8, dim_out: int = 2,
                 hidden: tuple = (128, 128, 64)):
        super().__init__()
        dims   = [dim_in] + list(hidden) + [dim_out]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:       # tanh só nas camadas ocultas
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self._xavier_init()

    def _xavier_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════
# 3. TREINO
# ══════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            loss = criterion(pred, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(Xb)
            n += len(Xb)
    return total_loss / n


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    """Retorna predições concatenadas em ordem (loader deve ter shuffle=False)."""
    model.eval()
    preds = []
    for Xb, _ in loader:
        preds.append(model(Xb.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def fit(model, train_loader, val_loader, cfg, device):
    """Treina com early stopping e ReduceLROnPlateau."""
    criterion = nn.HuberLoss(delta=cfg["delta"])
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=cfg["lr"],
                                  weight_decay=cfg["wd"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=cfg["lr_pat"], min_lr=1e-6)

    best_val, no_imp = float("inf"), 0
    history = {"train": [], "val": [], "lr": []}

    print(f"\n{'Época':>6}  {'Treino':>10}  {'Val':>10}  {'LR':>10}  {'s':>6}")
    print("─" * 50)

    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss  = run_epoch(model, val_loader,  criterion, optimizer, device, train=False)
        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        history["train"].append(tr_loss)
        history["val"].append(va_loss)
        history["lr"].append(lr_now)

        if ep % 10 == 0 or ep == 1:
            print(f"{ep:>6}  {tr_loss:>10.6f}  {va_loss:>10.6f}  "
                  f"{lr_now:>10.2e}  {time.time()-t0:>5.1f}s")

        if va_loss < best_val - 1e-7:
            best_val, no_imp = va_loss, 0
            torch.save(model.state_dict(), CKPT)
        else:
            no_imp += 1
            if no_imp >= cfg["es_pat"]:
                print(f"\n  Early stopping na época {ep} "
                      f"(melhor val = {best_val:.6f})")
                break

    model.load_state_dict(torch.load(CKPT, map_location=device))
    print(f"\n  Melhor val loss = {best_val:.6f}")
    return history


# ══════════════════════════════════════════════════════════════════════════
# 4. MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════

def metrics(y_true_raw: np.ndarray,
            y_pred_norm: np.ndarray,
            sy: StandardScaler) -> dict:
    """Calcula métricas no espaço original (rad/s²)."""
    y_pred_raw = sy.inverse_transform(y_pred_norm)
    out = {}
    for i, name in enumerate(["α₁", "α₂"]):
        t, p = y_true_raw[:, i], y_pred_raw[:, i]
        e    = t - p
        out[name] = {
            "R²":   1 - np.sum(e**2) / np.sum((t - t.mean())**2),
            "RMSE": float(np.sqrt(np.mean(e**2))),
            "MAE":  float(np.mean(np.abs(e))),
            "ρ_S":  float(stats.spearmanr(t, p).statistic),
        }
    return out, y_pred_raw


def print_metrics(m: dict, title: str):
    print(f"\n  {'─'*52}")
    print(f"  {title}  (rad/s²)")
    print(f"  {'─'*52}")
    print(f"  {'':4}  {'R²':>8}  {'RMSE':>10}  {'MAE':>10}  {'ρ_Spearman':>12}")
    for name, v in m.items():
        print(f"  {name:4}  {v['R²']:>8.4f}  {v['RMSE']:>10.4f}  "
              f"{v['MAE']:>10.4f}  {v['ρ_S']:>12.4f}")


# ══════════════════════════════════════════════════════════════════════════
# 5. FIGURA DE DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════════════

def plot_diagnostics(history, y_tr_raw, yp_tr, y_te_raw, yp_te):

    C = {"bg":"#0D1117","panel":"#161B22","grid":"#21262D","text":"#E6EDF3",
         "a1":"#58A6FF","a2":"#F87171","a3":"#3FB950","a4":"#D2A8FF",
         "a5":"#FFA657","slate":"#64748B"}

    def sa(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["slate"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(C["grid"]); sp.set_linewidth(0.6)
        ax.grid(True, color=C["grid"], lw=0.4, alpha=0.7)
        if title:  ax.set_title(title,  color=C["text"], fontsize=8.5,
                                fontweight="bold", pad=5)
        if xlabel: ax.set_xlabel(xlabel, color=C["slate"], fontsize=8)
        if ylabel: ax.set_ylabel(ylabel, color=C["slate"], fontsize=8)

    rng = np.random.default_rng(42)
    fig = plt.figure(figsize=(20, 18), facecolor=C["bg"])
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.40)

    # ── (a) Curva de aprendizado ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0:2])
    sa(ax, "(a) Curva de aprendizado", "Época", "Perda Huber (normalizada)")
    ep = range(1, len(history["train"]) + 1)
    ax.semilogy(ep, history["train"], color=C["a1"], lw=2, label="Treino")
    ax.semilogy(ep, history["val"],   color=C["a2"], lw=2, label="Val")
    best_ep = int(np.argmin(history["val"])) + 1
    ax.axvline(best_ep, color=C["a5"], lw=1.5, ls="--",
               label=f"Melhor val (ép. {best_ep})")
    ax.legend(fontsize=8, facecolor=C["panel"], labelcolor=C["text"])

    # ── (b) Learning rate schedule ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2:4])
    sa(ax, "(b) Learning rate", "Época", "LR")
    ax.semilogy(ep, history["lr"], color=C["a4"], lw=2)

    # ── Para cada variável: scatter + resíduos ────────────────────────────
    VAR_COLS = [(C["a1"], C["a3"]), (C["a2"], C["a4"])]
    var_names = ["α₁  (angle_accel1)", "α₂  (angle_accel2)"]

    for vi in range(2):
        col_s, col_r = VAR_COLS[vi]
        row = vi + 1

        # Scatter: predito vs real
        ax = fig.add_subplot(gs[row, 0:2])
        sa(ax, f"({'cd'[vi]}) Predito vs Real — {var_names[vi]}  |  Teste ep.4",
           "α real (rad/s²)", "α predito (rad/s²)")

        idx = rng.choice(len(y_te_raw), size=3000, replace=False)
        ax.scatter(y_te_raw[idx, vi], yp_te[idx, vi],
                   s=0.8, alpha=0.3, color=col_s)

        lo = min(y_te_raw[:, vi].min(), yp_te[:, vi].min())
        hi = max(y_te_raw[:, vi].max(), yp_te[:, vi].max())
        ax.plot([lo, hi], [lo, hi], color=C["a5"], lw=1.8,
                ls="--", label="y = x  (perfeito)")

        e    = y_te_raw[:, vi] - yp_te[:, vi]
        r2   = 1 - np.sum(e**2) / np.sum((y_te_raw[:, vi] -
                                            y_te_raw[:, vi].mean())**2)
        rmse = np.sqrt(np.mean(e**2))
        ax.text(0.04, 0.94,
                f"R² = {r2:.4f}\nRMSE = {rmse:.3f} rad/s²",
                transform=ax.transAxes, color=col_s,
                fontsize=8.5, va="top", style="italic")
        ax.legend(fontsize=8, facecolor=C["panel"], labelcolor=C["text"])

        # Distribuição de resíduos
        ax = fig.add_subplot(gs[row, 2:4])
        sa(ax, f"({'ef'[vi]}) Resíduos — {var_names[vi]}",
           "Resíduo (rad/s²)", "Densidade")

        for y_raw, yp, lbl, col in [
            (y_te_raw, yp_te,  "Teste ep.4",  col_s),
            (y_tr_raw, yp_tr,  "Treino ep.0-2", C["slate"]),
        ]:
            res = y_raw[:, vi] - yp[:, vi]
            ax.hist(res, bins=80, density=True,
                    color=col, alpha=0.55, edgecolor="none", label=lbl)

        res_te = y_te_raw[:, vi] - yp_te[:, vi]
        mu, sg = res_te.mean(), res_te.std()
        xr = np.linspace(res_te.min(), res_te.max(), 300)
        ax.plot(xr, stats.norm.pdf(xr, mu, sg),
                color=C["a5"], lw=1.8, ls="--", label="Normal ref.")
        ax.legend(fontsize=8, facecolor=C["panel"], labelcolor=C["text"])
        ax.text(0.97, 0.94, f"μ = {mu:+.2f}\nσ = {sg:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                color=col_s, fontsize=8.5, style="italic")

    fig.suptitle(
        "MLP tanh — Pêndulo Invertido Duplo\n"
        "Treino: ep.0-2  ·  Val: ep.3  ·  Teste: ep.4  (condição inicial não vista)",
        color=C["text"], fontsize=11, fontweight="bold", y=0.998
    )
    plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"\n  [OK] Figura salva: {OUT_FIG}")


# ══════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():

    CFG = {
        "hidden":  (128, 128, 64),
        "lr":      1e-3,
        "wd":      1e-5,
        "batch":   512,
        "epochs":  200,
        "delta":   1.0,   # Huber δ em espaço normalizado
        "lr_pat":  10,    # ReduceLROnPlateau patience
        "es_pat":  30,    # EarlyStopping patience
    }

    print("=" * 60)
    print("  MLP tanh — Pêndulo Invertido Duplo")
    print("=" * 60)

    # ── Dados ──────────────────────────────────────────────────────────────
    (train_loader, train_eval_loader, val_loader, test_loader,
     y_tr_raw, y_te_raw, sx, sy) = load_splits(CSV, CFG["batch"])

    # ── Modelo ─────────────────────────────────────────────────────────────
    model = MLPtanh(
        dim_in  = len(FEATURES),
        dim_out = len(TARGETS),
        hidden  = CFG["hidden"],
    ).to(DEVICE)

    print(f"\n  Arquitetura : {len(FEATURES)} → {CFG['hidden']} → {len(TARGETS)}")
    print(f"  Parâmetros  : {model.n_params:,}")
    print(f"  Ativação    : tanh  (confirma H4)")
    print(f"  Perda       : Huber δ={CFG['delta']}  (confirma H5)")
    print(f"  Dispositivo : {DEVICE}")

    # ── Treino ─────────────────────────────────────────────────────────────
    t0 = time.time()
    history = fit(model, train_loader, val_loader, CFG, DEVICE)
    print(f"\n  Tempo total de treino: {time.time()-t0:.1f}s")

    # ── Predições alinhadas (shuffle=False obrigatório) ────────────────────
    yp_tr_norm = predict(model, train_eval_loader, DEVICE)   # treino
    yp_te_norm = predict(model, test_loader,       DEVICE)   # teste

    # ── Métricas no espaço original ────────────────────────────────────────
    m_tr, yp_tr = metrics(y_tr_raw, yp_tr_norm, sy)
    m_te, yp_te = metrics(y_te_raw, yp_te_norm, sy)

    print_metrics(m_tr, "Treino  (ep.0-2)")
    print_metrics(m_te, "Teste   (ep.4 — nunca visto)")

    # ── Figura ────────────────────────────────────────────────────────────
    plot_diagnostics(history, y_tr_raw, yp_tr, y_te_raw, yp_te)

    # ── Salvar ───────────────────────────────────────────────────────────
    torch.save({
        "model_state_dict": model.state_dict(),
        "config":           CFG,
        "features":         FEATURES,
        "targets":          TARGETS,
        "scaler_X_mean":    sx.mean_,
        "scaler_X_scale":   sx.scale_,
        "scaler_y_mean":    sy.mean_,
        "scaler_y_scale":   sy.scale_,
        "metrics_train":    m_tr,
        "metrics_test":     m_te,
        "history":          history,
    }, OUT_PT)
    print(f"  [OK] Modelo salvo: {OUT_PT}")

    # ── Resumo final ──────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  RESUMO FINAL")
    print(f"{'═'*60}")
    for split, m in [("Treino", m_tr), ("Teste", m_te)]:
        for var, v in m.items():
            print(f"  {split:6} {var}: "
                  f"R²={v['R²']:.4f}  "
                  f"RMSE={v['RMSE']:.4f} rad/s²  "
                  f"MAE={v['MAE']:.4f}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
