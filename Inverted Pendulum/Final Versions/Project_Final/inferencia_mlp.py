"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Inferência — MLP tanh · Pêndulo Invertido Duplo                         ║
║  Carrega o modelo treinado (.pt) e oferece três modos de uso:            ║
║                                                                          ║
║  1. predict_single  — prediz α para um único estado                      ║
║  2. predict_batch   — prediz α para um DataFrame/CSV inteiro             ║
║  3. rollout         — integra a dinâmica passo a passo (Euler/RK4)       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════
# 1. DEFINIÇÃO DO MODELO
#    (deve ser idêntica à usada no treino)
# ══════════════════════════════════════════════════════════════════════════

class MLPtanh(nn.Module):
    def __init__(self, dim_in: int = 8, dim_out: int = 2,
                 hidden: tuple = (128, 128, 64)):
        super().__init__()
        dims   = [dim_in] + list(hidden) + [dim_out]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════
# 2. CLASSE DE INFERÊNCIA
# ══════════════════════════════════════════════════════════════════════════

class PendulumPredictor:
    """
    Carrega o checkpoint .pt e oferece predict_single, predict_batch
    e rollout.

    Uso mínimo:
        predictor = PendulumPredictor("mlp_tanh_pendulo.pt")
        alpha = predictor.predict_single(
            theta1=0.5, theta2=-0.3,
            omega1=1.2, omega2=-0.8,
            tau1=2.0,   tau2=0.5
        )
        print(alpha)   # {'alpha1': ..., 'alpha2': ...}
    """

    # Ordem das features — deve coincidir com o treino
    FEATURES = [
        "sin_theta1", "cos_theta1",
        "sin_theta2", "cos_theta2",
        "omega1",     "omega2",
        "tau1_dynamics", "tau2_dynamics",
    ]
    TARGETS = ["angle_accel1", "angle_accel2"]

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        self.device = self._resolve_device(device)
        self._load(checkpoint_path)
        print(f"[OK] Modelo carregado · dispositivo: {self.device}")
        print(f"     Arquitetura : {self.cfg['hidden']}")
        print(f"     Features    : {self.FEATURES}")
        print(f"     Targets     : {self.TARGETS}")

    # ── Setup ─────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        cfg    = ckpt.get("config", {})
        hidden = cfg.get("hidden", (128, 128, 64))
        self.cfg = cfg

        self.model = MLPtanh(
            dim_in  = len(self.FEATURES),
            dim_out = len(self.TARGETS),
            hidden  = hidden,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Scalers (salvos como arrays numpy no checkpoint)
        self.X_mean  = ckpt["scaler_X_mean"].astype(np.float32)
        self.X_scale = ckpt["scaler_X_scale"].astype(np.float32)
        self.y_mean  = ckpt["scaler_y_mean"].astype(np.float32)
        self.y_scale = ckpt["scaler_y_scale"].astype(np.float32)

    # ── Normalização / desnormalização ────────────────────────────────────

    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.X_mean) / self.X_scale

    def _denormalize_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_scale + self.y_mean

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(X.astype(np.float32)).to(self.device)

    # ── Construção do vetor de features ───────────────────────────────────

    @staticmethod
    def _state_to_features(theta1: float, theta2: float,
                            omega1: float, omega2: float,
                            tau1:   float, tau2:   float) -> np.ndarray:
        """
        Converte o estado físico (θ em radianos) para o vetor de features
        [sin θ₁, cos θ₁, sin θ₂, cos θ₂, ω₁, ω₂, τ₁, τ₂].
        """
        return np.array([[
            np.sin(theta1), np.cos(theta1),
            np.sin(theta2), np.cos(theta2),
            omega1, omega2,
            tau1,   tau2,
        ]], dtype=np.float32)

    # ══════════════════════════════════════════════════════════════════════
    # MODO 1 — predição para um único estado
    # ══════════════════════════════════════════════════════════════════════

    def predict_single(self,
                       theta1: float, theta2: float,
                       omega1: float, omega2: float,
                       tau1:   float, tau2:   float) -> dict:
        """
        Prediz as acelerações angulares para um único estado do sistema.

        Parâmetros
        ----------
        theta1, theta2 : ângulos dos elos em radianos
        omega1, omega2 : velocidades angulares em rad/s
        tau1, tau2     : torques em N·m

        Retorna
        -------
        dict com 'alpha1' e 'alpha2' em rad/s²
        """
        X_raw  = self._state_to_features(theta1, theta2, omega1, omega2, tau1, tau2)
        X_norm = self._normalize_X(X_raw)

        with torch.no_grad():
            y_norm = self.model(self._to_tensor(X_norm)).cpu().numpy()

        y_raw = self._denormalize_y(y_norm)[0]
        return {"alpha1": float(y_raw[0]), "alpha2": float(y_raw[1])}

    # ══════════════════════════════════════════════════════════════════════
    # MODO 2 — predição em lote (DataFrame ou CSV)
    # ══════════════════════════════════════════════════════════════════════

    def predict_batch(self, source, batch_size: int = 1024) -> pd.DataFrame:
        """
        Prediz acelerações para um conjunto de estados.

        Parâmetros
        ----------
        source : str (caminho para CSV) ou pd.DataFrame
                 Deve conter as colunas sin_theta1, cos_theta1,
                 sin_theta2, cos_theta2, omega1, omega2,
                 tau1_dynamics, tau2_dynamics.
                 Opcionalmente pode conter theta1, theta2 (escalares)
                 — nesse caso as colunas sin/cos são calculadas aqui.

        Retorna
        -------
        pd.DataFrame com colunas alpha1_pred e alpha2_pred
        """
        if isinstance(source, str):
            df = pd.read_csv(source)
        else:
            df = source.copy()

        # Se θ bruto vier sem sin/cos, calcular
        if "sin_theta1" not in df.columns and "theta1" in df.columns:
            df["sin_theta1"] = np.sin(df["theta1"])
            df["cos_theta1"] = np.cos(df["theta1"])
        if "sin_theta2" not in df.columns and "theta2" in df.columns:
            df["sin_theta2"] = np.sin(df["theta2"])
            df["cos_theta2"] = np.cos(df["theta2"])

        # Verificar colunas necessárias
        missing = [c for c in self.FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas ausentes no DataFrame: {missing}")

        X_raw  = df[self.FEATURES].values.astype(np.float32)
        X_norm = self._normalize_X(X_raw)

        preds = []
        with torch.no_grad():
            for start in range(0, len(X_norm), batch_size):
                batch = self._to_tensor(X_norm[start:start + batch_size])
                preds.append(self.model(batch).cpu().numpy())

        y_norm = np.concatenate(preds, axis=0)
        y_raw  = self._denormalize_y(y_norm)

        result = df.copy()
        result["alpha1_pred"] = y_raw[:, 0]
        result["alpha2_pred"] = y_raw[:, 1]
        return result

    # ══════════════════════════════════════════════════════════════════════
    # MODO 3 — rollout: integração passo a passo
    # ══════════════════════════════════════════════════════════════════════

    def rollout(self,
                theta1_0: float, theta2_0: float,
                omega1_0: float, omega2_0: float,
                tau_sequence: np.ndarray,
                dt: float = 0.002,
                method: str = "euler") -> pd.DataFrame:
        """
        Integra a dinâmica do pêndulo passo a passo usando a MLP.

        Parâmetros
        ----------
        theta1_0, theta2_0 : condição inicial dos ângulos (rad)
        omega1_0, omega2_0 : condição inicial das velocidades (rad/s)
        tau_sequence       : array (N, 2) com [τ₁, τ₂] em cada passo
        dt                 : passo de integração em segundos (padrão: 0.002)
        method             : "euler" ou "rk4"

        Retorna
        -------
        pd.DataFrame com colunas:
            t, theta1, theta2, omega1, omega2,
            tau1, tau2, alpha1_pred, alpha2_pred
        """
        assert method in ("euler", "rk4"), "method deve ser 'euler' ou 'rk4'"
        tau_sequence = np.asarray(tau_sequence, dtype=np.float64)
        if tau_sequence.ndim == 1:
            tau_sequence = tau_sequence.reshape(-1, 2)
        N = len(tau_sequence)

        # Estado inicial
        theta1, theta2 = float(theta1_0), float(theta2_0)
        omega1, omega2 = float(omega1_0), float(omega2_0)

        records = []

        def alpha_from_state(th1, th2, om1, om2, t1, t2):
            """Chama a rede e retorna (α₁, α₂)."""
            res = self.predict_single(th1, th2, om1, om2, t1, t2)
            return res["alpha1"], res["alpha2"]

        for i in range(N):
            tau1, tau2 = tau_sequence[i, 0], tau_sequence[i, 1]
            t_now = i * dt

            # Registrar estado ANTES da integração
            a1, a2 = alpha_from_state(theta1, theta2, omega1, omega2,
                                       tau1, tau2)
            records.append({
                "t":           t_now,
                "theta1":      theta1, "theta2":      theta2,
                "omega1":      omega1, "omega2":      omega2,
                "tau1":        tau1,   "tau2":        tau2,
                "alpha1_pred": a1,     "alpha2_pred": a2,
            })

            # Integração
            if method == "euler":
                omega1  += a1 * dt
                omega2  += a2 * dt
                theta1  += omega1 * dt
                theta2  += omega2 * dt

            else:  # RK4
                # k1
                a1k1, a2k1 = alpha_from_state(theta1, theta2,
                                               omega1, omega2, tau1, tau2)
                # k2
                om1_k2 = omega1 + 0.5*dt*a1k1
                om2_k2 = omega2 + 0.5*dt*a2k1
                th1_k2 = theta1 + 0.5*dt*om1_k2
                th2_k2 = theta2 + 0.5*dt*om2_k2
                a1k2, a2k2 = alpha_from_state(th1_k2, th2_k2,
                                               om1_k2, om2_k2, tau1, tau2)
                # k3
                om1_k3 = omega1 + 0.5*dt*a1k2
                om2_k3 = omega2 + 0.5*dt*a2k2
                th1_k3 = theta1 + 0.5*dt*om1_k3
                th2_k3 = theta2 + 0.5*dt*om2_k3
                a1k3, a2k3 = alpha_from_state(th1_k3, th2_k3,
                                               om1_k3, om2_k3, tau1, tau2)
                # k4
                om1_k4 = omega1 + dt*a1k3
                om2_k4 = omega2 + dt*a2k3
                th1_k4 = theta1 + dt*om1_k4
                th2_k4 = theta2 + dt*om2_k4
                a1k4, a2k4 = alpha_from_state(th1_k4, th2_k4,
                                               om1_k4, om2_k4, tau1, tau2)
                # Combinar
                omega1 += (dt/6) * (a1k1 + 2*a1k2 + 2*a1k3 + a1k4)
                omega2 += (dt/6) * (a2k1 + 2*a2k2 + 2*a2k3 + a2k4)
                theta1 += (dt/6) * ((omega1 - dt*a1k1/6) +
                                     2*(om1_k2 + om1_k3) + om1_k4)
                theta2 += (dt/6) * ((omega2 - dt*a2k1/6) +
                                     2*(om2_k2 + om2_k3) + om2_k4)

        return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════
# 3. DEMONSTRAÇÃO
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import os

    # ── Caminho do checkpoint ──────────────────────────────────────────────
    # Altere para o caminho do seu .pt
    CHECKPOINT = "C:\\Users\\Guilherme\\Mestrado\\Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Project_Final\\Results\\mlp_tanh_pendulo.pt"

    if not Path(CHECKPOINT).exists():
        print(f"[ERRO] Arquivo não encontrado: {CHECKPOINT}")
        print("       Altere a variável CHECKPOINT para o caminho correto.")
        exit(1)

    # ── Carregar modelo ────────────────────────────────────────────────────
    predictor = PendulumPredictor(CHECKPOINT)

    print("\n" + "="*60)

    # ── MODO 1: predição para um único estado ──────────────────────────────
    print("\n  MODO 1 — predição para um único estado")
    print("  " + "─"*50)

    resultado = predictor.predict_single(
        theta1 =  0.5,    # rad
        theta2 = -0.3,    # rad
        omega1 =  1.2,    # rad/s
        omega2 = -0.8,    # rad/s
        tau1   =  2.0,    # N·m
        tau2   =  0.5,    # N·m
    )
    print(f"  Entrada : θ₁=0.5 rad, θ₂=-0.3 rad, ω₁=1.2 rad/s, ω₂=-0.8 rad/s")
    print(f"  Saída   : α₁ = {resultado['alpha1']:+.4f} rad/s²")
    print(f"            α₂ = {resultado['alpha2']:+.4f} rad/s²")

    # ── MODO 2: predição em lote sobre CSV ────────────────────────────────
    CSV = "C:\\Users\\Guilherme\\Mestrado\\Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Project_Final\\Data Processed\\pendulum_dataset_tidy_with_acceleration.csv"
    if Path(CSV).exists():
        print(f"\n  MODO 2 — predição em lote sobre {CSV}")
        print("  " + "─"*50)

        df = pd.read_csv(CSV)
        # Usar apenas episódio 4 (teste) como exemplo
        df_ep4 = df[df["episode"] == 4].head(200)

        df_pred = predictor.predict_batch(df_ep4)

        # Comparar com valores reais
        e1 = df_pred["angle_accel1"] - df_pred["alpha1_pred"]
        e2 = df_pred["angle_accel2"] - df_pred["alpha2_pred"]
        r2_1 = 1 - (e1**2).sum() / ((df_pred["angle_accel1"] -
                                       df_pred["angle_accel1"].mean())**2).sum()
        r2_2 = 1 - (e2**2).sum() / ((df_pred["angle_accel2"] -
                                       df_pred["angle_accel2"].mean())**2).sum()

        print(f"  Amostras avaliadas : {len(df_pred)}")
        print(f"  R² α₁ : {r2_1:.6f}   RMSE: {np.sqrt((e1**2).mean()):.4f} rad/s²")
        print(f"  R² α₂ : {r2_2:.6f}   RMSE: {np.sqrt((e2**2).mean()):.4f} rad/s²")

        # Salvar predições
        OUT_CSV = "C:\\Users\\Guilherme\\Mestrado\\Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Project_Final\\Results\\predicoes_ep4.csv"
        df_pred.to_csv(OUT_CSV, index=False)
        print(f"  Predições salvas em: {OUT_CSV}")

    # ── MODO 3: rollout ───────────────────────────────────────────────────
    print("\n  MODO 3 — rollout de 2 segundos (Euler, dt=0.002s)")
    print("  " + "─"*50)

    N_STEPS = 1000          # 1000 × 0.002s = 2 segundos
    DT      = 0.002         # passo de tempo do MuJoCo

    # Torques constantes como exemplo
    # Na prática: substituir por sequência de controle real
    tau_seq = np.zeros((N_STEPS, 2))
    tau_seq[:, 0] = 1.0   # τ₁ constante
    tau_seq[:, 1] = 0.0   # τ₂ zero

    traj = predictor.rollout(
        theta1_0 =  0.1,    # condição inicial θ₁
        theta2_0 =  0.05,   # condição inicial θ₂
        omega1_0 =  0.0,    # parado inicialmente
        omega2_0 =  0.0,
        tau_sequence = tau_seq,
        dt = DT,
        method = "euler",   # ou "rk4" para maior precisão
    )

    print(f"  Passos integrados : {len(traj)}")
    print(f"  Tempo simulado    : {traj['t'].iloc[-1]:.3f}s")
    print(f"\n  Primeiros 5 passos:")
    print(traj[["t","theta1","theta2","omega1","omega2",
                 "alpha1_pred","alpha2_pred"]].head(5).to_string(index=False))
    print(f"\n  Últimos 5 passos:")
    print(traj[["t","theta1","theta2","omega1","omega2",
                 "alpha1_pred","alpha2_pred"]].tail(5).to_string(index=False))

    # Salvar trajetória
    OUT_ROLLOUT = "C:\\Users\\Guilherme\\Mestrado\\Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Project_Final\\Results\\MLP_Results\\rollout_2s.csv"
    traj.to_csv(OUT_ROLLOUT, index=False)
    print(f"\n  Trajetória salva em: {OUT_ROLLOUT}")

    print("\n" + "="*60)
    print("  Concluído.")
    print("="*60)
