"""
chaos_per_episode.py
====================
Gera 12 graficos a partir do dataset do pendulo invertido duplo:

  SCATTER CAOS (6 figuras):
    - 5 individuais (um por episodio)
    - 1 com todos sobrepostos

  CHAOS TOPOLOGY (6 figuras):
    - 5 individuais (um por episodio)
    - 1 com todos sobrepostos

Uso:
    python chaos_per_episode.py

Saida: pasta ./output_chaos/ com 12 arquivos PNG.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from scipy.ndimage import uniform_filter1d
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# =================================================================
#  CONFIGURACAO
# =================================================================
CSV_PATH      = "D:\\Trabalhos\\Smart Agri\\Double_Inverted_Pendulum\\Double-Inverted-Pendulum\\Inverted Pendulum\\Final Versions\\Data Processed\\pendulum_dataset_tidy_with_acceleration.csv"
OUTPUT_DIR    = "output_chaos"
W_LOCAL       = 300    # janela para metricas locais (~0.6 s a 2ms/step)
W_CORR        = 500    # janela correlacao rolling
STEP_CORR     = 20     # passo da correlacao rolling
LAMBDA_THRESH = 0.05   # limiar lambda -> caotico
DT            = 0.002  # timestep (s)
N_SCATTER     = 4000   # pontos no scatter (subsample)

CHAOS_CMAP = LinearSegmentedColormap.from_list(
    'chaos', ['#0C447C','#1D9E75','#EF9F27','#D85A30','#A32D2D'], N=256
)
EP_COLORS = ['#00d4ff','#ff6b35','#7fff6b','#bf7fff','#ffcc00']

plt.rcParams.update({
    'figure.facecolor': '#080c10',
    'axes.facecolor':   '#0d1318',
    'axes.edgecolor':   '#1a2a38',
    'axes.labelcolor':  '#c2cdd8',
    'xtick.color':      '#4a5a6a',
    'ytick.color':      '#4a5a6a',
    'text.color':       '#c2cdd8',
    'grid.color':       '#1a2a38',
    'grid.linewidth':   0.5,
    'font.family':      'DejaVu Sans',
    'font.size':        9,
    'axes.titlesize':   10,
    'axes.titleweight': 'bold',
    'axes.titlecolor':  '#e4edf6',
    'axes.labelsize':   8.5,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =================================================================
#  FUNCOES DE ANALISE
# =================================================================

def local_lyapunov(o1, o2, W=300, step=30, theiler=20):
    n     = len(o1)
    lam   = np.full(n, np.nan)
    state = np.column_stack([o1, o2])
    for start in range(0, n - W, step):
        seg  = state[start:start + W]
        divs = []
        for i in range(0, len(seg) - 1, 5):
            dists = np.linalg.norm(seg - seg[i], axis=1)
            dists[max(0, i - theiler): min(len(seg), i + theiler)] = np.inf
            nn = np.argmin(dists)
            d0 = dists[nn]
            if d0 < 1e-9: continue
            fi  = min(i + 1, len(seg) - 1)
            fnn = min(nn + 1, len(seg) - 1)
            d1  = np.linalg.norm(seg[fi] - seg[fnn])
            if d1 > 1e-9:
                divs.append(np.log(d1 / d0))
        if divs:
            lam[start + W // 2] = np.mean(divs)
    mask = ~np.isnan(lam)
    if mask.sum() > 2:
        lam = np.interp(np.arange(n), np.where(mask)[0], lam[mask])
    return lam


def compute_metrics(ep_df):
    o1   = ep_df['omega1'].values.astype(float)
    o2   = ep_df['omega2'].values.astype(float)
    a1   = ep_df['angle_accel1'].values.astype(float)
    a2   = ep_df['angle_accel2'].values.astype(float)
    t    = ep_df['time'].values.astype(float)
    n    = len(t)

    speed     = np.sqrt(o1**2 + o2**2)
    arc_len   = uniform_filter1d(speed, size=W_LOCAL) * W_LOCAL * DT
    accel_mag = uniform_filter1d(np.sqrt(a1**2 + a2**2), size=W_LOCAL)
    Ek        = 0.5 * (o1**2 + o2**2)
    lam       = local_lyapunov(o1, o2, W=W_LOCAL, step=30)
    lam_sm    = uniform_filter1d(lam, size=W_LOCAL)

    t_c, r_arc, r_accel, r_Ek = [], [], [], []
    for end in range(W_CORR, n, STEP_CORR):
        sl = slice(end - W_CORR, end)
        la = lam[sl]; ar = arc_len[sl]; ac = accel_mag[sl]; ek = Ek[sl]
        mk = ~np.isnan(la)
        if mk.sum() < 50: continue
        t_c.append(t[end - W_CORR // 2])
        r_arc.append(  pearsonr(ar[mk], la[mk])[0])
        r_accel.append(pearsonr(ac[mk], la[mk])[0])
        r_Ek.append(   pearsonr(ek[mk], la[mk])[0])

    return dict(t=t, o1=o1, o2=o2, speed=speed, arc_len=arc_len,
                accel_mag=accel_mag, Ek=Ek, lam=lam, lam_smooth=lam_sm,
                is_chaotic=lam_sm > LAMBDA_THRESH,
                t_corr=np.array(t_c),
                r_arc=np.array(r_arc),
                r_accel=np.array(r_accel),
                r_Ek=np.array(r_Ek))


def shade_chaos(ax, t, is_chaotic):
    in_block = False; x0 = None
    for i in range(len(t)):
        if is_chaotic[i] and not in_block:
            x0 = t[i]; in_block = True
        elif not is_chaotic[i] and in_block:
            ax.axvspan(x0, t[i], color='#ff3c3c', alpha=0.09, linewidth=0)
            in_block = False
    if in_block:
        ax.axvspan(x0, t[-1], color='#ff3c3c', alpha=0.09, linewidth=0)


# =================================================================
#  LOAD + COMPUTE
# =================================================================
print("Carregando dataset...")
df       = pd.read_csv(CSV_PATH)
episodes = sorted(df['episode'].unique())
print(f"  {len(episodes)} episodios, {len(df)} linhas\n")

print("Computando metricas (Lyapunov local ~30s)...")
metrics = {}
for ep in episodes:
    print(f"  Ep {ep}...", end=' ', flush=True)
    sub = df[df['episode'] == ep].reset_index(drop=True)
    metrics[ep] = compute_metrics(sub)
    m = metrics[ep]
    print(f"lambda_mean={np.nanmean(m['lam']):.4f}  "
          f"frac_caotica={m['is_chaotic'].mean():.1%}")

all_lam = np.concatenate([metrics[ep]['lam'] for ep in episodes])
VMIN    = np.nanpercentile(all_lam, 5)
VMAX    = np.nanpercentile(all_lam, 95)
NORM    = Normalize(vmin=VMIN, vmax=VMAX)
print("\nMetricas prontas. Gerando graficos...\n")


# =================================================================
#  SCATTER CAOS — individual
# =================================================================
def plot_scatter_episode(ep, m, path):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#080c10')
    fig.suptitle(
        f'Scatter Caos — Episodio {ep}   '
        f'lambda_mean={np.nanmean(m["lam"]):.4f}  |  '
        f'Fracao caotica={m["is_chaotic"].mean():.1%}',
        fontsize=12, color='#e4edf6', y=0.99, fontweight='bold')

    gs = gridspec.GridSpec(2, 3, hspace=0.44, wspace=0.32,
                           top=0.93, bottom=0.08, left=0.06, right=0.97)

    n   = len(m['t'])
    idx = np.random.choice(n, min(N_SCATTER, n), replace=False)
    ts  = m['t'][idx];       ls = m['lam'][idx]
    sps = m['speed'][idx];   acs= m['accel_mag'][idx]; Eks= m['Ek'][idx]
    gd  = ~np.isnan(ls)
    ts=ts[gd]; ls=ls[gd]; sps=sps[gd]; acs=acs[gd]; Eks=Eks[gd]
    tn  = (ts - ts.min()) / max(ts.max() - ts.min(), 1e-9)

    pairs = [
        (sps, '||omega|| (rad/s)',       'Velocidade no espaco de fases', m['r_arc']),
        (acs, '||alpha|| (rad/s2) local','Aceleracao angular local',      m['r_accel']),
        (Eks, 'Ek = 0.5*(w1^2+w2^2)',   'Energia cinetica proxy',        m['r_Ek']),
    ]

    for col, (xdata, xlabel, title, r_roll) in enumerate(pairs):
        ax_sc = fig.add_subplot(gs[0, col])
        sc = ax_sc.scatter(xdata, ls, c=tn, cmap='plasma', s=3,
                           alpha=0.55, linewidths=0, norm=Normalize(0, 1))
        r_g, _ = pearsonr(xdata, ls)
        ax_sc.axhline(0, color='#ffffff33', lw=0.8)
        ax_sc.axhline(LAMBDA_THRESH, color='#ffcc00', lw=0.9, ls='--', alpha=0.7)
        ax_sc.text(0.97, 0.96, f'r = {r_g:.3f}',
                   transform=ax_sc.transAxes, ha='right', va='top',
                   fontsize=9, color='#ffcc00',
                   bbox=dict(boxstyle='round,pad=0.3', fc='#0d1318', alpha=0.8))
        ax_sc.set_xlabel(xlabel); ax_sc.set_ylabel('lambda_local (nats/passo)')
        ax_sc.set_title(title); ax_sc.grid(True)
        if col == 0:
            cb = plt.colorbar(sc, ax=ax_sc, fraction=0.04, pad=0.02)
            cb.set_label('Tempo norm.', fontsize=7, color='#c2cdd8')
            plt.setp(cb.ax.yaxis.get_ticklabels(), color='#c2cdd8', fontsize=7)

        ax_r = fig.add_subplot(gs[1, col])
        tc   = m['t_corr']; rr = r_roll
        if len(tc) > 0:
            ax_r.plot(tc, rr, color=EP_COLORS[ep % len(EP_COLORS)], lw=1.2)
            ax_r.fill_between(tc, 0, rr, where=rr > 0,  color='#00d4ff', alpha=0.18)
            ax_r.fill_between(tc, 0, rr, where=rr <= 0, color='#ff3c3c', alpha=0.18)
            ic_at_tc = m['is_chaotic'][
                np.searchsorted(m['t'], tc).clip(0, len(m['t'])-1)]
            shade_chaos(ax_r, tc, ic_at_tc)
        ax_r.axhline(0, color='#ffffff44', lw=0.8)
        ax_r.set_ylim(-1, 1)
        ax_r.set_xlabel('Tempo (s)'); ax_r.set_ylabel('Pearson r (rolling)')
        ax_r.set_title(f'Correlacao rolling (W={W_CORR*DT:.1f}s)')
        ax_r.grid(True)

    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor='#080c10')
    plt.close()
    print(f"  Salvo: {path}")


# =================================================================
#  CHAOS TOPOLOGY — individual
# =================================================================
def plot_topology_episode(ep, m, path):
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor('#080c10')
    fig.suptitle(
        f'Chaos Topology — Episodio {ep}   '
        f'lambda_mean={np.nanmean(m["lam"]):.4f}  |  '
        f'Frac. caotica={m["is_chaotic"].mean():.1%}  |  '
        f'lambda_max={np.nanmax(m["lam"]):.4f}',
        fontsize=12, color='#e4edf6', y=0.99, fontweight='bold')

    gs = gridspec.GridSpec(2, 3, hspace=0.44, wspace=0.32,
                           top=0.93, bottom=0.08, left=0.06, right=0.97)

    o1=m['o1']; o2=m['o2']; lam=m['lam']; t=m['t']; lam_sm=m['lam_smooth']

    # Painel 1: scatter espaco de fases x lambda
    ax1 = fig.add_subplot(gs[0, 0])
    sc1 = ax1.scatter(o1[::2], o2[::2], c=lam[::2], cmap=CHAOS_CMAP, norm=NORM,
                      s=0.8, alpha=0.7, linewidths=0)
    cb1 = plt.colorbar(sc1, ax=ax1, fraction=0.04, pad=0.02)
    cb1.set_label('lambda_local', fontsize=7, color='#c2cdd8')
    plt.setp(cb1.ax.yaxis.get_ticklabels(), color='#c2cdd8', fontsize=7)
    ax1.axhline(0, color='#1a2a38', lw=0.5); ax1.axvline(0, color='#1a2a38', lw=0.5)
    ax1.set_xlabel('omega1 (rad/s)'); ax1.set_ylabel('omega2 (rad/s)')
    ax1.set_title('Espaco de fases (w1,w2) — cor = lambda_local'); ax1.grid(True)

    # Painel 2: hexbin lambda medio
    ax2 = fig.add_subplot(gs[0, 1])
    hb  = ax2.hexbin(o1, o2, C=lam, gridsize=38, cmap=CHAOS_CMAP, norm=NORM,
                     reduce_C_function=np.nanmean, linewidths=0.1)
    cb2 = plt.colorbar(hb, ax=ax2, fraction=0.04, pad=0.02)
    cb2.set_label('lambda medio/celula', fontsize=7, color='#c2cdd8')
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color='#c2cdd8', fontsize=7)
    ax2.axhline(0, color='#ffffff15', lw=0.5); ax2.axvline(0, color='#ffffff15', lw=0.5)
    ax2.set_xlabel('omega1 (rad/s)'); ax2.set_ylabel('omega2 (rad/s)')
    ax2.set_title('Hexbin: lambda medio por zona do espaco de fases'); ax2.grid(True)

    # Painel 3: ||omega|| x lambda colorido por Ek
    ax3 = fig.add_subplot(gs[0, 2])
    idx = np.random.choice(len(o1), min(N_SCATTER, len(o1)), replace=False)
    sp  = m['speed'][idx]; la = lam[idx]; ek = m['Ek'][idx]
    gd  = ~np.isnan(la)
    sc3 = ax3.scatter(sp[gd], la[gd], c=ek[gd], cmap='inferno', s=3, alpha=0.6,
                      linewidths=0,
                      norm=Normalize(np.percentile(ek,2), np.percentile(ek,98)))
    cb3 = plt.colorbar(sc3, ax=ax3, fraction=0.04, pad=0.02)
    cb3.set_label('Ek (proxy)', fontsize=7, color='#c2cdd8')
    plt.setp(cb3.ax.yaxis.get_ticklabels(), color='#c2cdd8', fontsize=7)
    ax3.axhline(0, color='#ffffff33', lw=0.8)
    ax3.axhline(LAMBDA_THRESH, color='#ffcc00', lw=0.8, ls='--', alpha=0.7)
    r_g, _ = pearsonr(sp[gd], la[gd])
    ax3.text(0.97, 0.96, f'r = {r_g:.3f}', transform=ax3.transAxes,
             ha='right', va='top', fontsize=9, color='#ffcc00',
             bbox=dict(boxstyle='round,pad=0.3', fc='#0d1318', alpha=0.8))
    ax3.set_xlabel('||omega|| (rad/s)'); ax3.set_ylabel('lambda_local (nats/passo)')
    ax3.set_title('||omega|| x lambda — cor = Energia cinetica Ek'); ax3.grid(True)

    # Painel 4 (ocupa linha inteira): timeline lambda
    ax4 = fig.add_subplot(gs[1, :])
    pts  = np.array([t, lam_sm]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap=CHAOS_CMAP, norm=NORM, linewidth=2.0)
    lc.set_array(lam_sm[:-1])
    ax4.add_collection(lc)
    ax4.set_xlim(t[0], t[-1])
    y_lo = min(lam_sm.min() * 1.3, -0.05)
    y_hi = lam_sm.max() * 1.3
    ax4.set_ylim(y_lo, y_hi)
    ax4.axhline(0, color='#ffffff55', lw=0.8)
    ax4.axhline(LAMBDA_THRESH, color='#ffcc00', lw=0.9, ls='--',
                alpha=0.8, label=f'Limiar lambda={LAMBDA_THRESH}')
    shade_chaos(ax4, t, m['is_chaotic'])
    plt.colorbar(lc, ax=ax4, fraction=0.01, pad=0.01).ax.tick_params(colors='#c2cdd8')
    ax4.set_xlabel('Tempo (s)'); ax4.set_ylabel('lambda_local suavizado')
    ax4.set_title('Timeline de Regime — Caotico (vermelho) vs. Linear (azul/verde)')
    ax4.legend(fontsize=8, framealpha=0.2); ax4.grid(True)

    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor='#080c10')
    plt.close()
    print(f"  Salvo: {path}")


# =================================================================
#  SCATTER CAOS — todos sobrepostos
# =================================================================
def plot_scatter_all(metrics, path):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#080c10')
    fig.suptitle(
        'Scatter Caos — Todos os Episodios Sobrepostos\n'
        'Cor = episodio  |  Linha inferior = correlacao rolling',
        fontsize=12, color='#e4edf6', y=0.99, fontweight='bold')

    gs = gridspec.GridSpec(2, 3, hspace=0.44, wspace=0.32,
                           top=0.93, bottom=0.08, left=0.06, right=0.97)

    cfg = [
        ('speed',     '||omega|| (rad/s)',       'Velocidade x lambda', 'r_arc'),
        ('accel_mag', '||alpha|| (rad/s2) local','Aceleracao x lambda', 'r_accel'),
        ('Ek',        'Ek = 0.5*(w1^2+w2^2)',   'Energia x lambda',    'r_Ek'),
    ]

    for col, (key, xlabel, title, rkey) in enumerate(cfg):
        ax_sc = fig.add_subplot(gs[0, col])
        ax_r  = fig.add_subplot(gs[1, col])

        for ep in episodes:
            m   = metrics[ep]; n = len(m['t'])
            idx = np.random.choice(n, min(N_SCATTER // len(episodes), n), replace=False)
            xd  = m[key][idx]; la = m['lam'][idx]
            gd  = ~np.isnan(la)
            ax_sc.scatter(xd[gd], la[gd],
                          color=EP_COLORS[ep % len(EP_COLORS)],
                          s=2, alpha=0.38, linewidths=0,
                          label=f'Ep {ep}')
            if len(m['t_corr']) > 0:
                ax_r.plot(m['t_corr'], m[rkey],
                          color=EP_COLORS[ep % len(EP_COLORS)],
                          lw=1.0, alpha=0.75, label=f'Ep {ep}')

        ax_sc.axhline(0, color='#ffffff33', lw=0.8)
        ax_sc.axhline(LAMBDA_THRESH, color='#ffcc00', lw=0.9, ls='--', alpha=0.7)
        ax_sc.set_xlabel(xlabel); ax_sc.set_ylabel('lambda_local (nats/passo)')
        ax_sc.set_title(title); ax_sc.grid(True)
        if col == 0:
            ax_sc.legend(fontsize=7.5, framealpha=0.2, markerscale=3)

        ax_r.axhline(0, color='#ffffff44', lw=0.8)
        ax_r.set_ylim(-1, 1)
        ax_r.set_xlabel('Tempo (s)'); ax_r.set_ylabel('Pearson r (rolling)')
        ax_r.set_title(f'r rolling todos episodios (W={W_CORR*DT:.1f}s)')
        ax_r.grid(True)
        if col == 0:
            ax_r.legend(fontsize=7.5, framealpha=0.2)

    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor='#080c10')
    plt.close()
    print(f"  Salvo: {path}")


# =================================================================
#  CHAOS TOPOLOGY — todos sobrepostos
# =================================================================
def plot_topology_all(metrics, path):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#080c10')
    fig.suptitle(
        'Chaos Topology — Todos os Episodios Sobrepostos\n'
        'Espaco de fases, hexbin lambda, ||omega||xlambda e timeline comparativa',
        fontsize=12, color='#e4edf6', y=0.99, fontweight='bold')

    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.32,
                           top=0.93, bottom=0.08, left=0.06, right=0.97)

    # Painel 1: espaco de fases sobrepostos
    ax1 = fig.add_subplot(gs[0, 0])
    for ep in episodes:
        m = metrics[ep]; n = len(m['o1']); s = max(1, n // 2000)
        ax1.scatter(m['o1'][::s], m['o2'][::s],
                    color=EP_COLORS[ep % len(EP_COLORS)],
                    s=0.8, alpha=0.40, linewidths=0, label=f'Ep {ep}')
    ax1.axhline(0, color='#1a2a38', lw=0.5); ax1.axvline(0, color='#1a2a38', lw=0.5)
    ax1.set_xlabel('omega1 (rad/s)'); ax1.set_ylabel('omega2 (rad/s)')
    ax1.set_title('Espaco de fases — todos os episodios')
    ax1.legend(fontsize=7.5, framealpha=0.2, markerscale=4); ax1.grid(True)

    # Painel 2: hexbin lambda medio global
    ax2 = fig.add_subplot(gs[0, 1])
    all_o1  = np.concatenate([metrics[ep]['o1']  for ep in episodes])
    all_o2  = np.concatenate([metrics[ep]['o2']  for ep in episodes])
    all_lam = np.concatenate([metrics[ep]['lam'] for ep in episodes])
    hb = ax2.hexbin(all_o1, all_o2, C=all_lam, gridsize=40,
                    cmap=CHAOS_CMAP, norm=NORM,
                    reduce_C_function=np.nanmean, linewidths=0.1)
    cb2 = plt.colorbar(hb, ax=ax2, fraction=0.04, pad=0.02)
    cb2.set_label('lambda medio/celula', fontsize=7, color='#c2cdd8')
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color='#c2cdd8', fontsize=7)
    ax2.set_xlabel('omega1 (rad/s)'); ax2.set_ylabel('omega2 (rad/s)')
    ax2.set_title('Hexbin lambda medio — todos os episodios'); ax2.grid(True)

    # Painel 3: ||omega|| x lambda sobrepostos
    ax3 = fig.add_subplot(gs[0, 2])
    for ep in episodes:
        m   = metrics[ep]; n = len(m['o1'])
        idx = np.random.choice(n, min(N_SCATTER // len(episodes), n), replace=False)
        sp  = m['speed'][idx]; la = m['lam'][idx]; gd = ~np.isnan(la)
        ax3.scatter(sp[gd], la[gd],
                    color=EP_COLORS[ep % len(EP_COLORS)],
                    s=2, alpha=0.40, linewidths=0, label=f'Ep {ep}')
    ax3.axhline(0, color='#ffffff33', lw=0.8)
    ax3.axhline(LAMBDA_THRESH, color='#ffcc00', lw=0.9, ls='--', alpha=0.7)
    ax3.set_xlabel('||omega|| (rad/s)'); ax3.set_ylabel('lambda_local (nats/passo)')
    ax3.set_title('||omega|| x lambda — todos os episodios')
    ax3.legend(fontsize=7.5, framealpha=0.2, markerscale=3); ax3.grid(True)

    # Painel 4 (linha inteira): timeline lambda todos episodios
    ax4 = fig.add_subplot(gs[1, :])
    for ep in episodes:
        m   = metrics[ep]; t = m['t']; ls = m['lam_smooth']
        col = EP_COLORS[ep % len(EP_COLORS)]
        ax4.plot(t, ls, color=col, lw=1.0, alpha=0.75, label=f'Ep {ep}')
        ax4.fill_between(t, 0, ls, where=m['is_chaotic'], color=col, alpha=0.08)
    ax4.axhline(0, color='#ffffff44', lw=0.8)
    ax4.axhline(LAMBDA_THRESH, color='#ffcc00', lw=0.9, ls='--',
                label=f'Limiar lambda={LAMBDA_THRESH}')
    ax4.set_xlabel('Tempo (s)'); ax4.set_ylabel('lambda_local suavizado')
    ax4.set_title('Timeline de Regime — todos os episodios sobrepostos')
    ax4.legend(fontsize=8, framealpha=0.2, ncol=6); ax4.grid(True)

    # Boxplot inset
    ax_b = fig.add_axes([0.375, 0.10, 0.12, 0.27])
    ax_b.set_facecolor('#080c10')
    bp_data = [metrics[ep]['lam'][~np.isnan(metrics[ep]['lam'])] for ep in episodes]
    bp = ax_b.boxplot(bp_data, patch_artist=True, widths=0.55,
                      medianprops=dict(color='#ffcc00', lw=1.5),
                      whiskerprops=dict(color='#4a5a6a'),
                      capprops=dict(color='#4a5a6a'),
                      flierprops=dict(marker='.', ms=1, alpha=0.3, color='#4a5a6a'))
    for patch, ep in zip(bp['boxes'], episodes):
        patch.set_facecolor(EP_COLORS[ep % len(EP_COLORS)]); patch.set_alpha(0.6)
    ax_b.set_xticks(range(1, len(episodes)+1))
    ax_b.set_xticklabels([f'E{ep}' for ep in episodes], fontsize=7, color='#4a5a6a')
    ax_b.set_ylabel('lambda_local', fontsize=7, color='#c2cdd8')
    ax_b.set_title('Dist. lambda', fontsize=7, color='#e4edf6')
    ax_b.tick_params(colors='#4a5a6a', labelsize=7)
    ax_b.axhline(LAMBDA_THRESH, color='#ffcc00', lw=0.8, ls='--', alpha=0.7)
    ax_b.grid(True, axis='y', alpha=0.4)

    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor='#080c10')
    plt.close()
    print(f"  Salvo: {path}")


# =================================================================
#  GERAR OS 12 GRAFICOS
# =================================================================
print("-" * 55)
print("SCATTER CAOS — individuais (5 figuras)")
print("-" * 55)
for ep in episodes:
    plot_scatter_episode(ep, metrics[ep],
                         os.path.join(OUTPUT_DIR, f"scatter_chaos_ep{ep}.png"))

print()
print("-" * 55)
print("SCATTER CAOS — todos sobrepostos (1 figura)")
print("-" * 55)
plot_scatter_all(metrics, os.path.join(OUTPUT_DIR, "scatter_chaos_all.png"))

print()
print("-" * 55)
print("CHAOS TOPOLOGY — individuais (5 figuras)")
print("-" * 55)
for ep in episodes:
    plot_topology_episode(ep, metrics[ep],
                          os.path.join(OUTPUT_DIR, f"chaos_topology_ep{ep}.png"))

print()
print("-" * 55)
print("CHAOS TOPOLOGY — todos sobrepostos (1 figura)")
print("-" * 55)
plot_topology_all(metrics, os.path.join(OUTPUT_DIR, "chaos_topology_all.png"))

print()
print("=" * 55)
print(f"12 figuras salvas em ./{OUTPUT_DIR}/")
print("-" * 55)
for fname in sorted(os.listdir(OUTPUT_DIR)):
    sz = os.path.getsize(os.path.join(OUTPUT_DIR, fname)) // 1024
    print(f"  {fname:42s}  {sz:4d} KB")
print("=" * 55)
