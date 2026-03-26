#!/usr/bin/env python3
"""
Final Visualization: Compare all 3 equation recovery methods.
=============================================================
Method 1: SINDy on True VF (direct from simulation data)
Method 2: SINDy on Neural ODE VF (indirect, via learned vector field)  
Method 3: PySR on Neural ODE VF (symbolic regression on learned VF)

Produces:
  - Figure 1: Scatter plots of predicted vs true derivatives (6 panels)
  - Figure 2: Forward simulation of recovered equations vs true ODE (trajectory comparison)
"""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(__file__))
from sir_pipeline import NeuralODE

SAVE_DIR = 'results'

# ──────────────────────────────────────────────────────────────────
# RECOVERED EQUATIONS (hardcoded from our runs)
# ──────────────────────────────────────────────────────────────────

# SINDy on True VF (R²=1.0, perfect)
def sindy_true_dS(S, I, b, g):  return -1.0 * b * S * I
def sindy_true_dI(S, I, b, g):  return +1.0 * b * S * I - 1.0 * g * I

# SINDy on Neural ODE VF (best sparse result at thr=0.2)
def sindy_node_dS(S, I, b, g):  return -0.8990 * b * S * I
def sindy_node_dI(S, I, b, g):  return +0.8892 * b * S * I - 0.9671 * g * I

# PySR on Neural ODE VF
def pysr_node_dS(S, I, b, g):   return -0.49869 * S * I
def pysr_node_dI(S, I, b, g):   return (b * S - g) * I

# True SIR equations (ground truth)
def true_dS(S, I, b, g):        return -b * S * I
def true_dI(S, I, b, g):        return  b * S * I - g * I


# ──────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────

def load_data():
    """Load dataset and Neural ODE predictions."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = np.load(os.path.join(SAVE_DIR, 'sir_dataset.npz'))
    params, t_grid, mean_traj = data['params'], data['t_grid'], data['mean_traj']

    model = NeuralODE(hidden=128).to(device)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pt'), map_location=device))
    model.eval()

    # Query Neural ODE VF
    all_S, all_I, all_b, all_g, all_dS_node, all_dI_node = [], [], [], [], [], []
    for idx in range(len(params)):
        b_val, g_val = params[idx]
        traj = mean_traj[idx]
        y_torch = torch.tensor(traj, dtype=torch.float32).to(device)
        model.vf.set_params(b_val, g_val)
        with torch.no_grad():
            dy = model.vf(0.0, y_torch).cpu().numpy()
        all_S.append(traj[:, 0]); all_I.append(traj[:, 1])
        all_b.append(np.full(len(traj), b_val))
        all_g.append(np.full(len(traj), g_val))
        all_dS_node.append(dy[:, 0]); all_dI_node.append(dy[:, 1])

    S = np.concatenate(all_S)
    I = np.concatenate(all_I)
    b = np.concatenate(all_b)
    g = np.concatenate(all_g)
    dS_node = np.concatenate(all_dS_node)
    dI_node = np.concatenate(all_dI_node)

    # True derivatives
    dS_true = -b * S * I
    dI_true =  b * S * I - g * I

    # Filter out equilibrium points
    mag = np.sqrt(dS_true**2 + dI_true**2)
    mask = mag > 1e-4

    return (S[mask], I[mask], b[mask], g[mask],
            dS_true[mask], dI_true[mask],
            dS_node[mask], dI_node[mask],
            params, t_grid, mean_traj, model, device)


# ──────────────────────────────────────────────────────────────────
# FIGURE 1: Scatter plots — Predicted vs True derivatives
# ──────────────────────────────────────────────────────────────────

def fig1_scatter_comparison(S, I, b, g, dS_true, dI_true, dS_node, dI_node):
    """6-panel scatter: rows = dS/dt, dI/dt; cols = 3 methods."""
    
    methods = [
        ("SINDy (True Data)", sindy_true_dS, sindy_true_dI, '#2196F3'),
        ("SINDy (Neural ODE)", sindy_node_dS, sindy_node_dI, '#FF9800'),
        ("PySR (Neural ODE)", pysr_node_dS, pysr_node_dI, '#4CAF50'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, (name, fn_dS, fn_dI, color) in enumerate(methods):
        pred_dS = fn_dS(S, I, b, g)
        pred_dI = fn_dI(S, I, b, g)

        for row, (tname, dt, dp) in enumerate([
            ('dS/dt', dS_true, pred_dS),
            ('dI/dt', dI_true, pred_dI),
        ]):
            ax = axes[row, col]
            ax.scatter(dt, dp, alpha=0.04, s=3, c=color, rasterized=True)

            lims = [min(dt.min(), dp.min()) * 1.05, max(dt.max(), dp.max()) * 1.05]
            ax.plot(lims, lims, 'k--', lw=1.5, alpha=0.6, label='Perfect (y=x)')

            ss_res = np.sum((dt - dp)**2)
            ss_tot = np.sum((dt - dt.mean())**2)
            r2 = 1 - ss_res / ss_tot
            rmse = np.sqrt(np.mean((dt - dp)**2))

            ax.set_title(f'{tname} — {name}\nR² = {r2:.4f} | RMSE = {rmse:.5f}', fontsize=11)
            ax.set_xlabel('True derivative', fontsize=10)
            ax.set_ylabel('Predicted derivative', fontsize=10)
            ax.legend(fontsize=9, loc='upper left')
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)

    plt.suptitle('Equation Recovery: Predicted vs True Derivatives\n'
                 'Ground Truth: dS/dt = −β·S·I,  dI/dt = β·S·I − γ·I',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig1_derivative_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────
# FIGURE 2: Forward simulation of recovered equations
# ──────────────────────────────────────────────────────────────────

def simulate_sir(dS_fn, dI_fn, beta, gamma, S0, I0, t_span, t_eval):
    """Simulate an SIR model using the given dS, dI functions."""
    def rhs(t, y):
        S, I, R = y
        ds = dS_fn(S, I, beta, gamma)
        di = dI_fn(S, I, beta, gamma)
        dr = -(ds + di)  # conservation: dR = -dS - dI
        return [ds, di, dr]

    sol = solve_ivp(rhs, t_span, [S0, I0, 1-S0-I0],
                    t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
    return sol.y  # (3, n_time)


def fig2_trajectory_comparison(params, t_grid, mean_traj, model, device):
    """Simulate recovered equations and compare trajectories."""

    # Pick 6 diverse parameter sets
    n_show = 6
    indices = np.linspace(0, len(params) - 1, n_show, dtype=int)
    t_span = [t_grid[0], t_grid[-1]]

    methods = [
        ("True ODE",           true_dS,       true_dI,       'black',   '-',  2.5),
        ("SINDy (True Data)",  sindy_true_dS, sindy_true_dI, '#2196F3', '--', 2.0),
        ("SINDy (Neural ODE)", sindy_node_dS, sindy_node_dI, '#FF9800', '--', 2.0),
        # ("PySR (Neural ODE)",  pysr_node_dS,  pysr_node_dI,  '#4CAF50', ':',  2.5),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()

    for ax_idx, data_idx in enumerate(indices):
        ax = axes[ax_idx]
        b_val, g_val = params[data_idx]
        traj = mean_traj[data_idx]  # (n_time, 3)
        S0, I0 = traj[0, 0], traj[0, 1]

        # Plot ensemble mean (stochastic data) as background
        ax.plot(t_grid, traj[:, 0], 'gray', lw=1, alpha=0.5)
        ax.plot(t_grid, traj[:, 1], 'gray', lw=1, alpha=0.5)
        ax.plot(t_grid, traj[:, 2], 'gray', lw=1, alpha=0.5)
        ax.fill_between([], [], color='gray', alpha=0.3, label='Stochastic data')

        for name, fn_dS, fn_dI, color, ls, lw in methods:
            try:
                sol = simulate_sir(fn_dS, fn_dI, b_val, g_val, S0, I0, t_span, t_grid)
                ax.plot(t_grid, sol[0], color=color, ls=ls, lw=lw, alpha=0.9)  # S
                ax.plot(t_grid, sol[1], color=color, ls=ls, lw=lw, alpha=0.9)  # I
                # Only label once per method
                ax.plot([], [], color=color, ls=ls, lw=lw, label=name)
            except Exception as e:
                ax.plot([], [], color=color, ls=ls, lw=lw, label=f'{name} (failed)')

        ax.set_title(f'β={b_val:.2f}, γ={g_val:.2f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Fraction')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc='right', framealpha=0.9)

    plt.suptitle('Forward Simulation: Recovered Equations vs True ODE\n'
                 'S(t) and I(t)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig2_trajectory_simulation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────
# FIGURE 3: Neural ODE vector field quality check
# ──────────────────────────────────────────────────────────────────

def fig3_neural_ode_quality(S, I, b, g, dS_true, dI_true, dS_node, dI_node):
    """2-panel scatter: Neural ODE derivatives vs True derivatives."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, tname, dt, dp in [
        (axes[0], 'dS/dt', dS_true, dS_node),
        (axes[1], 'dI/dt', dI_true, dI_node),
    ]:
        ax.scatter(dt, dp, alpha=0.03, s=3, c='steelblue', rasterized=True)
        lims = [min(dt.min(), dp.min()) * 1.05, max(dt.max(), dp.max()) * 1.05]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect (y=x)')

        r2 = 1 - np.sum((dt - dp)**2) / np.sum((dt - dt.mean())**2)
        rmse = np.sqrt(np.mean((dt - dp)**2))

        ax.set_title(f'{tname}: Neural ODE vs True\nR² = {r2:.4f} | RMSE = {rmse:.5f}', fontsize=12)
        ax.set_xlabel('True derivative', fontsize=11)
        ax.set_ylabel('Neural ODE derivative', fontsize=11)
        ax.legend(fontsize=10)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    plt.suptitle('Neural ODE Vector Field Quality\n'
                 'How well did the Neural ODE learn the true SIR dynamics?',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'fig3_neural_ode_quality.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading data and Neural ODE...")
    (S, I, b, g, dS_true, dI_true, dS_node, dI_node,
     params, t_grid, mean_traj, model, device) = load_data()
    print(f"  {len(S)} active data points")

    print("\nGenerating Figure 1: Derivative scatter comparison...")
    fig1_scatter_comparison(S, I, b, g, dS_true, dI_true, dS_node, dI_node)

    print("Generating Figure 2: Forward trajectory simulation...")
    fig2_trajectory_comparison(params, t_grid, mean_traj, model, device)

    print("Generating Figure 3: Neural ODE quality check...")
    fig3_neural_ode_quality(S, I, b, g, dS_true, dI_true, dS_node, dI_node)

    # ── Print final equations ──
    print("\n" + "="*60)
    print("RECOVERED EQUATIONS SUMMARY")
    print("="*60)
    print("\nGround Truth:")
    print("  dS/dt = -β·S·I")
    print("  dI/dt =  β·S·I - γ·I")
    print("\n1. SINDy (True Data → STLSQ):")
    print("  dS/dt = -1.0000·β·S·I")
    print("  dI/dt = +1.0000·β·S·I - 1.0000·γ·I")
    print("  → PERFECT RECOVERY ✅")
    print("\n2. SINDy (Neural ODE → STLSQ, threshold=0.2):")
    print("  dS/dt = -0.8990·β·S·I")
    print("  dI/dt = +0.8892·β·S·I - 0.9671·γ·I")
    print("  → Correct structure, ~10% coefficient error ⚠️")
    print("\n3. PySR (Neural ODE → Symbolic Regression):")
    print("  dS/dt = -0.4987·S·I")
    print("  dI/dt = (β·S - γ)·I  =  β·S·I - γ·I")
    print("  → dI/dt: EXACT STRUCTURE ✅  |  dS/dt: structure ok, β absorbed ⚠️")

    print("\n✅ All visualizations saved to results/")
