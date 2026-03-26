#!/usr/bin/env python3
"""
From Noise to Law — Recovering the SIR Equations from Stochastic Data
======================================================================
FINAL VERSION — All fixes integrated, drop-in ready.

Step 1: Gillespie stochastic SIR simulator + dataset generation
Step 2: Neural ODE training (ReduceLROnPlateau + early stopping)

"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Gillespie Stochastic SIR Simulator + Dataset Generation
# ═════════════════════════════════════════════════════════════════════════════

def gillespie_sir(beta, gamma, N, I0, t_max, seed=None):
    """
    Exact stochastic simulation of SIR using the Gillespie algorithm.
    Returns event-time arrays (t, S, I, R) — NOT on a uniform grid.
    """
    rng = np.random.default_rng(seed)
    S, I, R = N - I0, I0, 0
    t = 0.0
    t_traj  = [t]
    S_traj  = [S]
    I_traj  = [I]
    R_traj  = [R]

    while t < t_max and I > 0:
        r_infect = beta * S * I / N
        r_recover = gamma * I
        r_total   = r_infect + r_recover
        if r_total == 0:
            break
        dt = rng.exponential(1.0 / r_total)
        t += dt
        if t > t_max:
            break
        if rng.random() < r_infect / r_total:
            S -= 1;  I += 1
        else:
            I -= 1;  R += 1
        t_traj.append(t)
        S_traj.append(S)
        I_traj.append(I)
        R_traj.append(R)

    return (np.array(t_traj), np.array(S_traj),
            np.array(I_traj), np.array(R_traj))


def interpolate_trajectory(t_raw, X_raw, t_grid):
    """Resample step-function onto uniform grid via forward-fill."""
    idx = np.searchsorted(t_raw, t_grid, side='right') - 1
    idx = np.clip(idx, 0, len(X_raw) - 1)
    return X_raw[idx]


def deterministic_sir(beta, gamma, N, I0, t_grid):
    """Solve deterministic SIR ODE for reference comparison."""
    def rhs(t, y):
        S, I, R = y
        return [-beta*S*I/N,  beta*S*I/N - gamma*I,  gamma*I]
    sol = solve_ivp(
        rhs, [t_grid[0], t_grid[-1]], [N - I0, float(I0), 0.0],
        t_eval=t_grid, method='RK45', rtol=1e-8, atol=1e-10
    )
    return sol.y / N   # (3, n_time) normalised


def generate_dataset(
    beta_range=(0.3, 0.8),
    gamma_range=(0.05, 0.25),
    n_beta=8,
    n_gamma=8,
    N=1000,
    I0=10,
    t_max=100,
    n_time=200,
    n_runs=200,
    seed=42,
    save_path='sir_dataset.npz',
):
    """
    Simulate n_runs Gillespie trajectories at each (β,γ) grid point.
    Saves:
        params     : (n_params, 2)           — [beta, gamma]
        t_grid     : (n_time,)
        mean_traj  : (n_params, n_time, 3)   — mean [S/N, I/N, R/N]
    """
    rng    = np.random.default_rng(seed)
    betas  = np.linspace(*beta_range,  n_beta)
    gammas = np.linspace(*gamma_range, n_gamma)
    t_grid = np.linspace(0, t_max, n_time)

    param_list     = []
    mean_traj_list = []

    for beta in betas:
        for gamma in gammas:
            S_runs = np.zeros((n_runs, n_time))
            I_runs = np.zeros((n_runs, n_time))
            R_runs = np.zeros((n_runs, n_time))

            for k in range(n_runs):
                t_r, S_r, I_r, R_r = gillespie_sir(
                    beta, gamma, N, I0, t_max,
                    seed=int(rng.integers(0, 10_000_000))
                )
                S_runs[k] = interpolate_trajectory(t_r, S_r, t_grid)
                I_runs[k] = interpolate_trajectory(t_r, I_r, t_grid)
                R_runs[k] = interpolate_trajectory(t_r, R_r, t_grid)

            mean = np.stack([
                S_runs.mean(axis=0) / N,
                I_runs.mean(axis=0) / N,
                R_runs.mean(axis=0) / N,
            ], axis=-1)

            param_list.append([beta, gamma])
            mean_traj_list.append(mean)
            print(f"  β={beta:.3f}  γ={gamma:.3f}  peak I={mean[:,1].max():.3f}")

    params    = np.array(param_list)
    mean_traj = np.array(mean_traj_list)

    np.savez(save_path, params=params, t_grid=t_grid, mean_traj=mean_traj)
    print(f"\nDataset saved → {save_path}")
    print(f"  {len(params)} parameter points  |  {n_time} time steps  |  {n_runs} runs/point")
    return params, t_grid, mean_traj


def plot_stochastic_vs_deterministic(save_dir='.'):
    """Visualise individual runs vs ensemble mean vs deterministic ODE."""
    beta, gamma, N, I0 = 0.5, 0.15, 1000, 10
    t_max, n_runs = 100, 50
    t_grid = np.linspace(0, t_max, 300)

    all_S = np.zeros((n_runs, len(t_grid)))
    all_I = np.zeros((n_runs, len(t_grid)))
    all_R = np.zeros((n_runs, len(t_grid)))

    for i in range(n_runs):
        t_r, S_r, I_r, R_r = gillespie_sir(beta, gamma, N, I0, t_max, seed=i)
        all_S[i] = interpolate_trajectory(t_r, S_r, t_grid) / N
        all_I[i] = interpolate_trajectory(t_r, I_r, t_grid) / N
        all_R[i] = interpolate_trajectory(t_r, R_r, t_grid) / N

    det = deterministic_sir(beta, gamma, N, I0, t_grid)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    colors = ['steelblue', 'tomato', 'seagreen']
    labels = ['S(t)', 'I(t)', 'R(t)']

    for ax, D, det_c, c, lbl in zip(axes, [all_S, all_I, all_R], det, colors, labels):
        for run in D:
            ax.plot(t_grid, run, color=c, alpha=0.1, lw=0.5)
        ax.plot(t_grid, D.mean(axis=0), color=c, lw=2.5, label='Ensemble Mean')
        ax.plot(t_grid, det_c, 'k--', lw=2, label='ODE (true)')
        ax.set_title(lbl, fontsize=13)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Fraction of population')
        ax.legend()

    plt.suptitle(
        f'Gillespie SIR  β={beta}, γ={gamma}, N={N}, I₀={I0}  ({n_runs} runs)',
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    path = os.path.join(save_dir, 'step1_stochastic_vs_deterministic.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Neural ODE  (parameter-conditioned)
# ═════════════════════════════════════════════════════════════════════════════

class SIRVectorField(nn.Module):
    """
    Learns the SIR vector field.
    Input : [S, I, β, γ]   (4-dim) — R excluded (R = 1-S-I, redundant)
    Output: [dS/dt, dI/dt, dR/dt]  (3-dim)

    Conservation law dS+dI+dR = 0 enforced by subtracting output row-mean.
    Architecture: 4 → 128 → 128 → 128 → 3  with Tanh activations.
    """
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
        )
        self.register_buffer('_beta',  torch.tensor(0.0))
        self.register_buffer('_gamma', torch.tensor(0.0))

    def set_params(self, beta, gamma):
        device = next(self.parameters()).device
        self._beta  = (beta.to(device)  if isinstance(beta,  torch.Tensor)
                       else torch.tensor(beta,  dtype=torch.float32, device=device))
        self._gamma = (gamma.to(device) if isinstance(gamma, torch.Tensor)
                       else torch.tensor(gamma, dtype=torch.float32, device=device))

    def forward(self, t, y):
        squeezed = (y.dim() == 1)
        if squeezed:
            y = y.unsqueeze(0)
        batch = y.shape[0]

        SI    = y[:, :2]                              # (batch, 2) — drop R column
        beta  = self._beta.expand(batch, 1)
        gamma = self._gamma.expand(batch, 1)
        inp   = torch.cat([SI, beta, gamma], dim=-1)  # (batch, 4)
        dy    = self.net(inp)                         # (batch, 3)

        # Enforce conservation: dS + dI + dR must sum to 0
        dy = dy - dy.mean(dim=-1, keepdim=True)

        return dy.squeeze(0) if squeezed else dy


class NeuralODE(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.vf = SIRVectorField(hidden)

    def forward(self, y0, t, beta, gamma):
        # Project initial condition onto probability simplex (S+I+R=1, all ≥ 0)
        # Prevents integration drift when noisy means have tiny rounding errors
        y0 = y0.clamp(min=0.0)
        y0 = y0 / y0.sum()
        self.vf.set_params(beta, gamma)
        return odeint(self.vf, y0, t, method='dopri5', rtol=1e-5, atol=1e-6)


def train_neural_ode(
    data_path='sir_dataset.npz',
    n_epochs=2000,
    lr=1e-3,
    batch_size=8,
    device=None,
    save_dir='.',
    # ── Early stopping parameters ──────────────────────────────────────────
    es_patience=200,      # epochs to wait after last improvement
    es_min_delta=1e-5,    # minimum improvement to count as progress
    es_target_loss=5e-4,  # early stopping WILL NOT fire above this loss
    # ── Scheduler parameters ──────────────────────────────────────────────
    sched_patience=30,    # epochs before LR halved
    sched_factor=0.5,     # LR multiplier on plateau
    sched_min_lr=1e-5,    # LR floor
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training device: {device}")
    print(f"Early stopping: patience={es_patience}, min_delta={es_min_delta:.0e}, "
          f"won't fire unless loss < {es_target_loss:.0e}")

    # ── Load data ────────────────────────────────────────────────────────────
    data         = np.load(data_path)
    params       = data['params']
    t_grid       = data['t_grid']
    mean_traj    = data['mean_traj']
    n_params     = len(params)

    t_torch      = torch.tensor(t_grid,    dtype=torch.float32).to(device)
    traj_torch   = torch.tensor(mean_traj, dtype=torch.float32).to(device)
    params_torch = torch.tensor(params,    dtype=torch.float32).to(device)

    # ── Model + Optimizer ───────────────────────────────────────────────────
    model     = NeuralODE(hidden=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ReduceLROnPlateau: only ever LOWERS the LR, never resets it upward.
    # This eliminates the violent loss spikes caused by CosineAnnealingWarmRestarts.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=sched_factor,
        patience=sched_patience,
        min_lr=sched_min_lr
    )

    # I-channel upweighting: I peaks at ~0.1–0.5 while S,R span [0,1]
    # Without this, flat MSE under-penalises I errors
    channel_weights = torch.tensor([1.0, 3.0, 1.0], device=device)

    # ── Early stopping state ────────────────────────────────────────────────
    best_loss      = float('inf')
    es_counter     = 0
    best_model_path = os.path.join(save_dir, 'best_model.pt')

    idx_all   = np.arange(n_params)
    loss_hist = []

    print(f"\nTraining: {n_params} trajectories | {n_epochs} max epochs | "
          f"batch={batch_size} | I-weight=3×")
    print("-" * 70)

    prev_lr = lr
    for epoch in range(1, n_epochs + 1):
        model.train()
        np.random.shuffle(idx_all)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_params, batch_size):
            batch_idx  = idx_all[start : start + batch_size]
            total_loss = torch.tensor(0.0, device=device)

            for b_idx in batch_idx:
                y0        = traj_torch[b_idx, 0, :]   # (3,)
                beta_val  = params_torch[b_idx, 0]
                gamma_val = params_torch[b_idx, 1]
                target    = traj_torch[b_idx]          # (n_time, 3)

                pred = model(y0, t_torch, beta_val, gamma_val)   # (n_time, 3)

                err        = (pred - target).pow(2)              # (n_time, 3)
                loss       = (err * channel_weights).mean()
                total_loss = total_loss + loss

            total_loss = total_loss / len(batch_idx)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        loss_hist.append(avg_loss)

        # ── Scheduler step (pass loss so plateau can be detected) ───────────
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < prev_lr:
            print(f"  → LR reduced: {prev_lr:.2e} → {current_lr:.2e} at epoch {epoch}")
            prev_lr = current_lr

        # ── Early stopping logic ─────────────────────────────────────────────
        # Rule: only consider stopping if loss is already below es_target_loss.
        # This guarantees we run the full 2000 epochs if the model hasn't
        # converged to a meaningful level yet.
        improved = avg_loss < (best_loss - es_min_delta)

        if improved:
            best_loss = avg_loss
            es_counter = 0
            torch.save(model.state_dict(), best_model_path)   # save best checkpoint
        else:
            es_counter += 1

        # Fire early stopping ONLY when both conditions are true:
        #   1. Loss is below target (we have a good enough model)
        #   2. No improvement for es_patience consecutive epochs
        if (avg_loss < es_target_loss) and (es_counter >= es_patience):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n✅ Early stopping at epoch {epoch} — "
                  f"loss={avg_loss:.2e} < {es_target_loss:.0e}, "
                  f"no improvement for {es_patience} epochs, lr={current_lr:.2e}")
            break
        # ─────────────────────────────────────────────────────────────────────

        if epoch % 50 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            flag = " ✓" if improved else f" (no improv {es_counter}/{es_patience})"
            print(f"Epoch {epoch:5d}/{n_epochs}  loss={avg_loss:.6f}  "
                  f"best={best_loss:.6f}  lr={current_lr:.2e}{flag}")

    # ── Load best checkpoint ─────────────────────────────────────────────────
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"\nLoaded best model (loss={best_loss:.6f}) from {best_model_path}")

    # ── Save final model ─────────────────────────────────────────────────────
    final_model_path = os.path.join(save_dir, 'neural_ode_sir.pt')
    torch.save(model.state_dict(), final_model_path)
    np.save(os.path.join(save_dir, 'loss_history.npy'), np.array(loss_hist))

    # ── Loss curve ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(loss_hist, color='steelblue', lw=1.2, label='Train loss')
    ax.axhline(es_target_loss, color='tomato', lw=1.5, ls='--',
               label=f'Early stop target ({es_target_loss:.0e})')
    ax.axhline(best_loss, color='seagreen', lw=1.5, ls=':',
               label=f'Best loss ({best_loss:.2e})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Weighted MSE (log scale)')
    ax.set_title('Neural ODE Training — ReduceLROnPlateau + Early Stopping')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'step2_loss_curve.png'), dpi=150)
    plt.close()

    print(f"Final model  → {final_model_path}")
    print(f"Best model   → {best_model_path}")
    print(f"Best loss    : {best_loss:.6f}")

    plot_trajectory_comparison(model, params_torch, traj_torch, t_torch, device, save_dir)
    plot_phase_portraits(model, params_torch, traj_torch, t_torch, device, save_dir)

    return model, t_torch, traj_torch, params_torch


def plot_trajectory_comparison(model, params_torch, traj_torch, t_torch, device, save_dir='.'):
    model.eval()
    n_show  = min(6, len(params_torch))
    indices = np.linspace(0, len(params_torch) - 1, n_show, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            y0        = traj_torch[idx, 0, :]
            beta_val  = params_torch[idx, 0]
            gamma_val = params_torch[idx, 1]
            target    = traj_torch[idx].cpu().numpy()
            pred      = model(y0, t_torch, beta_val, gamma_val).cpu().numpy()
            t_np      = t_torch.cpu().numpy()

            for j, (c, lbl) in enumerate(
                zip(['steelblue', 'tomato', 'seagreen'], ['S', 'I', 'R'])
            ):
                ax.plot(t_np, target[:, j], color=c, lw=2,         label=f'{lbl} true')
                ax.plot(t_np, pred[:, j],   color=c, lw=2, ls='--', label=f'{lbl} pred')

            ax.set_title(f"β={beta_val:.2f}, γ={gamma_val:.2f}", fontsize=11)
            ax.set_xlabel('Time'); ax.set_ylabel('Fraction')
            ax.legend(fontsize=7, ncol=2)

    plt.suptitle('Neural ODE: Predicted vs True Mean Dynamics', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, 'step2_trajectory_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Trajectory comparison → {path}")


def plot_phase_portraits(model, params_torch, traj_torch, t_torch, device, save_dir='.'):
    model.eval()
    n_show  = min(6, len(params_torch))
    indices = np.linspace(0, len(params_torch) - 1, n_show, dtype=int)
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap    = plt.cm.viridis
    colors  = [cmap(i / max(n_show - 1, 1)) for i in range(n_show)]

    with torch.no_grad():
        for i, idx in enumerate(indices):
            y0        = traj_torch[idx, 0, :]
            beta_val  = params_torch[idx, 0]
            gamma_val = params_torch[idx, 1]
            target    = traj_torch[idx].cpu().numpy()
            pred      = model(y0, t_torch, beta_val, gamma_val).cpu().numpy()

            ax.plot(target[:, 0], target[:, 1], color=colors[i], lw=2,
                    label=f'True β={beta_val:.2f}')
            ax.plot(pred[:, 0],   pred[:, 1],   color=colors[i], lw=2, ls='--')

    ax.set_xlabel('S'); ax.set_ylabel('I')
    ax.set_title('Phase Portrait: S vs I  (solid=true, dashed=predicted)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'step2_phase_portraits.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Phase portraits → {path}")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SIR Recovery Pipeline — Final")
    parser.add_argument('--step',        type=int,   default=0,
                        help='1=data, 2=train, 3=sindy, 0=all')
    parser.add_argument('--device',      type=str,   default=None)
    parser.add_argument('--epochs',      type=int,   default=2000)
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--n_runs',      type=int,   default=200,
                        help='Gillespie runs per parameter point')
    parser.add_argument('--save_dir',    type=str,   default='.')
    # Early stopping
    parser.add_argument('--es_patience', type=int,   default=200)
    parser.add_argument('--es_min_delta',type=float, default=1e-5)
    parser.add_argument('--es_target',   type=float, default=1e-5,
                        help='Early stopping only fires below this loss')
    args   = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    data_path = os.path.join(args.save_dir, 'sir_dataset.npz')
    run_all   = (args.step == 0)

    # ── Step 1: Generate Dataset ─────────────────────────────────────────────
    if run_all or args.step == 1:
        print("\n" + "="*60)
        print("STEP 1: Generating Stochastic SIR Dataset")
        print("="*60)
        generate_dataset(n_runs=args.n_runs, save_path=data_path)
        plot_stochastic_vs_deterministic(save_dir=args.save_dir)

    # ── Step 2: Train Neural ODE ─────────────────────────────────────────────
    if run_all or args.step == 2:
        print("\n" + "="*60)
        print("STEP 2: Training Neural ODE")
        print("="*60)
        train_neural_ode(
            data_path=data_path,
            n_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            save_dir=args.save_dir,
            es_patience=args.es_patience,
            es_min_delta=args.es_min_delta,
            es_target_loss=args.es_target,
        )

if __name__ == '__main__':
    main()