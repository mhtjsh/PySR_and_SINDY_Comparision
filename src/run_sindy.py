#!/usr/bin/env python3
"""
SINDy-only equation recovery (no Julia/PySR dependency).
Runs instantly on the trained Neural ODE and true derivatives.
"""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from sir_pipeline import NeuralODE

SAVE_DIR = 'results'
FEATURE_NAMES = [
    '1', 'svar', 'ivar', 'svar*ivar',
    'bvar*svar', 'bvar*ivar', 'bvar*svar*ivar',
    'gvar*svar', 'gvar*ivar', 'gvar*svar*ivar',
]

def query_neural_ode_vf(model, data_path, device):
    data = np.load(data_path)
    params, mean_traj = data['params'], data['mean_traj']
    all_S, all_I, all_b, all_g, all_dS, all_dI = [], [], [], [], [], []
    model.eval()
    for idx in range(len(params)):
        b_val, g_val = params[idx]
        traj = mean_traj[idx]
        y_torch = torch.tensor(traj, dtype=torch.float32).to(device)
        model.vf.set_params(b_val, g_val)
        with torch.no_grad():
            dy = model.vf(0.0, y_torch).cpu().numpy()
        all_S.append(traj[:, 0]); all_I.append(traj[:, 1])
        all_b.append(np.full(len(traj), b_val)); all_g.append(np.full(len(traj), g_val))
        all_dS.append(dy[:, 0]); all_dI.append(dy[:, 1])
    return tuple(np.concatenate(x) for x in [all_S, all_I, all_b, all_g, all_dS, all_dI])

def query_true_vf(data_path):
    data = np.load(data_path)
    params, mean_traj = data['params'], data['mean_traj']
    all_S, all_I, all_b, all_g, all_dS, all_dI = [], [], [], [], [], []
    for idx in range(len(params)):
        b_val, g_val = params[idx]
        S_arr, I_arr = mean_traj[idx, :, 0], mean_traj[idx, :, 1]
        all_S.append(S_arr); all_I.append(I_arr)
        all_b.append(np.full(len(S_arr), b_val)); all_g.append(np.full(len(S_arr), g_val))
        all_dS.append(-b_val * S_arr * I_arr)
        all_dI.append(b_val * S_arr * I_arr - g_val * I_arr)
    return tuple(np.concatenate(x) for x in [all_S, all_I, all_b, all_g, all_dS, all_dI])

def build_library(S, I, b, g):
    return np.column_stack([np.ones(len(S)), S, I, S*I, b*S, b*I, b*S*I, g*S, g*I, g*S*I])

def stlsq(Theta, y, threshold=0.1, max_iter=30):
    xi = np.linalg.lstsq(Theta, y, rcond=None)[0]
    for _ in range(max_iter):
        small = np.abs(xi) < threshold
        xi[small] = 0.0
        big = np.where(~small)[0]
        if len(big) == 0: break
        xi[big] = np.linalg.lstsq(Theta[:, big], y, rcond=None)[0]
    return xi

def fmt_eq(xi, target_name):
    terms = [f"({c:+.4f})*{FEATURE_NAMES[j]}" for j, c in enumerate(xi) if abs(c) > 1e-10]
    return f"{target_name} = {' '.join(terms) if terms else '0'}"

def run_sindy(S, I, b, g, dS, dI, label=""):
    print(f"\n{'='*60}")
    print(f"SINDy Recovery ({label})")
    print(f"{'='*60}")
    mag = np.sqrt(dS**2 + dI**2)
    mask = mag > 1e-4
    S, I, b, g, dS, dI = S[mask], I[mask], b[mask], g[mask], dS[mask], dI[mask]
    print(f"  Active points: {mask.sum()}")
    
    Theta = build_library(S, I, b, g)
    n = len(S)
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    results = {}
    
    for tname, ty in [('dS/dt', dS), ('dI/dt', dI)]:
        print(f"\n  --- {tname} ---")
        best_aicc, best_xi, best_thr = float('inf'), None, None
        for thr in thresholds:
            xi = stlsq(Theta, ty, threshold=thr)
            k = np.count_nonzero(xi)
            if k == 0: continue
            resid = ty - Theta @ xi
            mse = np.mean(resid**2)
            r2 = 1 - np.sum(resid**2) / np.sum((ty - ty.mean())**2)
            aicc = n * np.log(max(mse, 1e-300)) + 2*k + (2*k*(k+1))/max(n-k-1, 1)
            print(f"    thr={thr:.3f}  k={k:2d}  R²={r2:.6f}  AICc={aicc:.1f}  {fmt_eq(xi, tname)}")
            if aicc < best_aicc:
                best_aicc, best_xi, best_thr = aicc, xi.copy(), thr
        results[tname] = best_xi
        print(f"\n  ✅ Best {tname} (thr={best_thr}): {fmt_eq(best_xi, tname)}")
    
    return results, S, I, b, g, dS, dI

def make_sindy_plots(sindy_true, sindy_node, dat_t, dat_n):
    """Plot SINDy results for both strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for col, (label, sindy_res, dat) in enumerate([
        ("True VF", sindy_true, dat_t),
        ("Neural ODE VF", sindy_node, dat_n),
    ]):
        S, I, b, g, dS, dI = dat
        Theta = build_library(S, I, b, g)
        dS_true = -b * S * I
        dI_true = b * S * I - g * I
        
        for row, (tname, dt, xi) in enumerate([
            ('dS/dt', dS_true, sindy_res['dS/dt']),
            ('dI/dt', dI_true, sindy_res['dI/dt']),
        ]):
            ax = axes[row, col]
            dy_sindy = Theta @ xi
            ax.scatter(dt, dy_sindy, alpha=0.03, s=2, c='darkorange' if col == 0 else 'steelblue')
            lims = [min(dt.min(), dy_sindy.min()), max(dt.max(), dy_sindy.max())]
            ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect')
            r2 = 1 - np.sum((dt - dy_sindy)**2) / np.sum((dt - dt.mean())**2)
            ax.set_title(f'{tname} — SINDy ({label})\nR²={r2:.6f}', fontsize=11)
            ax.set_xlabel('True'); ax.set_ylabel('SINDy')
            ax.legend()
    
    plt.suptitle('SINDy Equation Recovery: True VF vs Neural ODE VF', fontsize=14)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'step3_sindy_recovery.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    data_path = os.path.join(SAVE_DIR, 'sir_dataset.npz')

    model = NeuralODE(hidden=128).to(device)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pt'), map_location=device))
    model.eval()
    print("Loaded Neural ODE")

    print("\n" + "█"*60)
    print("STRATEGY A: TRUE SIR derivatives")
    print("█"*60)
    dat_t = query_true_vf(data_path)
    sindy_true, *dat_t_filtered = run_sindy(*dat_t, "True VF")

    print("\n" + "█"*60)
    print("STRATEGY B: Neural ODE vector field")
    print("█"*60)
    dat_n = query_neural_ode_vf(model, data_path, device)
    sindy_node, *dat_n_filtered = run_sindy(*dat_n, "Neural ODE VF")

    make_sindy_plots(sindy_true, sindy_node, dat_t_filtered, dat_n_filtered)

    print("\n" + "="*60)
    print("FINAL SINDy SUMMARY")
    print("="*60)
    print("\nGround Truth:")
    print("  dS/dt = -β·S·I")
    print("  dI/dt =  β·S·I - γ·I")
    print("\nSINDy (True VF) — perfect baseline:")
    for tn in ['dS/dt', 'dI/dt']:
        print(f"  {fmt_eq(sindy_true[tn], tn)}")
    print("\nSINDy (Neural ODE VF) — from learned model:")
    for tn in ['dS/dt', 'dI/dt']:
        print(f"  {fmt_eq(sindy_node[tn], tn)}")
    
    print("\n✅ SINDy recovery complete!")
