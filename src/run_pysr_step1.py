#!/usr/bin/env python3
"""
Step 1: Prepare PySR data using torch (saves to .npz)
Run this FIRST, then run run_pysr_step2.py (which never imports torch).
"""
import os, sys, numpy as np, torch

sys.path.insert(0, os.path.dirname(__file__))
from sir_pipeline import NeuralODE

SAVE_DIR = 'results'

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    data_path = os.path.join(SAVE_DIR, 'sir_dataset.npz')

    data = np.load(data_path)
    params, mean_traj = data['params'], data['mean_traj']

    # Load Neural ODE
    model = NeuralODE(hidden=128).to(device)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pt'), map_location=device))
    model.eval()
    print("Loaded Neural ODE")

    # --- True VF ---
    all_S_t, all_I_t, all_b_t, all_g_t, all_dS_t, all_dI_t = [], [], [], [], [], []
    for idx in range(len(params)):
        b_val, g_val = params[idx]
        S_arr, I_arr = mean_traj[idx, :, 0], mean_traj[idx, :, 1]
        all_S_t.append(S_arr); all_I_t.append(I_arr)
        all_b_t.append(np.full(len(S_arr), b_val)); all_g_t.append(np.full(len(S_arr), g_val))
        all_dS_t.append(-b_val * S_arr * I_arr)
        all_dI_t.append(b_val * S_arr * I_arr - g_val * I_arr)

    # --- Neural ODE VF ---
    all_S_n, all_I_n, all_b_n, all_g_n, all_dS_n, all_dI_n = [], [], [], [], [], []
    for idx in range(len(params)):
        b_val, g_val = params[idx]
        traj = mean_traj[idx]
        y_torch = torch.tensor(traj, dtype=torch.float32).to(device)
        model.vf.set_params(b_val, g_val)
        with torch.no_grad():
            dy = model.vf(0.0, y_torch).cpu().numpy()
        all_S_n.append(traj[:, 0]); all_I_n.append(traj[:, 1])
        all_b_n.append(np.full(len(traj), b_val)); all_g_n.append(np.full(len(traj), g_val))
        all_dS_n.append(dy[:, 0]); all_dI_n.append(dy[:, 1])

    # Save
    out_path = os.path.join(SAVE_DIR, 'pysr_input_data.npz')
    np.savez(out_path,
        # True VF
        S_true=np.concatenate(all_S_t), I_true=np.concatenate(all_I_t),
        b_true=np.concatenate(all_b_t), g_true=np.concatenate(all_g_t),
        dS_true=np.concatenate(all_dS_t), dI_true=np.concatenate(all_dI_t),
        # Neural ODE VF
        S_node=np.concatenate(all_S_n), I_node=np.concatenate(all_I_n),
        b_node=np.concatenate(all_b_n), g_node=np.concatenate(all_g_n),
        dS_node=np.concatenate(all_dS_n), dI_node=np.concatenate(all_dI_n),
    )
    print(f"Saved PySR input data → {out_path}")
    print("Now run: python run_pysr_step2.py")

if __name__ == '__main__':
    main()
