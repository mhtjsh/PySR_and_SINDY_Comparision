#!/usr/bin/env python3
"""
Step 2: Run PySR without importing torch.
This avoids the torch/juliacall segfault.

Usage:
  python run_pysr_step1.py   # first: generate data (uses torch)
  python run_pysr_step2.py   # then:  run PySR (no torch!)
"""
import os
import numpy as np
import pandas as pd
from pysr import PySRRegressor

SAVE_DIR = 'results'


def run_pysr_target(X_df, y, target_name, label):
    """Run PySR for one target variable."""
    # Subsample for speed
    n_sub = min(5000, len(y))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y), n_sub, replace=False)

    X_sub = X_df.iloc[idx].reset_index(drop=True)
    y_sub = y[idx]

    print(f"\n  PySR for {target_name} ({label}) — {n_sub} samples")
    print(f"  y range: [{y_sub.min():.5f}, {y_sub.max():.5f}]")

    model = PySRRegressor(
        niterations=80,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        populations=30,
        population_size=50,
        maxsize=12,
        parsimony=0.01,
        model_selection="best",
        timeout_in_seconds=300,
        random_state=42,
        deterministic=True,
        parallelism="serial",
        turbo=False,
        progress=True,
        temp_equation_file=os.path.join(SAVE_DIR, f"pysr_{label}_{target_name}.csv"),
    )
    model.fit(X_sub, y_sub)

    print(f"\n  Pareto front for {target_name}:")
    if hasattr(model, 'equations_') and model.equations_ is not None:
        for _, row in model.equations_.iterrows():
            print(f"    complexity={row['complexity']:2.0f}  loss={row['loss']:.6e}  {row['equation']}")
    print(f"\n  ✅ Best {target_name}: {model.sympy()}")
    return model


def main():
    data_path = os.path.join(SAVE_DIR, 'pysr_input_data.npz')
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run run_pysr_step1.py first!")
        return

    data = np.load(data_path)
    print("Loaded PySR input data")

    for label, suffix in [("TrueVF", "true"), ("NeuralODE", "node")]:
        S = data[f'S_{suffix}']
        I = data[f'I_{suffix}']
        b = data[f'b_{suffix}']
        g = data[f'g_{suffix}']
        dS = data[f'dS_{suffix}']
        dI = data[f'dI_{suffix}']

        # Filter active dynamics
        mag = np.sqrt(dS**2 + dI**2)
        mask = mag > 1e-4
        S, I, b, g, dS, dI = S[mask], I[mask], b[mask], g[mask], dS[mask], dI[mask]
        print(f"\n{'='*60}")
        print(f"PySR Recovery — {label} ({mask.sum()} active points)")
        print(f"{'='*60}")

        X_df = pd.DataFrame({'svar': S, 'ivar': I, 'bvar': b, 'gvar': g})

        m_dS = run_pysr_target(X_df, dS, 'dS_dt', label)
        m_dI = run_pysr_target(X_df, dI, 'dI_dt', label)

        print(f"\n  Summary for {label}:")
        print(f"    dS/dt: {m_dS.sympy()}")
        print(f"    dI/dt: {m_dI.sympy()}")

    print("\n  Ground truth:")
    print("    dS/dt = -bvar * svar * ivar")
    print("    dI/dt =  bvar * svar * ivar - gvar * ivar")
    print("\n✅ PySR complete!")


if __name__ == '__main__':
    main()
