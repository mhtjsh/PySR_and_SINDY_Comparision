# PySR & SINDY Comparision
# PySR and SINDy Comparison on SIR Model

## Project Overview
This repository compares equation recovery techniques. The project utilizes PySR and SINDy. The workflow involves an SIR dataset. The Gillespie Algorithm creates the dataset. A Neural ODE trains on the data. PySR recovers equations from the Neural ODE model. SINDy recovers equations from the data. The comparison evaluates both methods.

## Repository Structure
The `src/` directory contains the Python scripts:
* `sir_pipeline.py`: Generates the SIR dataset and handles the Neural ODE training.
* `run_pysr_step1.py` & `run_pysr_step2.py`: Run the PySR implementation for equation recovery.
* `run_sindy.py`: Executes the SINDy implementation.
* `visualize_results.py`: Produces plots and charts.

## Methodology

### Data Generation
The Gillespie Algorithm simulates the SIR dynamics. This approach introduces fluctuations. 

### Neural ODE Training
A Neural ODE models the system dynamics. The model learns the derivatives from the states. The loss curve shows the training progression.

### Equation Recovery
* **PySR**: Symbolic regression recovers equations from the Neural ODE vector field.
* **SINDy**: SINDy uses a dictionary of candidate functions to recover the equations.

## Results

### Trajectory Simulations
The simulation contrasts Gillespie outputs with model trajectories.
![Stochastic vs Deterministic](results/step1_stochastic_vs_deterministic.png)

### Neural ODE Training Loss
The Neural ODE converges over epochs.
![Loss Curve](results/step2_loss_curve.png)

### Trajectory Comparison
The trajectory from the Neural ODE matches the data.
![Trajectory Comparison](results/step2_trajectory_comparison.png)

### Phase Portraits
The phase portraits map the states.
![Phase Portraits](results/step2_phase_portraits.png)

### SINDy Recovery
SINDy recovers the coefficient matrix.
![SINDy Recovery](results/step3_sindy_recovery.png)

## Conclusion
The comparison highlights differences in equation recovery mechanisms between PySR and SINDy.
