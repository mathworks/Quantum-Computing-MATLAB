# Multi-Beam Satellite Activation using Quantum Optimization

This example shows how a single-resource beam activation problem maps to a weighted maximum independent set problem, then to a QUBO that can be solved with Tabu search and QAOA. It is written for MATLAB users who are comfortable with Optimization Toolbox and binary optimization, but are new to quantum optimization workflows and the MATLAB Support Package for Quantum Computing.

The base instance is deliberately small: it is a teaching template, not a quantum advantage benchmark. The same modeling pattern becomes more relevant as a building block in richer planning models with variables such as beam, gateway, frequency, and time-slot choices.

## File Manifest

| File | Description |
|---|---|
| `BeamActivation.m` | Main plain-text Live Script - open in MATLAB and run section by section |
| `mis2qubo.m` | Utility: converts a weighted maximum independent set problem, represented as a graph or adjacency matrix, to a `qubo` object |
| `README.md` | This file |

## Setup Instructions

1. Open MATLAB. MATLAB R2025a or later is recommended for plain-text Live Script rendering.
2. Install **Optimization Toolbox** - required for the classical problem-based MILP baseline.
3. Install the **MATLAB Support Package for Quantum Computing** - required for `qubo`, Tabu search, QAOA, and quantum circuit workflows.
4. Set `BeamActivation/` as the current folder, or add it to the MATLAB path so that `mis2qubo.m` is visible.
5. Open `BeamActivation.m` and run it section by section. Edit the beam layout, conflict rule, demand pattern, and result-scoring logic to adapt the template to your own max-weight independent set problem.

## Notes

- The early modeling and MILP sections use standard Optimization Toolbox concepts: binary decision variables, a linear reward, and pairwise conflict constraints.
- The QUBO section shows the main quantum-optimization bridge: hard conflict constraints become quadratic penalties in an unconstrained binary objective.
- The same `qprob` object is used for the classical QUBO heuristic and for QAOA, which keeps the comparison focused on solver workflow rather than reformulating the problem multiple times.
- QAOA in this example runs locally as a simulated hybrid quantum-classical algorithm. The QAOA result includes the optimized `quantumCircuit` and state probabilities; the hardware section shows how to inspect and submit that returned circuit. The cloud execution block is commented out so the example runs without credentials.
- When adapting the template, keep the penalty coefficient large enough that violating a conflict is more expensive than the reward gained by selecting conflicting nodes.
- Hardware measurement results are bit strings. For a new domain, update the scoring and visualization code so those bit strings are translated back into meaningful application metrics.
