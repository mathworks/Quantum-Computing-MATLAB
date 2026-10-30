# Example Spec

## 1. Example Summary

- **Example name**: `BeamActivation`
- **Domain**: Multi-beam satellite activation under interference
- **Primary MATLAB audience**: MATLAB users familiar with optimization workflows, but new to quantum optimization
- **Template status**: `Complete`

## 2. Problem Statement

- A satellite operator must decide which spot beams to activate in a time slot.
- Each beam activation is binary: on or off.
- The objective is to maximize total served demand or throughput.
- Adjacent or overlapping beams that share a channel cannot both be active because of interference.

## 3. Why This Is a Good Quantum-Template Candidate

- The decision variables are naturally binary.
- The interference constraints are naturally pairwise, so they map cleanly to quadratic penalties.
- The quantum-friendly slice is a weighted subset-selection problem with conflicts.
- A MATLAB user would normally solve this with `intlinprog`, heuristics, or brute force at small scale.

## 4. Intended Learning Outcomes

Users should leave this example understanding:

1. How to recognize a binary conflict problem as a good QUBO candidate.
2. How to translate a classical ILP-style formulation into a QUBO matrix with rewards and penalties.
3. How to compare classical, classical-QUBO, and QAOA workflows on the same problem instance.

## 5. Tactical Scope

- The demonstrated instance is a small hex-grid beam layout.
- Demand is represented by a simple tactical traffic pattern rather than full RF or orbital modeling.
- Interference is modeled through adjacency in the beam layout.
- Out of scope for the first version:
  - multiple time slots
  - channel assignment beyond binary activation
  - non-binary power levels
  - temporal coupling across slots
  - full physical beam overlap modeling

## 6. Classical Formulation

- Baseline solver: `intlinprog`
- Objective form: maximize weighted sum of active beams
- Constraint form: for each conflict pair, `x_i + x_j <= 1`
- How correctness will be checked: brute-force enumeration at the demonstrated scale

## 7. QUBO Formulation

- Binary variable meaning: `x_i = 1` means beam `i` is active
- Diagonal term meaning: reward for activating a beam, encoded as negative demand
- Off-diagonal term meaning: penalty for activating conflicting beams together
- Penalty coefficient approach: choose `lambda > max(demand)` as a safe starting point
- Any feasibility checker needed: yes, an explicit conflict-violation checker is used

## 8. Quantum Workflow

- Primary quantum method: `QAOA`
- High-level built-in path: `solve(qprob, Algorithm=qaoa)`
- Any advanced or optional "under the hood" content: none in the current version beyond tuning-oriented explanation
- What should be compared against the classical baseline:
  - selected beams
  - total throughput
  - feasibility
  - whether the returned solution matches the classical optimum

## 9. Visuals

- Domain plot: beam conflict graph with demand shown through marker size and color
- QUBO plot: heatmap of the QUBO matrix
- Result comparison plot: classical vs QAOA selected-beam overlays, plus tuning plots

## 10. Dependencies

- Required MATLAB release: `R2024b or later`
- Required MathWorks products:
  - MATLAB Support Package for Quantum Computing
  - Optimization Toolbox
- Any optional products: none

## 11. Deliverables

- [x] Main MATLAB script
- [x] `README.md`
- [x] `SPEC.md`
- [x] MATLAB verification completed
- [x] Example entered in `EXAMPLES_INDEX.md`

## 12. Verification Notes

- Date verified: `2026-05-26`
- Verified by: user
- MATLAB execution status: user reported the current version runs correctly
- Known limitations:
  - local QAOA timing reflects classical simulation cost, not hardware scaling
  - the example is intentionally limited to a small tactical binary-activation slice

## 13. Adaptation Points

Users should be able to replace:

- Instance generation: beam geometry and demand pattern
- Conflict / constraint logic: adjacency or incompatibility rule
- Reward and penalty model: throughput weights and conflict penalties

## 14. Open Questions

- Should later versions include a second, slightly richer satellite example with time-slot structure?
- Should this example eventually include an optional advanced section showing a custom QAOA circuit?
- Should the repo standardize how verified output snapshots are handled in Live Script exports?
