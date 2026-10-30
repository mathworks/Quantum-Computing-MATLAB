# Clinical Trial Site Selection using Quantum Optimization

This example shows how to use [MATLAB&reg; Support for Quantum Computing](https://www.mathworks.com/products/quantum-computing.html) to map a small synthetic clinical trial site-selection problem from a classical binary optimization workflow to a QUBO solved with Tabu search and QAOA. It can then submit the optimized QAOA circuit to Amazon&reg; Web Services (AWS&reg;) Braket&reg; for sampling on SV1 simulation or available gate-model quantum hardware. It is written for MATLAB users who know Optimization Toolbox&trade; workflows and want a concrete introduction to MATLAB Support Package for Quantum Computing.

The base instance has 8 synthetic candidate sites, 4 selected sites, exact regional targets, pairwise overlap penalties, and optional hard incompatibility pairs. It is an operational planning template, not a clinical decision support system.

## File Manifest

| File | Description |
|---|---|
| `ClinicalTrialSiteSelection.m` | Self-contained plain-text Live Script with local functions |
| `+awsbraket/` | Backend profiles and shared AWS Braket measurement helpers |
| `awsConfig.env` | Template for AWS credential environment variables |
| `results-*.mat` | Optional offline replay data saved from previous AWS runs |
| `README.md` | This file |

## Setup Instructions

1. Open MATLAB R2024b or later.
2. Install **Optimization Toolbox** for the `intlinprog` baseline.
3. Install **MATLAB Support Package for Quantum Computing** for `qubo`, `tabuSearch`, and QAOA.
4. Set `ClinicalTrialSiteSelection/` as the current folder.
5. Open `ClinicalTrialSiteSelection.m` and run it section by section.

## AWS Braket Setup

Section 7 can submit the optimized QAOA circuit to AWS Braket. Follow the MathWorks documentation for setup: [Run a quantum circuit on hardware using AWS](https://www.mathworks.com/help/matlab/math/run-quantum-circuit-on-hardware-using-AWS.html).

In the script, choose one backend profile: `awsbraket.sv1()`, `awsbraket.ionqForte1()`, or `awsbraket.rigettiCepheus()`. SV1 is the default profile because it verifies the AWS workflow with noiseless simulation. IonQ Forte&reg; 1 is the more appropriate hardware profile for this dense QAOA circuit, but it is more costly to run and may involve queue time. Rigetti Cepheus&trade; is included as a lower-cost hardware pipeline check; expect a broad distribution and do not expect it to reproduce the SV1 result reliably. Update `awsS3Path`, `awsNumShots`, and `awsWaitTimeoutSeconds` for your account and selected backend. `awsS3Path` is passed directly to `QuantumDeviceAWS`; AWS Braket records the exact output location in the task details. The script loads AWS credential environment variables from `awsConfig.env`; customize that setup if you use a different credential workflow.

AWS submission is off by default. To submit a new AWS task, change the Section 7 `setenv("RUN_ON_AWS","false")` line to `"true"` before running that section. When submission is off, the script reads `offlineResultsFile` only if you set it to a MAT-file saved from a matching Braket run. Offline replay is only valid when the saved results came from the same 8-site circuit, data, QAOA settings, backend, and shot count you want to discuss.

The backend profiles keep provider-specific details out of the main Live Script. `awsbraket.ionqForte1()` handles Forte 1 angle bounds, while `awsbraket.sv1()` and `awsbraket.rigettiCepheus()` submit the circuit unchanged. AWS results are fetched with `fetchOutput`, which returns a MATLAB `quantum.gate.QuantumMeasurement` object. Save that object to a MAT-file for offline replay instead of parsing backend-specific Braket JSON.

## Notes

- `mapX` and `mapY` are synthetic 0-to-1 teaching coordinates. They are used only to create a simple overlap matrix; they are not real geography.
- The QUBO penalty variables are named by purpose: `countPenalty`, `regionPenalty`, and `conflictPenalty`. They are individually tunable, even though the demo starts them at the same value.
- Cost and capacity are reported as operational metrics but are not hard constraints in the base version.
- QAOA runs locally as a simulated hybrid quantum-classical algorithm to tune the circuit angles. The AWS Braket step samples the fixed optimized circuit; it does not rerun the hybrid QAOA loop on AWS.
- The default `qaoaNumLayers=2` setting gives better probability concentration for this example than one layer. Reducing to one layer can produce results with noisy hardware, but it usually spreads probability across more bit strings.
- The example can run on SV1 or available gate-model QPUs, but it does not claim quantum advantage. Hardware runs can produce broad distributions even when the task completes successfully, especially when the compiled circuit is deep for the selected backend.
- The example uses only synthetic site data. Do not replace it with patient-level data, protected health information, real investigator names, or claims about actual clinical trial sites unless the surrounding workflow is approved for that use.
- Reading results from the Rigetti output requires the May 2026 or newer version of the support package
