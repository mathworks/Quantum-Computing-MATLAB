# Portfolio Optimization using VQE-CVaR

This example demonstrates a simulation of the Variational Quantum Eigensolver routine using the Conditional Value at Risk
aggregation function [1]. A simple mean-variance portfolio optimization problem [2] is considered as application in 
computational finance.  

The full classical simulation of the VQE optimizes a set of rotation angles used in the variational circuit. The angles from the final
iteration are used in the circuit sent to a real quantum processor. The solution bitstring is measured on the hardware with 
the highest probability.

## Required Products
- MATLAB&reg; Support Package for Quantum Computing
- Global Optimization Toolbox

## References 
[1] Improving Variational Quantum Optimization using CVaR (Barkoutsos et al. 2019)
[2] https://www.mathworks.com/help/finance/asset-allocation-case-study.html