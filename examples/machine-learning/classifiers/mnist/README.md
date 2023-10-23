# Quantum Classifier Network for MNIST Images  

This example simulates the training of a quantum neural network for a binary classification
task on the MNIST dataset. The quantum network is trained locally and evaluated 
on a remote device. The data pre-processing and circuit struture is inspired from [1].

The same binary classification is compared to a fully-classical neural network [2].

The training progress of the quantum network is displayed below and is comparable
to Figure 3 of [1]. 

![](mnist-training-results.png?raw=true)

## Required Products
- MATLAB&reg; Support Package for Quantum Computing
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox

## References 
[1] Hierarchical quantum classifiers (Grant et al. 2018)
[2] https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html
