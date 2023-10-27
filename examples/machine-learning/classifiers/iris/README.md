## Classifying Iris with Classical and Quantum Models

This example compares classical and quantum models for a binary classification
task on the Iris dataset. Using labeled data from [2], both models are 
trained and evaluated on unseen test data.

The quantum model is based on a Tree Tensor Network (TTN) circuit structure
[1] with post-processing. It was trained classically, and tested on real
quantum hardware as well as simulation. The classical model is a conventional
bagged decision tree. 

The dataset is available with the Statistics and Machine Learning Toolbox.
It can be accessed by opening the example [2]:
```matlab
openExample('stats/classdemo')
```

## Simulation and Hardware Test Results 

Models were trained using ~66% of the data, and tested on the remaining ~33%.
Diagonal elements represent the correct prediction.

![](confusionTestClassical.png?raw=true)

![](confusionTestQuantum.png?raw=true)

Only 10 samples from the test set were ran on real hardware. The IBM Lagos 
quantum device used 500 shots to classify each sample and the IonQ Harmony 
device used 100 shots.

![](confusionTestIBM.png?raw=true)

![](confusionTestAWS.png?raw=true)

## Required Products
- MATLAB&reg; Support Package for Quantum Computing
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox

## References 
[1] Hierarchical quantum classifiers (Grant et al. 2018)
[2] https://www.mathworks.com/help/stats/classification-example.html