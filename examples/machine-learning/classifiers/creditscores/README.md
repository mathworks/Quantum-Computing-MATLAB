## Classifying Credit Ratings with Classical and Quantum Models

This example compares classical and quantum models for a binary classification
task on synthetic credit rating data. Using labeled data from [2], both models are 
trained and evaluated on unseen test data. New samples of unlabeled data
are used to compare the simulated predictions of both models.

The quantum model is based on a Tree Tensor Network (TTN) circuit structure
[1] with post-processing. It was trained classically, and tested on real
quantum hardware as well as simulation. The classical model is a conventional
bagged decision tree. 

Details of data pre-processing are shown in the ```processCreditData.m```
function. The raw data is available with the Statistics and Machine 
Learning Toolbox. It can be accessed by opening the example [2]:
```matlab
openExample('stats/creditratingdemo')
```

## Simulation and Hardware Test Results 

Models were trained using 99% of the data, and tested on the remaining 1%.
Diagonal elements represent the correct test prediction. The simulated 
quantum model performs slighly better on test data than the classical
decision tree model.

![](confusionTestClassical.png?raw=true)

![](confusionTestQuantum.png?raw=true)

The IBM Algiers quantum device was used with 1000 shots to classify the
test set.

![](confusionTestHardwareAlgiers.png?raw=true)

The Aria 1 quantum device was used with 100 shots to classify 10 samples
from the test set.

![](confusionTestHardwareAria1.png?raw=true)

## Simulation Test Results on New Data

Using new and unlabeled data, both models were simulated to test the
similarity of their predictions. Here, the diagonal elements represent the
two models making the same prediction on unknown samples. The models agree 
on many predictions, but the true accuracy is unknown.

![](confusionUnlabeledTestBoth.png?raw=true)

## Required Products
- MATLAB&reg; Support Package for Quantum Computing
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox

## References 
[1] Hierarchical quantum classifiers (Grant et al. 2018)
[2] https://www.mathworks.com/help/stats/credit-rating-by-bagging-decision-trees.html