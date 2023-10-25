## Classifying Credit Ratings with Classical and Quantum Models

This example compares classical and quantum models for a binary classification
task on synthetic credit rating data. Using labeled data from [2], both models are 
trained and evaluated on unseen test data. A new credit dataset of unlabeled 
data is used to compare the predictions of both models.

The quantum circuit is based on Tree Tensor Network (TTN) structure [1] and
is simulated without noise during training and test evaluation.  

Data is available with the Statistics and Machine Learning Toolbox. It can
be accessed by opening the example [2]:
```matlab
openExample('stats/creditratingdemo')
```

## Results on Labeled and Unlabeled Test Data 

In the two figures below, diagonal elements represent the respective model
making the correct prediction on unseen labeled samples.

![](confusionTestClassical.png?raw=true)

![](confusionTestQuantum.png?raw=true)

Here, the diagonal elements represent the two models making the same 
prediction on an unknown sample.

![](confusionUnlabeledTestBoth.png?raw=true)

## Required Products
- MATLAB&reg; Support Package for Quantum Computing
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox

## References 
[1] Hierarchical quantum classifiers (Grant et al. 2018)
[2] https://www.mathworks.com/help/stats/credit-rating-by-bagging-decision-trees.html