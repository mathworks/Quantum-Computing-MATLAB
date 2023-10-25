## Classifying Credit Ratings with Classical and Quantum Models

This example compares classical and quantum models for a binary classification
task on credit rating data. Using labeled data from [2], both models are 
trained and evaluated on unseen test data. The generalization of both models 
is compared by also evaluating on unlabeled data. 

The quantum model is based on [1], and is fully simulated without noise
during training and testing. 

Data is available with the Statistics and Machine Learning Toolbox. It can
be accessed by opening the example [2]:
```matlab
openExample('stats/creditratingdemo')
```

openExample('stats/creditratingdemo')

## Results on Labeled and Unlabeled Test Data 

The diagonal elements represent the models making the correct prediction

![](confusionTestClassical.png.png.png?raw=true)

![](confusionTestQuantum.png.png?raw=true)

The diagonal elements represent the number of times the two models make
the same prediction.

![](confusionUnlabeledTestBoth.png?raw=true)

## Required Products
- MATLAB&reg; Support Package for Quantum Computing
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox

## References 
[1] Hierarchical quantum classifiers (Grant et al. 2018)
[2] https://www.mathworks.com/help/stats/credit-rating-by-bagging-decision-trees.html