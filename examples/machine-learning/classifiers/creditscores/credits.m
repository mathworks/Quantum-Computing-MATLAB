%% Classifying Credit Ratings 

% Load and process the data into labeled samples with 4 features
labeledData = readtable("CreditRating_Historical.dat");
[X, Y] = processCreditData(labeledData);

% numeric (1,4) -> categorical {better, worse}

numSamples = size(X,1);
numFeatures = size(X,2);
numClasses = 2;

rng default

% Use 99% of the data to train the models
partition = cvpartition(numSamples, "HoldOut", 0.01);
idx = partition.training;

trainX = X(idx,:);
trainY = Y(idx);
testX = X(~idx,:);
testY = Y(~idx);

%% Evaluate Classical Model

% Ensemble of bagged decision trees
numTrees = 25;
dTree = TreeBagger(numTrees,trainX,trainY);

predY = predict(dTree, testX);

figure
confusionchart(testY, categorical(predY))
title('Classical Model Predictions on Labeled Data')

%% Evaluate Quantum Model

% Quantum network using a qubit to encode each feature with classical
% output processing
% layers = [
%     featureInputLayer(numFeatures)
%     quantumCircuitLayer
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];
% 
% options = trainingOptions("adam", ...
%     MiniBatchSize=20, ...
%     InitialLearnRate=0.01, ...
%     ExecutionEnvironment="gpu", ...
%     Verbose=false, ...
%     MaxEpochs=50, ...
%     Plots="training-progress");
% 
% net = trainNetwork(trainX,trainY,layers,options);

model = load('trainedNetwork.mat');
net = model.net;

predY = classify(net, testX);

figure
confusionchart(testY, predY)
title('Quantum Model Predictions on Labeled Data')

%% Model Generalization

% The quantum and classical models were tested using unseen labeled data.
% Now compare the predictions of both models on unlabeled data.

unlabeledData = readtable("CreditRating_NewCompanies.dat");
X = processCreditData(unlabeledData);

Y2 = predict(dTree, X);

Y1 = classify(net, X);

figure
cm = confusionchart(categorical(Y2), Y1);
cm.XLabel = "Classical";
cm.YLabel = "Quantum";
title('Model Predictions on Unlabeled Data')

%% Helper to Process Credit Rating Data

function [X,Y] = processCreditData(data)
% Ratings {A,AA,AAA,B} are labeled as better and {BB,BBB,C} are labeled
% as worse. Use 4 features to classify a sample as worse or better: 

% (RE_TA)    - Retained Earnings / Total Assets
% (EBIT_TA)  - Earnings Before Interests and Taxes / Total Assets 
% (MVE_BVTD) - Market Value of Equity / Book Value of Total Debt 
% (S_TA)     - Sales / Total Assets 

arguments
    data table
end

vars = data.Properties.VariableNames;

% Choose features and rescale them in the range [0 pi/2]
features = vars(3:6);
data = normalize(data,"range", [0 pi/2], "DataVariables",features);

numSamples = size(data, 1);
numFeatures = length(features);

hasLabels = ismember('Rating', vars);
if hasLabels
    Y = categorical(repmat("worse", [numSamples 1]));
else
    Y = missing;
end

X = zeros(numSamples, numFeatures);
for ii  = 1:numSamples
    X(ii,:) = data{ii, features};
    if hasLabels
        if ismember(data.Rating(ii), {'A','AA','AAA','B'})
            Y(ii) = categorical("better");
        end
    end
end
end