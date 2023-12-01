
% Load and process the data into labeled samples with 4 features
labeledData = readtable("CreditRating_Historical.dat");
[X, Y] = processCreditData(labeledData);

% numeric features (1,4) -> categorical label {better, worse}

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

% Save the test data so it can be loaded when running on hardware
save('testDataCredit.mat', 'testX', 'testY')

%% Evaluate Classical Model

% Ensemble of bagged decision trees
numTrees = 25;
dTree = TreeBagger(numTrees,trainX,trainY);

predY = predict(dTree, testX);

figure
accur = sum(testY==categorical(predY))/numel(testY);
confusionchart(testY, categorical(predY))
title("Classical Test Results")

%% Evaluate Quantum Model

% Quantum TTN circuit (Figure 6 [1]) with classical processing
% on the expectation value of a qubit 
% layers = [
%     featureInputLayer(numFeatures)
%     quantumCircuitLayer
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];
% 
% options = trainingOptions("adam", ...
%     MiniBatchSize=50, ...
%     InitialLearnRate=0.01, ...
%     ExecutionEnvironment="cpu", ...
%     Verbose=false, ...
%     MaxEpochs=25, ...
%     Plots="training-progress");
% 
% net = trainNetwork(trainX,trainY,layers,options);
% save('trainedNetwork.mat', 'net')

trained = load('trainedNetwork.mat');
net = trained.net;
 
predY = classify(net, testX);

figure
accur = sum(testY==predY)/numel(testY);
confusionchart(testY, predY);
title("Simulation Test Results")

%% Evaluate Models on New Data

% The quantum and classical models were tested using unseen labeled data.
% Now compare the predictions of both models on new unlabeled data.

% unlabeledData = readtable("CreditRating_NewCompanies.dat");
% X = processCreditData(unlabeledData);
% 
% Y1 = predict(dTree, X);
% Y1 = categorical(Y1);
% 
% Y2 = classify(net, X);
% 
% figure
% accur = sum(Y1==Y2)/size(X,1);
% cm = confusionchart(Y1, Y2);
% cm.XLabel = "Classical";
% cm.YLabel = "Quantum";