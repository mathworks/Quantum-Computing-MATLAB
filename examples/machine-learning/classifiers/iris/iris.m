
% Example from Statistics and Machine Learning Toolbox
data = readtable("fisheriris.csv");
[X,Y] = processIrisData(data);

gscatter(X(:,1), X(:,2), Y,'rgb','osd');
xlabel('Sepal length');
ylabel('Sepal width');

numSamples = size(X,1);
numFeatures = size(X,2);
numClasses = 2;

rng default

partition = cvpartition(numSamples, "HoldOut", 0.33);
idx = partition.training;

trainX = X(idx,:);
trainY = Y(idx);
testX = X(~idx,:);
testY = Y(~idx);

save('testDataIris.mat', 'testX', "testY")

%% Classical Model

% Bagged decision tree
numTrees = 25;
dTree = TreeBagger(numTrees,trainX,trainY);

predY = predict(dTree, testX);

figure
accur = sum(testY==predY)/numel(testY);
confusionchart(testY, categorical(predY));
title('Classical Test Accuracy: '+string(accur))

%% Quantum Model

% Build the quantum network
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
%     ExecutionEnvironment="cpu", ...
%     Verbose=false, ...
%     MaxEpochs=10, ...
%     Plots="training-progress");
% 
% net = trainNetwork(trainX,trainY,layers,options);
% save('trainedIrisModel.mat', 'net')

trained = load('trainedIrisModel.mat');
net = trained.net;

predY = classify(net, testX);

figure
accur = sum(testY==predY)/numel(testY);
confusionchart(testY, predY);
title('Quantum Simulation Test Accuracy: '+string(accur))

function [X,Y] = processIrisData(data)
% See Table 2 [1]
arguments
    data table
end

% Select data from two of the three classes for binary classification
classes = {'setosa', 'virginica'};
selection = ismember(data.Species, classes);
data = data(selection, :);

% Rescale features in the range [0 pi/2]
vars = data.Properties.VariableNames;
features = vars(1:4);
data = normalize(data,"range", [0 pi/2], "DataVariables",features);

numSamples = size(data, 1);
numFeatures = length(features);

Y = categorical(repmat(classes(1), [numSamples 1]));
X = zeros(numSamples, numFeatures);
for ii  = 1:numSamples
    X(ii,:) = data{ii, features};
    if isequal(data.Species{ii}, classes{2})
        Y(ii) = categorical(classes(2));
    end
end
end