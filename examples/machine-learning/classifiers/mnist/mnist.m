%% MNIST Data 

% Create a datastore of MNIST images
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Take data corresponding to digits 0 and 1
classLabels = categorical(["0"; "1"]);
numClasses = length(classLabels);

% Partition into train and test sets
rng default
[imdsTrain,imdsTest] = splitEachLabel(imds,0.8,'randomize','Include',classLabels);

% Restrict labels to the desired classes
imdsTrain.Labels = categorical(imdsTrain.Labels, classLabels);
imdsTest.Labels = categorical(imdsTest.Labels, classLabels);

%% Classical Network

% Build and train the classical network
layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    InitialLearnRate=0.01, ...
    ExecutionEnvironment="gpu", ...
    MaxEpochs=4, ...
    Verbose=false);

classicalNet = trainNetwork(imdsTrain,layers,options);

predY = classify(classicalNet, imdsTest);
confusionchart(imdsTest.Labels, predY);

%% Quantum Network

% Process ImageDataStore into features and labels
[trainX,trainY] = processIMDS(imdsTrain);
[testX,testY] = processIMDS(imdsTest);

% Build the quantum network and train it using local resources
inputSize = size(trainX,2);
layers = [
    featureInputLayer(inputSize)
    PQCLayer("local")
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions("adam", ...
    MiniBatchSize=20, ...
    InitialLearnRate=0.1, ...
    ExecutionEnvironment="gpu", ...
    Verbose=true, ...
    Plots="training-progress");

quantumNetLocal = trainNetwork(trainX,trainY,layers,options);

% Test the network using remote resources by constructing a new network with
% the locally trained parameters. The PQCLayer will now execute on the 
% remote device.
dev = quantum.backend.QuantumDeviceAWS('sv1');

layers = [
    quantumNetLocal.Layers(1)
    PQCLayer(dev, quantumNetLocal.Layers(2).Weights)
    quantumNetLocal.Layers(3:5)
    ];

quantumNetRemote = SeriesNetwork(layers);

% Submit 1 sample to evaluate the model on the remote device
predY = classify(quantumNetRemote, testX(1,:));

%% Helper functions
function [X,Y] = processIMDS(imds)
% Convert MNIST images into features data X and labels Y
% Input images (28,28) -> PCA features (1,8)
arguments
    imds (1,1) matlab.io.datastore.ImageDatastore
end

Y = categorical(imds.Labels);

numSamples = size(imds.Labels,1);
X = zeros(numSamples, 28*28);
for i = 1:numSamples
    X(i,:) = reshape(double(readimage(imds, i)) ./ 255, [1 28*28]);
end

[~, scores, ~, ~, ~] = pca(X, NumComponents=8);
X = rescale(scores, 0, pi/2);
end
