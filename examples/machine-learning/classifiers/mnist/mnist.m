
% Create a datastore of MNIST images
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Take data corresponding to digits 0 and 1
classLabels = ["0" "1"];
numClasses = length(classLabels);

% Partition into train and test sets
rng default
[imdsTrain,imdsTest] = splitEachLabel(imds,0.8,'randomize','Include',classLabels);

% Process images into features and labels
[trainX,trainY] = processIMDS(imdsTrain);
[testX,testY] = processIMDS(imdsTest);

% Build the network and train it using local resources
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

localNet = trainNetwork(trainX,trainY,layers,options);

% Test the network using remote resources

dev = quantum.backend.QuantumDeviceAWS('sv1');

% Construct a new network using the locally trained parameters, but specify
% the PQCLayer to execute on the remote device.
layers = [
    localNet.Layers(1)
    PQCLayer(dev, localNet.Layers(2).Weights)
    localNet.Layers(3:5)
    ];

remoteNet = SeriesNetwork(layers);

% Submit 1 sample to evaluate the model 
predY = classify(remoteNet, testX(1,:));


function [X,Y] = processIMDS(imds)
% Helper to process MNIST images into features data X and labels Y
% Input images (28,28) -> PCA features (1,8)
arguments
    imds (1,1) matlab.io.datastore.ImageDatastore
end

classLabels = unique(imds.Labels);
Y = categorical(imds.Labels, classLabels);

numSamples = size(imds.Labels,1);
X = zeros(numSamples, 28*28);
for i = 1:numSamples
    X(i,:) = reshape(double(readimage(imds, i)) ./ 255, [1 28*28]);
end

[~, scores, ~, ~, ~] = pca(X, NumComponents=8);
X = rescale(scores, 0, pi/2);
end
