
% Create a datastore of MNIST images
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Take some data corresponding to digits 0 and 1
classLabels = ["0" "1"];
numClasses = length(classLabels);
numSamplesPerClass = 500;
N = numClasses*numSamplesPerClass;

rng default
[imds01,imdsOthers] = splitEachLabel(imds,numSamplesPerClass,'randomize','Include',classLabels);

X = zeros(N, 16);
Y = categorical(imds01.Labels, classLabels);

% Downscale samples 4x4 images of greyscale intensity
for i = 1:N
x = imresize(readimage(imds01,i), [4 4]);
X(i, :) = reshape( double(x) ./ 255 , [1 16]);
end

% Features are the rescaled 8 principle components with highest variance
[~, scores, ~, ~, ~] = pca(X, NumComponents=8);
X = rescale(scores, 0, pi/2);

% Build and train the network
inputSize = size(X,2);
layers = [
    featureInputLayer(inputSize)
    PQCLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions("adam", ...
    MiniBatchSize=20, ...
    InitialLearnRate=0.1, ...
    ExecutionEnvironment="gpu", ...
    Verbose=true, MaxEpochs=30, Plots="training-progress");

net = trainNetwork(X,Y,layers,options);