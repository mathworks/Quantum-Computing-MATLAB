
% Load the trained network
% trained = load('trainedNetwork.mat');
% trainedLayers = trained.net.Layers;

% Account setup required for constructing QuantumDeviceIBM can be found at:
% https://www.mathworks.com/help/matlab/math/run-quantum-circuit-on-hardware-using-IBM.html

% Construct a new network using the classically trained weights
% device = quantum.backend.QuantumDeviceIBM("ibm_algiers", UseSession=true, AccountName="...");
% layers = [
%     trainedLayers(1)
%     quantumCircuitLayerIBM(device, trainedLayers(2).Weights)
%     trainedLayers(3:5)
%     ];
% net = SeriesNetwork(layers);

% Load and classify test data
testdata = load('testDataCredit.mat');
% predY = classify(net, testdata.testX);

results = load('ibm_algiers_pred_1000.mat');
predY = results.predY;

figure
accur = sum(testdata.testY==predY)/numel(testdata.testY);
confusionchart(testdata.testY, predY);
title("IBM Algiers Test Results")