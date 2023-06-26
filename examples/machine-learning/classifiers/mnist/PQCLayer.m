classdef PQCLayer < nnet.layer.Layer
    properties (Learnable)
        Weights
    end

    methods
        function layer = PQCLayer
            numLearnables = 14;
            layer.Weights = rand(numLearnables,1);
        end

        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z. This method is used during
            % prediction.

             Z = expval(X, layer.Weights);
        end

        function [Z,memory] = forward(layer, X)
            % Z = forward(layer, X) forwards the input data X through the
            % layer and outputs the result Z. This method is used during
            % training.

            Z = expval(X, layer.Weights);
            memory = [];
        end

        function [dLdX,dLdW] = backward(layer,X,~,dLdZ,~)
            % Backward propagate the derivative of the loss
            % function through the layer.
            %
            % Inputs:
            %         layer   - Layer to backward propagate through
            %         X       - Layer input data
            %         Z       - Layer output data
            %         dLdZ    - Derivative of loss with respect to layer
            %                   output
            %         memory  - Memory value from forward function
            % Outputs:
            %         dLdX   - Derivative of loss with respect to layer input
            %         dLdW   - Derivative of loss with respect to
            %                  learnable parameters

            s = pi/4;
            dLdW = zeros(size(layer.Weights),'like',layer.Weights);
            
            for i=1:size(layer.Weights, 1)
                % Parameter-Shift 
                WPlus = layer.Weights;
                WPlus(i) = WPlus(i) + s;
                ZPlus = expval(X, WPlus);

                WMinus = layer.Weights;
                WMinus(i) = WMinus(i) - s;
                ZMinus = expval(X,WMinus);   

                dZdWi = (ZPlus - ZMinus)/(2*sin(s));
                dLdW(i) = sum(dLdZ .* dZdWi, 'all');
            end

            % Make this all zero since this gradient is not going to be
            % used during training.
            dLdX = zeros(size(X), 'like', X);
        end
    end
end

function Z = expval(X, weights)

numSamples = size(X,2);
numQubits = size(X,1); % each qubit encodes a pixel
Z = zeros(1,numSamples,'like',X);
readout = 6; 

% TTN Classifier 
paramGates = [ryGate(1:numQubits, weights(1:numQubits))
              cxGate(1:2:numQubits-1, 2:2:numQubits)
              ryGate([2 3 6 7], weights(numQubits+1:end-2))
              cxGate([2 6], [3 7])
              ryGate([3 6], weights(end-1:end))
              cxGate(3, 6)];

for i = 1:numSamples
    % Encode features and compute expectation value 
    % of Pauli Z operator on the measured qubit
    encodingGates = [ryGate(1:numQubits, 2*X(:,i))];
    mdl = quantumCircuit([encodingGates; paramGates]);
    sv = simulate(mdl);
    Z(i) = probability(sv,readout,"0") - probability(sv,readout,"1");
end
end