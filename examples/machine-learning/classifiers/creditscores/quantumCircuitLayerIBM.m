classdef quantumCircuitLayerIBM < nnet.layer.Layer
    properties (Learnable)
        Weights
    end

    properties
        Backend
    end

    methods
        function layer = quantumCircuitLayerIBM(backend, weights)
            arguments
                backend
                weights
            end

            assert(isa(backend, "quantum.backend.QuantumDevice"))

            layer.Backend = backend;
            layer.Weights = weights;
        end

        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z. This method is used during
            % prediction.

            Z = expval(layer, X, layer.Weights);
        end

        function [Z,memory] = forward(layer, X)
            % Z = forward(layer, X) forwards the input data X through the
            % layer and outputs the result Z. This method is used during
            % training.

            Z = expval(layer, X, layer.Weights);
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
                ZPlus = expval(layer, X, WPlus);

                WMinus = layer.Weights;
                WMinus(i) = WMinus(i) - s;
                ZMinus = expval(layer, X,WMinus);

                dZdWi = (ZPlus - ZMinus)/(2*sin(s));
                dLdW(i) = sum(dLdZ .* dZdWi, 'all');
            end

            % Make this all zero since this gradient is not going to be
            % used during training.
            dLdX = zeros(size(X), 'like', X);
        end

        function Z = expval(layer, X, weights)

            if all(X == [1;1;1;1])
                % Don't run the circuit during the validation step of the
                % network
                Z = zeros(1, 'like', X);
                return
            end

            numSamples = size(X,2);
            numQubits = size(X,1);
            Z = zeros(1,numSamples,'like',X);
            readout = 3;

            % TTN Classifier, see Figure 6 [1]
            paramGates = [ryGate(1:numQubits, weights(1:numQubits))
                cxGate([1 4], [2 3])
                ryGate([2 3], weights(numQubits+1:end-1))
                cxGate(2,3)
                ryGate(readout, weights(end))
                ];
            
            % Submit a task for each data sample
            tasks = cell(numSamples, 1);
            for i = 1:numSamples
                encodingGates = [ryGate(1:numQubits, 2*X(:,i))];
                mdl = quantumCircuit([encodingGates; paramGates]);

                task = run(mdl, layer.Backend, NumShots=1000);
                tasks{i} = task;
            end
            
            for i = 1:numSamples
                t = tasks{i};
                wait(t)
                result = fetchOutput(t);
                % Compute expectation value of Pauli Z operator on the measured qubit
                Z(i) = probability(result,readout,"0") - probability(result,readout,"1");
            end
        end
    end
end