classdef PQCLayer < nnet.layer.Layer
    % Custom PQC layer example.

    properties (Learnable)
        % Define layer learnable parameters.
        A
        B
    end

    methods
        function layer = PQCLayer
            % Set layer name.
            layer.Name = "PQC";

            % Set layer description.
            layer.Description = "Layer containing a parameterized " + ...
                "quantum circuit (PQC)";

            % Initialize learnable parameter.
            layer.A = 1;
            layer.B = 2;
        end

        function Z = predict(layer,X)
            % Z = predict(layer,X) forwards the input data X through the
            % layer and outputs the result Z at prediction time.
            Z = computeZ(X,layer.A,layer.B);
        end

        function [dLdX,dLdA,dLdB] = backward(layer,X,Z,dLdZ,memory)
            % Backpropagate the derivative of the loss
            % function through the layer.
            %
            % Inputs:
            %     layer   - Layer though which data backpropagates
            %     X       - Layer input data
            %     Z       - Layer output data
            %     dLdZ    - Derivative of loss with respect to layer
            %               output
            %     memory  - Memory value from forward function
            % Outputs:
            %     dLdX   - Derivative of loss with respect to layer input
            %     dLdA   - Derivative of loss with respect to learnable
            %              parameter A
            %     dLdB   - Derivative of loss with respect to learnable
            %              parameter B

            s = pi/4;
            ZPlus = computeZ(X,layer.A + s,layer.B);
            ZMinus = computeZ(X,layer.A - s,layer.B);
            dZdA = X(1,:).*((ZPlus - ZMinus)./(2*sin(X(1,:).*s)));
            dLdA = sum(dLdZ.*dZdA,"all");

            ZPlus = computeZ(X,layer.A,layer.B + s);
            ZMinus = computeZ(X,layer.A,layer.B - s);
            dZdB = X(2,:).*(((ZPlus - ZMinus)./(2*sin(X(2,:).*s))));
            dLdB = sum(dLdZ.*dZdB,"all");

            % Set the gradients with respect to x and y to zero
            % because the QNN does not use them during training.
            dLdX = zeros(size(X),"like",X);
        end
    end
end

function Z = computeZ(X, A, B)
    numSamples = size(X,2);
    x1 = X(1,:);
    x2 = X(2,:);
    Z = zeros(1,numSamples,"like",X);
    for i = 1:numSamples
        circ = quantumCircuit(2);
        circ.Gates = [rxGate(1,x1(i)*A); rxGate(2,x2(i)*B); cxGate(1,2)];
        s = simulate(circ);
        Z(i) = probability(s,2,"0") - probability(s,2,"1");
    end
end