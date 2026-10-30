function aws = ionqForte1()
%ionqForte1 Configuration profile for IonQ Forte 1 on AWS Braket.
aws.DeviceName = "Forte 1";
aws.Region = "us-east-1";
aws.prepareCircuit = @(circuit) makeBoundedAngleCircuit(circuit, pi/2);
end
function boundedCircuit = makeBoundedAngleCircuit(circuit, maxAbsAngle)
%makeBoundedAngleCircuit Split rotation gates to satisfy backend angle limits.
    originalGates = circuit.Gates;
    boundedGates = quantum.gate.SimpleGate.empty(0,1);
    for k = 1:numel(originalGates)
        gateType = string(originalGates(k).Type);
        if ismember(gateType, ["rx","rz","rzz"]) && ~isempty(originalGates(k).Angles)
            theta = mod(originalGates(k).Angles(1) + pi, 2*pi) - pi;
            numPieces = max(1, ceil(abs(theta)/maxAbsAngle));
            thetaPiece = theta/numPieces;
            for piece = 1:numPieces
                switch gateType
                    case "rx"
                        boundedGates(end+1,1) = rxGate(originalGates(k).TargetQubits, thetaPiece);
                    case "rz"
                        boundedGates(end+1,1) = rzGate(originalGates(k).TargetQubits, thetaPiece);
                    case "rzz"
                        targetQubits = originalGates(k).TargetQubits;
                        boundedGates(end+1,1) = rzzGate(targetQubits(1), targetQubits(2), thetaPiece);
                end
            end
        else
            boundedGates(end+1,1) = originalGates(k);
        end
    end
    boundedCircuit = quantumCircuit(boundedGates, circuit.NumQubits);
end
