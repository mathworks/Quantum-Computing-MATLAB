function aws = sv1()
%sv1 Configuration profile for AWS Braket SV1 simulation.
aws.DeviceName = "sv1";
aws.Region = "us-west-1";
aws.prepareCircuit = @(circuit) circuit;
end
