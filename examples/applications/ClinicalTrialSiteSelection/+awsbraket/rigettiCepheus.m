function aws = rigettiCepheus()
%rigettiCepheus Configuration profile for Rigetti Cepheus on AWS Braket.
aws.DeviceName = "Cepheus-1-108Q";
aws.Region = "us-west-1";
aws.prepareCircuit = @(circuit) circuit;
end
