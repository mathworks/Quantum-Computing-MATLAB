function [measurementTable, measurement] = loadSavedResults(matFile)
%loadSavedResults Load saved AWS measurement data from a MAT-file.
saved = load(matFile);
if isfield(saved, "awsMeasurement")
    measurement = saved.awsMeasurement;
elseif isfield(saved, "measurement")
    measurement = saved.measurement;
elseif isfield(saved, "awsMeasurements")
    measurement = saved.awsMeasurements;
else
    error("ClinicalTrialSiteSelection:UnsupportedSavedResults", ...
        "Expected %s to contain awsMeasurement, measurement, or awsMeasurements.", ...
        matFile)
end
measurementTable = awsbraket.measurementToTable(measurement);
end
