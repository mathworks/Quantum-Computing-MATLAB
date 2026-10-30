function measurementTable = measurementToTable(measurement)
%measurementToTable Convert quantum measurement results to a scoring table.
if istable(measurement)
    measurementTable = measurement(:, ["State","Count","Probability"]);
else
    State = string(measurement.MeasuredStates(:));
    Count = double(measurement.Counts(:));
    Probability = double(measurement.Probabilities(:));
    measurementTable = table(State, Count, Probability);
end

if ~isempty(measurementTable)
    measurementTable = sortrows(measurementTable, "Probability", "descend");
end
end
