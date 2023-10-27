function [X,Y] = processCreditData(data)
% Ratings {A,AA,AAA,B} are labeled as better and {BB,BBB,C} are labeled
% as worse. Use 4 features to classify a sample as worse or better: 

% (RE_TA)    - Retained Earnings / Total Assets
% (EBIT_TA)  - Earnings Before Interests and Taxes / Total Assets 
% (MVE_BVTD) - Market Value of Equity / Book Value of Total Debt 
% (S_TA)     - Sales / Total Assets 

arguments
    data table
end

vars = data.Properties.VariableNames;

% Choose features and rescale them in the range [0 pi/2]
features = vars(3:6);
data = normalize(data,"range", [0 pi/2], "DataVariables",features);

numSamples = size(data, 1);
numFeatures = length(features);

hasLabels = ismember('Rating', vars);
if hasLabels
    Y = categorical(repmat("worse", [numSamples 1]));
else
    % Unused default output when data doesn't have labels
    Y = missing;
end

X = zeros(numSamples, numFeatures);
for ii  = 1:numSamples
    X(ii,:) = data{ii, features};
    if hasLabels
        if ismember(data.Rating(ii), {'A','AA','AAA','B'})
            Y(ii) = categorical("better");
        end
    end
end
end