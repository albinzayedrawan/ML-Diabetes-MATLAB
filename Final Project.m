%% 
% Display the number of missing values in each column

missingValues = sum(ismissing(diabetes))
varNames = diabetes.Properties.VariableNames;
disp('Missing Values in Each Column:');
disp(table(varNames', missingValues', 'VariableNames', {'Variable', 'MissingValues'}));
%% 
% Numerical columns for scaling

numericalColumns = {'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'}
%% 
% Z-Score scaling to standardize and scale all the numerical features. Scaling 
% this way subtracts the mean and divides by the standard deviation.

diabetes{:, numericalColumns} = zscore(diabetes{:, numericalColumns})
%% 
% Display mean and standard deviation after scaling to verify that the Z-Score 
% scaling worked. We can see that the means are all close to zero and the standard 
% deviation is 1 for all features so the scaling has therefore been successfully 
% applied.

scaledStats = array2table([mean(diabetes{:, numericalColumns}); std(diabetes{:, numericalColumns})], 'VariableNames', numericalColumns);
scaledStats.Properties.RowNames = {'Mean', 'Std'};
disp('Mean and Standard Deviation After Scaling:');
disp(scaledStats);
%% 
% Calculate and display the correlation matrix

correlationMatrix = corr(diabetes{:, numericalColumns});
disp('Correlation Matrix:');
disp(array2table(correlationMatrix, 'VariableNames', numericalColumns, 'RowNames', numericalColumns));
%% 
% Filter out highly correlated features to avoid multicollinearity (there are 
% none at this threshold of 0.7)

corrThreshold = 0.7; 
highlyCorrelated = find(abs(correlationMatrix) > corrThreshold & eye(size(correlationMatrix)) == 0);
disp('Highly Correlated Features:');
disp(highlyCorrelated);
%% 
% Univariate Feature Selection using the 'Minimum Redundancy Maximum Relevance' 
% matlab function to evaluate the relevance of each feature individually with 
% the target variable

featureRanking = fscmrmr(diabetes{:, numericalColumns}, diabetes.Outcome);
disp('Ranked Features using MRMR:');
disp(array2table(featureRanking, 'VariableNames', numericalColumns));
%% 
% Train a decision tree model

treeModel = fitctree(diabetes{:, numericalColumns}, diabetes.Outcome);
%% 
% Display feature importance based on the decision tree model

importance = predictorImportance(treeModel);
disp('Feature Importance from Decision Tree Model:');
disp(array2table(importance, 'VariableNames', numericalColumns));
%% 
% Feature engineering: Create BMI categories

edges = [0 18.5 24.9 29.9 Inf];
labels = {'Underweight', 'Normal Weight', 'Overweight', 'Obese'};
diabetes.BMICategory = categorical(discretize(diabetes.BMI, edges), 1:numel(labels),labels);
diabetes = removevars(diabetes, "BMICategory");
%% 
% Create BMI categories based on percentiles

percentiles = prctile(diabetes.BMI, [0 25 50 75 100]);
labels = {'Underweight', 'Normal Weight', 'Overweight', 'Obese'};
diabetes.BMICategory = categorical(discretize(diabetes.BMI, percentiles), 1:numel(labels), labels);
%% 
% Create age groups based on percentiles

percentiles_age = prctile(diabetes.Age, [0 33.3 66.6 100]);
age_labels = {'Young', 'Adult', 'Senior'};
diabetes.AgeGroup = categorical(discretize(diabetes.Age, percentiles_age), 1:numel(age_labels), age_labels);
%% 
% Display the updated dataset

disp(diabetes);
%% 
% |Data Quality Report and additional measures|

disp('Summary Statistics:');
disp(summary(diabetes));

% Data Quality report for Continuous Features
disp('Data Quality report for Continuous Features:');
dataQualityCont = table();
for i = 1:length(numericalColumns)
    colName = numericalColumns{i};
    colData = diabetes.(colName);
    
    count = numel(colData);
    missing = sum(ismissing(colData));
    cardinality = numel(unique(colData));
    minimum = min(colData);
    firstQuartile = prctile(colData, 25);
    meanVal = mean(colData);
    medianVal = median(colData);
    thirdQuartile = prctile(colData, 75);
    maximum = max(colData);
    stdDev = std(colData);
    
    colStats = table(count, missing, cardinality, minimum, firstQuartile, meanVal, medianVal, thirdQuartile, maximum, stdDev);
    dataQualityCont = [dataQualityCont; colStats];
end
dataQualityCont.Properties.RowNames = numericalColumns;
disp(dataQualityCont);

% Data Quality report for Categorical Features
categoricalColumns = {'BMICategory', 'AgeGroup'};
disp('Data Quality report for Categorical Features:');
dataQualityCat = table();
for i = 1:length(categoricalColumns)
    colName = categoricalColumns{i};
    colData = diabetes.(colName);
    
    count = numel(colData);
    missing = sum(ismissing(colData));
    cardinality = numel(categories(colData));
    
    [modeVal, modeFreq] = mode(colData);
    modePercent = modeFreq / count * 100;
    
    [secondModeVal, secondModeFreq] = mode(colData(colData ~= modeVal));
    secondModePercent = secondModeFreq / count * 100;
    
    colStats = table(count, missing, cardinality, modeVal, modeFreq, modePercent, secondModeVal, secondModeFreq, secondModePercent);
    dataQualityCat = [dataQualityCat; colStats];
end
dataQualityCat.Properties.RowNames = categoricalColumns;
disp(dataQualityCat);

% Histograms for Continuous Features
figure;
for i = 1:length(numericalColumns)
    subplot(3, 3, i);
    histogram(diabetes.(numericalColumns{i}));
    title(['Distribution of ' numericalColumns{i}]);
end

% Boxplots for Continuous Features
figure;
for i = 1:length(numericalColumns)
    subplot(3, 3, i);
    boxplot(diabetes.(numericalColumns{i}));
    title(['Boxplot of ' numericalColumns{i}]);
end

% Histogram of target feature distribution
figure;
outcomeCounts = histcounts(diabetes.Outcome, 'BinEdges', [0 0.5 1.5]);
bar([0, 1], outcomeCounts);
title('Class Distribution');
xlabel('Outcome');
ylabel('Count');
xticks([0, 1]);
xticklabels({'Class 0', 'Class 1'});

% Check for Duplicate Rows
duplicateRows = diabetes(ismember(diabetes, unique(diabetes, 'rows', 'stable', 'rows')), :);
disp('Duplicate Rows:');
disp(duplicateRows);

% Check for Zero Values in Numerical Columns
zeroValues = diabetes(diabetes{:, numericalColumns} == 0, :);
disp('Rows with Zero Values in Numerical Columns:');
disp(zeroValues);

% Missing values recheck
missingValuesAfterProcessing = sum(ismissing(diabetes{:, numericalColumns}));
disp('Missing Values Recheck:');
disp(table(numericalColumns', missingValuesAfterProcessing', 'VariableNames', {'Variable', 'MissingValues'}));
%% 
% Splitting the datasest into training and testing sets (70% training, 30% testing)

% Calculate the number of observations
n = height(diabetes);

% Create a partition for training and testing using cvpartition
part = cvpartition(n, 'HoldOut', 0.3);

% Split the data into training and test sets
idxTrain = training(part);
tblTrain = diabetes(idxTrain, :);

idxTest = test(part);
tblTest = diabetes(idxTest, :);
%% 
% Display the number of observations in the training and test sets

disp(['Number of Observations in Training Set: ' num2str(sum(idxTrain))]);
disp(['Number of Observations in Test Set: ' num2str(sum(idxTest))]);

% DT model
treeModel = fitctree(tblTrain{:, numericalColumns}, tblTrain.Outcome);
yPred = predict(treeModel, tblTest{:, numericalColumns});

% Confusion matrix
confMat = confusionmat(tblTest.Outcome, yPred);

% Calculate precision, recall, and F1 score
precision = confMat(2,2) / sum(confMat(:,2)); % True Positives / (True Positives + False Positives)
recall = confMat(2,2) / sum(confMat(2,:));    % True Positives / (True Positives + False Negatives)
F1Score = 2 * (precision * recall) / (precision + recall);

% Display the results
disp('Confusion Matrix:');
disp(confMat);
disp(['Precision: ' num2str(precision)]);
disp(['Recall: ' num2str(recall)]);
disp(['F1 Score: ' num2str(F1Score)]);
%% 
% Boxplot for Testing Accuracy for five best algorithms

experimentResults = table2array(experimentResults)
boxplot(experimentResults, 'Labels', {'Linear SVM', 'Binary GLM Logistic Regression', 'Efficient Logistic Regression', 'Efficient Linear SVM', 'Coarse Gaussian SVM'})
xlabel("Algorithm")
ylabel("Testing Accuracy")

% Calculate the IQR, range + median for each algorithm for spread analysis
algorithmLabels = {'Linear SVM', 'Binary GLM Logistic Regression', 'Efficient Logistic Regression', 'Efficient Linear SVM', 'Coarse Gaussian SVM'};
iqrValues = iqr(experimentResults);
iqrTable = table(algorithmLabels', iqrValues', 'VariableNames', {'Algorithm', 'IQR'});
disp(iqrTable);

rangeValues = range(experimentResults);
rangeTable = table(algorithmLabels', rangeValues', 'VariableNames', {'Algorithm', 'Range'});
disp(rangeTable);

medians = median(experimentResults);
medianTable = table(algorithmLabels', medians', 'VariableNames', {'Algorithm', 'Median'});
disp(medianTable);
%% 
% Boxplot for Validation Accuracy for five best algorithms

experimentResults2 = table2array(experimentResults2)
boxplot(experimentResults2, 'Labels', {'Linear SVM', 'Binary GLM Logistic Regression', 'Efficient Logistic Regression', 'Efficient Linear SVM', 'Coarse Gaussian SVM'})
xlabel("Algorithm")
ylabel("Validation Accuracy")

% Calculate the IQR, range + median for each algorithm for spread analysis
algorithmLabels = {'Linear SVM', 'Binary GLM Logistic Regression', 'Efficient Logistic Regression', 'Efficient Linear SVM', 'Coarse Gaussian SVM'};
iqrValues = iqr(experimentResults2);
iqrTable = table(algorithmLabels', iqrValues', 'VariableNames', {'Algorithm', 'IQR'});
disp(iqrTable);

rangeValues = range(experimentResults2);
rangeTable = table(algorithmLabels', rangeValues', 'VariableNames', {'Algorithm', 'Range'});
disp(rangeTable);

medians = median(experimentResults2);
medianTable = table(algorithmLabels', medians', 'VariableNames', {'Algorithm', 'Median'});
disp(medianTable);
%% 
% Boxplot for Prediction Speed for five best algorithms

experimentResults3 = table2array(experimentResults3)
boxplot(experimentResults3, 'Labels', {'Linear SVM', 'Binary GLM Logistic Regression', 'Efficient Logistic Regression', 'Efficient Linear SVM', 'Coarse Gaussian SVM'})
xlabel("Algorithm")
ylabel("Prediction Speed")

% Calculate the IQR, range + median for each algorithm for spread analysis
algorithmLabels = {'Linear SVM', 'Binary GLM Logistic Regression', 'Efficient Logistic Regression', 'Efficient Linear SVM', 'Coarse Gaussian SVM'};
iqrValues = iqr(experimentResults3);
iqrTable = table(algorithmLabels', iqrValues', 'VariableNames', {'Algorithm', 'IQR'});
disp(iqrTable);

rangeValues = range(experimentResults3);
rangeTable = table(algorithmLabels', rangeValues', 'VariableNames', {'Algorithm', 'Range'});
disp(rangeTable);

medians = median(experimentResults3);
medianTable = table(algorithmLabels', medians', 'VariableNames', {'Algorithm', 'Median'});
disp(medianTable);
%% 
% Boxplot for Training Time for five best algorithms

experimentResults4 = table2array(experimentResults4)
boxplot(experimentResults4, 'Labels', {'Linear SVM', 'Binary GLM Logistic Regression', 'Efficient Logistic Regression', 'Efficient Linear SVM', 'Coarse Gaussian SVM'})
xlabel("Algorithm")
ylabel("Training Time")

% Calculate the IQR, range + median for each algorithm for spread analysis
algorithmLabels = {'Linear SVM', 'Binary GLM Logistic Regression', 'Efficient Logistic Regression', 'Efficient Linear SVM', 'Coarse Gaussian SVM'};
iqrValues = iqr(experimentResults4);
iqrTable = table(algorithmLabels', iqrValues', 'VariableNames', {'Algorithm', 'IQR'});
disp(iqrTable);

rangeValues = range(experimentResults4);
rangeTable = table(algorithmLabels', rangeValues', 'VariableNames', {'Algorithm', 'Range'});
disp(rangeTable);

medians = median(experimentResults4);
medianTable = table(algorithmLabels', medians', 'VariableNames', {'Algorithm', 'Median'});
disp(medianTable);
%% 
% Bargraph for Testing Accuracies for all algorithms (before narrowed down top 
% 5)

algorithms = experimentResults5.Algorithm;
testingAccuracyMean = experimentResults5.TestingAccuracyMean;

figure;
bar([testingAccuracyMean]);
xlabel('Algorithm');
ylabel('Testing Accuracy Mean');
title('Testing Accuracy Mean for 24 Different Algorithms');

xticks(1:length(algorithms));
xticklabels(algorithms);
xtickangle(45);