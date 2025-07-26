% predict_plant_growth_rf_bayesopt.m
% This script provides a high-performance approach to plant growth prediction using Random Forest.
% Key features include:
% 1. Data Preprocessing: Converts descriptive ranges into numerical midpoints.
% 2. Automated Hyperparameter Tuning: Utilizes Bayesian Optimization to efficiently find the
%    optimal 'NumTrees', 'MinLeafSize', 'MaxNumSplits', and 'NumVariablesToSample' for each target variable 
%    by minimizing cross-validation error.
% 3. Feature Importance Analysis: Calculates and displays which environmental factors are most influential.
% 4. Final Model Training: Trains robust Random Forest models using the optimized hyperparameters on the full dataset.
% 5. Prediction: Generates final predictions on the unseen test data.
% 6. Performance Evaluation: Displays model performance metrics and validation results.
% 7. Output: Displays tuning results, feature importance, and final predictions.

% --- Initialization ---
clear;
clc;
close all;
fprintf('=== Plant Growth Prediction using Random Forest with Bayesian Optimization ===\n');
fprintf('Starting model training and optimization process...\n\n');

% --- Helper Function: Convert Range String to Numerical Midpoint ---
function num_val = convertRangeToMidpoint(range_str)
    % Remove units and extra characters
    clean_str = regexprep(range_str, {'%', 'RH', 'lux', 'ppm', '\s+'}, '');
    parts = strsplit(clean_str, '-');
    
    if length(parts) == 2
        start_val = str2double(parts{1});
        end_val = str2double(parts{2});
        if ~isnan(start_val) && ~isnan(end_val)
            num_val = (start_val + end_val) / 2;
        else
            num_val = NaN;
        end
    elseif length(parts) == 1
        num_val = str2double(parts{1});
    else
        num_val = NaN;
    end
end

% --- 1. Load and Preprocess Data ---
fprintf('1. Loading and preprocessing data...\n');
try
    T_train = readtable('datatraining.xlsx', 'TextType', 'string');
    T_test = readtable('datatest.xlsx', 'TextType', 'string');
    fprintf('   ✓ Data files loaded successfully\n');
    fprintf('   Training data size: %d rows, %d columns\n', height(T_train), width(T_train));
    fprintf('   Test data size: %d rows, %d columns\n', height(T_test), width(T_test));
catch ME
    fprintf('   ✗ Error loading data files: %s\n', ME.message);
    return;
end

% Define columns to convert
colsToConvert = {'Moisture', 'Humidity', 'LightIntensity', 'CO2Concentration'};
fprintf('\n2. Converting feature columns to numerical values...\n');

for i = 1:length(colsToConvert)
    colName = colsToConvert{i};
    if ismember(colName, T_train.Properties.VariableNames)
        T_train.(colName) = cellfun(@convertRangeToMidpoint, cellstr(T_train.(colName)));
        T_test.(colName) = cellfun(@convertRangeToMidpoint, cellstr(T_test.(colName)));
        fprintf('   ✓ Column "%s" converted successfully\n', colName);
    else
        fprintf('   ⚠ Warning: Column "%s" not found in data\n', colName);
    end
end

% Check for missing values
missing_train = sum(isnan(T_train{:, colsToConvert}), 'all');
missing_test = sum(isnan(T_test{:, colsToConvert}), 'all');
if missing_train > 0 || missing_test > 0
    fprintf('   ⚠ Warning: Found %d missing values in training data, %d in test data\n', missing_train, missing_test);
end

% --- 2. Define Features (X) and Targets (y) ---
fprintf('\n3. Preparing feature matrices and target variables...\n');
X_train = T_train{:, colsToConvert};
X_test = T_test{:, colsToConvert};
featureNames = colsToConvert;

% Extract target variables
targetVars = {'LeafLength', 'LeafArea', 'StemGirth'};
for i = 1:length(targetVars)
    if ~ismember(targetVars{i}, T_train.Properties.VariableNames)
        fprintf('   ✗ Error: Target variable "%s" not found in training data\n', targetVars{i});
        return;
    end
end

y_train_LeafLength = T_train.LeafLength;
y_train_LeafArea = T_train.LeafArea;
y_train_StemGirth = T_train.StemGirth;

fprintf('   ✓ Feature matrix dimensions: %dx%d\n', size(X_train));
fprintf('   ✓ Test matrix dimensions: %dx%d\n', size(X_test));
fprintf('   ✓ Target variables prepared: %s\n', strjoin(targetVars, ', '));

% --- 3. Define Hyperparameter Optimization ---
fprintf('\n4. Setting up Bayesian Optimization for hyperparameter tuning...\n');

% Define optimizable variables for Random Forest
vars = [optimizableVariable('NumTrees', [10, 200], 'Type', 'integer');
        optimizableVariable('MinLeafSize', [1, 50], 'Type', 'integer');
        optimizableVariable('MaxNumSplits', [1, 100], 'Type', 'integer');
        optimizableVariable('NumVariablesToSample', [1, length(featureNames)], 'Type', 'integer')];

% Helper function for hyperparameter optimization
function results = optimizeRandomForest(X, y, vars, targetName)
    fprintf('   Optimizing hyperparameters for %s...\n', targetName);
    
    % Objective function using cross-validation
    objectiveFcn = @(params) crossValObjective(X, y, params);
    
    % Run Bayesian optimization
    results = bayesopt(objectiveFcn, vars, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'MaxObjectiveEvaluations', 50, ...
        'Verbose', 0, ...
        'PlotFcn', []);
end

% Cross-validation objective function
function loss = crossValObjective(X, y, params)
    % Create Random Forest with current parameters
    rf = TreeBagger(params.NumTrees, X, y, ...
        'Method', 'regression', ...
        'MinLeafSize', params.MinLeafSize, ...
        'MaxNumSplits', params.MaxNumSplits, ...
        'NumVariablesToSample', params.NumVariablesToSample, ...
        'OOBPrediction', 'on');
    
    % Use Out-of-Bag error as the loss metric
    loss = oobError(rf, 'Mode', 'ensemble');
    loss = loss(end); % Get the final OOB error
end

% --- 4. Hyperparameter Tuning for Each Target Variable ---
fprintf('\n5. Performing hyperparameter optimization...\n');

% Optimize for LeafLength
results_LL = optimizeRandomForest(X_train, y_train_LeafLength, vars, 'LeafLength');
bestParams_LL = results_LL.XAtMinObjective;

% Optimize for LeafArea
results_LA = optimizeRandomForest(X_train, y_train_LeafArea, vars, 'LeafArea');
bestParams_LA = results_LA.XAtMinObjective;

% Optimize for StemGirth
results_SG = optimizeRandomForest(X_train, y_train_StemGirth, vars, 'StemGirth');
bestParams_SG = results_SG.XAtMinObjective;

% Display optimization results
fprintf('\n6. Hyperparameter optimization results:\n');
fprintf('   LeafLength - NumTrees: %d, MinLeafSize: %d, MaxNumSplits: %d, NumVariablesToSample: %d (OOB Loss: %.4f)\n', ...
    bestParams_LL.NumTrees, bestParams_LL.MinLeafSize, bestParams_LL.MaxNumSplits, bestParams_LL.NumVariablesToSample, results_LL.MinObjective);
fprintf('   LeafArea   - NumTrees: %d, MinLeafSize: %d, MaxNumSplits: %d, NumVariablesToSample: %d (OOB Loss: %.4f)\n', ...
    bestParams_LA.NumTrees, bestParams_LA.MinLeafSize, bestParams_LA.MaxNumSplits, bestParams_LA.NumVariablesToSample, results_LA.MinObjective);
fprintf('   StemGirth  - NumTrees: %d, MinLeafSize: %d, MaxNumSplits: %d, NumVariablesToSample: %d (OOB Loss: %.4f)\n', ...
    bestParams_SG.NumTrees, bestParams_SG.MinLeafSize, bestParams_SG.MaxNumSplits, bestParams_SG.NumVariablesToSample, results_SG.MinObjective);

% --- 5. Train Final Models with Optimal Hyperparameters ---
fprintf('\n7. Training final Random Forest models with optimal hyperparameters...\n');

% Helper function to train final model and calculate feature importance
function [model, importance] = trainFinalModel(X, y, params, featureNames, targetName)
    % Train Random Forest with optimal parameters
    model = TreeBagger(params.NumTrees, X, y, ...
        'Method', 'regression', ...
        'MinLeafSize', params.MinLeafSize, ...
        'MaxNumSplits', params.MaxNumSplits, ...
        'NumVariablesToSample', params.NumVariablesToSample, ...
        'OOBPrediction', 'on', ...
        'OOBPredictorImportance', 'on');
    
    % Get feature importance from Random Forest
    importance = model.OOBPermutedPredictorDeltaError;
    
    % Normalize importance scores to percentages
    if sum(importance) > 0
        importance = importance / sum(importance) * 100;
    end
    
    % Display feature importance
    fprintf('   Feature importance for %s:\n', targetName);
    [sorted_importance, idx] = sort(importance, 'descend');
    for i = 1:length(featureNames)
        fprintf('     %s: %.2f%%\n', featureNames{idx(i)}, sorted_importance(i));
    end
    fprintf('   Out-of-Bag Error: %.4f\n', oobError(model, 'Mode', 'ensemble'));
    fprintf('\n');
end

% Train final models
[model_LeafLength, importance_LL] = trainFinalModel(X_train, y_train_LeafLength, bestParams_LL, featureNames, 'LeafLength');
[model_LeafArea, importance_LA] = trainFinalModel(X_train, y_train_LeafArea, bestParams_LA, featureNames, 'LeafArea');
[model_StemGirth, importance_SG] = trainFinalModel(X_train, y_train_StemGirth, bestParams_SG, featureNames, 'StemGirth');

% --- 6. Model Performance Evaluation ---
fprintf('8. Evaluating model performance on training data...\n');

function evaluateModel(model, X, y, targetName)
    predictions = predict(model, X);
    predictions = str2double(predictions); % Convert cell array to numeric
    
    mse = mean((y - predictions).^2);
    rmse = sqrt(mse);
    mae = mean(abs(y - predictions));
    r_squared = 1 - sum((y - predictions).^2) / sum((y - mean(y)).^2);
    
    fprintf('   %s Performance:\n', targetName);
    fprintf('     RMSE: %.4f\n', rmse);
    fprintf('     MAE:  %.4f\n', mae);
    fprintf('     R²:   %.4f\n', r_squared);
    fprintf('\n');
end

evaluateModel(model_LeafLength, X_train, y_train_LeafLength, 'LeafLength');
evaluateModel(model_LeafArea, X_train, y_train_LeafArea, 'LeafArea');
evaluateModel(model_StemGirth, X_train, y_train_StemGirth, 'StemGirth');

% --- 7. Generate Final Predictions ---
fprintf('9. Generating predictions for test data...\n');

predictions_LeafLength = predict(model_LeafLength, X_test);
predictions_LeafArea = predict(model_LeafArea, X_test);
predictions_StemGirth = predict(model_StemGirth, X_test);

% Convert cell arrays to numeric arrays
predictions_LeafLength = str2double(predictions_LeafLength);
predictions_LeafArea = str2double(predictions_LeafArea);
predictions_StemGirth = str2double(predictions_StemGirth);

fprintf('   ✓ Predictions generated for %d test samples\n', length(predictions_LeafLength));

% --- 8. Display and Save Results ---
fprintf('\n10. Final Results:\n');

% Create results table
resultsTable = table(predictions_LeafLength, predictions_LeafArea, predictions_StemGirth, ...
    'VariableNames', {'PredictedLeafLength', 'PredictedLeafArea', 'PredictedStemGirth'});

% Display results
fprintf('=== Final Predictions (Random Forest with Bayesian Optimization) ===\n');
disp(resultsTable);

% Optional: Save results to CSV file
try
    writetable(resultsTable, 'plant_growth_predictions_rf_bayesopt.csv');
    fprintf('\n   ✓ Results saved to "plant_growth_predictions_rf_bayesopt.csv"\n');
catch
    fprintf('\n   ⚠ Could not save results to CSV file\n');
end

% --- 9. Additional Random Forest Analysis ---
fprintf('\n=== Random Forest Model Analysis ===\n');

% Display OOB Error evolution for each model
fprintf('Out-of-Bag Error Analysis:\n');
fprintf('   LeafLength model - Final OOB Error: %.4f\n', oobError(model_LeafLength, 'Mode', 'ensemble'));
fprintf('   LeafArea model   - Final OOB Error: %.4f\n', oobError(model_LeafArea, 'Mode', 'ensemble'));
fprintf('   StemGirth model  - Final OOB Error: %.4f\n', oobError(model_StemGirth, 'Mode', 'ensemble'));

% --- 10. Summary ---
fprintf('\n=== Model Training Summary ===\n');
fprintf('Algorithm: Random Forest with Bayesian Optimization\n');
fprintf('Features used: %s\n', strjoin(featureNames, ', '));
fprintf('Target variables: %s\n', strjoin(targetVars, ', '));
fprintf('Optimization method: Bayesian Optimization with Out-of-Bag Error\n');
fprintf('Number of test predictions: %d\n', height(resultsTable));
fprintf('Random Forest advantages:\n');
fprintf('  • Reduced overfitting compared to single Decision Tree\n');
fprintf('  • Built-in feature importance calculation\n');
fprintf('  • Out-of-Bag error estimation without separate validation set\n');
fprintf('  • Better generalization through ensemble learning\n');
fprintf('\nProcess completed successfully!\n');