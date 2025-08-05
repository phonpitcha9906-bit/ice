% predict_plant_growth_rf_alternative.m
% This script provides a high-performance approach to plant growth prediction using a Random Forest model.
% Key features include:
% 1. Data Preprocessing: Converts descriptive ranges into numerical midpoints.
% 2. Automated Hyperparameter Tuning: Utilizes Grid Search to find the
%    optimal 'MinLeafSize', 'NumPredictorsToSample', and 'NumTrees' for each target variable 
%    by minimizing the Out-of-Bag (OOB) error.
% 3. Feature Importance Analysis: Calculates and displays which environmental factors are most influential.
% 4. Final Model Training: Trains robust Random Forest models using the optimized hyperparameters.
% 5. Prediction: Generates final predictions on the unseen test data.
% 6. Performance Evaluation: Displays model performance metrics and validation results.
% 7. Output: Displays tuning results, feature importance, and final predictions.

% --- Initialization ---
clear;
clc;
close all;
fprintf('=== Plant Growth Prediction using Random Forest with Grid Search Optimization ===\n');
fprintf('Starting model training and optimization process...\n\n');

% --- Helper Function: Convert Range String to Numerical Midpoint ---
function num_val = convertRangeToMidpoint(range_str)
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

colsToConvert = {'Moisture', 'Humidity', 'LightIntensity', 'CO2Concentration'};
fprintf('\n2. Converting feature columns to numerical values...\n');
for i = 1:length(colsToConvert)
    colName = colsToConvert{i};
    T_train.(colName) = cellfun(@convertRangeToMidpoint, cellstr(T_train.(colName)));
    T_test.(colName) = cellfun(@convertRangeToMidpoint, cellstr(T_test.(colName)));
    fprintf('   ✓ Column "%s" converted successfully\n', colName);
end

% --- 2. Define Features (X) and Targets (y) ---
fprintf('\n3. Preparing feature matrices and target variables...\n');
X_train = T_train{:, colsToConvert};
X_test = T_test{:, colsToConvert};
featureNames = colsToConvert;
targetVars = {'LeafLength', 'LeafArea', 'StemGirth'};
y_train_LeafLength = T_train.LeafLength;
y_train_LeafArea = T_train.LeafArea;
y_train_StemGirth = T_train.StemGirth;
fprintf('   ✓ Feature matrix dimensions: %dx%d\n', size(X_train));
fprintf('   ✓ Target variables prepared: %s\n', strjoin(targetVars, ', '));

% --- 3. Define Hyperparameter Grid Search ---
fprintf('\n4. Setting up Grid Search for hyperparameter tuning...\n');
% Define parameter grids
minLeafSizes = [1, 5, 10, 15, 20];
numPredictorsToSample = [1, 2, 3, 4]; % Up to number of features
numTrees = [50, 100, 150, 200];

function [bestParams, bestError] = gridSearchRandomForest(X, y, minLeafSizes, numPredictorsToSample, numTrees, targetName)
    fprintf('   Optimizing hyperparameters for %s using Grid Search...\n', targetName);
    
    bestError = inf;
    bestParams = struct();
    totalCombinations = length(minLeafSizes) * length(numPredictorsToSample) * length(numTrees);
    currentCombination = 0;
    
    for mls = minLeafSizes
        for npts = numPredictorsToSample
            for nt = numTrees
                currentCombination = currentCombination + 1;
                
                % Train model with current parameters
                try
                    model = TreeBagger(nt, X, y, ...
                        'Method', 'regression', ...
                        'MinLeafSize', mls, ...
                        'NumPredictorsToSample', npts, ...
                        'OOBPrediction', 'on');
                    
                    % Calculate OOB error
                    oobErr = oobError(model, 'Mode', 'ensemble');
                    
                    % Update best parameters if this is better
                    if oobErr < bestError
                        bestError = oobErr;
                        bestParams.MinLeafSize = mls;
                        bestParams.NumPredictorsToSample = npts;
                        bestParams.NumTrees = nt;
                    end
                    
                    % Progress indicator
                    if mod(currentCombination, 10) == 0 || currentCombination == totalCombinations
                        fprintf('     Progress: %d/%d combinations tested\n', currentCombination, totalCombinations);
                    end
                    
                catch ME
                    fprintf('     Warning: Error with parameters [%d, %d, %d]: %s\n', mls, npts, nt, ME.message);
                end
            end
        end
    end
    
    fprintf('     Best parameters found - MinLeafSize: %d, NumPredictorsToSample: %d, NumTrees: %d\n', ...
        bestParams.MinLeafSize, bestParams.NumPredictorsToSample, bestParams.NumTrees);
    fprintf('     Best OOB Error: %.4f\n', bestError);
end

% --- 4. Hyperparameter Tuning for Each Target Variable ---
fprintf('\n5. Performing hyperparameter optimization...\n');
[bestParams_LL, bestError_LL] = gridSearchRandomForest(X_train, y_train_LeafLength, minLeafSizes, numPredictorsToSample, numTrees, 'LeafLength');
[bestParams_LA, bestError_LA] = gridSearchRandomForest(X_train, y_train_LeafArea, minLeafSizes, numPredictorsToSample, numTrees, 'LeafArea');
[bestParams_SG, bestError_SG] = gridSearchRandomForest(X_train, y_train_StemGirth, minLeafSizes, numPredictorsToSample, numTrees, 'StemGirth');

fprintf('\n6. Hyperparameter optimization results:\n');
fprintf('   LeafLength - MinLeafSize: %d, NumPredictorsToSample: %d, NumTrees: %d (OOB Error: %.4f)\n', ...
    bestParams_LL.MinLeafSize, bestParams_LL.NumPredictorsToSample, bestParams_LL.NumTrees, bestError_LL);
fprintf('   LeafArea   - MinLeafSize: %d, NumPredictorsToSample: %d, NumTrees: %d (OOB Error: %.4f)\n', ...
    bestParams_LA.MinLeafSize, bestParams_LA.NumPredictorsToSample, bestParams_LA.NumTrees, bestError_LA);
fprintf('   StemGirth  - MinLeafSize: %d, NumPredictorsToSample: %d, NumTrees: %d (OOB Error: %.4f)\n', ...
    bestParams_SG.MinLeafSize, bestParams_SG.NumPredictorsToSample, bestParams_SG.NumTrees, bestError_SG);

% --- 5. Train Final Models with Optimal Hyperparameters ---
fprintf('\n7. Training final Random Forest models with optimal hyperparameters...\n');
function [model, importance] = trainFinalModel(X, y, params, featureNames, targetName)
    model = TreeBagger(params.NumTrees, X, y, ...
        'Method', 'regression', ...
        'MinLeafSize', params.MinLeafSize, ...
        'NumPredictorsToSample', params.NumPredictorsToSample, ...
        'OOBPredictorImportance', 'on', ...
        'PredictorNames', featureNames);
    
    % Use built-in permutation importance
    importance = model.OOBPermutedPredictorDeltaError;
    if sum(importance) > 0
        importance = importance / sum(importance) * 100;
    end
    
    fprintf('   Feature importance for %s:\n', targetName);
    [sorted_importance, idx] = sort(importance, 'descend');
    for i = 1:length(featureNames)
        fprintf('     %s: %.2f%%\n', featureNames{idx(i)}, sorted_importance(i));
    end
    fprintf('\n');
end

[model_LeafLength, ~] = trainFinalModel(X_train, y_train_LeafLength, bestParams_LL, featureNames, 'LeafLength');
[model_LeafArea, ~] = trainFinalModel(X_train, y_train_LeafArea, bestParams_LA, featureNames, 'LeafArea');
[model_StemGirth, ~] = trainFinalModel(X_train, y_train_StemGirth, bestParams_SG, featureNames, 'StemGirth');

% --- 6. Model Performance Evaluation ---
fprintf('8. Evaluating model performance on training data...\n');
function evaluateModel(model, X, y, targetName)
    predictions = predict(model, X);
    mse = mean((y - predictions).^2);
    rmse = sqrt(mse);
    mae = mean(abs(y - predictions));
    r_squared = 1 - sum((y - predictions).^2) / sum((y - mean(y)).^2);
    
    fprintf('   %s Performance:\n', targetName);
    fprintf('     RMSE: %.4f\n', rmse);
    fprintf('     MAE:  %.4f\n', mae);
    fprintf('     R²:   %.4f\n\n', r_squared);
end

evaluateModel(model_LeafLength, X_train, y_train_LeafLength, 'LeafLength');
evaluateModel(model_LeafArea, X_train, y_train_LeafArea, 'LeafArea');
evaluateModel(model_StemGirth, X_train, y_train_StemGirth, 'StemGirth');

% --- 7. Generate Final Predictions ---
fprintf('9. Generating predictions for test data...\n');
predictions_LeafLength = predict(model_LeafLength, X_test);
predictions_LeafArea = predict(model_LeafArea, X_test);
predictions_StemGirth = predict(model_StemGirth, X_test);
fprintf('   ✓ Predictions generated for %d test samples\n', length(predictions_LeafLength));

% --- Load actual data for classification evaluation ---
fprintf('\n--- Loading actual data for Classification Model Evaluation ---\n');
actual_file = 'datatest.xlsx'; % Assuming 'datatest.xlsx' contains the actual values for the test set
try
    actual_data_table = readtable(actual_file);
    actual_data.LeafLength = actual_data_table.LeafLength;
    actual_data.LeafArea = actual_data_table.LeafArea;
    actual_data.StemGirth = actual_data_table.StemGirth;
    fprintf('   ✓ Actual test data loaded successfully from "%s"\n', actual_file);
catch ME
    fprintf('   ✗ Error loading actual data file: %s\n', ME.message);
    % Continue without classification evaluation if file doesn't exist
    actual_data = [];
end

% --- Classification Model Evaluation (F1-score, Precision, Recall, Accuracy) ---
if ~isempty(actual_data)
    fprintf('\n--- Classification Model Evaluation (F1-score, Precision, Recall, Accuracy) ---\n');
    
    % Define thresholds - adjust these based on your data analysis
    threshold_leaf_length = 12.0;  % Adjust based on your data distribution
    threshold_leaf_area = 16.5;    % Adjust based on your data distribution
    threshold_stem_girth = 2.5;    % Adjust based on your data distribution
    
    % --- Leaf Length Classification ---
    fprintf('\n--- Leaf Length Classification ---\n');
    actual_class_leaf = (actual_data.LeafLength > threshold_leaf_length);
    pred_class_leaf = (predictions_LeafLength > threshold_leaf_length);
    [cm_leaf, f1_leaf, precision_leaf, recall_leaf, accuracy_leaf] = ...
        calculate_classification_metrics(actual_class_leaf, pred_class_leaf);
    disp('Confusion Matrix for Leaf Length (Actual Rows, Predicted Cols):');
    disp(cm_leaf);
    fprintf('  Accuracy:  %.2f%%\n', accuracy_leaf * 100);
    fprintf('  Precision: %.2f%%\n', precision_leaf * 100);
    fprintf('  Recall:    %.2f%%\n', recall_leaf * 100);
    fprintf('  F1-Score:  %.2f%%\n', f1_leaf * 100);
    
    % --- Leaf Area Classification ---
    fprintf('\n--- Leaf Area Classification ---\n');
    actual_class_area = (actual_data.LeafArea > threshold_leaf_area);
    pred_class_area = (predictions_LeafArea > threshold_leaf_area);
    [cm_area, f1_area, precision_area, recall_area, accuracy_area] = ...
        calculate_classification_metrics(actual_class_area, pred_class_area);
    disp('Confusion Matrix for Leaf Area (Actual Rows, Predicted Cols):');
    disp(cm_area);
    fprintf('  Accuracy:  %.2f%%\n', accuracy_area * 100);
    fprintf('  Precision: %.2f%%\n', precision_area * 100);
    fprintf('  Recall:    %.2f%%\n', recall_area * 100);
    fprintf('  F1-Score:  %.2f%%\n', f1_area * 100);
    
    % --- Stem Girth Classification ---
    fprintf('\n--- Stem Girth Classification ---\n');
    actual_class_stem = (actual_data.StemGirth > threshold_stem_girth);
    pred_class_stem = (predictions_StemGirth > threshold_stem_girth);
    [cm_stem, f1_stem, precision_stem, recall_stem, accuracy_stem] = ...
        calculate_classification_metrics(actual_class_stem, pred_class_stem);
    disp('Confusion Matrix for Stem Girth (Actual Rows, Predicted Cols):');
    disp(cm_stem);
    fprintf('  Accuracy:  %.2f%%\n', accuracy_stem * 100);
    fprintf('  Precision: %.2f%%\n', precision_stem * 100);
    fprintf('  Recall:    %.2f%%\n', recall_stem * 100);
    fprintf('  F1-Score:  %.2f%%\n', f1_stem * 100);
end

% --- 8. Display and Save Results ---
fprintf('\n10. Final Results:\n');
resultsTable = table(predictions_LeafLength, predictions_LeafArea, predictions_StemGirth, ...
    'VariableNames', {'PredictedLeafLength', 'PredictedLeafArea', 'PredictedStemGirth'});
fprintf('=== Final Predictions (Random Forest with Grid Search Optimization) ===\n');
disp(resultsTable);

try
    writetable(resultsTable, 'plant_growth_predictions_rf_gridsearch.csv');
    fprintf('\n   ✓ Results saved to "plant_growth_predictions_rf_gridsearch.csv"\n');
catch
    fprintf('\n   ⚠ Could not save results to CSV file\n');
end

% --- 9. Summary ---
fprintf('\n=== Model Training Summary ===\n');
fprintf('Algorithm: Random Forest with Grid Search Optimization\n');
fprintf('Features used: %s\n', strjoin(featureNames, ', '));
fprintf('Target variables: %s\n', strjoin(targetVars, ', '));
fprintf('Optimization method: Grid Search using Out-of-Bag Error\n');
fprintf('Number of test predictions: %d\n', height(resultsTable));
fprintf('\nProcess completed successfully!\n');

% =========================================================================================
% Local Function: calculate_classification_metrics
% Function to calculate Confusion Matrix and Classification Metrics
% =========================================================================================
function [cm, f1, precision, recall, accuracy] = calculate_classification_metrics(actual_labels, predicted_labels)
    % Ensure inputs are logical or 0/1 numeric
    actual_labels = logical(actual_labels);
    predicted_labels = logical(predicted_labels);
    
    % Calculate Confusion Matrix components
    TP = sum(actual_labels & predicted_labels);      % True Positive
    TN = sum(~actual_labels & ~predicted_labels);    % True Negative
    FP = sum(~actual_labels & predicted_labels);     % False Positive
    FN = sum(actual_labels & ~predicted_labels);     % False Negative
    
    % Create Confusion Matrix (rows are actual, columns are predicted)
    % [TP  FN]
    % [FP  TN]
    cm = [TP, FN; FP, TN];
    
    % Calculate Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN);
    
    % Handle division by zero for Precision/Recall
    if (TP + FP) == 0
        precision = 0;
    else
        precision = TP / (TP + FP);
    end
    
    if (TP + FN) == 0
        recall = 0;
    else
        recall = TP / (TP + FN);
    end
    
    if (precision + recall) == 0
        f1 = 0;
    else
        f1 = 2 * (precision * recall) / (precision + recall);
    end
end