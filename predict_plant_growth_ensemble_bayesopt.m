% predict_plant_growth_ensemble_bayesopt.m
% This script provides a high-performance approach to plant growth prediction using Ensemble Methods.
% Key features include:
% 1. Data Preprocessing: Converts descriptive ranges into numerical midpoints.
% 2. Automated Hyperparameter Tuning: Utilizes Bayesian Optimization to efficiently find the
%    optimal ensemble parameters (NumTrees, MinLeafSize, MaxNumSplits, NumPredictorsToSample)
%    for each target variable by minimizing cross-validation error.
% 3. Feature Importance Analysis: Calculates and displays which environmental factors are most influential.
% 4. Final Model Training: Trains robust Ensemble models using the optimized hyperparameters on the full dataset.
% 5. Prediction: Generates final predictions on the unseen test data.
% 6. Performance Evaluation: Displays model performance metrics and validation results.
% 7. Output: Displays tuning results, feature importance, and final predictions.

% --- Initialization ---
clear;
clc;
close all;
fprintf('=== Plant Growth Prediction using Ensemble Methods with Bayesian Optimization ===\n');
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

% Check for missing values and handle them
missing_train = sum(isnan(T_train{:, colsToConvert}), 'all');
missing_test = sum(isnan(T_test{:, colsToConvert}), 'all');
if missing_train > 0 || missing_test > 0
    fprintf('   ⚠ Warning: Found %d missing values in training data, %d in test data\n', missing_train, missing_test);
    % Fill missing values with median
    for i = 1:length(colsToConvert)
        colName = colsToConvert{i};
        medianVal = median(T_train.(colName), 'omitnan');
        T_train.(colName)(isnan(T_train.(colName))) = medianVal;
        T_test.(colName)(isnan(T_test.(colName))) = medianVal;
    end
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

% Define optimizable variables for Ensemble Methods (TreeBagger/Random Forest)
vars = [optimizableVariable('NumTrees', [50, 300], 'Type', 'integer');
        optimizableVariable('MinLeafSize', [1, 20], 'Type', 'integer');
        optimizableVariable('MaxNumSplits', [5, 100], 'Type', 'integer');
        optimizableVariable('NumPredictorsToSample', [1, length(featureNames)], 'Type', 'integer')];

% Helper function for hyperparameter optimization
function results = optimizeEnsemble(X, y, vars, targetName, featureNames)
    fprintf('   Optimizing hyperparameters for %s...\n', targetName);
    
    % Objective function using cross-validation
    objectiveFcn = @(params) crossValObjectiveEnsemble(X, y, params, featureNames);
    
    % Run Bayesian optimization
    results = bayesopt(objectiveFcn, vars, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'MaxObjectiveEvaluations', 40, ...
        'Verbose', 0, ...
        'PlotFcn', []);
end

% Cross-validation objective function for ensemble methods
function loss = crossValObjectiveEnsemble(X, y, params, featureNames)
    try
        % Create TreeBagger (Random Forest) with current parameters
        mdl = TreeBagger(params.NumTrees, X, y, ...
            'Method', 'regression', ...
            'MinLeafSize', params.MinLeafSize, ...
            'MaxNumSplits', params.MaxNumSplits, ...
            'NumPredictorsToSample', params.NumPredictorsToSample, ...
            'OOBPrediction', 'on', ...
            'PredictorNames', featureNames);
        
        % Use Out-of-Bag (OOB) error as validation metric
        oobError = oobError(mdl);
        loss = oobError(end);  % Take the final OOB error
        
        % If OOB error is not available, use cross-validation
        if isnan(loss) || loss <= 0
            % Perform 5-fold cross-validation manually
            cv = cvpartition(length(y), 'KFold', 5);
            mse_scores = zeros(cv.NumTestSets, 1);
            
            for fold = 1:cv.NumTestSets
                trainIdx = cv.training(fold);
                testIdx = cv.test(fold);
                
                X_train_fold = X(trainIdx, :);
                y_train_fold = y(trainIdx);
                X_val_fold = X(testIdx, :);
                y_val_fold = y(testIdx);
                
                % Train model on fold
                mdl_fold = TreeBagger(params.NumTrees, X_train_fold, y_train_fold, ...
                    'Method', 'regression', ...
                    'MinLeafSize', params.MinLeafSize, ...
                    'MaxNumSplits', params.MaxNumSplits, ...
                    'NumPredictorsToSample', params.NumPredictorsToSample, ...
                    'PredictorNames', featureNames);
                
                % Predict on validation set
                y_pred_fold = predict(mdl_fold, X_val_fold);
                mse_scores(fold) = mean((y_val_fold - y_pred_fold).^2);
            end
            
            loss = mean(mse_scores);
        end
        
    catch ME
        % If error occurs, return a high loss value
        loss = 1e6;
        fprintf('   Warning: Error in objective function: %s\n', ME.message);
    end
end

% --- 4. Hyperparameter Tuning for Each Target Variable ---
fprintf('\n5. Performing hyperparameter optimization...\n');

% Optimize for LeafLength
results_LL = optimizeEnsemble(X_train, y_train_LeafLength, vars, 'LeafLength', featureNames);
bestParams_LL = results_LL.XAtMinObjective;

% Optimize for LeafArea
results_LA = optimizeEnsemble(X_train, y_train_LeafArea, vars, 'LeafArea', featureNames);
bestParams_LA = results_LA.XAtMinObjective;

% Optimize for StemGirth
results_SG = optimizeEnsemble(X_train, y_train_StemGirth, vars, 'StemGirth', featureNames);
bestParams_SG = results_SG.XAtMinObjective;

% Display optimization results
fprintf('\n6. Hyperparameter optimization results:\n');
fprintf('   LeafLength - NumTrees: %d, MinLeafSize: %d, MaxNumSplits: %d, NumPredictors: %d (CV Loss: %.4f)\n', ...
    bestParams_LL.NumTrees, bestParams_LL.MinLeafSize, bestParams_LL.MaxNumSplits, ...
    bestParams_LL.NumPredictorsToSample, results_LL.MinObjective);
fprintf('   LeafArea   - NumTrees: %d, MinLeafSize: %d, MaxNumSplits: %d, NumPredictors: %d (CV Loss: %.4f)\n', ...
    bestParams_LA.NumTrees, bestParams_LA.MinLeafSize, bestParams_LA.MaxNumSplits, ...
    bestParams_LA.NumPredictorsToSample, results_LA.MinObjective);
fprintf('   StemGirth  - NumTrees: %d, MinLeafSize: %d, MaxNumSplits: %d, NumPredictors: %d (CV Loss: %.4f)\n', ...
    bestParams_SG.NumTrees, bestParams_SG.MinLeafSize, bestParams_SG.MaxNumSplits, ...
    bestParams_SG.NumPredictorsToSample, results_SG.MinObjective);

% --- 5. Train Final Models with Optimal Hyperparameters ---
fprintf('\n7. Training final Ensemble models with optimal hyperparameters...\n');

% Helper function to train final model and calculate feature importance
function [model, importance] = trainFinalEnsembleModel(X, y, params, featureNames, targetName)
    % Train TreeBagger with optimal parameters
    model = TreeBagger(params.NumTrees, X, y, ...
        'Method', 'regression', ...
        'MinLeafSize', params.MinLeafSize, ...
        'MaxNumSplits', params.MaxNumSplits, ...
        'NumPredictorsToSample', params.NumPredictorsToSample, ...
        'OOBPredictorImportance', 'on', ...
        'PredictorNames', featureNames);
    
    % Get feature importance from TreeBagger
    importance = model.OOBPermutedPredictorDeltaError;
    
    % Normalize importance scores to percentages
    if ~isempty(importance) && sum(importance) > 0
        importance = max(0, importance);  % Ensure non-negative
        importance = importance / sum(importance) * 100;
    else
        % If OOB importance is not available, calculate manually
        importance = calculatePermutationImportance(model, X, y, featureNames);
    end
    
    % Display feature importance
    fprintf('   Feature importance for %s:\n', targetName);
    [sorted_importance, idx] = sort(importance, 'descend');
    for i = 1:length(featureNames)
        fprintf('     %s: %.2f%%\n', featureNames{idx(i)}, sorted_importance(i));
    end
    fprintf('\n');
end

% Helper function to calculate permutation importance manually
function importance = calculatePermutationImportance(model, X, y, featureNames)
    % Get baseline predictions
    baseline_predictions = predict(model, X);
    baseline_mse = mean((y - baseline_predictions).^2);
    
    % Calculate importance by permuting each feature
    importance = zeros(1, length(featureNames));
    for i = 1:length(featureNames)
        X_permuted = X;
        X_permuted(:, i) = X_permuted(randperm(size(X, 1)), i);
        permuted_predictions = predict(model, X_permuted);
        permuted_mse = mean((y - permuted_predictions).^2);
        importance(i) = permuted_mse - baseline_mse;
    end
    
    % Normalize importance scores
    importance = max(0, importance);
    if sum(importance) > 0
        importance = importance / sum(importance) * 100;
    end
end

% Train final models
[model_LeafLength, importance_LL] = trainFinalEnsembleModel(X_train, y_train_LeafLength, bestParams_LL, featureNames, 'LeafLength');
[model_LeafArea, importance_LA] = trainFinalEnsembleModel(X_train, y_train_LeafArea, bestParams_LA, featureNames, 'LeafArea');
[model_StemGirth, importance_SG] = trainFinalEnsembleModel(X_train, y_train_StemGirth, bestParams_SG, featureNames, 'StemGirth');

% --- 6. Model Performance Evaluation ---
fprintf('8. Evaluating model performance on training data...\n');

function evaluateEnsembleModel(model, X, y, targetName)
    predictions = predict(model, X);
    mse = mean((y - predictions).^2);
    rmse = sqrt(mse);
    mae = mean(abs(y - predictions));
    r_squared = 1 - sum((y - predictions).^2) / sum((y - mean(y)).^2);
    
    fprintf('   %s Performance:\n', targetName);
    fprintf('     RMSE: %.4f\n', rmse);
    fprintf('     MAE:  %.4f\n', mae);
    fprintf('     R²:   %.4f\n', r_squared);
    
    % Additional ensemble-specific metrics
    if isprop(model, 'OOBIndices') && ~isempty(model.OOBIndices)
        oob_predictions = oobPredict(model);
        if ~isempty(oob_predictions)
            oob_rmse = sqrt(mean((y - oob_predictions).^2));
            fprintf('     OOB RMSE: %.4f\n', oob_rmse);
        end
    end
    fprintf('\n');
end

evaluateEnsembleModel(model_LeafLength, X_train, y_train_LeafLength, 'LeafLength');
evaluateEnsembleModel(model_LeafArea, X_train, y_train_LeafArea, 'LeafArea');
evaluateEnsembleModel(model_StemGirth, X_train, y_train_StemGirth, 'StemGirth');

% --- 7. Generate Final Predictions ---
fprintf('9. Generating predictions for test data...\n');

predictions_LeafLength = predict(model_LeafLength, X_test);
predictions_LeafArea = predict(model_LeafArea, X_test);
predictions_StemGirth = predict(model_StemGirth, X_test);

fprintf('   ✓ Predictions generated for %d test samples\n', length(predictions_LeafLength));

% --- 8. Display and Save Results ---
fprintf('\n10. Final Results:\n');

% Create results table
resultsTable = table(predictions_LeafLength, predictions_LeafArea, predictions_StemGirth, ...
    'VariableNames', {'PredictedLeafLength', 'PredictedLeafArea', 'PredictedStemGirth'});

% Display results
fprintf('=== Final Predictions (Ensemble Methods with Bayesian Optimization) ===\n');
disp(resultsTable);

% Optional: Save results to CSV file
try
    writetable(resultsTable, 'plant_growth_predictions_ensemble_bayesopt.csv');
    fprintf('\n   ✓ Results saved to "plant_growth_predictions_ensemble_bayesopt.csv"\n');
catch
    fprintf('\n   ⚠ Could not save results to CSV file\n');
end

% --- 9. Model Visualization (Optional) ---
fprintf('\n11. Generating model insights...\n');

% Plot feature importance comparison
figure('Position', [100, 100, 1200, 400]);

% Combine importance data
importanceData = [importance_LL; importance_LA; importance_SG];
targetNames = {'LeafLength', 'LeafArea', 'StemGirth'};

% Create grouped bar chart
subplot(1, 2, 1);
bar(importanceData');
set(gca, 'XTickLabel', featureNames);
xlabel('Features');
ylabel('Importance (%)');
title('Feature Importance by Target Variable');
legend(targetNames, 'Location', 'best');
grid on;

% Plot model performance comparison
subplot(1, 2, 2);
% Calculate R² scores for comparison
r2_scores = zeros(1, 3);
models = {model_LeafLength, model_LeafArea, model_StemGirth};
targets = {y_train_LeafLength, y_train_LeafArea, y_train_StemGirth};

for i = 1:3
    pred = predict(models{i}, X_train);
    r2_scores(i) = 1 - sum((targets{i} - pred).^2) / sum((targets{i} - mean(targets{i})).^2);
end

bar(r2_scores);
set(gca, 'XTickLabel', targetNames);
xlabel('Target Variables');
ylabel('R² Score');
title('Model Performance (R² Scores)');
ylim([0, 1]);
grid on;

% Add text labels on bars
for i = 1:length(r2_scores)
    text(i, r2_scores(i) + 0.02, sprintf('%.3f', r2_scores(i)), ...
        'HorizontalAlignment', 'center');
end

% Save the plot
try
    saveas(gcf, 'ensemble_model_analysis.png');
    fprintf('   ✓ Model analysis plot saved as "ensemble_model_analysis.png"\n');
catch
    fprintf('   ⚠ Could not save analysis plot\n');
end

% --- 10. Summary ---
fprintf('\n=== Model Training Summary ===\n');
fprintf('Algorithm: Ensemble Methods (TreeBagger/Random Forest) with Bayesian Optimization\n');
fprintf('Features used: %s\n', strjoin(featureNames, ', '));
fprintf('Target variables: %s\n', strjoin(targetVars, ', '));
fprintf('Optimization method: Bayesian Optimization with Out-of-Bag Validation\n');
fprintf('Number of test predictions: %d\n', height(resultsTable));
fprintf('Average number of trees: %.1f\n', mean([bestParams_LL.NumTrees, bestParams_LA.NumTrees, bestParams_SG.NumTrees]));
fprintf('Average R² score: %.4f\n', mean(r2_scores));
fprintf('\nProcess completed successfully!\n');
fprintf('Note: This ensemble approach provides similar benefits to XGBoost:\n');
fprintf('- Multiple tree ensemble for better accuracy\n');
fprintf('- Built-in feature selection via NumPredictorsToSample\n');
fprintf('- Robust out-of-bag validation\n');
fprintf('- Automatic handling of overfitting\n');