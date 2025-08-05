% test_script.m
% Test script to verify the fixed plant growth prediction script works

clear;
clc;

fprintf('=== Testing Fixed Plant Growth Prediction Script ===\n\n');

% First, create sample data
fprintf('Step 1: Creating sample data...\n');
try
    create_sample_data;
    fprintf('✓ Sample data created successfully\n\n');
catch ME
    fprintf('✗ Error creating sample data: %s\n', ME.message);
    return;
end

% Check if data files exist
if exist('datatraining.xlsx', 'file') && exist('datatest.xlsx', 'file')
    fprintf('✓ Data files found\n\n');
else
    fprintf('✗ Data files not found\n');
    return;
end

% Run the fixed prediction script
fprintf('Step 2: Running prediction script...\n');
try
    predict_plant_growth_rf_alternative;
    fprintf('\n✓ Prediction script completed successfully!\n');
catch ME
    fprintf('\n✗ Error in prediction script: %s\n', ME.message);
    fprintf('Error occurred at line: %s\n', ME.stack(1).name);
end

fprintf('\n=== Test Complete ===\n');