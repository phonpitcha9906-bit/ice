% create_sample_data.m
% This script creates sample training and test data for the plant growth prediction model

clear;
clc;

fprintf('Creating sample data files for plant growth prediction...\n');

% Set random seed for reproducibility
rng(42);

% Define sample size
n_train = 100;
n_test = 20;

% Define feature ranges (these will be converted to range strings)
moisture_ranges = {'20-30%', '30-40%', '40-50%', '50-60%', '60-70%'};
humidity_ranges = {'40-50% RH', '50-60% RH', '60-70% RH', '70-80% RH', '80-90% RH'};
light_ranges = {'1000-2000 lux', '2000-3000 lux', '3000-4000 lux', '4000-5000 lux', '5000-6000 lux'};
co2_ranges = {'300-400 ppm', '400-500 ppm', '500-600 ppm', '600-700 ppm', '700-800 ppm'};

% Function to generate synthetic target values based on features
function targets = generate_targets(moisture_mid, humidity_mid, light_mid, co2_mid)
    % Normalize features to 0-1 scale for easier calculation
    moisture_norm = (moisture_mid - 25) / 45;  % 25-70 range
    humidity_norm = (humidity_mid - 45) / 45;  % 45-90 range
    light_norm = (light_mid - 1500) / 4500;    % 1500-6000 range
    co2_norm = (co2_mid - 350) / 450;          % 350-800 range
    
    % Generate leaf length (8-16 cm range)
    leaf_length = 8 + 8 * (0.3*moisture_norm + 0.2*humidity_norm + 0.4*light_norm + 0.1*co2_norm) + 0.5*randn();
    leaf_length = max(6, min(18, leaf_length)); % Constrain to reasonable range
    
    % Generate leaf area (10-25 cm² range)
    leaf_area = 10 + 15 * (0.25*moisture_norm + 0.25*humidity_norm + 0.35*light_norm + 0.15*co2_norm) + 1*randn();
    leaf_area = max(8, min(28, leaf_area)); % Constrain to reasonable range
    
    % Generate stem girth (1-4 cm range)
    stem_girth = 1 + 3 * (0.4*moisture_norm + 0.15*humidity_norm + 0.3*light_norm + 0.15*co2_norm) + 0.2*randn();
    stem_girth = max(0.5, min(5, stem_girth)); % Constrain to reasonable range
    
    targets = [leaf_length, leaf_area, stem_girth];
end

% Function to convert midpoint back to range string
function range_str = midpoint_to_range(midpoint, ranges)
    % Find the range that contains this midpoint
    for i = 1:length(ranges)
        range_str_temp = ranges{i};
        clean_str = regexprep(range_str_temp, {'%', 'RH', 'lux', 'ppm', '\s+'}, '');
        parts = strsplit(clean_str, '-');
        if length(parts) == 2
            start_val = str2double(parts{1});
            end_val = str2double(parts{2});
            range_mid = (start_val + end_val) / 2;
            if abs(midpoint - range_mid) < 7.5 % Within reasonable tolerance
                range_str = range_str_temp;
                return;
            end
        end
    end
    % If no exact match, use the closest one
    range_str = ranges{randi(length(ranges))};
end

% Generate training data
fprintf('Generating training data (%d samples)...\n', n_train);
train_data = {};

for i = 1:n_train
    % Randomly select ranges
    moisture_idx = randi(length(moisture_ranges));
    humidity_idx = randi(length(humidity_ranges));
    light_idx = randi(length(light_ranges));
    co2_idx = randi(length(co2_ranges));
    
    moisture_str = moisture_ranges{moisture_idx};
    humidity_str = humidity_ranges{humidity_idx};
    light_str = light_ranges{light_idx};
    co2_str = co2_ranges{co2_idx};
    
    % Convert to midpoints for target generation
    moisture_mid = 25 + 10*(moisture_idx-1) + 5; % Approximate midpoints
    humidity_mid = 45 + 10*(humidity_idx-1) + 5;
    light_mid = 1500 + 1000*(light_idx-1) + 500;
    co2_mid = 350 + 100*(co2_idx-1) + 50;
    
    % Generate target values
    targets = generate_targets(moisture_mid, humidity_mid, light_mid, co2_mid);
    
    train_data{i, 1} = moisture_str;
    train_data{i, 2} = humidity_str;
    train_data{i, 3} = light_str;
    train_data{i, 4} = co2_str;
    train_data{i, 5} = targets(1); % LeafLength
    train_data{i, 6} = targets(2); % LeafArea
    train_data{i, 7} = targets(3); % StemGirth
end

% Create training table
train_table = cell2table(train_data, 'VariableNames', ...
    {'Moisture', 'Humidity', 'LightIntensity', 'CO2Concentration', 'LeafLength', 'LeafArea', 'StemGirth'});

% Generate test data
fprintf('Generating test data (%d samples)...\n', n_test);
test_data = {};

for i = 1:n_test
    % Randomly select ranges
    moisture_idx = randi(length(moisture_ranges));
    humidity_idx = randi(length(humidity_ranges));
    light_idx = randi(length(light_ranges));
    co2_idx = randi(length(co2_ranges));
    
    moisture_str = moisture_ranges{moisture_idx};
    humidity_str = humidity_ranges{humidity_idx};
    light_str = light_ranges{light_idx};
    co2_str = co2_ranges{co2_idx};
    
    % Convert to midpoints for target generation
    moisture_mid = 25 + 10*(moisture_idx-1) + 5;
    humidity_mid = 45 + 10*(humidity_idx-1) + 5;
    light_mid = 1500 + 1000*(light_idx-1) + 500;
    co2_mid = 350 + 100*(co2_idx-1) + 50;
    
    % Generate target values
    targets = generate_targets(moisture_mid, humidity_mid, light_mid, co2_mid);
    
    test_data{i, 1} = moisture_str;
    test_data{i, 2} = humidity_str;
    test_data{i, 3} = light_str;
    test_data{i, 4} = co2_str;
    test_data{i, 5} = targets(1); % LeafLength
    test_data{i, 6} = targets(2); % LeafArea
    test_data{i, 7} = targets(3); % StemGirth
end

% Create test table
test_table = cell2table(test_data, 'VariableNames', ...
    {'Moisture', 'Humidity', 'LightIntensity', 'CO2Concentration', 'LeafLength', 'LeafArea', 'StemGirth'});

% Save to Excel files
try
    writetable(train_table, 'datatraining.xlsx');
    fprintf('✓ Training data saved to datatraining.xlsx\n');
catch ME
    fprintf('✗ Error saving training data: %s\n', ME.message);
end

try
    writetable(test_table, 'datatest.xlsx');
    fprintf('✓ Test data saved to datatest.xlsx\n');
catch ME
    fprintf('✗ Error saving test data: %s\n', ME.message);
end

% Display sample data
fprintf('\nSample training data (first 5 rows):\n');
disp(train_table(1:min(5, height(train_table)), :));

fprintf('\nSample test data (first 5 rows):\n');
disp(test_table(1:min(5, height(test_table)), :));

fprintf('\nData generation completed successfully!\n');
fprintf('You can now run the plant growth prediction script.\n');