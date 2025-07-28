% LSTM Model for Plant Growth Prediction
% กำหนดชื่อไฟล์ข้อมูล
training_file = 'datatraining.xlsx';
test_file = 'datatest.xlsx';
actual_file = 'turedata.xlsx'; % ไฟล์สำหรับค่าจริงของชุดทดสอบ

% โหลดข้อมูลชุดฝึกและชุดทดสอบ
fprintf('Loading data files...\n');
training_data = readtable(training_file);
test_data = readtable(test_file);
actual_data = readtable(actual_file); % โหลดค่าจริงของชุดทดสอบ
fprintf('Data loaded.\n');

% กำหนดคอลัมน์ฟีเจอร์
feature_columns = {'Moisture', 'Humidity', 'LightIntensity', 'CO2Concentration'};

% ตรวจสอบและแปลงข้อมูลเชิงหมวดหมู่เป็นตัวเลข (หากจำเป็น)
fprintf('Checking and converting categorical features to numeric...\n');
for i = 1:length(feature_columns)
    col = feature_columns{i};
    % ตรวจสอบว่าคอลัมน์มีข้อมูลข้อความหรือ categorical array หรือไม่
    if iscellstr(training_data.(col)) || iscategorical(training_data.(col))
        fprintf('  Converting "%s" column from categorical/text to numeric using double(categorical()).\n', col);
        training_data.(col) = double(categorical(training_data.(col)));
        test_data.(col) = double(categorical(test_data.(col)));
    else
        fprintf('  "%s" column is already numeric. No conversion needed.\n', col);
    end
end
fprintf('Categorical conversion check complete.\n');

% เลือกฟีเจอร์สำหรับการฝึกและทดสอบ
X_train = training_data{:, feature_columns};
X_test = test_data{:, feature_columns};

% ตัวแปรตามสำหรับการฝึก (Response variables for training)
y_train_leaf = training_data.LeafLength;
y_train_area = training_data.LeafArea;
y_train_stem = training_data.StemGirth;

% --- Data Preparation for LSTM ---
fprintf('\nPreparing data for LSTM models...\n');

% สำหรับ LSTM เราใช้วิธี sliding window แทนการสร้าง sequence ปกติ
% เพื่อไม่ให้ข้อมูลลดลง
sequence_length = 3; % ลดลงเป็น 3 เพื่อให้ได้ข้อมูลมากขึ้น

% Function สำหรับสร้าง sequences แบบใหม่ (overlapping windows)
function [X_seq, y_seq] = create_sequences_improved(X, y, seq_len)
    num_samples = size(X, 1);
    num_features = size(X, 2);
    
    % สร้าง sequences โดยใช้การทำซ้ำข้อมูล (data augmentation)
    X_seq = zeros(num_samples, seq_len, num_features);
    y_seq = y;
    
    for i = 1:num_samples
        for j = 1:seq_len
            % ใช้ circular indexing เพื่อไม่ให้ข้อมูลลดลง
            idx = mod(i + j - 2, num_samples) + 1;
            X_seq(i, j, :) = X(idx, :);
        end
    end
end

% สร้าง sequences สำหรับข้อมูลฝึก (ใหม่)
[X_train_seq_leaf, y_train_seq_leaf] = create_sequences_improved(X_train, y_train_leaf, sequence_length);
[X_train_seq_area, y_train_seq_area] = create_sequences_improved(X_train, y_train_area, sequence_length);
[X_train_seq_stem, y_train_seq_stem] = create_sequences_improved(X_train, y_train_stem, sequence_length);

% สร้าง sequences สำหรับข้อมูลทดสอบ (ใหม่)
[X_test_seq_leaf, ~] = create_sequences_improved(X_test, zeros(size(X_test, 1), 1), sequence_length);
[X_test_seq_area, ~] = create_sequences_improved(X_test, zeros(size(X_test, 1), 1), sequence_length);
[X_test_seq_stem, ~] = create_sequences_improved(X_test, zeros(size(X_test, 1), 1), sequence_length);

% แสดงขนาดข้อมูลหลังสร้าง sequences
fprintf('Training data size after sequencing: %d samples\n', size(X_train_seq_leaf, 1));
fprintf('Test data size after sequencing: %d samples\n', size(X_test_seq_leaf, 1));

% Normalization ของข้อมูลแบบปรับปรุง
fprintf('Normalizing data for LSTM...\n');

% ใช้ mapminmax แทน normalize เพื่อความสอดคล้องกับ FNN
% Normalize input sequences
X_train_combined = reshape(X_train_seq_leaf, [], size(X_train_seq_leaf, 3));
[X_train_norm_flat, input_settings] = mapminmax(X_train_combined', -1, 1);
X_train_norm = reshape(X_train_norm_flat', size(X_train_seq_leaf));

% Apply same normalization to test data
X_test_combined_leaf = reshape(X_test_seq_leaf, [], size(X_test_seq_leaf, 3));
X_test_norm_flat_leaf = mapminmax('apply', X_test_combined_leaf', input_settings);
X_test_norm_leaf = reshape(X_test_norm_flat_leaf', size(X_test_seq_leaf));

X_test_combined_area = reshape(X_test_seq_area, [], size(X_test_seq_area, 3));
X_test_norm_flat_area = mapminmax('apply', X_test_combined_area', input_settings);
X_test_norm_area = reshape(X_test_norm_flat_area', size(X_test_seq_area));

X_test_combined_stem = reshape(X_test_seq_stem, [], size(X_test_seq_stem, 3));
X_test_norm_flat_stem = mapminmax('apply', X_test_combined_stem', input_settings);
X_test_norm_stem = reshape(X_test_norm_flat_stem', size(X_test_seq_stem));

% Normalize target variables ด้วย mapminmax
[y_train_leaf_norm, leaf_output_settings] = mapminmax(y_train_seq_leaf', -1, 1);
y_train_leaf_norm = y_train_leaf_norm';

[y_train_area_norm, area_output_settings] = mapminmax(y_train_seq_area', -1, 1);
y_train_area_norm = y_train_area_norm';

[y_train_stem_norm, stem_output_settings] = mapminmax(y_train_seq_stem', -1, 1);
y_train_stem_norm = y_train_stem_norm';

fprintf('Data preparation for LSTM complete.\n');

% --- สร้างโมเดล LSTM ---
fprintf('\nCreating LSTM models...\n');

% กำหนดพารามิเตอร์ LSTM ที่ปรับแล้วสำหรับข้อมูลขนาดเล็ก
numFeatures = size(X_train, 2); % จำนวนฟีเจอร์
numResponses = 1; % จำนวน output (regression)

% สร้าง LSTM layers สำหรับ Leaf Length (แบบเรียบง่าย)
layers_leaf = [
    sequenceInputLayer(numFeatures)
    lstmLayer(16, 'OutputMode', 'last')  % ลดขนาดให้เหมาะกับข้อมูลเล็ก
    dropoutLayer(0.3)
    fullyConnectedLayer(8)
    tanhLayer                            % ใช้ tanh แทน relu
    fullyConnectedLayer(numResponses)
    regressionLayer
];

% สร้าง LSTM layers สำหรับ Leaf Area (แตกต่างเล็กน้อย)
layers_area = [
    sequenceInputLayer(numFeatures)
    lstmLayer(20, 'OutputMode', 'last')  % ขนาดต่างกันเล็กน้อย
    dropoutLayer(0.2)
    fullyConnectedLayer(12)
    reluLayer
    fullyConnectedLayer(numResponses)
    regressionLayer
];

% สร้าง LSTM layers สำหรับ Stem Girth (แตกต่างอีก)
layers_stem = [
    sequenceInputLayer(numFeatures)
    lstmLayer(24, 'OutputMode', 'last')  % ขนาดต่างกันอีก
    dropoutLayer(0.25)
    fullyConnectedLayer(10)
    sigmoidLayer                         % ใช้ sigmoid
    fullyConnectedLayer(6)
    reluLayer
    fullyConnectedLayer(numResponses)
    regressionLayer
];

fprintf('LSTM architectures created.\n');

% --- กำหนดตัวเลือกการฝึก (Training Options) ---
fprintf('Setting training options...\n');

% ปรับ training options สำหรับข้อมูลขนาดเล็ก
options = trainingOptions('adam', ...
    'MaxEpochs', 150, ...                  % ลด epochs
    'MiniBatchSize', 4, ...                % ลด batch size สำหรับข้อมูลเล็ก
    'InitialLearnRate', 0.01, ...          % เพิ่ม learning rate
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.7, ...        % ปรับ drop factor
    'LearnRateDropPeriod', 30, ...         % ลด drop period
    'GradientThreshold', 2, ...            % เพิ่ม gradient threshold
    'ValidationFrequency', 5, ...          % ลด validation frequency
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

fprintf('Training options set.\n');

% --- เตรียมข้อมูลสำหรับการฝึก LSTM ---
% แปลงข้อมูลเป็น cell arrays สำหรับ LSTM โดยใช้ข้อมูลที่แตกต่างกันสำหรับแต่ละโมเดล

% สำหรับ Leaf Length
X_train_cell_leaf = cell(size(X_train_norm, 1), 1);
for i = 1:size(X_train_norm, 1)
    X_train_cell_leaf{i} = squeeze(X_train_norm(i, :, :))';
end

% สำหรับ Leaf Area - ใช้ข้อมูลที่ normalize แยกต่างหาก
X_train_area_combined = reshape(X_train_seq_area, [], size(X_train_seq_area, 3));
[X_train_area_norm_flat, input_settings_area] = mapminmax(X_train_area_combined', -1, 1);
X_train_area_norm = reshape(X_train_area_norm_flat', size(X_train_seq_area));

X_train_cell_area = cell(size(X_train_area_norm, 1), 1);
for i = 1:size(X_train_area_norm, 1)
    X_train_cell_area{i} = squeeze(X_train_area_norm(i, :, :))';
end

% สำหรับ Stem Girth - ใช้ข้อมูลที่ normalize แยกต่างหาก
X_train_stem_combined = reshape(X_train_seq_stem, [], size(X_train_seq_stem, 3));
[X_train_stem_norm_flat, input_settings_stem] = mapminmax(X_train_stem_combined', -1, 1);
X_train_stem_norm = reshape(X_train_stem_norm_flat', size(X_train_seq_stem));

X_train_cell_stem = cell(size(X_train_stem_norm, 1), 1);
for i = 1:size(X_train_stem_norm, 1)
    X_train_cell_stem{i} = squeeze(X_train_stem_norm(i, :, :))';
end

% เตรียม test data สำหรับแต่ละโมเดล
% Leaf test data
X_test_cell_leaf = cell(size(X_test_norm_leaf, 1), 1);
for i = 1:size(X_test_norm_leaf, 1)
    X_test_cell_leaf{i} = squeeze(X_test_norm_leaf(i, :, :))';
end

% Area test data
X_test_norm_area_flat = mapminmax('apply', X_test_combined_area', input_settings_area);
X_test_norm_area_corrected = reshape(X_test_norm_area_flat', size(X_test_seq_area));

X_test_cell_area = cell(size(X_test_norm_area_corrected, 1), 1);
for i = 1:size(X_test_norm_area_corrected, 1)
    X_test_cell_area{i} = squeeze(X_test_norm_area_corrected(i, :, :))';
end

% Stem test data
X_test_norm_stem_flat = mapminmax('apply', X_test_combined_stem', input_settings_stem);
X_test_norm_stem_corrected = reshape(X_test_norm_stem_flat', size(X_test_seq_stem));

X_test_cell_stem = cell(size(X_test_norm_stem_corrected, 1), 1);
for i = 1:size(X_test_norm_stem_corrected, 1)
    X_test_cell_stem{i} = squeeze(X_test_norm_stem_corrected(i, :, :))';
end

% เพิ่มการ shuffle ข้อมูลเพื่อลดการทำซ้ำ
fprintf('Adding randomization to training data...\n');
rng(42); % set seed for reproducibility
shuffle_idx = randperm(length(X_train_cell_leaf));

X_train_cell_leaf = X_train_cell_leaf(shuffle_idx);
y_train_leaf_norm = y_train_leaf_norm(shuffle_idx);

X_train_cell_area = X_train_cell_area(shuffle_idx);
y_train_area_norm = y_train_area_norm(shuffle_idx);

X_train_cell_stem = X_train_cell_stem(shuffle_idx);
y_train_stem_norm = y_train_stem_norm(shuffle_idx);

% --- ฝึกโมเดล LSTM ---
fprintf('\nStarting LSTM models training...\n');

% ฝึกโมเดล LSTM สำหรับ Leaf Length
fprintf('Training LSTM model for Leaf Length...\n');
net_lstm_leaf = trainNetwork(X_train_cell_leaf, y_train_leaf_norm, layers_leaf, options);
fprintf('Leaf Length LSTM model trained.\n');

% ฝึกโมเดล LSTM สำหรับ Leaf Area
fprintf('Training LSTM model for Leaf Area...\n');
net_lstm_area = trainNetwork(X_train_cell_area, y_train_area_norm, layers_area, options);
fprintf('Leaf Area LSTM model trained.\n');

% ฝึกโมเดล LSTM สำหรับ Stem Girth
fprintf('Training LSTM model for Stem Girth...\n');
net_lstm_stem = trainNetwork(X_train_cell_stem, y_train_stem_norm, layers_stem, options);
fprintf('Stem Girth LSTM model trained.\n');

fprintf('All LSTM models training complete.\n');

% --- ทำนายผลด้วยโมเดล LSTM ---
fprintf('\nMaking predictions with LSTM models...\n');

% ทำนายผล
y_pred_leaf_norm = predict(net_lstm_leaf, X_test_cell_leaf);
y_pred_area_norm = predict(net_lstm_area, X_test_cell_area);
y_pred_stem_norm = predict(net_lstm_stem, X_test_cell_stem);

% Denormalize predictions ด้วย mapminmax reverse
y_pred_leaf_lstm = mapminmax('reverse', y_pred_leaf_norm', leaf_output_settings)';
y_pred_area_lstm = mapminmax('reverse', y_pred_area_norm', area_output_settings)';
y_pred_stem_lstm = mapminmax('reverse', y_pred_stem_norm', stem_output_settings)';

fprintf('LSTM predictions complete and denormalized.\n');

% แสดงผลการทำนาย
disp(' ');
disp('--- LSTM Predictions on Test Data ---');
fprintf('Number of test samples: %d\n', length(y_pred_leaf_lstm));
fprintf('Number of actual samples: %d\n', length(actual_data.LeafLength));

disp('LSTM Predictions for Leaf Length:');
for i = 1:length(y_pred_leaf_lstm)
    fprintf('Sample %d: Predicted = %.4f, Actual = %.4f\n', i, y_pred_leaf_lstm(i), actual_data.LeafLength(i));
end

disp(' ');
disp('LSTM Predictions for Leaf Area:');
for i = 1:length(y_pred_area_lstm)
    fprintf('Sample %d: Predicted = %.4f, Actual = %.4f\n', i, y_pred_area_lstm(i), actual_data.LeafArea(i));
end

disp(' ');
disp('LSTM Predictions for Stem Girth:');
for i = 1:length(y_pred_stem_lstm)
    fprintf('Sample %d: Predicted = %.4f, Actual = %.4f\n', i, y_pred_stem_lstm(i), actual_data.StemGirth(i));
end

% --- การประเมินผลสำหรับ LSTM Models ---
fprintf('\n=== LSTM Model Performance Evaluation ===\n');

% ตอนนี้เราใช้ข้อมูลครบแล้ว ไม่ต้องปรับข้อมูลจริง
actual_leaf_adjusted = actual_data.LeafLength;
actual_area_adjusted = actual_data.LeafArea;
actual_stem_adjusted = actual_data.StemGirth;

% คำนวณ metrics สำหรับ LSTM models
% Leaf Length LSTM
mae_leaf_lstm = mean(abs(y_pred_leaf_lstm - actual_leaf_adjusted));
rmse_leaf_lstm = sqrt(mean((y_pred_leaf_lstm - actual_leaf_adjusted).^2));
r2_leaf_lstm = 1 - sum((actual_leaf_adjusted - y_pred_leaf_lstm).^2) / sum((actual_leaf_adjusted - mean(actual_leaf_adjusted)).^2);
mape_leaf_lstm = mean(abs((actual_leaf_adjusted - y_pred_leaf_lstm) ./ actual_leaf_adjusted)) * 100;

% Leaf Area LSTM
mae_area_lstm = mean(abs(y_pred_area_lstm - actual_area_adjusted));
rmse_area_lstm = sqrt(mean((y_pred_area_lstm - actual_area_adjusted).^2));
r2_area_lstm = 1 - sum((actual_area_adjusted - y_pred_area_lstm).^2) / sum((actual_area_adjusted - mean(actual_area_adjusted)).^2);
mape_area_lstm = mean(abs((actual_area_adjusted - y_pred_area_lstm) ./ actual_area_adjusted)) * 100;

% Stem Girth LSTM
mae_stem_lstm = mean(abs(y_pred_stem_lstm - actual_stem_adjusted));
rmse_stem_lstm = sqrt(mean((y_pred_stem_lstm - actual_stem_adjusted).^2));
r2_stem_lstm = 1 - sum((actual_stem_adjusted - y_pred_stem_lstm).^2) / sum((actual_stem_adjusted - mean(actual_stem_adjusted)).^2);
mape_stem_lstm = mean(abs((actual_stem_adjusted - y_pred_stem_lstm) ./ actual_stem_adjusted)) * 100;

% แสดงผลการประเมิน LSTM
fprintf('\n--- LSTM Model Performance Metrics ---\n');
fprintf('Leaf Length LSTM:\n');
fprintf('  MAE = %.4f, RMSE = %.4f, R² = %.4f, MAPE = %.2f%%\n', mae_leaf_lstm, rmse_leaf_lstm, r2_leaf_lstm, mape_leaf_lstm);
fprintf('Leaf Area LSTM:\n');
fprintf('  MAE = %.4f, RMSE = %.4f, R² = %.4f, MAPE = %.2f%%\n', mae_area_lstm, rmse_area_lstm, r2_area_lstm, mape_area_lstm);
fprintf('Stem Girth LSTM:\n');
fprintf('  MAE = %.4f, RMSE = %.4f, R² = %.4f, MAPE = %.2f%%\n', mae_stem_lstm, rmse_stem_lstm, r2_stem_lstm, mape_stem_lstm);

% --- สร้างกราฟแสดงผลการเปรียบเทียบ ---
fprintf('\nCreating comparison plots...\n');

% สร้างกราฟเปรียบเทียบค่าจริงกับค่าทำนาย
figure('Position', [100, 100, 1200, 400]);

% Leaf Length
subplot(1, 3, 1);
scatter(actual_leaf_adjusted, y_pred_leaf_lstm, 'filled', 'MarkerFaceColor', [0.2, 0.6, 0.8]);
hold on;
plot([min(actual_leaf_adjusted), max(actual_leaf_adjusted)], [min(actual_leaf_adjusted), max(actual_leaf_adjusted)], 'r--', 'LineWidth', 2);
xlabel('Actual Leaf Length');
ylabel('Predicted Leaf Length (LSTM)');
title(sprintf('Leaf Length: R² = %.3f', r2_leaf_lstm));
grid on;
axis equal;
axis tight;

% Leaf Area
subplot(1, 3, 2);
scatter(actual_area_adjusted, y_pred_area_lstm, 'filled', 'MarkerFaceColor', [0.8, 0.4, 0.2]);
hold on;
plot([min(actual_area_adjusted), max(actual_area_adjusted)], [min(actual_area_adjusted), max(actual_area_adjusted)], 'r--', 'LineWidth', 2);
xlabel('Actual Leaf Area');
ylabel('Predicted Leaf Area (LSTM)');
title(sprintf('Leaf Area: R² = %.3f', r2_area_lstm));
grid on;
axis equal;
axis tight;

% Stem Girth
subplot(1, 3, 3);
scatter(actual_stem_adjusted, y_pred_stem_lstm, 'filled', 'MarkerFaceColor', [0.2, 0.8, 0.4]);
hold on;
plot([min(actual_stem_adjusted), max(actual_stem_adjusted)], [min(actual_stem_adjusted), max(actual_stem_adjusted)], 'r--', 'LineWidth', 2);
xlabel('Actual Stem Girth');
ylabel('Predicted Stem Girth (LSTM)');
title(sprintf('Stem Girth: R² = %.3f', r2_stem_lstm));
grid on;
axis equal;
axis tight;

sgtitle('LSTM Model: Actual vs Predicted Values');

% --- สร้างตารางสรุปผลการประเมิน ---
fprintf('\n=== Performance Summary Table ===\n');
metrics_table = table(...
    {'Leaf Length'; 'Leaf Area'; 'Stem Girth'}, ...
    [mae_leaf_lstm; mae_area_lstm; mae_stem_lstm], ...
    [rmse_leaf_lstm; rmse_area_lstm; rmse_stem_lstm], ...
    [r2_leaf_lstm; r2_area_lstm; r2_stem_lstm], ...
    [mape_leaf_lstm; mape_area_lstm; mape_stem_lstm], ...
    'VariableNames', {'Target_Variable', 'MAE', 'RMSE', 'R_Squared', 'MAPE_Percent'});

disp(metrics_table);

% --- บันทึกโมเดล ---
fprintf('\nSaving trained LSTM models...\n');
save('lstm_plant_growth_models.mat', 'net_lstm_leaf', 'net_lstm_area', 'net_lstm_stem', ...
     'leaf_output_settings', 'area_output_settings', 'stem_output_settings', ...
     'input_settings', 'sequence_length', 'numFeatures');
fprintf('Models saved to lstm_plant_growth_models.mat\n');

fprintf('\n=== LSTM Plant Growth Prediction Analysis Complete ===\n');