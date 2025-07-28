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

% สำหรับ LSTM เราต้องจัดเตรียมข้อมูลเป็น sequence
% กำหนดความยาวของ sequence (time steps)
sequence_length = 5; % จำนวน time steps ที่ใช้ในการทำนาย

% Function สำหรับสร้าง sequences
function [X_seq, y_seq] = create_sequences(X, y, seq_len)
    num_samples = size(X, 1) - seq_len + 1;
    num_features = size(X, 2);
    
    X_seq = zeros(num_samples, seq_len, num_features);
    y_seq = zeros(num_samples, 1);
    
    for i = 1:num_samples
        X_seq(i, :, :) = X(i:i+seq_len-1, :);
        y_seq(i) = y(i+seq_len-1);
    end
end

% สร้าง sequences สำหรับข้อมูลฝึก
[X_train_seq_leaf, y_train_seq_leaf] = create_sequences(X_train, y_train_leaf, sequence_length);
[X_train_seq_area, y_train_seq_area] = create_sequences(X_train, y_train_area, sequence_length);
[X_train_seq_stem, y_train_seq_stem] = create_sequences(X_train, y_train_stem, sequence_length);

% สร้าง sequences สำหรับข้อมูลทดสอบ
[X_test_seq_leaf, ~] = create_sequences(X_test, zeros(size(X_test, 1), 1), sequence_length);
[X_test_seq_area, ~] = create_sequences(X_test, zeros(size(X_test, 1), 1), sequence_length);
[X_test_seq_stem, ~] = create_sequences(X_test, zeros(size(X_test, 1), 1), sequence_length);

% Normalization ของข้อมูล
fprintf('Normalizing data for LSTM...\n');

% Normalize input features
X_train_norm = normalize(X_train_seq_leaf, 3); % normalize across features (3rd dimension)
X_test_norm_leaf = normalize(X_test_seq_leaf, 3);
X_test_norm_area = normalize(X_test_seq_area, 3);
X_test_norm_stem = normalize(X_test_seq_stem, 3);

% Normalize target variables
y_train_leaf_norm = normalize(y_train_seq_leaf);
y_train_area_norm = normalize(y_train_seq_area);
y_train_stem_norm = normalize(y_train_seq_stem);

% เก็บค่า mean และ std สำหรับ denormalization
leaf_mean = mean(y_train_seq_leaf);
leaf_std = std(y_train_seq_leaf);
area_mean = mean(y_train_seq_area);
area_std = std(y_train_seq_area);
stem_mean = mean(y_train_seq_stem);
stem_std = std(y_train_seq_stem);

fprintf('Data preparation for LSTM complete.\n');

% --- สร้างโมเดล LSTM ---
fprintf('\nCreating LSTM models...\n');

% กำหนดพารามิเตอร์ LSTM ที่ปรับแล้ว
numHiddenUnits = 64; % จำนวน hidden units ใน LSTM layer
numFeatures = size(X_train, 2); % จำนวนฟีเจอร์
numResponses = 1; % จำนวน output (regression)

% สร้าง LSTM layers สำหรับ Leaf Length
layers_leaf = [
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits/2, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(numResponses)
    regressionLayer
];

% สร้าง LSTM layers สำหรับ Leaf Area
layers_area = [
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits/2, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(numResponses)
    regressionLayer
];

% สร้าง LSTM layers สำหรับ Stem Girth
layers_stem = [
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits/2, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(numResponses)
    regressionLayer
];

fprintf('LSTM architectures created.\n');

% --- กำหนดตัวเลือกการฝึก (Training Options) ---
fprintf('Setting training options...\n');

options = trainingOptions('adam', ...
    'MaxEpochs', 300, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 50, ...
    'GradientThreshold', 1, ...
    'ValidationFrequency', 10, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

fprintf('Training options set.\n');

% --- เตรียมข้อมูลสำหรับการฝึก LSTM ---
% แปลงข้อมูลเป็น cell arrays สำหรับ LSTM
X_train_cell_leaf = cell(size(X_train_norm, 1), 1);
X_train_cell_area = cell(size(X_train_norm, 1), 1);
X_train_cell_stem = cell(size(X_train_norm, 1), 1);

for i = 1:size(X_train_norm, 1)
    X_train_cell_leaf{i} = squeeze(X_train_norm(i, :, :))';
    X_train_cell_area{i} = squeeze(X_train_norm(i, :, :))';
    X_train_cell_stem{i} = squeeze(X_train_norm(i, :, :))';
end

% เตรียม test data
X_test_cell_leaf = cell(size(X_test_norm_leaf, 1), 1);
X_test_cell_area = cell(size(X_test_norm_area, 1), 1);
X_test_cell_stem = cell(size(X_test_norm_stem, 1), 1);

for i = 1:size(X_test_norm_leaf, 1)
    X_test_cell_leaf{i} = squeeze(X_test_norm_leaf(i, :, :))';
    X_test_cell_area{i} = squeeze(X_test_norm_area(i, :, :))';
    X_test_cell_stem{i} = squeeze(X_test_norm_stem(i, :, :))';
end

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

% Denormalize predictions
y_pred_leaf_lstm = y_pred_leaf_norm * leaf_std + leaf_mean;
y_pred_area_lstm = y_pred_area_norm * area_std + area_mean;
y_pred_stem_lstm = y_pred_stem_norm * stem_std + stem_mean;

fprintf('LSTM predictions complete and denormalized.\n');

% แสดงผลการทำนาย
disp(' ');
disp('--- LSTM Predictions on Test Data ---');
disp('LSTM Predictions for Leaf Length:');
disp(y_pred_leaf_lstm);
disp('LSTM Predictions for Leaf Area:');
disp(y_pred_area_lstm);
disp('LSTM Predictions for Stem Girth:');
disp(y_pred_stem_lstm);

% --- การประเมินผลสำหรับ LSTM Models ---
fprintf('\n=== LSTM Model Performance Evaluation ===\n');

% เนื่องจากเราใช้ sequences, จำนวนการทำนายจะน้อยกว่าข้อมูลเดิม
% ต้องปรับข้อมูลจริงให้ตรงกัน
actual_leaf_adjusted = actual_data.LeafLength(sequence_length:end);
actual_area_adjusted = actual_data.LeafArea(sequence_length:end);
actual_stem_adjusted = actual_data.StemGirth(sequence_length:end);

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
     'leaf_mean', 'leaf_std', 'area_mean', 'area_std', 'stem_mean', 'stem_std', ...
     'sequence_length', 'numFeatures');
fprintf('Models saved to lstm_plant_growth_models.mat\n');

fprintf('\n=== LSTM Plant Growth Prediction Analysis Complete ===\n');