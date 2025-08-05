% plant_growth_prediction_basic.m
% โปรแกรมทำนายการเจริญเติบโตของพืชแบบสำเร็จรูป
% ใช้ Multiple Linear Regression และ k-Nearest Neighbors
% ไม่ต้องใช้ Toolbox เพิ่มเติม - รันได้ทันทีใน MATLAB พื้นฐาน

clear;
clc;
close all;

fprintf('=== โปรแกรมทำนายการเจริญเติบโตของพืช (พร้อมใช้งาน) ===\n');
fprintf('เริ่มต้นการประมวลผล...\n\n');

% --- ฟังก์ชันแปลงข้อมูลช่วงเป็นตัวเลข ---
function num_val = convertRangeToNumber(range_str)
    if ischar(range_str) || isstring(range_str)
        clean_str = regexprep(char(range_str), {'%', 'RH', 'lux', 'ppm', '\s+'}, '');
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
    else
        num_val = double(range_str);
    end
end

% --- สร้างข้อมูลตัวอย่าง ---
fprintf('1. สร้างข้อมูลตัวอย่างสำหรับการทดสอบ...\n');

% ข้อมูลฝึกสอน (100 ตัวอย่าง)
rng(123); % ตั้งค่า random seed เพื่อผลลัพธ์ที่ซ้ำได้

n_train = 100;
n_test = 20;

% สร้างข้อมูล features แบบสุ่ม
moisture_vals = 25 + 40*rand(n_train,1);      % 25-65%
humidity_vals = 45 + 40*rand(n_train,1);      % 45-85% RH  
light_vals = 1500 + 4000*rand(n_train,1);     % 1500-5500 lux
co2_vals = 350 + 400*rand(n_train,1);         % 350-750 ppm

% สร้างข้อมูล targets ที่มีความสัมพันธ์กับ features
% สูตรจำลองการเจริญเติบโตของพืช
noise_factor = 0.1;

leaf_length = 8 + 0.15*moisture_vals + 0.08*humidity_vals + 0.002*light_vals + 0.01*co2_vals + ...
              noise_factor*8*randn(n_train,1);
leaf_length = max(6, min(20, leaf_length));

leaf_area = 10 + 0.2*moisture_vals + 0.12*humidity_vals + 0.003*light_vals + 0.015*co2_vals + ...
            noise_factor*10*randn(n_train,1);
leaf_area = max(8, min(30, leaf_area));

stem_girth = 1 + 0.05*moisture_vals + 0.02*humidity_vals + 0.0008*light_vals + 0.004*co2_vals + ...
             noise_factor*2*randn(n_train,1);
stem_girth = max(0.5, min(6, stem_girth));

% จัดข้อมูลฝึกสอน
X_train = [moisture_vals, humidity_vals, light_vals, co2_vals];
y_train = [leaf_length, leaf_area, stem_girth];

fprintf('   ✓ สร้างข้อมูลฝึกสอน: %d ตัวอย่าง\n', n_train);

% สร้างข้อมูลทดสอบ
moisture_test = 25 + 40*rand(n_test,1);
humidity_test = 45 + 40*rand(n_test,1);
light_test = 1500 + 4000*rand(n_test,1);
co2_test = 350 + 400*rand(n_test,1);

X_test = [moisture_test, humidity_test, light_test, co2_test];

% สร้างข้อมูลจริงสำหรับการประเมิน
leaf_length_true = 8 + 0.15*moisture_test + 0.08*humidity_test + 0.002*light_test + 0.01*co2_test + ...
                   noise_factor*8*randn(n_test,1);
leaf_length_true = max(6, min(20, leaf_length_true));

leaf_area_true = 10 + 0.2*moisture_test + 0.12*humidity_test + 0.003*light_test + 0.015*co2_test + ...
                 noise_factor*10*randn(n_test,1);
leaf_area_true = max(8, min(30, leaf_area_true));

stem_girth_true = 1 + 0.05*moisture_test + 0.02*humidity_test + 0.0008*light_test + 0.004*co2_test + ...
                  noise_factor*2*randn(n_test,1);
stem_girth_true = max(0.5, min(6, stem_girth_true));

y_test_true = [leaf_length_true, leaf_area_true, stem_girth_true];

fprintf('   ✓ สร้างข้อมูลทดสอบ: %d ตัวอย่าง\n', n_test);

% --- Multiple Linear Regression ---
fprintf('\n2. ฝึกสอนโมเดล Multiple Linear Regression...\n');

% Normalize features
X_mean = mean(X_train);
X_std = std(X_train);
X_train_norm = (X_train - X_mean) ./ X_std;
X_test_norm = (X_test - X_mean) ./ X_std;

% Add bias term (intercept)
X_train_norm = [ones(n_train,1), X_train_norm];
X_test_norm = [ones(n_test,1), X_test_norm];

% Train separate model for each target
target_names = {'Leaf Length', 'Leaf Area', 'Stem Girth'};
predictions_lr = zeros(n_test, 3);
coefficients = cell(3,1);

for i = 1:3
    % Linear regression: beta = (X'X)^(-1)X'y
    coefficients{i} = (X_train_norm' * X_train_norm) \ (X_train_norm' * y_train(:,i));
    predictions_lr(:,i) = X_test_norm * coefficients{i};
    
    fprintf('   ✓ ฝึกสอนโมเดลสำหรับ %s สำเร็จ\n', target_names{i});
end

% --- k-Nearest Neighbors ---
fprintf('\n3. ฝึกสอนโมเดล k-Nearest Neighbors (k=5)...\n');

k = 5;
predictions_knn = zeros(n_test, 3);

for i = 1:n_test
    % คำนวณระยะทาง Euclidean
    distances = sqrt(sum((X_train_norm(2:end,:) - X_test_norm(i,2:end)).^2, 2));
    
    % หา k ตัวใกล้ที่สุด
    [~, nearest_idx] = sort(distances);
    nearest_k = nearest_idx(1:k);
    
    % ทำนายโดยเฉลี่ยจาก k เพื่อนบ้าน
    predictions_knn(i,:) = mean(y_train(nearest_k,:));
end

fprintf('   ✓ ฝึกสอนโมเดล k-NN สำเร็จ\n');

% --- Ensemble (รวมผลทำนายจาก 2 โมเดล) ---
fprintf('\n4. รวมผลทำนายจากทั้ง 2 โมเดล...\n');

% น้ำหนักสำหรับแต่ละโมเดล (สามารถปรับได้)
weight_lr = 0.6;
weight_knn = 0.4;

predictions_ensemble = weight_lr * predictions_lr + weight_knn * predictions_knn;

fprintf('   ✓ รวมผลทำนายสำเร็จ (LR: %.1f%%, k-NN: %.1f%%)\n', weight_lr*100, weight_knn*100);

% --- ประเมินประสิทธิภาพ ---
fprintf('\n5. ประเมินประสิทธิภาพของโมเดล...\n');

function [rmse, mae, r2] = evaluate_performance(y_true, y_pred)
    mse = mean((y_true - y_pred).^2);
    rmse = sqrt(mse);
    mae = mean(abs(y_true - y_pred));
    ss_res = sum((y_true - y_pred).^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    r2 = 1 - (ss_res / ss_tot);
end

fprintf('\n--- ผลการประเมิน Linear Regression ---\n');
for i = 1:3
    [rmse, mae, r2] = evaluate_performance(y_test_true(:,i), predictions_lr(:,i));
    fprintf('%s: RMSE=%.3f, MAE=%.3f, R²=%.3f\n', target_names{i}, rmse, mae, r2);
end

fprintf('\n--- ผลการประเมิน k-NN ---\n');
for i = 1:3
    [rmse, mae, r2] = evaluate_performance(y_test_true(:,i), predictions_knn(:,i));
    fprintf('%s: RMSE=%.3f, MAE=%.3f, R²=%.3f\n', target_names{i}, rmse, mae, r2);
end

fprintf('\n--- ผลการประเมิน Ensemble ---\n');
for i = 1:3
    [rmse, mae, r2] = evaluate_performance(y_test_true(:,i), predictions_ensemble(:,i));
    fprintf('%s: RMSE=%.3f, MAE=%.3f, R²=%.3f\n', target_names{i}, rmse, mae, r2);
end

% --- Classification Evaluation ---
fprintf('\n6. ประเมินผลแบบ Classification...\n');

% กำหนด threshold สำหรับแต่ละตัวแปร
thresholds = [12, 18, 3]; % สำหรับ leaf_length, leaf_area, stem_girth

function [accuracy, precision, recall, f1] = classification_metrics(y_true, y_pred, threshold)
    true_class = y_true > threshold;
    pred_class = y_pred > threshold;
    
    tp = sum(true_class & pred_class);
    tn = sum(~true_class & ~pred_class);
    fp = sum(~true_class & pred_class);
    fn = sum(true_class & ~pred_class);
    
    accuracy = (tp + tn) / length(true_class);
    
    if (tp + fp) == 0
        precision = 0;
    else
        precision = tp / (tp + fp);
    end
    
    if (tp + fn) == 0
        recall = 0;
    else
        recall = tp / (tp + fn);
    end
    
    if (precision + recall) == 0
        f1 = 0;
    else
        f1 = 2 * precision * recall / (precision + recall);
    end
end

fprintf('\n--- Classification Performance (Ensemble) ---\n');
for i = 1:3
    [acc, prec, rec, f1] = classification_metrics(y_test_true(:,i), predictions_ensemble(:,i), thresholds(i));
    fprintf('%s (threshold=%.1f): Accuracy=%.2f%%, Precision=%.2f%%, Recall=%.2f%%, F1=%.2f%%\n', ...
        target_names{i}, thresholds(i), acc*100, prec*100, rec*100, f1*100);
end

% --- Feature Importance (จากค่าสัมประสิทธิ์ของ Linear Regression) ---
fprintf('\n7. ความสำคัญของ Features...\n');

feature_names = {'Moisture', 'Humidity', 'Light Intensity', 'CO2 Concentration'};

fprintf('\n--- Feature Importance (Linear Regression Coefficients) ---\n');
for i = 1:3
    fprintf('\n%s:\n', target_names{i});
    coeff_abs = abs(coefficients{i}(2:end)); % ไม่รวม intercept
    [~, sorted_idx] = sort(coeff_abs, 'descend');
    
    total_importance = sum(coeff_abs);
    for j = 1:length(feature_names)
        importance_pct = coeff_abs(sorted_idx(j)) / total_importance * 100;
        fprintf('  %s: %.1f%%\n', feature_names{sorted_idx(j)}, importance_pct);
    end
end

% --- แสดงผลทำนาย ---
fprintf('\n8. ผลการทำนาย...\n');

% สร้างตารางผลลัพธ์
fprintf('\n=== ผลการทำนายการเจริญเติบโตของพืช ===\n');
fprintf('%-4s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
    'ID', 'LeafLen_True', 'LeafLen_Pred', 'LeafArea_True', 'LeafArea_Pred', 'StemGirth_True', 'StemGirth_Pred');
fprintf('%-4s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
    '---', '------------', '------------', '-------------', '-------------', '--------------', '--------------');

for i = 1:min(10, n_test) % แสดง 10 ตัวอย่างแรก
    fprintf('%-4d %-12.2f %-12.2f %-12.2f %-12.2f %-12.2f %-12.2f\n', ...
        i, y_test_true(i,1), predictions_ensemble(i,1), ...
        y_test_true(i,2), predictions_ensemble(i,2), ...
        y_test_true(i,3), predictions_ensemble(i,3));
end

if n_test > 10
    fprintf('... และอีก %d ตัวอย่าง\n', n_test - 10);
end

% --- บันทึกผลลัพธ์ ---
fprintf('\n9. บันทึกผลลัพธ์...\n');

% สร้างตาราง
results_table = table((1:n_test)', X_test(:,1), X_test(:,2), X_test(:,3), X_test(:,4), ...
    y_test_true(:,1), predictions_ensemble(:,1), ...
    y_test_true(:,2), predictions_ensemble(:,2), ...
    y_test_true(:,3), predictions_ensemble(:,3), ...
    'VariableNames', {'ID', 'Moisture', 'Humidity', 'LightIntensity', 'CO2Concentration', ...
    'LeafLength_True', 'LeafLength_Pred', 'LeafArea_True', 'LeafArea_Pred', ...
    'StemGirth_True', 'StemGirth_Pred'});

try
    writetable(results_table, 'plant_growth_results.csv');
    fprintf('   ✓ บันทึกผลลัพธ์ลงไฟล์ plant_growth_results.csv สำเร็จ\n');
catch
    fprintf('   ⚠ ไม่สามารถบันทึกไฟล์ CSV ได้\n');
end

% --- สรุปผล ---
fprintf('\n=== สรุปผลการทำงาน ===\n');
fprintf('✓ อัลกอริทึม: Multiple Linear Regression + k-Nearest Neighbors\n');
fprintf('✓ จำนวนข้อมูลฝึกสอน: %d ตัวอย่าง\n', n_train);
fprintf('✓ จำนวนข้อมูลทดสอบ: %d ตัวอย่าง\n', n_test);
fprintf('✓ Features: %s\n', strjoin(feature_names, ', '));
fprintf('✓ Targets: %s\n', strjoin(target_names, ', '));
fprintf('✓ โมเดลรวม: Linear Regression (%.0f%%) + k-NN (%.0f%%)\n', weight_lr*100, weight_knn*100);

% คำนวณประสิทธิภาพเฉลี่ย
avg_rmse = mean([evaluate_performance(y_test_true(:,1), predictions_ensemble(:,1)), ...
                 evaluate_performance(y_test_true(:,2), predictions_ensemble(:,2)), ...
                 evaluate_performance(y_test_true(:,3), predictions_ensemble(:,3))]);

fprintf('✓ RMSE เฉลี่ย: %.3f\n', avg_rmse);
fprintf('\nการประมวลผลเสร็จสิ้น! 🌱\n');

% --- ฟังก์ชันทำนายใหม่ (สำหรับใช้งานจริง) ---
fprintf('\n--- ตัวอย่างการใช้งานฟังก์ชันทำนาย ---\n');

function predictions = predict_plant_growth(moisture, humidity, light, co2, coefficients_cell, X_mean, X_std, X_train_norm, y_train, k)
    % Normalize input
    X_new = [moisture, humidity, light, co2];
    X_new_norm = (X_new - X_mean) ./ X_std;
    X_new_norm = [1, X_new_norm]; % Add bias
    
    % Linear Regression predictions
    pred_lr = zeros(1,3);
    for i = 1:3
        pred_lr(i) = X_new_norm * coefficients_cell{i};
    end
    
    % k-NN predictions
    distances = sqrt(sum((X_train_norm(2:end,:) - X_new_norm(2:end)).^2, 2));
    [~, nearest_idx] = sort(distances);
    nearest_k = nearest_idx(1:k);
    pred_knn = mean(y_train(nearest_k,:));
    
    % Ensemble
    weight_lr = 0.6;
    weight_knn = 0.4;
    predictions = weight_lr * pred_lr + weight_knn * pred_knn;
end

% ตัวอย่างการทำนาย
example_moisture = 50;    % 50%
example_humidity = 70;    % 70% RH
example_light = 3000;     % 3000 lux
example_co2 = 500;        % 500 ppm

example_pred = predict_plant_growth(example_moisture, example_humidity, example_light, example_co2, ...
                                   coefficients, X_mean, X_std, X_train_norm, y_train, k);

fprintf('ตัวอย่างการทำนาย:\n');
fprintf('Input: Moisture=%.0f%%, Humidity=%.0f%%RH, Light=%.0flux, CO2=%.0fppm\n', ...
    example_moisture, example_humidity, example_light, example_co2);
fprintf('Prediction: LeafLength=%.2f cm, LeafArea=%.2f cm², StemGirth=%.2f cm\n', ...
    example_pred(1), example_pred(2), example_pred(3));

fprintf('\n🎉 โปรแกรมพร้อมใช้งาน! สามารถปรับแต่งและใช้ฟังก์ชัน predict_plant_growth() ได้\n');