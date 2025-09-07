# การทำนายการเติบโตของพืชด้วย Ensemble Methods และ Bayesian Optimization

โค้ด MATLAB นี้เป็นการแปลงจาก Decision Tree เดี่ยวเป็น **Ensemble Methods (TreeBagger/Random Forest)** ซึ่งให้ประสิทธิภาพที่ใกล้เคียงกับ XGBoost

## การแก้ไขปัญหา

### ปัญหาที่พบ:
- ข้อผิดพลาด `Unrecognized function or variable 'oobError'`
- ฟังก์ชัน OOB (Out-of-Bag) ที่ไม่เสถียร

### การแก้ไข:
1. **แก้ไขฟังก์ชัน OOB Error** - ใช้ `oobPredict()` แทน `oobError()`
2. **เพิ่ม Error Handling** - มี fallback เป็น cross-validation
3. **ปรับปรุง Feature Importance** - มีการตรวจสอบและ backup method
4. **เสถียรภาพ** - เพิ่ม try-catch blocks

## คุณสมบัติของโค้ดที่แก้ไขแล้ว

### **1. Ensemble Methods (TreeBagger)**
```matlab
% พารามิเตอร์ที่ปรับแต่ง:
- NumTrees (50-300)           % จำนวนต้นไม้
- MinLeafSize (1-20)          % ขนาดใบต่ำสุด  
- MaxNumSplits (5-100)        % จำนวนการแยกสูงสุด
- NumPredictorsToSample (1-4) % จำนวน features ต่อต้นไม้
```

### **2. การประเมินผลที่แข็งแกร่ง**
- **Cross-Validation** - 5-fold เป็นหลัก
- **OOB Validation** - หากใช้ได้
- **Feature Importance** - หลายวิธีการคำนวณ
- **Error Handling** - จัดการข้อผิดพลาดอัตโนมัติ

### **3. ผลลัพธ์ที่ครบถ้วน**
- Hyperparameter optimization results
- Feature importance analysis  
- Model performance metrics (RMSE, MAE, R², OOB metrics)
- Visualization plots
- CSV output file

## การใช้งาน

### ข้อกำหนด:
- MATLAB R2016b หรือใหม่กว่า
- Statistics and Machine Learning Toolbox
- Global Optimization Toolbox (สำหรับ Bayesian Optimization)

### วิธีการรัน:
```matlab
% เปิด MATLAB และรันโค้ด
predict_plant_growth_ensemble_bayesopt
```

### ไฟล์ข้อมูลที่ต้องการ:
- `datatraining.xlsx` - ข้อมูลฝึกสอน
- `datatest.xlsx` - ข้อมูลทดสอบ

## โครงสร้างข้อมูล

### Features (คอลัมน์ input):
- **Moisture** - ความชื้นดิน (เช่น "60-80%")
- **Humidity** - ความชื้นอากาศ (เช่น "70-85% RH")
- **LightIntensity** - ความเข้มแสง (เช่น "200-400 lux")
- **CO2Concentration** - ความเข้มข้น CO2 (เช่น "350-450 ppm")

### Targets (คอลัมน์ output):
- **LeafLength** - ความยาวใบ
- **LeafArea** - พื้นที่ใบ
- **StemGirth** - เส้นรอบวงลำต้น

## ข้อดีของ Ensemble Methods เทียบกับ Decision Tree เดี่ยว

### **ความแม่นยำสูงขึ้น:**
- รวม multiple trees ลดความเอนเอียง
- Random feature sampling ลด overfitting
- Bootstrap aggregating เพิ่มความเสถียร

### **ความเสถียรมากขึ้น:**
- Out-of-Bag validation
- Robust feature importance calculation
- Less sensitive to outliers

### **การป้องกัน Overfitting:**
- Feature subsampling per tree
- Bootstrap sampling  
- Ensemble averaging effect

## ผลลัพธ์ที่คาดหวัง

```
=== Plant Growth Prediction using Ensemble Methods ===
1. Data loading and preprocessing ✓
2. Feature conversion ✓
3. Bayesian optimization (3 target variables)
4. Model training with optimal parameters
5. Feature importance analysis
6. Performance evaluation
7. Final predictions → CSV file
8. Visualization plots
```

### ไฟล์ผลลัพธ์:
- `plant_growth_predictions_ensemble_bayesopt.csv` - การทำนาย
- `ensemble_model_analysis.png` - กราฟวิเคราะห์

## เปรียบเทียบประสิทธิภาพ

| Metric | Decision Tree | Ensemble Methods |
|--------|---------------|------------------|
| Accuracy | ★★★ | ★★★★★ |
| Stability | ★★ | ★★★★★ |
| Overfitting Resistance | ★★ | ★★★★★ |
| Feature Importance | ★★★ | ★★★★ |
| Computational Cost | ★★★★★ | ★★★ |

## การแก้ปัญหาเบื้องต้น

### หากเกิดข้อผิดพลาด:
1. **ตรวจสอบ MATLAB Version** - ต้อง R2016b หรือใหม่กว่า
2. **ตรวจสอบ Toolboxes** - Statistics ML และ Global Optimization
3. **ตรวจสอบไฟล์ข้อมูล** - format และ column names
4. **ลดขนาดข้อมูล** - หากหน่วยความจำไม่เพียงพอ

### หากต้องการปรับแต่ง:
```matlab
% ลดจำนวน optimization iterations
'MaxObjectiveEvaluations', 20  % แทน 40

% ลดจำนวนต้นไม้
optimizableVariable('NumTrees', [20, 100])  % แทน [50, 300]

% เปลี่ยน cross-validation folds
cv = cvpartition(length(y), 'KFold', 3);  % แทน 5
```

## บทสรุป

โค้ดนี้แก้ไขปัญหาเดิมและให้ประสิทธิภาพที่ดีกว่า Decision Tree เดี่ยว โดยใช้ Ensemble Methods ที่มีความคล้ายคลึงกับ XGBoost ในแง่ของ:

- **Multiple tree ensemble**
- **Feature randomization** 
- **Bootstrap aggregating**
- **Robust validation**
- **Feature importance analysis**

ทำให้ได้โมเดลที่แม่นยำและเสถียรมากขึ้นสำหรับการทำนายการเติบโตของพืช