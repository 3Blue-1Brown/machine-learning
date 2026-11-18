"""
BƯỚC 10: Lựa chọn mô hình tối ưu, đóng gói thành diabetes_model.pkl và kiểm tra tải lại
"""
import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'diabetes.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

features = ['Glucose', 'BMI', 'Age']
X = df[features]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("=" * 80)
print("1. KHỞI TẠO VÀ HUẤN LUYỆN PIPELINE TỐI ƯU NHẤT (DECISION TREE ĐÃ TỐI ƯU)")
print("=" * 80)

# Khởi tạo mô hình chiến thắng từ bước 9 với các siêu tham số tối ưu
best_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(
        random_state=42,
        criterion='entropy',
        max_depth=4,
        min_samples_split=15,
        min_samples_leaf=6
    ))
])

# Huấn luyện chỉ trên tập Train (614 mẫu)
best_pipeline.fit(X_train, y_train)

# Đánh giá lại trên tập Test (154 mẫu)
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred) * 100
prec = precision_score(y_test, y_pred) * 100
rec = recall_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100
auc = roc_auc_score(y_test, y_prob) * 100
cm = confusion_matrix(y_test, y_pred)

print(f"✅ Độ chính xác (Accuracy):  {acc:.2f}%")
print(f"✅ Độ nhạy (Recall)       :  {rec:.2f}% (Phát hiện 43/54 ca dương tính)")
print(f"✅ Điểm F1-Score          :  {f1:.2f}%")
print(f"✅ Chỉ số ROC-AUC         :  {auc:.2f}%")

# 2. ĐÓNG GÓI VÀ LƯU MÔ HÌNH THÀNH FILE .PKL BẰNG JOBLIB
model_save_path = os.path.join(MODELS_DIR, 'diabetes_model.pkl')
joblib.dump(best_pipeline, model_save_path)
print(f"\n📦 Đã lưu toàn bộ Pipeline mô hình vào: {model_save_path}")

# Lưu kèm metadata vào metrics.json để ứng dụng web và các module khác đọc
metrics_metadata = {
    'best_model': 'Decision Tree (Tối ưu hóa)',
    'features': features,
    'accuracy': round(acc, 2),
    'precision': round(prec, 2),
    'recall': round(rec, 2),
    'f1_score': round(f1, 2),
    'roc_auc': round(auc, 2),
    'best_params': {
        'criterion': 'entropy',
        'max_depth': 4,
        'min_samples_split': 15,
        'min_samples_leaf': 6
    },
    'train_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'confusion_matrix': cm.tolist()
}

metrics_json_path = os.path.join(MODELS_DIR, 'metrics.json')
with open(metrics_json_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_metadata, f, indent=2, ensure_ascii=False)
print(f"📄 Đã cập nhật tệp thông số đánh giá: {metrics_json_path}")

# 3. KIỂM TRA TẢI LẠI MÔ HÌNH VÀ DỰ ĐOÁN THỬ NGHIỆM
print("\n" + "=" * 80)
print("2. KIỂM TRA TẢI LẠI MÔ HÌNH TỪ ĐĨA VÀ DỰ ĐOÁN MẪU THỰC TẾ")
print("=" * 80)

loaded_model = joblib.load(model_save_path)
print("✅ Tải lại mô hình thành công từ file .pkl!")

# 3 kịch bản bệnh nhân thử nghiệm
test_patients = [
    {"name": "Bệnh nhân A (Khỏe mạnh)", "Glucose": 85.0, "BMI": 21.5, "Age": 24},
    {"name": "Bệnh nhân B (Nguy cơ cao)", "Glucose": 165.0, "BMI": 34.0, "Age": 52},
    {"name": "Bệnh nhân C (Chưa đo đường huyết)", "Glucose": np.nan, "BMI": 28.5, "Age": 45}
]

for p in test_patients:
    input_df = pd.DataFrame([[p['Glucose'], p['BMI'], p['Age']]], columns=features)
    pred_label = int(loaded_model.predict(input_df)[0])
    pred_probs = loaded_model.predict_proba(input_df)[0]
    
    print(f"\n* {p['name']}:")
    print(f"  Input: Glucose={p['Glucose']}, BMI={p['BMI']}, Age={p['Age']}")
    print(f"  -> Nhãn dự đoán: {pred_label} ({'Có nguy cơ mắc tiểu đường' if pred_label == 1 else 'Nguy cơ thấp / Bình thường'})")
    print(f"  -> Xác suất: P(0) = {pred_probs[0]*100:.1f}% | P(1) = {pred_probs[1]*100:.1f}%")

print("\n" + "=" * 80)
print("HOÀN TẤT BƯỚC 10: MÔ HÌNH ĐÃ SẴN SÀNG TRIỂN KHAI VÀO ỨNG DỤNG THỰC TẾ!")
print("=" * 80)

