"""
BƯỚC 8: Đánh giá mô hình - Confusion Matrix, ROC Curve và Bảng so sánh 4 thuật toán
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# Thiết lập phong cách đồ họa
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'diabetes.csv')
IMAGES_DIR = os.path.join(BASE_DIR, 'static', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

features = ['Glucose', 'BMI', 'Age']
X = df[features]
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=4, min_samples_split=10, min_samples_leaf=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5, min_samples_leaf=4, n_jobs=-1),
    'Support Vector Machine (SVM)': SVC(random_state=42, kernel='rbf', C=1.0, probability=True)
}

results = {}
cms = {}
roc_data = {}

for name, clf in models.items():
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100
    auc = roc_auc_score(y_test, y_prob) * 100
    
    results[name] = {
        'Accuracy (%)': round(acc, 2),
        'Precision (%)': round(prec, 2),
        'Recall (%)': round(rec, 2),
        'F1-Score (%)': round(f1, 2),
        'ROC-AUC (%)': round(auc, 2)
    }
    cms[name] = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, auc)

# 1. XUẤT BẢNG SO SÁNH HIỆU NĂNG TỔNG HỢP
print("=" * 80)
print("BẢNG SO SÁNH HIỆU NĂNG THỰC TẾ 4 MÔ HÌNH TRÊN TẬP KIỂM THỬ (154 MẪU)")
print("=" * 80)
metrics_df = pd.DataFrame(results).T
print(metrics_df.to_string())

# 2. VẼ ĐƯỜNG CONG ROC CURVE
plt.figure(figsize=(8, 6.5), dpi=300)
colors = ['#3182ce', '#38a169', '#dd6b20', '#805ad5']
for (name, (fpr, tpr, auc_val)), c in zip(roc_data.items(), colors):
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.2f}%)', color=c, lw=2.2)

plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Phân loại ngẫu nhiên (AUC = 50%)')
plt.title('Đường cong ROC Curve so sánh 4 mô hình Machine Learning', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Tỷ lệ Dương tính giả (False Positive Rate - 1 - Specificity)', fontsize=10)
plt.ylabel('Tỷ lệ Dương tính thật (True Positive Rate - Recall)', fontsize=10)
plt.legend(loc='lower right', frameon=True, fontsize=9.5)
plt.tight_layout()
roc_path = os.path.join(IMAGES_DIR, 'roc_curves.png')
plt.savefig(roc_path)
plt.close()
print(f"\n✅ Đã xuất biểu đồ ROC Curve: {roc_path}")

# 3. VẼ MA TRẬN NHẦM LẪN 4 MÔ HÌNH (LƯỚI 2x2)
fig, axes = plt.subplots(2, 2, figsize=(10.5, 9), dpi=300)
axes = axes.flatten()
for i, (name, cm) in enumerate(cms.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'},
                xticklabels=['Dự đoán 0', 'Dự đoán 1'],
                yticklabels=['Thực tế 0', 'Thực tế 1'])
    axes[i].set_title(f"{name}\nAcc: {results[name]['Accuracy (%)']:.1f}% | Rec: {results[name]['Recall (%)']:.1f}%",
                      fontsize=11, fontweight='bold', pad=10)
    axes[i].set_ylabel('Nhãn thực tế', fontsize=10)
    axes[i].set_xlabel('Nhãn dự đoán', fontsize=10)

plt.suptitle('Ma trận nhầm lẫn (Confusion Matrix) 4 mô hình trên 154 mẫu tập Test', fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout()
cm_path = os.path.join(IMAGES_DIR, 'confusion_matrix_all.png')
plt.savefig(cm_path)
plt.close()
print(f"✅ Đã xuất biểu đồ Confusion Matrix: {cm_path}")

# 4. VẼ BIỂU ĐỒ CỘT SO SÁNH 4 CHỈ SỐ
fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=300)
metrics_to_plot = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']
x = np.arange(len(metrics_to_plot))
width = 0.20

for i, (name, scores) in enumerate(results.items()):
    vals = [scores[m] for m in metrics_to_plot]
    bars = ax.bar(x + (i - 1.5) * width, vals, width, label=name, color=colors[i], alpha=0.9, edgecolor='white')
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_title('So sánh các chỉ số đánh giá giữa 4 thuật toán trên tập Test', fontsize=12, fontweight='bold', pad=15)
ax.set_ylabel('Tỷ lệ phần trăm (%)', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot, fontsize=10, fontweight='bold')
ax.set_ylim(0, 105)
ax.legend(loc='lower right', frameon=True, fontsize=9)
plt.tight_layout()
comp_path = os.path.join(IMAGES_DIR, 'model_comparison.png')
plt.savefig(comp_path)
plt.close()
print(f"✅ Đã xuất biểu đồ so sánh cột: {comp_path}")

