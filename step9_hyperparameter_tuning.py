"""
BƯỚC 9: Tối ưu hóa siêu tham số (Hyperparameter Tuning) bằng GridSearchCV & StratifiedKFold
"""
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'diabetes.csv')
    df = pd.read_csv(DATA_PATH)

    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_cols] = df[zero_cols].replace(0, np.nan)

    features = ['Glucose', 'BMI', 'Age']
    X = df[features]
    y = df['Outcome']

    # CHIA TẬP TRAIN (80%) VÀ TEST (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # THIẾT LẬP STRATIFIED K-FOLD ĐÁNH GIÁ CHÉO 5-FOLD TRÊN TẬP TRAIN
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=" * 85)
    print("BẮT ĐẦU TỐI ƯU SIÊU THAM SỐ VỚI GRIDSEARCHCV TRÊN 5-FOLD CROSS VALIDATION (TẬP TRAIN)")
    print("=" * 85)

    # Cấu hình không gian tham số (Parameter Grids) cho từng thuật toán
    tuning_configs = {
        'Logistic Regression': {
            'pipeline': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            'params': {
                'classifier__C': [0.01, 0.1, 1.0, 10.0],
                'classifier__solver': ['lbfgs', 'liblinear']
            }
        },
        'Decision Tree': {
            'pipeline': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', DecisionTreeClassifier(random_state=42))
            ]),
            'params': {
                'classifier__max_depth': [3, 4, 5, 6],
                'classifier__min_samples_split': [5, 10, 15],
                'classifier__min_samples_leaf': [2, 4, 6],
                'classifier__criterion': ['gini', 'entropy']
            }
        },
        'Random Forest': {
            'pipeline': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(random_state=42))
            ]),
            'params': {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [3, 4, 5],
                'classifier__min_samples_leaf': [2, 4]
            }
        },
        'Support Vector Machine (SVM)': {
            'pipeline': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', SVC(random_state=42, probability=True))
            ]),
            'params': {
                'classifier__C': [0.1, 1.0, 5.0],
                'classifier__kernel': ['rbf', 'linear'],
                'classifier__gamma': ['scale', 'auto']
            }
        }
    }

    best_models = {}
    tuning_summary = []

    for name, cfg in tuning_configs.items():
        print(f"\n---> Đang tìm kiếm siêu tham số tối ưu: 【{name}】...")
        
        # GridSearchCV chỉ chạy trên X_train (Tuyệt đối không chạm vào X_test)
        grid_search = GridSearchCV(
            estimator=cfg['pipeline'],
            param_grid=cfg['params'],
            scoring='f1',        # Ưu tiên tối ưu hóa F1-Score cho bài toán y tế
            cv=cv,
            n_jobs=1,            # Chạy an toàn tuần tự trên Windows
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_models[name] = best_estimator
        
        # Đánh giá mô hình đã tối ưu trên tập kiểm thử độc lập (Test Set 154 mẫu)
        y_pred = best_estimator.predict(X_test)
        y_prob = best_estimator.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, zero_division=0) * 100
        rec = recall_score(y_test, y_pred, zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, zero_division=0) * 100
        auc = roc_auc_score(y_test, y_prob) * 100
        
        tuning_summary.append({
            'Thuật toán': name,
            'Siêu tham số tối ưu': str({k.replace('classifier__', ''): v for k, v in best_params.items()}),
            'Accuracy (%)': round(acc, 2),
            'Recall (%)': round(rec, 2),
            'F1-Score (%)': round(f1, 2),
            'ROC-AUC (%)': round(auc, 2)
        })
        
        print(f"     + Điểm CV F1-Score trung bình: {grid_search.best_score_*100:.2f}%")
        print(f"     + Tham số tốt nhất           : { {k.replace('classifier__', ''): v for k, v in best_params.items()} }")
        print(f"     + Đánh giá trên tập Test     : Acc={acc:.2f}% | Recall={rec:.2f}% | F1={f1:.2f}% | AUC={auc:.2f}%")

    print("\n" + "=" * 85)
    print("BẢNG TỔNG HỢP KẾT QUẢ SAU KHI TỐI ƯU SIÊU THAM SỐ (TEST SET 154 MẪU)")
    print("=" * 85)
    summary_df = pd.DataFrame(tuning_summary)
    print(summary_df[['Thuật toán', 'Accuracy (%)', 'Recall (%)', 'F1-Score (%)', 'ROC-AUC (%)']].to_string(index=False))

if __name__ == '__main__':
    main()
