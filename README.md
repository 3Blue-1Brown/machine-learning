# HỆ THỐNG HỖ TRỢ DỰ ĐOÁN NGUY CƠ TIỂU ĐƯỜNG (DIAPREDICT ML)

> **Đồ án môn học:** Học Máy (Machine Learning)  
> **Bộ dữ liệu:** Pima Indians Diabetes Dataset (Kaggle / UCI Machine Learning Repository)  
> **Công nghệ:** Python 3.14, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Flask, Joblib  
> **Hiệu năng mô hình tối ưu (Decision Tree):** **Accuracy: 81.82%** | **Recall: 79.63%** | **F1-Score: 75.44%** | **ROC-AUC: 84.01%**

---

## 1. GIỚI THIỆU ĐỀ TÀI
Tiểu đường (Đái tháo đường) là một trong những bệnh mạn tính nguy hiểm hàng đầu, gây ra nhiều biến chứng nghiêm trọng về tim mạch, suy thận và thị lực nếu không được phát hiện kịp thời.

Đồ án này nghiên cứu và trực tiếp phát triển một giải pháp Machine Learning hoàn chỉnh hỗ trợ sàng lọc sớm nguy cơ mắc bệnh tiểu đường dựa trên **3 chỉ số lâm sàng cơ bản và phổ biến nhất**:
1. **Glucose (Lượng đường huyết trong huyết tương - mg/dL)**
2. **BMI (Chỉ số khối cơ thể - $kg/m^2$)**
3. **Age (Độ tuổi - Năm)**

Đồng thời, đồ án tiến hành đối sánh khoa học giữa **Kịch bản A (3 đặc trưng cơ bản)** và **Kịch bản B (Toàn bộ 8 đặc trưng)** để làm rõ vai trò của kỹ thuật lựa chọn đặc trưng (Feature Selection).

---

## 2. CẤU TRÚC THƯ MỤC DỰ ÁN
```text
diabetes_prediction/
│
├── dataset/
│   └── diabetes.csv           # Dữ liệu chuẩn Pima Indians Diabetes (768 dòng, 9 cột)
│
├── models/
│   ├── diabetes_model.pkl     # Pipeline tối ưu đã lưu (Imputer + Scaler + DecisionTree)
│   └── metrics.json           # Thông số đo đạc thực tế của 4 mô hình & siêu tham số
│
├── notebooks/
│   └── diabetes_analysis.ipynb # Jupyter Notebook phân tích EDA và huấn luyện từng bước
│
├── static/
│   ├── css/style.css          # Giao diện y tế hiện đại (Responsive, Blue & White)
│   ├── js/script.js           # Form validation, tính nhanh BMI & hiệu ứng UX
│   └── images/                # Biểu đồ trực quan hóa y tế chất lượng cao
│       ├── outcome_distribution.png
│       ├── feature_distributions.png
│       ├── correlation_matrix.png
│       ├── confusion_matrix_all.png
│       ├── roc_curves.png
│       └── model_comparison.png
│
├── templates/                 # Giao diện Web Flask (HTML5 & Jinja2)
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   └── about.html
│
├── app.py                     # Máy chủ Flask Backend & RESTful API (/api/predict)
├── predict.py                 # Chương trình dự đoán bệnh nhân mới qua dòng lệnh CLI
├── train_model.py             # Script huấn luyện tự động
├── step8_evaluation.py        # Đánh giá 4 mô hình, vẽ ROC Curve & Confusion Matrix
├── step9_hyperparameter_tuning.py # Tối ưu hóa siêu tham số (GridSearchCV & 5-Fold CV)
├── step10_save_best_model.py  # Đóng gói mô hình tốt nhất vào models/
├── requirements.txt           # Danh mục thư viện phụ thuộc
├── report.md                  # Báo cáo học thuật toàn diện đồ án
├── presentation.md            # Slide thuyết trình & Bộ câu hỏi vấn đáp Q&A bảo vệ đồ án
└── README.md
```

---

## 3. CÀI ĐẶT MÔI TRƯỜNG & THƯ VIỆN

```bash
# 1. Di chuyển vào thư mục dự án
cd diabetes_prediction

# 2. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 4. HƯỚNG DẪN THỰC THI DỰ ÁN

### 4.1. Huấn luyện mô hình & Tối ưu hóa siêu tham số (GridSearchCV)
Lệnh này sẽ thực hiện tìm kiếm siêu tham số tối ưu qua 5-Fold Stratified Cross-Validation:
```bash
python step9_hyperparameter_tuning.py
```

### 4.2. Đóng gói mô hình tối ưu nhất vào `models/diabetes_model.pkl`
```bash
python step10_save_best_model.py
```

### 4.3. Thử nghiệm dự đoán bệnh nhân mới bằng dòng lệnh CLI
- **Cách 1: Nhập tham số trực tiếp:**
  ```bash
  python predict.py --glucose 165 --bmi 34.0 --age 52
  ```
- **Cách 2: Chế độ tương tác nhập tay:**
  ```bash
  python predict.py
  ```

### 4.4. Khởi chạy Ứng dụng Web Flask
```bash
python app.py
```
Mở trình duyệt web bất kỳ và truy cập: 👉 **`http://127.0.0.1:5000/`**

---

## 5. KẾT QUẢ THỰC NGHIỆM ĐO ĐẠC THỰC TẾ (TEST SET 154 MẪU)

| Thuật toán | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | Xếp hạng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Tối ưu)** | **81.82%** | **71.67%** | **79.63%** | **75.44%** | **84.01%** | 🏆 **Mô hình chọn** |
| **Random Forest (100 cây)** | 75.97% | 67.35% | 61.11% | 64.08% | 81.46% | Hạng 2 |
| **Support Vector Machine (SVM)** | 72.73% | 62.00% | 57.41% | 59.62% | 78.03% | Hạng 3 |
| **Logistic Regression** | 72.73% | 62.00% | 57.41% | 59.62% | 78.76% | Hạng 4 |

> 🏆 **Lý do lựa chọn mô hình Decision Tree:**  
> Trong sàng lọc y tế, **Recall (Độ nhạy)** là tiêu chí được ưu tiên số 1 nhằm tránh bỏ sót bệnh nhân (False Negative). **Decision Tree** đạt **Recall 79.63%** (phát hiện 43/54 ca bệnh), đồng thời giữ **Accuracy cao nhất (81.82%)** và tính giải thích minh bạch cao (Explainable AI).

---

## 6. TUYÊN BỐ MIỄN TRỪ TRÁCH NHIỆM Y KHOA
Dự án được xây dựng trong khuôn khổ đồ án môn học **Học Máy (Machine Learning)** nhằm mục đích học tập thuật toán và minh họa công nghệ, **tuyệt đối không có giá trị thay thế chẩn đoán chuyên môn, xét nghiệm máu hoặc phác đồ điều trị của bác sĩ chuyên khoa**.
