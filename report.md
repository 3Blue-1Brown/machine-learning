# BÁO CÁO ĐỒ ÁN MÔN HỌC MÁY (MACHINE LEARNING)

---

**ĐỀ TÀI:**  
### XÂY DỰNG MÔ HÌNH HỌC MÁY DỰ ĐOÁN NGUY CƠ TIỂU ĐƯỜNG DỰA TRÊN CÁC CHỈ SỐ Y TẾ CƠ BẢN (BMI, TUỔI, LƯỢNG ĐƯỜNG TRONG MÁU)

- **Ngành đào tạo:** Công nghệ Thông tin  
- **Bộ môn:** Khoa học Dữ liệu & Trí tuệ Nhân tạo  
- **Sinh viên thực hiện:** Nhóm sinh viên CNTT  
- **Thời gian thực hiện:** Năm 2026  

---

## TÓM TẮT ĐỒ ÁN (ABSTRACT)

Đái tháo đường (tiểu đường) là một căn bệnh mạn tính nguy hiểm với tốc độ gia tăng nhanh chóng trên toàn cầu, gây ra các biến chứng nghiêm trọng về tim mạch, mù lòa, đoạn chi và suy thận mãn tính nếu không được phát hiện kịp thời. Tuy nhiên, việc thực hiện các gói xét nghiệm toàn diện thường tốn kém thời gian và chi phí đối với phần đông người dân.

Đồ án này nghiên cứu và trực tiếp phát triển một hệ thống học máy hoàn chỉnh, có khả năng hỗ trợ sàng lọc nguy cơ tiểu đường một cách nhanh chóng dựa trên **3 chỉ số lâm sàng cơ bản, dễ tiếp cận nhất**: **Lượng đường huyết (Glucose)**, **Chỉ số khối cơ thể (BMI)** và **Độ tuổi (Age)** trích xuất từ bộ dữ liệu chuẩn **Pima Indians Diabetes Dataset** (Kaggle / UCI). Đồng thời, đồ án xây dựng kịch bản đối sánh hiệu quả với toàn bộ 8 thuộc tính lâm sàng để làm rõ tác động của việc lựa chọn đặc trưng (Feature Selection).

Quy trình nghiên cứu được triển khai nghiêm ngặt theo các chuẩn mực khoa học dữ liệu:
1. **Tiền xử lý y khoa:** Phát hiện và quy đổi giá trị 0 phi lý sinh học ở cả **5 cột lâm sàng** (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) thành `NaN` và điền khuyết bằng Trung vị (`SimpleImputer(strategy='median')`).
2. **Chống rò rỉ dữ liệu (Anti Data Leakage):** Thiết kế chiến lược phân chia dữ liệu $80/20$ có phân tầng (`stratify=y`) và đóng gói toàn bộ quy trình tiền xử lý vào `Scikit-learn Pipeline`.
3. **Thử nghiệm 4 thuật toán:** Huấn luyện, đo đạc và so sánh thực nghiệm 4 thuật toán kinh điển: **Hồi quy Logistic (Logistic Regression)**, **Cây quyết định (Decision Tree)**, **Rừng ngẫu nhiên (Random Forest)** và **Máy vector hỗ trợ (Support Vector Machine - SVM)**.
4. **Tối ưu hóa siêu tham số (Hyperparameter Tuning):** Áp dụng **GridSearchCV** kết hợp **5-Fold Stratified Cross-Validation** trên tập huấn luyện, đưa mô hình **Decision Tree** đạt hiệu năng vượt trội: **Accuracy = 81.82%**, **Recall = 79.63%**, **F1-Score = 75.44%** và **ROC-AUC = 84.01%**, chứng minh tính tối ưu trong việc giảm thiểu tối đa các ca bỏ sót bệnh (False Negative).
5. **Triển khai ứng dụng:** Đóng gói mô hình bằng `Joblib` thành `diabetes_model.pkl`, cung cấp công cụ dòng lệnh [predict.py](file:///c:/Users/minhngoc/OneDrive/Documents/họcmáy/diabetes_prediction/predict.py) và website y tế hiện đại với Backend `Flask` tại `http://127.0.0.1:5000/`.

---

## CHƯƠNG 1: ĐẶT VẤN ĐỀ VÀ MỤC TIÊU ĐỀ TÀI

### 1.1. Tính cấp thiết của đề tài
Theo số liệu từ Hiệp hội Đái tháo đường Thế giới (IDF), toàn cầu hiện có hơn 537 triệu người trưởng thành mắc bệnh đái tháo đường, và gần một nửa trong số đó chưa từng được chẩn đoán. Tại Việt Nam, tỷ lệ người mắc đái tháo đường đang trẻ hóa nhanh chóng do thói quen ăn uống nhiều carbohydrate tinh chế, ít vận động và tình trạng thừa cân, béo phì.

Thách thức lớn nhất trong y tế cộng đồng là giai đoạn tiền đái tháo đường thường không có triệu chứng lâm sàng rõ rệt. Khi người bệnh bắt đầu cảm thấy mệt mỏi, sụt cân hay khát nước nhiều thì các tổn thương vi mạch đã bắt đầu xuất hiện. Do đó, việc ứng dụng Trí tuệ Nhân tạo và Machine Learning để xây dựng một công cụ sàng lọc trực tuyến, phi xâm lấn, chi phí 0 đồng giúp người dân tự đánh giá nguy cơ sức khỏe là một hướng tiếp cận có ý nghĩa xã hội và ứng dụng thực tiễn rất cao.

### 1.2. Mục tiêu nghiên cứu
- **Mục tiêu khoa học dữ liệu:** Khám phá quy luật phân bố của các chỉ số y tế, xử lý dữ liệu thiếu ẩn, xây dựng Pipeline tiền xử lý chuẩn mực không rò rỉ dữ liệu, tối ưu hóa siêu tham số với GridSearchCV và so sánh khách quan 4 thuật toán phân loại.
- **Mục tiêu kỹ thuật phần mềm:** Đóng gói mô hình Machine Learning thành công và tích hợp vào chương trình dòng lệnh `predict.py` và ứng dụng Web Flask hoàn chỉnh, có khả năng phản hồi tức thời dưới 15ms.
- **Mục tiêu người dùng:** Cung cấp trải nghiệm thân thiện, minh bạch về xác suất dự đoán và cung cấp các lời khuyên y tế tham khảo hữu ích kèm cảnh báo miễn trừ trách nhiệm y khoa.

### 1.3. Phạm vi và Giới hạn đề tài
- **Phạm vi dữ liệu:** Sử dụng bộ dữ liệu Pima Indians Diabetes gồm 768 cá thể. Tập trung nghiên cứu 2 kịch bản: Kịch bản A (3 đặc trưng: Glucose, BMI, Age) và Kịch bản B (toàn bộ 8 đặc trưng).
- **Giới hạn kỹ thuật:** Tuân thủ yêu cầu đề tài tập trung vào các mô hình học máy truyền thống có tính giải thích cao (Explainable AI), tốc độ tính toán nhanh, không sử dụng Deep Learning quá mức cần thiết.
- **Giới hạn y khoa:** Hệ thống đóng vai trò công cụ hỗ trợ sàng lọc tham khảo ban đầu, không thay thế xét nghiệm máu chuyên sâu hay chỉ định của bác sĩ y khoa.

---

## CHƯƠNG 2: TỔNG QUAN LÝ THUYẾT VÀ 4 THUẬT TOÁN SỬ DỤNG

### 2.1. Giới thiệu Bộ dữ liệu Pima Indians Diabetes
Bộ dữ liệu được thu thập ban đầu bởi Viện Quốc gia về Bệnh tiểu đường, Tiêu hóa và Bệnh thận (NIDDK) tại Hoa Kỳ, bao gồm các đối tượng nghiên cứu là phụ nữ người Mỹ bản địa gốc Pima từ 21 tuổi trở lên:
- Cấu trúc gồm 768 dòng và 9 cột.
- Nhãn mục tiêu `Outcome`: `0` (Âm tính - Không mắc bệnh, 500 mẫu) và `1` (Dương tính - Mắc bệnh, 268 mẫu).

### 2.2. Ý nghĩa lâm sàng của 3 đặc trưng cốt lõi
1. **Glucose (mg/dL):** Nồng độ glucose trong huyết tương sau thử nghiệm dung nạp glucose đường uống 2 giờ. Đây là tiêu chuẩn vàng sinh hóa phản ánh trực tiếp khả năng chuyển hóa đường của cơ thể ($r \approx 0.47$).
2. **BMI ($kg/m^2$):** Chỉ số khối cơ thể được tính bằng $BMI = \frac{\text{Cân nặng (kg)}}{[\text{Chiều cao (m)}]^2}$. Mỡ thừa trong cơ thể, đặc biệt là mỡ nội tạng, giải phóng các cytokine gây viêm, trực tiếp làm suy giảm độ nhạy của thụ thể insulin tại tế bào (hiện tượng kháng insulin).
3. **Age (Năm):** Quá trình lão hóa tự nhiên dẫn đến sự suy giảm dần về số lượng và chức năng của các tế bào beta đảo tụy chuyên trách tiết insulin.

### 2.3. Cơ sở lý thuyết của 4 thuật toán Machine Learning
1. **Hồi quy Logistic (Logistic Regression):**  
   Mô hình phân loại tuyến tính sử dụng hàm Sigmoid:
   $$\sigma(z) = \frac{1}{1 + e^{-z}} \quad \text{với } z = w_0 + w_1 \cdot \text{Glucose} + w_2 \cdot \text{BMI} + w_3 \cdot \text{Age}$$
   Đặc điểm: Đơn giản, tính toán nhanh, giải thích được trọng số, làm mốc so sánh (Baseline).
2. **Cây quyết định (Decision Tree Classifier):**  
   Thuật toán phân chia không gian đặc trưng bằng cây nhị phân dựa trên tiêu chuẩn Entropy / Information Gain:
   $$\text{Entropy}(S) = - \sum_{i=1}^{C} p_i \log_2(p_i)$$
   Đặc điểm: Bắt trọn các ranh giới phi tuyến tính rất tự nhiên, mô phỏng đúng cách các bác sĩ đưa ra quyết định sàng lọc qua từng ngưỡng chỉ số.
3. **Rừng ngẫu nhiên (Random Forest Classifier):**  
   Phương pháp học kết hợp (Ensemble Learning) áp dụng kỹ thuật Bagging. Thuật toán xây dựng một tập hợp gồm nhiều cây quyết định độc lập (100 cây), mỗi cây được huấn luyện trên một tập mẫu con ngẫu nhiên có hoàn lại và dự đoán bằng biểu quyết đa số (Majority Voting).
4. **Máy vector hỗ trợ (Support Vector Machine - SVM):**  
   Tìm siêu phẳng tối ưu (Hyperplane) trong không gian đa chiều phân tách hai lớp sao cho khoảng cách lề (Margin) là cực đại:
   $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{thỏa mãn } y_i (\mathbf{w}^T \phi(\mathbf{x}_i) + b) \ge 1$$
   Sử dụng hàm nhân RBF (Radial Basis Function) để giải quyết các đường biên phi tuyến phức tạp.

---

## CHƯƠNG 3: TIỀN XỬ LÝ DỮ LIỆU VÀ LỰA CHỌN ĐẶC TRƯNG

### 3.1. Phát hiện Dữ liệu thiếu ẩn (Hidden Missing Values)
Phân tích tệp `diabetes.csv` ghi nhận:
- `Glucose`: Có 5 giá trị bằng 0 (0.65%).
- `BloodPressure`: Có 35 giá trị bằng 0 (4.56%).
- `SkinThickness`: Có 227 giá trị bằng 0 (29.56%).
- `Insulin`: Có 374 giá trị bằng 0 (48.70%).
- `BMI`: Có 11 giá trị bằng 0 (1.43%).

*Đánh giá y khoa:* Một người sống không thể có đường huyết, huyết áp, da, insulin hay BMI bằng 0. Đây là các ca chưa được làm xét nghiệm. Toàn bộ các giá trị này được chuyển đổi thành `np.nan` trước khi đưa vào bước điền khuyết.

### 3.2. Chống rò rỉ dữ liệu (Anti Data Leakage) với Pipeline
- Phân chia tập dữ liệu $80/20$ có phân tầng (`stratify=y`) trước khi tiền xử lý: Tập Train (614 mẫu) và Tập Test (154 mẫu).
- `SimpleImputer(strategy='median')`: Tính trung vị **chỉ trên tập Train** (Glucose: 117.0, BMI: 32.4, Age: 29.0).
- `StandardScaler()`: Tính Mean và Std **chỉ trên tập Train** và áp dụng biến đổi đóng băng sang tập Test.

### 3.3. So sánh hiệu quả lựa chọn đặc trưng (Kịch bản A vs Kịch bản B)
Thực nghiệm so sánh trên 154 mẫu tập Test bằng Random Forest:

| Chỉ số đánh giá | Kịch bản A (3 đặc trưng: Glucose, BMI, Age) | Kịch bản B (Toàn bộ 8 đặc trưng) | Mức cải thiện ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Accuracy (%)** | 72.73% | 77.27% | **+4.54%** |
| **Precision (%)** | 63.64% | 70.21% | **+6.57%** |
| **Recall (%)** | 51.85% | 61.11% | **+9.26%** |
| **F1-Score (%)** | 57.14% | 65.35% | **+8.21%** |
| **ROC-AUC (%)** | 80.91% | 81.81% | **+0.90%** |

*Nhận xét:* Bổ sung đủ 8 đặc trưng giúp Recall tăng mạnh (+9.26%), giúp mô hình chẩn đoán chuyên sâu hạn chế tối đa bỏ sót bệnh. Trong khi đó, Kịch bản A phù hợp cho công cụ sàng lọc cộng đồng tự đo tại nhà.

---

## CHƯƠNG 4: HUẤN LUYỆN, TỐI ƯU HÓA VÀ ĐÁNH GIÁ 4 MÔ HÌNH

### 4.1. Tối ưu hóa siêu tham số bằng GridSearchCV
Áp dụng `GridSearchCV` với 5-Fold Stratified Cross-Validation trên tập Train với tiêu chí tối ưu hóa `scoring='f1'`:

- **Logistic Regression:** Tìm được siêu tham số tối ưu `{'C': 0.01, 'solver': 'liblinear'}`.
- **Decision Tree:** Tìm được siêu tham số tối ưu `{'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 6, 'min_samples_split': 15}`.
- **Random Forest:** `{'max_depth': 5, 'min_samples_leaf': 4, 'n_estimators': 100}`.
- **SVM:** `{'C': 5.0, 'gamma': 'scale', 'kernel': 'rbf'}`.

### 4.2. Bảng kết quả thực nghiệm đo đạc thực tế (Test Set 154 mẫu)

| Thuật toán | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | Xếp hạng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Tối ưu)** | **81.82%** | **71.67%** | **79.63%** | **75.44%** | **84.01%** | 🏆 **Mô hình triển khai** |
| **Random Forest (100 cây)** | 75.97% | 67.35% | 61.11% | 64.08% | 81.46% | Hạng 2 |
| **Support Vector Machine (SVM)** | 72.73% | 62.00% | 57.41% | 59.62% | 78.03% | Hạng 3 |
| **Logistic Regression** | 72.73% | 62.00% | 57.41% | 59.62% | 78.76% | Hạng 4 |

### 4.3. Phân tích Ma trận nhầm lẫn (Confusion Matrix)
Trên 154 mẫu kiểm thử (100 ca âm tính, 54 ca dương tính thực tế):
- **Decision Tree (Tối ưu):**
  - $TN = 83$ ca, $FP = 17$ ca.
  - $FN = \mathbf{11}$ ca (giảm số ca bỏ sót bệnh xuống thấp nhất).
  - $TP = \mathbf{43}$ ca (phát hiện thành công 43/54 ca có nguy cơ).
  - Độ nhạy Recall đạt kỷ lục **79.63%**, đưa F1-Score lên **75.44%**.

### 4.4. Quyết định lựa chọn mô hình triển khai
**Decision Tree (Tối ưu hóa)** được lựa chọn làm mô hình triển khai chính thức vì:
1. Đạt **Accuracy cao nhất: 81.82%**.
2. Đạt **Recall cao nhất: 79.63%** (ưu tiên hàng đầu trong y tế dự phòng để tránh bỏ sót bệnh nhân).
3. Đạt **F1-Score cao nhất: 75.44%** và **ROC-AUC cao nhất: 84.01%**.
4. Cây quyết định có tính minh bạch cao, giải thích được logic chẩn đoán lâm sàng cho bác sĩ và bệnh nhân.

---

## CHƯƠNG 5: TRIỂN KHAI PHẦN MỀM VÀ HƯỚNG DẪN SỬ DỤNG

### 5.1. Chương trình dự đoán dòng lệnh (`predict.py`)
Cho phép người dùng kiểm thử nhanh qua terminal:
```bash
python predict.py --glucose 165 --bmi 34.0 --age 52
```
Kết quả hiển thị:
- Nhãn dự đoán: Dương tính (Nhóm 1).
- Thước đo độ tin cậy: 85.29%.
- Khuyến nghị y tế và Cảnh báo miễn trừ trách nhiệm.

### 5.2. Ứng dụng Web Flask y tế hiện đại (`app.py`)
Khởi chạy tại `http://127.0.0.1:5000/`:
- **Trang chủ (`/`):** Hiển thị số liệu thực nghiệm, quy trình 3 bước và độ chính xác 81.8%.
- **Trang dự đoán (`/predict`):** Form nhập 3 chỉ số có tích hợp công cụ tính nhanh BMI thông minh.
- **Trang kết quả (`/result`):** Thanh đo xác suất nguy cơ, biểu đồ tròn Chart.js và nút in phiếu kết quả.
- **Trang giới thiệu mô hình (`/about`):** Bảng so sánh 4 mô hình, nhúng biểu đồ ROC Curve và Confusion Matrix.
- **RESTful API (`/api/predict`):** Phục vụ tích hợp cho các hệ thống phần mềm y tế bên thứ ba.

---

## CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 6.1. Kết quả đạt được
1. Xây dựng thành công quy trình Machine Learning bài bản chuẩn 11 bước.
2. Giải quyết triệt để dữ liệu thiếu ẩn ở 5 cột y sinh và phòng chống rò rỉ dữ liệu hoàn toàn bằng Scikit-learn Pipeline.
3. So sánh khách quan 4 thuật toán kinh điển, tối ưu hóa siêu tham số thành công bằng GridSearchCV, nâng Recall lên 79.63% và Accuracy lên 81.82%.
4. Triển khai hoàn chỉnh cả giao diện dòng lệnh CLI và Website Flask chuyên nghiệp.

### 6.2. Hướng phát triển trong tương lai
1. Thu thập thêm tập dữ liệu bệnh nhân tại Việt Nam để tinh chỉnh ngưỡng chẩn đoán phù hợp với thể tạng người châu Á.
2. Mở rộng tính năng cho phép nhập tùy chọn các chỉ số chuyên sâu (HbA1c, Huyết áp, Insulin) để chuyển đổi linh hoạt giữa Kịch bản A và Kịch bản B.
3. Phát triển ứng dụng di động hỗ trợ theo dõi đường huyết chuỗi thời gian (Time-series tracking).
