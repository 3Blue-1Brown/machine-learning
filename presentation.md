# NỘI DUNG SLIDE THUYẾT TRÌNH & BỘ CÂU HỎI VẤN ĐÁP BẢO VỆ ĐỒ ÁN (Q&A)

---

## PHẦN 1: DÀN Ý SLIDE THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN (10 SLIDES CHUẨN)

### SLIDE 1: Trang tiêu đề
- **Tên đề tài:** Xây dựng mô hình Học máy dự đoán nguy cơ tiểu đường dựa trên các chỉ số y tế cơ bản (BMI, Tuổi, Lượng đường trong máu)
- **Học phần:** Học Máy (Machine Learning)
- **Sinh viên thực hiện:** Nhóm sinh viên CNTT
- **Giảng viên hướng dẫn:** Bộ môn Khoa học Dữ liệu & AI

### SLIDE 2: Đặt vấn đề & Mục tiêu nghiên cứu
- **Thực trạng:** Tiểu đường là bệnh mạn tính nguy hiểm, giai đoạn tiền đái tháo đường thường không có triệu chứng rõ rệt.
- **Thách thức:** Xét nghiệm máu tổng quát thường tốn kém thời gian và chi phí.
- **Giải pháp đề tài:** Xây dựng công cụ Machine Learning hỗ trợ sàng lọc sơ bộ nhanh chóng, phi xâm lấn dựa trên 3 chỉ số lâm sàng cơ bản: Glucose, BMI, Age.

### SLIDE 3: Khảo sát dữ liệu & Vấn đề dữ liệu thiếu ẩn
- Bộ dữ liệu chuẩn: **Pima Indians Diabetes Dataset** (768 ca, 9 thuộc tính).
- **Phát hiện quan trọng:** Các giá trị bằng 0 ở `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` là dữ liệu bị thiếu ẩn (người sống không thể có đường huyết hay BMI bằng 0).
- **Giải pháp:** Chuyển đổi toàn bộ số 0 ở 5 cột này thành `NaN`.

### SLIDE 4: Chiến lược tiền xử lý chống Rò rỉ dữ liệu (Anti Data Leakage)
- Chia tập dữ liệu Train (80% - 614 mẫu) và Test (20% - 154 mẫu) với `stratify=y` để bảo toàn tỷ lệ nhãn.
- Đóng gói quy trình vào **Scikit-learn Pipeline**:
  - `SimpleImputer(strategy='median')`: Tính trung vị **chỉ trên tập Train**.
  - `StandardScaler()`: Chuẩn hóa thang đo theo Mean và Std **chỉ trên tập Train**.
  - Áp dụng độc lập sang tập Test để đảm bảo tính khách quan tuyệt đối.

### SLIDE 5: So sánh 2 kịch bản lựa chọn đặc trưng
- **Kịch bản A (3 đặc trưng: Glucose, BMI, Age):** Accuracy 72.73%, Recall 51.85%, F1 57.14%. Phù hợp cho ứng dụng sàng lọc nhanh tại nhà.
- **Kịch bản B (Toàn bộ 8 đặc trưng):** Accuracy 77.27%, Recall 61.11%, F1 65.35%.
- **Kết luận:** Bổ sung đủ đặc trưng y khoa giúp tăng mạnh độ nhạy Recall (+9.26%), phù hợp cho chẩn đoán chuyên sâu tại bệnh viện.

### SLIDE 6: Huấn luyện và Tối ưu hóa 4 thuật toán (GridSearchCV)
- Thử nghiệm 4 thuật toán: **Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM)**.
- Áp dụng `GridSearchCV` với 5-Fold Stratified Cross-Validation trên tập Train:
  - Decision Tree tối ưu đạt siêu tham số: `criterion='entropy', max_depth=4, min_samples_leaf=6, min_samples_split=15`.

### SLIDE 7: Bảng kết quả thực nghiệm đo đạc thực tế (154 mẫu Test)
| Thuật toán | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Tối ưu)** | **81.82%** | **71.67%** | **79.63%** | **75.44%** | **84.01%** |
| **Random Forest (100 cây)** | 75.97% | 67.35% | 61.11% | 64.08% | 81.46% |
| **Support Vector Machine (SVM)** | 72.73% | 62.00% | 57.41% | 59.62% | 78.03% |
| **Logistic Regression** | 72.73% | 62.00% | 57.41% | 59.62% | 78.76% |

### SLIDE 8: Phân tích Ma trận nhầm lẫn & Đường cong ROC Curve
- **Recall của Decision Tree đạt 79.63%:** Phát hiện chính xác 43/54 ca có bệnh, giảm số ca bỏ sót bệnh (False Negative) xuống mức thấp nhất (chỉ 11 ca).
- **ROC-AUC đạt 84.01%:** Khả năng phân biệt bệnh nhân trên dải ngưỡng xác suất vượt trội.

### SLIDE 9: Sản phẩm phần mềm thực tế
- **Chương trình dòng lệnh `predict.py`:** Tiếp nhận tham số, phân loại nhãn và in phiếu kết quả tức thì.
- **Ứng dụng Web Flask y tế:** Giao diện trực quan, tính nhanh BMI, thanh đo xác suất, biểu đồ tròn Chart.js và tài liệu RESTful API.

### SLIDE 10: Kết luận & Hướng phát triển
- Đồ án hoàn thành bài bản đủ 11 bước khoa học dữ liệu.
- Định hướng tương lai: Mở rộng tập dữ liệu thể tạng người Việt Nam và tích hợp ứng dụng di động theo dõi đường huyết theo chuỗi thời gian.

---

## PHẦN 2: BỘ CÂU HỎI VẤN ĐÁP BẢO VỆ ĐỒ ÁN (Q&A) ĐIỂM 10

#### Câu 1: Tại sao trong bài toán y tế này, nhóm em lại ưu tiên chỉ số Recall hơn là Accuracy?
> **Trả lời:**  
> Trong sàng lọc y tế, hậu quả của hai loại sai lầm là hoàn toàn khác nhau:
> - **False Positive (Báo động giả):** Người khỏe mạnh bị mô hình báo có nguy cơ $\rightarrow$ Bệnh nhân đi xét nghiệm máu lại, phát hiện âm tính, không gây nguy hiểm tính mạng.
> - **False Negative (Bỏ sót bệnh):** Người thực sự mắc bệnh nhưng mô hình kết luận bình thường $\rightarrow$ Bệnh nhân chủ quan không điều trị, dẫn đến biến chứng suy thận, mù lòa hay đột quỵ.  
> Do đó, mục tiêu hàng đầu là **tối đa hóa Recall** để giảm thiểu tối đa số ca False Negative. Decision Tree tối ưu đạt Recall **79.63%** là lý do chính mô hình này được lựa chọn.

#### Câu 2: Em đã làm gì để phòng chống Data Leakage (Rò rỉ dữ liệu) trong quá trình tiền xử lý?
> **Trả lời:**  
> Nhóm em phòng chống Data Leakage bằng 2 kỹ thuật nghiêm ngặt:
> 1. Chia dữ liệu thành tập Train (80%) và Test (20%) trước tiên bằng `train_test_split(stratify=y)`.
> 2. Đóng gói toàn bộ bộ điền khuyết (`SimpleImputer`) và bộ chuẩn hóa (`StandardScaler`) vào `Pipeline`. Quá trình `.fit()` chỉ được phép học giá trị trung vị, trung bình và độ lệch chuẩn **nghiêm ngặt trên tập Train**. Tập Test chỉ được áp dụng `.transform()`, tuyệt đối không tham gia vào quá trình tính toán thông số tiền xử lý.

#### Câu 3: Tại sao lại chọn điền khuyết bằng Trung vị (Median) mà không dùng Trung bình (Mean)?
> **Trả lời:**  
> Các chỉ số sinh hóa y tế như Glucose và Insulin có phân bố lệch phải (Right-skewed) và chứa một số giá trị ngoại lai rất cao (như Glucose 199 mg/dL hay Insulin 846 mu U/ml). Giá trị trung bình (Mean) rất nhạy cảm và dễ bị kéo lệch bởi các giá trị ngoại lai này. Trong khi đó, **Trung vị (Median)** là giá trị chia đôi mẫu, có tính bền vững (Robust) cao hơn nhiều, phản ánh chính xác thể trạng của đa số dân số mẫu.

#### Câu 4: Vì sao Decision Tree đơn lẻ sau khi tối ưu lại có kết quả tốt hơn Random Forest trên bộ dữ liệu này?
> **Trả lời:**  
> Trên không gian đặc trưng tối giản chỉ gồm 3 thuộc tính (Glucose, BMI, Age), ranh giới phân tách y khoa có tính phân tầng ngưỡng rất rõ ràng (ví dụ: các mốc Glucose > 126 mg/dL và BMI > 25 kg/m²).  
> Khi áp dụng Random Forest, cơ chế lấy ngẫu nhiên tập con đặc trưng (Subsampling of features) và biểu quyết đa số của 100 cây đôi khi làm loãng sức mạnh phân tách trực diện của thuộc tính Glucose. Trong khi đó, một Cây quyết định đơn lẻ được khống chế độ sâu `max_depth=4`, tiêu chuẩn `entropy` và `min_samples_split=15` đã tập trung khai thác triệt để các ngưỡng cắt y khoa mà không bị phân tán, đem lại Recall và Accuracy cao hơn.

#### Câu 5: Hệ thống web của em nạp mô hình như thế nào? Có phải mỗi lần người dùng bấm dự đoán thì web lại huấn luyện lại không?
> **Trả lời:**  
> Tuyệt đối không. Hệ thống tuân thủ kiến trúc phân tách giữa **Training Phase** và **Inference Phase**:
> - Mô hình chỉ được huấn luyện 1 lần duy nhất từ dữ liệu và đóng gói thành file nhị phân `diabetes_model.pkl`.
> - Khi server Flask khởi động, file `.pkl` được nạp vào bộ nhớ RAM duy nhất 1 lần bằng thư viện `Joblib`.
> - Khi người dùng bấm dự đoán trên web, server chỉ gọi hàm `predict()` và `predict_proba()` trên dữ liệu đầu vào. Thời gian phản hồi chỉ mất dưới **15 mili-giây**, đảm bảo hiệu năng phục vụ hàng nghìn lượt truy cập đồng thời.
