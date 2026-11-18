"""
Chương trình: predict.py
Mục đích: Dự đoán nguy cơ tiểu đường cho bệnh nhân mới qua giao diện dòng lệnh (CLI).
Sử dụng mô hình Pipeline tối ưu đã lưu tại models/diabetes_model.pkl.
"""

import sys
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import argparse
import joblib
import numpy as np
import pandas as pd

# Đường dẫn đến mô hình đã đóng gói
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'diabetes_model.pkl')

# Thứ tự thuộc tính đầu vào BẮT BUỘC phải trùng khớp 100% với lúc huấn luyện
FEATURE_NAMES = ['Glucose', 'BMI', 'Age']


def load_trained_model():
    """Hàm tải mô hình Pipeline từ đĩa."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy file mô hình tại '{MODEL_PATH}'. "
            f"Vui lòng chạy 'python step10_save_best_model.py' hoặc 'python train_model.py' trước!"
        )
    return joblib.load(MODEL_PATH)


def validate_input(glucose, bmi, age):
    """Kiểm tra tính hợp lệ về mặt sinh học của số liệu nhập vào."""
    try:
        g = float(glucose)
        b = float(bmi)
        a = float(age)
    except (ValueError, TypeError):
        return None, "Lỗi: Tất cả các chỉ số phải là số thực hoặc số nguyên hợp lệ."

    if g < 0 or g > 500:
        return None, "Lỗi: Lượng đường huyết (Glucose) phải nằm trong khoảng từ 0 đến 500 mg/dL."
    if b < 0 or b > 100:
        return None, "Lỗi: Chỉ số khối cơ thể (BMI) phải nằm trong khoảng từ 0 đến 100 kg/m²."
    if a < 1 or a > 125:
        return None, "Lỗi: Độ tuổi (Age) phải nằm trong khoảng từ 1 đến 125 tuổi."

    return (g, b, a), None


def predict_patient(model, glucose, bmi, age):
    """
    Tiếp nhận chỉ số, xử lý và trả về nhãn phân loại kèm xác suất tin cậy.
    """
    # Nếu người dùng nhập 0 ở Glucose hoặc BMI (chưa xét nghiệm), chuyển thành NaN
    # để bộ SimpleImputer trong Pipeline tự động điền giá trị trung vị y khoa
    g_val = np.nan if glucose == 0 else glucose
    b_val = np.nan if bmi == 0 else bmi
    a_val = age

    # Tạo DataFrame đúng tên cột và thứ tự lúc huấn luyện
    input_df = pd.DataFrame([[g_val, b_val, a_val]], columns=FEATURE_NAMES)

    # Dự đoán bằng Pipeline
    pred_label = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    prob_neg = float(probabilities[0] * 100)
    prob_pos = float(probabilities[1] * 100)

    return {
        'prediction': pred_label,
        'label': 'DƯƠNG TÍNH - CẢNH BÁO: CÓ NGUY CƠ TIỂU ĐƯỜNG (Nhóm 1)' if pred_label == 1 else 'ÂM TÍNH - NGUY CƠ THẤP: BÌNH THƯỜNG (Nhóm 0)',
        'prob_negative': prob_neg,
        'prob_positive': prob_pos,
        'confidence': prob_pos if pred_label == 1 else prob_neg
    }


def display_results(glucose, bmi, age, res):
    """In phiếu kết quả trực quan trên terminal."""
    print("\n" + "=" * 70)
    print("           PHIẾU KẾT QUẢ SÀNG LỌC NGUY CƠ TIỂU ĐƯỜNG (AI ML)")
    print("=" * 70)
    print(f"📋 THÔNG TIN CHỈ SỐ BỆNH NHÂN NHẬP VÀO:")
    print(f"   1. Lượng đường huyết (Glucose) : {glucose:.1f} mg/dL {'(Đã tự động điền trung vị)' if glucose == 0 else ''}")
    print(f"   2. Chỉ số khối cơ thể (BMI)    : {bmi:.1f} kg/m² {'(Đã tự động điền trung vị)' if bmi == 0 else ''}")
    print(f"   3. Độ tuổi (Age)               : {int(age)} tuổi")
    print("-" * 70)
    print(f"🎯 KẾT QUẢ PHÂN LOẠI TỪ MÔ HÌNH:")
    print(f"   👉 KẾT LUẬN : {res['label']}")
    print(f"   👉 ĐỘ TIN CẬY: {res['confidence']:.2f}%")
    print("-" * 70)
    print(f"📊 PHÂN TÍCH XÁC SUẤT CHI TIẾT:")
    print(f"   - Xác suất Không tiểu đường (Nhóm 0): {res['prob_negative']:.2f}%")
    print(f"   - Xác suất Có nguy cơ tiểu đường (Nhóm 1): {res['prob_positive']:.2f}%")
    print("=" * 70)
    print("⚠️  TUYÊN BỐ MIỄN TRỪ TRÁCH NHIỆM Y KHOA (MEDICAL DISCLAIMER):")
    print("   Đây là sản phẩm học tập môn Học Máy (Machine Learning) minh họa thuật toán,")
    print("   TUYỆT ĐỐI KHÔNG CÓ GIÁ TRỊ THAY THẾ CHẨN ĐOÁN HOẶC CHỈ ĐỊNH CỦA BÁC SĨ CHUYÊN KHOA.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Dự đoán nguy cơ tiểu đường cho bệnh nhân mới")
    parser.add_argument('--glucose', type=float, help="Lượng đường huyết lúc đói (mg/dL)")
    parser.add_argument('--bmi', type=float, help="Chỉ số khối cơ thể BMI (kg/m²)")
    parser.add_argument('--age', type=float, help="Độ tuổi (năm)")

    args = parser.parse_args()

    model = load_trained_model()

    # Nếu truyền đủ tham số qua dòng lệnh CLI
    if args.glucose is not None and args.bmi is not None and args.age is not None:
        clean_vals, err = validate_input(args.glucose, args.bmi, args.age)
        if err:
            print(f"❌ {err}")
            sys.exit(1)
        glucose, bmi, age = clean_vals
    else:
        # Chế độ tương tác nhập trực tiếp từ bàn phím
        print("=" * 70)
        print(" CHƯƠNG TRÌNH DỰ ĐOÁN NGUY CƠ TIỂU ĐƯỜNG DỰA TRÊN MACHINE LEARNING")
        print("=" * 70)
        print("Vui lòng nhập các chỉ số lâm sàng theo yêu cầu dưới đây:")

        while True:
            try:
                g_in = input("-> Nhập Lượng đường huyết - Glucose (mg/dL, nhập 0 nếu chưa đo): ").strip()
                b_in = input("-> Nhập Chỉ số khối cơ thể - BMI (kg/m², ví dụ: 24.5): ").strip()
                a_in = input("-> Nhập Độ tuổi - Age (năm, ví dụ: 45): ").strip()

                clean_vals, err = validate_input(g_in, b_in, a_in)
                if err:
                    print(f"❌ {err}. Vui lòng nhập lại!\n")
                    continue
                glucose, bmi, age = clean_vals
                break
            except KeyboardInterrupt:
                print("\nĐã hủy chương trình.")
                sys.exit(0)

    # Tiến hành dự đoán và xuất kết quả
    res = predict_patient(model, glucose, bmi, age)
    display_results(glucose, bmi, age, res)


if __name__ == '__main__':
    main()

