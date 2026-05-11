# Pneumonia Detection using Support Vector Machine (SVM)

Dự án sử dụng mô hình Support Vector Machine (SVM) được xây dựng từ đầu (from scratch) để phân loại ảnh X-quang phổi nhằm phát hiện bệnh viêm phổi (Pneumonia).

---

##  Cấu trúc thư mục

* **`data_prep.py`**: Chứa các hàm tiền xử lý dữ liệu:
    * Đọc dữ liệu hình ảnh.
    * Chuyển đổi sang ảnh xám (Grayscale).
    * Thay đổi kích thước ảnh (**128x128**).
    * Chuẩn hóa (Normalization) và làm phẳng (Flatten) dữ liệu.
* **`model.py`**: Xây dựng lớp SVM tùy chỉnh:
    * Hàm mục tiêu: **Hinge Loss**.
    * Tối ưu hóa: Cập nhật **Gradient Descent** thủ công.
* **`evaluate.py`**: Chứa các công cụ đánh giá mô hình:
    * Tính toán các độ đo: **Precision, Recall, F1-Score**.
* **`main.py`**: File thực thi chính của dự án:
    * Nạp dữ liệu từ thư mục.
    * Huấn luyện hai biến thể mô hình.
    * So sánh kết quả và hiển thị báo cáo.
