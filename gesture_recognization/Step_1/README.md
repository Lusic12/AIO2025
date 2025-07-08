# Hướng dẫn tạo dữ liệu cử chỉ tay (Step 1: Data Collection)

## Mục đích

Thu thập dữ liệu landmarks từ bàn tay qua camera để tạo bộ dữ liệu huấn luyện cho AI nhận diện cử chỉ.

## Các bước thực hiện

### 1. Chuẩn bị

- Đảm bảo đã cài đặt đầy đủ thư viện Python theo `requirements.txt`.
- Đảm bảo file `../config.yaml` đã cấu hình các nhãn cử chỉ.
- Kết nối và kiểm tra camera hoạt động tốt.

### 2. Chạy chương trình thu thập dữ liệu

Mở terminal tại thư mục `Step_1` và chạy:

```bash
python data_collector.py
```

### 3. Quy trình thu thập cho từng bộ dữ liệu

Chương trình sẽ lần lượt yêu cầu bạn thu thập cho 3 bộ: **train**, **val**, **test**.

- **Nhấn Enter** để bắt đầu từng bộ (train/val/test).
- **Chọn cử chỉ**: Nhấn phím (a, b, c, ...) hoặc SPACE tương ứng với từng cử chỉ (theo config.yaml).
- **Bắt đầu ghi**: Nhấn lại cùng phím để bắt đầu ghi dữ liệu cho cử chỉ đó.
- **Thực hiện cử chỉ**: Đưa tay vào camera và thực hiện cử chỉ liên tục.
- **Dừng ghi**: Nhấn lại cùng phím để dừng ghi dữ liệu.
- **Lưu ảnh mẫu**: Nhấn phím `s` để lưu ảnh tham khảo cho cử chỉ hiện tại.
- **Chuyển cử chỉ khác**: Nhấn phím khác để chọn cử chỉ mới.
- **Thoát**: Nhấn `q` để kết thúc thu thập cho bộ hiện tại.

Lặp lại các bước trên cho từng bộ dữ liệu (train, val, test).

### 4. Kết quả

Sau khi hoàn thành, bạn sẽ có các file:

```
data/
  landmarks_train.csv
  landmarks_val.csv
  landmarks_test.csv

sample_images/
  <tên_cử_chỉ>_<mode>_sample.jpg
```

- Mỗi file CSV chứa dữ liệu landmarks và nhãn cử chỉ.
- Thư mục `sample_images` chứa ảnh mẫu cho từng cử chỉ.

### 5. Lưu ý

- Nên thu thập đủ số lượng mẫu cho mỗi cử chỉ ở cả 3 bộ train/val/test để đảm bảo chất lượng huấn luyện.
- Nếu thiếu file `landmarks_val.csv`, hãy chạy lại và chọn đúng mode `val` khi chương trình yêu cầu.
- Số lượng mẫu ở mỗi bộ phụ thuộc vào thời gian bạn ghi dữ liệu cho từng mode.

---

**Sau khi thu thập xong, chuyển sang Step 2 để huấn luyện mô hình.**