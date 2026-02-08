# KF Hand (VOT2022) — Kalman Filter bbox + visualize

Thư mục này minh hoạ cách dùng **Kalman Filter** để làm mượt / dự đoán bounding box (bbox) theo thời gian, dựa trên dataset VOT2022 (hand).

- Code chính: [present2_kf_hand.py](present2_kf_hand.py)
- Dữ liệu: [vot2022_hand/groundtruth.txt](vot2022_hand/groundtruth.txt) và thư mục ảnh `vot2022_hand/images/`

## Cách chạy

Chạy từ thư mục `AIO2025/kf_hand`:

```bash
python present2_kf_hand.py
```

Output:
- Tạo video `predict.mp4` (ghi từng frame đã vẽ bbox)
- Hiển thị cửa sổ OpenCV để xem trực tiếp

## Ý nghĩa các màu trong visualize

Trong vòng lặp `main_bbox_tracking()` code vẽ 3 bbox (OpenCV dùng hệ màu **BGR**):

- **Xanh dương** `(255, 0, 0)`: **Groundtruth (GT)** — bbox “chuẩn” đọc từ `groundtruth.txt`
- **Hồng tím** `(255, 0, 255)`: **Measurement/Detector (noisy)** — bbox GT nhưng được cộng nhiễu Gaussian để mô phỏng detector bị nhiễu
- **Trắng** `(255, 255, 255)`: **Kalman estimate** — bbox ước lượng sau Kalman Filter (predict + update)

## Pipeline trong code (từ dữ liệu → vẽ)

Mỗi frame, code làm theo thứ tự:

1) Đọc GT bbox từ file
- `bbox = [l, t, w, h]`

2) Vẽ GT (màu xanh)
- `draw_bbox(bbox, img, (255,0,0))`

3) Tạo measurement nhiễu (màu hồng)
- `bbox_n = bbox + N(0, std_meas)`
- Sau đó `clamp_bbox(...)` để đảm bảo `w,h >= min_size`

4) Kalman Filter
- `kf.predict()` dùng mô hình động học để dự đoán state tiếp theo
- `kf.update(z)` cập nhật lại state theo measurement `z=[l,t,w,h]`

5) Vẽ Kalman estimate (màu trắng)
- `x_est = kf.update(...)` trả về `[l,t,w,h]`
- `draw_bbox(x_est, img, (255,255,255))`

## Kalman Filter đang track cái gì?

State 6 chiều:

\[
x = [l, t, w, h, dl, dt]^T
\]

- `l,t`: toạ độ góc trái-trên
- `w,h`: width/height (track trực tiếp, không log)
- `dl,dt`: vận tốc của `l,t`

### Ma trận `F` (transition)
- `l,t` theo **constant velocity**: $l_{k+1}=l_k+dl_k\Delta t$; $t_{k+1}=t_k+dt_k\Delta t$
- `w,h` giữ nguyên (random-walk sẽ được cho phép qua `Q`)

### Ma trận `H` (measurement)
- Measurement chỉ đo được `[l,t,w,h]` nên `H` lấy 4 phần tử đầu của state.

## Nhiễu `Q` và `R` (tóm tắt)

- `R`: nhiễu đo (measurement noise). Code tạo `D0 = diag([std_meas,...])` rồi `R = D0 D0^T` ⇒ `R` là đường chéo với phương sai `std_meas^2`.
- `Q`: nhiễu quá trình (process noise).
  - Phần `np.kron(np.eye(2), B0)` tạo 2 block cho 2 trục `l` và `t` theo mô hình vị trí–vận tốc với gia tốc nhiễu.
  - Phần `B1` tạo random-walk noise cho `w,h`.
  - `Q = B B^T` để ra hiệp phương sai đúng kích thước 6x6.

## Phụ thuộc

- `numpy`
- `scipy` (để dùng `block_diag`)
- `opencv-python` (import `cv2`)

Nếu thiếu package:

```bash
pip install numpy scipy opencv-python
```
