# 🤲 Hand Gesture Recognition & Relay Control

Hệ thống nhận diện cử chỉ tay thông minh để điều khiển relay Modbus RTU hoặc mô phỏng với AI.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green)](https://mediapipe.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org)

## ✨ Tính năng chính

- 🎯 **Nhận diện cử chỉ real-time** với độ chính xác cao
- 🔌 **Điều khiển relay Modbus RTU** (9600 baud) 
- 🎮 **Chế độ mô phỏng** với hiệu ứng đèn sinh động
- 🛡️ **Chống nhiễu gesture** với kỹ thuật debouncing
- 📊 **Thu thập và huấn luyện dữ liệu** tự động
- ⚙️ **Cấu hình linh hoạt** qua YAML

## 📁 Cấu trúc dự án

```
|- gesture_recognition/
   |- common/
   |  |- models.py               # Các lớp và hàm dùng chung
   |
   |- hardware/
   |  |- modbus_controller.py    # Lớp giao tiếp relay qua Modbus RTU
   |
   |- processing/
   |  |- data_collector.py       # Thu thập landmarks từ webcam
   |
   |- training/
   |  |- gesture_trainer_recognizer.py  # Script huấn luyện CLI
   |  |- gesture_trainer_recognizer.ipynb # Notebook minh họa
   |  |- models/                 # Chứa mô hình đã huấn luyện (.pth)
   |
   |- data/
   |  |- landmarks_train.csv     # Dữ liệu huấn luyện
   |  |- landmarks_test.csv      # Dữ liệu kiểm tra
   |  |- landmarks_val.csv       # Dữ liệu validation
   |
   |- main_controller.py         # Chạy real-time nhận diện + relay
   |- config.yaml                # Cấu hình cử chỉ
   |- requirements.txt           # Các thư viện cần cài
   |- README.md                  # Hướng dẫn tổng quát
```

## 🚀 Cài đặt nhanh


### 1. (Khuyến nghị) Thiết lập môi trường ảo
```powershell
# Tạo môi trường ảo
conda create -n gesture_env python==3.10

# Kích hoạt môi trường ảo
conda activate gesture_env

```

### 2. Cài đặt Python dependencies
```powershell
# Cài đặt thư viện cần thiết
pip install -r requirements.txt

# Kiểm tra cài đặt
python -c "import torch, cv2, mediapipe; print('✅ Setup OK!')"
```

## 🎮 Sử dụng

### Chế độ Demo - Mô phỏng (Simulation)
```powershell
# Chạy với hiệu ứng đèn ảo, an toàn không cần phần cứng
python main_controller --simulation
```
🎯 **Tính năng:** Hiển thị 3 đèn ảo, chuyển màu real-time theo cử chỉ

### Chế độ Production - Phần cứng thật
```powershell
# Liệt kê cổng COM có sẵn
python main_controller --list-ports

# Kết nối với relay module
python main_controller --port COM3
```

### Tùy chọn nâng cao
```powershell
python main_controller \
  --model training/models/custom_model.pth \
  --config config.yaml \
  --resolution 1920x1080 \
  --simulation
```

## 🎯 Các cử chỉ hỗ trợ

| Cử chỉ | Mô tả | Hành động |
|--------|-------|-----------|
| 👌 **turn_on** | Cử chỉ OK (tròn) | Bật **tất cả** relay |
| ✊ **turn_off** | Nắm tay lại | Tắt **tất cả** relay |
| 1️⃣ **light1_on** | Số 1 (giơ 1 ngón) | Bật relay số 1 |
| 2️⃣ **light1_off** | Số 2 (giơ 2 ngón) | Tắt relay số 1 |
| 3️⃣ **light2_on** | Số 3 (giơ 3 ngón) | Bật relay số 2 |
| 4️⃣ **light2_off** | Số 4 (giơ 4 ngón) | Tắt relay số 2 |

## 🔧 Phím tắt trong ứng dụng

- `q`: Thoát ứng dụng
- `r`: Reset tất cả đèn về trạng thái tắt
- `ESC`: Thoát khẩn cấp

## 🎓 Workflow Development (Dành cho dev)

### 1. Thu thập dữ liệu mới
```powershell
python -m processing.data_collector
```
**Hướng dẫn:** Nhấn phím tương ứng cử chỉ -> Thu thập 50-100 mẫu mỗi cử chỉ

### 2. Huấn luyện model với dữ liệu mới
```powershell
python training.gesture_trainer_recognizer
```
**Thời gian:** 2-5 phút (tùy số lượng dữ liệu)

### 3. Test và deploy
```powershell
# Test với model mới
python main_controller --model training/models/new_model.pth --simulation
```

## ⚙️ Cấu hình system

### File `config.yaml`
```yaml
gestures:
  0: "light1_on"     # Số 1 - Bật đèn 1
  1: "light1_off"    # Số 2 - Tắt đèn 1  
  2: "light2_on"     # Số 3 - Bật đèn 2
  3: "light2_off"    # Số 4 - Tắt đèn 2
  4: "turn_off"      # Nắm tay - Tắt tất cả
  5: "turn_on"       # OK (tròn) - Bật tất cả

sensor_settings:
  max_hands: 1                    # Chỉ nhận diện 1 bàn tay
  detection_confidence: 0.7       # Độ tin cậy phát hiện
  tracking_confidence: 0.5        # Độ tin cậy tracking

relay_settings:
  baudrate: 9600                  # Chuẩn Modbus RTU
  timeout: 2                      # Timeout (giây)
  debounce_frames: 8              # Số frame xác nhận gesture
```

## Troubleshooting

### Lỗi thường gặp

**Camera không hoạt động:**
```powershell
# Thử camera index khác
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

**Relay không phản hồi:**
```powershell
# Kiểm tra COM ports
python main_controller --list-ports

# Test kết nối Modbus
python main_controller --test --port COM3
```


## Yêu cầu hệ thống

- **Python 3.10+**
- **Webcam HD (720p+)**
- **4GB RAM** (khuyến nghị 8GB)
- **USB-Serial converter** (nếu dùng relay thật)
- **Windows 10/11** (Linux/Mac: experimental)



**Made with ❤️ by AI VIET NAM - STA ANH MINH**
