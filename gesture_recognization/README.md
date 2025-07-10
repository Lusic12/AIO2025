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
AIO2025/
├── gesture_recognization/
│   ├── 🧠 common/               # Các lớp AI và utility chung
│   │   └── models.py            # HandGestureModel, HandLandmarksDetector
│   ├── ⚡ hardware/             # Điều khiển phần cứng
│   │   └── modbus_controller.py # ModbusMaster cho relay
│   ├── 📊 processing/           # Thu thập và xử lý dữ liệu
│   │   └── data_collector.py    # Thu thập landmarks từ camera
│   ├── 🎓 training/             # Huấn luyện mô hình AI
│   │   ├── gesture_trainer_recognizer.py
│   │   ├── hand_gesture_recgonition.ipynb
│   │   └── models/              # Model đã huấn luyện (.pth)
│   ├── 🚀 app/                  # Ứng dụng chính
│   │   └── main_controller.py   # Điều khiển relay qua gesture
│   ├── 💾 data/                 # Dữ liệu landmarks (CSV)
│   ├── config.yaml              # Cấu hình cử chỉ và tham số
│   ├── requirements.txt         # Dependencies Python
│   └── README.md                # Hướng dẫn chi tiết
```

## 🚀 Cài đặt nhanh

### 1. Cài đặt Python dependencies
```powershell
# Cài đặt thư viện cần thiết
pip install -r requirements.txt

# Kiểm tra cài đặt
python -c "import torch, cv2, mediapipe; print('✅ Setup OK!')"
```

### 2. (Khuyến nghị) Thiết lập môi trường ảo
```powershell
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt (Windows)
.\.venv\Scripts\Activate.ps1

# Kích hoạt (Linux/Mac)
source .venv/bin/activate
```

## 🎮 Sử dụng

### Chế độ Demo - Mô phỏng (Simulation)
```powershell
# Chạy với hiệu ứng đèn ảo, an toàn không cần phần cứng
python -m gesture_recognization.app.main_controller --simulation
```
🎯 **Tính năng:** Hiển thị 3 đèn ảo, chuyển màu real-time theo cử chỉ

### Chế độ Production - Phần cứng thật
```powershell
# Liệt kê cổng COM có sẵn
python -m gesture_recognization.app.main_controller --list-ports

# Kết nối với relay module
python -m gesture_recognization.app.main_controller --port COM3
```

### Tùy chọn nâng cao
```powershell
python -m gesture_recognization.app.main_controller \
  --model training/models/custom_model.pth \
  --config config.yaml \
  --resolution 1920x1080 \
  --simulation
```

## 🎯 Các cử chỉ hỗ trợ

| Cử chỉ | Mô tả | Hành động |
|--------|-------|-----------|
| ✋ **turn_on** | Bàn tay mở | Bật **tất cả** relay |
| ✊ **turn_off** | Nắm tay | Tắt **tất cả** relay |
| 👆 **light1_on** | Giơ 1 ngón | Bật relay số 1 |
| 🤏 **light1_off** | Chụm 1 ngón | Tắt relay số 1 |
| ✌️ **light2_on** | Giơ 2 ngón | Bật relay số 2 |
| 🤞 **light2_off** | Chụm 2 ngón | Tắt relay số 2 |

## 🔧 Phím tắt trong ứng dụng

- `q`: Thoát ứng dụng
- `r`: Reset tất cả đèn về trạng thái tắt
- `ESC`: Thoát khẩn cấp

## 🎓 Workflow Development (Dành cho dev)

### 1. Thu thập dữ liệu mới
```powershell
python -m processing.data_collector
```
📝 **Hướng dẫn:** Nhấn phím tương ứng cử chỉ → Thu thập 50-100 mẫu mỗi cử chỉ

### 2. Huấn luyện model với dữ liệu mới
```powershell
python -m training.gesture_trainer_recognizer
```
⏱️ **Thời gian:** 2-5 phút (tùy số lượng dữ liệu)

### 3. Test và deploy
```powershell
# Test với model mới
python -m app.main_controller --model training/models/new_model.pth --simulation
```

## ⚙️ Cấu hình system

### File `config.yaml`
```yaml
gestures:
  0: "turn_off"      # Tắt tất cả
  1: "light1_on"     # Bật đèn 1
  2: "light1_off"    # Tắt đèn 1
  3: "light2_on"     # Bật đèn 2  
  4: "light2_off"    # Tắt đèn 2
  5: "turn_on"       # Bật tất cả

sensor_settings:
  max_hands: 1                    # Chỉ nhận diện 1 bàn tay
  detection_confidence: 0.7       # Độ tin cậy phát hiện
  tracking_confidence: 0.5        # Độ tin cậy tracking

relay_settings:
  baudrate: 9600                  # Chuẩn Modbus RTU
  timeout: 2                      # Timeout (giây)
  debounce_frames: 8              # Số frame xác nhận gesture
```

## 🛠️ Troubleshooting

### ❌ Lỗi thường gặp

**Camera không hoạt động:**
```powershell
# Thử camera index khác
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

**Relay không phản hồi:**
```powershell
# Kiểm tra COM ports
python -m app.main_controller --list-ports

# Test kết nối Modbus
python -m app.main_controller --test --port COM3
```


## 📋 Yêu cầu hệ thống

- 🐍 **Python 3.10+**
- 📷 **Webcam HD (720p+)**
- 💾 **4GB RAM** (khuyến nghị 8GB)
- ⚡ **USB-Serial converter** (nếu dùng relay thật)
- 🖥️ **Windows 10/11** (Linux/Mac: experimental)



**Made with ❤️ by AI VIET NAM - STA ANH MINH**
