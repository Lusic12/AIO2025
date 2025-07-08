# ESP32 Hand Gesture Recognition Project

Dự án nhận diện cử chỉ tay để điều khiển ESP32 thông qua camera và AI.

## Cấu trúc dự án

```
esp32-http-server/
├── src/
│   └── esp32-http-server.ino      # Mã nguồn ESP32
├── gesture_recognization/
│   ├── config.yaml                # Cấu hình cử chỉ
│   ├── requirements.txt           # Thư viện Python
│   ├── main.py                    # Script chính (entry point)
│   ├── test_esp32_connection.py   # Test kết nối ESP32
│   ├── Step_1/                    # Thu thập dữ liệu
│   │   ├── data_collector.py      # Module thu thập dữ liệu
│   │   └── generate_landmark_data.py  # Utility functions
│   ├── Step_2/                    # Training và Recognition
│   │   └── gesture_trainer_recognizer.py
│   └── Step_3/                    # Gesture Control
│       ├── gesture_control.py     # Module điều khiển (phiên bản cũ)
│       ├── http_esp32.py          # Module điều khiển ESP32 qua HTTP
│       ├── relay_controller.py    # Module điều khiển Relay qua Modbus
│       ├── controller.py          # Module base điều khiển Modbus
│       └── README.md              # Hướng dẫn Step 3
├── platformio.ini                 # Cấu hình PlatformIO
└── wokwi.toml                    # Cấu hình Wokwi Simulator
```

## Hướng dẫn sử dụng

### Bước 1: Cài đặt môi trường

1. **Cài đặt Python dependencies:**
   ```bash
   cd gesture_recognization
   pip install -r requirements.txt
   ```

2. **Cài đặt PlatformIO** (cho ESP32):
   - Cài PlatformIO IDE extension trong VS Code
   - Hoặc cài PlatformIO Core: `pip install platformio`

### Bước 2: Cấu hình cử chỉ

Chỉnh sửa file `config.yaml` để định nghĩa các cử chỉ:

```yaml
gestures:
  0: "turn_off"      # Tắt tất cả đèn
  1: "light1_on"     # Bật đèn số 1
  2: "light1_off"    # Tắt đèn số 1
  3: "light2_on"     # Bật đèn số 2
  4: "light2_off"    # Tắt đèn số 2
```

### Bước 3: Khởi tạo môi trường trên anaconda

```
conda create -n env gesture python==3.10
conda activate
```

### Bước 3: Thu thập dữ liệu (Step 1)

```bash
cd Step_1
python data_collector.py
```

**Hướng dẫn thu thập:**
- Nhấn phím tương ứng với cử chỉ (a, b, c, d, e...)
- Nhấn cùng phím 2 lần để bắt đầu/dừng ghi dữ liệu
- Nhấn 's' để lưu ảnh mẫu
- Nhấn 'q' để thoát

### Bước 4: Huấn luyện mô hình (Step 2)

```bash
cd Step_2
python gesture_trainer_recognizer.py
```

Chọn chức năng **1** để training model.

### Bước 5: Chuẩn bị ESP32

1. **Upload code lên ESP32:**
   ```bash
   pio run --target upload
   ```

2. **Kiểm tra IP của ESP32:**
   ```bash
   pio device monitor
   ```

### Bước 6: Điều khiển thiết bị (Step 3)

#### Điều khiển ESP32 qua HTTP:
```bash
cd Step_3
python http_esp32.py
```

#### Điều khiển Relay qua Modbus:
```bash
cd Step_3
python relay_controller.py
```

#### (Phiên bản cũ) Điều khiển đa chức năng:
```bash
cd Step_3
python gesture_control.py
```

Hoặc sử dụng script chính:
```bash
python main.py
# Chọn option 4: Điều khiển thiết bị
```

### Bước 7 (Tùy chọn): Test nhận diện (Step 2)

```bash
cd Step_2
python gesture_trainer_recognizer.py
```

Chọn chức năng **2** và nhập:
- Đường dẫn đến file model (.pth)
- IP của ESP32

## Workflow tổng thể

1. **ESP32 Setup**: ESP32 tạo HTTP server để nhận lệnh điều khiển
2. **Data Collection (Step 1)**: Thu thập landmarks từ camera cho các cử chỉ
3. **Model Training (Step 2)**: Huấn luyện neural network với dữ liệu đã thu thập
4. **Gesture Control (Step 3)**: Load model và điều khiển ESP32 real-time

## So sánh các Steps

| Step | Mục đích | Input | Output | 
|------|----------|--------|---------|
| **Step 1** | Thu thập dữ liệu | Camera + Cử chỉ tay | CSV files (landmarks) |
| **Step 2** | Training model | CSV files | Model (.pth) + Recognition |
| **Step 3** | Điều khiển ESP32 | Model + Camera | HTTP commands → ESP32 |

## Sự khác biệt Step 2 vs Step 3

### Step 2: Training + Recognition
- **Training**: Huấn luyện model từ dữ liệu
- **Recognition**: Test model với việc gửi HTTP requests
- **Focus**: Development và testing

### Step 3: Production Control  
- **Control**: Load model có sẵn và điều khiển
- **Optimized**: Tối ưu cho performance và stability
- **Features**: Smart gesture validation, cooldown, reconnection
- **Focus**: Production usage

## Troubleshooting

### Lỗi import libraries
```bash
pip install -r requirements.txt
```

### ESP32 không kết nối được
- Kiểm tra WiFi settings trong code ESP32
- Đảm bảo máy tính và ESP32 cùng mạng
- Kiểm tra IP address trong Serial Monitor

### Camera không hoạt động
- Kiểm tra quyền truy cập camera
- Thử thay đổi camera index (0, 1, 2...)

### Model accuracy thấp
- Thu thập thêm dữ liệu training
- Đảm bảo cử chỉ rõ ràng và nhất quán
- Tăng số epoch training

## Tính năng

- ✅ Thu thập dữ liệu tự động
- ✅ Training model với early stopping
- ✅ Real-time gesture recognition
- ✅ HTTP communication với ESP32
- ✅ Giao diện trực quan
- ✅ Configurable gestures

## Yêu cầu hệ thống

- Python 3.8+
- Webcam
- ESP32 development board
- WiFi network

## License

MIT License
