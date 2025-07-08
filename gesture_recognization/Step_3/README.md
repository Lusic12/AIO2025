# Step 3: Gesture Control

Module điều khiển thiết bị thông qua nhận diện cử chỉ tay đã được training. Hỗ trợ hai phương thức điều khiển:

1. **HTTP ESP32**: Điều khiển ESP32 qua giao thức HTTP (`http_esp32.py`)
2. **Modbus Relay**: Điều khiển relay qua giao thức Modbus RTU (`relay_controller.py`)

## Mục đích

Sử dụng model đã huấn luyện từ Step 2 để nhận diện cử chỉ tay real-time và gửi lệnh điều khiển đến thiết bị mục tiêu (ESP32 hoặc Relay).

## Tính năng chính

### 1. **Model Loading**
- Load model weights đã huấn luyện từ Step 2
- Tự động tìm model mới nhất hoặc chọn thủ công
- Kiểm tra tính tương thích của model

### 2. **Real-time Recognition**
- Nhận diện cử chỉ tay từ camera real-time
- Xử lý landmarks với MediaPipe
- Áp dụng confidence threshold và gesture stability

### 3. **ESP32 Control**
- Gửi HTTP requests đến ESP32
- Mapping cử chỉ thành lệnh điều khiển LED
- Test kết nối và retry mechanism

### 4. **Smart Control Logic**
- Gesture stability (cần N frames liên tiếp)
- Cooldown time giữa các lệnh
- Tránh spam commands

## Files trong thư mục

### `http_esp32.py`
Module điều khiển ESP32 thông qua HTTP requests:

- **`ESP32Controller`**: Quản lý kết nối và gửi lệnh đến ESP32 HTTP server
- **`ESP32GestureControl`**: Kết hợp nhận diện cử chỉ và điều khiển ESP32
- Hỗ trợ chế độ test: `python http_esp32.py --test`

### `relay_controller.py`
Module điều khiển Relay qua Modbus RTU:

- **`ModbusMaster`**: Quản lý kết nối và gửi lệnh Modbus RTU đến relay
- **`RelayGestureControl`**: Kết hợp nhận diện cử chỉ và điều khiển relay
- Hỗ trợ chế độ test: `python relay_controller.py --test`

### `controller.py`
Module base chứa các class ModbusMaster để điều khiển relay.

### `gesture_control.py`
Module kết hợp (phiên bản cũ) chứa:

#### Classes:
- **`HandGestureModel`**: Neural network (copy từ Step 2)
- **`HandLandmarksDetector`**: MediaPipe hand detection
- **`ESP32Controller`**: HTTP communication với ESP32
- **`GestureController`**: Main controller class

#### Functions:
- **`find_latest_model()`**: Tìm model mới nhất
- **`main()`**: Entry point

## Cách sử dụng

### Điều khiển ESP32 thông qua HTTP
Nếu bạn muốn điều khiển ESP32 qua giao thức HTTP:

```bash
cd Step_3
python http_esp32.py
```

### Điều khiển Relay thông qua Modbus RTU
Nếu bạn muốn điều khiển Relay qua giao thức Modbus:

```bash
cd Step_3
python relay_controller.py
```

### Chế độ Test (không dùng camera)
Test kết nối ESP32 HTTP:

```bash
cd Step_3
python http_esp32.py --test [optional_ip_address]
```

Test kết nối Relay Modbus:

```bash
cd Step_3
python relay_controller.py --test
```

### Chạy module cũ (hỗ trợ cả hai chế độ)
Để sử dụng phiên bản trước kết hợp cả hai phương thức:

```bash
cd Step_3
python gesture_control.py
```

### Import trong code khác
```python
# Điều khiển ESP32 HTTP
from http_esp32 import ESP32GestureControl

controller = ESP32GestureControl(
    model_path="path/to/model.pth",
    esp32_ip="192.168.1.100"
)
controller.run()

# Điều khiển Relay Modbus
from relay_controller import RelayGestureControl

controller = RelayGestureControl(
    model_path="path/to/model.pth"
)
controller.run()
```

## Configuration

### Gesture Mapping
```python
command_map = {
    0: None,        # turn_off - không làm gì
    1: "1",         # light1_on - toggle LED 1
    2: "1",         # light1_off - toggle LED 1
    3: "2",         # light2_on - toggle LED 2
    4: "2",         # light2_off - toggle LED 2
    5: "1",         # peace - toggle LED 1
    6: "2",         # ok - toggle LED 2
}
```

### Parameters
```python
confidence_threshold = 0.8      # Ngưỡng confidence tối thiểu
gesture_threshold = 15          # Số frame liên tiếp để xác nhận
cooldown_time = 2.0            # Thời gian chờ giữa các lệnh (giây)
```

## Workflow

1. **Load Model**: Đọc model weights từ Step 2
2. **Test ESP32**: Kiểm tra kết nối HTTP
3. **Start Camera**: Khởi động webcam
4. **Detect Hands**: Phát hiện landmarks từ MediaPipe
5. **Predict Gesture**: Dự đoán cử chỉ với model
6. **Validate Stability**: Kiểm tra tính ổn định (N frames)
7. **Send Command**: Gửi HTTP request đến ESP32
8. **Apply Cooldown**: Chờ trước lệnh tiếp theo

## Controls

### Keyboard Controls
- **`q`**: Quit/Thoát chương trình
- **`r`**: Reconnect ESP32

### Screen Display
- **Gesture name + confidence**: Tên cử chỉ và độ tin cậy
- **Progress bar**: Tiến độ gesture stability
- **ESP32 status**: Trạng thái kết nối (🟢/🔴)

## Requirements

### Model từ Step 2
```
Step_2/models/
└── hand_gesture_model_YYYYMMDD_HHMMSS.pth
```

### ESP32 Running
- ESP32 đã upload code và chạy HTTP server
- Cùng mạng WiFi với máy tính
- Port 80 mở và accessible

### Camera
- Webcam hoạt động bình thường
- Quyền truy cập camera

## Troubleshooting

### Lỗi không tìm thấy model
```bash
# Kiểm tra thư mục models
ls ../Step_2/models/
# Hoặc chạy Step 2 để training model
cd ../Step_2
python gesture_trainer_recognizer.py
```

### ESP32 không kết nối được
```bash
# Test kết nối ESP32 thủ công
cd Step_3
python http_esp32.py --test
# hoặc
cd ..
python test_esp32_connection.py
```

### Relay Modbus không kết nối được
```bash
# Test kết nối Modbus thủ công
cd Step_3
python relay_controller.py --test
```

### Kiểm tra cổng COM (cho Modbus)
```bash
# Windows
python -m serial.tools.list_ports

# Linux
python -m serial.tools.list_ports -v
```

### Camera không hoạt động
- Kiểm tra camera index (0, 1, 2...)
- Kiểm tra quyền truy cập camera
- Đóng các ứng dụng khác đang dùng camera

### Gesture không ổn định
- Tăng `gesture_threshold` (số frame cần thiết)
- Điều chỉnh `confidence_threshold`
- Cải thiện lighting và background

## Performance Tips

### Tối ưu độ chính xác
1. **Training data tốt**: Đảm bảo Step 1 thu thập dữ liệu đủ và chất lượng
2. **Lighting**: Ánh sáng đều, tránh backlight
3. **Background**: Nền đơn giản, ít nhiễu
4. **Hand position**: Giữ tay trong khung hình, khoảng cách ổn định

### Tối ưu hiệu suất
1. **Camera resolution**: Giảm độ phân giải nếu cần (720p thay vì 1080p)
2. **Model optimization**: Sử dụng model nhỏ hơn nếu accuracy chấp nhận được
3. **Frame rate**: Giảm FPS nếu không cần real-time cao

## Integration với ESP32

### HTTP Endpoints
```
GET http://ESP32_IP/              # Test connection
GET http://ESP32_IP/toggle/1      # Toggle LED 1
GET http://ESP32_IP/toggle/2      # Toggle LED 2
```

### Response Format
```
HTTP/1.1 200 OK
Content-Type: text/html
```

## So sánh hai phương thức điều khiển

| Tính năng | HTTP ESP32 | Modbus Relay |
|-----------|------------|--------------|
| **Giao thức** | HTTP | Modbus RTU |
| **Kết nối** | Không dây (WiFi) | Có dây (USB/Serial) |
| **Khoảng cách** | Xa (phạm vi WiFi) | Gần (cần kết nối cáp) |
| **Ổn định** | Phụ thuộc WiFi | Rất ổn định |
| **Độ trễ** | Cao hơn | Thấp hơn |
| **Thiết lập** | Cần cài đặt ESP32 | Cần cài đặt driver USB-Serial |
| **Mở rộng** | Dễ thêm nhiều ESP32 | Giới hạn bởi số cổng COM |
| **Ứng dụng** | IoT, điều khiển từ xa | Điều khiển công nghiệp |
| **File sử dụng** | `http_esp32.py` | `relay_controller.py` |

## Chọn phương thức nào?

- **HTTP ESP32**: Tốt cho các ứng dụng IoT, điều khiển từ xa không dây, dễ mở rộng
- **Modbus Relay**: Tốt cho điều khiển công nghiệp, ổn định hơn, phản hồi nhanh hơn

## Future Enhancements

- [ ] Support multiple ESP32 devices
- [ ] Support multiple Modbus slaves
- [ ] Custom gesture mapping via config file
- [ ] Voice feedback
- [ ] Gesture history logging
- [ ] Mobile app integration
- [ ] MQTT support
- [ ] Multi-hand detection
