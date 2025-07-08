"""
Step 3: Gesture Recognition Control Module
==========================================

Mô-đun điều khiển ESP32 thông qua nhận diện cử chỉ tay đã được training.
Load model đã huấn luyện và thực hiện nhận diện real-time để điều khiển thiết bị.

Author: ESP32 Gesture Recognition Project
"""

import os
import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import yaml
import requests
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class HandGestureModel(nn.Module):
    """Neural Network cho hand gesture recognition - copy từ Step 2"""
    
    def __init__(self, input_size=63, num_classes=5, hidden_sizes=[128, 64, 32]):
        super(HandGestureModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def predict_with_known_class(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            return torch.argmax(outputs, dim=1)


class HandLandmarksDetector:
    """Lớp phát hiện và trích xuất landmark từ bàn tay"""
    
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def extract_landmarks(self, frame):
        """Trích xuất landmarks từ frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            landmarks = []
            hand_landmarks = results.multi_hand_landmarks[0]  # Chỉ lấy bàn tay đầu tiên
            
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return landmarks, hand_landmarks
        
        return None, None
    
    def draw_landmarks(self, frame, hand_landmarks):
        """Vẽ landmarks lên frame"""
        if hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )


class ESP32Controller:
    """Lớp điều khiển ESP32 thông qua HTTP requests"""
    
    def __init__(self, esp32_ip: str = "192.168.1.100", port: int = 80):
        self.esp32_ip = esp32_ip
        self.port = port
        self.base_url = f"http://{esp32_ip}:{port}"
        self.toggle_url = f"{self.base_url}/toggle/"
        
        # Test kết nối
        self.is_connected = self._test_connection()
        
    def _test_connection(self) -> bool:
        """Test kết nối đến ESP32"""
        try:
            response = requests.get(self.base_url, timeout=3)
            if response.status_code == 200:
                print(f"✅ Kết nối ESP32 thành công: {self.base_url}")
                return True
            else:
                print(f"⚠️ ESP32 phản hồi với status code: {response.status_code}")
                return False
        except requests.ConnectionError:
            print(f"❌ Không thể kết nối đến ESP32: {self.base_url}")
            print("💡 Kiểm tra:")
            print("   - ESP32 đã bật và kết nối WiFi")
            print("   - IP address đúng")
            print("   - Cùng mạng với máy tính")
            return False
        except requests.Timeout:
            print(f"⏰ Timeout kết nối ESP32: {self.base_url}")
            return False
        except Exception as e:
            print(f"❌ Lỗi kết nối ESP32: {e}")
            return False
    
    def send_command(self, gesture_id: int, gesture_name: str) -> bool:
        """
        Gửi lệnh điều khiển tới ESP32 dựa trên gesture_id
        
        Args:
            gesture_id: ID của cử chỉ (0, 1, 2, 3, 4...)
            gesture_name: Tên cử chỉ (để log)
            
        Returns:
            bool: True nếu gửi thành công
        """
        if not self.is_connected:
            return False
            
        try:
            # Ánh xạ gesture tới LED control
            command_map = {
                0: None,        # turn_off - không làm gì
                1: "1",         # light1_on - toggle LED 1
                2: "1",         # light1_off - toggle LED 1
                3: "2",         # light2_on - toggle LED 2
                4: "2",         # light2_off - toggle LED 2
                5: "1",         # peace - toggle LED 1
                6: "2",         # ok - toggle LED 2
            }
            
            command = command_map.get(gesture_id)
            
            if command is None:
                return True  # turn_off command - không cần gửi HTTP
            
            url = f"{self.toggle_url}{command}"
            response = requests.get(url, timeout=2)
            
            if response.status_code == 200:
                print(f"🎯 Đã gửi lệnh '{gesture_name}' -> LED {command}")
                return True
            else:
                print(f"⚠️ Lỗi gửi lệnh: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Lỗi kết nối ESP32: {e}")
            return False
    
    def reconnect(self) -> bool:
        """Thử kết nối lại ESP32"""
        print("🔄 Đang thử kết nối lại ESP32...")
        self.is_connected = self._test_connection()
        return self.is_connected


class GestureController:
    """Lớp chính điều khiển nhận diện cử chỉ và gửi lệnh đến ESP32"""
    
    def __init__(self, model_path: str, esp32_ip: str = "192.168.1.100", 
                 config_path: str = "../config.yaml"):
        
        self.model_path = model_path
        self.config_path = config_path
        
        # Load cấu hình
        self.gesture_labels = self._load_gesture_config()
        self.num_classes = len(self.gesture_labels)
        
        # Load model
        self.model = self._load_model()
        
        # Khởi tạo detector và controller
        self.detector = HandLandmarksDetector()
        self.esp32_controller = ESP32Controller(esp32_ip)
        
        # Thiết lập parameters
        self.confidence_threshold = 0.8
        self.gesture_threshold = 15  # Số frame liên tiếp để xác nhận cử chỉ
        self.cooldown_time = 2.0     # Thời gian chờ giữa các lệnh (giây)
        
        print(f"✅ GestureController đã sẵn sàng với {self.num_classes} cử chỉ")
    
    def _load_gesture_config(self) -> Dict[int, str]:
        """Đọc cấu hình cử chỉ từ file YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config.get('gestures', {})
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file config: {self.config_path}")
            return {}
        except Exception as e:
            print(f"❌ Lỗi đọc config: {e}")
            return {}
    
    def _load_model(self) -> HandGestureModel:
        """Load model đã được training từ Step 2"""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Lấy thông tin từ checkpoint
            saved_labels = checkpoint.get('gesture_labels', {})
            saved_num_classes = checkpoint.get('num_classes', len(saved_labels))
            
            # Khởi tạo model
            model = HandGestureModel(num_classes=saved_num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"✅ Đã load model từ: {self.model_path}")
            print(f"   Model classes: {saved_num_classes}")
            print(f"   Gesture labels: {saved_labels}")
            
            return model
            
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file model: {self.model_path}")
            raise
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
            raise
    
    def predict_gesture(self, landmarks: List[float]) -> Tuple[Optional[int], float]:
        """Dự đoán cử chỉ từ landmarks"""
        if landmarks is None or len(landmarks) != 63:
            return None, 0.0
        
        try:
            # Chuyển đổi thành tensor
            landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(landmarks_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                return predicted.item(), confidence.item()
                
        except Exception as e:
            print(f"❌ Lỗi dự đoán: {e}")
            return None, 0.0
    
    def run_control(self, camera_index: int = 0, resolution: Tuple[int, int] = (1280, 720)):
        """
        Chạy vòng lặp chính điều khiển real-time
        
        Args:
            camera_index: Index của camera (0, 1, 2...)
            resolution: Độ phân giải camera
        """
        # Khởi tạo camera
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        if not cap.isOpened():
            print(f"❌ Không thể mở camera {camera_index}")
            return
        
        # Biến theo dõi trạng thái
        last_gesture = None
        gesture_count = 0
        last_command_time = 0
        
        print("\n" + "="*60)
        print("    🎮 GESTURE CONTROL STARTED")
        print("="*60)
        print("📹 Camera đang chạy...")
        print("🤲 Thực hiện cử chỉ trước camera để điều khiển ESP32")
        print("⌨️ Nhấn 'q' để thoát, 'r' để reconnect ESP32")
        print("="*60)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Không thể đọc frame từ camera")
                break
            
            # Trích xuất landmarks
            landmarks, hand_landmarks = self.detector.extract_landmarks(frame)
            
            # Vẽ landmarks nếu có
            if hand_landmarks:
                self.detector.draw_landmarks(frame, hand_landmarks)
            
            # Dự đoán và điều khiển
            if landmarks:
                gesture_id, confidence = self.predict_gesture(landmarks)
                
                if gesture_id is not None and confidence > self.confidence_threshold:
                    gesture_name = self.gesture_labels.get(gesture_id, f"Unknown_{gesture_id}")
                    
                    # Kiểm tra tính nhất quán của cử chỉ
                    if gesture_id == last_gesture:
                        gesture_count += 1
                    else:
                        last_gesture = gesture_id
                        gesture_count = 1
                    
                    # Gửi lệnh nếu cử chỉ ổn định và đã qua cooldown time
                    current_time = time.time()
                    if (gesture_count >= self.gesture_threshold and 
                        current_time - last_command_time > self.cooldown_time):
                        
                        success = self.esp32_controller.send_command(gesture_id, gesture_name)
                        if success:
                            last_command_time = current_time
                        
                        gesture_count = 0  # Reset để tránh gửi liên tục
                    
                    # Hiển thị thông tin trên frame
                    info_text = f"{gesture_name} ({confidence:.2f})"
                    progress_text = f"Progress: {gesture_count}/{self.gesture_threshold}"
                    
                    cv2.putText(frame, info_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, progress_text, (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    last_gesture = None
                    gesture_count = 0
            else:
                last_gesture = None
                gesture_count = 0
            
            # Hiển thị trạng thái ESP32
            esp32_status = "🟢 Connected" if self.esp32_controller.is_connected else "🔴 Disconnected"
            cv2.putText(frame, f"ESP32: {esp32_status}", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Hiển thị frame
            cv2.imshow('Gesture Control - ESP32', frame)
            
            # Xử lý phím nhấn
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Thoát chương trình...")
                break
            elif key == ord('r'):
                print("\n🔄 Reconnecting ESP32...")
                self.esp32_controller.reconnect()
        
        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Đã dọn dẹp tài nguyên")


def find_latest_model(models_dir: str = "../Step_2/models") -> Optional[str]:
    """Tìm file model mới nhất trong thư mục models"""
    try:
        if not os.path.exists(models_dir):
            return None
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if not model_files:
            return None
        
        # Sắp xếp theo thời gian tạo (mới nhất)
        model_files.sort(key=lambda x: os.path.getctime(os.path.join(models_dir, x)), reverse=True)
        latest_model = os.path.join(models_dir, model_files[0])
        
        return latest_model
        
    except Exception as e:
        print(f"❌ Lỗi tìm model: {e}")
        return None


def main():
    """Hàm main cho Step 3"""
    print("="*60)
    print("      STEP 3: GESTURE CONTROL")
    print("="*60)
    
    # Tìm model tự động hoặc nhập thủ công
    latest_model = find_latest_model()
    
    if latest_model:
        print(f"📁 Tìm thấy model mới nhất: {os.path.basename(latest_model)}")
        use_latest = input("Sử dụng model này? (y/n, mặc định: y): ").lower()
        
        if use_latest in ['', 'y', 'yes']:
            model_path = latest_model
        else:
            model_path = input("Nhập đường dẫn đến file model (.pth): ")
    else:
        print("❌ Không tìm thấy model trong ../Step_2/models/")
        model_path = input("Nhập đường dẫn đến file model (.pth): ")
    
    # Kiểm tra file model
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file model: {model_path}")
        return
    
    # Nhập IP ESP32
    esp32_ip = input("🌐 Nhập IP của ESP32 (mặc định: 192.168.1.100): ") or "192.168.1.100"
    
    try:
        # Khởi tạo controller
        controller = GestureController(model_path, esp32_ip)
        
        # Chạy điều khiển
        controller.run_control()
        
    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")


if __name__ == "__main__":
    main()
