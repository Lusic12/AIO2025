"""
HTTP ESP32 Controller Module
===========================

Module điều khiển ESP32 qua giao thức HTTP
Sử dụng cử chỉ tay để gửi lệnh điều khiển đến ESP32 HTTP server
"""

import os
import cv2
import time
import yaml
import torch
import numpy as np
import sys
import datetime
import requests
from pathlib import Path

# Thêm thư mục gốc vào path để import từ các Step khác
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import mediapipe as mp
    from torch import nn
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install required packages: pip install -r ../requirements.txt")
    sys.exit(1)


class ESP32Controller:
    """
    Class quản lý kết nối và điều khiển ESP32 qua HTTP requests.
    Hỗ trợ điều khiển 3 đèn LED hoặc relay kết nối với ESP32.
    """
    
    def __init__(self, ip_address="192.168.1.100", port=80, timeout=2):
        """
        Khởi tạo kết nối với ESP32 HTTP server.
        
        Args:
            ip_address: Địa chỉ IP của ESP32
            port: Cổng của HTTP server trên ESP32
            timeout: Thời gian chờ kết nối (giây)
        """
        self.base_url = f"http://{ip_address}:{port}"
        self.timeout = timeout
        
        # Thử kết nối
        try:
            self.test_connection()
            print(f"Connected to ESP32 at {self.base_url}")
        except Exception as e:
            raise ConnectionError(f"Could not connect to ESP32: {e}")
    
    def test_connection(self):
        """Kiểm tra kết nối với ESP32"""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            if response.status_code != 200:
                raise ConnectionError(f"ESP32 returned status code {response.status_code}")
            return True
        except requests.RequestException as e:
            raise ConnectionError(f"Connection error: {e}")
    
    def send_command(self, endpoint, params=None):
        """Gửi lệnh đến ESP32"""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                return True, response.json() if response.headers.get('content-type') == 'application/json' else response.text
            else:
                print(f"ESP32 returned status code {response.status_code}")
                return False, None
        except requests.RequestException as e:
            print(f"Request error: {e}")
            return False, None
    
    def switch_light(self, light_id, state):
        """Điều khiển đèn LED hoặc relay"""
        action = "on" if state else "off"
        return self.send_command(f"control/{light_id}/{action}")
    
    def switch_light_1(self, state):
        """Điều khiển đèn 1"""
        return self.switch_light(1, state)
    
    def switch_light_2(self, state):
        """Điều khiển đèn 2"""
        return self.switch_light(2, state)
    
    def switch_light_3(self, state):
        """Điều khiển đèn 3"""
        return self.switch_light(3, state)
    
    def all_on(self):
        """Bật tất cả các đèn"""
        success = True
        success &= self.switch_light_1(True)[0]
        time.sleep(0.1)
        success &= self.switch_light_2(True)[0]
        time.sleep(0.1)
        success &= self.switch_light_3(True)[0]
        return success
    
    def all_off(self):
        """Tắt tất cả các đèn"""
        success = True
        success &= self.switch_light_1(False)[0]
        time.sleep(0.1)
        success &= self.switch_light_2(False)[0]
        time.sleep(0.1)
        success &= self.switch_light_3(False)[0]
        return success
    
    def get_status(self):
        """Lấy trạng thái của các đèn"""
        success, data = self.send_command("status")
        if success:
            return data
        else:
            return {"light1": False, "light2": False, "light3": False}


class HandLandmarksDetector:
    """Phát hiện và trích xuất landmarks từ bàn tay sử dụng MediaPipe"""
    
    def __init__(self, max_hands=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def detect_hand(self, frame):
        """
        Phát hiện bàn tay và trích xuất landmarks
        """
        hands = []
        frame = cv2.flip(frame, 1)  # Lật ngang để dễ tương tác
        annotated_image = frame.copy()
        
        # Convert sang RGB (MediaPipe cần input RGB)
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract landmarks
                hand = []
                for landmark in hand_landmarks.landmark:
                    x, y, z = landmark.x, landmark.y, landmark.z
                    hand.extend([x, y, z])
                
                hands.append(hand)
                
        return hands, annotated_image


class GestureClassifier(nn.Module):
    """Neural Network cho phân loại cử chỉ tay"""
    
    def __init__(self, input_size=63, num_classes=7):
        super(GestureClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(128, num_classes),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def predict(self, x, threshold=0.9):
        """Dự đoán với confidence threshold"""
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        chosen_ind = torch.argmax(softmax_prob, dim=1)
        # Trả về -1 nếu confidence thấp hơn ngưỡng
        return torch.where(softmax_prob[0, chosen_ind] > threshold, chosen_ind, -1)


def label_dict_from_config_file(config_path):
    """Đọc cấu hình các lớp cử chỉ từ file YAML"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        
        if 'gestures' not in config_data:
            print(f"Error: 'gestures' section not found in {config_path}")
            return {}
            
        return config_data['gestures']
        
    except Exception as e:
        print(f"Error loading gesture config: {e}")
        return {}


class ESP32GestureControl:
    """Điều khiển ESP32 bằng cử chỉ tay qua HTTP"""
    
    def __init__(self, model_path, esp32_ip, config_path="../config.yaml", resolution=(1280, 720)):
        self.resolution = resolution
        self.height = 720
        self.width = 1280
        self.esp32_ip = esp32_ip

        # Khởi tạo các components
        self.detector = HandLandmarksDetector()
        self.status_text = None
        self.signs = label_dict_from_config_file(config_path)
        
        # Load model
        self.classifier = GestureClassifier(num_classes=len(self.signs))
        self.classifier.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.classifier.eval()

        # Khởi tạo ESP32 controller
        try:
            self.controller = ESP32Controller(ip_address=esp32_ip)
            print(f"ESP32 controller initialized successfully at {esp32_ip}")
            
            # Lấy trạng thái ban đầu của các đèn
            status = self.controller.get_status()
            if isinstance(status, dict):
                self.light1 = status.get("light1", False)
                self.light2 = status.get("light2", False)
                self.light3 = status.get("light3", False)
            else:
                self.light1 = self.light2 = self.light3 = False
            
        except Exception as e:
            print(f"Error initializing ESP32 controller: {e}")
            print("Please check your ESP32 connection and IP address")
            sys.exit(1)
            
        # Tracking variables
        self.last_command_time = time.time()
        self.command_debounce = 1.0  # 1 second debounce time
        
        print(f"ESP32 Gesture Control initialized with {len(self.signs)} gestures")
        print(f"Available gestures: {list(self.signs.values())}")
    
    def _log_action(self, gesture, action):
        """Ghi log hành động"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / "esp32_control.log"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_file, 'a') as f:
            f.write(f"{timestamp} | {gesture} | {action} | " 
                   f"Lights: [{int(self.light1)}, {int(self.light2)}, {int(self.light3)}]\n")

    def light_simulation(self, img):
        """Hiển thị trạng thái đèn trên giao diện"""
        # Append a white rectangle at the bottom of the image
        height, width, _ = img.shape
        rect_height = int(0.15 * height)
        rect_width = width
        white_rect = np.ones((rect_height, rect_width, 3), dtype=np.uint8) * 255

        # Draw a red border around the rectangle
        cv2.rectangle(white_rect, (0, 0), (rect_width, rect_height), (0, 0, 255), 2)

        # Calculate circle positions
        circle_radius = int(0.45 * rect_height)
        circle1_center = (int(rect_width * 0.25), int(rect_height / 2))
        circle2_center = (int(rect_width * 0.5), int(rect_height / 2))
        circle3_center = (int(rect_width * 0.75), int(rect_height / 2))

        # Draw the circles
        on_color = (0, 255, 255)  # Yellow
        off_color = (0, 0, 0)     # Black
        
        lights = [self.light1, self.light2, self.light3]
        centers = [circle1_center, circle2_center, circle3_center]
        labels = ["Light 1", "Light 2", "Light 3"]
        
        for i, (center, light, label) in enumerate(zip(centers, lights, labels)):
            color = on_color if light else off_color
            cv2.circle(white_rect, center, circle_radius, color, -1)
            
            # Add label
            text_y = center[1] + circle_radius + 15
            cv2.putText(white_rect, label, 
                      (center[0] - 30, text_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Append the white rectangle to the bottom of the image
        img = np.vstack((img, white_rect))
        return img

    def run(self):
        """Chạy hệ thống nhận diện và điều khiển"""
        cam = cv2.VideoCapture(0)
        cam.set(3, self.width)
        cam.set(4, self.height)
        
        if not cam.isOpened():
            print("Error: Could not open camera")
            return
            
        print("=== ESP32 GESTURE CONTROL SYSTEM ===")
        print(f"Connected to ESP32 at: {self.esp32_ip}")
        print("Controls:")
        print("- 'q': Quit")
        print("- 'r': Reset all lights")
        print("- 's': Sync status with ESP32")
        print(f"- Gestures: {list(self.signs.values())}")
        print("====================================")
            
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        try:
            while cam.isOpened():
                ret, frame = cam.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                    
                # Detect hand landmarks
                hand, img = self.detector.detect_hand(frame)
                
                # Process detected hand
                if len(hand) > 0:
                    with torch.no_grad():
                        # Convert landmarks to tensor
                        hand_landmark = torch.from_numpy(
                            np.array(hand[0], dtype=np.float32).reshape(1, -1)
                        )
                        
                        # Predict gesture
                        class_number = self.classifier.predict(hand_landmark, threshold=0.85).item()
                        
                        # Process valid gesture
                        if class_number != -1:
                            self.status_text = self.signs[class_number]
                            
                            # Debounce check
                            current_time = time.time()
                            if current_time - self.last_command_time >= self.command_debounce:
                                self.last_command_time = current_time
                                
                                # Process gesture commands
                                if self.status_text == "light1_on":
                                    if not self.light1:
                                        success, _ = self.controller.switch_light_1(True)
                                        if success:
                                            self.light1 = True
                                            self._log_action("light1_on", "Light 1 ON")
                                        
                                elif self.status_text == "light1_off":
                                    if self.light1:
                                        success, _ = self.controller.switch_light_1(False)
                                        if success:
                                            self.light1 = False
                                            self._log_action("light1_off", "Light 1 OFF")
                                        
                                elif self.status_text == "light2_on":
                                    if not self.light2:
                                        success, _ = self.controller.switch_light_2(True)
                                        if success:
                                            self.light2 = True
                                            self._log_action("light2_on", "Light 2 ON")
                                        
                                elif self.status_text == "light2_off":
                                    if self.light2:
                                        success, _ = self.controller.switch_light_2(False)
                                        if success:
                                            self.light2 = False
                                            self._log_action("light2_off", "Light 2 OFF")
                                        
                                elif self.status_text == "turn_on":
                                    if not (self.light1 and self.light2 and self.light3):
                                        if self.controller.all_on():
                                            self.light1 = self.light2 = self.light3 = True
                                            self._log_action("turn_on", "ALL ON")
                                        
                                elif self.status_text == "turn_off":
                                    if self.light1 or self.light2 or self.light3:
                                        if self.controller.all_off():
                                            self.light1 = self.light2 = self.light3 = False
                                            self._log_action("turn_off", "ALL OFF")
                        else:
                            self.status_text = "undefined command"
                else:
                    self.status_text = None
                
                # Add UI elements
                img = self.light_simulation(img)
                
                # Display gesture status
                if self.status_text:
                    cv2.putText(img, f"Gesture: {self.status_text}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Calculate and show FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1.0:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                cv2.putText(img, f"FPS: {fps:.1f}", 
                           (10, img.shape[0] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show ESP32 connection info
                cv2.putText(img, f"ESP32: {self.esp32_ip}", 
                           (img.shape[1] - 250, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display the frame
                cv2.namedWindow("ESP32 Gesture Control", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("ESP32 Gesture Control", self.resolution[0], self.resolution[1])
                cv2.imshow("ESP32 Gesture Control", img)
                
                # Process keys
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    # Reset all lights
                    if self.controller.all_off():
                        self.light1 = self.light2 = self.light3 = False
                        print("All lights reset")
                elif key == ord("s"):
                    # Sync with ESP32 status
                    try:
                        status = self.controller.get_status()
                        if isinstance(status, dict):
                            self.light1 = status.get("light1", False)
                            self.light2 = status.get("light2", False)
                            self.light3 = status.get("light3", False)
                            print(f"Synced with ESP32: Light1={self.light1}, Light2={self.light2}, Light3={self.light3}")
                    except:
                        print("Failed to sync with ESP32")
        
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Cleanup
            if cam.isOpened():
                cam.release()
            cv2.destroyAllWindows()
            
            # Turn off all lights before exit
            try:
                self.controller.all_off()
                print("All lights turned off")
            except:
                pass


def main():
    """Main function"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ESP32 HTTP Gesture Control System")
    parser.add_argument("--model", "-m", 
                      default="../Step_2/models/hand_gesture_model.pth",
                      help="Path to trained model file")
    parser.add_argument("--config", "-c", 
                      default="../config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--resolution", "-r", 
                      default="1280x720",
                      help="Display resolution (WIDTHxHEIGHT)")
    parser.add_argument("--ip", "-i", 
                      default="192.168.1.100",
                      help="ESP32 IP address")
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"Invalid resolution format: {args.resolution}. Using default 1280x720")
        resolution = (1280, 720)
    
    # Create and run gesture control system
    try:
        control_system = ESP32GestureControl(
            model_path=args.model,
            esp32_ip=args.ip,
            config_path=args.config,
            resolution=resolution
        )
        control_system.run()
    except KeyboardInterrupt:
        print("\nExiting by user request")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


# Test the ESP32 controller directly
def test_esp32_controller(ip_address="192.168.1.100"):
    """Test the ESP32 controller"""
    print(f"Testing ESP32 Controller at {ip_address}...")
    try:
        controller = ESP32Controller(ip_address=ip_address)
        print("Connected successfully. Testing light control...")
        
        for i in range(3):
            print(f"Test cycle {i+1}/3")
            
            # Bật từng đèn lần lượt
            print("Turning ON lights one by one...")
            controller.switch_light_1(True)
            time.sleep(1)
            controller.switch_light_2(True)
            time.sleep(1)
            controller.switch_light_3(True)
            time.sleep(1)
            
            # Tắt tất cả
            print("Turning OFF all lights...")
            controller.all_off()
            time.sleep(1)
            
            # Bật tất cả
            print("Turning ON all lights...")
            controller.all_on()
            time.sleep(1)
            
            # Tắt tất cả
            print("Turning OFF all lights...")
            controller.all_off()
            time.sleep(1)
        
        # Kiểm tra trạng thái
        status = controller.get_status()
        print(f"ESP32 status: {status}")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test mode
        if len(sys.argv) > 2:
            test_esp32_controller(sys.argv[2])
        else:
            test_esp32_controller()
    else:
        # Normal operation
        main()
