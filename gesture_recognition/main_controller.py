import os
import cv2
import time
import yaml
import torch
import numpy as np
import sys
import datetime
import platform
import serial
import subprocess
from serial.tools import list_ports
from hardware.modbus_controller import ModbusMaster
from pathlib import Path
import traceback
import argparse
from common.models import HandGestureModel, HandLandmarksDetector, label_dict_from_config_file
import mediapipe as mp
from torch import nn

class RelayGestureControl:
    """Relay control bằng hand gesture"""
    def __init__(self, model_path, config_path="../config.yaml", resolution=(640, 640), port=None, simulation=False):
        self.resolution = resolution
        self.height = 640
        self.width = 640
        self.port = port
        self.simulation = simulation
        self.detector = HandLandmarksDetector()
        self.status_text = None
        self.signs = label_dict_from_config_file(config_path)
        print(f"Loading model from: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            num_classes = len(self.signs)
            self.classifier = HandGestureModel(input_size=63, num_classes=num_classes)
            self.classifier.load_state_dict(model_dict)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            raise
        self.classifier.eval()
        self.controller = None
        if not simulation:
            try:
                self.controller = ModbusMaster(port=self.port)
            except Exception as e:
                print(f"Error connecting to hardware: {e}")
                print("Automatically switching to SIMULATION mode...")
                self.simulation = True
                self.controller = None
        if self.simulation:
            print("Simulation mode enabled. All relay commands will be simulated.")
        self.light1 = False
        self.light2 = False
        self.light3 = False
        self.last_command_time = time.time()
        self.command_debounce = 0.5
        # Debounce gesture
        self.gesture_buffer = []  # Lưu các gesture liên tiếp
        self.gesture_buffer_size = 8  # Số frame cần giống nhau để xác nhận
        self.last_stable_gesture = None
        print(f"Khởi tạo điều khiển relay bằng cử chỉ với {len(self.signs)} cử chỉ")
        print(f"Available gestures: {list(self.signs.values())}")

    def _control_relay(self, relay_num, state):
        if self.simulation:
            action = "ON" if state else "OFF"
            print(f"SIMULATION: Relay {relay_num} -> {action}")
            return
        try:
            if relay_num == 1:
                self.controller.switch_actuator_1(state)
            elif relay_num == 2:
                self.controller.switch_actuator_2(state)
            elif relay_num == 3:
                self.controller.switch_actuator_3(state)
        except Exception as e:
            print(f"Error controlling relay {relay_num}: {e}")

    def _control_all_relays(self, state):
        if self.simulation:
            action = "ON" if state else "OFF"
            print(f"SIMULATION: ALL RELAYS -> {action}")
            return
        try:
            if state:
                self.controller.all_on()
            else:
                self.controller.all_off()
        except Exception as e:
            print(f"Error controlling all relays: {e}")

    def light_simulation(self, img):
        """Hiển thị trạng thái đèn trên giao diện (chỉ dùng cho simulation)"""
        height, width, _ = img.shape
        rect_height = int(0.15 * height)
        rect_width = width
        white_rect = np.ones((rect_height, rect_width, 3), dtype=np.uint8) * 255
        # Viền đỏ
        cv2.rectangle(white_rect, (0, 0), (rect_width, rect_height), (0, 0, 255), 2)
        # Vị trí các đèn
        circle_radius = int(0.45 * rect_height)
        circle1_center = (int(rect_width * 0.25), int(rect_height / 2))
        circle2_center = (int(rect_width * 0.5), int(rect_height / 2))
        circle3_center = (int(rect_width * 0.75), int(rect_height / 2))
        on_color = (0, 255, 255)  # Vàng
        off_color = (0, 0, 0)     # Đen
        lights = [self.light1, self.light2, self.light3]
        centers = [circle1_center, circle2_center, circle3_center]
        labels = ["Light 1", "Light 2", "Light 3"]
        for i, (center, light, label) in enumerate(zip(centers, lights, labels)):
            color = on_color if light else off_color
            cv2.circle(white_rect, center, circle_radius, color, -1)
            cv2.circle(white_rect, center, circle_radius, (0,0,0), 2)
            cv2.putText(white_rect, label, (center[0]-circle_radius, center[1]+circle_radius+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        img = np.vstack((img, white_rect))
        return img

    def _debounce_gesture(self, gesture):
        """Chỉ trả về gesture nếu liên tục giống nhau N lần, ngược lại trả về None"""
        self.gesture_buffer.append(gesture)
        if len(self.gesture_buffer) > self.gesture_buffer_size:
            self.gesture_buffer.pop(0)
        # Nếu đủ buffer và tất cả giống nhau, xác nhận gesture ổn định
        if len(self.gesture_buffer) == self.gesture_buffer_size and all(g == gesture for g in self.gesture_buffer):
            if gesture != self.last_stable_gesture:
                self.last_stable_gesture = gesture
                return gesture
        return None

    def run(self):
        cam = cv2.VideoCapture(0)
        cam.set(3, self.width)
        cam.set(4, self.height)
        if not cam.isOpened():
            print("Lỗi: Không mở được camera!")
            return
        print("=== RELAY GESTURE CONTROL SYSTEM ===")
        print("Controls:")
        print("- 'q': Thoát")
        print("- 'r':Khởi động lại các đèn")
        print(f"- Gestures: {list(self.signs.values())}")
        print("=====================================")
        try:
            while cam.isOpened():
                ret, frame = cam.read()
                if not ret:
                    print("Lỗi: Không đọc được frame từ camera!")
                    break
                hand, img = self.detector.detect_hand(frame)
                stable_gesture = None
                if hand is not None and len(hand) > 0:
                    with torch.no_grad():
                        hand_landmark = torch.from_numpy(np.array(hand[0], dtype=np.float32).reshape(1, -1))
                        class_number = self.classifier.predict(hand_landmark, threshold=0.7).item()
                        if class_number != -1:
                            gesture = self.signs[class_number]
                            stable_gesture = self._debounce_gesture(gesture)
                            self.status_text = gesture
                        else:
                            self.status_text = "undefined command"
                            self.gesture_buffer.clear()
                else:
                    self.status_text = None
                    self.gesture_buffer.clear()
                if stable_gesture:
                    current_time = time.time()
                    if current_time - self.last_command_time >= self.command_debounce:
                        self.last_command_time = current_time
                        if stable_gesture == "light1_on" and not self.light1:
                            self.light1 = True
                            self._control_relay(1, True)
                        elif stable_gesture == "light1_off" and self.light1:
                            self.light1 = False
                            self._control_relay(1, False)
                        elif stable_gesture == "light2_on" and not self.light2:
                            self.light2 = True
                            self._control_relay(2, True)
                        elif stable_gesture == "light2_off" and self.light2:
                            self.light2 = False
                            self._control_relay(2, False)
                        elif stable_gesture == "turn_on":
                            self.light1 = self.light2 = self.light3 = True
                            self._control_all_relays(True)
                        elif stable_gesture == "turn_off":
                            self.light1 = self.light2 = self.light3 = False
                            self._control_all_relays(False)
                if self.status_text:
                    cv2.putText(img, f"Gesture: {self.status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                if self.simulation:
                    img = self.light_simulation(img)
                cv2.imshow("Relay Gesture Control", img)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self.light1 = self.light2 = self.light3 = False
                    self._control_all_relays(False)
                    print("Tất cả đèn đã được đặt lại OFF")
        except Exception as e:
            print(f"Lỗi trong vòng lặp chính: {e}")
            traceback.print_exc()
        finally:
            if cam.isOpened():
                cam.release()
            cv2.destroyAllWindows()
            try:
                self._control_all_relays(False)
                if self.controller:
                    self.controller.close()
                if self.simulation:
                    print("Kết thúc mô phỏng - Tất cả relay ảo đã tắt")
                else:
                    print("Tất cả relay đã tắt")
            except:
                pass
def main():
    parser = argparse.ArgumentParser(description="Relay Gesture Control System")
    parser.add_argument("--model", "-m", default="../Step_2/models/hand_gesture_model.pth", help="Path to trained model file")
    parser.add_argument("--config", "-c", default="../config.yaml", help="Path to configuration file")
    parser.add_argument("--resolution", "-r", default="640x640", help="Display resolution (WIDTHxHEIGHT)")
    parser.add_argument("--port", "-p", help="Specify COM port manually (e.g., COM3, COM4)")
    parser.add_argument("--list-ports", "-l", action="store_true", help="List all available COM ports and exit")
    parser.add_argument("--simulation", "-s", action="store_true", help="Run in simulation mode without hardware")
    args = parser.parse_args()
    if args.list_ports:
        print("Available COM ports:")
        for port in list_ports.comports():
            print(f"  {port}")
        return
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"Invalid resolution format: {args.resolution}. Using default 640x640")
        resolution = (640, 640)
    try:
        control_system = RelayGestureControl(
            model_path=args.model,
            config_path=args.config,
            resolution=resolution,
            port=args.port,
            simulation=getattr(args, 'simulation', False)
        )
        control_system.run()
    except KeyboardInterrupt:
        print("\nExiting by user request")
    except Exception as e:
        print(f"Error: {e}")
        if not getattr(args, 'simulation', False):
            print("Try running with --simulation flag to test gesture recognition")
            print("Command: python relay_controller.py --simulation")
        traceback.print_exc()

if __name__ == "__main__":
    main()

