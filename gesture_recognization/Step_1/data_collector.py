"""
Step 1: Hand Gesture Data Collection Module
==========================================

Mô-đun thu thập dữ liệu cử chỉ tay để huấn luyện mô hình AI.
Thu thập landmark data từ camera và lưu thành CSV files.

Sử dụng các functions từ generate_landmark_data.py để tương thích.

Author: ESP32 Gesture Recognition Project
"""

import os
import csv
import cv2
import mediapipe as mp
import yaml
from typing import Dict, Tuple, Optional, List
import numpy as np


def is_handsign_character(char: str) -> bool:
    """
    Kiểm tra ký tự có phải là phím hợp lệ để gán nhãn cử chỉ hay không.
    Từ generate_landmark_data.py
    """
    return ord('a') <= ord(char) < ord('q') or char == " "


def label_dict_from_config_file(yaml_path: str) -> Dict[int, str]:
    """
    Đọc cấu hình các lớp cử chỉ từ file YAML.
    Từ generate_landmark_data.py
    """
    try:
        with open(yaml_path, 'r', encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if 'gestures' not in config_data:
            raise KeyError("Key 'gestures' không tìm thấy trong file yaml")

        label_tag = config_data['gestures']

        for key, value in label_tag.items():
            if not isinstance(key, int) or not isinstance(value, str):
                raise ValueError(f"Invalid data type: key={key}, value={value}")
        print(f"Đã load thành công {len(label_tag)} gestures từ {yaml_path}")
        return label_tag
        
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file trong đường dẫn {yaml_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Lỗi khi đọc file YAML: {e}")
        raise
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        raise


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
    
    def detect_hand(self, frame: np.ndarray) -> Tuple[Optional[List], np.ndarray]:
        """
        Phát hiện bàn tay và trích xuất landmarks
        
        Args:
            frame: Khung hình từ camera
            
        Returns:
            Tuple của (landmarks, annotated_frame)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        annotated_frame = frame.copy()
        landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ landmarks lên frame
                self.mp_draw.draw_landmarks(
                    annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Trích xuất tọa độ landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                landmarks_list.append(landmarks)
        
        return landmarks_list if landmarks_list else None, annotated_frame


class HandDatasetWriter:
    """
    Lớp ghi dữ liệu landmarks vào file CSV
    Tương thích với HandDatasetWriter từ generate_landmark_data.py
    """
    
    def __init__(self, file_path: str, buffer_size: int = 100):
        """
        Khởi tạo writer với file path và buffer size
        """
        try: 
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self.file_path = file_path
            self.buffer_size = buffer_size
            self.sample_count = 0
            self.buffer_count = 0

            self.csv_file = open(self.file_path, 'w', newline="", encoding="utf-8") 
            self.file_writer = csv.writer(
                self.csv_file,
                delimiter=',',
                quotechar='|',
                quoting=csv.QUOTE_MINIMAL
            )

            # Ghi header cho file CSV
            header = []
            for i in range(21):  # MediaPipe có 21 landmarks cho mỗi bàn tay
                header.extend([f'x{i}', f'y{i}', f'z{i}'])
            header.append('label')
            self.file_writer.writerow(header)

            print(f'Đã khởi tạo file csv tại {self.file_path}')

        except Exception as e:
            print(f"Lỗi khi khởi tạo HandDatasetWriter: {e}")
            raise
    
    def add(self, hand_landmarks: List[float], label: int):
        """Thêm một mẫu dữ liệu vào CSV"""
        if len(hand_landmarks) != 63:
            raise ValueError(f"Số lượng landmarks không đúng: {len(hand_landmarks)} (cần 63)")
        
        try:
            # Chuyển đổi landmarks thành list phẳng
            flattened_landmarks = np.array(hand_landmarks).flatten().tolist()
            
            # Ghi vào file: [x1, y1, z1, x2, y2, z2, ..., label]
            row = flattened_landmarks + [label]
            self.file_writer.writerow(row)
            
            self.sample_count += 1
            self.buffer_count += 1
            
            # Auto flush buffer khi đạt ngưỡng
            if self.buffer_count >= self.buffer_size:
                self.flush()
                
        except Exception as e:
            print(f"Lỗi khi ghi dữ liệu: {e}")
            raise
    
    def flush(self):
        """Flush buffer và đảm bảo dữ liệu được ghi vào disk"""
        self.csv_file.flush()
        self.buffer_count = 0
    
    def close(self):
        """Đóng file CSV và giải phóng tài nguyên"""
        try:
            self.flush()  # Flush buffer cuối cùng
            self.csv_file.close()
            print(f"Đã đóng file {self.file_path}. Tổng số mẫu: {self.sample_count}")
        except Exception as e:
            print(f"Lỗi khi đóng file: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class GestureDataCollector:
    """Lớp chính để thu thập dữ liệu cử chỉ tay"""
    
    def __init__(self, config_path: str = "../config.yaml"):
        self.config_path = config_path
        # Sử dụng function từ generate_landmark_data.py
        self.gesture_labels = label_dict_from_config_file(config_path)
        self.detector = HandLandmarksDetector()
        
    def _load_gesture_config(self) -> Dict[int, str]:
        """Đọc cấu hình cử chỉ từ file YAML - sử dụng function từ generate_landmark_data.py"""
        return label_dict_from_config_file(self.config_path)
    
    def _is_valid_gesture_key(self, char: str) -> bool:
        """Kiểm tra phím nhấn có hợp lệ không - sử dụng function từ generate_landmark_data.py"""
        return is_handsign_character(char)
    
    def _char_to_label(self, char: str) -> int:
        """Chuyển đổi ký tự thành label"""
        if char == ' ':
            return 0
        return ord(char) - ord('a') + 1
    
    def collect_data(self, mode: str, data_path: str = "./data", 
                    img_path: str = "./sample_images", 
                    resolution: Tuple[int, int] = (1280, 720)):
        """
        Thu thập dữ liệu cho một mode cụ thể (train/val/test)
        
        Args:
            mode: Chế độ thu thập (train/val/test)
            data_path: Thư mục lưu file CSV
            img_path: Thư mục lưu ảnh mẫu
            resolution: Độ phân giải camera
        """
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(img_path, exist_ok=True)
        
        # Khởi tạo writer và camera
        csv_file = os.path.join(data_path, f"landmarks_{mode}.csv")
        writer = HandDatasetWriter(csv_file)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        current_label = None
        is_recording = False
        
        self._print_instructions(mode)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc frame từ camera!")
                break
            
            # Phát hiện bàn tay
            landmarks, annotated_frame = self.detector.detect_hand(frame)
            
            # Hiển thị thông tin trạng thái
            self._draw_status(annotated_frame, mode, current_label, is_recording)
            
            # Ghi dữ liệu nếu đang recording
            if is_recording and landmarks and current_label is not None:
                writer.add(landmarks[0], current_label)
                cv2.putText(annotated_frame, "RECORDING...", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            cv2.imshow(f'Hand Gesture Data Collection - {mode.upper()}', annotated_frame)
            
            # Xử lý phím nhấn
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key != 255:  # Có phím được nhấn
                char = chr(key)
                if self._is_valid_gesture_key(char):
                    new_label = self._char_to_label(char)
                    
                    if new_label in self.gesture_labels:
                        if current_label == new_label:
                            # Toggle recording
                            is_recording = not is_recording
                            action = "Bắt đầu" if is_recording else "Dừng"
                            print(f"{action} ghi dữ liệu cho: {self.gesture_labels[new_label]}")
                        else:
                            # Chọn cử chỉ mới
                            current_label = new_label
                            is_recording = False
                            print(f"Đã chọn cử chỉ: {self.gesture_labels[new_label]}")
                    else:
                        print(f"Cử chỉ {new_label} không được định nghĩa trong config!")
                
                elif key == ord('s') and landmarks and current_label is not None:
                    # Lưu ảnh mẫu
                    gesture_name = self.gesture_labels[current_label]
                    img_name = f"{gesture_name}_{mode}_sample.jpg"
                    img_path_full = os.path.join(img_path, img_name)
                    cv2.imwrite(img_path_full, annotated_frame)
                    print(f"Đã lưu ảnh mẫu: {img_name}")
        
        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()
        writer.close()
        print(f"Hoàn thành thu thập dữ liệu cho {mode}")
    
    def _print_instructions(self, mode: str):
        """In hướng dẫn sử dụng"""
        print(f"\n{'='*50}")
        print(f"THU THẬP DỮ LIỆU CHO {mode.upper()}")
        print(f"{'='*50}")
        print("Hướng dẫn sử dụng:")
        print("- Nhấn các phím tương ứng với cử chỉ:")
        for label, name in self.gesture_labels.items():
            if label == 0:
                key_char = "SPACE"
            else:
                key_char = chr(ord('a') + label - 1)
            print(f"  '{key_char}': {name}")
        print("\n- Nhấn cùng phím 2 lần để bắt đầu/dừng ghi")
        print("- Nhấn 's': Lưu ảnh mẫu")
        print("- Nhấn 'q': Thoát chương trình")
        print(f"{'='*50}\n")
    
    def _draw_status(self, frame: np.ndarray, mode: str, current_label: Optional[int], is_recording: bool):
        """Vẽ thông tin trạng thái lên frame"""
        # Vẽ nền cho text
        cv2.rectangle(frame, (5, 5), (600, 90), (0, 0, 0), -1)
        
        status_text = f"Mode: {mode.upper()}"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if current_label is not None:
            gesture_name = self.gesture_labels.get(current_label, "Unknown")
            gesture_text = f"Gesture: {gesture_name}"
            cv2.putText(frame, gesture_text, (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        recording_text = f"Recording: {'ON' if is_recording else 'OFF'}"
        color = (0, 255, 0) if is_recording else (0, 0, 255)
        cv2.putText(frame, recording_text, (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    """Hàm main để chạy thu thập dữ liệu"""
    collector = GestureDataCollector()
    
    # Danh sách các dataset cần thu thập
    datasets = ["train", "val", "test"]
    
    for dataset in datasets:
        input(f"\nNhấn Enter để bắt đầu thu thập dữ liệu {dataset}...")
        collector.collect_data(dataset)
        print(f"Hoàn thành thu thập dữ liệu {dataset}\n")
    
    print("Hoàn thành thu thập tất cả dữ liệu!")


if __name__ == "__main__":
    main()
