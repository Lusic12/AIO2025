import os
import csv
import cv2
import mediapipe as mp
import yaml
from typing import Dict, Tuple, Optional, List
import numpy as np

# Add project root to path for cross-step imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gesture_recognization.common.models import HandLandmarksDetector, label_dict_from_config_file

def is_handsign_character(char: str) -> bool:
    """
    Check if the character is a valid key for gesture labeling.
    From generate_landmark_data.py
    """
    return ord('a') <= ord(char) < ord('q') or char == " "


class HandDatasetWriter:
    """Ghi dữ liệu landmarks vào file CSV"""
    
    def __init__(self, file_path: str):
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        self.file_path = file_path
        self.sample_count = 0
        
        # Mở file CSV để ghi
        self.csv_file = open(self.file_path, 'w', newline="", encoding="utf-8")
        self.file_writer = csv.writer(self.csv_file, delimiter=',')
        
        # Tạo header: x0,y0,z0,x1,y1,z1,...,x20,y20,z20,label
        header = []
        for i in range(21):  # 21 landmarks
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        header.append('label')  # Cột cuối là nhãn
        
        self.file_writer.writerow(header)
        print(f'Tạo file CSV: {self.file_path}')
    
    def add(self, hand_landmarks: List[float], label: int):
        """
        Thêm một mẫu dữ liệu vào CSV
        
        Args:
            hand_landmarks: List 63 số (21 landmarks x 3 tọa độ)
            label: Nhãn cử chỉ (0-5)
        """
        if len(hand_landmarks) != 63:
            raise ValueError(f"Cần 63 landmarks, nhận được {len(hand_landmarks)}")
        
        # Tạo row: [x0,y0,z0,...,x20,y20,z20,label]
        row = hand_landmarks + [label]
        self.file_writer.writerow(row)
        self.sample_count += 1
    
    def close(self):
        """Đóng file và hiển thị thống kê"""
        self.csv_file.close()
        print(f"Hoàn thành! Tổng cộng: {self.sample_count} mẫu")
        print(f"File lưu tại: {self.file_path}")

class GestureDataCollector:
    """Điều khiển thu thập dữ liệu cử chỉ tay"""
    
    def __init__(self, config_path: str = "../config.yaml"):
        # Đọc danh sách gesture từ config
        self.gesture_labels = label_dict_from_config_file(config_path)
        
        # Khởi tạo detector
        self.detector = HandLandmarksDetector()
        
        print("=== THU THẬP DỮ LIỆU CỬ CHỈ TAY ===")
        print(f"Có {len(self.gesture_labels)} loại cử chỉ:")
        for gid, gname in self.gesture_labels.items():
            print(f"  Phím '{chr(ord('a') + gid)}': {gname}")
    
    def collect_data(self, mode: str):
        """
        Thu thập dữ liệu cho một dataset (train/val/test)
        
        Args:
            mode: Loại dataset ('train', 'val', 'test')
        """
        print(f"\n=== THU THẬP DỮ LIỆU {mode.upper()} ===")
        print("Cách sử dụng:")
        print("- Nhấn 'a' đến 'f' để chọn cử chỉ")
        print("- Nhấn lần 2 để bắt đầu/dừng ghi")
        print("- Nhấn 'q' để thoát")
        print("=" * 40)
        
        # Tạo file CSV
        data_folder = "./data"
        os.makedirs(data_folder, exist_ok=True)
        csv_file = os.path.join(data_folder, f"landmarks_{mode}.csv")
        writer = HandDatasetWriter(csv_file)
        
        # Khởi tạo camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Trạng thái thu thập
        current_label = None    # Cử chỉ hiện tại
        is_recording = False    # Có đang ghi không?
        frame_count = 0         # Số frame đã ghi
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không đọc được từ camera!")
                break
            
            # Phát hiện tay
            landmarks, annotated_frame = self.detector.detect_hand(frame)
            
            # Ghi dữ liệu nếu đang recording
            if is_recording and landmarks and current_label is not None:
                writer.add(landmarks[0], current_label)
                frame_count += 1
                
                # Hiển thị trạng thái ghi
                cv2.putText(annotated_frame, f"DANG GHI: {frame_count}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.circle(annotated_frame, (50, 120), 20, (0, 0, 255), -1)
            
            # Hiển thị trạng thái hiện tại
            if current_label is not None:
                gesture_name = self.gesture_labels.get(current_label, "Unknown")
                status = f"Cu chi: {gesture_name} | Ghi: {'BAT' if is_recording else 'TAT'}"
                cv2.putText(annotated_frame, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Hiển thị có phát hiện tay không
            if landmarks:
                cv2.putText(annotated_frame, "PHAT HIEN TAY", 
                           (10, annotated_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(f'Thu thap du lieu - {mode.upper()}', annotated_frame)
            
            # Xử lý phím bấm
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif ord('a') <= key <= ord('f'):
                # Chọn cử chỉ (a=0, b=1, ..., f=5)
                new_label = key - ord('a')
                if new_label in self.gesture_labels:
                    if current_label == new_label:
                        # Bật/tắt ghi cho cử chỉ hiện tại
                        is_recording = not is_recording
                        gesture_name = self.gesture_labels[new_label]
                        if is_recording:
                            frame_count = 0
                            print(f"Bắt đầu ghi: {gesture_name}")
                        else:
                            print(f"Dừng ghi. Đã ghi {frame_count} mẫu")
                    else:
                        # Chọn cử chỉ mới
                        current_label = new_label
                        is_recording = False
                        frame_count = 0
                        gesture_name = self.gesture_labels[new_label]
                        print(f"Chọn cử chỉ: {gesture_name}")
        
        # Dọn dẹp
        cap.release()
        cv2.destroyAllWindows()
        writer.close()
        print(f"Hoàn thành thu thập dữ liệu {mode}")


def main():
    """Main function to run data collection"""
    collector = GestureDataCollector()
    
    # List of datasets to collect
    datasets = ["train", "val", "test"]
    
    for dataset in datasets:
        input(f"\nPress Enter to start collecting {dataset} data...")
        collector.collect_data(dataset)
        print(f"Finished collecting {dataset} data\n")
    
    print("Finished collecting all data!")


if __name__ == "__main__":
    main()
