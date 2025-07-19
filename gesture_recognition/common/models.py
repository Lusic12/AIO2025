import yaml
import torch
import numpy as np
import mediapipe as mp
import cv2
from torch import nn
from typing import Dict, Tuple, List, Optional

class HandGestureModel(nn.Module):
    """Mô hình neural network cho phân loại cử chỉ tay"""
    
    def __init__(self, input_size=63, num_classes=6):
        super(HandGestureModel, self).__init__()
        
        # Mạng neural network với 4 lớp
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),    # Lớp 1: 63 -> 256
            nn.ReLU(),                     # Hàm kích hoạt
            nn.Dropout(0.1),               # Dropout tránh overfitting
            
            nn.Linear(256, 128),           # Lớp 2: 256 -> 128
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),            # Lớp 3: 128 -> 64
            nn.ReLU(),
            
            nn.Linear(64, num_classes)     # Lớp output: 64 -> 6
        )
    
    def forward(self, x):
        """Tính toán forward pass"""
        return self.network(x)
    
    def predict(self, x, threshold=0.7):
        """
        Dự đoán với ngưỡng tin cậy
        
        Args:
            x: Input tensor
            threshold: Ngưỡng tin cậy (0.7 = 70%)
            
        Returns:
            predicted: Class được dự đoán hoặc -1 nếu không tin cậy
        """
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = torch.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)
            
            # Trả về -1 nếu tin cậy thấp
            if max_prob.item() < threshold:
                return torch.tensor(-1)
            return predicted

class HandLandmarksDetector:
    """Phát hiện và trích xuất tọa độ 21 điểm trên bàn tay"""
    
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.5):
        # Khởi tạo MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,        # Xử lý video (không phải ảnh)
            max_num_hands=1,                # Chỉ nhận diện 1 tay
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def detect_hand(self, frame: np.ndarray) -> Tuple[Optional[List], np.ndarray]:
        """
        Phát hiện tay và trả về tọa độ 21 điểm landmarks
        
        Args:
            frame: Khung hình từ camera (BGR format)
            
        Returns:
            landmarks_list: Danh sách tọa độ 21 điểm (x,y,z) hoặc None
            annotated_frame: Khung hình có vẽ landmarks
        """
        # Chuyển đổi BGR sang RGB (MediaPipe yêu cầu)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Phát hiện tay
        results = self.hands.process(rgb_frame)
        
        # Sao chép frame để vẽ landmarks
        annotated_frame = frame.copy()
        landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ landmarks lên frame
                self.mp_draw.draw_landmarks(
                    annotated_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Trích xuất tọa độ
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmarks_list.append(landmarks)
        
        return landmarks_list if landmarks_list else None, annotated_frame

def label_dict_from_config_file(config_path: str = "config.yaml") -> Dict[int, str]:
    """
    Đọc danh sách gesture từ file config
    
    Args:
        config_path: Đường dẫn file config.yaml
        
    Returns:
        Dict mapping từ ID gesture sang tên gesture
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config.get('gestures', {})
    except FileNotFoundError:
        print(f"Không tìm thấy file: {config_path}")
        return {}
    except Exception as e:
        print(f"Lỗi đọc config: {e}")
        return {}