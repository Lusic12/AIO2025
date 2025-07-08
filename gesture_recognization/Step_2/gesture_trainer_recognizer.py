"""
Step 2: Hand Gesture Training and Recognition Module
==================================================

Mô-đun huấn luyện mô hình và nhận diện cử chỉ tay để điều khiển ESP32.

Author: ESP32 Gesture Recognition Project
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import yaml
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy


class HandGestureDataset(Dataset):
    """Dataset class cho hand gesture landmarks"""
    
    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path)
        self.features = self.data.iloc[:, :-1].values.astype(np.float32)  # Tất cả cột trừ cột cuối
        self.labels = self.data.iloc[:, -1].values.astype(np.int64)       # Cột cuối là label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


class HandGestureModel(nn.Module):
    """Neural Network cho hand gesture recognition"""
    def __init__(self, input_size=63, num_classes=6):
        super(HandGestureModel, self).__init__()
        
        self.network = nn.Sequential(nn.Linear(input_size,256),
                                   nn.ReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(256,128),
                                   nn.ReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(128,64),
                                   nn.Linear(64,num_classes)
                                   )
        
    
    def forward(self, x):
        return self.network(x)
    
    def get_logits(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            return torch.argmax(outputs, dim=1)


class EarlyStopper:
    """Early stopping để tránh overfitting"""
    
    def __init__(self, patience=30, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def early_stop(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class HandGestureTrainer:
    """Lớp training model"""
    
    def __init__(self, config_path: str = "../config.yaml"):
        self.config_path = config_path
        self.gesture_labels = self._load_gesture_config()
        self.num_classes = len(self.gesture_labels)
        
    def _load_gesture_config(self) -> Dict[int, str]:
        """Đọc cấu hình cử chỉ từ file YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config.get('gestures', {})
        except FileNotFoundError:
            print(f"Không tìm thấy file config: {self.config_path}")
            return {}
    
    def prepare_data(self, data_folder: str = "../Step_1/data"):
        """Chuẩn bị dữ liệu training và validation"""
        train_path = os.path.join(data_folder, "landmarks_train.csv")
        val_path = os.path.join(data_folder, "landmarks_val.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError("Không tìm thấy file dữ liệu. Hãy chạy Step 1 trước!")
        
        train_dataset = HandGestureDataset(train_path)
        val_dataset = HandGestureDataset(val_path)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader, epochs=200):
        """Huấn luyện mô hình"""
        model = HandGestureModel(num_classes=self.num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        early_stopper = EarlyStopper()
        
        best_val_loss = float('inf')
        best_model_state = None
        
        print(f"Bắt đầu huấn luyện với {self.num_classes} lớp cử chỉ...")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_acc = Accuracy(num_classes=self.num_classes, task='multiclass')
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_acc.update(model.get_logits(data), target)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_acc = Accuracy(num_classes=self.num_classes, task='multiclass')
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    val_acc.update(model.get_logits(data), target)
            
            # Tính loss và accuracy trung bình
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = train_acc.compute().item()
            val_accuracy = val_acc.compute().item()
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Lưu model tốt nhất
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                print(f"  -> New best model!")
            
            # Early stopping
            if early_stopper.early_stop(avg_val_loss):
                print(f"Early stopping tại epoch {epoch+1}")
                break
        
        # Lưu model tốt nhất
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return model
    
    def save_model(self, model, save_path="./models"):
        """Lưu model đã huấn luyện"""
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(save_path, f"hand_gesture_model_{timestamp}.pth")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'gesture_labels': self.gesture_labels,
            'num_classes': self.num_classes
        }, model_path)
        
        print(f"Model đã được lưu tại: {model_path}")
        return model_path

def main():
    print("=== STEP 2: TRAINING VÀ RECOGNITION ===")
    print("Training model")
    
    
    trainer = HandGestureTrainer()
    train_loader, val_loader = trainer.prepare_data()
    model = trainer.train_model(train_loader, val_loader)
    model_path = trainer.save_model(model)
    print(f"Hoàn thành training! Model lưu tại: {model_path}")
if __name__ == "__main__":
    main()
