import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import yaml

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gesture_recognition.common.models import label_dict_from_config_file, HandGestureModel

# Load configuration
with open('..\config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

train_path = config['Dataset']['train']
val_path = config['Dataset']['val']
test_path = config['Dataset']['test']

class HandGestureDataset(Dataset):
    """Dataset PyTorch cho dữ liệu cử chỉ tay"""
    
    def __init__(self, csv_path: str):
        print(f"Đang tải dữ liệu từ: {csv_path}")
        
        # Đọc file CSV
        self.data = pd.read_csv(csv_path)
        print(f"Kích thước dữ liệu: {self.data.shape}")
        
        # Tách features (63 cột đầu) và labels (cột cuối)
        self.features = self.data.iloc[:, :-1].values.astype(np.float32)
        self.labels = self.data.iloc[:, -1].values.astype(np.int64)
        
        # Hiển thị thống kê
        unique_labels = np.unique(self.labels)
        print(f"Các lớp có trong dữ liệu: {unique_labels}")
        for label in unique_labels:
            count = np.sum(self.labels == label)
            print(f"  Lớp {label}: {count} mẫu")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Trả về (features, label) dưới dạng tensor
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

class EarlyStopper:
    """Early stopping để ngăn overfitting"""
    
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
    """Điều khiển huấn luyện mô hình cử chỉ tay"""
    
    def __init__(self, config_path: str = "../config.yaml"):
        # Đọc cấu hình
        self.gesture_labels = label_dict_from_config_file(config_path)
        self.num_classes = len(self.gesture_labels)
        
        print("=== HUẤN LUYỆN MÔ HÌNH ===")
        print(f"Số lớp cử chỉ: {self.num_classes}")
        for gid, gname in self.gesture_labels.items():
            print(f"  {gid}: {gname}")
    
    def prepare_data(self, batch_size: int = 32, train_path=None, val_path=None):
        """Chuẩn bị dữ liệu train và validation"""
        # Kiểm tra file tồn tại
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Không tìm thấy: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Không tìm thấy: {val_path}")
        
        # Tạo dataset
        train_dataset = HandGestureDataset(train_path)
        val_dataset = HandGestureDataset(val_path)
        
        # Tạo data loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader, epochs=200):
        """Huấn luyện mô hình"""
        print(f"\nBắt đầu huấn luyện {epochs} epochs...")
        
        # Khởi tạo mô hình
        model = HandGestureModel(input_size=63, num_classes=self.num_classes)
        criterion = nn.CrossEntropyLoss()  # Hàm loss
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer
        early_stopper = EarlyStopper(patience=20, min_delta=0.01)  # Khởi tạo early stopping
        # Lưu mô hình tốt nhất
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # === TRAINING PHASE ===
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            # === VALIDATION PHASE ===
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            # Tính toán metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            
            print(f"Epoch {epoch+1}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            # Lưu mô hình tốt nhất
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                print(f" Mô hình tốt nhất mới!")
            
            # Kiểm tra early stopping
            if early_stopper.early_stop(avg_val_loss):
                print(f"\nEarly stopping tại epoch {epoch+1}")
                break
        
        # Load mô hình tốt nhất
        if best_model_state:
            model.load_state_dict(best_model_state)
            print(f"\nĐã load mô hình tốt nhất (Val Loss: {best_val_loss:.4f})")
        
        return model
    
    def save_model(self, model, save_path="./models"):
        """Lưu mô hình đã huấn luyện"""
        os.makedirs(save_path, exist_ok=True)
        
        # Tạo tên file với timestamp
        model_filename = f"hand_gesture_model.pth"
        model_path = os.path.join(save_path, model_filename)
        
        # Lưu mô hình và metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'gesture_labels': self.gesture_labels,
            'num_classes': self.num_classes,
            'input_size': 63,
        }, model_path)
        
        
        return model_path

    def test_model(self, model, test_path):
        """Kiểm tra mô hình với test set"""
        test_dataset = HandGestureDataset(test_path)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"\nKết quả trên test set:")
        print(f"  Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
        
        return test_accuracy, avg_test_loss

def main():
    print("TRAINING MODEL")
    
    trainer = HandGestureTrainer()
    train_loader, val_loader = trainer.prepare_data(batch_size=32, train_path=train_path, val_path=val_path)
    model = trainer.train_model(train_loader, val_loader)
    model_path = trainer.save_model(model)
    
    print(f"Training complete! Model saved at: {model_path}")
    
    # Test model on the test dataset
    print("=== TESTING THE MODEL ===")
    test_accuracy, test_loss = trainer.test_model(model, test_path)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
