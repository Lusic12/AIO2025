"""
Main entry point for ESP32 Hand Gesture Recognition Project
===========================================================

Script chính để chạy toàn bộ quy trình từ thu thập dữ liệu đến nhận diện cử chỉ.
"""

import os
import sys

def print_menu():
    """In menu lựa chọn chức năng"""
    print("\n" + "="*60)
    print("      ESP32 HAND GESTURE RECOGNITION PROJECT")
    print("="*60)
    print("1. Thu thập dữ liệu (Step 1: Data Collection)")
    print("2. Huấn luyện mô hình (Step 2: Model Training)")
    print("3. Nhận diện thời gian thực (Step 2: Real-time Recognition)")
    print("4. Điều khiển thiết bị (Step 3: Gesture Control)")
    print("5. Test kết nối ESP32")
    print("6. Kiểm tra cài đặt")
    print("7. Thoát")
    print("="*60)

def check_requirements():
    """Kiểm tra các thư viện cần thiết"""
    required_packages = [
        'cv2', 'mediapipe', 'torch', 'torchvision', 
        'numpy', 'pandas', 'yaml', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'mediapipe':
                import mediapipe
            elif package == 'torch':
                import torch
            elif package == 'torchvision':
                import torchvision
            elif package == 'numpy':
                import numpy
            elif package == 'pandas':
                import pandas
            elif package == 'yaml':
                import yaml
            elif package == 'requests':
                import requests
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Thiếu các thư viện sau:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Chạy lệnh sau để cài đặt:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("✅ Tất cả thư viện đã được cài đặt!")
        return True

def run_data_collection():
    """Chạy thu thập dữ liệu"""
    print("\n🔄 Khởi động module thu thập dữ liệu...")
    
    try:
        sys.path.append('./Step_1')
        from data_collector import main as collect_main
        collect_main()
    except ImportError as e:
        print(f"❌ Lỗi import: {e}")
        print("💡 Đảm bảo file Step_1/data_collector.py tồn tại")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def run_training():
    """Chạy huấn luyện mô hình"""
    print("\n🔄 Khởi động module huấn luyện...")
    
    try:
        sys.path.append('./Step_2')
        from gesture_trainer_recognizer import HandGestureTrainer
        
        trainer = HandGestureTrainer()
        train_loader, val_loader = trainer.prepare_data()
        model = trainer.train_model(train_loader, val_loader)
        model_path = trainer.save_model(model)
        print(f"✅ Hoàn thành training! Model lưu tại: {model_path}")
        
    except FileNotFoundError:
        print("❌ Không tìm thấy dữ liệu training!")
        print("💡 Hãy chạy Step 1 (Thu thập dữ liệu) trước")
    except ImportError as e:
        print(f"❌ Lỗi import: {e}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def run_recognition():
    """Chạy nhận diện thời gian thực (Step 2)"""
    print("\n🔄 Khởi động module nhận diện...")
    
    try:
        sys.path.append('./Step_2')
        from gesture_trainer_recognizer import GestureRecognizer
        
        # Tìm file model mới nhất
        models_dir = "./Step_2/models"
        if not os.path.exists(models_dir):
            print("❌ Không tìm thấy thư mục models!")
            print("💡 Hãy chạy training trước")
            return
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if not model_files:
            print("❌ Không tìm thấy file model!")
            print("💡 Hãy chạy training trước")
            return
        
        # Sử dụng model mới nhất
        latest_model = max(model_files)
        model_path = os.path.join(models_dir, latest_model)
        
        print(f"📁 Sử dụng model: {latest_model}")
        esp32_ip = input("🌐 Nhập IP của ESP32 (mặc định: 192.168.1.100): ") or "192.168.1.100"
        
        recognizer = GestureRecognizer(model_path, esp32_ip)
        recognizer.run_recognition()
        
    except ImportError as e:
        print(f"❌ Lỗi import: {e}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")


def run_gesture_control():
    """Chạy điều khiển thiết bị (Step 3)"""
    print("\n🎮 Điều khiển thiết bị bằng cử chỉ tay (Step 3)...")
    
    try:
        sys.path.append('./Step_3')
        
        # Import hàm tìm model từ các module
        try:
            from http_esp32 import find_latest_model
        except ImportError:
            from gesture_control import find_latest_model
        
        # Tìm model tự động
        latest_model = find_latest_model()
        
        if latest_model:
            print(f"📁 Tìm thấy model mới nhất: {os.path.basename(latest_model)}")
            use_latest = input("Sử dụng model này? (y/n, mặc định: y): ").lower()
            
            if use_latest in ['', 'y', 'yes']:
                model_path = latest_model
            else:
                model_path = input("Nhập đường dẫn đến file model (.pth): ")
        else:
            print("❌ Không tìm thấy model trong ./Step_2/models/")
            model_path = input("Nhập đường dẫn đến file model (.pth): ")
        
        if not os.path.exists(model_path):
            print(f"❌ Không tìm thấy file model: {model_path}")
            return
        
        # Chọn loại điều khiển
        print("\n📋 Chọn loại thiết bị điều khiển:")
        print("1. ESP32 qua HTTP")
        print("2. Relay qua Modbus RTU")
        print("3. Chế độ kết hợp (phiên bản cũ)")
        
        device_choice = input("👉 Chọn thiết bị (1-3): ").strip()
        
        if device_choice == "1":
            # Điều khiển ESP32 qua HTTP
            from http_esp32 import ESP32GestureControl
            
            esp32_ip = input("🌐 Nhập IP của ESP32 (mặc định: 192.168.1.100): ") or "192.168.1.100"
            
            # Khởi tạo và chạy controller
            controller = ESP32GestureControl(model_path=model_path, esp32_ip=esp32_ip)
            controller.run()
            
        elif device_choice == "2":
            # Điều khiển Relay qua Modbus
            from relay_controller import RelayGestureControl
            
            # Khởi tạo và chạy controller
            controller = RelayGestureControl(model_path=model_path)
            controller.run()
            
        elif device_choice == "3":
            # Chế độ kết hợp (phiên bản cũ)
            from gesture_control import GestureController
            
            esp32_ip = input("🌐 Nhập IP của ESP32 (mặc định: 192.168.1.100): ") or "192.168.1.100"
            
            # Khởi tạo và chạy controller
            controller = GestureController(model_path, esp32_ip)
            controller.run_control()
            
        else:
            print("❌ Lựa chọn không hợp lệ!")
        
    except ImportError as e:
        print(f"❌ Lỗi import: {e}")
        print("💡 Đảm bảo các file trong Step_3 tồn tại")
    except Exception as e:
        print(f"❌ Lỗi: {e}")


def run_esp32_test():
    """Test kết nối ESP32"""
    print("\n🔧 Khởi động test ESP32...")
    
    try:
        from test_esp32_connection import main as test_main
        test_main()
    except ImportError as e:
        print(f"❌ Lỗi import: {e}")
        print("💡 Đảm bảo file test_esp32_connection.py tồn tại")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def main():
    """Hàm main"""
    while True:
        print_menu()
        choice = input("👉 Chọn chức năng (1-7): ").strip()
        
        if choice == "1":
            run_data_collection()
        elif choice == "2":
            run_training()
        elif choice == "3":
            run_recognition()
        elif choice == "4":
            run_gesture_control()
        elif choice == "5":
            run_esp32_test()
        elif choice == "6":
            print("\n🔍 Kiểm tra cài đặt...")
            check_requirements()
        elif choice == "7":
            print("\n👋 Tạm biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")
        
        input("\n⏸️  Nhấn Enter để tiếp tục...")

if __name__ == "__main__":
    main()
