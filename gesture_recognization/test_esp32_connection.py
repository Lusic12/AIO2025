"""
ESP32 Connection Test Script
============================

Script để test kết nối và gửi lệnh đến ESP32 server.
"""

import requests
import time

def test_esp32_connection(esp32_ip="192.168.1.100", port=80):
    """
    Test kết nối đến ESP32 server
    
    Args:
        esp32_ip: IP address của ESP32
        port: Port của ESP32 server (mặc định 80)
    """
    base_url = f"http://{esp32_ip}:{port}"
    
    print(f"🔍 Testing connection to ESP32 at {base_url}")
    
    try:
        # Test kết nối cơ bản
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ ESP32 server is responding!")
            print(f"📄 Response preview: {response.text[:100]}...")
        else:
            print(f"⚠️ ESP32 responded with status code: {response.status_code}")
            
    except requests.ConnectionError:
        print("❌ Cannot connect to ESP32!")
        print("💡 Check:")
        print("   - ESP32 is powered on")
        print("   - WiFi connection is established")
        print("   - IP address is correct")
        print("   - Computer and ESP32 are on same network")
        return False
    except requests.Timeout:
        print("⏰ Connection timeout!")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def test_led_control(esp32_ip="192.168.1.100", port=80):
    """
    Test điều khiển LED
    
    Args:
        esp32_ip: IP address của ESP32
        port: Port của ESP32 server
    """
    base_url = f"http://{esp32_ip}:{port}"
    
    print(f"\n🔧 Testing LED control...")
    
    # Test các lệnh điều khiển LED
    test_commands = [
        ("LED 1", f"{base_url}/toggle/1"),
        ("LED 2", f"{base_url}/toggle/2"),
    ]
    
    for led_name, url in test_commands:
        try:
            print(f"🔄 Testing {led_name}...")
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                print(f"✅ {led_name} command sent successfully!")
            else:
                print(f"⚠️ {led_name} command failed (status: {response.status_code})")
            
            time.sleep(1)  # Delay giữa các lệnh
            
        except Exception as e:
            print(f"❌ Error testing {led_name}: {e}")

def simulate_gesture_commands(esp32_ip="192.168.1.100"):
    """
    Mô phỏng gửi các lệnh cử chỉ
    """
    print(f"\n🎭 Simulating gesture commands...")
    
    gestures = {
        "turn_off": None,      # Không gửi lệnh
        "light1_on": f"http://{esp32_ip}/toggle/1",
        "light1_off": f"http://{esp32_ip}/toggle/1", 
        "light2_on": f"http://{esp32_ip}/toggle/2",
        "light2_off": f"http://{esp32_ip}/toggle/2",
    }
    
    for gesture_name, url in gestures.items():
        if url:
            try:
                print(f"🤏 Simulating gesture: {gesture_name}")
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    print(f"✅ Gesture '{gesture_name}' executed successfully!")
                else:
                    print(f"⚠️ Gesture '{gesture_name}' failed")
                    
                time.sleep(2)  # Delay giữa các cử chỉ
                
            except Exception as e:
                print(f"❌ Error with gesture '{gesture_name}': {e}")
        else:
            print(f"⏸️ Gesture '{gesture_name}': No action (turn off)")

def main():
    """Hàm main để chạy test"""
    print("="*60)
    print("        ESP32 CONNECTION TEST SCRIPT")
    print("="*60)
    
    # Nhập IP của ESP32
    esp32_ip = input("🌐 Enter ESP32 IP address (default: 192.168.1.100): ").strip()
    if not esp32_ip:
        esp32_ip = "192.168.1.100"
    
    print(f"\n🎯 Target ESP32: {esp32_ip}")
    
    # Test kết nối cơ bản
    if test_esp32_connection(esp32_ip):
        print("\n" + "="*50)
        
        # Menu lựa chọn
        while True:
            print("\nSelect test option:")
            print("1. Test LED control")
            print("2. Simulate gesture commands")
            print("3. Exit")
            
            choice = input("👉 Choose (1-3): ").strip()
            
            if choice == "1":
                test_led_control(esp32_ip)
            elif choice == "2":
                simulate_gesture_commands(esp32_ip)
            elif choice == "3":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice!")
    else:
        print("\n💡 Please check ESP32 connection and try again.")

if __name__ == "__main__":
    main()
