import serial
from serial.tools import list_ports
import time
import sys

def test_multiple_configs(port_name):
    """Test với nhiều cấu hình khác nhau"""
    print(f"Testing {port_name} with multiple configurations...")
    
    configs = [
        # Thử các baudrate phổ biến
        {'baudrate': 115200, 'timeout': 1, 'parity': serial.PARITY_NONE},
        {'baudrate': 9600, 'timeout': 2, 'parity': serial.PARITY_NONE},
        {'baudrate': 19200, 'timeout': 1, 'parity': serial.PARITY_NONE},
        {'baudrate': 38400, 'timeout': 1, 'parity': serial.PARITY_NONE},
        
        # Thử với flow control
        {'baudrate': 9600, 'timeout': 1, 'rtscts': False, 'dsrdtr': False},
        {'baudrate': 9600, 'timeout': 1, 'xonxoff': False},
        
        # Thử với parity khác
        {'baudrate': 9600, 'timeout': 1, 'parity': serial.PARITY_EVEN},
        {'baudrate': 9600, 'timeout': 1, 'parity': serial.PARITY_ODD},
    ]
    
    for i, config in enumerate(configs, 1):
        try:
            print(f"  Config {i}: {config}")
            ser = serial.Serial(port_name, **config)
            print(f"  ✓ SUCCESS with config {i}!")
            
            # Test gửi/nhận data
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            
            # Thử gửi lệnh Modbus relay
            relay_on = b'\x01\x05\x00\x00\xFF\x00\x8C\x3A'
            ser.write(relay_on)
            time.sleep(0.1)
            
            if ser.in_waiting > 0:
                response = ser.read(ser.in_waiting)
                print(f"  Response: {response.hex()}")
            else:
                print("  No response (normal for some devices)")
            
            ser.close()
            return config
            
        except Exception as e:
            print(f"  ✗ Config {i} failed: {str(e)[:50]}...")
            continue
    
    return None

def reset_com_port():
    """Reset COM port qua Device Manager"""
    print("Attempting to reset COM port...")
    try:
        import subprocess
        # Disable và enable lại port
        result = subprocess.run([
            'powershell', '-Command',
            'Get-PnpDevice | Where-Object {$_.FriendlyName -like "*CH340*"} | Disable-PnpDevice -Confirm:$false; Start-Sleep 2; Get-PnpDevice | Where-Object {$_.FriendlyName -like "*CH340*"} | Enable-PnpDevice -Confirm:$false'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Port reset successful")
            time.sleep(3)  # Đợi port khởi tạo lại
            return True
        else:
            print("✗ Port reset failed")
            return False
    except:
        print("Could not reset port (need admin rights)")
        return False

if __name__ == "__main__":
    # Tìm COM port
    ports = list_ports.comports()
    ch340_port = None
    
    for port in ports:
        if "CH340" in port.description:
            ch340_port = port.device
            break
    
    if not ch340_port:
        print("No CH340 device found!")
        sys.exit(1)
    
    print(f"Found CH340 at {ch340_port}")
    
    # Thử reset port trước
    print("\n=== Attempting port reset ===")
    reset_com_port()
    
    # Test với nhiều cấu hình
    print(f"\n=== Testing configurations ===")
    working_config = test_multiple_configs(ch340_port)
    
    if working_config:
        print(f"\n✓ Working configuration found: {working_config}")
        print("You can use this configuration in your main script.")
    else:
        print("\n✗ No working configuration found!")
        print("\nManual troubleshooting steps:")
        print("1. Download latest CH340 driver from: http://www.wch-ic.com/downloads/CH341SER_EXE.html")
        print("2. Uninstall current driver in Device Manager")
        print("3. Install new driver")
        print("4. Reboot computer")
        print("5. Try different USB cable/port")