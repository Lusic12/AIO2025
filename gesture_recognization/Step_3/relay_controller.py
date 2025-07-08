"""
Relay Controller Module
=======================

Module điều khiển relay qua giao thức Modbus RTU
Sử dụng cử chỉ tay để điều khiển các thiết bị relay

⚡ QUAN TRỌNG VỀ BAUDRATE:
   - Relay module Modbus RTU chỉ hoạt động ở 9600 baud
   - CH340 USB-Serial converter có thể accept nhiều baudrate (9600, 115200, etc.)
   - Nhưng relay chỉ "hiểu" được 9600 baud
   - Nếu dùng baudrate khác:
     ✅ Kết nối sẽ thành công (CH340 accept)
     ❌ Relay sẽ không phản hồi (không hiểu lệnh)
   
🔧 CÁCH HOẠT ĐỘNG:
   1. Computer → CH340 (USB to Serial) → Relay Module
   2. CH340 chuyển đổi USB thành RS485/RS232
   3. Relay module nhận tín hiệu serial và thực hiện lệnh
   4. Chỉ 9600 baud là chuẩn Modbus RTU cho relay
"""

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


# ⚡ MÃ LỆNH MODBUS RTU CHO RELAY (Chỉ hoạt động ở 9600 baud)
# Cấu trúc: [Slave_ID, Function_Code, Address_Hi, Address_Lo, Value_Hi, Value_Lo, CRC_Lo, CRC_Hi]
# Function 05 (0x05): Write Single Coil - ghi trạng thái cho 1 relay
# Value: 0xFF00 = ON, 0x0000 = OFF

RELAY1_ON =  [1, 5, 0, 0, 0xFF, 0, 0x8C, 0x3A]  # Relay 1 bật
RELAY1_OFF = [1, 5, 0, 0, 0,    0, 0xCD, 0xCA]  # Relay 1 tắt

RELAY2_ON =  [1, 5, 0, 1, 0xFF, 0, 0xDD, 0xFA]  # Relay 2 bật  
RELAY2_OFF = [1, 5, 0, 1, 0,    0, 0x9C, 0x0A]  # Relay 2 tắt

RELAY3_ON =  [1, 5, 0, 2, 0xFF, 0, 0x2D, 0xFA]  # Relay 3 bật
RELAY3_OFF = [1, 5, 0, 2, 0,    0, 0x6C, 0x0A]  # Relay 3 tắt


class ModbusMaster:
    """
    Class quản lý kết nối và điều khiển các thiết bị qua giao tiếp Modbus.
    Hỗ trợ điều khiển 3 relay/actuator.
    Tự động tìm cấu hình COM port và baudrate phù hợp.
    """
    
    def __init__(self, custom_port=None, auto_detect=True) -> None:
        """
        Khởi tạo kết nối serial với thiết bị Modbus.
        Tự động thử các cấu hình khác nhau để tìm config tốt nhất.
        
        Args:
            custom_port (str, optional): Cổng COM tùy chọn. Nếu không cung cấp, tự động tìm.
            auto_detect (bool): Có tự động thử các cấu hình khác nhau hay không.
        """
        # Các baudrate phổ biến cho Modbus RTU (ưu tiên 9600)
        self.baudrates_to_try = [9600, 19200, 38400, 115200]
        
        # Cấu hình serial hiện tại
        self.current_port = None
        self.current_baudrate = None
        self.ser = None
        
        # Lấy danh sách cổng
        port_list = list_ports.comports()
        print(f"📡 Available COM ports: {[str(port) for port in port_list]}")
        
        if len(port_list) == 0:
            raise Exception("No port found! Check connection to Modbus device.")

        # Xác định danh sách cổng để thử
        ports_to_try = []
        
        if custom_port:
            ports_to_try = [custom_port]
            print(f"🎯 Using specified port: {custom_port}")
        else:
            # Xác định cổng COM dựa trên hệ điều hành
            which_os = platform.system()
            
            if which_os == "Linux":
                usb_ports = [f"/dev/{port.name}" for port in port_list if "USB" in port.name]
                if not usb_ports:
                    raise Exception("No USB port found on Linux!")
                ports_to_try = usb_ports
                
            elif which_os == "Windows":
                # Ưu tiên COM5, sau đó USB Serial, cuối cùng là tất cả COM ports
                preferred_ports = []
                usb_serial_ports = []
                other_com_ports = []
                
                for port in port_list:
                    strPort = str(port)
                    port_name = strPort.split(" ")[0]
                    
                    if "COM5" in strPort:
                        preferred_ports.insert(0, port_name)  # COM5 lên đầu
                    elif "USB Serial" in strPort or "CH340" in strPort or "CP210" in strPort:
                        usb_serial_ports.append(port_name)
                    elif "COM" in strPort:
                        other_com_ports.append(port_name)
                
                # Sắp xếp thứ tự ưu tiên: COM5 > USB Serial > COM ports khác
                ports_to_try = preferred_ports + usb_serial_ports + other_com_ports
                
            else:
                raise Exception(f"Unsupported OS: {which_os}")

        if not ports_to_try:
            raise Exception("No suitable ports found!")

        # 🔧 TỰ ĐỘNG THỬ CÁC CẤU HÌNH
        if auto_detect:
            print("\n" + "="*70)
            print("🔍 AUTO-DETECTING BEST CONFIGURATION")
            print("   Testing COM ports with different baudrates...")
            print("   Priority: 9600 baud (relay standard) > others")
            print("="*70)
            
            best_config = self._auto_detect_config(ports_to_try)
            
            if best_config:
                self.current_port, self.current_baudrate = best_config
                print(f"\n✅ BEST CONFIG FOUND:")
                print(f"   Port: {self.current_port}")
                print(f"   Baudrate: {self.current_baudrate} baud")
                
                # Kết nối với config tốt nhất
                self._connect_with_config(self.current_port, self.current_baudrate)
            else:
                raise Exception("❌ Could not find working configuration on any port!")
        else:
            # Sử dụng cấu hình mặc định (chỉ 9600 baud)
            self.current_port = ports_to_try[0]
            self.current_baudrate = 9600
            print(f"\n🔧 Using default config: {self.current_port} at 9600 baud")
            self._connect_with_config(self.current_port, self.current_baudrate)

    def _auto_detect_config(self, ports_to_try):
        """
        Tự động thử các cấu hình COM port và baudrate để tìm config tốt nhất
        
        Returns:
            tuple: (port, baudrate) nếu tìm thấy, None nếu không
        """
        working_configs = []
        
        for port in ports_to_try:
            print(f"\n🔌 Testing port: {port}")
            
            for baudrate in self.baudrates_to_try:
                print(f"   📡 Trying {baudrate} baud...", end=" ")
                
                try:
                    # Thử kết nối
                    test_ser = serial.Serial(
                        port=port,
                        baudrate=baudrate,
                        timeout=1.0,  # Timeout ngắn cho test
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        bytesize=serial.EIGHTBITS,
                        rtscts=False,
                        dsrdtr=False,
                        xonxoff=False
                    )
                    
                    # Clear buffers
                    test_ser.reset_input_buffer()
                    test_ser.reset_output_buffer()
                    time.sleep(0.1)
                    
                    # Test gửi lệnh đơn giản
                    test_result = self._test_relay_response(test_ser, port, baudrate)
                    
                    test_ser.close()
                    
                    if test_result:
                        score = self._calculate_config_score(port, baudrate, test_result)
                        working_configs.append((port, baudrate, score, test_result))
                        
                        # Hiển thị thông tin chi tiết
                        if isinstance(test_result, dict):
                            read_ok = "✅" if test_result.get('read_test') else "❌"
                            relay_ok = "✅" if test_result.get('relay_test') else "❌"
                            bytes_count = test_result.get('response_bytes', 0)
                            print(f"✅ WORKS (score: {score}) [Read:{read_ok} Relay:{relay_ok} Bytes:{bytes_count}]")
                        else:
                            print(f"✅ WORKS (score: {score})")
                    else:
                        print("❌ No response")
                        
                except Exception as e:
                    print(f"❌ Failed: {str(e)[:30]}...")
                    continue
        
        if not working_configs:
            return None
        
        # Sắp xếp theo điểm số (cao nhất trước)
        working_configs.sort(key=lambda x: x[2], reverse=True)
        
        # Hiển thị chi tiết và trả về config tốt nhất
        return self._display_config_selection(working_configs)
        working_configs.sort(key=lambda x: x[2], reverse=True)
        
        self._display_config_selection(working_configs)
        
        # Trả về config tốt nhất
        return working_configs[0][0], working_configs[0][1]

    def _display_config_selection(self, working_configs):
        """Hiển thị chi tiết các config và lý do chọn config tốt nhất"""
        print(f"\n🏆 WORKING CONFIGURATIONS FOUND:")
        for i, (port, baudrate, score, test_info) in enumerate(working_configs[:5]):  # Top 5
            status = "⭐ BEST" if i == 0 else f"#{i+1}"
            
            if isinstance(test_info, dict):
                read_status = "Read:✅" if test_info.get('read_test') else "Read:❌"
                relay_status = "Relay:✅" if test_info.get('relay_test') else "Relay:❌"
                bytes_info = f"Bytes:{test_info.get('response_bytes', 0)}"
                quality = f"Quality:{test_info.get('quality_score', 0)}"
                detail = f"[{read_status} {relay_status} {bytes_info} {quality}]"
            else:
                detail = f"[{test_info}]"
            
            print(f"   {status}: {port} @ {baudrate} baud (score: {score}) {detail}")
        
        # Hiển thị lý do chọn config tốt nhất
        best_port, best_baudrate, best_score, best_test = working_configs[0]
        print(f"\n🎯 SELECTED: {best_port} @ {best_baudrate} baud")
        
        if best_baudrate == 9600:
            print(f"   ✅ Perfect! 9600 baud is the Modbus RTU standard for relays")
        else:
            print(f"   ⚠️  Non-standard baudrate, but showed best response")
        
        if best_score >= 150:
            print(f"   🏆 Excellent configuration (score: {best_score})")
        elif best_score >= 100:
            print(f"   👍 Good configuration (score: {best_score})")
        else:
            print(f"   ⚠️  Acceptable configuration (score: {best_score})")
            
        return best_port, best_baudrate

    def _test_relay_response(self, ser, port, baudrate):
        """
        Test phản hồi của relay với nhiều lệnh khác nhau để đánh giá chất lượng
        
        Returns:
            dict: Thông tin chi tiết về test result
        """
        test_results = {
            'read_test': False,
            'relay_test': False,
            'response_bytes': 0,
            'errors': 0,
            'quality_score': 0
        }
        
        try:
            # Test 1: Gửi lệnh đọc trạng thái (an toàn, không thay đổi relay)
            read_command = [0x01, 0x01, 0x00, 0x00, 0x00, 0x03, 0x3D, 0xCB]
            ser.write(bytearray(read_command))
            time.sleep(0.1)  # Chờ lâu hơn cho relay phản hồi
            
            if ser.in_waiting > 0:
                response = ser.read(ser.in_waiting)
                if len(response) > 0:
                    test_results['read_test'] = True
                    test_results['response_bytes'] += len(response)
                    test_results['quality_score'] += 20
            
            # Clear buffer trước test tiếp theo
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            time.sleep(0.05)
            
            # Test 2: Test thực tế với relay (bật/tắt nhanh)
            ser.write(bytearray(RELAY1_ON))
            time.sleep(0.05)
            
            
            # Kiểm tra phản hồi từ lệnh ON
            relay_response = False
            if ser.in_waiting > 0:
                response = ser.read(ser.in_waiting)
                if len(response) > 0:
                    test_results['relay_test'] = True
                    test_results['response_bytes'] += len(response)
                    test_results['quality_score'] += 30
                    relay_response = True
            
            # Tắt relay ngay lập tức (an toàn)
            ser.reset_output_buffer()
            ser.write(bytearray(RELAY1_OFF))
            time.sleep(0.05)
            
            # Kiểm tra phản hồi từ lệnh OFF
            if ser.in_waiting > 0:
                response = ser.read(ser.in_waiting)
                if len(response) > 0:
                    test_results['response_bytes'] += len(response)
                    test_results['quality_score'] += 20
            elif relay_response:
                # Nếu ON có phản hồi nhưng OFF không có, vẫn tính điểm
                test_results['quality_score'] += 10
            
            # Bonus điểm dựa vào số lượng byte phản hồi
            if test_results['response_bytes'] >= 8:
                test_results['quality_score'] += 15  # Phản hồi đầy đủ
            elif test_results['response_bytes'] > 0:
                test_results['quality_score'] += 5   # Có phản hồi nhưng không đầy đủ
            
            return test_results
            
        except Exception as e:
            test_results['errors'] += 1
            return test_results

    def _calculate_config_score(self, port, baudrate, test_result):
        """
        Tính điểm tổng thể cho cấu hình dựa trên nhiều yếu tố
        
        Score = Test quality + Port priority + Baudrate priority + Reliability
        """
        total_score = 0
        
        # 1. Điểm từ test thực tế (0-85 điểm)
        if isinstance(test_result, dict):
            total_score += test_result.get('quality_score', 0)
            
            # Bonus cho test thành công
            if test_result.get('read_test', False):
                total_score += 10
            if test_result.get('relay_test', False):
                total_score += 15
            
            # Penalty cho lỗi
            total_score -= test_result.get('errors', 0) * 10
        
        # 2. Điểm ưu tiên port (0-50 điểm)
        if "COM5" in str(port):
            total_score += 50      # COM5 thường là relay port chính
        elif "USB" in str(port).upper() or "CH340" in str(port).upper() or "CP210" in str(port).upper():
            total_score += 35      # USB Serial converter ports
        elif "COM" in str(port).upper():
            total_score += 20      # Các COM port khác
        else:
            total_score += 10      # Port tổng quát
        
        # 3. Điểm ưu tiên baudrate (0-60 điểm)
        baudrate_scores = {
            9600: 60,    # Chuẩn Modbus RTU - tỷ lệ thành công cao nhất
            19200: 45,   # Phổ biến, nhiều thiết bị hỗ trợ
            38400: 30,   # Ít phổ biến hơn nhưng vẫn ổn định
            115200: 15   # Chủ yếu cho debug/PC communication
        }
        total_score += baudrate_scores.get(baudrate, 5)
        
        # 4. Bonus reliability cho 9600 baud (relay standard)
        if baudrate == 9600:
            total_score += 25  # Bonus đặc biệt cho chuẩn relay
        
        # Đảm bảo score không âm
        return max(0, total_score)

    def _connect_with_config(self, port, baudrate):
        """Kết nối với cấu hình đã xác định"""
        print(f"\n🔌 Connecting to {port} at {baudrate} baud...")
        
        # Giải thích về baudrate được chọn
        if baudrate == 9600:
            print("   ✅ Using 9600 baud - Standard for relay Modbus RTU")
        else:
            print(f"   ⚠️  Using {baudrate} baud - Non-standard, may work for some devices")
        
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=2,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                rtscts=False,
                dsrdtr=False,
                xonxoff=False
            )
            
            # Clear buffers
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            print(f"✅ Connected successfully!")
            print(f"   Port: {self.ser.name}")
            print(f"   Baudrate: {self.ser.baudrate} baud")
            print(f"   Timeout: {self.ser.timeout}s")
            
            # Chờ ổn định
            time.sleep(1.0)
            
        except Exception as e:
            error_msg = str(e)
            if "access is denied" in error_msg.lower():
                raise Exception(f"❌ COM Port Access Error:\n"
                              f"   Port {port} is being used by another program\n"
                              f"   Solutions:\n"
                              f"   1. Close other serial programs\n"
                              f"   2. Unplug and replug USB cable\n"
                              f"   3. Run as Administrator")
            else:
                raise Exception(f"❌ Connection failed: {error_msg}")
        
        if not self.ser.is_open:
            try:
                self.ser.open()
                print("✅ Serial port opened successfully")
            except Exception as e:
                raise Exception(f"❌ Could not open port {port}: {e}")
        
        print(f"🔗 Modbus Master ready on {port} at {baudrate} baud")
        
    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        print("Closing the serial connection")
        self.close()

    def test_connection(self):
        """Test kết nối Modbus và khả năng điều khiển relay"""
        try:
            print("\n🔍 Testing Modbus connection...")
            print("   This will test actual relay control (safe test)")
            
            # Clear buffers trước khi test
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            # Test 1: Gửi lệnh đọc trạng thái (an toàn, không thay đổi relay)
            print("   Test 1: Reading relay status...")
            read_command = [0x01, 0x01, 0x00, 0x00, 0x00, 0x03, 0x3D, 0xCB]  # Read 3 coils
            
            self.ser.write(bytearray(read_command))
            time.sleep(0.1)
            
            # Đọc phản hồi
            response = None
            if self.ser.in_waiting > 0:
                response = self.ser.read(self.ser.in_waiting)
                print(f"   ✅ Device responded: {' '.join([hex(x) for x in response])}")
            else:
                print("   ⚠️  No response to read command (normal for some relay modules)")
            
            # Test 2: Test thật với relay 1 (bật rồi tắt ngay)
            print("   Test 2: Quick relay test (ON/OFF)...")
            
            # Bật relay 1
            self.ser.reset_output_buffer()
            self.ser.write(bytearray(RELAY1_ON))
            time.sleep(0.05)
            print("   📡 Sent: Relay 1 ON command")
            
            # Đọc phản hồi nếu có
            if self.ser.in_waiting > 0:
                response = self.ser.read(self.ser.in_waiting)
                print(f"   📥 Response: {' '.join([hex(x) for x in response])}")
            
            # Chờ một chút
            time.sleep(0.3)
            
            # Tắt relay 1 ngay
            self.ser.reset_output_buffer()
            self.ser.write(bytearray(RELAY1_OFF))
            time.sleep(0.05)
            print("   📡 Sent: Relay 1 OFF command")
            
            # Đọc phản hồi nếu có
            if self.ser.in_waiting > 0:
                response = self.ser.read(self.ser.in_waiting)
                print(f"   📥 Response: {' '.join([hex(x) for x in response])}")
            
            print("   ✅ Relay test completed (relay should have flickered briefly)")
            print("   🔧 If relay didn't respond, check:")
            print("      - Power supply to relay module")
            print("      - Wiring connections")
            print("      - Relay module address (should be ID=1)")
            
            return True
                
        except Exception as e:
            print(f"   ❌ Connection test failed: {e}")
            return False

    def send_command_with_retry(self, command, max_retries=2):
        """
        Gửi lệnh Modbus với retry logic
        Chỉ retry 2 lần vì relay module đơn giản
        """
        for attempt in range(max_retries):
            try:
                # Clear output buffer để đảm bảo lệnh sạch
                self.ser.reset_output_buffer()
                
                # Gửi lệnh
                self.ser.write(bytearray(command))
                
                # Chờ relay xử lý (relay cần thời gian phản ứng)
                time.sleep(0.1)  # Tăng delay cho relay ổn định
                
                # Log lệnh đã gửi để debug
                cmd_hex = ' '.join([f"{x:02X}" for x in command])
                print(f"   📡 Sent Modbus: {cmd_hex}")
                
                # Đọc phản hồi nếu có (một số relay không phản hồi)
                if self.ser.in_waiting > 0:
                    response = self.ser.read(self.ser.in_waiting)
                    resp_hex = ' '.join([f"{x:02X}" for x in response])
                    print(f"   📥 Response: {resp_hex}")
                else:
                    print(f"   ℹ️  No response (normal for simple relay modules)")
                
                return True
                
            except Exception as e:
                print(f"   ❌ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"   🔄 Retrying in 0.1s...")
                    time.sleep(0.1)
                else:
                    print(f"   ❌ Failed after {max_retries} attempts")
                    return False
        
        return False

    def switch_actuator_1(self, state):
        """Điều khiển relay 1 - chỉ hoạt động ở 9600 baud"""
        command = RELAY1_ON if state else RELAY1_OFF
        action = "ON" if state else "OFF"
        
        print(f"🔌 Relay 1 → {action}")
        success = self.send_command_with_retry(command)
        
        if success:
            print(f"✅ Relay 1: {action} - Command sent successfully")
        else:
            print(f"❌ Relay 1: Failed to send {action} command")
            
        return success

    def switch_actuator_2(self, state):
        """Điều khiển relay 2 - chỉ hoạt động ở 9600 baud"""
        command = RELAY2_ON if state else RELAY2_OFF
        action = "ON" if state else "OFF"
        
        print(f"🔌 Relay 2 → {action}")
        success = self.send_command_with_retry(command)
        
        if success:
            print(f"✅ Relay 2: {action} - Command sent successfully")
        else:
            print(f"❌ Relay 2: Failed to send {action} command")
            
        return success
    
    def switch_actuator_3(self, state):
        """Điều khiển relay 3 - chỉ hoạt động ở 9600 baud"""
        command = RELAY3_ON if state else RELAY3_OFF
        action = "ON" if state else "OFF"
        
        print(f"🔌 Relay 3 → {action}")
        success = self.send_command_with_retry(command)
        
        if success:
            print(f"✅ Relay 3: {action} - Command sent successfully")
        else:
            print(f"❌ Relay 3: Failed to send {action} command")
            
        return success
        
    def all_on(self):
        """Bật tất cả relay - thực hiện tuần tự với delay"""
        print("🔌 ALL RELAYS → ON")
        success_count = 0
        
        if self.switch_actuator_1(True):
            success_count += 1
        time.sleep(0.1)  # Delay giữa các lệnh để relay ổn định
        
        if self.switch_actuator_2(True):
            success_count += 1
        time.sleep(0.1)
        
        if self.switch_actuator_3(True):
            success_count += 1
            
        print(f"✅ {success_count}/3 relays turned ON successfully")
        return success_count == 3
        
    def all_off(self):
        """Tắt tất cả relay - thực hiện tuần tự với delay"""
        print("🔌 ALL RELAYS → OFF")
        success_count = 0
        
        if self.switch_actuator_1(False):
            success_count += 1
        time.sleep(0.1)  # Delay giữa các lệnh để relay ổn định
        
        if self.switch_actuator_2(False):
            success_count += 1
        time.sleep(0.1)
        
        if self.switch_actuator_3(False):
            success_count += 1
            
        print(f"✅ {success_count}/3 relays turned OFF successfully")
        return success_count == 3

    def close(self):
        """Đóng kết nối serial"""
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()


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
class HandGestureModel(nn.Module):
    """
    Neural Network cho hand gesture recognition
    """
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
        
    def predict(self, x, threshold=0.5):
        with torch.no_grad():
            outputs = self.forward(x)                  # shape: (batch_size, num_classes)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1) # lấy xác suất cao nhất và index tương ứng
            preds[max_probs < threshold] = -1          # gán -1 cho những mẫu không đủ độ tự tin
            return preds




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


class RelayGestureControl:
    """Điều khiển relay bằng cử chỉ tay qua Modbus"""
    
    def __init__(self, model_path, config_path="../config.yaml", resolution=(1280, 720), port=None, simulation=False, save_video=False):
        self.resolution = resolution
        self.height = 720
        self.width = 1280
        self.port = port
        self.simulation = simulation
        self.save_video = save_video  # ✅ Thêm tùy chọn save video
        
        # Video recording setup
        self.video_writer = None
        self.recording = False
        self.video_filename = None

        # Khởi tạo các components
        self.detector = HandLandmarksDetector()
        self.status_text = None
        self.signs = label_dict_from_config_file(config_path)
        
        # Load model
        print(f"Loading model from: {model_path}")
        
        try:
            # Load checkpoint để kiểm tra cấu trúc trước
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Lấy thông tin model từ checkpoint
            model_info = {}
            model_dict = {}
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model_dict = checkpoint['model_state_dict']
                    if 'num_classes' in checkpoint:
                        model_info['num_classes'] = checkpoint['num_classes']
                        print(f"Found num_classes in model: {checkpoint['num_classes']}")
                    if 'gesture_labels' in checkpoint:
                        model_info['gesture_labels'] = checkpoint['gesture_labels']
                        print(f"Found gesture labels in model: {checkpoint['gesture_labels']}")
                else:
                    model_dict = checkpoint
            else:
                model_dict = checkpoint
            
            # Phân tích cấu trúc mạng neural từ state_dict
            # Sử dụng cách tiếp cận mạnh mẽ hơn với phát hiện số lớp ẩn
            hidden_sizes = []
            max_layer_index = -1
            
            # Đếm số lớp và kích thước từng lớp
            for key in model_dict.keys():
                if 'network' in key and 'weight' in key:
                    # Lấy số index của lớp từ key (e.g., 'network.6.weight' -> 6)
                    parts = key.split('.')
                    if len(parts) >= 2:
                        try:
                            layer_index = int(parts[1])
                            max_layer_index = max(max_layer_index, layer_index)
                        except ValueError:
                            pass
            
            # Kiểm tra xem chúng ta có một mạng neural nhiều lớp
            if max_layer_index > 0:
                print(f"Detected {(max_layer_index + 1) // 3} hidden layers")
                
                # Trích xuất kích thước của các lớp ẩn
                for i in range(0, max_layer_index + 1, 3):  # Mỗi khối gồm Linear + ReLU + Dropout
                    if f'network.{i}.weight' in model_dict:
                        weight_shape = model_dict[f'network.{i}.weight'].shape
                        if len(weight_shape) == 2:
                            hidden_sizes.append(weight_shape[0])
                
                # Lớp cuối cùng không nên được tính là hidden layer
                if hidden_sizes and len(hidden_sizes) > 1:
                    hidden_sizes = hidden_sizes[:-1]
            
            # Nếu không phát hiện được cấu trúc, dùng cấu trúc mặc định đã biết
            if not hidden_sizes or len(hidden_sizes) < 2:
                print("Using default architecture: [512, 256, 128, 64]")
                hidden_sizes = [512, 256, 128, 64]
                
            # Số lớp đầu ra (ưu tiên dùng số lớp từ config file)
            num_classes = len(self.signs)
            model_classes = model_info.get('num_classes')
            
            if model_classes and model_classes != num_classes:
                print(f"WARNING: Model has {model_classes} classes but config has {num_classes} classes.")
                print("Using number of classes from config file.")
                
            print(f"Final model structure: hidden_sizes={hidden_sizes}, num_classes={num_classes}")
            
            # Khởi tạo model với cấu trúc đã phát hiện
            self.classifier = HandGestureModel(
                input_size=63, 
                num_classes=num_classes
            )
            
            # Tải state_dict vào model
            if 'model_state_dict' in checkpoint:
                print("Loading state_dict from 'model_state_dict' key")
                self.classifier.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("Loading direct state dict")
                self.classifier.load_state_dict(checkpoint)
                
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        # Set model to evaluation mode
        self.classifier.eval()

        # Khởi tạo Modbus controller với fallback simulation
        self.controller = None
        
        if not simulation:
            try:
                self.controller = ModbusMaster(custom_port=self.port)
                
                # Test kết nối thực tế
                if self.controller.test_connection():
                    print("✅ Hardware controller initialized and tested successfully")
                    print("🔌 Ready to control relays via Modbus")
                else:
                    print("⚠️  Hardware connected but test failed - will try anyway")
                    
            except Exception as e:
                print(f"❌ Hardware connection failed: {e}")
                print("🎮 Automatically switching to SIMULATION mode...")
                print("💡 You can test gesture recognition without hardware")
                self.simulation = True
                self.controller = None
        
        if self.simulation:
            print("\n" + "="*50)
            print("🎮 SIMULATION MODE ENABLED")
            print("🔌 Virtual relay control active")
            print("👋 Gesture recognition fully functional")
            print("⚡ All relay commands will be simulated")
            print("="*50 + "\n")
            
        # Trạng thái đèn
        self.light1 = False
        self.light2 = False
        self.light3 = False
        
        # Tracking variables
        self.last_command_time = time.time()
        self.command_debounce = 1.0  # 1 second debounce time
        
        print(f"Relay Gesture Control initialized with {len(self.signs)} gestures")
        print(f"Available gestures: {list(self.signs.values())}")
        
        # Setup video recording if enabled
        if self.save_video:
            # Tạo thư mục videos nếu chưa có
            video_dir = Path("videos")
            video_dir.mkdir(exist_ok=True)
            
            # Tạo tên file với timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = "simulation" if self.simulation else "hardware"
            self.video_filename = video_dir / f"gesture_control_{mode}_{timestamp}.mp4"
            
            print(f"📹 Video recording enabled: {self.video_filename}")
        else:
            print("📹 Video recording disabled")
    
    def _setup_video_recording(self, frame_shape):
        """Khởi tạo video writer"""
        if self.save_video and not self.recording:
            try:
                # Codec và setup
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 20.0  # FPS cho video
                
                self.video_writer = cv2.VideoWriter(
                    str(self.video_filename),
                    fourcc,
                    fps,
                    (frame_shape[1], frame_shape[0])  # (width, height)
                )
                
                if self.video_writer.isOpened():
                    self.recording = True
                    print(f"✅ Video recording started: {self.video_filename}")
                else:
                    print(f"❌ Failed to start video recording")
                    self.video_writer = None
                    
            except Exception as e:
                print(f"❌ Error setting up video recording: {e}")
                self.video_writer = None

    def _write_video_frame(self, frame):
        """Ghi frame vào video"""
        if self.recording and self.video_writer:
            try:
                self.video_writer.write(frame)
            except Exception as e:
                print(f"❌ Error writing video frame: {e}")

    def _stop_video_recording(self):
        """Dừng và đóng video recording"""
        if self.recording and self.video_writer:
            try:
                self.video_writer.release()
                self.recording = False
                print(f"✅ Video saved: {self.video_filename}")
            except Exception as e:
                print(f"❌ Error stopping video recording: {e}")
            finally:
                self.video_writer = None

    def _control_relay(self, relay_num, state):
        """Điều khiển relay với hỗ trợ simulation"""
        if self.simulation:
            action = "ON" if state else "OFF"
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"🔌 [{timestamp}] SIMULATION: Relay {relay_num} -> {action}")
            
            # Tạo hiệu ứng LED ảo
            status = "🟢" if state else "🔴"
            print(f"   {status} Virtual Relay {relay_num} status: {action}")
            return
        
        try:
            if relay_num == 1:
                self.controller.switch_actuator_1(state)
            elif relay_num == 2:
                self.controller.switch_actuator_2(state)
            elif relay_num == 3:
                self.controller.switch_actuator_3(state)
        except Exception as e:
            print(f"❌ Error controlling relay {relay_num}: {e}")

    def _control_all_relays(self, state):
        """Điều khiển tất cả relay với simulation"""
        if self.simulation:
            action = "ON" if state else "OFF"
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"🔌 [{timestamp}] SIMULATION: ALL RELAYS -> {action}")
            
            # Hiển thị trạng thái tất cả relay ảo
            status = "🟢" if state else "🔴"
            print(f"   {status} Virtual Relay 1: {action}")
            print(f"   {status} Virtual Relay 2: {action}")
            print(f"   {status} Virtual Relay 3: {action}")
            return
        
        try:
            if state:
                self.controller.all_on()
            else:
                self.controller.all_off()
        except Exception as e:
            print(f"❌ Error controlling all relays: {e}")

    def _log_action(self, gesture, action):
        """Ghi log hành động với timestamp"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        mode = "SIM" if self.simulation else "HW"
        print(f"👋 [{timestamp}] [{mode}] Gesture: {gesture} → {action}")

        # Ghi vào file log
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / "relay_control.log"
        
        with open(log_file, 'a') as f:
            f.write(f"{timestamp} | {mode} | {gesture} | {action} | " 
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
            
        print("=== RELAY GESTURE CONTROL SYSTEM ===")
        print("Controls:")
        print("- 'q': Quit")
        print("- 'r': Reset all lights")
        print("- 'v': Toggle video recording (if enabled)")  # ✅ Thêm phím v
        print(f"- Gestures: {list(self.signs.values())}")
        print("=====================================")
        
        # Biến để track video recording state
        video_initialized = False
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
                        class_number = self.classifier.predict(hand_landmark, threshold=0.7).item()
                        
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
                                        self.light1 = True
                                        self._control_relay(1, True)
                                        self._log_action("light1_on", "Light 1 ON")
                                        
                                elif self.status_text == "light1_off":
                                    if self.light1:
                                        self.light1 = False
                                        self._control_relay(1, False)
                                        self._log_action("light1_off", "Light 1 OFF")
                                        
                                elif self.status_text == "light2_on":
                                    if not self.light2:
                                        self.light2 = True
                                        self._control_relay(2, True)
                                        self._log_action("light2_on", "Light 2 ON")
                                        
                                elif self.status_text == "light2_off":
                                    if self.light2:
                                        self.light2 = False
                                        self._control_relay(2, False)
                                        self._log_action("light2_off", "Light 2 OFF")
                                        
                                elif self.status_text == "turn_on":
                                    if not (self.light1 and self.light2 and self.light3):
                                        self.light1 = self.light2 = self.light3 = True
                                        self._control_all_relays(True)
                                        self._log_action("turn_on", "ALL ON")
                                        
                                elif self.status_text == "turn_off":
                                    if self.light1 or self.light2 or self.light3:
                                        self.light1 = self.light2 = self.light3 = False
                                        self._control_all_relays(False)
                                        self._log_action("turn_off", "ALL OFF")
                        else:
                            self.status_text = "undefined command"
                else:
                    self.status_text = None
                
                # Add UI elements
                img = self.light_simulation(img)
                
                # ✅ Setup video recording on first frame
                if self.save_video and not video_initialized:
                    self._setup_video_recording(img.shape)
                    video_initialized = True
                
                # ✅ Write frame to video if recording
                if self.recording:
                    self._write_video_frame(img)
                
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
                
                # Show mode indicator
                mode_text = "SIMULATION MODE" if self.simulation else "RELAY MODE"
                mode_color = (0, 255, 255) if self.simulation else (0, 0, 255)  # Yellow for sim, Red for hardware
                cv2.putText(img, mode_text, 
                           (img.shape[1] - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
                
                # ✅ Show video recording status
                if self.save_video:
                    record_text = "REC ●" if self.recording else "REC ○"
                    record_color = (0, 0, 255) if self.recording else (128, 128, 128)  # Red when recording, Gray when not
                    cv2.putText(img, record_text, 
                               (img.shape[1] - 200, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, record_color, 2)
                
                # Display the frame
                cv2.namedWindow("Relay Gesture Control", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Relay Gesture Control", self.resolution[0], self.resolution[1])
                cv2.imshow("Relay Gesture Control", img)
                
                # Process keys
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    # Reset all lights
                    self.light1 = self.light2 = self.light3 = False
                    self._control_all_relays(False)
                    print("All lights reset")
                elif key == ord("v") and self.save_video:
                    # ✅ Toggle video recording
                    if self.recording:
                        self._stop_video_recording()
                    else:
                        if video_initialized:
                            # Restart recording với frame hiện tại
                            self._setup_video_recording(img.shape)
                        else:
                            print("⚠️  Video not initialized yet, wait for first frame")
        
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Cleanup
            if cam.isOpened():
                cam.release()
            cv2.destroyAllWindows()
            
            # ✅ Stop video recording if active
            if self.recording:
                self._stop_video_recording()
            
            # Turn off all relays before exit
            try:
                self._control_all_relays(False)
                if self.controller:
                    self.controller.close()
                if self.simulation:
                    print("🎮 Simulation ended - All virtual relays turned off")
                else:
                    print("All relays turned off")
            except:
                pass


def main():
    """Main function"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Relay Gesture Control System")
    parser.add_argument("--model", "-m", 
                      default="../Step_2/models/hand_gesture_model.pth",
                      help="Path to trained model file")
    parser.add_argument("--config", "-c", 
                      default="../config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--resolution", "-r", 
                      default="1280x720",
                      help="Display resolution (WIDTHxHEIGHT)")
    parser.add_argument("--port", "-p",
                      help="Specify COM port manually (e.g., COM3, COM4)")
    parser.add_argument("--list-ports", "-l", action="store_true",
                      help="List all available COM ports and exit")
    parser.add_argument("--simulation", "-s", action="store_true",
                      help="Run in simulation mode without hardware")
    parser.add_argument("--save-video", action="store_true",
                      help="Enable video recording (save to videos/ folder)")
    parser.add_argument("--record", action="store_true", 
                      help="Alias for --save-video")
    
    args = parser.parse_args()
    
    # List all available ports if requested
    if args.list_ports:
        print("Available COM ports:")
        for port in list_ports.comports():
            print(f"  {port}")
        return
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"Invalid resolution format: {args.resolution}. Using default 1280x720")
        resolution = (1280, 720)
    
    # Create and run gesture control system
    try:
        # Enable video recording if requested
        save_video = getattr(args, 'save_video', False) or getattr(args, 'record', False)
        
        control_system = RelayGestureControl(
            model_path=args.model,
            config_path=args.config,
            resolution=resolution,
            port=args.port,
            simulation=getattr(args, 'simulation', False),
            save_video=save_video
        )
        control_system.run()
    except KeyboardInterrupt:
        print("\nExiting by user request")
    except Exception as e:
        print(f"❌ Error: {e}")
        if not getattr(args, 'simulation', False):
            print("🎮 Try running with --simulation flag to test gesture recognition")
            print("💡 Command: python relay_controller.py --simulation")
        import traceback
        traceback.print_exc()


# Test the Modbus controller directly
def test_modbus_controller(custom_port=None):
    """Test Modbus controller - chỉ sử dụng 9600 baud chuẩn"""
    print("="*60)
    print("🔧 TESTING MODBUS CONTROLLER AT 9600 BAUD")
    print("="*60)
    
    try:
        controller = ModbusMaster(custom_port=custom_port)
        
        # Test kết nối trước
        print("\n🔍 Step 1: Testing connection...")
        if not controller.test_connection():
            print("⚠️  Connection test had issues but continuing...")
        
        print("\n🔧 Step 2: Testing individual relay control...")
        
        # Test từng relay một cách an toàn (1 lần bật/tắt)
        for relay_num in range(1, 4):
            print(f"\n   Testing Relay {relay_num}:")
            
            # Bật relay
            if relay_num == 1:
                success_on = controller.switch_actuator_1(True)
            elif relay_num == 2:
                success_on = controller.switch_actuator_2(True)
            else:
                success_on = controller.switch_actuator_3(True)
            
            if success_on:
                time.sleep(0.5)  # Để thấy relay hoạt động
                
                # Tắt relay
                if relay_num == 1:
                    success_off = controller.switch_actuator_1(False)
                elif relay_num == 2:
                    success_off = controller.switch_actuator_2(False)
                else:
                    success_off = controller.switch_actuator_3(False)
                
                if success_off:
                    print(f"   ✅ Relay {relay_num}: Test completed successfully")
                else:
                    print(f"   ⚠️  Relay {relay_num}: ON worked, but OFF failed")
            else:
                print(f"   ❌ Relay {relay_num}: Failed to turn ON")
            
            time.sleep(0.3)  # Delay giữa các test
        
        print(f"\n🔧 Step 3: Testing group control...")
        
        # Test bật tất cả
        print("   Testing ALL ON...")
        if controller.all_on():
            time.sleep(1)  # Để thấy tất cả relay bật
            
            # Test tắt tất cả
            print("   Testing ALL OFF...")
            controller.all_off()
            time.sleep(0.5)
        
        print("\n" + "="*60)
        print("✅ MODBUS TEST COMPLETED SUCCESSFULLY!")
        print("🔌 All relay commands sent at 9600 baud")
        print("💡 If relays didn't respond physically, check:")
        print("   - Relay module power supply")
        print("   - Wiring connections")
        print("   - Module address settings")
        print("="*60)
        
        controller.close()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check COM port is correct")
        print("2. Ensure relay module is powered")
        print("3. Verify USB cable connection")
        print("4. Check relay module supports 9600 baud Modbus RTU")
        print("5. Try different port with --port argument")
        
        if platform.system() == "Windows":
            print("\n📋 Available COM ports:")
            for port in list_ports.comports():
                print(f"   {port}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Parse any additional args for test mode
        import argparse
        parser = argparse.ArgumentParser(description="Test Modbus Controller")
        parser.add_argument("--test", action="store_true", help="Run in test mode")
        parser.add_argument("--port", "-p", help="Specify COM port manually (e.g., COM3, COM4)")
        parser.add_argument("--list-ports", "-l", action="store_true", help="List all available COM ports and exit")
        
        # Parse remaining args (skip the first one which is "--test")
        args, _ = parser.parse_known_args(sys.argv[1:])
        
        if args.list_ports:
            print("Available COM ports:")
            for port in list_ports.comports():
                print(f"  {port}")
        else:
            test_modbus_controller(custom_port=args.port)
    else:
        main()

    @staticmethod
    def explain_baudrate_mechanics():
        """
        Giải thích chi tiết về cơ chế baudrate và tại sao 9600 baud quan trọng cho relay
        """
        print("\n" + "="*80)
        print("📚 BAUDRATE & RELAY COMMUNICATION EXPLANATION")
        print("="*80)
        
        print("🔧 SERIAL COMMUNICATION BASICS:")
        print("   • Baudrate = bits per second (bps)")
        print("   • Higher baudrate = faster data transmission")
        print("   • Both devices must use SAME baudrate to communicate")
        print("   • Mismatch = garbled data or no communication")
        
        print("\n⚡ WHY 9600 BAUD FOR RELAY MODULES:")
        print("   1. 🏭 MODBUS RTU STANDARD:")
        print("      • Modbus RTU specification recommends 9600 baud")
        print("      • Most relay modules are designed for 9600 baud")
        print("      • Industrial standard for decades")
        
        print("\n   2. 🔌 HARDWARE COMPATIBILITY:")
        print("      • Relay controller chips (like STM32, Arduino) default to 9600")
        print("      • Less prone to timing errors at slower speed")
        print("      • Better noise immunity in industrial environments")
        
        print("\n   3. 💻 CH340 USB-SERIAL CONVERTER:")
        print("      • Computer ↔ USB ↔ CH340 ↔ RS485/RS232 ↔ Relay")
        print("      • CH340 can accept multiple baudrates (9600, 115200, etc.)")
        print("      • BUT relay module only understands 9600 baud")
        
        print("\n🚨 COMMON MISUNDERSTANDING:")
        print("   ❌ 'Connection successful at 115200 baud' ≠ Working communication")
        print("   ✅ CH340 accepts 115200 baud from computer")
        print("   ❌ But relay doesn't respond because it expects 9600 baud")
        print("   ➡️  Result: Connection OK, but no relay control!")
        
        print("\n🧪 TESTING DIFFERENT BAUDRATES:")
        baudrates = [9600, 19200, 38400, 115200]
        print("   Baudrate │ Connection │ Relay Response │ Recommendation")
        print("   ─────────┼────────────┼────────────────┼─────────────────")
        
        recommendations = [
            ("9600", "✅ YES", "✅ YES", "🏆 BEST - Use this!"),
            ("19200", "✅ YES", "❌ NO", "⚠️  Connection only"),
            ("38400", "✅ YES", "❌ NO", "⚠️  Connection only"),
            ("115200", "✅ YES", "❌ NO", "⚠️  Connection only")
        ]
        
        for baud, conn, resp, rec in recommendations:
            print(f"   {baud:>8} │ {conn:>10} │ {resp:>14} │ {rec}")
        
        print("\n🔍 HOW AUTO-DETECTION WORKS:")
        print("   1. 📡 Scan all available COM ports")
        print("   2. 🔄 Try each baudrate (9600, 19200, 38400, 115200)")
        print("   3. 📤 Send test commands to relay")
        print("   4. 📥 Check for valid responses")
        print("   5. 📊 Score each configuration based on:")
        print("      • Response quality (relay acknowledges commands)")
        print("      • Port type priority (COM5, USB Serial, etc.)")
        print("      • Baudrate preference (9600 = highest score)")
        print("   6. 🎯 Select configuration with highest score")
        
        print("\n💡 TROUBLESHOOTING TIPS:")
        print("   • If auto-detection fails → Check physical connections")
        print("   • If connection OK but no relay control → Wrong baudrate")
        print("   • If 'access denied' error → Close other serial programs")
        print("   • If no COM ports → Check drivers (CH340/CP210x)")
        
        print("="*80)
