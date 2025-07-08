"""
Controller Module for Relay/Actuator Control
===========================================

Module quản lý kết nối và điều khiển các thiết bị relay/actuator qua giao tiếp Modbus.
"""

import platform
import serial
from serial.tools import list_ports
import time

# Modbus commands with CRC (2 last bytes are CRC)
RELAY1_ON =  [1, 5, 0, 0, 0xFF, 0, 0x8C, 0x3A]
RELAY1_OFF = [1, 5, 0, 0, 0,    0, 0xCD, 0xCA]

RELAY2_ON =  [1, 5, 0, 1, 0xFF, 0, 0xDD, 0xFA]
RELAY2_OFF = [1, 5, 0, 1, 0,    0, 0x9C, 0x0A]

RELAY3_ON =  [1, 5, 0, 2, 0xFF, 0, 0x2D, 0xFA]
RELAY3_OFF = [1, 5, 0, 2, 0,    0, 0x6C, 0x0A]


class ModbusMaster:
    """
    Class quản lý kết nối và điều khiển các thiết bị qua giao tiếp Modbus.
    Hỗ trợ điều khiển 3 relay/actuator.
    """
    
    def __init__(self) -> None:
        """
        Khởi tạo kết nối serial với thiết bị Modbus.
        Tự động tìm cổng USB phù hợp dựa vào hệ điều hành.
        """
        port_list = list_ports.comports()
        if len(port_list) == 0:
            raise Exception("No port found! Check connection to Modbus device.")

        # Xác định cổng COM dựa trên hệ điều hành
        which_os = platform.system()
        if which_os == "Linux":
            name_ports = list(filter(lambda name: "USB" in name, map(lambda port: port.name, port_list)))
            if not name_ports:
                raise Exception("No USB port found on Linux!")
            portName = "/dev/" + name_ports[0]
            print(f"Connected to {portName}")
        elif which_os == "Windows":
            portName = None
            for port in port_list:
                strPort = str(port)
                if "USB Serial" in strPort:
                    splitPort = strPort.split(" ")
                    portName = splitPort[0]
                    break
            if not portName:
                raise Exception("No USB Serial port found on Windows!")
            print(f"Connected to {portName}")
        else:
            raise Exception(f"Unsupported OS: {which_os}")

        # Cấu hình serial
        self.ser = serial.Serial(portName)
        self.ser.baudrate = 9600
        self.ser.stopbits = serial.STOPBITS_ONE
        self.ser.parity = serial.PARITY_NONE
        self.ser.bytesize = serial.EIGHTBITS
        print(f"Serial config: {self.ser.baudrate} baud, {self.ser.bytesize} bits, "
              f"Parity: {self.ser.parity}, Stop bits: {self.ser.stopbits}")
        
    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        print("Closing the serial connection")
        self.close()

    def switch_actuator_1(self, state):
        """Điều khiển relay/actuator 1"""
        if state:
            self.ser.write(bytearray(RELAY1_ON))
        else:
            self.ser.write(bytearray(RELAY1_OFF))
        # Chờ cho relay hoạt động
        time.sleep(0.01)

    def switch_actuator_2(self, state):
        """Điều khiển relay/actuator 2"""
        if state:
            self.ser.write(bytearray(RELAY2_ON))
        else:
            self.ser.write(bytearray(RELAY2_OFF))
        # Chờ cho relay hoạt động
        time.sleep(0.01)
    
    def switch_actuator_3(self, state):
        """Điều khiển relay/actuator 3"""
        if state:
            self.ser.write(bytearray(RELAY3_ON))
        else:
            self.ser.write(bytearray(RELAY3_OFF))
        # Chờ cho relay hoạt động
        time.sleep(0.01)
        
    def all_on(self):
        """Bật tất cả relay/actuator"""
        self.switch_actuator_1(True)
        time.sleep(0.03)
        self.switch_actuator_2(True)
        time.sleep(0.03)
        self.switch_actuator_3(True)
        
    def all_off(self):
        """Tắt tất cả relay/actuator"""
        self.switch_actuator_1(False)
        time.sleep(0.03)
        self.switch_actuator_2(False)
        time.sleep(0.03)
        self.switch_actuator_3(False)

    def close(self):
        """Đóng kết nối serial"""
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()


# Test kết nối và điều khiển relay
if __name__ == "__main__":
    print("Testing Modbus Controller...")
    try:
        controller = ModbusMaster()
        print("Connected successfully. Testing relay control...")
        
        for i in range(3):
            print(f"Test cycle {i+1}/3")
            
            # Bật từng relay lần lượt
            print("Turning ON relays one by one...")
            controller.switch_actuator_1(True)
            time.sleep(1)
            controller.switch_actuator_2(True)
            time.sleep(1)
            controller.switch_actuator_3(True)
            time.sleep(1)
            
            # Tắt tất cả
            print("Turning OFF all relays...")
            controller.all_off()
            time.sleep(1)
            
            # Bật tất cả
            print("Turning ON all relays...")
            controller.all_on()
            time.sleep(1)
            
            # Tắt tất cả
            print("Turning OFF all relays...")
            controller.all_off()
            time.sleep(1)
        
        print("Test completed successfully!")
        controller.close()
        
    except Exception as e:
        print(f"Error during test: {e}")
