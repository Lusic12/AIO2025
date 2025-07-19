import serial
import time

RELAY1_ON =  [1, 5, 0, 0, 0xFF, 0, 0x8C, 0x3A]  #  Kích hoạt mở relay 1
RELAY1_OFF = [1, 5, 0, 0, 0,    0, 0xCD, 0xCA]  #  Kích hoạt tắt relay 1

RELAY2_ON =  [1, 5, 0, 1, 0xFF, 0, 0xDD, 0xFA]  #  Kích hoạt mở relay 2
RELAY2_OFF = [1, 5, 0, 1, 0,    0, 0x9C, 0x0A]  #  Kích hoạt tắt relay 2

RELAY3_ON =  [1, 5, 0, 2, 0xFF, 0, 0x2D, 0xFA]  #  Kích hoạt mở relay 3
RELAY3_OFF = [1, 5, 0, 2, 0,    0, 0x6C, 0x0A]  #  Kích hoạt tắt relay 3

class ModbusMaster:
    """
    Simple Modbus relay controller: Kết nối cổng COM với buadrate 9600.
    """
    def __init__(self, port):
        # Open serial connection to relay at 9600 baud
        self.ser = serial.Serial(
            port=port,
            baudrate=9600,
            timeout=2
        )
        print(f"Connected to {port} at 9600 baud")

    def switch_actuator_1(self, state):
        # Kích hoạt 1 mở or tắt
        cmd = RELAY1_ON if state else RELAY1_OFF
        self.ser.write(bytearray(cmd))

    def switch_actuator_2(self, state):
        # Kích hoạt 2 mở or tắt
        cmd = RELAY2_ON if state else RELAY2_OFF
        self.ser.write(bytearray(cmd))

    def switch_actuator_3(self, state):
        # Kích hoạt 3 mở or tắt
        cmd = RELAY3_ON if state else RELAY3_OFF
        self.ser.write(bytearray(cmd))

    def all_on(self):
        self.switch_actuator_1(True)
        time.sleep(0.1)
        self.switch_actuator_2(True)
        time.sleep(0.1)
        self.switch_actuator_3(True)
        print(f"Cả 3 relay đều mở")

    def all_off(self):
        self.switch_actuator_1(False)
        time.sleep(0.1)
        self.switch_actuator_2(False)
        time.sleep(0.1)
        self.switch_actuator_3(False)
        print(f"Cả 3 relay đều tắt")

    def close(self):
        # Đóng cổng serial
        self.ser.close()
