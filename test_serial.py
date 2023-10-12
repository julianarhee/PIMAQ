import serial
import time

arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600)

def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline() #.decode('utf-8').rstrip()
    return data

try:
    while True:
        num = input("Enter string: ")
        value = write_read(num)
        print(value)
except serial.SerialException as e:
    print(e)
finally:
    arduino.close()


