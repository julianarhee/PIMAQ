####!/usr/bin/env python3



from re import A
import serial as pyserial
import time
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='Multi-camera acquisition in Python.')

    parser.add_argument('--port', default='/dev/ttyACM0', type=str,
         help='port for arduino (default: /dev/ttyACM0)')

    parser.add_argument('-f', '--frame_rate', default=10, type=float,
         help='frame rate (default: 10Hz)')


    args = parser.parse_args()

    # port = "/dev/cu.usbmodem145201"
    port = '/dev/ttyACM0' #args.port
    baudrate = 115200
    print("# Please specify a port and a baudrate")
    print("# using hard coded defaults " + port + " " + str(baudrate))

    arduino = pyserial.Serial(port, baudrate, timeout=0.5)
    time.sleep(1)
    #flushBuffer()
    sys.stdout.flush()
    print("Connected serial port...")

    frame_rate = args.frame_rate

    time.sleep(0.5)
    arduino.write(b'S%d\r' % args.frame_rate)   
    time.sleep(1)



if __name__=='__main__':
    main()
