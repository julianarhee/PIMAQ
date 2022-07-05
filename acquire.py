####!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   record_and_sync_frames.py
@Time    :   2022/02/01 12:27:07
@Author  :   julianarhee 
@Contact :   juliana.rhee@gmail.com

Record frames with Basler (pypylon) with output trigger (audo abt)

'''
from multiprocessing import set_start_method
try:
    import pyrealsense2 as rs
except ImportError as e:
    print('pyrealsense not found, cannot acquire realsense cameras...')
    rs = None
from re import A, U
import numpy as np
import cv2
import time
# import matplotlib.pyplot as plt
import h5py
import os
import argparse
import multiprocessing as mp
import yaml
import warnings
import sys
import serial as pyserial

#from concurrent.futures import ProcessPoolExecutor as PoolExec


# import queue
if rs is not None:
    from devices import Realsense 
import realsense_utils
import traceback
try:
    from devices import PointGrey
except ImportError as e:
    print('PySpin not found, can''t acquire from FLIR cameras')
    PointGrey = None
try:
    from devices import Basler
except ImportError as e:
    print('Basler not found, can''t acquire from Basler cameras')
    Basler = None

#from functools import wraps


#def waitpid(func):
#    cache = {}
#
#    @wraps(func)
#    def wrapper(pid, options):
#        try:
#            wpid, status = func(pid, options)
#            if wpid > 0:
#                cache[wpid] = status
#        except OSError as e'csv', #:
#            if pid in cache:
#                return pid, cache[pid]
#            else:
#                raise e
#        else:
#            return wpid, status
#
#    return wrapper
#
#os.waitpid = waitpid(os.waitpid)
#

def initialize_and_loop(tuple_list_item): #config, camname, cam, args, experiment, start_t): #, arduino):
    config, camname, cam, args, experiment, start_t = tuple_list_item
    if cam['type'] == 'Realsense':
        device = Realsense(serial=cam['serial'], 
            start_t=start_t,
            options=config['realsense_options'],
            save=args.save,
            savedir=config['savedir'], 
            experiment=experiment,
            name=camname,
            movie_format=args.movie_format,
            metadata_format='hdf5', 
            uncompressed=config['realsense_options']['uncompressed'],
            preview=args.preview,
            verbose=args.verbose,
            master=cam['master'],
            codec=config['codec']
            )
    elif cam['type'] == 'PointGrey':
        device = PointGrey(serial=cam['serial'], 
            start_t=start_t, 
            options=cam['options'],
            save=args.save, 
            savedir=config['savedir'],
            experiment=experiment,
            name=camname,
            movie_format=args.movie_format, 
            metadata_format='hdf5', 
            uncompressed=False, # setting to False always because you don't need to calibrate it
            preview=args.preview,
            verbose=args.verbose,
            strobe=cam['strobe'],
            codec=config['codec'],
            )
    elif cam['type'] == 'Basler':
        device = Basler(serial=cam['serial'], 
            start_t=start_t, 
            options=cam['options'],
            save=args.save, 
            savedir=config['savedir'],
            experiment=experiment,
            name=camname,
            movie_format=args.movie_format, 
            metadata_format='csv', #'hdf5', 
            uncompressed=False, # setting to False always because you don't need to calibrate it
            preview=args.preview,
            verbose=args.verbose,
            strobe=cam['strobe'],
            codec=config['codec'],
            acquisition_fps=args.acquisition_fps,
            videowrite_fps = args.videowrite_fps,
            experiment_duration = args.experiment_duration
            )

    else:
        raise ValueError('Invalid camera type: %s' %cam['type'])
    # sync_mode = 'master' if serial == args.master else 'slave'
    if cam['master'] in [True, 'True']:
        sleep_time = 1 #np.random.randn()+3
        time.sleep(sleep_time)
        device.start()
        time.sleep(2)

        # Set up arduino for trigger
        arduino = initialize_arduino(port=args.port, baudrate=115200)

        arduino.write(b'S%d\r' % args.acquisition_fps)   
        print("***Arduino started***")
        device.arduino = arduino

    else:
        sleep_time = 1 #np.random.randn()+3
        time.sleep(sleep_time)
        device.start()
        time.sleep(2)

    #time.sleep(5)
    #arduino.write(b'S%d\r' % args.acquisition_fps)   
    #print("***Arduino started***")

    #print("STARTED DEVICE")
    # runs until keyboard interrupt!
    try:
        device.loop()
        #return camname, cam['serial']   
    except KeyboardInterrupt:
        if cam['master'] in [True, 'True']:
            arduino.write(b'Q\r')
            print("Closed Arduino")

    return ("done")


def initialize_arduino(port='/dev/ttyACM0', baudrate=115200):
    # Set up arduino for trigger
    # port = "/dev/cu.usbmodem145201"
    #port = args.port
    #baudrate = 115200
    #print("# Please specify a port and a baudrate")
    #print("# using hard coded defaults " + port + " " + str(baudrate))
    arduino = pyserial.Serial(port, baudrate, timeout=0.5)
    time.sleep(1)
    #flushBuffer()
    sys.stdout.flush()
    print("Connected serial port...")
    time.sleep(2)

    #serial_queue = mp.Queue()
    return arduino #, serial_queue

def send_to_arduino(arduino, serial_queue, acquisition_fps):
    # send data to ardunio, maybe in a loop
    # sleeping 30ms after each update
    while True:
        data = serial_queue.get()
        print(data)
        if data=='q':
            arduino.write(b'Q\r')
            print("***Arduino stopped***")
            break

        arduino.write(b'S%d\r' % acquisition_fps)   
        print("***Arduino started***")

 

def main():
    parser = argparse.ArgumentParser(description='Multi-camera acquisition in Python.')
    parser.add_argument('-n','--name', type=str, default='JB999',
        help='Base name for directories. Example: mouse ID')
    parser.add_argument('-c', '--config', type=str, default='config.yaml', 
        help='Configuration for acquisition. Defines number of cameras, serial numbers, etc.')
    parser.add_argument('-p', '--preview', default=False, action='store_true',
        help='Show preview in opencv window')
    parser.add_argument('-s', '--save', default=False, action='store_true',
        help='Use this flag to save to disk. If not passed, will only view')
    parser.add_argument('-v', '--verbose', default=False,action='store_true',
        help='Use this flag to print debugging commands.')
    parser.add_argument('--movie_format', default='opencv',
        choices=['hdf5','opencv', 'ffmpeg', 'directory'], type=str,
        help='Method to save files to movies. Dramatically affects performance and filesize')

    parser.add_argument('--port', default='/dev/ttyACM0', type=str,
         help='port for arduino (default: /dev/ttyACM0)')

    parser.add_argument('-r', '--acquisition_fps', default=10, type=float,
         help='Acquisition frame rate (default: 10 Hz)')
    parser.add_argument('-w', '--videowrite_fps', default=10, type=float,
         help='Video save frame rate (default: 10 Hz)')
    parser.add_argument('-d', '--experiment_duration', default=np.inf, type=float,
         help='Experiment dur in minutes (default: inf.)')

    args = parser.parse_args()

#    if args.movie_format == 'ffmpeg':
#        warnings.filterwarnings('ffmpeg uses lots of CPU resources. ' + 
#            '60 Hz, 640x480 fills RAM in 5 minutes. Consider opencv')

    if rs is not None:
        serials = realsense_utils.enumerate_connected_devices()
        if args.verbose:
            print('Realsense Serials: ', serials)

    if os.path.isfile(args.config):
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        raise ValueError('Invalid config file: %s' %args.config)


    # Set up arduino for trigger
    # port = "/dev/cu.usbmodem145201"
    #arduino=None
    arduino = initialize_arduino(port=args.port, baudrate=115200)
    arduino.write(b'Q\r') # % args.acquisition_fps)   
    print("***Arduino cleared***")
    arduino.close()

    acquisition_fps = args.acquisition_fps
    videowrite_fps = args.videowrite_fps

    """
    Originally I wanted to initialize each device, then pass each device to "run_loop" 
    in its own process. However, pyrealsense2 config objects and pyrealsense2 pipeline objects
    are not pickle-able, and python pickles arguments before passing them to a process. Therefore,
    you have to initialize the configuration and the pipeline from within the process already!
    """
    start_t = time.perf_counter()
    #tuples = []
    # make a name for this experiment
    experiment = '%s_%s' %(args.name, time.strftime('%y%m%d_%H%M%S', time.localtime()))
    if args.save:
        directory = os.path.join(config['savedir'], experiment)
        if not os.path.isdir(directory):
            os.makedirs(directory)
            with open(os.path.join(directory, 'loaded_config_file.yaml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    tuple_list=[]
    for camname, cam in config['cams'].items():
        print(camname)
        tup = (config, camname, cam, args, experiment, start_t) #, serial_queue) #, arduino)
        #p = mp.Process(target=initialize_and_loop, args=(tup,))
        #p.start()
        tuple_list.append(tup)

#    if len(config['cams']) >1 :
#        print("Multi-camera")
#
#        try:
#            procs=[]
#            for camname, cam in config['cams'].items():
#                print(camname)
#                #cv2.namedWindow(camname, cv2.WINDOW_NORMAL)
#                #time.sleep(2)
#                tup = (config, camname, cam, args, experiment, start_t)
#                p = mp.Process(target=initialize_and_loop, args=(config, camname, cam, args, experiment, start_t, arduino))
#                p.start()
#                print(camname, p)
#                procs.append(p) #camname, p))
#                tuples.append(tup)
#            if args.verbose:
#                print('Tuples created, starting...')
#            # send triggers 
#            #arduino.write(b'S%d\r' % frame_rate)
#        except KeyboardInterrupt: 
#            for proc in procs:
#                print(proc, proc.is_alive())
#                proc.terminate()
#                proc.join()
#        finally: 
#            for proc in procs:
#                proc.join()
#                if proc.is_alive():
#                    print('alive', proc)
#                    time.sleep(1)
#                else:
#                    print('dead:', proc)

    if len(config['cams']) >1 :
        print("Multi-camera")
        #with mp.Pool(len(config['cams'])) as pool:
        pool = mp.Pool(len(config['cams'])) 
        try:
            #res = pool.starmap(initialize_and_loop, tuple_list)
            #res = pool.map(initialize_and_loop, *zip(*tuple_list))
            res = pool.map(initialize_and_loop, tuple_list)
            time.sleep(5)
            print('mapped')
            time.sleep(1)
            print("STARTED")

            key = cv2.waitKey(1) & 0xFF
            if key==27:
                raise(KeyboardInterrupt)
            #elif key==ord('s'):
            #    arduino.write(b'S%d\r' % args.acquisition_fps)   
            #    print("***Arduino started***")

        except KeyboardInterrupt:
            print('User interrupted acquisition')
            print("pool closed")
        finally:
            #pool.shutdown()
            pool.terminate()
            pool.join()
            pool.close()

    else:
        tuple_list = [(config, camname, cam, args, experiment, start_t)]
        assert len(tuple_list) == 1
        initialize_and_loop(*tuple_list[0])

    #if args.preview:
    cv2.destroyAllWindows()
    for p in mp.active_children():
        print(p)
        p.terminate()

    #arduino.write(b'Q\r')
    #print('Stopped arduino....')

    # get all active child processes
    active = mp.active_children()
    print(f'Active Children: {len(active)}')
    # terminate all active children
    for child in active:
        child.terminate()
    # block until all children have closed
    for child in active:
        child.join()
        child.close()

if __name__=='__main__':
    set_start_method("spawn")
    main()

    #os.killpg(0, signal.SIGKILL) # kill all processes in my group