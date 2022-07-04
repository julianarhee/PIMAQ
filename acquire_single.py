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
#        except OSError as e:
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

def initialize_and_loop(config, camname, cam, args, experiment, start_t, arduino):

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
            metadata_format='hdf5', 
            uncompressed=False, # setting to False always because you don't need to calibrate it
            preview=args.preview,
            verbose=args.verbose,
            strobe=cam['strobe'],
            codec=config['codec'],
            acquisition_rate=args.frame_rate
            )

    else:
        raise ValueError('Invalid camera type: %s' %cam['type'])
    # sync_mode = 'master' if serial == args.master else 'slave'
    if cam['master']:
        sleep_time = 5 #np.random.randn()+3
        time.sleep(sleep_time)
        device.start()
    else:
        sleep_time = 5 #np.random.randn()+3
        time.sleep(sleep_time)
        device.start()
    #print("STARTED DEVICE")
    # runs until keyboard interrupt!
    arduino.write(b'S%d\r' % args.frame_rate)   
    time.sleep(1)

    device.loop()

    print("out of loop")

    #return camname, cam['serial']   

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

    parser.add_argument('-f', '--frame_rate', default=10, type=float,
         help='frame rate (default: 10Hz)')

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
    port = args.port
    baudrate = 115200
    print("# Please specify a port and a baudrate")
    print("# using hard coded defaults " + port + " " + str(baudrate))
    arduino = pyserial.Serial(port, baudrate, timeout=0.5)
    time.sleep(1)
    #flushBuffer()
    sys.stdout.flush()
    print("Connected serial port...")

    frame_rate = args.frame_rate

    """
    Originally I wanted to initialize each device, then pass each device to "run_loop" 
    in its own process. However, pyrealsense2 config objects and pyrealsense2 pipeline objects
    are not pickle-able, and python pickles arguments before passing them to a process. Therefore,
    you have to initialize the configuration and the pipeline from within the process already!
    """
    start_t = time.perf_counter()
    tuples = []
    # make a name for this experiment
    experiment = '%s_%s' %(args.name, time.strftime('%y%m%d_%H%M%S', time.localtime()))
    if args.save:
        directory = os.path.join(config['savedir'], experiment)
        if not os.path.isdir(directory):
            os.makedirs(directory)
            with open(os.path.join(directory, 'loaded_config_file.yaml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    for camname, cam in config['cams'].items():
        print(camname)
        tup = (config, camname, cam, args, experiment, start_t)
        #p = mp.Process(target=initialize_and_loop, args=(tup,))
        #p.start()
        tuples.append(tup)


    if len(config['cams']) >1 :
        print("Multi-camera")
        try:
            procs=[]
            for camname, cam in config['cams'].items():
                print(camname)
                cv2.namedWindow(camname, cv2.WINDOW_NORMAL)
                #time.sleep(2)
                tup = (config, camname, cam, args, experiment, start_t)
                p = mp.Process(target=initialize_and_loop, args=(config, camname, cam, args, experiment, start_t, arduino))
                p.start()
                print(camname, p)
                procs.append(p) #camname, p))
                tuples.append(tup)
            if args.verbose:
                print('Tuples created, starting...')
            # send triggers 
            #arduino.write(b'S%d\r' % frame_rate)
        except KeyboardInterrupt: 
            for proc in procs:
                print(proc, proc.is_alive())
                proc.terminate()
                proc.join()
        finally: 
            for proc in procs:
                proc.join()
                if proc.is_alive():
                    print('alive', proc)
                    time.sleep(1)
                else:
                    print('dead:', proc)
                #p.terminate()
                #p.join()
                #p.close()
                
#        with mp.Pool(len(config['cams'])) as p:
#            try:
#                p.starmap(initialize_and_loop, tuples)
#                print('mapped')
#                key = cv2.waitKey(1)
#                if key==27:
#                    raise(KeyboardInterrupt)
#            except KeyboardInterrupt:
#                p.close()
#                p.join()
#                print('User interrupted acquisition')
#            finally:
#                p.close()
#                p.join()
#
        print("pool closed")
    else:
        tuples = [(config, camname, cam, args, experiment, start_t)]
        assert len(tuples) == 1
        initialize_and_loop(*tuples[0])

    #if args.preview:
    cv2.destroyAllWindows()


    arduino.write(b'Q\r')
    print('Stopped arduino....')



if __name__=='__main__':
    main()