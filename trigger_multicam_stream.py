####!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   record_and_sync_frames.py
@Time    :   2022/02/01 12:27:07
@Author  :   julianarhee 
@Contact :   juliana.rhee@gmail.com

Record frames with Basler (pypylon) with output trigger (audo abt)

'''
#%%
from pypylon import pylon
#from pypylon import genicam
import copy
import array
import sys
import errno
import os
import optparse
import hashlib
import serial
import cv2 #from scipy.misc import imsave
import threading
import time
import subprocess
from datetime import datetime
#from libtiff import TIFF

from queue import Queue
import numpy as np
import multiprocessing as mp
import traceback

import imageio #from imageio import get_writer

#from samples.configurationeventprinter import ConfigurationEventPrinter
#from samples.imageeventprinter import ImageEventPrinter

current_ms_time = lambda: int(round(time.time() * 1000))


class FFMPEG_VideoWriter:
    """ A class for FFMPEG-based video writing.

    A class to write videos using ffmpeg. ffmpeg will write in a large
    choice of formats.

    Parameters
    -----------

    filename
      Any filename like 'video.mp4' etc. but if you want to avoid
      complications it is recommended to use the generic extension
      '.avi' for all your videos.

    size
      Size (width,height) of the output video in pixels.

    fps
      Frames per second in the output video file.

    codec
      FFMPEG codec. It seems that in terms of quality the hierarchy is
      'rawvideo' = 'png' > 'mpeg4' > 'libx264'
      'png' manages the same lossless quality as 'rawvideo' but yields
      smaller files. Type ``ffmpeg -codecs`` in a terminal to get a list
      of accepted codecs.

      Note for default 'libx264': by default the pixel format yuv420p
      is used. If the video dimensions are not both even (e.g. 720x405)
      another pixel format is used, and this can cause problem in some
      video readers.

    audiofile
      Optional: The name of an audio file that will be incorporated
      to the video.

    preset
      Sets the time that FFMPEG will take to compress the video. The slower,
      the better the compression rate. Possibilities are: ultrafast,superfast,
      veryfast, faster, fast, medium (default), slow, slower, veryslow,
      placebo.

    bitrate
      Only relevant for codecs which accept a bitrate. "5000k" offers
      nice results in general.

    withmask
      Boolean. Set to ``True`` if there is a mask in the video to be
      encoded.

    """

    def __init__(self, filename, size, fps, codec="libx264", audiofile=None,
                 preset="fast", bitrate='5000000', pixfmt="yuv420p",
                 logfile=None, threads=None, ffmpeg_params=None):

        if logfile is None:
            logfile = subprocess.PIPE

        self.filename = filename
        self.codec = codec
        self.ext = self.filename.split(".")[-1]

        self._pix_fmt = None
        self._depth = None
        self._size = None

        # order is important
        cmd = [
            #"ffmpeg-4.2.1-win64-static/bin/ffmpeg",
            "/bin/ffmpeg", 
            '-y', # overwrite 
            '-loglevel', 'error' if logfile == subprocess.PIPE else 'info',
            #'-vcodec', 'libx264',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            #'-s', '%dx%d' % (size[1], size[0]),
            #'-pix_fmt', 'yuv420p', #pixfmt,
            '-r', '%.02f' % fps,
            '-i', '-', '-an'
        ]

            #'-c:v', 'h264_nvenc', 
            #'-b:v', '1M'


        cmd.extend([
            #'-c:v', 'h264_nvenc', #_nvenc',
            '-vcodec', codec,
            '-preset', preset
        ])
        if ffmpeg_params is not None:
            cmd.extend(ffmpeg_params)
        if bitrate is not None:
            cmd.extend([
                '-b', bitrate
            ])
        if threads is not None:
            cmd.extend(["-threads", str(threads)])

        if ((codec == 'libx264') and
                (size[0] % 2 == 0) and
                (size[1] % 2 == 0)):
            print("** adjusting**")
#            cmd.extend([
#                '-pix_fmt', 'yuv420p'
#            ])
        cmd.extend([
            filename
        ])

        popen_params = {"stdout": subprocess.DEVNULL,
                        "stderr": logfile,
                        "stdin": subprocess.PIPE}

        # This was added so that no extra unwanted window opens on windows
        # when the child process is created
        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

        print(cmd)
        self.proc = subprocess.Popen(cmd, **popen_params)


    def write_frame(self, im): #img_array):
        """ Writes one frame in the file."""

        # Get props of image
        h, w = im.shape[:2]
        size = w, h
        depth = 1 if im.ndim == 2 else im.shape[2]

        # Ensure that image is in uint8
        im = imageio.core.util.image_as_uint(im, bitdepth=8)
        # To be written efficiently, ie. without creating an immutable
        # buffer, by calling im.tobytes() the array must be contiguous.
        if not im.flags.c_contiguous:
            # checkign the flag is a micro optimization.
            # the image will be a numpy subclass. See discussion
            # https://github.com/numpy/numpy/issues/11804
            im = np.ascontiguousarray(im)

        # Set size and initialize if not initialized yet
        if self._size is None:
            map = {1: "gray", 2: "gray8a", 3: "rgb24", 4: "rgba"}
            self._pix_fmt = map.get(depth, None)
            if self._pix_fmt is None:
                raise ValueError("Image must have 1, 2, 3 or 4 channels")
            self._size = size
            self._depth = depth
            #self._initialize()


        try:
               self.proc.stdin.write(im.tobytes()) #im_array) #img_array.tobytes())
        except IOError as err:
            print(err)
            _, ffmpeg_error = self.proc.communicate()
            error = (str(err) + ("\n\nMoviePy error: FFMPEG encountered "
                                 "the following error while writing file %s:"
                                 "\n\n %s" % (self.filename, str(ffmpeg_error))))

            if b"Unknown encoder" in ffmpeg_error:

                error = error+("\n\nThe video export "
                  "failed because FFMPEG didn't find the specified "
                  "codec for video encoding (%s). Please install "
                  "this codec or change the codec when calling "
                  "write_videofile. For instance:\n"
                  "  >>> clip.write_videofile('myvid.webm', codec='libvpx')")%(self.codec)

            elif b"incorrect codec parameters ?" in ffmpeg_error:

                 error = error+("\n\nThe video export "
                  "failed, possibly because the codec specified for "
                  "the video (%s) is not compatible with the given "
                  "extension (%s). Please specify a valid 'codec' "
                  "argument in write_videofile. This would be 'libx264' "
                  "or 'mpeg4' for mp4, 'libtheora' for ogv, 'libvpx for webm. "
                  "Another possible reason is that the audio codec was not "
                  "compatible with the video codec. For instance the video "
                  "extensions 'ogv' and 'webm' only allow 'libvorbis' (default) as a"
                  "video codec."
                  )%(self.codec, self.ext)

            elif  b"encoder setup failed" in ffmpeg_error:

                error = error+("\n\nThe video export "
                  "failed, possibly because the bitrate you specified "
                  "was too high or too low for the video codec.")

            elif b"Invalid encoder type" in ffmpeg_error:

                error = error + ("\n\nThe video export failed because the codec "
                  "or file extension you provided is not a video")

            raise IOError(error)
        except Exception as e:
            traceback.print_exc()


    def close(self):
        if self.proc:
            self.proc.stdin.close()
            if self.proc.stderr is not None:
                self.proc.stderr.close()
            self.proc.wait()

        self.proc = None

    # Support the Context Manager protocol, to ensure that resources are cleaned up.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def extract_options(options):

    parser = optparse.OptionParser()
    parser.add_option('--save-dir', action="store", dest="save_dir", default="/home/julianarhee/Videos/basler", help="out path directory [default: /home/julianarhee/Videos/basler]")
    parser.add_option('--output-format', action="store", dest="output_format", type="choice", choices=['png', 'npz'], default='png', help="out file format, png or npz [default: png]")
    parser.add_option('--save', action='store_true', dest='save_images', default=False, help='Flag to save images to disk.')
    parser.add_option('--basename', action="store", dest="basename", default="acquisition", help="basename for saved acquisition")

    parser.add_option('--write-process', action="store_true", dest="save_in_separate_process", default=True, help="spawn process for disk-writer [default: True]")
    parser.add_option('--write-thread', action="store_false", dest="save_in_separate_process", help="spawn threads for disk-writer")
    parser.add_option('-f', '--frame-rate', action="store", dest="frame_rate", help="requested frame rate", type="float", default=30.0)

    parser.add_option('-W', '--width', action="store", dest="width", help="image width", type="int", default=1920)
    parser.add_option('-H', '--height', action="store", dest="height", help="image height", type="int", default=1200)
    parser.add_option('-x', '--exposure', action="store", dest="exposure", help="exposure (us))", type='float', default=68295.0) #16670)

    parser.add_option('-t', '--duration', action="store", dest="duration", help="recording duration (min)", type='float', default=np.inf)

    parser.add_option('--port', action="store", dest="port", help="port for arduino (default: /dev/ttyACM0)", default='/dev/ttyACM0')
    parser.add_option('--disable', action='store_false', dest='enable_framerate', default=True, help='Flag to disable acquisition frame rate setting.')

    parser.add_option('--no-trigger', action='store_false', dest='send_trigger', default=True, help='Flag to disable arduino trigger output.')

    (options, args) = parser.parse_args()

    return options


# ############################################
# Save functions
# ############################################

def create_dirs(save_dir, basename='acquisition'):
    tstamp_fmt = '%Y%m%d-%H%M%S'
    datestr = datetime.now().strftime(tstamp_fmt) 
    basename = '%s_%s' % (datestr, basename)
    dst_dir = os.path.join(save_dir, basename)
    frame_write_dir = os.path.join(dst_dir, 'frames')
    try:
        os.makedirs(frame_write_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
        pass 
    frametimes_fpath = os.path.join(dst_dir, 'frame_times.txt')
    performance_fpath = os.path.join(dst_dir, 'performance.txt')

    return frame_write_dir, frametimes_fpath, performance_fpath


# ############################################
# Camera functions
# ############################################

def connect_to_camera(connect_retries=50, max_cams=2): #frame_rate=20., acquisition_line='Line3', enable_framerate=True,
                     #send_trigger=False,  width=1200, height=1200, exposure=16670):
    print('Searching for camera...')
    #max_cams = 2
    cameras = None
    # get transport layer factory
    tlFactory = pylon.TlFactory.GetInstance()

    # get the camera list 
    devices = tlFactory.EnumerateDevices()
    print('Connecting to cameras...')   

    # Create array of cameras
    n = 0
    while cameras is None and n < connect_retries:
        try:
            cameras = pylon.InstantCameraArray(min(len(devices), max_cams))
            l = cameras.GetSize()
            #pylon.TlFactory.GetInstance().CreateFirstDevice())
            print("L", l)
            print(cameras)
            #time.sleep(0.5)
            #camera.Open()
            #print("Bound to device:" % (camera.GetDeviceInfo().GetModelName()))

        except Exception as e:
            print('.')
            time.sleep(0.1)
            camera = None
            n += 1

    if cameras is None:
        try:
            #import opencv_fallback
            #camera = opencv_fallback.Camera(0)
            print("Bound to OpenCV fallback camera.")
        except Exception as e2:
            print("Could not load OpenCV fallback camera")
            print(e2)
            exit()
    else:
        for ix, cam in enumerate(cameras):
            cam.Attach(tlFactory.CreateDevice(devices[ix]))
            #camera.Open()
            print("Bound to device: %s" % (cam.GetDeviceInfo().GetModelName()))
            # setup named window per camera
            window_name = f'Camera-{ix:03}'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # open camera 
    cameras.Open()
    # store a unique number for each camera to identify the incoming images
    for idx, cam in enumerate(cameras):
        camera_serial = cam.DeviceInfo.GetSerialNumber()
        print(f"set context {idx} for camera {camera_serial}")
        cam.SetCameraContext(idx)

    print("Success!")

    return cameras

def set_camera_properties(cameras, frame_rate=20., acquisition_line='Line4', enable_framerate=True,
                     send_trigger=False,  width=1200, height=1200, exposure=16670):

    # acquisition settings
    for i, cam in enumerate(cameras):
        cam.AcquisitionFrameRateEnable = enable_framerate
        cam.AcquisitionFrameRate = frame_rate
        if not send_trigger: #enable_framerate:
            cam.AcquisitionMode.SetValue('Continuous')
            print("Setting acquisition frame rate: %.2f Hz" % cam.AcquisitionFrameRate())
            for trigger_type in ['FrameStart', 'FrameBurstStart']:
                cam.TriggerSelector = trigger_type
                cam.TriggerMode = "Off"
        else: 

            #cam.AcquisitionMode.SetValue('Continuous')

            cam.AcquisitionFrameRate = frame_rate
            # Set  trigger
            # get clean powerup state
            cam.UserSetSelector = "Default"
            cam.UserSetLoad.Execute()
            cam.TriggerSelector = "FrameStart"
            cam.TriggerMode = "On"
            cam.TriggerDelay.SetValue(0)
            cam.TriggerActivation = 'RisingEdge' 
            #cam.AcquisitionMode.SetValue('SingleFrame')
            cam.AcquisitionMode.SetValue('Continuous')
            #cam.AcquisitionStatusSelector="FrameTriggerWait"

            # Set IO lines:
            cam.TriggerSource.SetValue("Line4")
            cam.LineSelector.SetValue("Line4") #acquisition_line) # select GPIO 1
            cam.LineMode.SetValue('Input')     # Set as input
            #uam.LineStatus = False #.SetValue(False)
            # Output:
            #cam.LineSelector.SetValue('Line4')
            #cam.LineMode.SetValue('Output')
            #cam.LineSource.SetValue('UserOutput3') # Set source signal to User Output 1
            #cam.UserOutputSelector.SetValue('UserOutput3')
            #cam.UserOutputValue.SetValue(False)

            # setup trigger and acquisition control
            #cam.TriggerSource.SetValue(acquisition_line)
            #camera.TriggerSelector.SetValue('AcquisitionStart')
            #cam.TriggerActivation = 'RisingEdge'
         
            # Trigger On Line 3 FrameStart Rising Edge, Exposure Out Line 2.
            #cam.LineSelector.SetValue('Line4')
            #cam.LineMode.SetValue('Output')
            #cam.LineSource.SetValue('ExposureActive')

            cam.ChunkModeActive = True
            cam.ChunkEnable = True
 
        # Set image format:
        cam.Width.SetValue(width) #(960)
        cam.Height.SetValue(height) #(600)
        cam.PixelFormat.SetValue('Mono8')
        cam.ExposureMode.SetValue('Timed')
        cam.ExposureTime.SetValue(exposure) #(40000)

        try:
            actual_framerate = cam.ResultingFrameRate.GetValue()
            assert cam.AcquisitionFrameRate() <= cam.ResultingFrameRate(), "Unable to acquieve desired frame rate (%.2f Hz)" % float(cam.AcquisitionFrameRate.GetValue())
        except AssertionError:
            cam.AcquisitionFrameRate.SetValue(float(cam.ResultingFrameRate.GetValue()))
            print("Set acquisition rate to: %.2f" % cam.AcquisitionFrameRate())
        print('Final frame rate: %.2f Hz' % (cam.AcquisitionFrameRate()))
         
    return cameras

def convert_image(grabResult):
    #im_native = res.Array
    #print(im_native.shape)
    im_to_show = converter.Convert(grabResult)
    im_array = im_to_show.GetArray()

    return im_array

#


#


# compute a hash from the current time so that we don't accidentally overwrite old data
#run_hash = hashlib.md5(str(time.time())).hexdigest()

class SampleImageEventHandler(pylon.ImageEventHandler):
    def OnImageGrabbed(self, cam, grabResult):
        #print("CSampleImageEventHandler::OnImageGrabbed called.")
        cam.UserOutputValue.SetValue(True)
        #camera.UserOutputValue.SetValue(True)

class TriggeredImage(pylon.ImageEventHandler):
    def __init__(self):
        super().__init__()
        self.grab_times = []
    def OnImageGrabbed(self, camera, grabResult):
        self.grab_times.append(grabResult.TimeStamp)

class ImageEventPrinter(pylon.ImageEventHandler):
    def OnImagesSkipped(self, camera, countOfSkippedImages):
        print("OnImagesSkipped event for device ", camera.GetDeviceInfo().GetModelName())
        print(countOfSkippedImages, " images have been skipped.")
        print()

    def OnImageGrabbed(self, camera, grabResult):
        print("OnImageGrabbed event for device ", camera.GetDeviceInfo().GetModelName())

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            print("SizeX: ", grabResult.GetWidth())
            print("SizeY: ", grabResult.GetHeight())
            img = grabResult.GetArray()
            print("Gray values of first row: ", img[0])
            print()
        else:
            print("Error: ", camera.GetDeviceInfo().GetModelName(), grabResult.GetErrorCode(), grabResult.GetErrorDescription())

# ############################################
# Main 
# ############################################
        
if __name__ == '__main__':

    optsE = extract_options(sys.argv[1:])

    # Arduino serial port
    port = optsE.port
    send_trigger=optsE.send_trigger
    
    # Camera settings:
    frame_rate = optsE.frame_rate
    frame_period = float(1/frame_rate)
    width = optsE.width
    height = optsE.height
    exposure = optsE.exposure
    duration = optsE.duration
    duration_sec = duration*60.0

    # Acquisition settings
    acquire_images = True
    save_images = optsE.save_images #True
    save_dir = optsE.save_dir
    output_format = optsE.output_format
    save_in_separate_process = optsE.save_in_separate_process   
    save_as_png = True #output_format=='tiff'
    basename = optsE.basename
    
    # Make the output path if it doesn't already exist
    #frame_write_dir, frametimes_fpath, performance_fpath = creaet_dirs(save_dir)

    time.sleep(1.0)
#%%
    # -------------------------------------------------------------
    # Set up serial connection
    # -------------------------------------------------------------
    if send_trigger:
        # port = "/dev/cu.usbmodem145201"
        baudrate = 115200
        print("# Please specify a port and a baudrate")
        print("# using hard coded defaults " + port + " " + str(baudrate))
        ser = serial.Serial(port, baudrate, timeout=0.5)
        time.sleep(1)
        #flushBuffer()
        sys.stdout.flush()
        print("Connected serial port...")
    else:
        ser=None
#%%
    # -------------------------------------------------------------
    # Camera Setup
    # ------------------------------------------------------------     
    enable_framerate = optsE.enable_framerate
    acquisition_line = 'Line4'
    camera = None
    if acquire_images:
        cameras = connect_to_camera(max_cams=2)

        cameras = set_camera_properties(cameras, frame_rate=frame_rate, acquisition_line=acquisition_line, 
                                   enable_framerate=enable_framerate, send_trigger=send_trigger, width=width, height=height,
                            exposure=exposure) 
    # Attach event handlers:
    for cam in cameras:
        #cam.RegisterImageEventHandler(SampleImageEventHandler(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)

        cam.RegisterConfiguration(pylon.ConfigurationEventHandler(), pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_Delete)

        #cam.RegisterConfiguration(pylon.ConfigurationEventHandler(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)
        #cam.RegisterImageEventHandler(pylon.ImageEventHandler(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)


        #cam.RegisterImageEventHandler(ImageEventPrinter(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)

        # create event handler instance
        #image_timestamps = TriggeredImage()

        # register handler
        # remove all other handlers
        #cam.RegisterImageEventHandler(image_timestamps, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_None)


    time.sleep(1)
    print("Camera ready!")

    # converting to opencv bgr format  
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

#%%
    # -------------------------------------------------------------
    # Set up a thread to write stuff to disk
    # -------------------------------------------------------------
    if save_in_separate_process:
        im_queue = mp.Queue()
    else:
        im_queue = Queue()


    frames_per_dir=18000
    def save_images_to_disk():
        print('Disk-saving thread active...')

        # Create frame metadata file:
        frame_period = float(1./frame_rate)

        date_fmt = '%Y%m%d-%H%M%S%f'
        datestr = datetime.now().strftime(date_fmt)
        
        #frametimes_fpath = os.path.join(dst_dir, 'frames_%03d.txt' % file_counter)
        #print("Created outfile: %s" % frametimes_fpath) #serial_outfile)
        #serial_file = open(frametimes_fpath, 'w+') #open(serial_outfile, 'w+')
        #serial_file.write('count\tframe\tframe_ID\tframe_tstamp\timage_num\tacq_trigger\tframe_trigger\trelative_time\trelative_camera_time\tcontext\n')

        n = 0
        # intialize save dir
        # Make the output path if it doesn't already exist
        frame_write_dir, frametimes_fpath, performance_fpath = create_dirs(save_dir)

        #frametimes_fpath = os.path.join(dst_dir, 'frames_%03d.txt' % file_counter)
        print("Created outfile: %s" % frametimes_fpath) #serial_outfile)
        serial_file = open(frametimes_fpath, 'w+') #open(serial_outfile, 'w+')
        serial_file.write('count\tframe\tframe_ID\tframe_tstamp\timage_num\tacq_trigger\tframe_trigger\trelative_time\trelative_camera_time\tcontext\n')

        file_counter=0
        prev_counter=0
        curr_n=1
        prev_n=1

        start_time = time.perf_counter() #process_time() #clock() 
      
        # start get loop 
        result = im_queue.get()
        while result is not None: 
            (camID, metadata) = result
            prev_n = copy.copy(curr_n)
            curr_n = metadata['ID']

            if prev_n==curr_n and n>=frames_per_dir:

                #output performance
                nframes = copy.copy(n) #metadata['frame_count']
                acq_duration=time.perf_counter()-start_time
                print('Total Time: %.3f sec' % acq_duration)
                expected_frames=int(np.ceil(float(acq_duration/frame_period)))+1
                print('Actual Frame Count = '+str(nframes+1))
                print('Expected Frame Count = '+str(expected_frames), acq_duration, frame_period)

                # write performance to file
                performance_file = open(performance_fpath,'w+')
                performance_file.write('frame_rate\tframe_period\tacq_duration\tframe_count\texpected_frame_count\tmissing_frames\n')
                performance_file.write('%10.4f\t%10.4f\t%10.4f\t%i\t%i\t%i\n'%\
                    (frame_rate, frame_period, acq_duration, nframes, expected_frames, expected_frames-nframes))
                performance_file.close()
                serial_file.close()

                # start new save dir
                frame_write_dir, frametimes_fpath, performance_fpath = create_dirs(save_dir)
                print("Created outfile: %s" % frametimes_fpath) #serial_outfile)
                serial_file = open(frametimes_fpath, 'w+') #open(serial_outfile, 'w+')
                serial_file.write('count\tframe\tframe_ID\tframe_tstamp\timage_num\tacq_trigger\tframe_trigger\trelative_time\trelative_camera_time\tcontext\n')
               
                prev_counter=copy.copy(file_counter)
                file_counter = file_counter+1
                n=0
                prev_n=0
                curr_n=0

            if n==0:
                print("Restarting clock")
                start_time = time.perf_counter() #process_time() #clock() 
                cam_start_time = metadata['tstamp']

            if prev_n != curr_n: #prev_counter == file_counter:
                n += 1  
                 
#            name = 'frame-%06d_cam%02d' % (int(metadata['ID']), metadata['context_value'])
#            print(name)
#            if save_as_png:
#                fpath = os.path.join(frame_write_dir, '%s.png' % name)
#                #tiff = TIFF.open(fpath, mode='w')
#                #tiff.write_image(im_array)
#                #tiff.close()
#                cv2.imwrite(fpath, im_array)
#            else:
#                fpath = os.path.join(frame_write_dir, '%s.npz' % name)
#                np.savez_compressed(fpath, im_array)

            serial_file.write('\t'.join([str(s) for s in [metadata['frame_count'], n, metadata['ID'], metadata['tstamp'], metadata['number'], metadata['acq_trigger'], metadata['frame_trigger'], str(time.process_time()-start_time), (metadata['tstamp']-cam_start_time)/1E9, str(metadata['context_value'])]])  + '\n')

            result = im_queue.get()

        disk_writer_alive = False 
        print('Disk-saving thread inactive...')
        serial_file.flush()
        serial_file.close()
        print("Closed data file...")

    if save_in_separate_process:
        disk_writer = mp.Process(target=save_images_to_disk)
    else:
        disk_writer = threading.Thread(target=save_images_to_disk)

    if save_images:
        disk_writer.daemon = True
        disk_writer.start()

#%%
    for cam in cameras:
        # Frame triggering  
        cam.LineSelector.SetValue(acquisition_line) 
        sync_line = cam.LineSelector.GetValue()
        sync_state = cam.LineStatus.GetValue()
        print("Waiting for Acquisition Start trigger...", sync_line, sync_state)
        while sync_state is False: 
            #print("[%s] trigger" % sync_line, sync_state)
            sync_state = cam.LineStatus.GetValue()
        print("... ... MW trigger received!")
        cam.AcquisitionStart.Execute()


    # set up timers and counters
    nframes = 0
    t = 0
    last_t = None
    report_interval=1.0
    report_period = frame_rate*report_interval # frames
    timeout_time = 1000  # time (ms) of timeout when retreviging result (timeout time must be longer than exposure time, or else error)
    exp_start_time = time.perf_counter() # start timer in frac. seconds

    if send_trigger:
        #byte_string = str.encode('S')
        #ser.write(str.encode('S')) #('S')#start arduino trigger
        ser.write(b'S%d\r' % frame_rate)

        print('Triggered arduino....')

    time.sleep(1)

    elapsed_time=0
 
    # -------------------------------------------------------------
    # Start acquiring
    # -------------------------------------------------------------
    ser.write(10) #str.encode('10'))

    #cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) #Only) #GrabStrategy_OneByOne)
    cameras.StartGrabbing(pylon.GrabStrategy_OneByOne) #, pylon.GrabLoop_ProvidedByInstantCamera) # first in, first out
    #cam.StartGrabbingMax(100, pylon.GrabStrategy_LatestImages, pylon.GrabLoop_ProvidedByInstantCamera)

    print('Beginning imaging [Hit ESC to quit]...')
    movieA = os.path.join(save_dir, '%sA.mp4' % basename)
    movieB = os.path.join(save_dir, '%sB.mp4' % basename)
    print("OUT: %s | %s" % (movieA, movieB))
    
#    command = ['/bin/ffmpeg',
#               #'-y',
#               '-i', '-', '-an',
#               #'-an',
#               '-c:v', 'h264_nvenc',
#               '-b:v', '1M', '-preset', 'fast', outfile]
               #'-vcodec', 'h264', 
               #'-r', '50',
               #'-f', 'rtsp',
               #'-rtsp_transport',
               #'tcp','rtsp://192.168.1.xxxx:5555/live.sdp']

    #p = subprocess.Popen(command, stdin=subprocess.PIPE) 
    writerA = FFMPEG_VideoWriter(movieA, (cameras[0].Height(), cameras[0].Width()), fps=frame_rate*2, pixfmt='yuv420p') # pixfmt="uyvy422")
    writerB =  FFMPEG_VideoWriter(movieB, (cameras[1].Height(), cameras[1].Width()), fps=frame_rate*2, pixfmt='yuv420p')

#    writerA = imageio.get_writer(
#        movieA,  # mkv players often support H.264
#        fps=20,  # FPS is in units Hz; should be real-time.
#        codec='libx264',  # When used properly, this is basically
#                          # "PNG for video" (i.e. lossless)
#        quality=None,  # disables variable compression
#        #pixelformat='rgb24',  # keep it as RGB colours
#        ffmpeg_params=[  # compatibility with older library versions
#            '-preset',  # set to faster, veryfast, superfast, ultrafast
#            'fast',     # for higher speed but worse compression
#            '-crf',  # quality; set to 0 for lossless, but keep in mind
#            '0'     # that the camera probably adds static anyway
#        ]
#    ) 
#    writerB = imageio.get_writer(
#        movieA,  # mkv players often support H.264
#        fps=20,  # FPS is in units Hz; should be real-time.
#        codec='libx264',  # When used properly, this is basically
#                          # "PNG for video" (i.e. lossless)
#        quality=None,  # disables variable compression
#        #pixelformat='rgb24',  # keep it as RGB colours
#        ffmpeg_params=[  # compatibility with older library versions
#            '-preset',  # set to faster, veryfast, superfast, ultrafast
#            'fast',     # for higher speed but worse compression
#            '-crf',  # quality; set to 0 for lossless, but keep in mind
#            '0'     # that the camera probably adds static anyway
#        ]
#    ) 

    while cameras.IsGrabbing():  # send images as stream until Ctrl-C
        #grabResult = cameras.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)
        grabResult = cameras.RetrieveResult(timeout_time, pylon.TimeoutHandling_ThrowException)
        if grabResult.GetNumberOfSkippedImages():
            print("Skipped ", grabResult.GetNumberOfSkippedImages(), " image.")
 
        cameraContextValue = grabResult.GetCameraContext()
        frame_state = cameras[cameraContextValue].UserOutputValue.GetValue()

        if grabResult.GrabSucceeded():
            image = grabResult.Array
            if cameraContextValue==0:
                writerA.write_frame(image)
                #writerA.append_data(image) 
                #p.stdin.write(image)
                #proc.stdin.write(img_array.tobytes()))
            else:
                writerB.write_frame(image)
                #writerB.append_data(image)
            meta = {'tstamp': grabResult.GetTimeStamp(), #res.TimeStamp, 
                    'ID': grabResult.ID,
                    'number': grabResult.ImageNumber,
                    'frame_count': nframes,
                    'acq_trigger': sync_state,
                    'frame_trigger': frame_state,
                    'context_value': cameraContextValue}
        
            im_queue.put((cameraContextValue, meta))

            #image = resize_compress(image)
            imshow = convert_image(grabResult)
            #p.stdin.write(image.to_bytes())
            nframes += 1

        grabResult.Release() 
        # Show image:
        #cv2.imshow('cam_window', im_array)
        window_name = f'Camera-{cameraContextValue:03}'
        cv2.imshow(window_name, imshow)


#    while True: #cameras.IsGrabbing(): #while cam actually grabbing data
#        #print("loop")
        t = time.time() 
#        #while camera.IsGrabbing():
#        # Grab a frame:
#        #camera.WaitForFrameTriggerReady(100)
#        res = cameras.RetrieveResult(timeout_time, pylon.TimeoutHandling_ThrowException)
#        #res = cam.GrabOne(pylon.waitForever)
#        if res.GetNumberOfSkippedImages():
#            print("Skipped ", res.GetNumberOfSkippedImages(), " image.")
#
#
#        # When the cameras in the array are created the camera context value
#        # is set to the index of the camera in the array.
#        # The camera context is a user settable value.
#        # This value is attached to each grab result and can be used
#        # to determine the camera that produced the grab result.
#        cameraContextValue = res.GetCameraContext()
#        print(res.GetTimeStamp())
#
#        # Print the index and the model name of the camera.
#        #print("Camera ", cameraContextValue, ": ", cameras[cameraContextValue].GetDeviceInfo().GetModelName())
#        sync_state = cameras[cameraContextValue].LineStatus.GetValue()
#        #print("context:", cameraContextValue)
#        if res.GrabSucceeded():
#            # Access img data:
#            im_native = res.Array
#            #print(im_native.shape)
#            im_to_show = converter.Convert(res)
#            im_array = im_to_show.GetArray()
#            frame_state = cameras[cameraContextValue].UserOutputValue.GetValue()
#            meta = {'tstamp': res.GetTimeStamp(), #res.TimeStamp, 
#                    'ID': res.ID,
#                    'number': res.ImageNumber,
#                    'frame_count': nframes,
#                    'acq_trigger': sync_state,
#                    'frame_trigger': frame_state,
#                    'context_value': cameraContextValue}
#            if save_images:
#                im_queue.put((im_native, meta))
#            nframes += 1
#            res.Release()
#
#        # Show image:
#        #cv2.imshow('cam_window', im_array)
#        window_name = f'Camera-{cameraContextValue:03}'
#        cv2.imshow(window_name, im_array)
#
#        #cameras[cameraContextValue].UserOutputValue.SetValue(False)
#
        # Break out of the while loop if ESC registered
        elapsed_time = time.perf_counter() - exp_start_time
        key = cv2.waitKey(1)
        #sync_state = cameras[cameraContextValue].LineStatus.GetValue()
        #if key == 27 or sync_state is False or (elapsed_time>duration_sec): # ESC
        if key == 27 or (elapsed_time>duration_sec): # ESC
            print("Sync:", sync_state)
            #writerA.close()
            #writerB.close()
            break
        grabResult.Release()

        if nframes % report_period == 0:
            if last_t is not None:
                print('avg frame rate: %f [Hit ESC to quit]' % (report_period / (t - last_t)))
                print('ID: %i, nframes: %i, %s' % (meta['ID'], nframes, meta['tstamp']) )
            last_t = t


    #cam.AcquisitionStop.Execute()
    #camera.AcquisitionStart.Execute()

    # Relase the resource:
    for cam in cameras:
        cam.UserOutputValue.SetValue(False) 
        cam.StopGrabbing()
    cameras.Close() 

    cv2.destroyAllWindows()

    if im_queue is not None:
        im_queue.put(None)

    writerA.close()
    writerB.close()

    #ffmpeg -y -r 10 -start_number 971874 -i /mnt/nvme1n1p2/Videos/basler/20220702-002605_acquisition/frames/frame-%06d_cam01.png -c:v h264_nvenc /home/julianarhee/Videos/baslertest2.mp4



#
#
#    print(str.format('total recording time: {} min',(elapsed_time/60)))
#
##%%
#    # Stop arduino
#    #if send_trigger:
#        #ser.write(str.encode('F')) #('S')#start arduino trigger
#        #ser.write('F')#start arduino trigger
#
#        ser.write(b'Q\r')
#        print('Stopped arduino....')
#
#    print('Acquisition Finished!')
#
#    # Save images in buffer
#    print("Saving images, recording dur was: %.2f min" % (elapsed_time/60.)) 
#    if save_images:
#        hang_time = time.time()
#        nag_time = 0.05
#
#        sys.stdout.write('Waiting for disk writer to catch up (this may take a while)...')
#        sys.stdout.flush()
#        waits = 0
#        while not im_queue.empty():
#            now = time.time()
#            if (now - hang_time) > nag_time:
#                sys.stdout.write('.')
#                sys.stdout.flush()
#                hang_time = now
#                waits += 1
#        print(waits)
#        print("\n")
#
#        if not im_queue.empty():
#            print("WARNING: not all images have been saved to disk!")
#
#        disk_writer_alive = False
#        if save_in_separate_process and disk_writer is not None:
#            print("Terminating disk writer...")
#            disk_writer.join()
#            # disk_writer.terminate() 
#        # disk_writer.join()
#        print('Disk writer terminated')        
#
#
#    # Stop arduino
#    if send_trigger:
#        ser.write(str.encode('F')) #('S')#start arduino trigger
#        #ser.write('F'.encode())
#        #serial_file.flush()
#        #serial_file.close()
#        print('Stopped arduino....')
#
#
#
et