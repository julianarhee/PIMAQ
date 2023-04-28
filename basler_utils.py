from pypylon import pylon
import time
import pypylon
import traceback

#def enumerate_connected_devices():
#    """
#    Enumerate the connected Intel RealSense devices
#    Parameters:
#    -----------
#    context           : rs.context()
#                         The context created for using the realsense library
#    Return:
#    -----------
#    connect_device : array
#                       Array of enumerated devices which are connected to the PC
#    """
#    context = rs.context()
#    connect_device = []
#    for d in context.devices:
#        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
#            connect_device.append(d.get_info(rs.camera_info.serial_number))
#    return connect_device
#
def get_serial_number(cam):
    return(
        cam.DeviceInfo.GetSerialNumber())
        
        #.ToString()
   # )


def connect_to_devices(max_cams=2, connect_retries=50):
    """
    Enumerate the connected Basler devices
    Parameters:
    -----------
    context           : rs.context()
                         The context created for using the realsense library
    Return:
    -----------
    connect_device : array
                       Array of enumerated devices which are connected to the PC
    """
    print('Searching for camera...')

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

    for ix, cam in enumerate(cameras):
        cam.Attach(tlFactory.CreateDevice(devices[ix]))
        #camera.Open()
        print("Bound to device: %s" % (cam.GetDeviceInfo().GetModelName()))

    # open camera 
    cameras.Open()
    # store a unique number for each camera to identify the incoming images
    for idx, cam in enumerate(cameras):
        camera_serial = cam.DeviceInfo.GetSerialNumber()
        print(f"set context {idx} for camera {camera_serial}")
        cam.SetCameraContext(idx)

    print("Success!")

    return cameras


def set_camera_properties(cameras, frame_rate=20., acquisition_line='Line4', enable_framerate=False,
                     send_trigger=True,  width=1200, height=1200, exposure=16670):

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
            #cam.AcquisitionMode.SetValue('Continuous')
            cam.AcquisitionStatusSelector="FrameTriggerWait"

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
    # converting to opencv bgr format  
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    #im_native = res.Array
    #print(im_native.shape)
    im_to_show = converter.Convert(grabResult)
    im_array = im_to_show.GetArray()

    return im_array



def set_value(nodemap, nodename, value):
    try:
        node = nodemap.GetNode(nodename)
        nodeval, typestring = get_nodeval_and_type(node)

        if typestring in [pypylon.genicam.IFloat, pypylon.genicam.IInteger]:
            assert(value <= nodeval.GetMax() and value >= nodeval.GetMin())
            if typestring == pypylon.genicam.IInteger:
                assert(type(value)==int)
                if pypylon.genicam.IsAvailable(nodeval) and pypylon.genicam.IsWritable(nodeval):
                    nodeval.SetValue(value)
                else:
                    raise ValueError('Node not writable or available: %s' %nodename)
            elif typestring == pypylon.genicam.IFloat:
                assert(type(value) in [float, int])
                if pypylon.genicam.IsAvailable(nodeval) and pypylon.genicam.IsWritable(nodeval):
                    nodeval.SetValue(float(value))
                else:
                    raise ValueError('Node not writable or available: %s' %nodename)
        elif typestring == pypylon.genicam.IEnumeration:
            assert(type(value)==str)
            entry = nodeval.GetEntryByName(value)
            if entry is None:
                print('Valid entries: ')
                entrylist = nodeval.GetEntries()
                for entry in entrylist:
                    print(entry)
                    #print(entry.GetName())
                raise ValueError('Invalid entry!: %s' %value)
                #else:
                #entry = PySpin.CEnumEntryPtr(entry)
            if pypylon.genicam.IsAvailable(entry) and pypylon.genicam.IsReadable(entry):
                nodeval.SetIntValue(entry.GetValue())
            else:
                raise ValueError('Entry not readable!')
        elif typestring == pypylon.genicam.IBoolean:
            assert(type(value)==bool)
            if pypylon.genicam.IsAvailable(nodeval) and pypylon.genicam.IsWritable(nodeval):
                nodeval.SetValue(value)
            else:
                raise ValueError('Node not writable or available: %s' %nodename)

    except Exception as e:# PySpin.SpinnakerException as e:
        print("ERROR setting:", nodename, value)
        traceback.print_exc()
        raise ValueError('Error: %s' %e)

#        nodetype = node.GetPrincipalInterfaceType()
#
#        nodeval, typestring = get_nodeval_and_type(node)
#
#        assert(PySpin.IsWritable(nodeval), '%s is not writable!' %nodename)
#
#        if typestring == 'int' or typestring == 'float':
#            assert(value <= nodeval.GetMax() and value >= nodeval.GetMin())
#        if typestring == 'int':
#            assert(type(value)==int)
#            if PySpin.IsAvailable(nodeval) and PySpin.IsWritable(nodeval):
#                nodeval.SetValue(value)
#            else:
#                raise ValueError('Node not writable or available: %s' %nodename)
#
#        elif typestring == 'float':
#            assert(type(value)==float)
#            if PySpin.IsAvailable(nodeval) and PySpin.IsWritable(nodeval):
#                nodeval.SetValue(value)
#            else:
#                raise ValueError('Node not writable or available: %s' %nodename)
#        elif typestring == 'enum':
#            assert(type(value)==str)
#
#            entry = nodeval.GetEntryByName(value)
#
#            if entry is None:
#                print('Valid entries: ')
#                entrylist = nodeval.GetEntries()
#                for entry in entrylist:
#                    print(entry.GetName())
#                raise ValueError('Invalid entry!: %s' %value)
#            else:
#                entry = PySpin.CEnumEntryPtr(entry)
#            if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
#                nodeval.SetIntValue(entry.GetValue())
#            else:
#                raise ValueError('Entry not readable!')
#            # PySpin.CEnumEntryPtr
#        elif typestring == 'bool':
#            assert(type(value)==bool)
#            if PySpin.IsAvailable(nodeval) and PySpin.IsWritable(nodeval):
#                nodeval.SetValue(value)
#                
#    raise ValueError('Node not writable or available: %s' %nodename)
#    except Exception as e:# PySpin.SpinnakerException as e:
#        print("ERROR setting:", nodename, value)
#        raise ValueError('Error: %s' %e)
#

def turn_strobe_on(nodemap, line, trigger_selector='FrameStart', line_output=None, line_source='ExposureActive'): # strobe_duration=0.0):
    '''
    # is using external hardware trigger, select line_output to record actual on times (LineSource = 'ExposureActive')
    # check camera model for which lines can be out/in

    # Set  trigger
    # get clean powerup state -- now in self.cleanup_powerup_state()
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
    '''

    assert(type(line)==int)
    #assert(type(strobe_duration)==float)
    
    set_value(nodemap, 'TriggerSelector', trigger_selector)
    set_value(nodemap, 'TriggerMode', 'On')
    set_value(nodemap, 'TriggerSource', 'Line3')

    set_value(nodemap, 'TriggerDelay', 0)
    set_value(nodemap, 'TriggerActivation', 'RisingEdge')
    #set_value(nodemap, 'AcquisitionMode', 'Continuous') # must be continuous for external frame trigger
    set_value(nodemap, 'AcquisitionStatusSelector', 'FrameTriggerWait')
    set_value(nodemap, 'AcquisitionBurstFrameCount', 1)

    # Set trigger source 
    linestr = 'Line%d'%line
    # set the line selector to this line so that we change the following
    # values for Line2, for example, not Line0
    set_value(nodemap, 'LineSelector', linestr)
    # one of input, trigger, strobe, output
    set_value(nodemap, 'LineMode', 'Input') #'strobe')

    # set output
    if line_output is not None:
        linestr_out = 'Line%d' % line_output
        set_value(nodemap, 'LineSelector', linestr_out)
        set_value(nodemap, 'LineMode', 'Output')
        set_value(nodemap, 'LineSource', line_source)

    
def print_value(nodemap, nodename):
    assert(type(nodename)==str)
    node = nodemap.GetNode(nodename)
    nodeval, typestring = get_nodeval_and_type(node)
    if typestring == 'enum':
        # GetCurrentEntry
        print(nodename, typestring, nodeval.ToString())
    else:
        print(nodename, typestring, nodeval.GetValue())

def get_nodeval_and_type(node):

#    nodetype = node.GetPrincipalInterfaceType()
#    if nodetype== PySpin.intfIString:
#        nodeval = PySpin.CStringPtr(node)
#        typestring = 'string'
#    elif nodetype== PySpin.intfIInteger:
#        nodeval = PySpin.CIntegerPtr(node)
#        typestring = 'int'
#    elif nodetype== PySpin.intfIFloat:
#        nodeval = PySpin.CFloatPtr(node)
#        typestring = 'float'
#    elif nodetype== PySpin.intfIBoolean:
#        nodeval = PySpin.CBooleanPtr(node)
#        typestring = 'bool'
#    elif nodetype == PySpin.intfIEnumeration:
#        nodeval = PySpin.CEnumerationPtr(node)
#        typestring = 'enum'
#    elif nodetype == PySpin.intfICommand:
#        nodeval = PySpin.CCommandPtr(node)
#        typestring = 'command'
#    else:
#        raise ValueError('Invalid node type: %s' %nodetype)
#        
#    return(nodeval, typestring)
#
    return(node, type(node))

