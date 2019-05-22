# panopticon
Acquisition and processing for multiple realsense cameras for mouse behavior. 

# installing PySpin
* Go to the Spinnaker downloads page at FLIR (formerly PointGrey) [here.](https://www.flir.com/products/spinnaker-sdk/)
* Click Download Now. Note: I had to disable AdBlock, etc. to get the website to show the download button.
* Click your OS (e.g. Windows)
* Click Latest Spinnaker Web Installer. Download the x64 version and install. Note the version!!
  * You might be able to get away with installing only "Visual Studio runtimes and drivers."
* Go back one page and click on the "Latest Python Spinnaker"
  * Note: a full version of these instructions can be found in the README in this zip file!
  * Download the .zip file corresponding to your python installation and OS version. E.g., my python version is 3.6 and my OS is 64 bit, so I downloaded `spinnaker_python-1.20.0.15-cp36-cp36m-win_amd64`
    * The `cp36` means python 3.6, and `amd64` means 64 bit.
* Unzip this file
* `cd` into this file location. There should be a file like `spinnaker_python-1.23.0.27-cp36-cp36m-win_amd64.whl`
* activate your anaconda or pip environment!!!
* run `python -m ensurepip` to ensure that pip is installed
* run `python -m pip install --upgrade pip numpy`. This makes sure that `numpy` is installed
* Finally, to install PySpin, run `python -m pip install spinnaker_python-1.x.x.x-cp36-cp36m-win_amd64.whl`, with of course the correct filename for your version.
* To verify installation, make sure your point grey is connected and run `python Examples\Python3\Acquisition.py`.
* Potential bugs
  * It's very important to run the `Acquistion.py` example to catch any potential bugs.
  * I encountered a strange bug relating to my `anaconda` numpy version. It said something about `MKL_DNN` or something.
    * If this happens to you, run `python -m pip install --upgrade --force-reinstall spinnaker_python-1.20.0.15-cp36-cp36m-win_amd64.whl` (with the correct version). This will reinstall the correct numpy version.
