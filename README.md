# RadarScenes Tools

## About
![viewer example](https://github.com/oleschum/radar_scenes/blob/master/doc/viewer.png?raw=true)

This python package provides some helper scripts for the [RadarScenes](http://radar-scenes.com) dataset. 

Among others, the package contains a viewer for the radar data and camera images from the dataset.

## Installation

The package is designed for Python versions `>=3.6` and can be installed using `pip`. Installation using `pip` is the
recommended method. 

The alternative is to clone this repository and manually install the package using the `setup.py`.

### Virtual Environment
It is *highly recommended* to install the package in its own virtual environment. To do so, create a virtual environment 
prior to installation of the package:

```
python3 -m venv ~/.virtualenvs/radar_scenes
```
This will create a python virtual environment called `radar_scenes` in the folder `.virtualenvs` in your home directory.

This environment can be activated via
```
source ~/.virtualenvs/radar_scenes/bin/activate
```
An active virtual environment is indicated by a preceding `(radar_scenes)` line before the usual bash prompt.

Once the virtual environment is active, the package can be installed with the command

```
pip install radar_scenes
```
You do not have to clone this repository for the installation.

There are multiple guides available which give more information about virtual environments and installation of python
packages, e.g. on [python.org](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
### System wide installation
If no virtual environment is desired (again, this is __discouraged__!) , the package can be installed using the global version of `pip`
```
pip3 install --user radar_scenes
```
The flag `--user` can be omitted if a system wide installation is desired (may require root privileges).

### Windows Setup Hints
When getting error message  
*"qt.qpa.plugin: Could not load the Qt platform plugin "windows" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem."*  
don't give up. Instead, you should try this:

- set an additional environment variable pointing to your anaconda plugins directory
(e.g. *set QT_PLUGIN_PATH=<anaconda3-directory>\Library\plugins*)
- copy the right dlls into the directory above - hint taken from [LINK](https://stackoverflow.com/questions/41994485/how-to-fix-could-not-find-or-load-the-qt-platform-plugin-windows-while-using-m):
  - copy the following files ...
    - \Anaconda3\Lib\site-packages\PySide2\plugins\platforms\qminimal.dll
    - \Anaconda3\Lib\site-packages\PySide2\plugins\platforms\qoffscreen.dll
    - \Anaconda3\Lib\site-packages\PySide2\plugins\platforms\qwindows.dll  
    to:
    - \Anaconda3\Library\plugins\platforms



## Citation
Please refer to www.radar-scenes.com to get instructions on how to cite the data set. 


## Usage

After successful installation, the `radar_scenes` package is available in your python environment.

### Radar Data Viewer
During installation, the command `rad_viewer` is made available. If you have installed the package into a virtual
environment, this command is only available while the virtual environment is active.

Calling `rad_viewer` launches the radar data viewer. As an optional command line argument, a path to a `*.json` file 
from the RadarScenes dataset can be provided. The sequence will then be loaded directly on start up.

Example:
```
(radar_scenes)
$ rad_viewer ~/datasets/radar_scenes/data/sequence_128/scenes.json
```

The time slider itself or the arrow keys on your keyboard can be used to scroll through the sequence.

## License
This project is licensed under the terms of the MIT license.

Notice, however, that the RadarScenes data set itself comes with a different license.
