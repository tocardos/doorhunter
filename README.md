small web app to monitor camera from home

Requirements & Installation

sudo apt-get install python3-picamera2
sudo apt-get install libcamera-v4l2
sudo apt-get install libcamera-tools
 sudo apt-get install libcamera-apps
sudo apt-get install python3-opencv
sudo apt-get install python3-xmltodict
sudo apt-get install python3-flask-sqlalchemy
sudo apt-get install python3-huawei-lte-api
sudo apt-get install python3-flask-socketio
sudo apt-get install python3-flask-cors
sudo apt-get install ffmpeg

to install dlib without using venv :

sudo apt-get install build-essential cmake
sudo apt-get install libboost-all-dev
 sudo pip3 install dlib --break-system-packages
sudo pip3 install eventlet --break-system-packages

testing yunet : needs opencv4.1 minimum, using venv to avoid breaking distro and keep system package and latest opencv
python3 -m venv --system-site-packages yunet
pip3 install opencv-python
pip install quart

sudo apt-get install git-lfs

sudo apt-get install dbus-x11
export $(dbus-launch)
sudo apt-get install python3-yaml
sudo apt-get install python3-nose
sudo apt-get install python3-metaconfig

