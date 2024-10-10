#!/usr/bin/env python3

import asyncio
from quart import Quart, websocket, render_template_string,render_template,send_from_directory
from quart import request, redirect, url_for, flash,jsonify
from sqlalchemy import select

from picamera2 import Picamera2,MappedArray
from picamera2.encoders import MJPEGEncoder,H264Encoder,Quality
from picamera2.outputs import FileOutput,CircularOutput
from queue import Queue
from threading import Thread

import io
import cv2
import dlib
import numpy as np
import time
from datetime import datetime
import psutil
import json
import os
import logging
#from yunet import YuNet
# import from local definitions
from config import init_db, get_db, DatabaseOperations, Contact
from config import dongle, connection_status_processor
from config import wifi_con,connection_status_wifi,get_hotspot_status
from functools import wraps
# import for async functions
import threading
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HOTSPOT_UUID = "fa1185cf-e8a2-4ad2-9b0e-bc1704f71b36"

app = Quart(__name__)
app.secret_key = 'your-secret-key'  # needed for flash messages
# add the test for 4G connection
app.context_processor(connection_status_processor)
app.context_processor(connection_status_wifi)
# for async functions in main thread
executor = ThreadPoolExecutor()

# Global variable to store the Monica instance for multi access
Monica = None
processing_face_det = 'Yunet'  # Initial processing mode for facedetect
processing_state = True  # Initial processing state
min_area = 500 # minimum area for knn detection
# value is based on frame of 320*240= 76,800 pixels
# full body is about 53*80 pixels = 4240 pixels ( area)
# partial body detection is lower than 4240, so on the average 750

# To hold the current state of the sliders
slider_state = {
    'motion': 2,          # Default to "No Detection"
    'face_detect': 2,         # Default to "No Detection"
    'frameRatio': 2    # Default to "Frame Ratio 1"
}
face_detect_mapping = {
    0: 'Yunet',
    1: 'no processing',
    2: 'ssd',
    3: 'hog'
}
motion_detect_mapping = {
    0: 'mog',
    1: 'no processing',
    2: 'blurred',
    3: 'knn'
}
connected_clients = []

framerate = 25 # 25 frames per second, more than enough
preroll = 3 # preroll time of video recording prior of detection
lores_size = (640,480)
main_size  = (1024,768)
decimation_factor =2 # lores downsizes by this facto to spare cpu

# trying to add an overlay, this is done on the main output and not before the encoder
overlay = np.zeros((300, 400, 4), dtype=np.uint8)
overlay[:150, 200:] = (255, 0, 0, 64)
overlay[150:, :200] = (0, 255, 0, 64)
overlay[150:, 200:] = (0, 0, 255, 64)

# method to incrust text into GPU before encoding
# to be done adding frame around face detection
colour = (123, 255, 0) # light gray
origin = (0, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 2
cpul= 0
detection_queue = Queue()  # Queue for signaling face detection results
motion_queue = Queue() # Queue for signaling motion detection
motion_queue_mog = Queue() # queue for mog rectangle detection
motion_queue_knn = Queue() # queue for knn rectangle detection
motion_queue_blur = Queue() # queue for motion blurre detection
detection_queue_hog = Queue() # queue for hog rectangle detection
face_detect_threshold = 0.80
#faces = []
#scores = []
# Global variable to track the number of active WebSocket connections
active_connections = 0





# io buffer used for streaming
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.queue = Queue()

    def write(self, buf):
        self.frame = buf
        self.queue.put(buf)

frame_queue = Queue()

class VideoProcessor:
    def __init__(self,app_loop,recording_dir='recordings'):

        self.app_loop = app_loop
        self.motion_detected = 0
        self._shutdown = threading.Event()
        self.face_detected = 0
        self.recording = False
        self.video_writer = None
        
        self.face_start_time = None
        self.ltime = None
        
        self.latest_frame= None
        self.processing = True
        self.running = True

        self.recording_dir = recording_dir
		# Get the current directory where the script is located
        self.current_directory = os.path.dirname(__file__)
        self.filename = None
        # Define the relative path to the subfolder
        self.relative_path = os.path.join(self.current_directory, self.recording_dir)


        # debug purpose
        self.fps = 0
        self.cpu_load=0
        # Ensure the directory exists, or create it
        os.makedirs(self.relative_path, exist_ok=True)
        self.picam2 = Picamera2()
        mode=self.picam2.sensor_modes[1]

        video_config = self.picam2.create_video_configuration(
            sensor={"output_size":mode['size'],'bit_depth':mode['bit_depth']},
            main={"size": main_size, "format": "YUV420"},
            lores={"size": lores_size, "format": "RGB888"},
            #controls={"FrameDurationLimits": (40000, 40000)},
            controls={'FrameRate': framerate}
        )
        self.picam2.configure(video_config)
        self.angle = 0
        self.output = StreamingOutput()
        # overlay timestamp in gpu
        #self.picam2.pre_callback = self.apply_timestamp
        # doing the overlay after processing is better to avoid impacting detection
        self.picam2.post_callback = self.apply_timestamp
        # setting gpu encoder
        self.encoder = H264Encoder() # by default quality is set to medium
        # use circular output to have a preroll of 1 second
        self.circ=CircularOutput(buffersize=framerate*preroll)
        # write encoder output to circ buffer
        #encoder.output=[circ]
        #encoder.name = "main"
        #picam2.encoders = encoder 
        self.picam2.start_encoder(self.encoder,self.circ,name="main")
        self.picam2.start_recording(MJPEGEncoder(), FileOutput(self.output), name="lores")

        # Load face detection model
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_detect_threshold = 0.80 # level above considered as good detection
        #self.face_detected = 0
        self.process_every_n_frames = 5
        self.frame_count = 0
        
        self.current_faces = []  # Store the last detected faces

        # Parameters for motion detection
        self.prev_frame = None
        self.motion_threshold = 30 # minimum threshold for frame comparison
        self.motion_maxval = 255 # max value for frame comparison
        # drawback when using gpu incrust, it plays a role in the motion detection, as the 
        # incrusted frame is compared to previous one
        self.motion_detect_level = 800 # level above which we consider motion detected, rule of thumb
        # parameter for mog motion detection
        
        
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500,
                                                          varThreshold=16, # typical value 8-25
                                                          detectShadows=False)
        # Create KNN background subtractor
        self.mog_threshold = 180
        self.knn_threshold = 180
        self.knn = cv2.createBackgroundSubtractorKNN(
                   history=500,
                   dist2Threshold=400, # default is 400 ( distance )
                   detectShadows=False
                ) 
        # Initialize the HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # so far knn seems to be faster with about 2ms per frame while mog is about 4ms
        # in order to decrease cpu load we will try working with half size frame
        self.lores_half_size = (lores_size[0]//decimation_factor,lores_size[1]//decimation_factor)
        # Load Yunet face detection model
        self.face_detector_yunet = cv2.FaceDetectorYN.create(
            model='/home/baddemanax/opencv_zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
            config='',
            input_size=self.lores_half_size,
            #score_threshold=0.9,
            #nms_threshold=0.3,
            #top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
            #backend_id=cv2.dnn.DNN_TARGET_CPU,
            #target_id=cv2.dnn.DNN_BACKEND_OPENCV
        )
        # Check for GPU support
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            print("OpenCL acceleration enabled")
        # Load the MobileNet SSD model
        self.net = cv2.dnn.readNetFromCaffe(
            'poc3/static/MobileNetSSD_deploy.prototxt',
            'poc3/static/MobileNetSSD_deploy.caffemodel'
        )
        # Attempt to optimize the model for OpenCV DNN backend
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # use lower quantisation 
        if hasattr(cv2.dnn, 'DNN_TARGET_CPU_FP16'):
           self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU_FP16)
        # Initialize class labels
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        
        # Performance settings
        self.conf_threshold = 0.5
        self.resolution = self.lores_half_size
        '''
        self.toggle_dlib= False
        self.toggle_yunet = True
        self.toggle_cv = False
        '''
        self.toggle_facedetect = face_detect_mapping[slider_state['face_detect']]
        self.toggle_motion = motion_detect_mapping[slider_state['motion']]

    #- - - - - - - - - - - 
    # ROTATE
    #- - - - - - - - - - -   
    def rotate(self,angle):
        self.angle= angle
    #- - - - - - - - - - -
    # TIMESTAMP and DETECTIONBOX
    #- - - - - - - - - - -
    def apply_timestamp(self,request):
        #global cpul
        #global faces,scores
        timestamp = time.strftime("%Y-%m-%d %X")
        # Get CPU load (percentage) over all cores
        #cpul = round((cpul + psutil.cpu_percent(interval=0))/2)
        #cpu_load = f" cpu load : {cpul}"  # interval=1 means it waits for 1 second

        with MappedArray(request, "lores") as m:
            cv2.putText(m.array, timestamp, origin, font, scale, colour, thickness)
            #cv2.putText(m.array, (cpu_load), (0,60), font, scale, colour, thickness)
        
            # check for motion detection results from queue
            if not motion_queue.empty():
                width,heigth = motion_queue.get()
                radius = 20
                cv2.circle(m.array, (width*decimation_factor - round(radius/2), radius), radius, (0, 0, 255), -1)
            if not motion_queue_mog.empty():
                large_contours = motion_queue_mog.get()
                for cnt in large_contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # we include decimetion factor as it is computed on smaller frame
                    x, y, w, h = x*decimation_factor,y*decimation_factor,w*decimation_factor,h*decimation_factor
                    cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 0, 200), 3)
            if not motion_queue_knn.empty():
                contours = motion_queue_knn.get()
                # Draw rectangles around large enough contours
                for contour in contours:
                    if cv2.contourArea(contour) > min_area:
                        (x, y, w, h) = cv2.boundingRect(contour)
                        # we include decimetion factor as it is computed on smaller frame
                        x, y, w, h = x*decimation_factor,y*decimation_factor,w*decimation_factor,h*decimation_factor
                        cv2.rectangle(m.array, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if not motion_queue_blur.empty():
                (x, y, w, h) = motion_queue_blur.get()
                #for contour in contours:
                #(x, y, w, h) = cv2.boundingRect(contours)
                # we include decimetion factor as it is computed on smaller frame
                x, y, w, h = x*decimation_factor,y*decimation_factor,w*decimation_factor,h*decimation_factor
                cv2.rectangle(m.array, (x, y), (x+w, y+h), (0, 255, 0), 2)


            # Check for face detection results from  queue
            if not detection_queue.empty():
                #print("face queue not empty")
                faces,scores = detection_queue.get()
                if self.toggle_facedetect == 'DLIB':

                    for i,face in enumerate(faces):
                        # face detected once above threshold to avoid bad detection
                    
                        if (scores[i]>face_detect_threshold):
                            #face_detected=1
                            print(f"face detected {scores[i]}")
                        
                            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                            x,y,w,h = x*decimation_factor,y*decimation_factor,w*decimation_factor,h*decimation_factor
                            cv2.rectangle(m.array, (x, y), (x + w, y + h), (255, 0, 0), 3)
                
                elif self.toggle_facedetect =='Yunet':
                        if (scores>face_detect_threshold):
                            print(f"face detected {scores}")
                            box = list(map(int, faces[:4]))
                            box = [value * decimation_factor for value in box]

                            cv2.rectangle(m.array, box, (255, 0, 0), 3)
            if not detection_queue_hog.empty():
                pick = detection_queue_hog.get()
                for (x1, y1, x2, y2) in pick:
                    x1,y1,x2,y2 = x1*decimation_factor,y1*decimation_factor,x2*decimation_factor,y2*decimation_factor
                    cv2.rectangle(m.array, (x1, y1), (x2, y2), (255, 0, 0), 2)

    #- - - - - - - - - - - - - - -
    # MOTION DETECTION KNN
    #- - - - - - - - - - - - - - -
    def motion_detect_knn(self,frame):
            
            tm= cv2.TickMeter()
            tm.start()
            self.motion_detected = 0
            fg_mask = self.knn.apply(frame)
            # Apply threshold to the mask
            # threshold value should be set as a parameter
            _, thresh = cv2.threshold(fg_mask, self.knn_threshold, 255, cv2.THRESH_BINARY)
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # put detection value in queue for ploting rectangle
            motion_queue_knn.put(contours)
            

            for contour in contours:
                if cv2.contourArea(contour) > min_area:
                    self.motion_detected =1
                    #(x, y, w, h) = cv2.boundingRect(contour)
                    #images.append(frame[y:y+h,x:x+w])

            images = None
            if len(contours)>=1 & self.motion_detected ==1:
                # Combine all contours into one array
                all_points = np.vstack(contours)
                # Find bounding box for all points
                x, y, w, h = cv2.boundingRect(all_points)
                images=(x,y,w,h)       

            tm.stop()
            #print(f"time for knn detection {round(tm.getTimeMilli(),2)} ms")
            tm.reset()
            return images
     #- - - - - - - - - - - - - - - - - 
     #  MOTION DETECTION MOG
     #- - - - - - - - - - - - - - - - -       
    def motion_detect_mog(self,frame):
        #self.frame_count += 1
        #if self.frame_count % self.process_every_n_frames == 0:
        #    self.frame_count=0
            tm= cv2.TickMeter()
            tm.start()
            # Apply background subtraction
            fg_mask = self.backSub.apply(frame)
            # apply global threshol to remove shadows
            retval, mask_thresh = cv2.threshold(fg_mask, self.mog_threshold, 255, cv2.THRESH_BINARY)
            # set the kernal
            # we can go to 5,5 if there is too much noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # Apply erosion to remove some noise
            mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_contour_area = min_area  # Define your minimum area threshold for medium range
            # the higher the closer the detection
            # filtering contours using list comprehension
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
            
            motion_queue_mog.put(large_contours)
            # in ordre to decrease prcessing we only search people in motion detected zone
            images = None
            if large_contours:
                
                all_points = np.vstack(contours)
                # Find bounding box for all points
                x, y, w, h = cv2.boundingRect(all_points)
                if w*h>min_area:
                    self.motion_detected=1
                    #images = frame[y:y+h,x:x+w]
                    images = (x,y,w,h)
            else:
                self.motion_detected=0
            
            
                    

            #for cnt in large_contours:
            #    x, y, w, h = cv2.boundingRect(cnt)
            #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 3)
            tm.stop()
            #print(f"time for mog detection {round(tm.getTimeMilli(),2)} ms")

            tm.reset()
            return images

    #- - - - - - - - - - - - - - -
    # FRAME DIFF DETECTION
    #- - - - - - - - - - - - - - -
    def motion_detect(self,frame):
        
        # Motion detection
        # to decrease computing load, processing is done every other frame
        # there is a preroll for the clip recording to compensate low latency
        #self.frame_count += 1
        #if self.frame_count % self.process_every_n_frames == 0:
        #    self.frame_count=0
            tm= cv2.TickMeter()
            if self.prev_frame is not None:
                tm.start()
                frame_delta = cv2.absdiff(self.prev_frame, frame)
                _, thresh = cv2.threshold(frame_delta, self.motion_threshold, self.motion_maxval, cv2.THRESH_BINARY)
                self.motion_detected = np.sum(thresh)/self.motion_maxval > self.motion_detect_level

            self.prev_frame = frame
            tm.stop()
            #print(f"time to detect motion : {round(tm.getTimeMilli(),2)} ms")
            tm.reset()
            # Draw motion and face detection results
            if self.motion_detected:
                print(f"motion detected {np.sum(thresh)}")
                height, width = frame.shape
                # drawing is moved to camera array
                #radius = 15
                #cv2.putText(frame, "Motion Detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #cv2.circle(frame, (width - radius, radius), radius, (0, 0, 255), -1)
                motion_queue.put((height, width))
            return frame
    #- - - - - - - - - - - - - - - - - - - -
    # FRAME DIFF BLURED
    #- - - - - - - - - - - - - - - - - - - -
    def motion_detect_blur(self,frame):
        # code based from pyimagesearch
        tm=cv2.TickMeter()
        
        tm.start()
        # reset motion flag
        self.motion_detected=0
        gray = cv2.GaussianBlur(frame, (21, 21), 0)
        # if the first frame is None, initialize it
        if self.prev_frame is None:
            self.prev_frame = gray.copy().astype("float")
            #continue
        # accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(gray, self.prev_frame, 0.5)
        # compute the absolute difference between the current frame and
        # first frame
        #frameDelta = cv2.absdiff(self.prev_frame, gray)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.prev_frame))
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		    cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1]
        
        for c in contours:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < min_area:
                
                continue
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            self.motion_detected=1
            (x, y, w, h) = cv2.boundingRect(c)
            motion_queue_blur.put((x,y,w,h))
        
        images = None
        if len(contours)>=1 & self.motion_detected ==1:
            # Combine all contours into one array
            all_points = np.vstack(contours)
            # Find bounding box for all points
            x, y, w, h = cv2.boundingRect(all_points)
            images=(x,y,w,h)
            #print(f"image size: {images}")   
        tm.stop()
        #print(f"time to detect motion blurred: {round(tm.getTimeMilli(),2)} ms")
        tm.reset()
        return images
    #- - - - - - - - - - - - - - - - - - - -           
    # FACE DETECTION YUNET
    #- - - - - - - - - - - - - - - - - - - - 
    def process_frame_yunet (self,frame):

         # Face detection using Yunet
        #self.frame_count += 1
        #if self.frame_count % self.process_every_n_frames == 0:
            #self.face_detected=0
            #self.frame_count = 0  # reset number to avoid overflow
            tm=cv2.TickMeter()
            # Detect faces using Yunet
            tm.start()
            _, faces = self.face_detector_yunet.detect(frame)
            # next line is to avoid none type is not iterable
            faces = faces if faces is not None else []
            tm.stop()
            print(f"yunet time :{round(tm.getTimeMilli(),2)} ms")
            tm.reset()
            # Draw rectangles around detected faces
            
            for face in faces:
                
                fc = list(map(int, face[:4]))
                score = face[-1]
                detection_queue.put((fc,score))
                if (score>self.face_detect_threshold):
                    self.face_detected=1
                    print('face detected')
                    # Store faces in a shared state to use in other parts of the app
                    self.current_faces = fc
                
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #- - - - - - - - - - - - - - - - - - - - - -
    # FACE DETECTION DLIB
    #- - - - - - - - - - - - - - - - - - - - - -
    def process_frame_dlib(self, frame):
   
        # face detection using dlib
        # to decrease computing load, processing is done every other frame
        # there is a preroll for the clip recording to compensate low latency
        #self.frame_count += 1
        #if self.frame_count % self.process_every_n_frames == 0:
        #    self.face_detected=0
            tm = cv2.TickMeter()
            # Detect faces using dlib
            self.frame_count = 0 # reset number to avoid overflow
            #faces = self.face_detector(frame)
            tm.start()
            faces,scores,idx = self.face_detector.run(frame,1,-1)
            tm.stop()
            print(f"dlib time :{round(tm.getTimeMilli(),2)} ms")
            tm.reset()
            #print(f"scores {scores}")
            # Signal detected faces (store face positions)
            detection_queue.put((faces,scores))
            # Draw rectangles around detected faces
            # moved to camera array avoid re generating frame in cpu
            
            for i,face in enumerate(faces):
                if (scores[i]>self.face_detect_threshold):
                    self.face_detected=1
                    #print(f"face detected {scores[i]}")
                    # Signal detected faces (store face positions)
                    #detection_queue.put(face)
                    # Store faces in a shared state to use in other parts of the app
                    self.current_faces = face
                    
                    #x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                    #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
    #- - - - - - - - - - - - - - - - - - - - - -
    # PERSON DETECTIOn HOG
    #- - - - - - - - - - - - - - - - - - - - - -  
    def process_frame_hog(self,frame):
        tm = cv2.TickMeter()
        tm.start()
        # Detect people in the image
        # hog is trained on 64*128 pixel window
        # winstride 4 more precise but slow, 6 compromise, 8 fast
        # padding 4 is minimum for small frame
        # scale 1.05 default
        (rects, weights) = self.hog.detectMultiScale(frame, winStride=(8, 8),
                                            padding=(8, 8), scale=1.02)
        # Convert the rectangles to the format expected by non_max_suppression
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        # Apply non-maxima suppression to the bounding boxes
        pick = self.non_max_suppression(rects, overlapThresh=0.65)
        detection_queue_hog.put(pick)
        if len(pick)>0:
            self.face_detected=1
        else:
            self.face_detected=0
		# Draw the final bounding boxes
        #for (x1, y1, x2, y2) in pick:
        #    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tm.stop()
        print(f"time for hog {round(tm.getTimeMilli(),2)} ms")
        tm.reset()
        
    def non_max_suppression(self,boxes, overlapThresh):
        if len(boxes) == 0:
            print("empty box for hog detection")
            return []

        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")

    #- - - - - - - - - - - - - - - - - - - - - - - - -
    # MOBILE SSD
    #- - - - - - - - - - - - - - - - - - - - - - - - - 
    def process_frame_ssd(self,frame):

        tm = cv2.TickMeter()
        tm.start()
        # Get frame dimensions
        (h, w) = frame.shape[:2]
        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        # Pass the blob through the network
        self.net.setInput(blob)
        detections = self.net.forward()
        
        persons = []
        
        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.conf_threshold:
                class_id = int(detections[0, 0, i, 1])
                
                if self.classes[class_id] == "person":
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    persons.append({
                        'bbox': (startX, startY, endX, endY),
                        'confidence': confidence
                    })
                    self.face_detected=1
                    print(f"person detected with confidence {confidence}")
            
        tm.stop()
        print(f"time for ssd {round(tm.getTimeMilli(),2)} ms")
        tm.reset()

    #- - - - - - - - - - - - - - - - - - - - - - - - -
    # CAMERA THREAD
    #- - - - - - - - - - - - - - - - - - - - - - - - -
    def camera_thread(self):
    
        frame_count=0
        prev_time=0
        prev = None # previous frame for np motion detection
        try:
            while not self._shutdown.is_set():
                frame = self.output.queue.get()

                frame_count +=1
                curr_time = time.time()
                # 
                if self.processing:
                    # to spare cpu processing, detection only every other frame
                    if frame_count % self.process_every_n_frames == 0:
                        # Decode the frame (as it is in a compressed format like MJPEG)
                        frame_array = np.frombuffer(frame, dtype=np.uint8)
                        if self.toggle_motion == "numpy":
                            if prev is not None:
                                 mse = np.square(np.subtract(frame_array, prev)).mean()
                                 if mse>7:
                                     print(f"np detection with level {mse}")
                                     self.motion_detected = 1
                                 else:
                                    self.motion_detected = 0
                            prev = frame_array
                        
                        
                        image = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                        # Resize the frame to half of lores size
                        image = cv2.resize(image, self.lores_half_size, interpolation=cv2.INTER_AREA)
                        # Convert the frame to grayscale as processing works on grayscale images
                        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        # Run motion detection
                        if self.toggle_motion == 'diff':
                            small_frame = self.motion_detect(gray_frame)
                        elif self.toggle_motion == 'mog':
                            small_frame = self.motion_detect_mog(gray_frame)
                        elif self.toggle_motion == 'knn':
                            small_frame = self.motion_detect_knn(gray_frame)
                        elif self.toggle_motion == 'blurred':
                            small_frame=self.motion_detect_blur(gray_frame)
	
                        # Run face detection on the frame if motion detected
                        self.face_detected=0 # reset detector flag every frame
                        if self.motion_detected:
                            if self.toggle_facedetect == 'DLIB':
                                self.process_frame_dlib(gray_frame)
                            elif self.toggle_facedetect == 'Yunet':
                                # yunet cannot work with grayscale
                                self.process_frame_yunet(image)
                            elif self.toggle_facedetect == 'hog':
                                
                                    self.process_frame_hog(gray_frame)
                            elif self.toggle_facedetect == 'ssd':
                                if small_frame is not None:
                                    (x,y,w,h)= small_frame
                                    self.process_frame_ssd(image[y:y+h,x:x+w])
                                else:
                                    self.process_frame_ssd(image)
                        # records clip 
                        self.handle_recording()
                        # adding a check  to avoid cpu dma useless
                    # 

                if curr_time - prev_time >= 1:
                    self.fps = frame_count / (curr_time - prev_time)
                    frame_count = 0
                    prev_time = curr_time
                    #print(f"frame per seconds {round(self.fps)}")
                    # Calculate CPU load
                    self.cpu_load = psutil.cpu_percent()
                    #print(f"cpu load {self.cpu_load}")

                if active_connections > 0: 
                    
                    if frame_queue.qsize() < 10:  # Limit queue size to prevent memory issues
                        frame_queue.put(frame)
                    else:
                        print("too many frames in queue")
                        frame_queue.get()  # Remove oldest frame
                        frame_queue.put(frame)
                #time.sleep(0.03)
        finally:
            self.picam2.stop_recording()
    
    def handle_recording(self):
        #print(f"handle recording {self.face_detected}")
        if self.face_detected:
            if not self.recording:
                self.recording = True
                # Get current date and time
                current_time = datetime.now()

                # Format the timestamp as YYMMDD_HHMM
                formatted_time = current_time.strftime("%y%m%d_%H%M%S")
                print(f"time {formatted_time}")
                self.filename = os.path.join(self.relative_path, f"detection_{formatted_time}")
                self.circ.fileoutput=self.filename
                self.circ.start()
                
            self.ltime = time.time()  # Set ltime when recording starts and continue until
            # no face detection
            #self.face_detected=0

        if self.recording and self.ltime is not None and time.time() - self.ltime > 8:
            self.recording = False
            #camera.stop_encoder()
            self.circ.stop()
            self.video_writer = None
            self.ltime = None  # Reset ltime when recording stops
            print("Saving file",self.filename)
            self.face_detected=0 # reset flag in case of processing every other frame
            print()
            print("Preparing an mp4 version")
            # convert file to mp4
            cmd = 'ffmpeg -nostats -loglevel 0 -r 30 -i ' + self.filename + ' -c copy ' + self.filename +'.mp4'
            os.system(cmd)
            # delete tmp file
            cmd ='rm ' + self.filename 
            os.system(cmd)
            # we need to await an async function
            #self.handle_sms(self.filename+".mp4")
            # if we are not in an async context we need a synchronous wrapper
            #result = self.sync_handle_sms(f"{self.filename}.mp4")
            asyncio.run_coroutine_threadsafe(
                        self.handle_sms(f"{self.filename}.mp4"), 
                        self.app_loop
                    ).result()
            asyncio.run_coroutine_threadsafe(
                        self.save_recording_metadata(f"{self.filename}.mp4"), 
                        self.app_loop
                    ).result()
            print("Removing h264 version")
            print("Waiting for next trigger")

    async def handle_sms(self, filename):
        msg = f"Motion Detected check the file: {filename}"
        try:
            async for db in get_db():
                success_count, total_contacts = await DatabaseOperations.send_sms_to_contacts(db, msg)
                logger.info(f"SMS sent successfully: {success_count}/{total_contacts}")
                return success_count, total_contacts
        except Exception as e:
            logger.error(f"Error sending SMS: {str(e)}")
            return 0, 0
    async def save_recording_metadata(self,filename):
        try:
            async for db in get_db():
                success = await DatabaseOperations.add_recording(db,filename)
                logger.info(f"success in writing recording metadata {success}")
        except Exception as e:
            logger.error(f"Error while writing recording metadata {e}")

    def shutdown(self):
        self._shutdown.set()
#------------------------------------------------------------------------------
#
#   INDEX
#
#------------------------------------------------------------------------------
@app.route('/')
async def index():
    status = await dongle.get_connection_status()
    #wifi_status = await wifi_con.get_active_connection_by_uuid(HOTSPOT_UUID)
    #print(wifi_status)
    #return await render_template('index.html', connection_status=status)
    return await render_template('index.html')



@app.route('/status')
async def status():
    #return await render_template_string(HTML_TEMPLATE)
    return await render_template('status.html')

#------------------------------------------------------------------------------
#
#   SETTINGS
#
#------------------------------------------------------------------------------

@app.route('/settings')
async def settings():
    #return await render_template_string(HTML_TEMPLATE)
    return await render_template('settings.html')

@app.route('/rotate_camera', methods=['POST'])
async def rotate_camera():
    form = await request.form
    angle = form.get('angle',0) 
    Monica.rotate(int(angle))
    return redirect(url_for('settings'))
#**********************
#
#   WIFI
#
#**********************
def error_handler(f):
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        try:
            return await f(*args, **kwargs)
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    return decorated_function

@app.route('/status_wifi')
@error_handler
async def status_wifi():
    is_active =  get_hotspot_status()
    #status = connection_status_wifi()
    #print(f"is active {is_active}")
    return jsonify({'status': 'active' if is_active else 'inactive'})

@app.route('/toggle_wifi/<action>')
@error_handler
async def toggle_wifi(action):
    if action not in ['on', 'off']:
        return jsonify({'error': 'Invalid action'}), 400
    
    connection = wifi_con.get_connection_by_uuid(HOTSPOT_UUID)
    if not connection:
        return jsonify({'error': f'Connection with UUID {HOTSPOT_UUID} not found'}), 404

    if action == 'on':
        # Activate the hotspot connection
        wifi_con.set_active_connection(connection)
        message = "Hotspot turned on"
    else:
        # Deactivate the hotspot connection
        active_conn = wifi_con.get_active_connection_by_uuid(HOTSPOT_UUID)
        if active_conn:
            active_conn.Delete()
        message = "Hotspot turned off"
    
    return jsonify({'success': True, 'message': message})

#--------------------------------------------------------------
#
#       CONTACT
#
#--------------------------------------------------------------

@app.route('/contact')
async def contact():
    contacts = []
    async for db in get_db():
        result = await db.execute(select(Contact))
        contacts = result.scalars().all()
    
    return await render_template('contact.html', contacts=contacts)

# Route to display the Add Contact page
@app.route('/add_contact')
async def add_contact():
    return await render_template('add_contact.html')
@app.route('/create_contact', methods=['POST'])

async def create_contact():
    form = await request.form
    contact_data = {
        'alias': form['alias'],
        'first_name': form['first_name'],
        'last_name': form['last_name'],
        'email': form['email'],
        'phone': form['phone']
    }

    async for db in get_db():
        try:
            await DatabaseOperations.add_contact(db, contact_data)
            await flash('Contact added successfully!', 'success')
        except Exception as e:
            await flash(f'Error adding contact: {str(e)}', 'error')
    
    return redirect(url_for('contact'))

@app.route('/update_contact/<int:id>', methods=['POST'])
async def update_contact(id):
    form = await request.form
    async for db in get_db():
        contact = await DatabaseOperations.get_contact_by_id(db, id)
        if contact:
            mail_send = 'mail_send' in form
            sms_send = 'sms_send' in form
            await DatabaseOperations.update_contact(db, contact, mail_send=mail_send, sms_send=sms_send)
    return redirect(url_for('contact'))

@app.route('/update_contact_checkbox', methods=['POST'])
async def update_contact_checkbox():
    form = await request.form
    contact_id = int(form.get('id'))
    field = form.get('field')
    value = form.get('value')

    async for db in get_db():
        contact = await DatabaseOperations.get_contact_by_id(db, contact_id)
        if contact:
            update_data = {
                field: True if value == '1' else False
            }
            await DatabaseOperations.update_contact(db, contact, **update_data)

    return redirect(url_for('contact'))

@app.route('/edit_contact/<int:id>', methods=['GET', 'POST'])
async def edit_contact(id):
    async for db in get_db():
        contact = await DatabaseOperations.get_contact_by_id(db, id)
        if not contact:
            await flash('Contact not found', 'error')
            return redirect(url_for('contact'))

        if request.method == 'POST':
            form = await request.form
            update_data = {
                'first_name': form.get('first_name'),
                'last_name': form.get('last_name'),
                'email': form.get('email'),
                'phone': form.get('phone')
            }
            await DatabaseOperations.update_contact(db, contact, **update_data)
            return redirect(url_for('contact'))

        return await render_template('edit_contact.html', contact=contact)


@app.route('/delete_contact/<int:id>', methods=['POST'])
async def delete_contact(id):
    async for db in get_db():
        contact = await DatabaseOperations.get_contact_by_id(db, id)
        if contact:
            try:
                await DatabaseOperations.delete_contact(db, contact)
                await flash('Contact deleted successfully!', 'success')
            except Exception as e:
                await flash(f'Error deleting contact: {str(e)}', 'error')
    return redirect(url_for('contact'))



# for android phone, they point to index.html
@app.route('/index.html')
async def index_index():
    #return await render_template_string(HTML_TEMPLATE)
    return await render_template('index.html')

#---------------------------------------------------------------------
#
#     WEBSOCKET
#
#---------------------------------------------------------------------
@app.websocket('/ws')
async def ws():
    global active_connections
    active_connections += 1

    loop = asyncio.get_event_loop()  # Get the current event loop
    try:
        while True:
        #while True:
            frame = await loop.run_in_executor(None, frame_queue.get)
            await websocket.send(frame)
           

    finally:
        print('DEBUG : one connexion less')
        active_connections -= 1

@app.websocket('/debug_data')
async def debug_data():
    global Monica
    while True:
        # Calculate frame rate
        fps = round(Monica.fps)
        
        # Calculate CPU load
        cpu_load = Monica.cpu_load

        # Get motion detection and face detection results
        #motion_detected = Monica.motion_detected
        #face_detected = Monica.face_detected
        #thanks chatgpt ;)
        motion_detected = bool(Monica.motion_detected) if isinstance(Monica.motion_detected, np.bool_) else Monica.motion_detected
        face_detected = bool(Monica.face_detected) if isinstance(Monica.face_detected, np.bool_) else Monica.face_detected
        
        # Send debug data to browser
        debug_data = {
            'fps': fps,
            'cpu_load': cpu_load,
            'motion_detected': motion_detected,
            'face_detected': face_detected
        }
        await websocket.send(json.dumps(debug_data))

        # Wait for a short period before sending the next debug data
        await asyncio.sleep(0.1)

@app.websocket('/toggle_facedetect')
async def toggle_facedetect():
    global Monica
    global connected_clients, processing_face_det,processing_state,face_detect_mapping
    # Register the new client
    connected_clients.append(websocket._get_current_object())
    # Send the current slider state to the newly connected client
    await websocket.send(json.dumps(slider_state))
    try:
        # Send the current processing mode to the new client
        
        #await websocket.send(json.dumps({'mode': processing_face_det, 'processing': processing_state}))
        
        # Listen for incoming messages (mode changes)
        while True:
            
            
            data = await websocket.receive()
            #mode = json.loads(data).get('mode')
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                print("Invalid JSON received")
                continue

            # Update the server's slider state based on the received message
            # Update server-side slider state based on the received message
            if message.get('type') == 'motion':
                mode_value = int(message.get('mode'))
                # Use the mapping to convert the integer to the corresponding mode
                Monica.toggle_motion = motion_detect_mapping.get(mode_value, 'no processing')
                print(f"toggle motion {Monica.toggle_motion}")
                slider_state['motion'] = message.get('mode')
            elif message.get('type') == 'face_detect':
                mode_value = int(message.get('mode'))
                # Use the mapping to convert the integer to the corresponding mode
                Monica.toggle_facedetect = face_detect_mapping.get(mode_value, 'no processing')
                print(f"toggle face detect {Monica.toggle_facedetect}")
                slider_state['face_detect'] = message.get('mode')
            elif message.get('type') == 'frameRatio':
                print(f"Received WebSocket message: {message}")
                mode_value = int(message.get('value'))
                Monica.process_every_n_frames=mode_value
                print(f"process every {Monica.process_every_n_frames} frame")
                slider_state['frameRatio'] = message.get('value')
            
            '''
            if 'mode' in message and message['mode'] in ['DLIB', 'Yunet']:
                # Update the global processing mode
                processing_face_det = message['mode']
                Monica.toggle_dlib = (processing_face_det == 'DLIB')
                Monica.toggle_yunet = (processing_face_det == 'Yunet')
                print(f'Switched to {processing_face_det} mode')
            
            if 'processing' in message:
                # Update the global processing state
                processing_state = message['processing']
                Monica.processing = processing_state
                print(f'Processing {"started" if processing_state else "stopped"}')
            '''    
                # Broadcast the mode and state change to all clients
            for client in connected_clients:
                try:
                    #await client.send(json.dumps({'mode': processing_face_det, 'processing': processing_state}))
                    await client.send(json.dumps(slider_state))
                    
                except Exception as e:
                    print(f"Failed to send message to client: {e}")
    except Exception as e:
        print(f"WebSocket error: {e}")

    finally:
        # Remove the client from the connected list if the connection is closed
        connected_clients.remove(websocket._get_current_object())
#------------------------------------------------------------------
#
#       HISTORY
#
#------------------------------------------------------------------
# Route to serve the watch recordings page
"""
@app.route('/history')
async def history():
    async for db in get_db():
    
        recordings = await DatabaseOperations.get_recordings(db)
        print(recordings)
    return await render_template('db_history.html', recordings=recordings)

"""
@app.route('/history')
async def history():
    # Path to the recordings directory
    recordings_path = os.path.join(app.root_path, 'recordings')
    
    # Get a list of all MP4 files in the recordings directory
    recordings = [f for f in os.listdir(recordings_path) if f.endswith('.mp4')]
    
    # Render the history.html template and pass the list of recordings
    return await render_template('history.html', recordings=recordings)

# Serve individual MP4 files from the 'static/recordings' directory
@app.route('/static/recordings/<filename>')
async def serve_recording(filename):
    print(f"{filename}")
    recordings_path = os.path.join(app.root_path, 'recordings')
    return await send_from_directory(recordings_path, filename)

#---------------------------------------------------------------------
#
# 4G PROCESSING
#
#---------------------------------------------------------------------

# Add a route to manually send SMS
@app.route('/send_sms/<phone_number>/<message>')
async def send_sms(phone_number, message):
    success = await dongle.send_sms(phone_number, message)
    if success:
        await flash('SMS sent successfully!', 'success')
    else:
        await flash('Failed to send SMS', 'error')
    return redirect(url_for('index'))


#---------------------------------------------------------------
#
#               BEFORE SERVING
#
# the server and the process need to start before a client connects
# we init the database
# we start the thread videoprocessor
#
#---------------------------------------------------------------

@app.before_serving
async def startup():
    global Monica
    logger.info("Starting up application...")
    await init_db()
    # Store the main event loop
    main_loop = asyncio.get_event_loop()
    Monica = VideoProcessor(main_loop)
    Thread(target=Monica.camera_thread, daemon=True).start()
    # Test connection status at startup
    status = await dongle.get_connection_status()
    logger.info(f"Initial 4G connection status: {'Connected' if status else 'Disconnected'}")

@app.after_serving
async def shutdown():
    Monica.shutdown()
    executor.shutdown()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)