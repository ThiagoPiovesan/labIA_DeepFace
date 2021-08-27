"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""

from icecream import ic
import os
import time
import argparse
import uuid
import pika
import base64
import numpy as np
from imageio import imread
import io
import json
import ast
import threading
import queue
q=queue.Queue()

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import Jetson.GPIO as GPIO
from obj_tracking import CentroidTracker

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from micro_services.request import addNewInference

def __init__(self, cls_dict):
    self.cls_dict = cls_dict
WINDOW_NAME = 'TrtYOLODemo'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args

def writeFiles(body):
    # Convert the data to object
    body = ast.literal_eval(body.decode('utf-8'))

    # grab the variables
    img1 = body['0']
    cls_image = body['1']
    confs = body['2']

    # build the image
    img3 = imread(io.BytesIO(base64.b64decode(img1)))                           
    img = cv2.cvtColor(np.array(img3), cv2.COLOR_BGR2RGB)                                                                                                                                                                                                                                                                                                                                                                                                                                   
    
    # Gerate an uuid for the image
    uuidImage = uuid.uuid4().hex

    # Variables configs
    directory = '/home/jetsonnano/captured_images/'
    img_name_withoutInference = ''                                                                                                                                                                                                  
    img_name_withInference = '' 
    txt_name = ''
    #create_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    #create the directory to store the images
    try:
        if not os.path.exists(directory + 'Img' + str(uuidImage)):
            os.makedirs(directory + 'Img' + str(uuidImage))
    except OSError:
        print('Error: Creating directory. ' + directory)

    # Grab the path
    path = directory + 'Img' + str(uuidImage) + '/'                                                                                                                                                                                                                             

    # Save the images without yolo-inference
    img_name = path + 'img' + str(uuidImage) + '.png'
    img_name_withoutInference = 'img' + str(uuidImage) + '.png'
    txt_name = 'img' + str(uuidImage)
    cv2.imwrite(img_name, img) 

    # Write the .txt file with the classes
    f = open(path + 'img' + str(uuidImage) + ".txt", "x")
    i = 0
    for i in range (0, len(cls_image)):
        accuracy = '{:.2f}'.format(confs[i] * 100)
        f.write(str(cls_image[i]) + ' ' + str(accuracy) + '\n')
        i += 1

    f.close()
        
    #print('confs:', confs)
    #print('class:', cls_image)    

    # Write the image with the boxes
    img_name = path + 'imgYolo' + str(uuidImage) + '.png'
    img_name_withInference = 'imgYolo' + str(uuidImage) + '.png'
    cv2.imwrite(img_name, img)

    addNewInference(path, img_name_withoutInference, img_name_withInference, txt_name)
    # print("{} written!".format(img_name))
    return 

def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()

    # Aux variable
    last_time = time.time()

    # RabbitMQ configs
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='inference')

    # Set GPIO configs
    #GPIO.setwarnings(False)
    #GPIO.setmode(GPIO.BOARD)
    #GPIO.setup(15, GPIO.IN)         #Read output from PIR motion sensor

    ct = CentroidTracker()
    auxId = -1

    while True:
        # try:
        #     ret, frame = cam.read()
        # except Exception as e:
        #     print(str(e))
        #     continue

        ret, frame = cam.read()

        if frame is None or ret is not True:
                continue

        try:
        
            cv2.imshow('frame',frame)
            # sensor = GPIO.input(15)

            boxes, confs, clss = trt_yolo.detect(frame, conf_th)
            results = vis.draw_bboxes(frame, boxes, confs, clss)
            frame = results[0]
            frame = show_fps(frame, fps)

            cls_image = results[1]

            objects = ct.update(boxes)

            #print(objects.items())
            
            for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame

                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                if(objectID > auxId):

                    # Convert captured image to JPG
                    ret, buffer = cv2.imencode('.jpg', frame)

                    # Convert to base64 encoding and show start of data
                    base = base64.b64encode(buffer).decode('utf-8')
                    data = {
                        0: base,
                        1: cls_image,
                        2: confs.tolist()
                    }

                    # data = base64.b64encode(data)

                    # Send the image tho rabbitMQ
                    channel.basic_publish(exchange='',  
                            routing_key='chanel2',
                            body=json.dumps(data))
                    #writeFiles(frame, cls_image, confs)

                    
                    auxId = objectID
                    
                    

            #cv2.imshow('frame', frame) 
            
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            #elif sensor and (round(time.time() - last_time) % 60 == 20):

                ## Update the Aux variable
                #last_time = time.time()

                ## Convert captured image to JPG
                #ret, buffer = cv2.imencode('.jpg', frame)

                ## Convert to base64 encoding and show start of data
                #base = base64.b64encode(buffer)
                
                ## Send the image tho rabbitMQ
                #channel.basic_publish(exchange='',
                        #routing_key='inference',
                        #body=base)
                ## connection.close()
        except Exception as e:
            print(str(e))
            continue



def main():
    # Check Args
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)
    
    # yolo configs
    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    # Connect to camera
    cam = cv2.VideoCapture()
    cam.open("rtsp://admin:Pti2389!!@192.168.0.100:554/h264/ch1/sub")

    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
    
    # Loop to detect
    loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis)

    # Close all
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
