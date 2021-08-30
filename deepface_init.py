#-------------------------------------------#
# Project: DeepFace - Facial Attribute Analysis 
#          -> Emotion, Age, Gender, Race
#
# Author: Thiago Piovesan
# Date: 27 de Agosto de 2021
# Teste: Versão 1.0
#-------------------------------------------#
# Importação de Bibliotecas
from numpy.lib.npyio import save
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import json
import time
import base64
import numpy as np
import datetime
import pandas as pd

#-------------------------------------------#
# Captura contínua da câmera
def loop_and_detect(cam):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
    """
#-------------------------------------------#
# Inicialização de Variáveis
    c_timer = 0
    fps = 0.0
    tic = time.time()
#-------------------------------------------#
    # Aux timer variable
    last_time = datetime.datetime.now().second
#-------------------------------------------#
    save_frames = []
    # ct = CentroidTracker()
    # auxId = -1
#-------------------------------------------#
# Início do Loop ---
    while True:
        # try:
        #     ret, frame = cam.read()
        # except Exception as e:
        #     print(str(e))
        #     continue
#-----------------------------------------------------------------#
        ret, frame = cam.read()
        if frame is None or ret is not True:
            continue

        #-----------------------------------------------------------------#
        try:
            c_timer = datetime.datetime.now().second
            save_frames.append(frame)               # Salva os frames em um vetor
            # cv2.imshow('frame',frame)

            # if(objectID > auxId):
        #-----------------------------------------------------------------#
            # Convert captured image to JPG
            toc = time.time()
        
        # Pega os frames acumulados ao longo de 1 segundo
            if abs(c_timer-last_time) == 1:
                print(str(abs(c_timer-last_time)))
                # ret, buffer = cv2.imencode('.jpg', frame)
                
                cv2.imwrite('images/teste.jpg', save_frames[0]) 

                # Convert Frame to base64
                # base64_buffer = base64.b64encode(buffer)

                captures = DeepFace.analyze(save_frames[0], actions=['emotion'], enforce_detection=True)
            
                emotion = captures['emotion']
                print(emotion)

                last_time = c_timer
                save_frames = []
            #-----------------------------------------------------------------#

            cv2.imshow('Sorria, você está sendo analizado', frame) 
        #-----------------------------------------------------------------#   
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
        #-------------------------------------------# 
        # Tecla para sair do stream

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
        #-------------------------------------------# 
        except Exception as e:
            print(str(e))
            continue
#-------------------------------------------# 
# Função principal
def main():

    # Connect to camera
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Sorria, você está sendo analizado',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sorria, você está sendo analizado', 640,640)
    # cam.open("rtsp://admin:Pti2389!!@192.168.0.100:554/h264/ch1/sub")

    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
    
    # Loop to detect
    loop_and_detect(cam)

    # Close all
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()