#-------------------------------------------#
# Project: DeepFace - Facial Attribute Analysis 
#          -> Emotion, Age, Gender, Race
#
# Author: Thiago Piovesan
# Date: 27 de Agosto de 2021
# Teste: Versão 1.0
#-------------------------------------------#
# Importação de Bibliotecas
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import json
import time
import base64

#-------------------------------------------#
# Captura contínua da câmera
def loop_and_detect(cam, conf_th):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      conf_th: confidence/score threshold for object detection.
    """
#-------------------------------------------#
# Inicialização de Variáveis
    c_timer = 0
    full_scrn = False
    fps = 0.0
    tic = time.time()
#-------------------------------------------#
    # Aux variable
    last_time = time.time()
#-------------------------------------------#
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
#-------------------------------------------#
# Pega 1 frame a cada 30 frames -> 2 frames por segundo
        # if c_timer == 30:
        ret, frame = cam.read()
        if frame is None or ret is not True:
                continue

        #     c_timer = 0
        # else:
        #     c_timer = c_timer + 1
        #-------------------------------------------#

        #-------------------------------------------#
        try:
        
            cv2.imshow('frame',frame)

            # if(objectID > auxId):
        #-------------------------------------------#
            # Convert captured image to JPG
            toc = time.time()
            if (1.0 / (toc - tic)) >= 120:

                ret, buffer = cv2.imencode('.jpg', frame)
                
                cv2.imwrite('images/teste.jpg', frame) 

                # Convert JPG image to base64
                # base64_buffer = base64.b64encode(frame)

                captures = DeepFace.analyze('images/teste.jpg', actions=['emotion'])
            
                # print(captures['emotion'])
                emotion = captures['emotion']
                print(emotion)

            cv2.imshow('frame', frame) 
        #-------------------------------------------#   
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
    # cam.open("rtsp://admin:Pti2389!!@192.168.0.100:554/h264/ch1/sub")

    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
    
    # Loop to detect
    loop_and_detect(cam, conf_th=0.3)

    # Close all
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()