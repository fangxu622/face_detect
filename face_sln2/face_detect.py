#import numpy as np
import cv2
from pyseeta import Detector


import argparse




parser = argparse.ArgumentParser(description='face detection')
parser.add_argument('-vp','--video', default="",
                    help='path to video,example: -vp /home/uses/xx.mp4"')
parser.add_argument('-s','--scale',type=float, default=0.8,
                    help='resize the video,Range of value:[0,1]')
parser.add_argument('-sf','--skipFrame', default=2, type=int,
                    help='the number of skip frame')
parser.add_argument('-m','--model', default="./faceFront.bin",
                    help='the number of skip frame')

#image_color = cv2.imread('data/test.jpg', cv2.IMREAD_COLOR)
#image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
#faces = detector.detect(image_gray)
global args
args = parser.parse_args()

#cv2.imshow('test', image_color)

def face_dect(video,scale,skipFrame,modelPath):

    cap = cv2.VideoCapture(video)
    detector = Detector(modelPath)
    detector.set_min_face_size(30)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        i = i + 1
        # print(ret)
        if not (i % skipFrame  == 0):
            pass
        gray_1 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray_1, None, fx=scale, fy=scale, interpolation=2)
        faces = detector.detect(gray)
        for i, face in enumerate(faces):
            # print('({0},{1},{2},{3}) score={4}'.format(face.left, face.top, face.right, face.bottom, face.score))
            cv2.rectangle(frame, (int(face.left / scale), int(face.top / scale)),
                          (int(face.right / scale), int(face.bottom / scale)),
                          (0, 255, 0), thickness=1)

            # cv2.putText(gray, str(i), (face.left, face.bottom), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
            #         thickness=1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    if not args.video:
        print("please input video path (at least),example: -vp /home/uses/xx.mp4")
    else:
        face_dect(video=args.video,
                  scale=args.scale,
                  skipFrame=args.skipFrame,modelPath=args.model)
