import cv2
import numpy as np

#video import
cap = cv2.VideoCapture('video2.mp4')
min_width_rect=80    #min width rectangle
min_height_rect=80   #min height rectangle
count_line_position=550

#Initializing substructor algorithm
algo=cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    r,frame1= cap.read()
    grey= cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur= cv2.GaussianBlur(grey,(3,3),5)
    img_sub= algo.apply(blur)
    dilat= cv2.dilate(img_sub,np.ones((5,5)))
    kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada= cv2.morphologyEx(dilat,cv2.MORPH_CLOSE, kernel)
    dilatada= cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE, kernel)
    counterSahpe,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,122,0),4)

    for (i,c) in enumerate(counterSahpe):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter=(w>=min_width_rect) and (h>=min_height_rect)
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        vehicle= frame1[y: y+h, x: x+w]
        filename='savedImage.jpg'
        cv2.imshow("vehicle", vehicle)
        cv2.imwrite(filename,vehicle)


    #cv2.imshow('Detector',dilatada)
    cv2.imshow('Video Original',frame1)
    if (cv2.waitKey(30)==13):
        break
cv2.destroyAllWindows() 
cap.release()
