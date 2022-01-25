from typing import final
import cv2 
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*"XVID")
outputFile = cv2.VideoWriter("output.avi",fourcc,20.0,(640,480))
cap = cv2.VideoCapture(0)
image = cv2.

time.sleep(2)
bg = 0
for i in range(60):
    ret,bg = cap.read()

bg = np.flip(bg,axis=1)
while(cap.isOpened()):
    ret,img = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480)) 
    image = cv2.resize(image, (640, 480)) 
  
  
    u_black = np.array([104, 153, 70]) 
    l_black = np.array([30, 30, 0]) 
  
    mask = cv2.inRange(frame, l_black, u_black) 
    res = cv2.bitwise_and(frame, frame, mask = mask) 
  
    f = frame - res 
    f = np.where(f == 0, image, f) 

    img = np.flip(img,axis=1)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lowered = np.array([30,30,0])
    upperred = np.array([100,150,70])
    mask1 = cv2.inRange(hsv,lowered,upperred)
    lowered = np.array([190,12,70])
    upperred = np.array([290,12,6])
    mask2 = cv2.inRange(hsv,lowered,upperred)

    mask1 = mask1+mask2
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((1,1),np.uint8))
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE,np.ones((1,1),np.uint8))
    mask2 = cv2.bitwise_not(mask1)
    res1 = cv2.bitwise_and(img,img,mask = mask2)
    res2 = cv2.bitwise_and(bg,bg,mask = mask1)

    finalOutput =cv2.addWeighted(res1,1,res2,1,0)
    outputFile.write(finalOutput)
    cv2.imshow("magic",finalOutput)
    cv2.waitKey(1)

cap.release()
outputFile.release()
cv2.distroyallwindows()