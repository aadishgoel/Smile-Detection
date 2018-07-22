import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)
font=cv2.FONT_HERSHEY_COMPLEX_SMALL
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces, faceRejectLevels, faceLevelWeights= face_cascade.detectMultiScale3(gray, 1.3, 5, outputRejectLevels=True)
    f=0
    for(x,y,w,h) in faces:
        if (round(faceLevelWeights[f][0],3)) <= 5: continue
        #print(round(faceLevelWeights[f][0],3))         To Display Face Confidence
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(img, str(round(faceLevelWeights[f][0],3)), (x,y),font, 1, (255,255,255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes, rejectLevels, levelWeights = smile_cascade.detectMultiScale3(roi_gray, outputRejectLevels=True)
        i=0
        for(ex,ey,ew,eh) in eyes:
            if(round(levelWeights[i][0],3)>=3.5):
                #print(round(levelWeights[i][0],3))       To Display Smile Confidence
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
                cv2.putText(roi_color,str(round(levelWeights[i][0],3)),(ex,ey), font,1,(255,255,255),2)
            i+=1
        f+=1
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
