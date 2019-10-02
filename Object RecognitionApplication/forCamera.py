import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('Haarcascades\haarcascade_frontalface_default.xml')
# apple_cascade = cv2.CascadeClassifier('Haarcascades\haarcascade_apple.xml')
# chair_cascade = cv2.CascadeClassifier('Haarcascades\haarcascade_chair.xml')
# car_cascade = cv2.CascadeClassifier('Haarcascades\haarcascade_car.xml')
eye_cascade = cv2.CascadeClassifier('Haarcascades\haarcascade_eye.xml')
# banana_cascade = cv2.CascadeClassifier('Haarcascades\haarcascade_banana.xml')
# dolphin_cascade = cv2.CascadeClassifier('Haarcascades\haarcascade_dolphin.xml')
# fullbody_cascade = cv2.CascadeClassifier('Haarcascades\haarcascade_fullbody.xml')
# gun_cascade = cv2.CascadeClassifier('Haarcascades\haarcascade_gun.xml')
# bear_cascade = cv2.CascadeClassifier('Haarcascades\haarcascade_bear.xml')


font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    img = cv2.flip(img,90)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray2 = np.array(gray, 'uint8')
    #faces = face_cascade.detectMultiScale(gray2)
    
    faces = face_cascade.detectMultiScale(gray,1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        cv2.putText(img,'FACE',(x+w-150,y+h+25), font, 1, (255,255,255), 2, cv2.LINE_AA)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
     
    #apples = apple_cascade.detectMultiScale(gray,30,30)
    #for(x,y,w,h) in apples:
     #   cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0),2)
      #  cv2.putText(img,'apple',(x+w-60,y+h-60), font, 0.5, (0,255,255), 2, cv2.LINE_AA)

    # cars = car_cascade.detectMultiScale(gray,1.2,2)
    # for (x,y,w,h) in cars:
    #     cv2.putText(img,'car',(x+w-60,y+h-60), font, 0.5, (0,255,255), 2, cv2.LINE_AA)

    #banana = banana_cascade.detectMultiScale(gray, 2, 40)
    #for (x,y,w,h) in banana:
     #   cv2.rectangle(img,(x,y),(x+w,y+h),(250,255,210),2)
      #  cv2.putText(img,'Banana',(x+w-60,y+h-60), font, 0.5, (11,255,24), 2, cv2.LINE_AA)

    # dolphin = dolphin_cascade.detectMultiScale(gray,20,30)
    # for (x,y,w,h) in dolphin:
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #     cv2.putText(img,'dolphin',(x+w-60,y+h-60), font, 1, (11,255,24), 2, cv2.LINE_AA)

    # chair = chair_cascade.detectMultiScale(gray, 10, 50)
    # for (x,y,w,h) in chair:
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(200,215,210),2)
    #     cv2.putText(img,'chair',(x+w-60,y+h-60), font, 0.5, (11,255,24), 2, cv2.LINE_AA)

   
    # fullBody = fullbody_cascade.detectMultiScale(gray,2,2)    
    # for (x,y,w,h) in fullBody:  
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(11,255,255),2)
    #     cv2.putText(img,'full Body',(x+w-60,y+h-60), font, 0.5, (11,255,255), 2, cv2.LINE_AA)
    # gun = gun_cascade.detectMultiScale(gray,1.7,40)    
    # for (x,y,w,h) in gun:  
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(11,255,255),2)
    #     cv2.putText(img,'Gun',(x+w-60,y+h-60), font, 0.5, (11,255,255), 2, cv2.LINE_AA)
    # bear = bear_cascade.detectMultiScale(gray,10,50)    
    # for (x,y,w,h) in bear:  
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(11,255,255),2)
    #     cv2.putText(img,'Bear',(x+w-60,y+h-60), font, 0.5, (11,255,255), 2, cv2.LINE_AA)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
