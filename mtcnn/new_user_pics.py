import cv2
import numpy as np
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from tkinter import*

root = Tk()
my_label = Label(root,text = 'click on C to capture photo and Q to finish')
my_label2 = Label(root,text='close this window to start')
my_label2.config(font=("Courier", 25))
my_label.config(font=("Courier", 30))


my_label.pack()
my_label2.pack()
 

root.mainloop()


face_cascade= cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
num=0
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        print(x,y,w,h)
        #region of interest
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        #to save the pics of the face
        
        
        color = (255,0,0)
        stroke = 2 
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)

    cv2.imshow('frame',frame)
    
    if cv2.waitKey(20)&0xFF==ord('c'):
         cv2.imwrite(f'pics/{str(num)}.png',frame)
         num+=1

    
    elif cv2.waitKey(20)&0xFF==ord('q'):

        break

cap.release()
cv2.destroyAllWindows







