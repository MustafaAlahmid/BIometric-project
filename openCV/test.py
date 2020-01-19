import cv2
import numpy as np
import pickle
from PIL import Image
import os 
print(cv2.__file__)

# use the face cascades to detect faces 
face_cascade = cv2.CascadeClassifier('/home/azzurri/Desktop/openCV/cascades/data/haarcascade_frontalface_alt2.xml')
#useing the open cv recognizer for 
recognizer = cv2.face.LBPHFaceRecognizer_create()
# read the our  trained model 
recognizer.read('/home/azzurri/Desktop/openCV/face-trainner.yml')


# creat the labels 
labels = {"person name":1}
# load the labels from the model 
with open("face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


# creat 3 variables for counting the correct and the wrong prediction 
correct = 0
wrong = 0
number = 0
#load the test dataset 
test_data= '/home/azzurri/Desktop/openCV/datasets/test'
 
# load all pictures and take thier labels detect the faces and crop the face then detect the face  
for root, dirs, files in os.walk(test_data):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()

            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")

            # get the id and the confedence from the recognizer for the predectid picture 

            id_, conf = recognizer.predict(image_array)
            if conf >=30 and conf<=120:
                if labels[id_] == label:
                    correct+=1
                    number+=1
                    print(conf)
                else:
                    wrong+=1
                    number+=1
# print the results 
print('number of tested pictures = ',number)
print('wrong predict',wrong)
print('correct predict',correct)

