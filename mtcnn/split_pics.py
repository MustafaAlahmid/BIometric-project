import os
import shutil
import random
import numpy as np
from tkinter import*
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from numpy import load
from keras.models import load_model
from numpy import expand_dims
from numpy import savez_compressed



def split_pic(username):
       pics = next(os.walk('pics/'))[2] #dir is your directory path as string
       n = len(pics)/2
       l = int(n)

       data=np.array(pics)
       val = data[l::]
       train = data[0:l]
       os.mkdir(f"datasets/validation/{username}")
       os.mkdir(f"datasets/train/{username}") 

       for pic in val:
              shutil.move(f"pics/{pic}", f"datasets/validation/{username}")

       for pic in train:
              shutil.move(f"pics/{pic}", f"datasets/train/{username}")


       def extract_face(filename, required_size=(160, 160)):
              image = Image.open(filename)
              image = image.convert('RGB')
              pixels = asarray(image)
              detector = MTCNN()
              results = detector.detect_faces(pixels)
              x1, y1, width, height = results[0]['box']
              x1, y1 = abs(x1), abs(y1)
              x2, y2 = x1 + width, y1 + height
              face = pixels[y1:y2, x1:x2]
              image = Image.fromarray(face)
              image = image.resize(required_size)
              face_array = asarray(image)
              return face_array

       def load_faces(directory):
              faces = list()
              # enumerate files
              for filename in listdir(directory):
                     # path
                     path = directory + filename
                     # get face
                     face = extract_face(path)
                     # store
                     faces.append(face)
              return faces

       def load_dataset(directory):
              X, y = list(), list()
              # enumerate folders, on per class
              for subdir in listdir(directory):
                     # path
                     path = directory + subdir + '/'
                     # skip any files that might be in the dir
                     if not isdir(path):
                            continue
                     # load all faces in the subdirectory
                     faces = load_faces(path)
                     # create labels
                     labels = [subdir for _ in range(len(faces))]
                     # summarize progress
                     print('>loaded %d examples for class: %s' % (len(faces), subdir))
                     # store
                     X.extend(faces)
                     y.extend(labels)
              return asarray(X), asarray(y)



              
       # load train dataset
       trainX, trainy = load_dataset(f"datasets/train/{username}")
       print(trainX.shape, trainy.shape)
       # load test dataset
       testX, testy = load_dataset(f"datasets/validation/{username}")
       data = load('cinema_class.npz')
       data.append(trainX,trainy,testX,testy)
       savez_compressed('cinema_class.npz', trainX, trainy, testX, testy)


              ##
              #############################
       # get the face embedding for one face
       def get_embedding(model, face_pixels):
              # scale pixel values
              face_pixels = face_pixels.astype('float32')
              # standardize pixel values across channels (global)
              mean, std = face_pixels.mean(), face_pixels.std()
              face_pixels = (face_pixels - mean) / std
              # transform face into one sample
              samples = expand_dims(face_pixels, axis=0)
              # make prediction to get embedding
              yhat = model.predict(samples)
              return yhat[0]
       data = load('cinema_class.npz')
       trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
       print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
       # load the facenet model
       model = load_model('facenet_keras.h5')
       print('Loaded Model')
       # convert each face in the train set to an embedding
       newTrainX = list()
       for face_pixels in trainX:
              embedding = get_embedding(model, face_pixels)
              newTrainX.append(embedding)
       newTrainX = asarray(newTrainX)
       print(newTrainX.shape)
       # convert each face in the test set to an embedding
       newTestX = list()
       for face_pixels in testX:
              embedding = get_embedding(model, face_pixels)
              newTestX.append(embedding)
       newTestX = asarray(newTestX)
       print(newTestX.shape)
       # save arrays to one file in compressed format
       savez_compressed('cinema_class_embeddings.npz', newTrainX, trainy, newTestX, testy)

       ################################################################

       # load dataset
       data = load('cinema_class_embeddings.npz')
       trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
       #print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
       # normalize input vectors
       in_encoder = Normalizer(norm='l2')
       trainX = in_encoder.transform(trainX)
       testX = in_encoder.transform(testX)
       # label encode targets
       out_encoder = LabelEncoder()
       out_encoder.fit(trainy)
       trainy = out_encoder.transform(trainy)
       testy = out_encoder.transform(testy)
       # fit model

       svc = SVC(kernel='rbf',gamma=0.7,probability=True).fit(trainX, trainy)
       yhat_train = svc.predict(trainX)
       yhat_test = svc.predict(testX)
       yhat_prob = svc.predict_proba(testX)





root = Tk()
e = Entry(root,width = 30)
e.pack()
e.get()
e.insert(0,'')
hello = e.get()

def myclick():
       hell0 = e.get()
       my_label = Label(root,text = hell0)
       my_label.pack()       
       split_pic(hell0)
my_button = Button(root,text='Add user',padx =20, pady=20, command = myclick, fg='blue')
my_button.pack()
root.mainloop()







