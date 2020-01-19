from os import listdir
import os
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
import pickle 
from sklearn.externals import joblib 
from keras.models import load_model
import numpy as np 
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
import pandas as pd

# extract the faces from the photo 
def extract_face(filename, required_size=(160, 160)):
	image = Image.open(filename)
	image = image.convert('RGB')
	pixels = asarray(image)
	#us the detector MRCNN
	detector = MTCNN()
	# detect faces 
	results = detector.detect_faces(pixels)
	# extract the bounding box from the face 
	x1, y1, width, height = results[0]['box']
	# fixing bugs 
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize the image to required size 
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# load images and extract faces for all images in a directory
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

# load the dataset from all the dir and subdir
def load_dataset(directory):
	X = list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + '/'
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
	return asarray(X)

def get_embedding(model, face_pixels):
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]


# load test dataset
test_real = load_dataset('my_pic/')
# save arrays to one file in compressed format
savez_compressed('real.npz',test_real)


trainX = asarray(test_real)
model = load_model('facenet_keras.h5')
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
#print(newTrainX.shape)
# save arrays to one file in compressed format
savez_compressed('real-embeddings.npz', newTrainX)



# load faces
testX_faces = trainX

# load face embeddings
data = load('cinema_class_embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
data1 = load('real-embeddings.npz')
test_real = data1['arr_0']
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# test model on a random example from the test dataset
for i in range(1):

	random_face_pixels = testX_faces[i]
	random_face_emb = test_real[i]
	random_face_class = testy[i]

	#fit the model

	model = pickle.load(open('finalized_model.pkl', 'rb'))

	# prediction for the face
	samples = expand_dims(random_face_emb, axis=0)
	yhat_class = model.predict(samples)
	yhat_prob = model.predict_proba(samples)

	# get the name
	class_index = yhat_class[0]
	class_probability = yhat_prob[0,class_index] * 100
	predict_names = out_encoder.inverse_transform(yhat_class)
	if (class_probability>20):
		print('Predicted: %s' % (predict_names[0]),class_probability)
		print(model.predict(samples))
		# plot for fun
		pyplot.imshow(random_face_pixels)
		title = '%s' % (predict_names[0],)
		pyplot.title(title)
		pyplot.show()
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		print(now)
		df1 = pd.DataFrame({'name': [title,],
		'log in time': [current_time]})
		df=pd.read_csv('login.csv')
		df.append(df1)
		df.to_csv('login.csv')


	else:
		print(class_probability)
		img = cv2.imread('mn.jpg')
		cv2.imshow('Who are you',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()






	
	
