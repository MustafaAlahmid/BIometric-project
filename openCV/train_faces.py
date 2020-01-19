import cv2
import os
import numpy as np
from PIL import Image
import pickle

# our dirictor variables 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "/home/azzurri/Desktop/openCV/datasets")
thresholds = [20,40,60,80,100,120]
#face detection and recognizer 
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

for threshold in thresholds:
		
	recognizer = cv2.face.LBPHFaceRecognizer_create(threshold=threshold)# parameters are 

	#creating variables for labels id and x and y data 
	current_id = 0
	label_ids = {}
	y_labels = []
	x_train = []
	labeles = []

	#loops to load all the data set
	for root, dirs, files in os.walk(image_dir):
		for file in files:
			if file.endswith("png") or file.endswith("jpg"):
				path = os.path.join(root, file)
				label = os.path.basename(root).replace(" ", "-").lower()
				#print(label, path)
				if not label in label_ids:
					label_ids[label] = current_id
					current_id += 1
				id_ = label_ids[label]
				#print(label_ids)
				#load the image and convert to gray then resize it 
				pil_image = Image.open(path).convert("L") # grayscale
				size = (550, 550)
				final_image = pil_image.resize(size, Image.ANTIALIAS)
				image_array = np.array(final_image, "uint8")
				#print(image_array)
				#face cascade detector used KNN algorithm
				# minNeigbhors gave a better results at 3 than 5 or 7 or 1  
				faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=3)
				# region of interest 
				for (x,y,w,h) in faces:
					roi = image_array[y:y+h, x:x+w]
					x_train.append(roi)
					y_labels.append(id_)



	with open("face-labels.pickle", 'wb') as f:
		pickle.dump(label_ids, f)

	#train the recognizer and save it 
	recognizer.train(x_train, np.array(y_labels))
	recognizer.save("face-trainner.yml")

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
						#print(conf)
					else:
						wrong+=1
						number+=1
	# print the results 
	print(f'---------------threshold = {threshold} --------------------')
	print('number of tested pictures = ',number)
	print('wrong predict',wrong)
	print('correct predict',correct)

