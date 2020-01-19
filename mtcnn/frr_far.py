from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# # we here compute the False accaptance rate and false rejection rate 
# we will load the test data which contain over 200 pictures 
# crop the faces and put them in numpy array then encode them as we did before with train test data 
#the dataset contain pictures from imposter and geniune  with 50%  percentage for each 

data_gen = load('cinema_class.npz')
data_imposter= load('test_1.npz')
testX_imposter_faces = data_imposter['arr_0']
test_X_gen_faces = data_gen['arr_2']
print(test_X_gen_faces.shape)
print(testX_imposter_faces.shape)
# load face embeddings
data_imposter = load("test_1_embeddings.npz")
data_gen = load('cinema_class_embeddings.npz')
trainX_gen, trainy_gen, testX_gen, testy_gen = data_gen['arr_0'], data_gen['arr_1'],data_gen['arr_2'],data_gen['arr_3']
testX_imposter, testy_imposter = data_imposter['arr_0'], data_imposter['arr_1']
print(testX_gen.shape)
print(testX_imposter.shape)
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX_gen = in_encoder.transform(trainX_gen)
testX_imposter = in_encoder.transform(testX_imposter)
testX_gen = in_encoder.transform(testX_gen)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy_gen)
trainy_gen = out_encoder.transform(trainy_gen)
testy_gen=out_encoder.transform(testy_gen)
# fit model
model = pickle.load(open('finalized_model.pkl', 'rb'))
model.fit(trainX_gen, trainy_gen)
# test model on imposter faces 

# a is a list where we will calculate the probability of the false accptance rate FAR
# the model will predect the face and then check the matching probablity and add it to the a list  
a = []
for i in range(100):
        random_face_pixels = testX_imposter_faces[i]
        random_face_emb = testX_imposter[i]
        random_face_class = testy_imposter[i]
        samples = expand_dims(random_face_emb, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        intp = int((class_probability))
        #print(f'Predicted: {intp} %')
        a.append(intp)
 # far is a list that we will save the False accaptance rate in each threshold 
 # threshold is the list of thresold and it will go from 0% to 100%       
far = []
threshold = []
for i in range(100):
        num = 0

        for x in a:
                if x>i:
                        num+=1
        #print(i,num)
        far.append(num)
        threshold.append(i)

far = np.array(far)
print('FAR: ',far)
print('-----------------------------------------------------------')


b = []
for i in range(100):
        random_face_pixels = test_X_gen_faces[i]
        random_face_emb = testX_gen[i]
        random_face_class = testy_gen[i]
        face_name = out_encoder.inverse_transform([random_face_class])
        # prediction for the face
        samples = expand_dims(random_face_emb, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_threshold = out_encoder.inverse_transform(yhat_class)
        if predict_threshold[0]==face_name[0]:
                intp = int((class_probability))
                #print(f'Predicted: {intp} %')
                b.append(intp)
# print(b)
frr = []
for i in range(100):
        num = 0

        for x in b:
                if x<i:
                        num+=1
        #print(i,num)
        frr.append(num)


frr = np.array(frr)
print('FRR: ',frr)
print('-----------------------------------------------------------')


for i  in range(100):
        a = frr[i]
        b = far[i]
        if a == b:
                EER= a
                print('EER = ',i)

threshold = np.array(threshold)

plt.plot(threshold,frr,'--b',far,'--r')
plt.plot(15,EER,'ro') 

plt.xlabel('threshold')
plt.title('FAR,FRR and EER')
plt.axis([0, 100, 0, 100])
plt.show()

plt.plot(threshold,frr,'--b')
plt.xlabel('threshold')
plt.title('FRR')
plt.axis([0, 100, 0, 100])
plt.show()


plt.plot(threshold,far,'--r')
plt.xlabel('threshold')
plt.title('FAR')

plt.axis([5, 20, 0, 100])
plt.show()



fig, ax = plt.subplots()

ax.plot(threshold, far, 'r--', label='FAR')
ax.plot(threshold, frr, 'g--', label='FRR')
plt.xlabel('Threshold')
plt.plot(15,EER,'ro', label='EER') 


legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')

plt.show()

