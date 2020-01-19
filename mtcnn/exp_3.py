from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn import metrics
import numpy as np 
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot 
from sklearn.metrics import roc_curve
from sklearn.metrics import auc  



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

score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
report = classification_report(testy, yhat_test)
matrics = confusion_matrix(testy,yhat_test)





binary_confusion_matrix = ConfusionMatrix(testy, yhat_test)
binary_confusion_matrix.plot(normalized=True)
plt.show()

print(report)
#cm.print_stats()

print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
