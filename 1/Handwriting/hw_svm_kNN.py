import numpy as np
from sklearn import svm
from sklearn import neighbors
import pandas as pd


def readTrainingData(dp = "training.csv"):
    trainingData = np.loadtxt(dp, delimiter = ",",skiprows = 1)#skip the first row of string
    label = trainingData[:,0]#split the label and training data
    data = trainingData[:,1:785]
    return data,label

def readTestingData(dp = "testing.csv"):
    testingData = np.loadtxt(dp, delimiter = ",",skiprows = 1)#skip the first row of string
    data = testingData[:,:784]
    return data



class SVM():
    def preprocessing(self,data,label):
        return data,label
    def testing(self,data):
        return data
    def SVMmodel(self):
        data = readTrainingData()
        tdata = readTestingData()

        clf = svm.SVC()
        clf.fit(data[0],data[1]) #training

        predictions1 = [int(a) for a in clf.predict(data[0][19000:,:])] #predicting

        num_correct = sum(int(a == y) for a, y in zip(predictions1, data[1][19000:])) #results
        print ("%s of %s test values correct." % (num_correct, len(data[1][19000:])))

        predictions2 = [str(b) for b in clf.predict(tdata)]
        result = pd.DataFrame(data=predictions2)
        result.to_csv('resultSVM.csv',index=False,header=False,sep=',',encoding='utf-8')

class kNN():
    def preprocessing(self,data,label):
        return data,label
    def testing(self,data):
        return data
    def kNNmodel(self):
        data = readTrainingData()
        tdata = readTestingData()

        clf = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
        clf.fit(data[0],data[1]) #training

        predictions = [int(a) for a in clf.predict(data[0][19000:,:])] #predicting

        num_correct = sum(int(a == y) for a, y in zip(predictions, data[1][19000:])) #results
        print ("%s of %s test values correct." % (num_correct, len(data[1][19000:])))

        predictions2 = [int(b) for b in clf.predict(tdata)]
        result = pd.DataFrame(data=predictions2)
        result.to_csv('resultkNN.csv',index=False,header=False,sep=',',encoding='utf-8')


s1 = SVM()
s1.SVMmodel()

s2 = kNN()
s2.kNNmodel()


