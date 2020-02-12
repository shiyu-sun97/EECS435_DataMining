from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,784))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))


handwritingClassTest()



def SVMmodel(self):
        data = self.readTrainingData()
        dataT = int(len(data)*SVM.por)
        dataP = len(data) - dataT
        clf = svm.SVC()

        clf.fit(data[0][:-dataT],data[1][:-dataT])

        predictions = [int(a) for a in clf.predict(data[0][dataP:])]
        num_correct = sum(int(a == y) for a, y in zip(predictions, data[1][dataP:]))
        print ("%s of %s test values correct." % (num_correct, dataT))


s = SVM()
s.SVMmodel()



'''
    def preprocessing(self, dataPath, splitor=',', droplines = 1):
            opfile = open(dataPath, 'r', encoding = 'utf-8')
            dData = opfile.readlines()[droplines:]
            opfile.close()
            dResult = []
            dVector = []
            for line in dData:
                items = list(map(int,line[:-1].split(splitor)))
                assert len(items) == 785
                dResult.append(items[0])
                dVector.append(items[1:])
                #print(items[:])
            #print(dVector[0:])
            #print('/n')
            #print(dResult)
            return dResult,dVector
             #[((entry[1:], entry[0]) for entry in line.split(splitor)) for line in dData]
    '''
