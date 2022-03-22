# -*- coding: utf-8 -*-

__author__ = 'mike-bowles'

import urllib
import numpy
from sklearn import datasets, linear_model
from sklearn.metrics import roc_curve, auc
import pylab as plt

#read data from uci data repository
target_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
response = urllib.request.urlopen(target_url)

# 내가 추가한 것 - 51 line에서 string 처리를 못해서 #############################
data = response.read().decode("ascii")
###############################################################################

#arrange data into list for labels and list of lists for attributes
xList = []
labels = []
tList = []

# 데이터 세트의 행과 열 카운트 ##################################################
# 레이블을 리스트로, 속성을 리스트의 리스트로 데이터 정리 ########################
aa = 0
for line in data.split('\n'):
    if len(line) > 0:
        idx = 0
        for row in line.split(','):
            tList.insert(idx, row)
            idx = idx + 1    
        aList = tList.copy()        
        # label 저장
        if(aList[-1] == 'M'):
            labels.append(1.0)
        else:
            labels.append(0.0)
        # remove label from data frame
        aList.pop()
        # convert data frame to floats
        floatRow = [float(num) for num in aList]
        xList.append(floatRow)
        aa = aa +1
        if aa == 208:
            aa = 209
        else:
            tList.clear()

#divide attribute matrix and label vector into training(2/3 of data) and test sets (1/3 of data)
indices = range(len(xList))
xListTest = [xList[i] for i in indices if i%3 == 0 ]
xListTrain = [xList[i] for i in indices if i%3 != 0 ]
labelsTest = [labels[i] for i in indices if i%3 == 0]
labelsTrain = [labels[i] for i in indices if i%3 != 0]

# form list of list input into numpy arrays to match input class for scikit-learn linear model
xTrain = numpy.array(xListTrain); yTrain = numpy.array(labelsTrain); xTest = numpy.array(xListTest); yTest = numpy.array(labelsTest)

alphaList = [0.1**i for i in [-3, -2, -1, 0,1, 2, 3, 4, 5]]

aucList = []
for alph in alphaList:
    rocksVMinesRidgeModel = linear_model.Ridge(alpha=alph)
    rocksVMinesRidgeModel.fit(xTrain, yTrain)
    fpr, tpr, thresholds = roc_curve(yTest,rocksVMinesRidgeModel.predict(xTest))
    roc_auc = auc(fpr, tpr)
    aucList.append(roc_auc)
print("AUC             alpha")
for i in range(len(aucList)):
    print(aucList[i], alphaList[i])

# plot auc values versus alpha values
# AUC 값이 1에 근접하는 것은 성능이 좋은 것. 0.5에 가까우면 좋지 않다는 의미
# AUC의 목표는 MSE(Mean Square Error)와 같이 최대화 하는 것이며 alpha=1.0 일때
# 날카로운 피크를 보임.
x = [-3, -2, -1, 0,1, 2, 3, 4, 5]
plt.plot(x, aucList)
plt.xlabel('-log(alpha)')
plt.ylabel('AUC')
plt.show()

# visualize the performance of the best classifier
indexBest = aucList.index(max(aucList))
alph = alphaList[indexBest]
rocksVMinesRidgeModel = linear_model.Ridge(alpha=alph)
rocksVMinesRidgeModel.fit(xTrain, yTrain)

#scatter plot of actual vs predicted
plt.scatter(rocksVMinesRidgeModel.predict(xTest), yTest, s=100, alpha=0.25)
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.show()