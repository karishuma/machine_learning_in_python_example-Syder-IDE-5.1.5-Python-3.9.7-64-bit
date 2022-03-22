# -*- coding: utf-8 -*-

__author__ = 'mike-bowles'

# 바위와 기뢰 데이터로 트레이닝한 분류의 성능 예측 ###############################

import urllib
import numpy
import random
from sklearn import datasets, linear_model
from sklearn.metrics import roc_curve, auc
import pylab as pl

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot

def confusionMatrix(predicted, actual, threshold):
    if len(predicted) != len(actual): return -1
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    for i in range(len(actual)):
        if actual[i] > 0.5:     # labels that are 1.0  (positive examples)
            if predicted[i] > threshold:
                tp += 1.0       # correctly predicted positive
            else:
                fn += 1.0       # incorrectly predicted negative
        else:                   # labels that are 0.0 (negative examples)
            if predicted[i] < threshold:
                tn += 1.0       # correctly predicted negative
            else:
                fp += 1.0       # incorrectly predicted positive
    rtn = [tp, fn, fp, tn]
    return rtn


#read data from uci data repository
target_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
response = urllib.request.urlopen(target_url)

# 내가 추가한 것 - 51 line에서 string 처리를 못해서 #############################
data = response.read().decode("ascii")
###############################################################################

xList = []
labels = []
tList = []

# arrange data into list for labels and list of lists for attributes
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

# divide attribute matrix and label vector into training(2/3 of data) and test sets (1/3 of data)
indices = range(len(xList))
xListTest = [xList[i] for i in indices if i%3 == 0 ]    # 1/3은 테스트용
xListTrain = [xList[i] for i in indices if i%3 != 0 ]   # 2/3은 학습용
labelsTest = [labels[i] for i in indices if i%3 == 0]   # 1/3은 테스트용
labelsTrain = [labels[i] for i in indices if i%3 != 0]  # 2/3은 학습용

# form list of list input into numpy arrays to match input class for scikit-learn linear model
# List를 numpy array 형으로 변환
xTrain = numpy.array(xListTrain) 
yTrain = numpy.array(labelsTrain) 
xTest = numpy.array(xListTest) 
yTest = numpy.array(labelsTest)

# check shapes to see what they look like
print("\nShape of xTrain array", xTrain.shape)
print("Shape of yTrain array", yTrain.shape)
print("Shape of xTest array", xTest.shape)
print("Shape of yTest array", yTest.shape)

# train linear regression class in scikit-learn linear model
# 원본 데이터 세트의 레이블을 M->-1.0과 R->0.0의 숫자로 변환한 데이터에 분류를 위한
# 학습을 한다. 그리고 최소제곱회귀분석, OLS, Ordinary leaset square regression을
# 이용하여 선형모델을 fitting. 
rocksVMinesModel = linear_model.LinearRegression()
rocksVMinesModel.fit(xTrain,yTrain)

# 학습용 데이터세트를 이용하여 학습모델의 예측 결과를 만든다. 
# generate predictions on in-sample error
trainingPredictions = rocksVMinesModel.predict(xTrain)
print("\nSome values predicted by model")
print(trainingPredictions[0:5])
print(trainingPredictions[-6:-1])
dataRow = pd.DataFrame(trainingPredictions)
dataRow.plot(color = "blue", legend = None)
plot.xlabel("Index")
plot.ylabel("predictions on in-sample error")
plot.show()

# 총 오류 수 = FP + FN, 검증 데이터(표본 외 데이터)의 오분류 오차율이 학습데이터
# (in-sample data)보다 크다. 표본 외 데이터의 성능은 새로운 관측치에 대한 예측오차
# 를 더 잘 나타낸다고 볼 수 있다.

# 결정 분기점, 최적 결정분기점 = 0.25, 이때 검증 데이터(표본 외 데이터)의 오분류 
# 오차율이 가장 작다.
# 결정분기점 (0.00) - 검증 데이터(표본 외 데이터)의 오분류 오차율 (28.57%)
# 결정분기점 (0.25) - 검증 데이터(표본 외 데이터)의 오분류 오차율 (24.29%)
# 결정분기점 (0.50) - 검증 데이터(표본 외 데이터)의 오분류 오차율 (25.71%)
# 결정분기점 (0.75) - 검증 데이터(표본 외 데이터)의 오분류 오차율 (30.00%)
# 결정분기점 (1.00) - 검증 데이터(표본 외 데이터)의 오분류 오차율 (38.57%)
decision_node_point = 0.25
# generate confusion matrix for predictions on training set (in-sample)
confusionMatTrain = confusionMatrix(trainingPredictions, yTrain, decision_node_point)
# pick threshold value and generate confusion matrix entries
tp = confusionMatTrain[0]
fn = confusionMatTrain[1]
fp = confusionMatTrain[2]
tn = confusionMatTrain[3]
print("\nConfusion Matrix of training set")
print("tp = " + str(tp) + "\tfn = " + str(fn) + "\n" + "fp = " + str(fp) + "\ttn = " + str(tn))
print ("오분류 오차율 = ", f'{(fn+fp)/(tp+fn+fp+tn)*100:.2f}', "%")

# Question
# 결정분기점이 바뀌면 trainingPredictions 이 변하는데 ROC, roc_auc는 왜 그대로?

# generate predictions on out-of-sample data (Test Set)
testPredictions = rocksVMinesModel.predict(xTest)
# generate confusion matrix from predictions on out-of-sample data
conMatTest = confusionMatrix(testPredictions, yTest, decision_node_point)
# pick threshold value and generate confusion matrix entries
tp = conMatTest[0]
fn = conMatTest[1]
fp = conMatTest[2]
tn = conMatTest[3]
print("\nConfusion Matrix of Test set")
print("tp = " + str(tp) + "\tfn = " + str(fn) + "\n" + "fp = " + str(fp) + "\ttn = " + str(tn))
print ("오분류 오차율 = ", f'{(fn+fp)/(tp+fn+fp+tn) * 100:.2f}', "%\n")

# generate ROC curve for in-sample
fpr, tpr, thresholds = roc_curve(yTrain,trainingPredictions)
roc_auc = auc(fpr, tpr)
print( 'AUC for in-sample ROC curve: %f' % roc_auc)

# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('In sample ROC rocks versus mines')
pl.legend(loc="lower right")
pl.show()

#generate ROC curve for out-of-sample
fpr, tpr, thresholds = roc_curve(yTest,testPredictions)
roc_auc = auc(fpr, tpr)
print( 'AUC for out-of-sample ROC curve: %f' % roc_auc)

# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Out-of-sample ROC rocks versus mines')
pl.legend(loc="lower right")
pl.show()