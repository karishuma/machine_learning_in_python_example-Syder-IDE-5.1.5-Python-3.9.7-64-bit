# -*- coding: utf-8 -*-

__author__ = 'mike-bowles'

import urllib
import numpy
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt

#read data into iterable
target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
response = urllib.request.urlopen(target_url)

# 내가 추가한 것 - 51 line에서 string 처리를 못해서 #############################
data = response.read().decode("ascii")
###############################################################################

xList = []
labels = []
names = []
firstLine = True
for line in data.split('\n'):
    # 칼럼 헤더 추출
    if firstLine:
        names = line.strip().split(";")
        firstLine = False
    else:
        # split on semi-colon & restore float-typed data in data frame
        row = line.strip().split(";")
        if len(row) > 1:
            # put labels in separate array
            labels.append(float(row[-1]))
            # remove label from row
            row.pop()
            # convert row to floats
            floatRow = [float(num) for num in row]
            xList.append(floatRow)

# divide attributes and labels into training and test sets
indices = range(len(xList))
xListTest = [xList[i] for i in indices if i%3 == 0 ]
xListTrain = [xList[i] for i in indices if i%3 != 0 ]
labelsTest = [labels[i] for i in indices if i%3 == 0]
labelsTrain = [labels[i] for i in indices if i%3 != 0]

xTrain = numpy.array(xListTrain); yTrain = numpy.array(labelsTrain); xTest = numpy.array(xListTest); yTest = numpy.array(labelsTest)

# 리지 복잡도 파라미터, alpha를 이용한 RMSE 도표 생성 ###########################
# 큰값(가장 복잡도가 낮은 모델)을 왼쪽 -> 작은값(가장 복잡도가 높은 모델)을 오른쪽
# 단계적 전진 회귀와 특성이 비슷하지만 리지회귀가 조금 더 낮다.
alphaList = [0.1**i for i in [0,1, 2, 3, 4, 5, 6]]
rmsError = []
for alph in alphaList:
    wineRidgeModel = linear_model.Ridge(alpha=alph)
    wineRidgeModel.fit(xTrain, yTrain)
    rmsError.append(numpy.linalg.norm((yTest-wineRidgeModel.predict(xTest)), 2)/sqrt(len(yTest)))

print("RMS Error             alpha")
for i in range(len(rmsError)):
    print(rmsError[i], alphaList[i])

#plot curve of out-of-sample error versus alpha
x = range(len(rmsError))
plt.plot(x, rmsError, 'k')
plt.xlabel('-log(alpha)')
plt.ylabel('Error (RMS)')
plt.show()

#Plot histogram of out of sample errors for best alpha value and scatter plot of actual versus predicted
#Identify index corresponding to min value, retrain with the corresponding value of alpha
#Use resulting model to predict against out of sample data.  Plot errors (aka residuals)
indexBest = rmsError.index(min(rmsError))
alph = alphaList[indexBest]
wineRidgeModel = linear_model.Ridge(alpha=alph)
wineRidgeModel.fit(xTrain, yTrain)
errorVector = yTest-wineRidgeModel.predict(xTest)
plt.hist(errorVector)
plt.xlabel("Bin Boundaries")
plt.ylabel("Counts")
plt.show()

plt.scatter(wineRidgeModel.predict(xTest), yTest, s=100, alpha=0.10)
plt.xlabel('Predicted Taste Score')
plt.ylabel('Actual Taste Score')
plt.show()