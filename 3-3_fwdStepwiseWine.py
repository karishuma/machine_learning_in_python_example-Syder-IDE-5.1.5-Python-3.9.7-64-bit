# -*- coding: utf-8 -*-

__author__ = 'mike-bowles'

import urllib
import numpy
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt

def xattrSelect(x, idxSet):
    #vtakes X matrix as list of list and returns subset containing columns in idxSet
    xOut = []
    for row in x:
        xOut.append([row[i] for i in idxSet])
    return(xOut)

# read data into iterable
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

# build list of attributes one-at-a-time - starting with empty
attributeList = []
# xList 1행의 크기(11, 속성의 갯수)로 인덱스 생성 (0~10)
index = range(len(xList[1])) 
indexSet = set(index)
indexSeq = []
oosError = []

# 단계적 전진 회귀 이
for i in index:
    attSet = set(attributeList) # attributeList에서 중복되지 않게 element 가져옴
    # attributes not in list already
    attTrySet = indexSet - attSet
    # form into list
    attTry = [ii for ii in attTrySet]
    errorList = []
    attTemp = []
    # try each attribute not in set to see which one gives least oos error
    for iTry in attTry:
        attTemp = [] + attributeList
        attTemp.append(iTry)
        # use attTemp to form training and testing sub matrices as list of lists
        xTrainTemp = xattrSelect(xListTrain, attTemp)
        xTestTemp = xattrSelect(xListTest, attTemp)
        #form into numpy arrays
        xTrain = numpy.array(xTrainTemp)
        yTrain = numpy.array(labelsTrain)
        xTest = numpy.array(xTestTemp)
        yTest = numpy.array(labelsTest)
        # use sci-kit learn linear regression
        wineQModel = linear_model.LinearRegression()
        wineQModel.fit(xTrain,yTrain)
        # use trained model to generate prediction and calculate rmsError
        rmsError = numpy.linalg.norm((yTest-wineQModel.predict(xTest)), 2) / sqrt(len(yTest))
        print(attTemp, attributeList, rmsError) # 칼럼(속성)을 붙이는 순서를 보여줌
        errorList.append(rmsError)
        attTemp = []
    # 속성을 붙일 때마다 error 리스트에서 최소값이 있는 인덱스를 찾아서
    iBest = numpy.argmin(errorList)
    # 해당 칼럼을 이전 최적 칼럼들(최적 subset)에 붙여서 1개씩 칼럼을 증가시켜 저장
    attributeList.append(attTry[iBest])
    # 선택된 (최적 subset)의 rmsError 저장
    oosError.append(errorList[iBest])

print("\nOut of sample error versus attribute set size" )
print(oosError)
# 예측 품질에 영향을 미치는 속성의 중요도 순으로 정렬 - 중요도에 따라 속성의 순위
# 부여 - 중요도가 높은 것으로 일부를 가지고 최적의 속성 집합을 만들 수 있다.
# 중요도가 낮은 것들은 배제
print("\n" + "Best attribute indices - 최적 subset으로 칼럼이 붙는 순서")
print(attributeList)
namesList = [names[i] for i in attributeList]
print("\n" + "Best attribute names - 최적 subset으로 칼럼이 붙는 순서")
print(namesList)

# Plot error versus number of attributes
# 9th와 10th 사이의 변화(성능감소)가 제일 적다.
x = range(len(oosError))
plt.plot(x, oosError, 'k')
plt.xlabel('Number of Attributes')
plt.ylabel('Error (RMS)')
plt.show()

# Plot histogram of out of sample errors for best number of attributes
# Identify index corresponding to min value, retrain with the corresponding attributes
# Use resulting model to predict against out of sample data.  Plot errors (aka residuals)
indexBest = oosError.index(min(oosError))
attributesBest = attributeList[1:(indexBest+1)]

# Define column-wise subsets of xListTrain and xListTest and convert to numpy
xTrainTemp = xattrSelect(xListTrain, attributesBest)
xTestTemp = xattrSelect(xListTest, attributesBest)
xTrain = numpy.array(xTrainTemp); xTest = numpy.array(xTestTemp)

# train and plot error histogram
wineQModel = linear_model.LinearRegression()
wineQModel.fit(xTrain,yTrain)
errorVector = yTest-wineQModel.predict(xTest)
plt.hist(errorVector)
# 에러 분포에 대한 히스토그램
plt.xlabel("Bin Boundaries - Distribution of Error")
plt.ylabel("Counts")
plt.show()

# scatter plot of actual versus predicted
plt.scatter(wineQModel.predict(xTest), yTest, s=100, alpha=0.10)
plt.xlabel('Predicted Taste Score - Predicted Label')
plt.ylabel('Actual Taste Score - Actual Label')
plt.show()