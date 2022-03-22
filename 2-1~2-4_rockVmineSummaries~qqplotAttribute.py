# -*- coding: utf-8 -*-

__author__ = 'mike-bowles'

import urllib
import sys
import numpy as np
import pylab
import scipy.stats as stats

#read data from uci data repository
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
response = urllib.request.urlopen(target_url)

tList = []
xList = []
labels = []

# 내가 추가한 것 - 26 line에서 string 처리를 못해서 #############################
data = response.read().decode("ascii")
###############################################################################

# 2-1 rockVmineSummaries.py
# 데이터 세트의 행과 열 카운트 ##################################################
# 레이블을 리스트로, 속성을 리스트의 리스트로 데이터 정리 ########################
aa=0
for line in data.split('\n'):
    if len(line) > 0:
        idx = 0
        for row in line.split(','):
            tList.insert(idx, row)
            idx = idx + 1    
        aList = tList.copy()
        xList.append(aList)
        aa = aa +1
        if aa == 208:
            aa = 209
        else:
            tList.clear()
print('\n')    
sys.stdout.write("Number of Rows of Data = " + str(len(xList)) + '\n')          # 208
sys.stdout.write("Number of Columns of Data = " + str(len(xList[1])) + '\n')    # 61
print('\n')
###############################################################################

nrow = len(xList)
ncol = len(xList[1])

type = [0]*3
colCounts = []

# 2-2 rockVmineContnts.py
# 데이터 세트에서 수치형 열과 범주형 열이 볓 개인지 확인 #########################
# 데이터 세트는 숫자(정수 또는 실수)인 경우, 문자열인 경우, 비어 있는 경우로 구성
# 되어 있으며 각 경우를 카운트 ################################################## 

for col in range(ncol):
    for row in xList:
        try:
            a = float(row[col])
            if isinstance(a, float):
                type[0] += 1
        except ValueError:
            if len(row[col]) > 0:
                type[1] += 1
            else:
                type[2] += 1
    colCounts.append(type)
    type = [0] * 3

sys.stdout.write("Col#" + '\t' + "Number" + '\t' + "Strings" + '\t' + "Other\n")

iCol = 0
for types in colCounts:
    sys.stdout.write(str(iCol) + '\t\t' + str(types[0]) + '\t\t' + str(types[1]) + '\t\t' + str(types[2]) + "\n")
    iCol += 1
print('\n')
###############################################################################    

# 세번째 칼럼에 대한 통계량 생성, generate summary statistics for column 3 ######
col = 3
colData = []

for row in xList:
    colData.append(float(row[col]))

# 2-4 qqplotAttribute.py 
# 3번빼 칼럼에 대한 Q-Q 도표 display
stats.probplot(colData, dist="norm", plot=pylab)
pylab.show()


# 2-3 rVMSummaryStats.py
colArray = np.array(colData)
colMean = np.mean(colArray)
colsd = np.std(colArray)
# 선택한 속성에 대한 평균과 표준편차의 통계량을 통하여 제작할 모델에 대한 통찰력을
# 높여준다.
sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' + "Standard Deviation = " + '\t ' + str(colsd) + "\n")

# 분위수 경계 계산, calculate quantile boundaries
ntiles = 4
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))

sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")

# 10개의 동일한 간격으로 수행, run again with 10 equal intervals
ntiles = 10

percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))

sys.stdout.write("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")

# 최종 칼럼에는 범주형 변수 포함, 
# The last column contains categorical variables
col = 60
colData = []
for row in xList:
    colData.append(row[col])

unique = set(colData)
sys.stdout.write("Unique Label Values \n")
print(unique)

# 개별 값이 있는 요소의 수를 오름 셈
# count up the number of elements having each value
catDict = dict(zip(list(unique),range(len(unique))))

catCount = [0]*2

for elt in colData:
    catCount[catDict[elt]] += 1

# 레이블 - M은 Mine, R은 Rock
sys.stdout.write("\nCounts for Each Value of Categorical Label \n")
print(list(unique))
print(catCount)
###############################################################################