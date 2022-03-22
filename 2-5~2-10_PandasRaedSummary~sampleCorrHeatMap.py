# -*- coding: utf-8 -*-

__author__ = 'mike-bowles'

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot
from random import uniform
from math import sqrt
import sys

#target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")

rocksVMines = pd.read_csv(target_url,header=None, prefix="V")

# 2-5 pandasReadSummarize.py
# 엑셀 형태로 읽어서 데이터 세트 통계정보 확인 ##################################
"""
print("\n")
print("Head")
print(rocksVMines.head())
print("\n")
print("Tail")
print(rocksVMines.tail())
#generate statistical summaries
summary = rocksVMines.describe()
print("\n")
print("Summary")
print(summary)
"""
###############################################################################

# 2-6 linePlots.py
# 전체 데이터 세트를 평행죄표계, parallel coordinate plot에 표시 ################
"""
for i in range(208):
    if rocksVMines.iat[i, 60] == "M":
        pcolor = "red"
    else:
        pcolor = "blue"
    # 연속된 데이터처럼 데이터 행으로 도표 그리기
    dataRow = rocksVMines.iloc[i, 0:60]
    dataRow.plot(color = pcolor)
plot.xlabel("Attribute Index")
plot.ylabel("Attribute Value")
plot.show()
"""
###############################################################################

# 2-7 corrPlot.py
# 속성의 쌍으로(속성과 레이즐을 이용하여) 대비도표, cross plot 또는 ##############
# 산점도, scatter plot 작성 ####################################################
# 각 열, column에 해당되는 60개의 속성, attribute는 60번의 다른 시간대, 주파수를
# 가진다. 따라서 인접한 시간대의 주파수 차이가 적을 것이어서 인접한 두 속성의 
# 상관관계가 더 크다. 2-3이 2-21보다 상관관계가 크다
# 수치형 속성간 상관관계 작성 - 두 속성의 상관관계가 높을수록 분포가 직선에 가까워짐
"""
dataRow2 = rocksVMines.iloc[1, 0:60]
dataRow3 = rocksVMines.iloc[2, 0:60]
plot.scatter(dataRow2, dataRow3)
plot.xlabel("2nd Attribute")
plot.ylabel("3rd Attribute")
plot.show()
dataRow21 = rocksVMines.iloc[20, 0:60]
plot.scatter(dataRow2, dataRow21)
plot.xlabel("2nd Attribute")
plot.ylabel("21th Attribute")
plot.show()
"""
###############################################################################

# 2-8 targetCorr.py
# 분류 목표와 수치 속성 간의 상관관계 ###########################################
# 35번째 속성을 사용한 이유는 평행죄표계 plot에서 보면 35번째에 속성에서 기뢰(red)
# 와 바위(blue) 간의 분리가 보이기 때문이다.
# change the targets to numeric values - M을 1로 R을 0으로 치환하여 plot
"""
target = []
for i in range(208):
    #assign 0 or 1 target value based on "M" or "R" labels
    if rocksVMines.iat[i,60] == "M":
        target.append(1.0)
    else:
        target.append(0.0)

# plot rows of data as if they were series data
# 이 경우 선을 따라 점들이 어떻게 분포되어 있는지 확인하기 어렵다.
dataRow = rocksVMines.iloc[0:208,35]
plot.scatter(dataRow, target)

plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()

# To improve the visualization, this version dithers the points a little
# and makes them somewhat transparent
# 즉, 각 값에 -0.1~0.1의 범위에서 일정하게 랜덤한 값을 더해서 세로축으로 약간 퍼지게
# 만들고 alpha=0.5 하여 점들을 불투명하게 하여 삼포도에서 중복되는 부분은 약간 검게
# 표시되도록 하였음
target = []
for i in range(208):
    # assign 0 or 1 target value based on "M" or "R" labels
    # and add some dither
    if rocksVMines.iat[i,60] == "M":
        target.append(1.0 + uniform(-0.1, 0.1))
    else:
        target.append(0.0 + uniform(-0.1, 0.1))

# plot rows of data as if they were series data
# plot에서 M=1은 왼쪽으로 몰려있고 R=0은 전체적으로 균등하게 분포하고 있어 밀도함수
# 로 보았을 때 35번째 속성이 0.5보다 작으면 M, 크면 R이라고 분류할 수 있다.
dataRow = rocksVMines.iloc[0:208,35]
plot.scatter(dataRow, target, alpha=0.5, s=120)

plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()
"""
###############################################################################

# 2-9 corrCalc.py
# 속성 2와 21의 피어슨 상관계수를 계산하여 상관관계의 정도를 계량화 ###############
# 속성 - u, v (벡터)
# /u = avg(u)
# delta(u) = u1-/u, ....., un-/u
# corr(u, v) = trans(delta(u))*delta(v) divide 
#              sqrt( trans(delta(u))*delta(v)) * trans(delta(v))*delta(u)) )
# calculate correlations between real-valued attributes
"""
dataRow2 = rocksVMines.iloc[1,0:60]
dataRow3 = rocksVMines.iloc[2,0:60]
dataRow21 = rocksVMines.iloc[20,0:60]

mean2 = 0.0; mean3 = 0.0; mean21 = 0.0
numElt = len(dataRow2)
for i in range(numElt):
    mean2 += dataRow2[i]/numElt
    mean3 += dataRow3[i]/numElt
    mean21 += dataRow21[i]/numElt

var2 = 0.0; var3 = 0.0; var21 = 0.0
for i in range(numElt):
    var2 += (dataRow2[i] - mean2) * (dataRow2[i] - mean2)/numElt
    var3 += (dataRow3[i] - mean3) * (dataRow3[i] - mean3)/numElt
    var21 += (dataRow21[i] - mean21) * (dataRow21[i] - mean21)/numElt

corr23 = 0.0; corr221 = 0.0
for i in range(numElt):
    corr23 += (dataRow2[i] - mean2) * \
              (dataRow3[i] - mean3) / (sqrt(var2*var3) * numElt)
    corr221 += (dataRow2[i] - mean2) * \
               (dataRow21[i] - mean21) / (sqrt(var2*var21) * numElt)

# 상관관계가 클수록 값이 크다.
sys.stdout.write("Correlation between attribute 2 and 3 \n")
print(corr23)
sys.stdout.write(" \n")

sys.stdout.write("Correlation between attribute 2 and 21 \n")
print(corr221)
sys.stdout.write(" \n")
"""
##############################################################################

# 2-10 sampleCorrHeatMap.py
# 히트맵을 이용하여 속성과 레이블의 상관관계 시각화 ##############################
# 데이터가 많을 때에는 피어슨 상관계수를 계산하고 계산 결과 값을 행렬로 정렬(행렬의 
# ij번째 성분은 i번째 속성과 j번째 속성의 상관계수임) 하여 히트맵으로 표시함
# 한 그룹의 속성 간 피어슨상관계수>0.7 인 경우를 다중공선성(multicollinearity)라고
# 하며 불안정한 예측이 이어질 수 있다. 목표와 상관관계는 다른 문제이며 목표와 상관
# 관계가 있다는 것은 예측관계가 있나는 것을 의미한다.
corMat = DataFrame(rocksVMines.corr())
plot.pcolor(corMat)
plot.show()
##############################################################################