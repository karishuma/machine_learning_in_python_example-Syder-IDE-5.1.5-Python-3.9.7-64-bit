# -*- coding: utf-8 -*-

__author__ = 'mike-bowles'

# 유리의 종류를 분류하는 다중 분류 문제 #########################################
# 결과가 2개가 아니라 여러개라는 것을 빼면 이항분류문제와 비슷 (바위, 기뢰 분류)
# wine 문제도 몇 개의 가능한 결과(테이스트 점수는 3~8 인 정수)가 있지만 순서관계가
# 5점은 3점보다 높지만 8점보다 낮다. 그러나 다중 클래스 문제에는 순서가 없다.
###############################################################################

import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plot

target_url = ("https://archive.ics.uci.edu/ml/machine-"
              "learning-databases/glass/glass.data")

glass = pd.read_csv(target_url,header=None, prefix="V")
glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si',
                 'K', 'Ca', 'Ba', 'Fe', 'Type']

print(glass.head())

#generate statistical summaries
summary = glass.describe()
print(summary)

# 2-15 glassSummary.py
# 정규화된 데이터 상자의 도표 ###################################################
# 분류 문제이기 때문에 아웃라이어의 수가 상당히 많다. 속성 값과 클래스 소속 간에는 
# 연속적인 관계가 필요하지 않다. 즉, 클래스 각각에 대하여 속성값에 따른 기대 근접도
# 가 필요없다. (여러 클래스 중 하나로 결정된다는 의미, 클래스 각각에 대해 하나의 표
# 본이 몇 % 확률로 각 클래스가 될 것인지 계산할 필요가 없다는 뜻이다.)
# 이 데이터의 또 다른 특성은 불균형, 즉 각 클래스의 표본이 9~76까지 다양하다. 대개
# 통계량은 가장 표본이 많은 클래스의 속성값에 영향을 받으며 다른 클래스의 멤버가 비 
# 숫한 속성을 가질 것이란 기대치도 없다.
"""
ncol1 = len(glass.columns)
glassNormalized = glass.iloc[:, 1:ncol1]
ncol2 = len(glassNormalized.columns)
summary2 = glassNormalized.describe()
for i in range(ncol2):
    mean = summary2.iloc[1, i]
    sd = summary2.iloc[2, i]
    glassNormalized.iloc[:,i:(i + 1)] = \
        (glassNormalized.iloc[:,i:(i + 1)] - mean) / sd
array = glassNormalized.values
plot.boxplot(array)
plot.xlabel("Attribute Index")
plot.ylabel(("Quartile Ranges - Normalized "))
plot.show()
"""
###############################################################################

# 2-16 glassParallelPlot.py
# 평행좌표도표 #################################################################
# 데이터는 각각의 가능한 결과 분류(6가지)에 따라 구분되는 색을 이용하여 그려짐.
# 레이블은 1~7까지 이지만 4가 없다. 
# 바위/기뢰 문제에서 평행좌표도표의 색은 2가지. 이는 두 개의 다른 레이블을 의미함.
# 와인문제와 전복문제의 회귀분석은 레이블이 어떤 수치 값일 수 있었다. 이에 대한 선은
# 다른 색의 범위에 따라 그려졌다. 
# 어두운 푸른 선 : 많은 속성에 대해 잘 모이고 다른 클래스와 잘 분리되어 있지만 몇몇
#                 속성에서는 데이터의 끝단에 위치 (해당 속성에는 아웃라이어)
# 밝은 푸른 선 : 어두운 푸른 선보다 수는 적고 모든 속성은 아니어도 어두운 푸른 선처
#               럼 끝단에 위치한다.
# 가운데 갈색 선 : 그룹으로 잘 모여 있지만 값의 중간 쯤에 위치
"""
glassNormalized = glass
ncols = len(glassNormalized.columns)
nrows = len(glassNormalized.index)
summary = glassNormalized.describe()
nDataCol = ncols - 1

# normalize except for labels
for i in range(ncols - 1):
    mean = summary.iloc[1, i]
    sd = summary.iloc[2, i]
    glassNormalized.iloc[:,i:(i + 1)] = \
        (glassNormalized.iloc[:,i:(i + 1)] - mean) / sd

# Plot Parallel Coordinate Graph with normalized values
for i in range(nrows):
    #plot rows of data as if they were series data
    dataRow = glassNormalized.iloc[i,1:nDataCol]
    labelColor = glassNormalized.iloc[i,nDataCol]/7.0
    dataRow.plot(color=plot.cm.RdYlBu(labelColor), alpha=0.5)

plot.xlabel("Attribute Index")
plot.ylabel(("Attribute Values"))
plot.show()
"""
###############################################################################

# Draw Heatmap for Figure 2-22
# 유리 데이터 상관관계의 히트맵 #################################################
# 속성 간의 상관관계가 낮다는 것을 볼 수 있다. 즉, 각 속성은 대부분 독립적이라는 의
# 미로 좋은 것이다. 이 문제의 목표는 구분되는 색 중 하나이기 때문에 상관관계 지도에
# 포함하지 않는다.이 때문에 상관관계 히트맵이 설명하는 영역은 제한된다.
ncols = len(glass.columns)
# calculate correlation matrix
corMat = DataFrame(glass.iloc[:, 1:(ncols - 1)].corr())
# visualize correlations using heatmap
plot.pcolor(corMat)
plot.show()
###############################################################################