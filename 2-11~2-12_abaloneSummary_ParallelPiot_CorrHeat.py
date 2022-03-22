# -*- coding: utf-8 -*-

__author__ = 'mike-bowles'

import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plot
from math import exp

target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases"
              "/abalone/abalone.data")

# read abalone data
abalone = pd.read_csv(target_url,header=None, prefix="V")
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                   'Shucked weight', 'Viscera weight', 'Shell weight',
                   'Rings']

print(abalone.head())
print(abalone.tail())

# print summary of data frame
summary = abalone.describe()
print(summary)

# 2-11 abaloneSummary.py
# 상자와 수염도표, box and whisker plot 으로 데이터 표시 ########################
"""
# 상자 안의 붉은 선 - 중앙값, 50번째 100분위수
# 상자 윗 라인 - 75번째 100분위수 (3사분위수)
# 상자 아래 라인 - 25번째 백분위수 (1사분위수)
# 상자 위, 아래의 수염은 상자 4분위 범위, 즉 25분위~75분위까지의 간격 = 상자의 높이
# 의 1.4배 - 배수는 조정 가능. 수염에서 멀리 떨어진 것은 아웃라이어로 볼 수 있음.

# box plot the real-valued attributes convert to array for plot routine
array = abalone.iloc[:,1:9].values
plot.boxplot(array)
plot.xlabel("Attribute Index")
plot.ylabel(("Quartile Ranges"))
plot.show()

# the last column (rings) is out of scale with the rest - remove and replot
# 8번, 나이테 속성을 제외하고 보면 1~7까지의 분포가 더 자세히 보임

array2 = abalone.iloc[:,1:8].values
plot.boxplot(array2)
plot.xlabel("Attribute Index")
plot.ylabel(("Quartile Ranges"))
plot.show()

# removing is okay but renormalizing the variables generalizes better.
# renormalize columns to zero mean and unit standard deviation
# this is a common normalization and desirable for other operations
# (like k-means clustering or k-nearest neighbors
# 정규화는 개별 칼럼을 중심화, centering하고 척도화, scaling하여 각 속성의 측정단위
# 를 동일하게 만드는 것.   

abaloneNormalized = abalone.iloc[:,1:9]

for i in range(8):
    mean = summary.iloc[1, i]
    sd = summary.iloc[2, i]
    abaloneNormalized.iloc[:,i:(i + 1)] = (
                    abaloneNormalized.iloc[:,i:(i + 1)] - mean) / sd

array3 = abaloneNormalized.values
plot.boxplot(array3)
plot.xlabel("Attribute Index")
plot.ylabel(("Quartile Ranges - Normalized "))
plot.show()
"""
##############################################################################

# 2-11 abaloneParallelPlot.py
# 전복의 나이를 알아내는 문제에서 변수, 즉 속성 간의 또는 속성과 레이블 간의 상관
# 관계에 대한 아이디어를 얻기 위하여 칼라로 표신된 평향좌표로 시각화 - 회귀 문제를 
# 풀기 위한 평행도표 이용 - 목적값을 기준으로 높은가, 낮은가에 따라 색으로 표시
# 숫자에 색을 부여하기 위해 [0.0, 1.0] 구간으로 압축

# get summary to use for scaling
minRings = summary.iloc[3,7] # min value of the number of rings
maxRings = summary.iloc[7,7] # max value of the number of rings
nrows = len(abalone.index)

# 전복의 나이테 수와 각 속성간의 직접적인 상관관계 나타냄
# 색채 스캐일은 어두운 붉은 갈색 - 노랑 - 옅은 파랑 - 진한 파랑의 순서로 변화
# summary에서 얻은 나이테 수의 min, max 값을 이용하여 각 행의 나이테 수를 
# [0.0, 1.0] 구간으로 압축함.
# 데이터 전체가 min~max 사이에서 넓게 퍼져있고 스케을을 축약하였기 때문에 대부분의
# 데이터가 색채 스케일 상의 중간쯤에 위치하게 된다. 그럼에도 불구하고, 각 속성과
# 각 표본에서 측정된 나이테의 개수 간에 유의미한 상관관계가 있다는 것을 보여줌.
# 비슷한 색채는 여러개의 속성의 유사한 값이 묶여져 있다. 
# 속성과 나이테의 수가 상관관계가 없다면 세로축 방향으로 color가 들쭉날쭉하게 나올
# 것임. 그러나, 세로축 방향으로 color가 색채 스캐일에 따라 이어지듯이 변하기 때문에
# 속성과 나이테 수의 상관관계 있다고 볼 수 있음.
# 짙은 오렌지 색 구역에 섞여있는 흐릿한 푸른색 선은 정확하게 예측하기 어려운 표본,
# 즉 속성과 나이테 수의 상관관계가 없는 셈플에 대한 측정이 있다는 것을 의미함.
for i in range(nrows):
    # plot rows of data as if they were series data
    dataRow = abalone.iloc[i,1:8]
    # 각 행의 나이테 수에 따라 plotting 할 8개 데이터의 color 결정
    labelColor = (abalone.iloc[i,8] - minRings) / (maxRings - minRings)
    dataRow.plot(color=plot.cm.RdYlBu(labelColor), alpha=0.5)

plot.xlabel("Attribute Index")
plot.xticks(fontsize=6)
plot.ylabel("Attribute Values")
plot.show()

meanRings = summary.iloc[1,7]
sdRings = summary.iloc[2,7]

# renormalize using mean and standard variation, then compress
# with logit function - Sigmoid Function = 1 / (1 + exp(-x))
# 아주 큰 음의 값은 거의 0으로 변환하고 아주 큰 양의 값은 거의 1로 변환하며 모든
# 범위의 색채를 잘(골고루) 사용하게 됨.
# 총 중량과 순 중량 구역에 옅은 푸른 선과 섞여있는 나이테 수가 많은 전복 견본에 해당
# 하는 아주 짙은 푸른선-노란색과 붉은색 견본이 몇 개 있음을 보여준다.
for i in range(nrows):
    # plot rows of data as if they were series data
    dataRow = abalone.iloc[i,1:8]
    normTarget = (abalone.iloc[i,8] - meanRings)/sdRings
    labelColor = 1.0/(1.0 + exp(-normTarget))
    dataRow.plot(color=plot.cm.RdYlBu(labelColor), alpha=0.5)

plot.xlabel("Attribute Index")
plot.xticks(fontsize=6)
plot.ylabel(("Attribute Values"))
plot.show()

###############################################################################

# 2-12 abaloneCorrHeat.py
# 상관관계 히트맵 사용 - 한 쌍의 상관관계 시각화 #################################
# 붉은 색은 높은 상관관계 푸른색은 낮은 상관관계를 나타내며 히트맵에서 가장 윗 행과
# 가장 오른쪽 칼럼이 목적값(나이테 수)가 된다. 따라서 여기서 푸른색은 목적값과 낮은
# 상관관계가 있다는 의미이다. 푸른섹은 전복중량과 나이테 수의 상관관계에 해당하며 
# 평행 좌표 도표와 동일한 결과를 확인 할 수 있다.
#calculate correlation matrix
corMat = DataFrame(abalone.iloc[:,1:9].corr())
#print correlation matrix
print(corMat)

#visualize correlations using heatmap
plot.pcolor(corMat)
plot.show()
###############################################################################