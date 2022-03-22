# -*- coding: utf-8 -*-


__author__ = 'mike-bowles'

import math

# 결과값, 목적값
target = [1.5, 2.1, 3.3, -4.7, -2.3, 0.75]
# 계산값
prediction = [0.5, 1.5, 2.1, -2.2, 0.1, -0.5]

error = []
for i in range(len(target)):
    error.append(target[i] - prediction[i])
print("Errors : ", error)

SquareError = []
AbsError = []
for val in error:
    SquareError.append(val*val)
    AbsError.append(abs(val))
print("Squared Error : ", SquareError)
print("Absolte value of Errore : ", AbsError)

MSE = sum(SquareError) / len(SquareError)
print("MSE = ", MSE)
print("RMSE = ", math.sqrt(MSE))

MAE = sum(AbsError) / len(AbsError)
print("MAE = ", MAE)

# 예측오차의 MSE와 타겟의 분산 또는 타겟 표준편차와 RMSE가 비슷하다면 예측 알고리즘
# 의 성능이 안 좋은 것이며 이 경우 예측 알고리즘을 대체하여 타겟의 평균을 단순 계산
# 하더라도 비슷한 수준일 것이다. 여기서의 오차는 RSME가 타겟의 표준편차의 1/2 수준
# 이므로 높은 성능을 의미한다. 
TargetDeviation = []
TargetMean = sum(target) / len(target)
for val in target:
    TargetDeviation.append((val - TargetMean) * (val - TargetMean))
TargetVariance = sum(TargetDeviation) / len(TargetDeviation)
print("Target Variance : ", TargetVariance)
print("Target Standard Deviation : ", math.sqrt(TargetVariance))