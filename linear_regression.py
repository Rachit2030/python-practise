import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

data = pd.read_csv(r'train.csv')
data_test = pd.read_csv(r'test.csv')
y = np.array(data.y)
x = np.array(data.x)
x_test = np.array(data_test.x)
y_test = np.array(data_test.y)
rows = np.size(x,0)
rows_test = np.size(x_test,0)
features_array = np.empty((rows,2),dtype=np.float64)
print(y.shape)
print(features_array.shape)
for i in range(rows):
    features_array[i]=(1,x[i])
#print(features_array)
feature_array_transpose = features_array.transpose()
#print(feature_array_transpose)
print(feature_array_transpose.shape)
# X transpose * X
x_transpose_mul_x = np.matmul(feature_array_transpose,features_array)
#print(x_transpose_mul_x)
print(x_transpose_mul_x.shape)
# inverse(X transpose * X)
x_transpose_mul_x = np.linalg.inv(x_transpose_mul_x)
#print(x_transpose_mul_x)
print(x_transpose_mul_x.shape)
# X transpose * Y
x_transpose_mul_y = np.matmul(feature_array_transpose,y)
print(x_transpose_mul_y)
print(x_transpose_mul_y.shape)
# multiplying inverse(X transpose * X) with X transpose * Y
multiply_both=np.matmul(x_transpose_mul_x,x_transpose_mul_y)
print(multiply_both)
y_predicted = np.empty((rows,1))
y_error = np.empty((rows,1))
y_rms=0
for i in range(rows_test):
    y_predicted[i] = multiply_both[0] + multiply_both[1]*x_test[i]
    y_error = ((y_test[i]-y_predicted[i])**2)*0.5
    y_rms = y_rms + y_error
y_rms = math.sqrt(y_rms)
print(y_rms)