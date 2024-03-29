import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r"data.csv")
print(data)
y1=np.array(data.y1)
y2=np.array(data.y2)
x1=np.array(data.x1)
x2=np.array(data.x2)
x3=np.array(data.x3)
y_vector = np.empty((5,2),dtype=float)
feature_vector = np.empty((5,4),dtype=float)
print(x1.size)
for i in range(5):
    feature_vector[i]=([1,x1[i],x2[i],x3[i]])
for i in range(5):
    y_vector[i]=([y1[i],y2[i]])
print(feature_vector)
feature_vector_transpose = feature_vector.transpose()
print(feature_vector_transpose)
feature_vector = np.matmul(feature_vector_transpose,feature_vector)
print(feature_vector)
feature_vector = np.linalg.inv(feature_vector)
print(feature_vector)
y = np.matmul(feature_vector_transpose,y_vector)
print(y)
beta_feature = np.matmul(feature_vector,y)
print(beta_feature)
# print(f'The equation of multiple linear regression is y = {beta_feature[1]} x1 + {beta_feature[2]} x2 + {beta_feature[0]}')
# x1_surf = np.array(np.meshgrid(np.linspace(data.x1.min(),data.x1.max(),100)))
# x2_surf = np.array(np.meshgrid(np.linspace(data.x2.min(),data.x2.max(),100)))
# print(x2_surf.shape)
