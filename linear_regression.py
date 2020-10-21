import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'test.csv')
y = np.array(data.y)
x = np.array(data.x)
rows = np.size(x,0)
features_array = np.empty((rows,2))
print(features_array.shape)
for i in range(rows):
    features_array[i]=(1,x[i])
print(features_array)
for i in range(rows):
    for j in range(columns):
