#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics

marks = {"CAT1":[12,13,20,19,17,18,24], "CAT2":[14,13,19,20,15,17,16]}
data = pd.DataFrame(marks)
#print(data)
plt.scatter(data['CAT1'], data['CAT2'], c='black')
plt.xlabel("CAT1")
plt.ylabel("CAT2")
plt.show()

X = data['CAT1'].values.reshape(-1,1)
y = data['CAT2'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(X, y)
print("LINEAR MODEL:Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

predictions = reg.predict(X)
df = pd.DataFrame({'ACTUAL': data['CAT2'], 'PREDICTED': predictions.flatten()})

plt.scatter(
    data['CAT1'],
    data['CAT2'],
    c='red'
)
plt.plot(
    data['CAT1'],
    predictions,
    c='yellow',
    linewidth=2
)
plt.show()

print('MEAN ABSOLUTE ERROR:', metrics.mean_absolute_error(data['CAT2'], predictions))
print('MEAN SQUARED ERROR:', metrics.mean_squared_error(data['CAT2'], predictions))
print('MEAN ROOT SQUARED ERROR:', np.sqrt(metrics.mean_squared_error(data['CAT2'], predictions)))

