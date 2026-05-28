#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("Advertising.csv")

df.head()


# In[3]:


df.info()


# In[4]:


sns.pairplot(df)
plt.show()


# In[5]:


X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

print("Model trained successfully!")


# In[6]:


y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)


# In[7]:


sample_prediction = model.predict([[230.1, 37.8, 69.2]])

print("Predicted Sales:", sample_prediction[0])


# In[8]:


sample_prediction = model.predict([[230.1, 37.8, 69.2]])

print("Predicted Sales:", sample_prediction[0])


# In[ ]:




