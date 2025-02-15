#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%matplotlib inline

# Generate a random data
np.random.seed(0)
area = 2.5 * np.random.randn(100) + 25
price = 25 * area + 5 + np.random.randint(20,50, size = len(area))


# In[1]:


data = np.array([area, price])
data = pd.DataFrame(data = data.T, columns=['area','price'])


# In[3]:


print(data.head())


# In[4]:


plt.scatter(data['area'], data['price'])
plt.show()


# In[5]:


W = sum(price*(area-np.mean(area))) / sum((area-np.mean(area))**2)
b = np.mean(price) - W*np.mean(area)


# In[6]:


print("The regression coefficients are", W,b)


# In[7]:


# predicted values
y_pred = W * area + b


# In[8]:


plt.plot(area, y_pred, color='red',label="Predicted Price")
plt.scatter(data['area'], data['price'], label="Training Data")
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.show()


# In[ ]:




