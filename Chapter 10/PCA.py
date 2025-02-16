#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# In[2]:


# Load data
((x_train, y_train), (_, _)) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.
x_train = x_train.astype(np.float32)

x_train = np.reshape(x_train, (x_train.shape[0], 784))


# In[3]:


mean = x_train.mean(axis = 1)

x_train = x_train - mean[:,None]


# In[4]:


s, u, v = tf.linalg.svd(x_train)


# In[5]:


s = tf.linalg.diag(s)


# In[6]:


print("Diagonal matrix shape: {} \nLeft Singular Matrix shape: {} \nRight Singular matrix shape: {}".
     format(s.shape,u.shape,v.shape))


# In[7]:


k = 3
pca = tf.matmul(u[:,0:k], s[0:k,0:k])


# In[8]:


print('original data shape',x_train.shape)
print('reduced data shape', pca.shape) 


# In[9]:


Set = sns.color_palette("Set2", 10)
color_mapping = {key:value for (key,value) in enumerate(Set)}
colors = list(map(lambda x: color_mapping[x], y_train))
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pca[:, 0], pca[:, 1],pca[:, 2], c=colors)
 


# In[ ]:




