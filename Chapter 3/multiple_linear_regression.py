#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.feature_column import numeric_column
from tensorflow.feature_column import categorical_column_with_vocabulary_list


# In[2]:


featcols = [
tf.feature_column.numeric_column("area"),
tf.feature_column.categorical_column_with_vocabulary_list("type",["bungalow","apartment"])
]


# In[3]:


def train_input_fn():
    features = {"area":[1000,2000,4000,1000,2000,4000],
              "type":["bungalow","bungalow","house",
                      "apartment","apartment","apartment"]}
    labels = [ 500 , 1000 , 1500 , 700 , 1300 , 1900 ]
    return features, labels


# In[4]:


model = tf.estimator.LinearRegressor(featcols)


# In[5]:


model.train(train_input_fn, steps=200)


# In[6]:


def predict_input_fn():
    features = {"area":[1500,1800],
              "type":["house","apartment"]}
    return features

predictions = model.predict(predict_input_fn)

print(next(predictions))
print(next(predictions))


# In[ ]:




