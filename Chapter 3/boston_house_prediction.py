#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
from tensorflow import feature_column as fc 
from tensorflow.keras.datasets import boston_housing

get_ipython().run_line_magic('load_ext', 'tensorboard')
# Clear any logs from previous runs
get_ipython().system('rm -rf ./logs/')


# In[2]:


tf.__version__


# In[3]:


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


# In[4]:


features = ['CRIM', 'ZN', 'INDUS','CHAS','NOX','RM','AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
 
x_train_df = pd.DataFrame(x_train, columns= features)
x_test_df = pd.DataFrame(x_test, columns= features)
y_train_df = pd.DataFrame(y_train, columns=['MEDV'])
y_test_df = pd.DataFrame(y_test, columns=['MEDV'])
x_train_df.head()


# In[5]:


feature_columns = []
for feature_name in features:
    feature_columns.append(fc.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)


# In[6]:


def estimator_input_fn(df_data, df_label, epochs=20, shuffle=True, batch_size=64):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(df_data), df_label))
        if shuffle:
            ds = ds.shuffle(100)
        ds = ds.batch(batch_size).repeat(epochs)
        return ds
    return input_function

train_input_fn = estimator_input_fn(x_train_df, y_train_df)
val_input_fn = estimator_input_fn(x_test_df, y_test_df, epochs=1, shuffle=False)


# In[7]:


linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns, model_dir = 'logs/func/')
linear_est.train(train_input_fn, steps=100)
result = linear_est.evaluate(val_input_fn)
print(result)


# In[8]:


result = linear_est.predict(val_input_fn)


# In[9]:


for pred,exp in zip(result, y_test[:32]):
    print("Predicted Value: ", pred['predictions'][0], "Expected: ", exp)


# In[10]:


get_ipython().run_line_magic('tensorboard', '--logdir logs/func')


# In[ ]:




