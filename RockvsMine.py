#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


# Load the dataset to a pandas DataFrame
sonar_data = pd.read_csv('sonar data.csv', header=None)


# In[3]:


# Display the first few rows of the DataFrame
sonar_data.head()


# In[4]:


# number of rows and columns
sonar_data.shape


# In[5]:


#statistical measures of the data
sonar_data.describe() 


# In[6]:


sonar_data[60].value_counts()


# M stands for Mine,
# R stands for Rock

# In[7]:


sonar_data.groupby(60).mean()


# In[8]:


# separating data and Labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]


# In[9]:


print(X)
print(Y)


# In[10]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)


# In[11]:


print(X.shape, X_train.shape, X_test.shape)


# In[12]:


print(X_train)
print(Y_train)


# In[13]:


model = LogisticRegression()


# In[14]:


#training the Logistic Regression model with training data
model.fit(X_train, Y_train)


# In[15]:


#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) 


# In[16]:


print('Accuracy on training data : ', training_data_accuracy)


# In[17]:


#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test) 


# In[21]:


print('Accuracy on test data : ', test_data_accuracy)


# In[22]:


import numpy as np

# Function to get input data from the user
def get_user_input():
    user_input = input("Enter the sonar data (comma-separated): ")
    # Convert the input string into a list of floats
    input_list = [float(i) for i in user_input.split(',')]
    return input_list

# Get user input data
input_data = get_user_input()

# Changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Assuming 'model' is a pre-trained model loaded earlier
# Predict the class
prediction = model.predict(input_data_reshaped)

# Print the prediction result
if prediction[0] == 'R':
    print('The object is a Rock')
else:
    print('The object is a Mine')


# In[ ]:




