import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title of the Streamlit app
st.title('Rock vs Mine Prediction App')

# Load the dataset to a pandas DataFrame
sonar_data = pd.read_csv('sonar data.csv', header=None)

# Show dataset information and preview
if st.checkbox("Show dataset information"):
    st.write(sonar_data.head())
    st.write(f"Dataset shape: {sonar_data.shape}")

# Display statistical measures of the data
if st.checkbox("Show statistical description of data"):
    st.write(sonar_data.describe())

# Display counts of 'R' and 'M' (Rock and Mine) in the dataset
if st.checkbox("Show value counts for Rock (R) and Mine (M)"):
    st.write(sonar_data[60].value_counts())

# M stands for Mine, R stands for Rock
st.write('**Note:** "M" stands for Mine, "R" stands for Rock.')

# Grouped data by class labels
if st.checkbox("Show grouped mean by Rock/Mine"):
    st.write(sonar_data.groupby(60).mean())

# Separating data and Labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Logistic Regression Model
model = LogisticRegression()

# Train the model
model.fit(X_train, Y_train)

# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
st.write(f"Accuracy on training data: {training_data_accuracy}")

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
st.write(f"Accuracy on test data: {test_data_accuracy}")

# Function to get input data from the user through the app
def get_user_input():
    user_input = st.text_input("Enter the sonar data (comma-separated, 60 values):")
    if user_input:
        input_list = [float(i) for i in user_input.split(',')]
        return np.asarray(input_list).reshape(1, -1)
    return None

# Get user input and predict
input_data = get_user_input()

if input_data is not None and input_data.shape[1] == 60:  # Ensure proper input shape
    prediction = model.predict(input_data)

    # Show the prediction result
    if prediction[0] == 'R':
        st.success('The object is a Rock')
    else:
        st.success('The object is a Mine')
else:
    if input_data is not None:
        st.warning("Please enter valid sonar data (60 comma-separated values).")

# Display dataset shape
st.write(f"Dataset Shape: {sonar_data.shape}")
