# RockvsMine-prediction
Rock vs Mine Prediction App
===========================

A machine learning application for predicting whether an object is a rock or a mine using SONAR data.

![image](https://github.com/user-attachments/assets/ba74ab98-1014-4453-8bb2-9bf61577513f)


This project uses a **Logistic Regression Model** to classify objects based on the SONAR dataset. The Rock\_vs\_Mine\_Predictor.ipynb file contains the source code for data preprocessing, model training, and testing. After training, the model is saved for convenient reuse. The web interface for this project is developed using **Streamlit**, providing a simple and interactive way to make predictions.

Features:
---------

*   **Data Preprocessing:** The SONAR data is preprocessed and split into training and testing sets.
    
*   **Model Training:** A Logistic Regression model is trained on the preprocessed data to predict whether the object is a rock or a mine.
    
*   **Model Evaluation:** The model’s accuracy is calculated, and its performance is assessed using standard metrics.
    
*   **User Interface:** Streamlit is used to create a web interface where users can upload data for prediction.
    

Dataset:
--------

The dataset used is the SONAR dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)). It consists of 208 instances with 60 features, which represent the sonar signals that bounce off either rocks or mines.

Using the Model:
----------------

The model is deployed through a web interface where users can upload SONAR data in CSV format and get a prediction of whether the object is a rock or a mine.



Note:
-----

This project is developed for **educational** purposes and is not intended for commercial use. For any issues or feedback, feel free to reach out.
