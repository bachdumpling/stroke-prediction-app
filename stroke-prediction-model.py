# Description: Trains a logistic regression model to predict stroke risk based on healthcare data.
'''
The code performs the following steps:
1. Imports the necessary libraries and modules.
2. Defines the preprocess_data function that reads the CSV file, encodes categorical variables using LabelEncoder, imputes missing values using SimpleImputer, and returns the preprocessed data, label encoder, and imputer.
3. Defines the train_model function that selects the key features, splits the data into training and testing sets, scales the features using StandardScaler, trains a logistic regression model with class weights, and returns the trained model and scaler.
4. Specifies the file path for the dataset.
5. Calls the preprocess_data function to preprocess the data.
6. Calls the train_model function to train the model.
7. Uses joblib.dump to save the trained model, scaler, label encoder, and imputer for later use.

This code trains a logistic regression model to predict stroke risk based on various features in the healthcare dataset. The model and preprocessors are saved using joblib for future use in making predictions on new data.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

def preprocess_data(file_path):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path)
    
    # Initialize a LabelEncoder for encoding categorical variables
    label_encoder = LabelEncoder()
    
    # Define the columns that contain categorical data
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    
    # Encode the categorical columns using LabelEncoder
    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column].astype(str))
    
    # Initialize a SimpleImputer for handling missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    # Impute missing values in the 'bmi' and 'avg_glucose_level' columns
    data[['bmi', 'avg_glucose_level']] = imputer.fit_transform(data[['bmi', 'avg_glucose_level']])
    
    # Return the preprocessed data, label encoder, and imputer
    return data, label_encoder, imputer

def train_model(data):
    # Define the key features for training the model
    key_features = ['age', 'hypertension', 'avg_glucose_level', 'bmi', 'ever_married', 'work_type']
    
    # Split the data into features (X) and target variable (y)
    X = data[key_features]
    y = data['stroke']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize a StandardScaler for scaling the features
    scaler = StandardScaler()
    
    # Scale the training features
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Initialize a logistic regression model with class weights
    model = LogisticRegression(random_state=42, class_weight='balanced')
    
    # Train the model on the scaled training data
    model.fit(X_train_scaled, y_train)
    
    # Return the trained model and scaler
    return model, scaler

# Run these functions and save the artifacts
file_path = 'healthcare-dataset-stroke-data.csv'

# Preprocess the data
data, label_encoder, imputer = preprocess_data(file_path)

# Train the model
model, scaler = train_model(data)

# Save the trained model, scaler, label encoder, and imputer using joblib
joblib.dump((model, scaler, label_encoder, imputer), 'stroke_model_pipeline.joblib')