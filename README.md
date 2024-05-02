# Stroke Prediction App

This repository contains a Stroke Prediction App that uses machine learning to predict the risk of stroke based on various health and demographic factors. The app is built using Streamlit, a Python library for creating interactive web applications.

#### Visit the app in production here: [https://stroke-prediction-app.streamlit.app/](https://stroke-prediction-app.streamlit.app/)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Running the Model](#running-the-model)
  - [Running the App](#running-the-app)
- [File Descriptions](#file-descriptions)
- [Dataset](#dataset)
- [Model](#model)
- [App](#app)
- [Contributing](#contributing)

## Installation

To run the Stroke Prediction App locally, please follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/stroke-prediction-app.git
   ```

2. Navigate to the project directory:
   ```
   cd stroke-prediction-app
   ```

3. Create a virtual environment (optional but recommended):
   ```
   python -m venv env
   source env/bin/activate  # For Unix/Linux
   env\Scripts\activate.bat  # For Windows
   ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Model

To train the stroke prediction model, run the following command:
```
python stroke-prediction-model.py
```

This will train the model using the dataset specified in `healthcare-dataset-stroke-data.csv` and save the trained model and preprocessors in `stroke_model_pipeline.joblib`.

### Running the App

To run the Stroke Prediction App, use the following command:
```
streamlit run stroke-prediction-app.py
```

This will start the Streamlit app in your default web browser. You can interact with the app by entering the required information and clicking the "Predict Stroke Risk" button to get the prediction result.

## File Descriptions

- `healthcare-dataset-stroke-data.csv`: The dataset used for training the stroke prediction model.
- `requirements.txt`: The file containing the list of required Python packages and their versions.
- `stroke_model_pipeline.joblib`: The saved trained model and preprocessors.
- `stroke-prediction-app.py`: The Streamlit app script for the Stroke Prediction App.
- `stroke-prediction-model.py`: The script for training the stroke prediction model.

## Dataset

The dataset used for training the stroke prediction model is stored in the `healthcare-dataset-stroke-data.csv` file. It contains various health and demographic factors along with the corresponding stroke outcome.

## Model

The stroke prediction model is trained using logistic regression. The model is trained on the provided dataset and saved in the `stroke_model_pipeline.joblib` file, along with the necessary preprocessors.

## App

The Stroke Prediction App is built using Streamlit. It provides a user-friendly interface for entering health and demographic information and predicting the risk of stroke based on the trained model.

## Contributing

Contributions to the Stroke Prediction App are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
