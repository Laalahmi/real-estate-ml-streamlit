Real Estate Price Prediction – ML Deployment with Streamlit

Live Application

You can access the deployed application here:

https://real-estate-ml-app-wu3lxwd9gqwc5ikf8kfyni.streamlit.app/

Overview

This project implements a machine learning pipeline to predict real estate prices based on various property attributes such as house size, number of rooms, location indicators, and amenities.

The machine learning workflow was first developed in a Jupyter Notebook and later modularized into reusable Python modules. The trained model is deployed as an interactive web application using Streamlit.

This project demonstrates important concepts in machine learning engineering including:

Code modularization

Feature engineering

Model training and evaluation

Logging and error handling

Deployment of ML models with Streamlit

This project was developed as part of the course:

CST2216 – Modularizing and Deploying ML Code
Algonquin College

Application

The Streamlit application allows users to enter property characteristics and obtain a predicted house price instantly.

Workflow of the application:

User enters house attributes in the sidebar.

The application performs feature transformations.

Inputs are scaled using the same scaler used during training.

The trained model predicts the house price.

The predicted price is displayed to the user.

Features Used in the Model

The dataset includes the following features:

squareMeters
numberOfRooms
hasYard
hasPool
floors
cityCode
cityPartRange
numPrevOwners
isNewBuilt
hasStormProtector
basement
attic
garage
hasStorageRoom
hasGuestRoom
houseAge

Target variable:

price

Feature engineering includes computing houseAge from the build year:

houseAge = current_year - year_built

Machine Learning Pipeline

The machine learning pipeline follows these steps:

Data loading

Feature engineering

Train-test split

Feature scaling

Model training

Model evaluation

Model serialization

Deployment with Streamlit

Model used:

Linear Regression

Scaling method:

MinMaxScaler

The trained model is saved using joblib.

Project Structure

real-estate-ml-streamlit

app.py – Streamlit application
train.py – main script to run training
README.md – project documentation
requirements.txt – project dependencies
.gitignore – ignored files

assets/ – application assets
 algonquin_logo.png

data/ – dataset used for training
 real_estate.csv

models/ – saved trained model
 real_estate_model.joblib

logs/ – application logs

notebooks/ – original experimentation notebook
 Real_Estate_EDA.ipynb

src/ – modular Python source code
 config.py – configuration settings
 data_loader.py – dataset loading functions
 features.py – feature engineering logic
 train.py – model training pipeline
 logger.py – logging configuration

Running the Project Locally
1. Clone the repository

git clone https://github.com/Laalahmi/real-estate-ml-streamlit.git

cd real-estate-ml-streamlit

2. Create a virtual environment

python -m venv .venv

Activate the environment

Windows

.venv\Scripts\activate

Mac or Linux

source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Train the model

python -m src.train

This will train the model and save it in the models folder.

5. Run the Streamlit application

python -m streamlit run app.py

The application will open in your browser at:

http://localhost:8501

Streamlit Application

The Streamlit interface allows users to:

Enter property features

Generate price predictions

View model information

Inspect input features used for prediction

The application also includes logging and error handling to ensure robustness.

Technologies Used

Python
Pandas
NumPy
Scikit-learn
Streamlit
Joblib
Pillow

Logging

Logging is implemented to track application activity including:

Model loading

Prediction requests

Runtime errors

Logs are stored in the logs directory.

Deployment

This project can be deployed on Streamlit Cloud by connecting the GitHub repository and specifying the main application file:

app.py

Author

Mohammed Laalahmi
Business Intelligence Systems Infrastructure
Algonquin College

Instructor

Dr. Umer Altaf
Algonquin College

Course

CST2216 – Modularizing and Deploying ML Code
