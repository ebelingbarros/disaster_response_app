# Disaster Response Web App

## 1. Libraries
The code presented in this repository was written in Python and Html. To work, it requires the following Python packages: 

- json
- plotly 
- pandas 
- nltk
- flask
- sklearn
- sqlalchemy
- sys
- numpy
- re
- pickle
- warnings

## 2. Project Overview
The project develops a web app which can be used to classify disaster messages (e.g. a hurricane or a flood) into several categories. As such, it could potentially be used by disaster workers to direct the obtained messages to the pertinent aid agencies. For this purposes, the app contains a Machine Learning model that categorizes the new messages received.

## 3. File Descriptions
- process_data.py: This code takes as its input csv files containing message data and message categories (labels), and creates an SQLite database containing a merged and cleaned version of this data.
- train_classifier.py: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
- ETL Pipeline Preparation.ipynb: The code and analysis contained in this Jupyter notebook was used in the development of process_data.py. process_data.py effectively automates this notebook.
- ML Pipeline Preparation.ipynb: The code and analysis contained in this Jupyter notebook was used in the development of train_classifier.py. In particular, it contains the analysis used to tune the ML model and determine which algorithm to use. train_classifier.py effectively automates the model fitting process contained in this notebook.
- data: This folder contains sample messages and categories datasets in csv format.
- app: This folder contains all of the files necessary to run and render the web app.

## 4. Running Instructions
### Run process_data.py
1. Save the data folder in the current working directory and process_data.py in the data folder.
2. From the current working directory, run the following command: ``` python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
### Run train_classifier.py
1. In the current working directory, create a folder called 'models' and save train_classifier.py in this.
2. From the current working directory, run the following command: ```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```
### Run the web app
1. Save the app folder in the current working directory.
2. Run the following command in the app directory: ```python run.py```
3. Go to http://localhost:3001/

## 5. Screenshot

<p align="center">
  <img width="97%" height="97%" src="https://github.com/ebelingbarros/disaster_response_app/blob/main/screen_example.png"> 
</p> 

## 6. Licensing, Authors, Acknowledgements
This app was completed as part of the Udacity Data Scientist Nanodegree. Code templates and data were provided by Udacity. The data was originally sourced by Udacity from Figure Eight.
