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

## 2. Project Overview
The project develops a web app which can be used to classify disaster messages (e.g. a hurricane or a flood) into several categories. As such, it could potentially be used by disaster workers to direct the obtained messages to the pertinent aid agencies. For this purpose, the app contains a Machine Learning model that categorizes the new messages received.

## 3. File Descriptions
- ETL Pipeline Preparation.ipynb: The code contained in this Jupyter notebook is used to develop process_data.py. Its input are csv files that contain message data and message categories. Through data cleaning steps, a SQLite database is created, which contains a merged and cleaned version of the data.
- messages.csv and categories.csv: these are the two files that are used as raw input by the previous notebook.
- ML Pipeline Preparation.ipynb: This Jupyter notebook's code is used to develop train_classifier.py. It takes the SQLite database that is produced by the previous notebook and uses its data to train and tune a Random Forest ML model for categorizing the messages. In addition, test evaluation metrics are generated as part of the training process. Its output is a pickle file containing the fitted model. 
- messages.db: this is the database created by SQLite.
- process_data.py: This is the file which is effectively run by the app for cleaning and preparing the data.
- train_classifier.py: This is the file which is effectively run by the app for running the ML model
- data: The folder contains the sample messages and categories datasets in csv format.
- app: This folder contains all of the files necessary to run and render the web app.

## 4. Running Instructions
1. From the current working directory, run the following command: ``` python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
2. From the current working directory, run the following command: ``python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```
2. Run the following command in the app directory: ```python run.py```
3. Go to http://localhost:3001/

## 5. Screenshots

<p align="center">
  <img width="97%" height="97%" src="https://github.com/ebelingbarros/disaster_response_app/blob/main/screenshot.png"> 
</p> 

This is a screenshot of the app, in which the generic message "Please help me" is typed. Although this message is rather useless from the point of view of the app's potential user, it shows that the app does a reasonably good job in categorizing the message as a request towards receiving aid.

<p align="center">
  <img width="97%" height="97%" src="https://github.com/ebelingbarros/disaster_response_app/blob/main/prediction.png"> 
</p> 

## 6. Licensing, Authors, Acknowledgements
The app was completed as requirement for the completion of the Udacity Data Scientist Nanodegree. As such, data and code templates were provided by Udacity. The data was originally obtained by Udacity from [Figure Eight/Appen](https://appen.com/). 
