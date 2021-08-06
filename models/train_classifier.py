# Import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

import warnings

warnings.simplefilter('ignore')


def load_data(database_filepath):
    """This function loads the messages and categories datasets and merges them
    
    Args:
    database_filename: a string with the file name of the SQLite database that contains the cleaned message data.
       
    Returns:
    X: it is a dataframe that contains the dataset with the features.
    Y: it is a dataframe that contains the dataset with the labels
    category_names: this is a list of strings that contains the category names.
    """
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM Messages", engine)
    
    # Create X and Y datasets
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    # Create list containing all category names
    category_names = list(Y.columns.values)
    
    return X, Y, category_names

def tokenize(text):
    """This function normalizes, tokenizes and stem the text strings
    
    Args:
    text: strings that contains messages to be processed
       
    Returns:
    stemmed: a list of strings which contain the normalized and stemmed word tokens
    """
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # Stem word tokens and remove stop words
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return stemmed

def performance_metric(y_true, y_pred):
    """This function calculates a median F1 score for all of the output classifiers

        Args:
        y_true: an array containing the actual labels.
        y_pred: an array containing the predicted labels.

        Returns:
        score: a float with the Median F1 score for all of the output classifiers
        """
    f1_list = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i])
        f1_list.append(f1)
        
    score = np.median(f1_list)
    return score
    
def build_model():
    """This function builds the machine learning pipeline
    
    Args:
    None
       
    Returns:
    cv: a gridsearchcv objec that that creates a 
    model object and determines optimal model parameters.
    """
    # Create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Create parameters dictionary
    parameters = {'vect__min_df': [1, 5],
                  'tfidf__use_idf':[True, False],
                  'clf__estimator__n_estimators':[10, 25], 
                  'clf__estimator__min_samples_split':[2, 5, 10],
                 'clf__estimator__class_weight': ['balanced']}
    
    # Create scorer
    scorer = make_scorer(performance_metric)
    
    # Create grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters, scoring = scorer, verbose = 10)
    return cv

def get_eval_metrics(actual, predicted, col_names):
    """This function calculates the evaluation metrics for the ML model
    
    Args:
    actual: an array that containins the actual labels.
    predicted: an array that contains the predicted labels.
    col_names: a list of strings that contains the names for each of the predicted fields.
       
    Returns:
    metrics_df: a aataframe that contains the accuracy, precision, recall 
    and f1 score for a set of actual and predicted labels.
    """
    metrics = []
    
    # Calculate evaluation metrics for each set of labels
    for i in range(len(col_names)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i], average='weighted')
        recall = recall_score(actual[:, i], predicted[:, i], average='weighted')
        f1 = f1_score(actual[:, i], predicted[:, i], average='weighted')
        
        metrics.append([accuracy, precision, recall, f1])
    
    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return metrics_df


def evaluate_model(model, X_test, Y_test, category_names):
    """This function returns test accuracy, precision, recall and F1 score for fitted model
    
    Args:
    model: model object. Fitted model object.
    X_test: dataframe. Dataframe containing test features dataset.
    Y_test: dataframe. Dataframe containing test labels dataset.
    category_names: list of strings. List containing category names.
    
    Returns:
    None
    """
    # Predict labels for test dataset
    Y_pred = model.predict(X_test)
    
    # Calculate and print evaluation metrics
    eval_metrics = get_eval_metrics(np.array(Y_test), Y_pred, category_names)
    print(eval_metrics)
    

def save_model(model, model_filepath):
    """A pickle of the fitted model
    
    Args:
    model: a model object with the fitted model.
    model_filepath: a string of the filepath where the fitted model will be saved
    
    Returns:
    None
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading the data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building the model...')
        model = build_model()
        
        print('Training the model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating the model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving the model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('You need to provide, as the first argument, the disaster messages database's filepath '\
              'and then the filepath of the pickle file, as the second argument, in order to '\
              'save the model. \n\nAn example: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
