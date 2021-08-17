# Import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    """The function loads the data and specifies the model that will be estimated.
    
    Input:
    Data stored in the SQLite database
       
    Output:
    Dependent and independent variables.
    """
        
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM Accidents", con=engine)
    
    X = df['message']
    Y = df.drop(['index', 'message', 'original', 'genre'], axis = 1)
    cat_names = df.drop(["index", "message", "original", "genre"], axis=1).columns
    return X, Y, cat_names
    
def tokenize(text):
    """
    The function converts the message into into tokens

    Input:
    Disaster response messages

    Output:
    Tokens to be used as inputs for the model
    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    The function builds a machine the machine learning model. First, the pipeline is created, then the parameters dictionary is 
    created, and lastly the grid search object is built.
    
    Input:
    None

    Output: 
    Grid search object
    """    

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'vect__min_df': [1, 5, 10],
                  'clf__estimator__n_estimators':[5, 10, 100], 
                  'clf__estimator__min_samples_split':[2, 5, 10],
                 'clf__estimator__class_weight': ['balanced']}
    
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose=15)
    return cv

def evaluate_model(model, X_test, Y_test, cat_names):
    """
    The function evaluates the model. A for loop for four different evaluation metrics (accuracy, f1, precision and recall) is run 
    for the category names.
    
    Inputs:
    model, X_test, Y_test, cat_names

    Output: 
    A printed dataframe with the evaluation metrics
    """    

    Y_pred = model.predict(X_test)
    evaluation_metrics = []
    
    # Calculate evaluation metrics for each set of labels
    for i in range(len(cat_names)):
        accuracy = accuracy_score(Y_test.iloc[:, i].values, Y_pred[:, i])
        f1 = f1_score(Y_test.iloc[:, i].values, Y_pred[:, i], average='weighted')
        precision = precision_score(Y_test.iloc[:, i].values, Y_pred[:, i], average='weighted')
        recall = recall_score(Y_test.iloc[:, i].values, Y_pred[:, i], average='weighted')
                
        evaluation_metrics.append([accuracy, precision, recall, f1])
                
    cols = ['Accuracy', 'Precision', 'Recall', 'F1']
    evaluation_metrics_df = pd.DataFrame(data = evaluation_metrics, index = cat_names, columns = cols)
    
    print(evaluation_metrics_df)
   
def save_model(model, model_filepath):
    """
    This function saves the model that will be used in the app in a pickle format.
    
    Inputs:
    model, model filepath

    Output:
    Model in pickle format.
    """

    pickle.dump(model, open(model_filepath, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading the data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, cat_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building the model...')
        model = build_model()
        
        print('Training the model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating the model...')
        evaluate_model(model, X_test, Y_test, cat_names)

        print('Saving the model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('You need to provide, as the first argument, a filepath for the disaster messages database'\
              'and then the filepath of the pickle file, as the second argument, in order to '\
              'save the model. \n\nAn example: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()