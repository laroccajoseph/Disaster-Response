import sys
import matplotlib as plt
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from pathlib import Path

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    '''Load the data from a give filepath'''
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql_table('dr', engine)  
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y


def tokenize(text):
    ''' This function will tokenize out text message data into a machine readable format. It takes in text data'''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    temp_text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens_pre = word_tokenize(temp_text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens_pre if word not in stop_words]
    return tokens


def build_model():
    '''This function creates the pipeline for the model to be created and used globally'''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters =  parameters = {  
        'clf__estimator__min_samples_split': [2, 4],
    }
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    return cv
    

def evaluate_model(model, X_test, Y_test):
    '''This function uses prediction through the model and returns a report of info'''
    Y_pred = model.predict(X_test)
    
    accuracy = (Y_pred == Y_test).mean()
    recall = recall_score(Y_test, Y_pred, average='macro')
    pre = precision_score(Y_test, Y_pred, average='macro')

    print("Recall:", recall)
    print("Precision:", pre)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)
    
def save_model(model, model_filepath):
    '''This function saves the model for future use'''
    path = open(model_filepath, 'wb')
    pickle.dump(model, path)
    path.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        print(database_filepath)
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()