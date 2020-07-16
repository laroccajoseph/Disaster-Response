import sys
import matplotlib as plt
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''This function takes in two file paths for csv files, then loads and merges them to a usable format, returning it'''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge( categories, left_on='id', right_on='id')
    cat = categories.categories.str.split(';', expand=True)
    categories = cat
    row = cat.iloc[0]
    
# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing

    row = row.str.replace('1', '')
    row = row.str.replace('0', '')
    row = row.str.replace('-', '')
    category_colnames = row
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df = df.drop(['categories'], axis = 1)
    df_new = pd.concat([df, categories], axis = 1)
    return df_new
    
def clean_data(df_new):
    '''Clean the dataframe of duplicates and other values that interfere with the ML process'''
    df_new.duplicated().sum()
    df_drp = df_new.drop_duplicates()
    df_drp.duplicated().sum()
    df = df_drp
    df = df.dropna()
    df.isnull().sum()
    df.loc[df['related'] == 2] = 0
    return df
    
def save_data(df, database_filename):
    '''Save the dataframe based on a given filepath'''
    name = 'sqlite:///' + database_filename
    engine = create_engine(name)
    df.to_sql('dr', engine, index=False, if_exists = 'replace') 
    return

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()