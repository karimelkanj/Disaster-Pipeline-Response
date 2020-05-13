import sys

# For ETL Pipeline
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the inputs of disater messages and categories as CSV format and returns a panda dataframe
    having the 2 files merged.
    Input:
        :messages_filepath: CSV of disater messages from figure Eight
        :categories_filepath: CSV of disater categories for the messages_filepath file
    Returns:
        :df: Pandas DataFrame merging the 2 CSV files provided.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories)

    return df


def clean_data(df):
    """
    Takes an dirty and untidy pandas dataframe and returns a tidy version of it in a format that 
    can be saved into an SQL database.
    Input:
        :df: A data frame that is return from the load_data() method 
    """
    
    column_names = df['categories'].str.split(';', expand=True).iloc[0,:]
    categories = df['categories'].str.split(';', expand=True)
    category_colnames = list(set([c_name[:-2] for c_name in column_names]))
    categories.columns = [c_name[:-2] for c_name in column_names]
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    df.drop(columns=['categories'], inplace = True)
    
    df = pd.concat([df, categories], axis = 1)

    df.drop_duplicates(inplace = True)

    return df


def save_data(df, database_filename):
    """
    Saves the a given pandas dataframe into a database_filename
    :Inputs:
        :df: pandas dataframe we want to save into a database
        :database_filename: The database filepath we wan to save the dataframe into
    :Outputs:
        None: the function only saves the data
    """
    engine = create_engine('sqlite:///data/InsertDatabaseName.db')
    df.to_sql('InsertTableName', engine, index=False)  
    

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