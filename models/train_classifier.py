# import libraries
import sys
from sklearn.externals import joblib

from sqlalchemy import create_engine
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pickle


import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    
    """
    load_data loads data that was saved in the process_data.py using the load_data function, into a pandas data frame and returns X, Y and category_names.
    
    Inputs:
        :database_file_path: The path of the database where our data was saved.
    Outputs:
        :X: The messages column in the needed dataframe.
        :Y: The one hot encoded classes (targets) columns.
        :category_names: The names of the categories in Y.
    """
    
    engine = create_engine('sqlite:///data/InsertDatabaseName.db')
    df = pd.read_sql_table('InsertTableName', engine, index_col=False)

    X = df.loc[:,'message']
    Y = df.iloc[:,4:]
    Y.drop(columns=['child_alone'], inplace = True)
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    
    """
    Preprocess the messages text data,by:
    1. Making them lower case.
    2. Remove the punctuations and replace then with spaces using regular expressions.
    3. Split text into a list of words using spaces.
    4. Removing stop words.
    5. Lemmatizing the words.
    5. Stemming the words.
    and finally the function returns the preprocessed list of words
    
    """
    punctuation_pattern = r"[^a-zA-Z0-9]"
    
    text = text.lower()
    text = re.sub(punctuation_pattern, " ", text) 
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    words = [PorterStemmer().stem(w) for w in words]
    
    return words


def build_model():
    
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('multiClassifier',MultiOutputClassifier(KNeighborsClassifier()))
                        ])
    parameters = {
        'multiClassifier__estimator__n_neighbors': [5, 10, 15],
        'multiClassifier__estimator__weights': ['uniform']
                 }
            
            
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1, verbose=3)    
    
    
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    
    # Evaluate Model
    for i, col in enumerate(category_names):
    
        print("Column {}: {}".format(i, col))
    
        y_true = list(Y_test.values[:, i])
        y_pred = list(Y_pred[:, i])
        target_names = ['is_{}'.format(col), 'is_not_{}'.format(col)]
        print(classification_report(y_true, y_pred, target_names=target_names))
        
    return
    
    


def save_model(model, model_filepath):
    
    pickle.dump(model, open(model_filepath, 'wb'))
    
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        model = model.best_estimator_
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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