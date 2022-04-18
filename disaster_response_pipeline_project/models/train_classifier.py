import sys
import pickle
import sqlite3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
#from custom_transformer import StartingVerbExtractor

from sklearn.base import BaseEstimator, TransformerMixin

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def load_data(database_filepath):
    path='sqlite:///'+database_filepath
    engine = create_engine(path)
    df = pd.read_sql_table('table_0',engine)

    X = df.message.values
    Y = df[df.columns[4:-1]].values
    category_names=df.columns[4:-1].tolist()
    return (X,Y,category_names)

def tokenize(text):
    #for test in X:
    words=word_tokenize(text)
    lemmatizer=WordNetLemmatizer()
    clean_tokens=[]
    for tok in words:
        clean_tok=lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    #text processing and model pipeline
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.neighbors import KNeighborsClassifier
    pipeline=Pipeline([
        ('features',FeatureUnion([
            ('text_pipeline',Pipeline([
                ('vect',CountVectorizer(tokenizer=tokenize)),
                ('tfidf',TfidfTransformer())
            ])),
            
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf',MultiOutputClassifier(KNeighborsClassifier()))
    ])
    #print(KNeighborsClassifier().get_params().keys())
    #print(MultiOutputClassifier(KNeighborsClassifier()).get_params().keys())
    #print(pipeline.get_params())
    
    #define parameters for GridSearchCV
    parameters = {
        #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        #'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        #'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__leaf_size': [30, 50],
        'clf__estimator__n_neighbors': [5, 7],
        
        #'features__transformer_weights': (
        #    {'text_pipeline': 1, 'starting_verb': 0.5},
        #    {'text_pipeline': 0.5, 'starting_verb': 1},
        #    {'text_pipeline': 0.8, 'starting_verb': 1},
        #)
    }
    
    #create gridsearch object and return as final model pipeline
    cv=GridSearchCV(pipeline,param_grid=parameters,verbose=10)
    
    return cv#pipeline#
    
def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    
    from sklearn.metrics import classification_report,confusion_matrix
    classreport=[]
    for i in range(len(Y_test[0])):
        classreport.append([category_names[i],classification_report(Y_test[:,i], Y_pred[:,i],zero_division=0)])
    df=pd.DataFrame(classreport)
    df.to_csv('classfication_report.csv')
    return
    
    '''
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    '''

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    #pickled_model = pickle.load(open('model.pkl', 'rb'))

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