## Import libraries ##

import pandas as pd
import sqlalchemy
import pathlib
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import sqlalchemy
import re
import nltk
import warnings
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
warnings.filterwarnings("ignore")


## load datasets ##

engine = sqlalchemy.create_engine('sqlite:///'+str(pathlib.Path(__file__).parent.resolve()).replace('models','data')+'\DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse',engine)

## all function ##

# function to tokenize

def tokenize(text):
    """
    tokenize text function

    Args:
        text (List): List of text or words to tokenize

    Returns:
        List: list of clean text
    """
    detected_urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

## Machine learning pipeline ##

pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC(C=1, loss='hinge', max_iter=50000, tol=0.001)))
    ])

## Clean data and split data ##

df = df[df.iloc[:,3:].sum(axis=1) != 0]
df.reset_index(drop=True,inplace=True)
X = df.iloc[:,:3]
Y = df.iloc[:,3:]
X = df['message'].values.tolist()
indexY = df.iloc[:,3:].columns[np.mean(Y,axis=0) != 0]
Y = Y[indexY].values.tolist()

## Train the model ##

pipeline.fit(X,Y)

## Export model ##

pickle.dump(pipeline, open(str(pathlib.Path(__file__).parent.resolve())+'\model.pkl', 'wb'))