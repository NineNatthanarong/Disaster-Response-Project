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

# function to calculate f1 score precision and recall
def metrics(model,X_test,y_test):
    """
    function to calculate f1 score precision and recall

    Args:
        model (model): List of text or words to tokenize
        X_test (List,String): List or String to calculate
        y_test (List,String): List or String to calculate

    Returns:
        ndarray:  Recall, f1 Score , Precision
    """
    Precision = []
    F1Score = []
    Recall = []
    from sklearn.metrics import precision_score,f1_score,recall_score
    for i in range(len(y_test)):
        y_pred = model.predict(X_test)[i]
        Precision.append(precision_score(y_test[i],y_pred,average='weighted'))
        F1Score.append(f1_score(y_test[i],y_pred,average='weighted'))
        Recall.append(recall_score(y_test[i],y_pred,average='weighted'))
    npPrecision = np.array(Precision) 
    npF1Score = np.array(F1Score)
    npRecall = np.array(Recall)
    npPrecision = npPrecision[npPrecision > 0]
    npF1Score = npF1Score[npF1Score > 0]
    npRecall = npRecall[npRecall > 0]
    print("Precision",np.mean(npPrecision))
    print("Recall   ",np.mean(npRecall))
    print("F1Score  ",np.mean(npF1Score))
    return np.mean(npPrecision),np.mean(npRecall),np.mean(npF1Score)

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
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.1,shuffle=True)

## Train the model ##

pipeline.fit(X_train, y_train)

## Show score ##

metrics(pipeline,X_test,y_test)

## Export model ##

pickle.dump(pipeline, open(str(pathlib.Path(__file__).parent.resolve())+'\model.pkl', 'wb'))