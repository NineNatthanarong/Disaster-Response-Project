## Import libraries ##
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import pathlib
import joblib
import sqlalchemy

## Set flask app ##
app = Flask(__name__)

## tokenize function ##
def tokenize(text):
    """
    tokenize text function

    Args:
        text (String): text or words to tokenize

    Returns:
        List: list of clean text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

## Load data ##
engine = sqlalchemy.create_engine('sqlite:///'+str(pathlib.Path(__file__).parent.resolve()).replace('app','data')+'\DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse',engine)

## Load model ##
model = joblib.load(str(pathlib.Path(__file__).parent.resolve()).replace('app','models')+'\model.pkl')

## index webpage displays ##
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    data = df.iloc[:,3:]
    most5 = data.sum().sort_values(ascending=False)[:10]
    least5 = data.sum().sort_values(ascending=False)[-10:]
    cor = data.corr().dropna(how='all',axis=0).dropna(how='all',axis=1)
    
    # create visuals
    graphs = [
        {
            'data': [{
                'x': most5.index.tolist(),
                'y': most5.values.tolist(),
                'type': 'bar',
                'marker':{
                    'color': ['rgba(222,45,38,0.8)', 'rgba(204,204,204,1)', 'rgba(204,204,204,1)', 'rgba(204,204,204,1)', 'rgba(204,204,204,1)','rgba(204,204,204,1)', 'rgba(204,204,204,1)', 'rgba(204,204,204,1)', 'rgba(204,204,204,1)', 'rgba(204,204,204,1)', 'rgba(204,204,204,1)']
                }
            }],
            'layout': {
                'title': '10 Most categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                },
                'font': {'size': 18 },
            }
        },
        {
            'data': [{
                'x': least5.index.tolist(),
                'y': least5.values.tolist(),
                'type': 'bar',
                'marker':{
                    'color': 'rgb(255, 178, 107)',
                    'opacity': 0.8,
                }
            }],
            'layout': {
                'title': '10 Minimal categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                },
                'font': {'size': 18 },
            }
        },
        {
            'data':  [{
                'values': genre_counts,
                'labels': genre_names,
                'type': 'pie',
                'textinfo': "label+percent",
        }],

            'layout': {
                'title': 'Proportion of each type',
                'font': {'size': 18},
            }
        },
        {
            'data': [{
                'z': cor.values,
                'type': 'heatmap',
                'x':cor.index,
                'y':cor.index,
            }],
            'layout': {
                'title': 'Correlation of categories',
                'font': {'size': 18 },
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

## Set port and host ##
def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()