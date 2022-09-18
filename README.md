# Disaster Response Project

Analyzing message data for disaster response

### Libraries used
- pandas==1.4.3
- numpy==1.22.3
- sqlalchemy==1.4.41
- re==2.2.1
- nltk=3.7
- sklearn==1.1.0
- json==2.0.9
- plotly==5.9.0
- joblib==1.1.0
- flask==2.1.2

### Installation
run with python3.*

`pip install -r requirement.txt`

### Run app
run with python3.*

set path
`cd app`

run
`python run.py`

### Content

#### Machine learning pipeline
I use LinearSVC Algorithm with parameter

[ C=1, loss='hinge', max_iter=50000, tol=0.001 ]

#### Web app
- Main page
![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)
![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)
- Analysis page
![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)
### File Description
**app/run.py** -> Python file to run flask app

**app/index.html and go.html** -> Page of flask app

**data/categories.csv and messages.csv** -> raw data

**data/DisasterResponse.db** -> Database of clean data

**data/process_data.py** -> ETL pipeline

**models/model.pkl** -> Machine learning model

**models/train.py** -> Machine learning pipeline

**requirement.txt** -> require libraries to run

### Authors

- [@NineNatthanarong](https://github.com/NineNatthanarong)
### Acknowledgements
- dataset reference
    - [Udacity](https://www.udacity.com/)
### License

[MIT](https://choosealicense.com/licenses/mit/)