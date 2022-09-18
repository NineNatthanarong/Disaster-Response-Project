# Disaster Response Project

Analyzing message data for disaster response

### Libraries used
- pandas==1.4.3
- numpy==1.22.3
- sqlalchemy==1.4.41
- re==2.2.1
- nltk==3.7
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
Use LinearSVC Algorithm with parameter

- Precision 0.9571006279772363
- Recall    0.9474695707879563
- F1Score   0.942995474232156

[ C=1, loss='hinge', max_iter=50000, tol=0.001 ]

#### Web app
- Test here

[https://disaster-response-project001.herokuapp.com/](https://disaster-response-project001.herokuapp.com/)

- Main page
![Main_page](https://github.com/NineNatthanarong/Disaster-Response-Project/blob/master/pic/shot1.png)
![Main_page](https://github.com/NineNatthanarong/Disaster-Response-Project/blob/master/pic/shot2.png)
- Analysis page
![Analysis_page](https://github.com/NineNatthanarong/Disaster-Response-Project/blob/master/pic/shot3.png)
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