## Extract ##

# Import libraries load datasets and drop drop duplicates by id
import pandas as pd
import sqlalchemy
import pathlib

messages = pd.read_csv(str(pathlib.Path(__file__).parent.resolve())+'/messages.csv').drop_duplicates(subset=['id'])
categories = pd.read_csv(str(pathlib.Path(__file__).parent.resolve())+'/categories.csv').drop_duplicates(subset=['id'])

## Transform ##

# Merge datasets with id
df = categories.merge(messages,on='id')

# Keep index of categories for accuracy in integration
index_of_categories = list(categories['id'])

# Split the values in the 'categories' column on the `;` character
categories = categories['categories'].str.split(';',expand=True)

# Select the first row to create a column.
row = categories.iloc[0,:]

# Use the first row to separate the text with a "-" and select the text.
category_colnames = row.apply(lambda X : X.split('-')[0])

# Create a column from the selected text.
categories.columns = list(category_colnames)

# Converts category values to just numbers 0 or 1.
categories = categories.apply(lambda X : X.apply(lambda Y : int(Y.split('-')[1])))

# Insert an index into the categories dataframe.
categories.insert(0, "id", index_of_categories)

# Remove the 'categories' row
df.drop('categories',axis=1,inplace=True)

# Merge datasets with id
df = df.merge(categories, how='inner', on='id')

# The 'original' row was removed because it contained too much erroneous data.
df.drop(['original'],axis=1,inplace=True)

## Load ##

# Exported clean data
engine = sqlalchemy.create_engine('sqlite:///'+str(pathlib.Path(__file__).parent.resolve())+'/DisasterResponse.db')
df.to_csv('DisasterResponse.csv', index=False)