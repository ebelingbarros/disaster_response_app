# Import packages
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Load messages dataset
messages = pd.read_csv('messages.csv')
    
# Load categories dataset
categories = pd.read_csv('categories.csv')
    
# Merge datasets
df = messages.merge(categories, how = 'left', on = ['id'])

print('Loading data...')

# create a dataframe of the 36 individual category columns
categories = categories.categories.str.split(";", n=36, expand=True,)

# select the first row of the categories dataframe and use this row to extract a list of new column names for categories.
categories.rename(columns=categories.iloc[0], inplace = True)
categories.columns = categories.columns.str.replace('[-1, -0]','')

#Convert category values to just numbers 0 or 1.
categories.replace(r'[a-zA-Z%]', '', regex=True, inplace=True)
categories.replace('[-, _, __]', '', regex=True, inplace=True)
categories = categories.apply(pd.to_numeric, errors="coerce")
categories.reset_index(level=0, inplace=True)

# drop the original categories column from `df`
df=df[['message', 'original', 'genre']]
df.reset_index(level=0, inplace=True)

# concatenate the original dataframe with the new `categories` dataframe
df = pd.merge(df, categories, on="index")

# drop duplicates
df.drop_duplicates(inplace = True)

print('Cleaning data...')
    
engine = create_engine('sqlite:///Messages.db')
df.to_sql('Messages', engine, index=False, if_exists='replace')
 
print('Saving data..')
print('Cleaned data saved to database!')
