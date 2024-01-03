import pandas as pd

data = pd.read_csv('MSFT.csv')
column = 'Adj Close'

if column in data:
    data.drop(columns=[column],inplace=True)
    data.to_csv('MSFT.csv',index=False)