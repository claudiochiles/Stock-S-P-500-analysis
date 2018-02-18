#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:07:26 2018

@author: Claudio
"""

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

#this part imports the data from yahoo finance and then creates a csv file
#of that data
start = dt.datetime(2007,9,12)
end = dt.datetime(2017,9,12)

#df is short for data frame
df = web.DataReader('^GSPC','yahoo',start, end)

df.to_csv('sp500.csv')
#transforms the column date into a date time index, index_col makes the 
#column the first one
"""df = pd.read_csv('amzn.csv',parse_dates=True,index_col= 0) 
#print(df.head())

print(df[['Open','High']].head()) #printing two columns to acompany date

#visualization
df['Adj Close'].plot()"""