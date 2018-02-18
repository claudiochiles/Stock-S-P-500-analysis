#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 21:58:14 2018

@author: Claudio
"""

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
#candle stick is a type of graph used for finance
from matplotlib.finance import candlestick_ohlc 
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

#transforms the column date into a date time index, index_col makes the 
#column the first one
df = pd.read_csv('amzn.csv',parse_dates=True,index_col= 0) 

#stock manipulation 100 dates averages (moving average) creates
#a new column as well, min_periods takes the NaN out of the data
#df['100ma'] = df['Adj Close'].rolling(window = 100,min_periods=0).mean()
df.dropna(inplace=True) #inplace modifies that data frame in place
#print(df.tail())
#subplots
ax1 = plt.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)
#sharex so the graphs are in the same axis so when you zoom one axis 
#the other one zooms as well
ax2 = plt.subplot2grid((6,1),(5,0),rowspan=1,colspan=1,sharex=ax1)
#we choose the index because in our case the date is the index
"""ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma']) 
ax2.bar(df.index, df['Volume'])"""
#plt.show()

#resample data into specific time frames you want
df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

#show dates in open high low close in mdates (matplotlibs dates)
df_ohlc.reset_index(inplace=True)
df_ohlc['Date']= df_ohlc['Date'].map(mdates.date2num)
#display dates in graphs as matplotlib (pretty stuff)
ax1.xaxis_date()
candlestick_ohlc(ax1, df_ohlc.values, width = 2, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)
#plt.show()

#grab data from S&P 500 or any list
#inspect a company and then find the td from tables and the ticker

