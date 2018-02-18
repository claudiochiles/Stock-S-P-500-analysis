#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:59:58 2018

@author: Claudio
"""
#beautiful soup is a lib for pulling out data out of HTML and XML files
import bs4 as bs
#saving objects into python so they can be used later in other scripts
#so in this case if we want another stock from the S&P 500 we can just have it
#inside python
import pickle
#will allow you to send HTTP/1.1 requests using Python.
import requests
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
#create new directories 
import os
import pandas as pd
import pandas_datareader.data as web
#fetch data google and yahoo other way
import quandl
import numpy as np
style.use('ggplot')
#get the information and save it from sp500 basically importing the tickers
def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text,"lxml")
    #finds the table in wikipedia that was source inspected that has
    #the information from sp500
    table = soup.find('table',{'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb")as f:
        pickle.dump(tickers, f)
        
        
    return tickers

#store a csv file in their own directory (store all data locally because
#it takes the program 20 min to get from yahoo, so we dont wanna repeat
#that process)

#save_sp500_tickers()

#so functions keep working from each other and dont reload 
def get_data_from_google(reload_sp500= False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle","rb")as f:
            tickers=pickle.load(f)
     #make that directory locally       
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
        
    start = dt.datetime(2010,1,1)
    end = dt.datetime(2016,12,31)
    #grab the ticker data. We could use tickers[25] to only have 25 stocks
    for ticker in tickers[:100]:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker,'yahoo',start,end)
            #df = quandl.get('WIKI/'+ticker,start_date = start, end_date =end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))
#get_data_from_google()

#compile all the information into one data frame from the information we pullled
#from quandl
def compile_data():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)
        
    main_df = pd.DataFrame()
    
    for count,ticker in enumerate(tickers[:71]):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace = True)
        #When inplace=True is passed, the data is renamed in place (it returns nothing)
        #when false or nothing performs the operation and returns a copy of the object
        df.rename(columns = {'Adj Close':ticker}, inplace = True)
        df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)
        
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')
        
        if count % 10 == 0:
            print(count)
            
    print(main_df.head())    
    main_df.to_csv('sp500_joined_closes.csv')   
#compile_data()
## visualize data for stocks   
def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    #df['AAPL'].plot()
    #plt.show()
    df_corr = df.corr()
    print(df_corr.head())
    #takes only data (numbers) inside the corr matrix
    data =  df_corr.values
    fig = plt.figure()
    #1 by 1 in plot number one
    ax = fig.add_subplot(1,1,1)
    #customizing the plot showing the correlation in color
    #if theres negative correlation if stock goes up other goes down
    heatmap = ax.pcolor(data,cmap = plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0])+0.5,minor = False)
    ax.set_yticks(np.arange(data.shape[1])+0.5,minor = False)
    #errases any gaps from the graph
    ax.invert_yaxis()
    #moves ticks from xaxis to the top
    ax.xaxis.tick_top()
    
    column_labels = df_corr.columns
    row_labels = df_corr.index
    
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    #limit of the color limit of the heatmap of correlationmatrix
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()

#visualize_data()