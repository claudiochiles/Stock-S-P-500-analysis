#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:08:37 2018

@author: Claudio
"""

import numpy as np
import pandas as pd
import pickle
from collections import Counter
## machine learning testing and training samples
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

##we're trying to create a model where if a stock after 7 days rises certain
##percentage then the model sells the stocks and if it goes down buys and so on
def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv',index_col = 0)
    tickers = df.columns.values.tolist()
    df.fillna(0,inplace=True)
    
    for i in range(1,hm_days+1):
        ##for example if its Amazon it will show AMAZN_thedayd
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker])/df[ticker]
    df.fillna(0, inplace=True)
    hello = df['{}_{}d'.format(ticker,i)]
    return tickers, df

##mapping functions
##args makes you pass any number of arguments you want and becomes iterable
def buy_sell_hold(*args):
    ##goes by row in the data frame to pass columns as parameters to work with that
    ##column row value. In this case we pass tomorrows price and the day after that 
    ##and so on (the whole week of percent changes)
    cols = [c for c in args]
    ## if they pass some required value we choose then we either buy or sell
    requirement = 0.028
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
        
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    
    ## for example AAPL_target
    ## try changing this mapping into something more elegant
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, 
       df['{}_1d'.format(ticker)],
        df['{}_2d'.format(ticker)],
         df['{}_3d'.format(ticker)],
          df['{}_4d'.format(ticker)],
           df['{}_5d'.format(ticker)],
            df['{}_6d'.format(ticker)],
             df['{}_7d'.format(ticker)]))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    ## Counter gives us the distribution
    print('Data spread:', Counter(str_vals))
    df.fillna(0, inplace = True)
    ## so when a stock prices goes from 0 to 3000 (inf) it doesn't count it as the percent
    ##change we're looking for
    df = df.replace([np.inf,-np.inf],np.nan)
    df.dropna(inplace=True)
    ##percent change data for all the companies(71 companies)
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf,-np.inf],0)
    df_vals.fillna(0, inplace = True)
    ## generally X feature sets (price changes, pct changes) and y labels (things
    ## that describe it)
    X = df_vals.values
    ## try using just vals instead of y
    ## y is the hold either -1, 0 or 1 target classifier
    y = df['{}_target'.format(ticker)].values
    
    return X,y, df
##machine learning part of the exercise
def do_ml(ticker):
    X,y,df = extract_featuresets(ticker)
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,
                                                                         test_size =
                                                                         0.25)
    #clf = neighbors.KNeighborsClassifier()
    ## make the machine vote by itself between 3 classifiers which one is best to use
    clf = VotingClassifier([('lsvc',svm.LinearSVC()),('knn',
                            neighbors.KNeighborsClassifier()),
                            ('rfor',RandomForestClassifier())])
    
    
    clf.fit(X_train,y_train)    
    confidence = clf.score(X_test, y_test)
    print('Accuracy:',confidence)
    predictions = clf.predict(X_test)
    ## to make differente predictions
    print('Predicted spread:', Counter(predictions))
    
    return confidence

do_ml('BAC')

    
    
    
        