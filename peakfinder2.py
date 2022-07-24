# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:07:42 2022

@author: jwang
"""
import numpy as np
import pandas as pd
from pandas import read_csv
from scipy.signal import argrelextrema
import heapq
import matplotlib.pyplot as plt

series = read_csv('c:/temp/01484683_IndianRiverInlet_2014-2022_test2.csv', header=1, parse_dates=[0], index_col=[0],names=['DateTime', 'wl'], squeeze=True)#)#, usecols=col_list)
#print(series)
df=series.to_frame(name='data')

all_dates =  np.unique(df.index.date)

%matplotlib widget 

import scipy.signal

print(f'num of data points = {len(df)}')
n = int(len(df)/len(all_dates)/3)

def get_peaks(dt, df_all, n, sign=1):    
    
    mask = (df_all.index >= dt - pd.DateOffset(2)) & (df_all.index <= dt + pd.DateOffset(2))
    df1 = df_all[mask]        
    maxd = list(sign*df1['data'])           
    df1['maxd'] = maxd               
    #indexes = argrelextrema(np.array(maxd), np.greater_equal, order=n)[0]
    indexes, _ = scipy.signal.find_peaks(np.array(maxd), height=np.mean(maxd)*1.5, distance=n)
    #print(indexes)
    #print(df1.iloc[indexes]['data'])
    df1['max'] = float("nan")    
    df1.iloc[indexes,df1.columns.get_loc('max')] = df1.iloc[indexes]['data']
    
    df1 = df1[df1.index.date == dt]
    df1 = df1.reset_index()    
    peak_max = list(df1['max'].fillna(df1['max'].min() - 1))          
    high_peaks = heapq.nlargest(2, list(range(len(peak_max))), key=peak_max.__getitem__)          

    return df1.iloc[high_peaks]


all_max_peaks = []
all_min_peaks = []
for dt in all_dates:   
    max_peak = get_peaks(dt,df,n)
    max_peak = max_peak.dropna()  
    all_max_peaks.append(max_peak)    
    min_peak = get_peaks(dt,df,n, -1)
    min_peak = min_peak.dropna()  
    all_min_peaks.append(min_peak)
    
df_all_max=pd.concat(all_max_peaks)
df_all_min=pd.concat(all_min_peaks)
#print(df_all_max)
plt.figure(figsize=(10, 7))
plt.scatter(df_all_max.DateTime, df_all_max['data'], c='r')
plt.scatter(df_all_min.DateTime, df_all_min['data'], c='g')

plt.plot(df)
plt.xticks(rotation=45)

df_all_max.to_csv('c:/temp/Daily2maxima.csv')
df_all_min.to_csv('c:/temp/Daily2minima.csv')
