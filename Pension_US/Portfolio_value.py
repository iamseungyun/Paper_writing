import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
import pickle
# import import_ipynb


def dyna_withdraw(retire_year, weight, k, years, withdraw_rate, df, portfolio_value) :
    start_date = datetime(retire_year, 1, 1)
    end_date = start_date + relativedelta(years=years)
    
    annual_withdraw = portfolio_value*withdraw_rate
    initial_weight = weight(retire_year, 0, k)
    
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    portfolio_values = []
    withdraw_amounts = []
    
    for i in range(df.shape[0]-1) :
        if i%12 == 0 :
            if i == 0 :
                portfolio_value -= annual_withdraw
                withdraw_amounts.append(annual_withdraw)
            else :
                if (((stock_value*(1+df.iloc[i]['Mkt']))+(bond_value*(1+df.iloc[i]['RF'])))*0.05)/df.iloc[i]['CPI'] > annual_withdraw/df.iloc[0]['CPI']  :
                    annual_withdraw *= 1.03
                elif (((stock_value*(1+df.iloc[i]['Mkt']))+(bond_value*(1+df.iloc[i]['RF'])))*0.05)/df.iloc[i]['CPI'] < annual_withdraw/df.iloc[0]['CPI'] :
                    annual_withdraw *= 0.97
                
                portfolio_value -= annual_withdraw*df.iloc[i]['CPI']/df.iloc[0]['CPI']*((0.99)**(i/12))
                withdraw_amounts.append(annual_withdraw*((0.99)**(i/12)))
        
        if i > 0 :        
            stock_value = portfolio_value*weight(retire_year, i-1, k)*(1+df.iloc[i]['Mkt'])
            bond_value = portfolio_value*(1-weight(retire_year, i-1, k))*(1+df.iloc[i]['RF'])
        else :
            stock_value = portfolio_value*initial_weight
            bond_value = portfolio_value*(1-initial_weight)
            
        portfolio_value = stock_value + bond_value
        
        if portfolio_value < 0 :
            portfolio_value = 0
            annual_withdraw = 0
            
        portfolio_values.append(portfolio_value*(df.iloc[0]['CPI']/df.iloc[i]['CPI']))
    
    stock_value = portfolio_value*weight(retire_year, 12*years-1, k)*(1+df.iloc[12*years]['Mkt'])
    bond_value = portfolio_value*(1-weight(retire_year, 12*years-1, k))*(1+df.iloc[12*years]['RF'])
    portfolio_value = stock_value + bond_value
    portfolio_values.append(portfolio_value*(df.iloc[0]['CPI']/df.iloc[12*years]['CPI']))
        
        
    portfolio_values = np.array(portfolio_values)
    portfolio_values = (portfolio_values/portfolio_values[0])*100
    withdraw_amounts = np.array(withdraw_amounts)
        
    return portfolio_values, withdraw_amounts

def cons_withdraw(retire_year, weight, k, years, withdraw_rate, df, portfolio_value) :
    Mkt_name = 'Mkt'
    print(Mkt_name)
    start_date = datetime(retire_year, 1, 1)
    print('start_date: ',start_date)
    end_date = start_date + relativedelta(years=years)
    print('end_date: ',end_date)
    
    annual_withdraw = portfolio_value*withdraw_rate
    initial_weight = weight(retire_year, 0, k)
    print('initial_weight: ',initial_weight)
    
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    portfolio_values = []
    withdraw_amounts = []
    
    for i in range(df.shape[0]-1) :
        if i%12 == 0 :
            portfolio_value -= annual_withdraw*((0.99)**(i/12))*(df.iloc[i]['CPI']/df.iloc[0]['CPI'])
        
            withdraw_amounts.append(annual_withdraw*((0.99)**(i/12)))
        
        if i > 0 :
            stock_value = portfolio_value*weight(retire_year, i-1, k)*(1+df.iloc[i][Mkt_name])
            bond_value = portfolio_value*(1-weight(retire_year, i-1, k))*(1+df.iloc[i]['RF'])
        else :
            stock_value = portfolio_value*initial_weight
            bond_value = portfolio_value*(1-initial_weight)
            
        portfolio_value = stock_value + bond_value
        
        if portfolio_value < 0 :
            portfolio_value = 0
            annual_withdraw = 0
            
        portfolio_values.append(portfolio_value*(df.iloc[0]['CPI']/df.iloc[i]['CPI']))
    
    stock_value = portfolio_value*weight(retire_year, 12*years-1, k)*(1+df.iloc[12*years][Mkt_name])
    bond_value = portfolio_value*(1-weight(retire_year, 12*years-1, k))*(1+df.iloc[12*years]['RF'])
    portfolio_value = stock_value + bond_value
    portfolio_values.append(portfolio_value*(df.iloc[0]['CPI']/df.iloc[12*years]['CPI']))
        
        
    portfolio_values = np.array(portfolio_values)
    portfolio_values = portfolio_values/portfolio_values[0]*100
    withdraw_amounts = np.array(withdraw_amounts)
        
    return portfolio_values, withdraw_amounts