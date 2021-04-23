# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:14:18 2021

@author: lenovo
"""
import numpy as np
import datetime
import pandas as pd
import math
from pylab import plt, mpl

# Get business days during the life of snowball. 
def get_bdays(maturity):
    issue_date = datetime.date.today()                                  #issue date of the snowball
    end_date = issue_date + datetime.timedelta(days=maturity*365)       #end date of the snowball
    bdate = pd.bdate_range(issue_date, end_date).date                   #business days during the whole life of the snowball
    bdate_list = [datetime.datetime.strftime(x,'%F') for x in bdate]    #datetime to str
    
    return bdate_list

# Generate geometric brownian motion paths. 
def get_path(r, sigma, num_bdays, num_path, maturity):
    S = np.zeros((num_bdays, num_path))
    S[0] = 1
    for t in range(1, num_bdays):
        S[t] = S[t-1] * np.exp((r-0.5*sigma**2) * (maturity/num_bdays) +\
        sigma * math.sqrt(maturity/num_bdays) * np.random.standard_normal(num_path))
    
    return S

# Plot simulation paths
def plot_path(S):
    plt.style.use('seaborn')
    mpl.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(10, 6))
    plt.plot(S, lw=1.5)
    plt.xlabel('time')
    plt.ylabel('index level')

# Generate coupon for each knock-out date
def get_coupon(annualized_coupon, maturity):
    return np.linspace(annualized_coupon/12,annualized_coupon*maturity,12*maturity)

# Calculate discounting factors for each knock-out date
def get_df(r, call_obs):
    issue_date = datetime.date.today()
    call_obs = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in call_obs]
    diff = []
    df = []
    for i in range(len(call_obs)):
        diff.append((call_obs[i].date() - issue_date).days)
        df.append(math.exp(-diff[i]/365*r))
        
    return df

# Get the value of snowball by discounting the payoff of all simulation paths
def get_value(path,        #simulation paths
             coupon,       #coupon
             call_barrier, #knock-out barrier
             call_obs,     #knock-out obeservation date
             ki_barrier,   #knock-in barrier
             ki_obs,       #knock-in obeservation date
             df,           #discount factors
             notional      #notional principle
            ):
    obs = path.loc[:,call_obs]>call_barrier                     #find knock-out paths
    called = obs.any(axis=1)                                    #find knock-out paths
    called_value = obs[called]*coupon*df
    called_value = called_value[called_value>0].min(axis=1)     #discount the coupon on knock-out dates

    ki = (path[~called].loc[:,ki_obs]<ki_barrier).any(axis=1)   #knock-in and no knock-out
    ki_value = (path[~called][ki].iloc[:,-1]-1)*df[-1]          #discount the loss for knock-in paths

    nki_value = (~ki).sum()*coupon[-1]*df[-1]                   #discount paths that do not knock-in either knock-out

    return (called_value.sum()+ki_value.sum()+nki_value)/path.shape[0]*notional #average all paths



r = 0.03                                                        #risk-free rate
sigma = 0.26                                                    #volatility of underlying
maturity = 1                                                    #life of the snowball
num_simulation = 10000                                          #number of simulation
annualized_coupon = 0.1991                                      #coupon offered by the dealer
bdays = get_bdays(maturity)                                     #get business days in the whole life of snowball
S = get_path(r, sigma, len(bdays), num_simulation, maturity)    #generate corresponding geometric brownian motion paths
path = pd.DataFrame(data = S, index = bdays).T                  #form the dataframe for simulated paths
coupon = get_coupon(annualized_coupon, maturity)                #generate coupon rate for each knock-out date
call_barrier = 1                                                #knock-out level
call_obs = bdays[0:len(bdays):22][1:]                           #knock-out observation date
if len(call_obs) != maturity * 12:                              #need to find a way to better address the function
    call_obs.append(bdays[-1])
ki_barrier = 0.75                                               #knock-in barrier
ki_obs = bdays                                                  #knock-in observation date
df = get_df(r, call_obs)                                        #generate the discount factors for each knock-out date
notional = 1
value = get_value(path, coupon, call_barrier, call_obs, ki_barrier, ki_obs, df, notional)
print('The value of the snowball is '+str(value))