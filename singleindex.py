import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def alpha_beta_calc(stockdata,marketdata):
    alphas = []
    betas = []
    
    RM = marketdata.to_numpy()
    RM = RM.reshape(-1,1)
    stockdata = stockdata.to_numpy()
    
    reg = LinearRegression()
    for i in range(stockdata.shape[1]):
        reg.fit(RM,stockdata[:,i].reshape(-1,1))
        betas.append(reg.coef_[0][0])
        alphas.append(reg.intercept_[0])
    return alphas,betas

def single_index(stockdata,marketdata,rf):
    alphas, betas = alpha_beta_calc(stockdata,marketdata)
    stockdata = stockdata.to_numpy()
    exp_return = stockdata.mean(0)
    
    ranking = (exp_return-rf)/betas
    return ranking

def ranked_index(stockdata,marketdata,rf):
    ranked = single_index(stockdata,marketdata,rf)
    names = stockdata.columns.to_numpy()
    names = names.tolist()
    df = pd.DataFrame([ranked], columns = names,index=['Excess return to beta'])
    df = df.transpose()
    return df

def get_by_rank(stockdata,marketdata,rf,num):
    df = ranked_index(stockdata,marketdata,rf)
    df_sorted = df.sort_values('Excess return to beta',ascending = False)
    
    head = df_sorted.head(num)
    names = head.index.tolist()
    
    return stockdata[names]