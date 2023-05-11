import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def alpha_beta_calc(stockdata,marketdata):
    '''Calculates alpha and beta values for each stock with respect to the marketdata.
    
    Args:
    stockdata (DataFrame): Stock data of historical prices.
    marketdata (DataFrame): Market data of historical prices.

    Returns:
    Two lists of floats that contains alpha values and the second list contains beta values
    for each stock.
    '''
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
    '''Calculates the ranking of stocks according to the single-index model.
    
    Args:
    stockdata (DataFrame): Stock data of historical returns.
    marketdata (DataFrame): Market data of historical returns.
    rf (float): The risk-free rate.
    
    Returns:
    The ranking of each stock based on the single-index model.
    '''

    alphas, betas = alpha_beta_calc(stockdata,marketdata)
    stockdata = stockdata.to_numpy()
    exp_return = stockdata.mean(0)
    
    ranking = (exp_return-rf)/betas
    return ranking

def ranked_index(stockdata,marketdata,rf):
    '''Calculates excess return to beta for a single index and ranks the stocks accordingly.
    
    Args:
       stockdata(DataFrame) : Stock data of historical returns for each stock.
       marketdata(DataFrame) : Market data of historical returns.
       rf(float) : The risk-free rate of return.
       
    Returns:
    The excess return to beta ranking of each stock.
    '''

    ranked = single_index(stockdata,marketdata,rf)
    names = stockdata.columns.to_numpy()
    names = names.tolist()
    df = pd.DataFrame([ranked], columns = names,index=['Excess return to beta'])
    df = df.transpose()
    return df

def get_by_rank(stockdata,marketdata,rf,num):
    '''Calculates the top N stocks sorted by the Excess return to beta ratio, as calculated by the single_index()function.
    
    Args:
    stockdata(DataFrame): Stock data of historical prices.
    marketdata(DataFrame): Market data of historical prices.
    rf(float): A risk-free rate return
    num(int): The number of top stocks to return.
    
    Returns:
    The top N stocks sorted by the Excess return to beta ratio.
    '''

    df = ranked_index(stockdata,marketdata,rf)
    df_sorted = df.sort_values('Excess return to beta',ascending = False)
    
    head = df_sorted.head(num)
    names = head.index.tolist()
    
    return stockdata[names]