from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os.path
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from DATA import get_filtered_stock_data as gfs
from DATA import read_data
from DATA import delete_pickle_file
from Effient_Frontier import get_cov_matrices
from Effient_Frontier import get_mean_matrices
from Effient_Frontier import efficient_frontier 
from Effient_Frontier import ESG_efficient_frontier
from Effient_Frontier import ESG_efficient_frontier_gw
from backtest import backtest
from Greenwashing import greenwashing

def mean_test(x):
    '''
    Performs a test on the mean calculations given a predefined return matrix x

    Args:
        x: A return matrix in numpy'''
    df = pd.DataFrame(x)
    mean_array = df.mean().to_numpy()
    mean = np.array([4,5,6])
    assert np.allclose(mean_array,mean), "The mean calculations failed"
    return "Passed"
    
def cov_test(x):
    '''
    Performs a test on the covariance calculations given a predefined return matrix x

    Args:
        x: A return matrix in numpy'''
    covtest = np.array([[9,9,9],
                [9,9,9],
                [9,9,9]])
    df = pd.DataFrame(x)
    cov = df.cov().to_numpy()
    assert np.allclose(cov,covtest), "The covariance calculations failed"
    return "Passed"
    
def returns_test(x):
    '''
    Performs a test on the returns calculations given a predefined return matrix x

    Args:
        x: A return matrix in numpy'''
    weights = np.array([1/3,1/3,1/3])
    test_return = 5
    df = pd.DataFrame(x)
    mean = df.mean().to_numpy()
    assert np.allclose(np.sum(weights @ mean),test_return), "The returns calculations failed"
    return "Passed"
    
def deviation_test(x):
    '''
    Performs a test on the standard deviation calculations given a predefined return matrix x

    Args:
        x: A return matrix in numpy'''
    df = pd.DataFrame(x)
    cov = df.cov().to_numpy()
    weights = np.array([1/3,1/3,1/3])
    assert np.allclose(np.sqrt(weights.T @ cov @ weights),3), "The standard deviation calculations failed"
    return "Passed"
    
def efficientfrontier_test(x):
    '''
    Performs a test on the efficient frontier calculations given a predefined return matrix x

    Args:
        x: A return matrix in numpy'''
    #file_path = "ESG_US.csv"
    #esg = pd.read_csv(file_path)
    
    col = np.array(['Isin', 'CurrencyCode', 'Ticker', 'company_name',
       'environment_grade', 'environment_level', 'environment_score',
       'esg_id', 'exchange_symbol', 'governance_grade',
       'governance_level', 'governance_score', 'last_processing_date',
       'social_grade', 'social_level', 'social_score', 'stock_symbol',
       'total', 'total_grade', 'total_level'])
    
    dat = np.array([['US0268747849', 'USD', 'System.Object[]',
        'American International Group, Inc.', 'A', 'High', 538, 390,
        'NYSE', 'BB', 'Medium', 314, 20230221, 'BB', 'Medium', 315,
        'AIG', 1167, 'BBB', 'High'],
       ['US03027X1000', 'USD', 'System.Object[]',
        'American Tower Corporation (REIT)', 'A', 'High', 515, 415,
        'NYSE', 'BB', 'Medium', 305, 20230221, 'BB', 'Medium', 314,
        'AMT', 1134, 'BBB', 'High'],
       ['US0304201033', 'USD', 'System.Object[]',
        'American Water Works Company, Inc.', 'AA', 'Excellent', 680,
        418, 'NYSE', 'BB', 'Medium', 300, 20230221, 'BBB', 'High', 407,
        'AWK', 1387, 'A', 'High']])
    
    esg = pd.DataFrame(dat,columns=col)
    
    df = pd.DataFrame(x)
    mean = df.mean().to_numpy()
    
    cov = df.cov().to_numpy()
    
    bounds = [(0, 1) for _ in range(len(mean))] 
    
    target = np.linspace(np.min(mean), np.max(mean), 100)
    
    rf = 0.0
    
    df = df.rename(columns={0: "US0268747849", 1: "US03027X1000",2: "US0304201033"})
    
    max_sharpe_ret, max_sharpe_vol, max_sharpe_sr, portfolio_esg, frontier, mu, stdevs_ , w_opt = efficient_frontier(mean, cov, target, rf, bounds, df, esg=esg, score='environment_score',get_plots = False)
    print(portfolio_esg)
    assert np.allclose(w_opt,np.array([0,0,1])), "The optimal weights calculations failed"
    assert portfolio_esg[0] == 680.0, "Weighted portfolio E score calculations failed"
    return "Passed","Passed"

def backtest_test(x):
    '''
    Performs a test on the backtest calculations given a predefined return matrix x

    Args:
        x: A return matrix in numpy'''
    result = np.array([4.5,9,1.5,1.5,3,6,1000,680])
    
    rf = 0.0
    col = np.array(['Isin', 'CurrencyCode', 'Ticker', 'company_name',
       'environment_grade', 'environment_level', 'environment_score',
       'esg_id', 'exchange_symbol', 'governance_grade',
       'governance_level', 'governance_score', 'last_processing_date',
       'social_grade', 'social_level', 'social_score', 'stock_symbol',
       'total', 'total_grade', 'total_level'])
    
    dat = np.array([['US0268747849', 'USD', 'System.Object[]',
        'American International Group, Inc.', 'A', 'High', 538, 390,
        'NYSE', 'BB', 'Medium', 314, 20230221, 'BB', 'Medium', 315,
        'AIG', 1167, 'BBB', 'High'],
       ['US03027X1000', 'USD', 'System.Object[]',
        'American Tower Corporation (REIT)', 'A', 'High', 515, 415,
        'NYSE', 'BB', 'Medium', 305, 20230221, 'BB', 'Medium', 314,
        'AMT', 1134, 'BBB', 'High'],
       ['US0304201033', 'USD', 'System.Object[]',
        'American Water Works Company, Inc.', 'AA', 'Excellent', 680,
        418, 'NYSE', 'BB', 'Medium', 300, 20230221, 'BBB', 'High', 407,
        'AWK', 1387, 'A', 'High']])
    
    esg = pd.DataFrame(dat,columns=col)
   
    
    score = 'environment_score'

    window = 2
    
    get_plots = False
    
    num = None
    
    df = pd.DataFrame(x)
    df = df.rename(columns={0: "US0268747849", 1: "US03027X1000",2: "US0304201033"})
    
    bt = backtest(rf,esg,df,score,window,get_plots=get_plots,num=None, test = True)
    
    btnp = bt.to_numpy()
    assert np.allclose(result,btnp), "Backtest calculations failed"
    return "Passed"

def optimizer_test(x):
    '''
    A test to see if the efficient frontier method will invest 100% into index 0 stock
    
    Args:
        x: A return matrix in numpy
    '''
    df = pd.DataFrame(x)
    mean = df.mean().to_numpy()
    
    cov = df.cov().to_numpy()
    
    bounds = [(0, 1) for _ in range(len(mean))] 
    
    target = np.linspace(np.min(mean), np.max(mean), 100)
    
    rf = 0.0
    
    max_sharpe_ret, max_sharpe_vol, max_sharpe_sr, portfolio_esg, frontier, mu, stdevs_ , w_opt = efficient_frontier(mean, cov, target, rf, bounds, df, esg=None, score=None,get_plots = False)
    assert np.allclose(w_opt,np.array([1,0,0])), "The optimizer test failed"
    return "Passed"
    
def test_all():
    '''
    Runs all other tests in the file
    '''
    x = np.array([[1,2,3],
                       [4,5,6],
                       [7,8,9]])
    
    matrix = np.array([[-1.0, -3, -5],
                   [2.0, 3.0, 4.0],
                   [3.0, 4, 5.0]])


    test_results = {
    'Mean calculation test': mean_test(x),
    'Covariance calculations test': cov_test(x),
    'Returns calculations test': returns_test(x),
    'Standard deviation calculations test': deviation_test(x),
    'Optimal weights calculations test': efficientfrontier_test(x)[0],
    'Optimizer test': optimizer_test(matrix),
    'Weighted E score of portfolio': efficientfrontier_test(x)[1],
    'Backtest calculations':  backtest_test(x),
    'All tests': 'Passed'}
    
    max_width = max(len(test_name)+3 for test_name in test_results.keys())

    # Print the test results with aligned messages
    for test_name, result in test_results.items():
        output = f'{test_name:{max_width}}: {result}'
        print(output)
        
if __name__ == '__main__':
    test_all()