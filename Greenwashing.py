import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import ShrunkCovariance

def greenwashing(returns,esg,ESG_threshold,rf,score):
    #Get the esg data and transform it such that it can be used in calculations
    ESG2 = 1000
    esgdf = pd.DataFrame((esg['Isin'],esg[score])).transpose()
    esgdf.index = list(esgdf["Isin"])
    esgdf = esgdf.drop('Isin',axis=1)
    
    topESG = esgdf.loc[returns.columns.to_list()]
    topESG = topESG[~topESG.index.duplicated(keep='first')].to_numpy()   
    def optimize_portfolio(weights, stock_data, ESG_data, ESG_threshold,rf):
        portfolio_return = np.sum(returns.mean() @ weights)
        #måske brug shrinkage cov
        npreturns = returns.to_numpy()
        shrinkage_cov = ShrunkCovariance(shrinkage=0.2).fit(npreturns).covariance_
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(shrinkage_cov, weights)))
        sharpe_ratio = (portfolio_return-rf) / portfolio_std_dev
        ESG_score = np.sum(weights @ topESG)
    
        # Apply the ESG constraint by adding a penalty term to the objective function
        if ESG_score < ESG_threshold or ESG_score > ESG2:
            #print("Før",sharpe_ratio)
            sharpe_ratio *= 0.1
            #print('Efter',sharpe_ratio)
            #print('Negativ',-sharpe_ratio)
        return -sharpe_ratio
    
    init_guess = np.ones(len(topESG)) / len(topESG)
    
    bounds = [(0, 1) for _ in range(len(topESG))]
    
    constraint = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
               {'type': 'ineq', 'fun': lambda x: np.dot(x,topESG) - ESG_threshold},
               {'type': 'ineq', 'fun': lambda x: ESG2 - np.dot(x,topESG)})

    result = minimize(optimize_portfolio, init_guess, args=(returns, esg, ESG_threshold,rf), bounds=bounds, constraints=constraint)
    
    realized_return = np.sum(returns.mean() @ result.x)
    npreturns = returns.to_numpy()
    shrinkage_cov = ShrunkCovariance(shrinkage=0.2).fit(npreturns).covariance_
    realized_std = np.sqrt(np.dot(result.x.T, np.dot(shrinkage_cov, result.x)))
    realized_ESG = np.sum(result.x @ topESG)
    sharp_ration = (realized_return-rf)/realized_std 
    
    
    # Print the results
    print('Optimal weights:', np.round(result.x,4))
    print('Optimized Sharpe ratio:', -result.fun)
    print('Sharp ratio', sharp_ration)
    print('Reazlied return', realized_return)
    print('Realized std',realized_std)
    print('ESG score', realized_ESG)
    
    return result.x, -result.fun, realized_return, realized_std, realized_ESG