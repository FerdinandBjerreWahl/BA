import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import ShrunkCovariance
from Greenwashing import greenwashing
from DATA import get_filtered_stock_data as gfs


def get_mean_matrices(returns):
    '''Calculates mean matrix from the returns 
    
    Args:
       returns(DataFrame) : A pandas DataFrame containing stock data of historical.      
     
    Returns:
        The mean matrix of returns, represented as numpy arrays.
    '''
    npreturns = returns.to_numpy()
    mean_matrix = np.array(returns.mean())
    return mean_matrix

def get_cov_matrices(returns, shrink = 0.2):
    '''Calculates the covariance matrix and a shrinkage covariance matrix
    
    Args:
       returns(DataFrame) : A pandas DataFrame containing stock data of historical.      
     
    Returns:
        The covariance matrix and a shrinkage covariance matrix both represented as numpy arrays of returns.
    '''
    npreturns = returns.to_numpy()
    shrinkage_cov = ShrunkCovariance(shrinkage=shrink).fit(npreturns).covariance_
    cov_matrix = np.array(returns.cov())
    return cov_matrix, shrinkage_cov

def objective_function(weights, cov_matrix):
    '''Computes the value of the objective function for a given set of weights and covariance matrix.
    
    Args:
        weights(numpy.ndarray) : A numpy array containing the weights for each stock in the portfolio.
        cov_matrix (numpy.ndarray) :  A numpy array containing the covariance matrix of stock returns.
    
    Returns:
        The value of the objective function for the given weights and covariance matrix.

    '''
    return np.dot(weights.T, np.dot(cov_matrix, weights))


def constraint_function(weights):
    '''Computes the value of the constraint function for a given set of weights.
    
    Args:
        weights(numpy.ndarray) : A numpy array containing the weights for each stock in the portfolio.
        
    Returns:
        The value of the constraint function for the given weights    
    '''
    return np.sum(weights) - 1


def efficient_frontier(mu, cov, target, rf, bounds, returns, esg=None, score=None,get_plots = False):
    ''' The function efficient_frontier calculates the efficient frontier for a given set of expected returns and a covariance matrix,and also finds the portfolio with the highest Sharpe ratio and calculates Score for this
    Args:
        mu(numpy.ndarray) : Expected returns for each stock
        cov(numpy.ndarray) : Covariance matrix for the stocks
        target(list) : Desired expected returns for the efficient frontier
        rf(float) : Risk-free rate
        bounds(tuple) : Containing the bounds for the weights of the portfolio
        esg(DataFrame) : Containing the ESG scores for the stocks
        returns(DataFrame) : Containing the historical returns for the stocks
        score(string) : Representing the ESG score to optimize for
    Returns:
        max_sharpe_ret : Expected return of the portfolio with the highest Sharpe ratio
        max_sharpe_vol : Volatility of the portfolio with the highest Sharpe ratio
        max_sharpe_sr : Sharpe ratio of the portfolio with the highest Sharpe ratio
        portfolio_esg : Containing the ESG score of the portfolio for each desired return in target
        frontier : The expected return, volatility, and Sharpe ratio for each point on the efficient frontier
        mu : Expected returns for use in plotting the efficient frontier
        stdevs_ : Repeated standard deviations for use in plotting the efficient frontier'''  
    def negativeSR(w, mu=mu, cov=cov, rf=rf):
        
        w = np.array(w)
        V = np.sqrt(w.T @ cov @ w)
        R = np.sum(mu * w)
        SR = (R - rf) / V
        return -1 * SR

    x0 = np.ones(len(mu)) / len(mu)
    
    frontier = []
    portfolio_esg = []
    if get_plots == True:
        for ret in target:
            def ret_constraint(weights):
                return np.dot(weights, mu) - ret

            res = minimize(objective_function, x0, args=(cov,), method='SLSQP', constraints=[{'type': 'eq', 'fun': constraint_function}, {'type': 'eq', 'fun': ret_constraint}], bounds=bounds)

            vol = np.sqrt(res.fun)

            sharpe = (ret - rf) / vol

            frontier.append((ret, vol, sharpe))
            
    w_opt = minimize(negativeSR, x0, method='SLSQP', bounds=bounds, options=({'disp':True,'maxiter':1000}), constraints=[{'type': 'eq', 'fun': constraint_function}]).x
    result = 0
    if score is not None or esg is not None:
        for count, col in enumerate(returns.columns):
            first = esg[esg['Isin'] == col]
            env = first[score]
            env = env.to_numpy()[0]
            result += w_opt[count] * env

    portfolio_esg.append(result)

    #frontier = np.array(frontier)

    #max_sharpe_idx = np.argmax(frontier[:, 2])
    #max_sharpe_ret, max_sharpe_vol, max_sharpe_sr = frontier[max_sharpe_idx]
    
    max_sharpe_ret = np.sum(mu * w_opt)
    max_sharpe_vol = np.sqrt(w_opt.T @ cov @ w_opt)
    max_sharpe_sr = (max_sharpe_ret - rf) / max_sharpe_vol
    
    stdevs = np.sqrt(np.diag(cov))
    stdevs_ = np.repeat(stdevs[:, np.newaxis], 100, axis=1)
    mu = np.repeat(mu[:, np.newaxis], 100, axis=1)
    
    frontier = np.array(frontier)

    return max_sharpe_ret, max_sharpe_vol, max_sharpe_sr, portfolio_esg, frontier, mu, stdevs_ , w_opt

def ESG_efficient_frontier(file_path, column_name, column_value, prefixes, start_date, end_date, time='y', operator=None, esg=None, rf=None, score=None,get_plots = False): 
    '''
    Computes the Max_sharp and ESG based on specified criteria and constraints. In this case all the stocks ESG scores needs do be over the tresholds to be in the optimal portfolio.
    Args:
        file_path (str): The path to the file containing the data.
        column_name (str): The name of the column containing the data.
        column_value (str): The value to filter the data column.
        prefixes (list): Prefixes for the data columns.
        start_date (str): The start date of the data range.
        end_date (str): The end date of the data range.
        time (str, optional): The time period of the data. Defaults to 'y'.
        lower_bound (float, optional): The lower bound value for the weights. Defaults to 0.
        operator (str, optional): The operator to use for filtering the data. Defaults to None.
        esg (str, optional): The ESG criterion to consider. Defaults to None.
        rf (float, optional): The risk-free rate. Defaults to None.
        score (str, optional): The scoring method for ranking stocks. Defaults to None.

    Returns:
        A tuple containing two lists - Max_sharp and ESG calculated from all the inputs
    ''' 
    thresholds = range(0,800,25) # create a list of thresholds to iterate over
    Max_sharp = []
    ESG = []
    for i in thresholds:
        print(i)
        data = gfs(file_path, column_name, column_value, prefixes, start_date, end_date, time='y', threshold=i, operator=operator)
        mu = get_mean_matrices(data)
        cov = get_cov_matrices(data)[1]
        bounds = [(0, 1) for _ in range(len(mu))]
        target = np.linspace(np.min(mu), np.max(mu), 100)
        efficient_frontier2 = efficient_frontier(mu, cov, target, rf, bounds, data, esg=esg, score=score,get_plots = get_plots)
        Max_sharp.append(efficient_frontier2[2])
        print('Sharp ratio:', efficient_frontier2[2])
        ESG.append(efficient_frontier2[3])
        print('ESG score:', efficient_frontier2[3])
    return Max_sharp,ESG



def ESG_efficient_frontier_gw(rf,returns,esg,score):
    '''
    Computes the Efficient ESG Frontier based on specified criteria and constraints. In this case all the stocks weighted average ESG scores needs do be over threshold to be in the optimal portfolio.

    Args:
        rf (float): The risk-free rate.
        returns (numpy.ndarray): The returns data for the stocks.
        esg (numpy.ndarray): The ESG data for the stocks.
        score (str): The scoring method for ranking stocks.

    Returns:
        tuple: A tuple containing two lists - Max_sharp and ESG from all the inputs
    '''
    thresholds = range(400,800,25) # create a list of thresholds to iterate over
    
    Max_sharp = []
    ESG = []
    for i in thresholds:
        print(i)
        weights, sharpe, realized_return, realized_std, ESG_score = greenwashing(returns,esg,i,rf,score)
        Max_sharp.append(sharpe)
        print('Sharp ratio',sharpe)
        ESG.append(ESG_score)             
    return Max_sharp,ESG