import numpy as np
from scipy.optimize import minimize

def get_cov_mean_matrices(returns):
    '''Calculates covariance matrix and mean matrix from the returns 
    
    Args:
       returns(DataFrame) : A pandas DataFrame containing stock data of historical.      
     
    Returns:
        The covariance matrix and mean matrix, both represented as numpy arrays.
    '''
    
    cov_matrix = np.array(returns.cov())
    mean_matrix = np.array(returns.mean())
    return cov_matrix, mean_matrix


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


def efficient_frontier(mu, cov, target, rf, bounds, esg, returns, score):
    
    
     ''' The function efficient_frontier calculates the efficient frontier for a given set of expected returns and a covariance matrix, and also finds the portfolio with the highest Sharpe ratio and calculates Score for this

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
        stdevs_ : Repeated standard deviations for use in plotting the efficient frontier
        '''  
    def negativeSR(w, mu=mu, cov=cov, rf=rf):
        w = np.array(w)
        V = np.sqrt(w.T @ cov @ w)
        R = np.sum(mu * w)
        SR = (R - rf) / V
        return -1 * SR

    x0 = np.ones(len(mu)) / len(mu)
    
    
    frontier = []
    portfolio_esg = []

    for ret in target:
        def ret_constraint(weights):
            return np.dot(weights, mu) - ret
        
        res = minimize(objective_function, x0, args=(cov,), method='SLSQP', constraints=[{'type': 'eq', 'fun': constraint_function}, {'type': 'eq', 'fun': ret_constraint}], bounds=bounds)

        vol = np.sqrt(res.fun)
      
        sharpe = (ret - rf) / vol

        frontier.append((ret, vol, sharpe))

        w_opt = minimize(negativeSR, x0, method='SLSQP', bounds=bounds, constraints=({'type': 'eq', 'fun': constraint_function})).x

        result = 0
        for count, col in enumerate(returns.columns):
            first = esg[esg['Isin'] == col]
            env = first[score]
            env = env.to_numpy()[0]
            result += w_opt[count] * env

    portfolio_esg.append(result)

    frontier = np.array(frontier)

    max_sharpe_idx = np.argmax(frontier[:, 2])
    max_sharpe_ret, max_sharpe_vol, max_sharpe_sr = frontier[max_sharpe_idx]

    stdevs = np.sqrt(np.diag(cov))
    stdevs_ = np.repeat(stdevs[:, np.newaxis], 100, axis=1)
    mu = np.repeat(mu[:, np.newaxis], 100, axis=1)

    return max_sharpe_ret, max_sharpe_vol, max_sharpe_sr, portfolio_esg, frontier, mu, stdevs_ 
