import pandas as pd
import numpy as np
from Effient_Frontier import get_mean_matrices
from Effient_Frontier import get_cov_matrices
from Effient_Frontier import efficient_frontier 

def backtest(rf, esg, returns, score, window,get_plots=False,num=None):
    '''
    Performs a backtest of a portfolio strategy over a given window of periods.

    Args:
        rf (float): The risk-free rate.
        esg (pd.DataFrame): The ESG data.
        returns (pd.DataFrame): The returns data for the stocks.
        score (str): The ESG score column name.
        window (int): The number of periods in each window.
        num (int): The number of top-ranked stocks to select for investing.

    Returns:
        pd.DataFrame: A DataFrame containing the backtest results.
    '''
    #Find the how many periods the data has
    n = returns.shape[0]
    
    #Create a list with the initial value 100. This is used to see how much an investment of 100 would have developed over the periods
    portfolio_value = [100]
    
    #Create a bunch of lists where values will be stored
    expected_srs = []
    expected_stds = []
    expected_returns = []
    
    realized_srs = []
    realized_stds = []
    realized_returns = []
    
    portfolio_esgs = []
    
    #Loop over the periods
    for i in range(n-window-1):
        #The current window
        rwindow = returns[i:i+window]
        
        if num is not None:
            #Create a dataframe of Sharpe ratios, then using the top 'num' stocks to select for investing
            if num > rwindow.shape[1]:
                num = rwindow.shape[1]
            SRs = (rwindow.mean()-rf)/rwindow.std()
            df = pd.DataFrame(SRs)
            df = df.rename(columns={0: 'Sharperatio'})
            df_sorted_desc = df.sort_values(by='Sharperatio', ascending=False)
            names = df_sorted_desc.head(num).index.tolist()
            num_window = rwindow[names]
            mu = get_mean_matrices(num_window)
            cov = get_cov_matrices(num_window)[1]
            
            target = np.linspace(np.min(mu), np.max(mu), 100)
            
            bounds = [(0, 1) for _ in range(len(mu))]
            
            max_sharpe_ret, max_sharpe_vol, max_sharpe_sr, portfolio_esg, frontier, mu, stdevs , w_opt  = efficient_frontier(mu, cov, target, rf, bounds, num_window, esg, score,get_plots)
        else:
            mu = get_mean_matrices(rwindow)
            cov = get_cov_matrices(rwindow)[1]
            #Set the target, used for the efficient frontier module 
            target = np.linspace(np.min(mu), np.max(mu), 100)
        
            #Create the bounds
            bounds = [(0, 1) for _ in range(len(mu))]
            max_sharpe_ret, max_sharpe_vol, max_sharpe_sr, portfolio_esg, frontier, mu, stdevs , w_opt  = efficient_frontier(mu, cov, target, rf, bounds, rwindow, esg, score,get_plots)

        
        #Get the mean returns for the next period
        next_mu_window = returns[i+window:i+window+1]
        next_cov_window = returns[i+window-1:i+window+1]
        
        if num is not None:
            next_mu_names = next_mu_window[names]
            next_mu = get_mean_matrices(next_mu_names)
            
            next_cov_names = next_cov_window[names]
            cov = get_cov_matrices(next_cov_names)[1]
        else:
            next_mu = get_mean_matrices(next_mu_window)
            cov = get_cov_matrices(next_cov_window)[1]

        
        #Get the covariance matrix from the current period i to the next i+1

        
        #check_for_zeros(rwindow)
        #check_for_zeros(cov)
        
        #Compute the return if 100% was invested in the weights w_opt and held for the next period i+1
        realized_return = np.sum(next_mu *w_opt)
        realized_std = np.sqrt(np.dot(w_opt.T, np.dot(cov, w_opt)))
        SR = (realized_return-rf)/realized_std
        
        portfolio_value.append(portfolio_value[i]*(1+realized_return))
        
        #print(w_opt)
        #A bunch of prints to see the results while code is running
        print("For window: ", i)
        print("Expected return: ", max_sharpe_ret)
        print("Realized return: ", realized_return)
        print("Expected volatility: ", max_sharpe_vol)
        print("Realized volatility: ", realized_std)
        print("Portfolio value: ",portfolio_value[i]*(1+realized_return))
        print("\n")
        
        #Append results to the lists
        expected_srs.append(max_sharpe_sr)
        expected_stds.append(max_sharpe_vol)
        expected_returns.append(max_sharpe_ret)
        
        realized_srs.append(SR)
        realized_stds.append(realized_std)
        realized_returns.append(realized_return)
        
        portfolio_esgs.append(portfolio_esg)

    #Reshape the lists such that they are column vectors
    col1 = np.reshape(expected_returns, (-1, 1))
    col2 = np.reshape(realized_returns, (-1, 1))
    col3 = np.reshape(expected_stds, (-1, 1))
    col4 = np.reshape(realized_stds, (-1, 1))
    col5 = np.reshape(expected_srs, (-1, 1))
    col6 = np.reshape(realized_srs, (-1, 1))
    col7 = np.reshape(portfolio_value[1:], (-1, 1))
    col8 = np.reshape(portfolio_esgs,(-1,1))


    # Combine the column vectors into a single NumPy array
    data = np.hstack((col1, col2, col3, col4, col5, col6, col7,col8))

    # Create a Pandas DataFrame from the NumPy array
    df = pd.DataFrame(data, columns=['Expected returns', 'Realized returns', 'Expected stds', 'Realized stds', 'Expected srs', 'Realized srs', 'Portfolio Value','Portfolio E Score'])
    
    return df

