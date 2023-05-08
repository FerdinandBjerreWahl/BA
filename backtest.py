import pandas as pd
import numpy as np
from Effient_Frontier import get_cov_mean_matrices
from Effient_Frontier import efficient_frontier 

def backtest(rf, esg, returns, score, window, num,lower_bound):
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
        
        #Create a dataframe of Sharpe ratios, then using the top 'num' stocks to select for investing
        SRs = (rwindow.mean()-rf)/rwindow.std()
        df = pd.DataFrame(SRs)
        df = df.rename(columns={0: 'Sharperatio'})
        df_sorted_desc = df.sort_values(by='Sharperatio', ascending=False)
        names = df_sorted_desc.head(num).index.tolist()
        num_window = rwindow[names]

        #Get the mean return of the stocks and the covariance matrix
        mu = get_cov_mean_matrices(num_window)[1]
        cov = get_cov_mean_matrices(num_window)[2]
        
        #Set the target, used for the efficient frontier module 
        target = np.linspace(np.min(mu), np.max(mu), 100)
        
        #Create the bounds
        bounds = [(0, 1) for _ in range(len(mu))]
        
        #Alot of computation here is unessecary for the backtest. 
        max_sharpe_ret, max_sharpe_vol, max_sharpe_sr, portfolio_esg, frontier, mu, stdevs , w_opt  = efficient_frontier(mu, cov, target, rf, bounds, esg, num_window, score,lower_bound)
        
        #Get the mean returns for the next period
        next_mu_window = returns[i+window:i+window+1]
        next_mu_names = next_mu_window[names]
        next_mu = get_cov_mean_matrices(next_mu_names)[1]
        
        #Get the covariance matrix from the current period i to the next i+1
        next_cov_window = returns[i+window-1:i+window+1]
        next_cov_names = next_cov_window[names]
        cov = get_cov_mean_matrices(next_cov_names)[2]
        
        #check_for_zeros(rwindow)
        #check_for_zeros(cov)
        
        #Compute the return if 100% was invested in the weights w_opt and held for the next period i+1
        realized_return = np.sum(next_mu *w_opt)
        realized_std = np.sqrt(np.dot(w_opt.T, np.dot(cov, w_opt)))
        SR = (realized_return-rf)/realized_std
        
        portfolio_value.append(portfolio_value[i]*(1+realized_return))
        
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