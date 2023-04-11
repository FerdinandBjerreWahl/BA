import numpy as np
from scipy.optimize import minimize

def get_cov_mean_matrices(returns):
    cov_matrix = np.array(returns.cov())
    mean_matrix = np.array(returns.mean())
    return cov_matrix, mean_matrix


def objective_function(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))


def constraint_function(weights):
    return np.sum(weights) - 1


def efficient_frontier(mu, cov, target, rf, bounds, esg, returns, score):
    def negativeSR(w, mu=mu, cov=cov, rf=rf):
        w = np.array(w)
        V = np.sqrt(w.T @ cov @ w)
        R = np.sum(mu * w)
        SR = (R - rf) / V
        return -1 * SR

    # Define the initial guess for the weights
    x0 = np.ones(len(mu)) / len(mu)

    # Calculate the efficient frontier
    frontier = []
    portfolio_esg = []

    for ret in target:
        # Define the constraint that the expected return of the portfolio equals the desired return
        def ret_constraint(weights):
            return np.dot(weights, mu) - ret

        # Use the minimize function to find the portfolio weights that minimize the objective function subject to the constraints
        res = minimize(objective_function, x0, args=(cov,), method='SLSQP', constraints=[{'type': 'eq', 'fun': constraint_function}, {'type': 'eq', 'fun': ret_constraint}], bounds=bounds)

        # Calculate the volatility of the portfolio
        vol = np.sqrt(res.fun)

        # Calculate the Sharpe ratio of the portfolio
        sharpe = (ret - rf) / vol

        # Append the results to the frontier
        frontier.append((ret, vol, sharpe))

        # Find Sharpe ratio optimized weights
        w_opt = minimize(negativeSR, x0, method='SLSQP', bounds=bounds, constraints=({'type': 'eq', 'fun': constraint_function})).x

        # Calculate the environmental score of the portfolio
        result = 0
        for count, col in enumerate(returns.columns):
            first = esg[esg['Isin'] == col]
            env = first[score]
            env = env.to_numpy()[0]
            result += w_opt[count] * env

    portfolio_esg.append(result)

    # Convert the frontier to a numpy array
    frontier = np.array(frontier)

    # Find the portfolio with the highest Sharpe ratio
    max_sharpe_idx = np.argmax(frontier[:, 2])
    max_sharpe_ret, max_sharpe_vol, max_sharpe_sr = frontier[max_sharpe_idx]

    # Calculate standard deviations of securities
    stdevs = np.sqrt(np.diag(cov))
    stdevs_ = np.repeat(stdevs[:, np.newaxis], 100, axis=1)
    mu = np.repeat(mu[:, np.newaxis], 100, axis=1)

    return max_sharpe_ret, max_sharpe_vol, max_sharpe_sr, portfolio_esg, frontier, mu, stdevs_
