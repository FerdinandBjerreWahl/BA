import pandas as pd


def rank_stocks(returns,num,rf):
    '''
    Ranks stocks based on their Sharpe ratios.

    Args:
        returns (pd.DataFrame): The returns data for the stocks.
        num (int): The number of top-ranked stocks to select.
        rf (float): The risk-free rate.

    Returns:
        The subset of the top-ranked stocks' based on their Sharpe ratios as a DataFrame from their returns.
    '''
    SRs = (returns.mean()-rf)/returns.std()
    df = pd.DataFrame(SRs)
    df = df.rename(columns={0: 'Sharperatio'})
    df_sorted_desc = df.sort_values(by='Sharperatio', ascending=False)
    names = df_sorted_desc.head(num).index.tolist()
    num_window = returns[names]
    return num_window