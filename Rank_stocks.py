import pandas as pd


def rank_stocks(returns,num,rf):
    SRs = (returns.mean()-rf)/returns.std()
    df = pd.DataFrame(SRs)
    df = df.rename(columns={0: 'Sharperatio'})
    df_sorted_desc = df.sort_values(by='Sharperatio', ascending=False)
    names = df_sorted_desc.head(num).index.tolist()
    num_window = returns[names]
    return num_window