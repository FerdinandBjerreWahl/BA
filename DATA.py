from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os.path


def read_data(file_path):
    return pd.read_csv(file_path)

def get_isins(data):
    return data['Isin'].tolist()


def concatenate_data(prefixes, isindata, start_date, end_date):
    yf.pdr_override()
    data_list = []
    for prefix in prefixes:
        filtered_isin = []
        for i in range(0, len(isindata)):
            if isindata[i][:len(prefix)] == prefix:
                filtered_isin.append(isindata[i])
        data = pdr.get_data_yahoo(filtered_isin, start=start_date, end=end_date)
        data_list.append(data)
    combined_data = pd.concat(data_list, sort=False, axis=1, join='inner')
    return combined_data

def remove_null_columns(data):
    null_columns = []
    for col in data.columns:
        if data[col].isnull().all():
            null_columns.append(col)
    return data.drop(columns=null_columns)



def filter_by_column(data,stock_data, column_name, column_value=None):
    if column_value is not None:
        filtered = data[data[column_name] == column_value]
        data_isin = get_isins(filtered)
        filtered_data = [c for c in stock_data.columns if c in data_isin]
        subset = stock_data[filtered_data]
        return subset
    else:
        return stock_data
    

    
def filter_by_value(stock_data, data, column_name, threshold=None, operator=None):
    if threshold is not None:
        if operator == 'leq':
            filtered=data[data[column_name] <= threshold]
            data_isin = get_isins(filtered)
            filtered_data = [c for c in stock_data.columns if c in data_isin]
            subset = stock_data[filtered_data]
            return subset

        if operator == 'geq':
            filtered=data[data[column_name] >= threshold]
            data_isin = get_isins(filtered)
            filtered_data = [c for c in stock_data.columns if c in data_isin]
            subset = stock_data[filtered_data]
            return subset

        else:
            return stock_data
    else:
        return stock_data
    

def to_date(data,time='y'):
    adj_pct = data.ffill().pct_change()
    adj_pct.index = pd.to_datetime(adj_pct.index)
    result = adj_pct.resample(time).sum()
    return result



def load_or_process_data(file_path, prefixes, start_date, end_date):
    
    pickle_file_path = f"{file_path}.pickle"
    if os.path.isfile(pickle_file_path):
        with open(pickle_file_path, "rb") as f:
            data = pickle.load(f)
            isins = pickle.load(f)
            stock_data = pickle.load(f)
    else:
        data = read_data(file_path)
        isins = get_isins(data)
        stock_data = concatenate_data(prefixes, isins, start_date, end_date)
        stock_data = remove_null_columns(stock_data)

       
        with open(pickle_file_path, "wb") as f:
            pickle.dump(data, f)
            pickle.dump(isins, f)
            pickle.dump(stock_data, f)
            
    return data, isins, stock_data



def get_filtered_stock_data(file_path, column_name, column_value, prefixes, start_date, end_date, time='y', threshold=None, operator=None):
    data, isins, stock_data = load_or_process_data(file_path, prefixes, start_date, end_date)
    stock_data = stock_data['Adj Close']
    
    if isinstance(threshold, int):
        filtered_data = filter_by_value(stock_data, data, column_name, threshold, operator)
    elif threshold is None:
        filtered_data = filter_by_column(data,stock_data, column_name, column_value)
    
    formatted_data = to_date(filtered_data, time)
    return formatted_data
