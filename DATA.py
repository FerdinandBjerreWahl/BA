from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os.path


def read_data(file_path):
    ''' Reads data from file path.
    
    Args:
        file_path (string): location of data file
        
    Returns:
        DataFrame: Input data
    '''
    return pd.read_csv(file_path)

def get_isins(data):
    ''' Isolates Isin codes from DataFrame
    
    Args:
        data (DataFrame): Pandas DataFrame containing esg data
    
    Returns:
        list: Isin codes
    '''
    return data['Isin'].tolist()


def concatenate_data(prefixes, isindata, start_date, end_date):
    ''' Fixes timezone issues in data by isolating by prefix and rejoining the data
    
    Args:
        prefixes (string): The different prefixes of isin codes in the loaded data
        isindata (list): Isin codes to process
        start_date (string): Start date of data
        end_date (string): End date of data
    
    Returns:
        DataFrame: Combined data with fixed timezones
    
    '''
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
    '''
    Remove columns from a DataFrame that contain null values (represented as 0).

    Args:
        data (pandas.DataFrame): The input DataFrame.

    Returns:
        A new DataFrame obtained by dropping the columns containing null values.
    '''
    null_columns = []
    for col in data.columns:
        if (data[col]==0).any():
            null_columns.append(col)
    return data.drop(columns=null_columns)

def filter_by_column(data,stock_data, column_name, column_value=None):
    '''
    Filter stock data based on a specific column and its value.

    Args:
        data (pandas.DataFrame): The data used for filtering.
        stock_data (pandas.DataFrame): The stock data to be filtered.
        column_name (str): The name of the column used for filtering.
        column_value (str): The value to filter the column by. If not provided, no filtering is applied.

    Returns:
        The filtered stock data based on the specified column and value, or the original stock data if no filtering is applied.
    '''
    if column_value is not None:
        filtered = data[data[column_name] == column_value]
        data_isin = get_isins(filtered)
        filtered_data = [c for c in stock_data.columns if c in data_isin]
        subset = stock_data[filtered_data]
        return subset
    else:
        return stock_data
    

    
def filter_by_value(stock_data, data, column_name, threshold=None, operator=None):
    '''
    Filter stock data based on a specific column and its value using a threshold and operator.

    Args:
        stock_data (pandas.DataFrame): The stock data to be filtered.
        data (pandas.DataFrame): The data used for filtering.
        column_name (str): The name of the column used for filtering.
        threshold (optional): The threshold value for the filtering. If not provided, no filtering is applied.
        operator (optional): The operator used for the comparison. Supported operators: 'leq' (less than or equal to),
                             'geq' (greater than or equal to). If not provided, no filtering is applied.

    Returns:
       The filtered stock data based on the specified column, threshold, and operator, or the original stock data if no filtering is applied '''
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
    

def to_date(data,time):
    '''
    Convert the data(stock_data) to a specified time frequency.

    Args:
        data (pandas.DataFrame): The data to be converted.
        time (str): The time frequency to which the data will be converted. Default is 'y' (yearly).

    Returns:
        The converted data resampled to the specified time frequency.
    '''
    adj_pct = data.ffill().pct_change()
    adj_pct.index = pd.to_datetime(adj_pct.index)
    result = adj_pct.resample(time).sum()
    return result

def load_or_process_data(file_path, prefixes, start_date, end_date, time):
    '''
    Load or process data based on the file path and specified parameters.

    Args:
        file_path (str): The path to the data file.
        prefixes (list): List of data prefixes.
        start_date (str): The start date for data processing.
        end_date (str): The end date for data processing.
        time (str): The time frequency for data resampling.

    Returns:
        A tuple containing the loaded or processed data, ISINs, and stock data.
    '''

    pickle_file_path = f"{file_path}.pickle"
    if os.path.isfile(pickle_file_path):
        with open(pickle_file_path, "rb") as f:
            loaded_data = pickle.load(f)
            data = loaded_data[0]
            isins = loaded_data[1]
            initial_stock_data = loaded_data[2]
            stock_data = to_date(initial_stock_data, time)
            stock_data = remove_null_columns(stock_data)
    else:
        data = read_data(file_path)
        isins = get_isins(data)
        initial_stock_data = concatenate_data(prefixes, isins, start_date, end_date)
        stock_data = to_date(initial_stock_data, time)
        stock_data = remove_null_columns(stock_data)

        with open(pickle_file_path, "wb") as f:
            pickle.dump((data, isins, initial_stock_data), f)
            
    return data, isins, stock_data



def delete_pickle_file(filename):
    '''
    Delete a pickle file.

    Args:
        filename (str): The name of the pickle file to be deleted.

    Returns:
        None 
    '''
    if os.path.exists(filename):
        os.remove(filename)
        print(f"File '{filename}' has been deleted.")
    else:
        print(f"File '{filename}' does not exist.")



def get_filtered_stock_data(file_path, column_name, column_value, prefixes, start_date, end_date, time='y', threshold=None, operator=None):
    '''
    Get filtered stock data based on specified criteria.

    Args:
        file_path (str): The path to the data file.
        column_name (str): The name of the column to filter on.
        column_value: The value to filter on. If `threshold` is provided, this argument is ignored.
        prefixes (list): A list of stock prefixes.
        start_date (str): The start date of the data range.
        end_date (str): The end date of the data range.
        time (str): The time frequency for resampling the data. Default is 'y' (yearly).
        threshold (int): The threshold value to use for filtering. Default is None.
        operator (str): The operator to use for comparison when filtering by threshold. 
                                 Possible values are 'leq' (less than or equal to) and 'geq' (greater than or equal to).
                                 Default is None.

    Returns:
        The filtered stock data as dataframe
    '''
    data, isins, stock_data = load_or_process_data(file_path, prefixes, start_date, end_date,time)
    stock_data = stock_data['Adj Close']
    
    if isinstance(threshold, int):
        filtered_data = filter_by_value(stock_data, data, column_name, threshold, operator)
    elif threshold is None:
        filtered_data = filter_by_column(data,stock_data, column_name, column_value)
        
    return filtered_data