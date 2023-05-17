import pandas as pd
import numpy as np
import inspect

def get_company_names(isin_array, dataframe):
    """
    Get the company names corresponding to Isin values from a DataFrame.

    Parameters:
        isin_array (numpy.ndarray): Array of Isin values.
        dataframe (pandas.DataFrame): DataFrame containing Isin and company name columns.

    Returns:
        List of company names corresponding to the given Isin values.
    """
    company_names = []
    for isin in isin_array:
        # Locate the row with matching ISIN and retrieve the company name
        company_name = dataframe.loc[dataframe['Isin'] == isin, 'company_name'].values

        if len(company_name) > 0:
            # Append the company name to the list if a match is found
            company_names.append(company_name[0])
        else:
            # Append None if no match is found for the ISIN
            company_names.append(None)
    
    return company_names

def print_function_comments(func):
    """
    Get the documentation from a function
    
    Parameters:
       func: The name of a function

    Returns:
        Nothing. But prints the comments/documentation of a function
    """
    comments = inspect.getdoc(func)
    if comments:
        print(f"Comments for {func.__name__}:")
        print(comments)
    else:
        print(f"No comments found for {func.__name__}.")
        

def named_weights(returns,w_opt):
    """
    Combines the weights with the names of the companies in a dataframe
    
    Parameters:
        returns: dataframe of returns used to get the w_opt parameter
        w_opt: a numpy array of optimal weights, obtained trough the efficient frontier module
    
    Returns:
        A dataframe of optimal weights with the associated names of the companies
     """
    combined_data = pd.DataFrame({'Column': get_company_names(returns.columns.to_numpy(), pd.read_csv("ESG_US.csv")), 'Weights': w_opt.round(3)})
    return combined_data

def user_weights(returns,w_opt):
    """
    A user friendly method to display the optimal weights
    
    Parameters:
        returns: dataframe of returns used to get the w_opt parameter
        w_opt: a numpy array of optimal weights, obtained trough the efficient frontier module
        
    Returns:
        A dataframe where the optimal weights for each company is listed. All weights which are 0 are removed for user friendlyness
    """
    combined_data = named_weights(returns,w_opt)
    display(combined_data[(combined_data != 0).all(1)].sum())
    df = combined_data[(combined_data != 0).all(1)]
    return df

