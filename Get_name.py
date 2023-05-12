import pandas as pd
import numpy as np

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
