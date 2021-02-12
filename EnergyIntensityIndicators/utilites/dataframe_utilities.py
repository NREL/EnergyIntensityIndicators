import pandas as pd
import numpy as np


# class DFUtilities:
#     """Typically shouldn't be a class, I'm not sure the optimal strategy when methods 
#     reference eachother
#     """


# @staticmethod
def calculate_log_changes(dataset):
    """Calculate the log changes
        Parameters:
        ----------
            dataset: dataframe

        Returns:
        -------
            log_ratio: dataframe

    """
    change = dataset.divide(dataset.shift().values).astype(float)
    
    log_ratio = change.apply(lambda col: np.log(col), axis=1)

    log_ratio_df = pd.DataFrame(data=log_ratio, index=dataset.index, columns=dataset.columns)

    return log_ratio_df

# @staticmethod
def use_intersection(data, intersection_):
    """Select portion of dataframe where index is in intersection_
    """

    if isinstance(data, pd.Series): 
        data_new = data.loc[intersection_]
    else:
        data_new = data.loc[intersection_, :]
        
    return data_new

def ensure_same_indices(df1, df2):
    """Returns two dataframes with the same indices
    purpose: enable dataframe operations such as multiply and divide between the two dfs
    """        
    if df1.empty or df2.empty:
        raise ValueError('at least one dataframe is empty')

    df1.index = df1.index.astype(int)
    df1.index = df1.index.rename('Year')

    df2.index = df2.index.astype(int)
    df2.index = df2.index.rename('Year')
    
    if df1.index.equals(df2.index):
        return df1, df2
    else:
        intersection_ = df1.index.intersection(df2.index)
        if len(intersection_) == 0: 
            raise ValueError('DataFrames do not contain any shared years')
        
        df1_new = use_intersection(df1, intersection_)
        df2_new = use_intersection(df2, intersection_)

        return df1_new, df2_new

# @staticmethod
def int_index(df):
    """Ensure df index is Year of type int
    """
    if 'Year' in df.columns:
        df = df.set_index('Year')
    else:
        df.index.name = 'Year'

    df.index = df.index.astype(int)
    return df

# @staticmethod
def create_total_column(df, total_label):
    """Create column from sum of all other columns, name with name of 
    level of aggregation
    """
    df_drop_str = df.select_dtypes(exclude='object')
    if len(df_drop_str.columns.tolist()) > 1:
        df[total_label] = df.drop(total_label, axis=1, errors='ignore').sum(axis=1, numeric_only=True)
    elif len(df_drop_str.columns.tolist()) == 1:
        df[total_label] = df[df_drop_str.columns]
    return df 
    
# @staticmethod
def select_value(dataframe, base_row, base_column):
    """Select value from dataframe as in Excel's @index function"""
    return dataframe.iloc[base_row, base_column].values()
    
# @staticmethod
def calculate_shares(dataset, total_label):
    """"sum row, calculate each type of energy as percentage of total
    Parameters
    ----------
    dataset: dataframe
        energy data
    
    Returns
    -------
    shares: dataframe
        contains shares of each energy category relative to total energy 
    """
    dataset[total_label] = dataset[total_label].replace(0, np.nan)
    shares = dataset.drop(total_label, axis=1).divide(dataset[total_label].values.reshape(len(dataset[total_label]), 1))
    return shares