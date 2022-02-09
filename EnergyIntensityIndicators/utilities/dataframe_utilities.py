import pandas as pd
import numpy as np
from functools import reduce
from pandas.core.algorithms import isin

from EnergyIntensityIndicators.utilities import loggers

logger = loggers.get_logger()


class DFUtilities:
    """Typically shouldn't be a class, Pytest seems to require
    """

    @staticmethod
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

        log_ratio_df = pd.DataFrame(data=log_ratio, index=dataset.index,
                                    columns=dataset.columns)

        return log_ratio_df

    @staticmethod
    def use_intersection(data, intersection_):
        """Select portion of dataframe where index is in intersection_
        """

        if isinstance(data, pd.Series):
            data_new = data.loc[intersection_]
        else:
            data_new = data.loc[intersection_, :]

        return data_new

    def ensure_same_indices(self, df1, df2):
        """Returns two dataframes with the same indices
        purpose: enable dataframe operations such as multiply and
        divide between the two dfs
        """

        if df1.empty or df2.empty:
            # print('df1:\n', df1)
            # print('df2:\n', df2)
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

            df1_new = self.use_intersection(df1, intersection_)
            df2_new = self.use_intersection(df2, intersection_)

            return df1_new, df2_new

    @staticmethod
    def int_index(df):
        """Ensure df index is Year of type int
        """
        if 'Year' in df.columns:
            df = df.set_index('Year')
        else:
            df.index.name = 'Year'

        df = df[df.index.notna()]
        df.index = df.index.astype(int)
        return df

    @staticmethod
    def create_total_column(df, total_label):
        """Create column from sum of all other columns, name
        with name of level of aggregation
        """
        df2 = df.copy()
        # print('df2:\n', df2)
        # print('type df2', type(df2))
        # print('total_label:', total_label)
        # print('df2 info:', df2.info())
        df2 = df2.fillna(np.nan)
        df_drop_str = \
            df2.apply(
                lambda col:
                    pd.to_numeric(
                        col, errors='ignore'), axis=1).dropna(how='all', axis=1)
        df_drop_str = df2.select_dtypes(exclude='object')

        if isinstance(df2, pd.Series):
            df2 = df2.to_frame()
        cols = df2.columns.tolist()
        #print('cols:', cols)
        if len(cols) > 1:
            df2[total_label] = df2.drop(
                total_label, axis=1, errors='ignore').sum(axis=1,
                                                          numeric_only=True)
        else:
            col = cols[0]
            #print('col:', col)
            df2 = df2.rename(
                columns={col: total_label})

        df2[total_label] = df2[total_label].astype(float)

        # print('df2 final:\n', df2)
        return df2

    @staticmethod
    def select_value(dataframe, base_row, base_column):
        """Select value from dataframe as in Excel's @index function"""
        return dataframe.iloc[base_row, base_column].values()

    def calculate_shares(self, df, total_label=None):
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
        dataset = df.copy()

        if isinstance(dataset.columns, pd.MultiIndex):
            level = dataset.columns.nlevels - 1
            levels = list(range(level))
            sums = dataset.sum(level=levels, axis=1)
            sums.columns = sums.columns.droplevel(levels[:-1])
            shares = dataset.div(sums, level=level-1,
                                 fill_value=np.nan)
        else:
            dataset[total_label] = dataset[total_label].replace(0, np.nan)
            # print('dataset:\n', dataset)
            shares = dataset.drop(total_label, axis=1).divide(
                dataset[total_label].values.reshape(len(dataset[total_label]), 1))
        return shares

    @staticmethod
    def merge_df_list(df_list, keep_cols=False):
        """Complete outer merge on a list of dataframes, merging on left and
        right index of each
        """
        if keep_cols:
            edit_df_list = []
            for df in df_list:
                df.index = df.index.astype(int)
                df = df.reset_index()
                edit_df_list.append(df)
            merged_data = \
                pd.concat(edit_df_list, axis=0, sort=False).groupby('Year').sum(min_count=1)
        else:
            merged_data = reduce(lambda df1, df2: df1.merge(df2, how='outer',
                                                            left_index=True,
                                                            right_index=True),
                                 df_list)
        return merged_data


if __name__ == '__main__':
    pass