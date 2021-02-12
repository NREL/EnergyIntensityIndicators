import pytest
import unittest
import pandas as pd
import os
import glob
import numpy as np

from EnergyIntensityIndicators.utilities import dataframe_utilities as df_utils
from EnergyIntensityIndicators.tests.utilities import TestingUtilities

class TestDFUtilities:

    @staticmethod
    def test_calculate_log_changes(acceptable_pct_difference=0.05):
        """Test for the dataframe_utilities calculate_log_changes method

        Args:
            acceptable_pct_difference (float, optional): [description]. Defaults to 0.05.
        """
        
        input_data = [[1.2759, 0.9869],
                      [1.2650, 0.9743],
                      [1.2579, 0.9910],
                      [1.2634, 0.9915],
                      [1.2396, 0.9906]]


        input_df = pd.DataFrame(input_data, 
                                     index=[1970, 1971, 1972, 1973, 1974], 
                                     columns=['All_Passenger', 'All_Freight'])

        log_ratio_df = df_utils.calculate_log_changes(input_df)
        log_ratio_df = log_ratio_df.round(4)
        comparison_output = [[np.nan, np.nan],
                             [-0.0086, -0.0129],
                             [-0.0056, 0.0170],
                             [0.0044, 0.0005],
                             [-0.0190, -0.0009]]

        comparison_df = pd.DataFrame(comparison_output, 
                                     index=[1970, 1971, 1972, 1973, 1974], 
                                     columns=['All_Passenger', 'All_Freight'])
        print('comparison_df:\n', comparison_df)
        print('log_ratio_df:\n', log_ratio_df)
        assert TestingUtilities(acceptable_pct_difference).pct_diff(comparison_df, log_ratio_df)

    def test_calculate_shares():
        pass