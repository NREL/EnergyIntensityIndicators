import pandas as pd
from sklearn import linear_model
import os


from EnergyIntensityIndicators.electricity import ElectricityIndicators
from tests.utilities import TestingUtilities

class TestElectricity:

    pnnl_directory = './tests/Indicators_Spreadsheets_2020'
    output_directory = './Results'
    acceptable_pct_difference = 0.05
    sector = 'electricity'
    utils = TestingUtilities(sector, acceptable_pct_difference)
    # directory = 
    level_of_aggregation = None
    lmdi_model = 'multiplicative'
    base_year = 1985
    end_year = 2018
    comm = ElectricityIndicators(directory, output_directory, 
                                 level_of_aggregation, 
                                 lmdi_model, 
                                 base_year, end_year)

    def 