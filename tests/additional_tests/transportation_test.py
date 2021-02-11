import pandas as pd
from sklearn import linear_model
import os


from EnergyIntensityIndicators.transportation import TransportationIndicators
from tests.utilities import TestingUtilities

class TestTransportation:

    pnnl_directory = './tests/Indicators_Spreadsheets_2020'
    output_directory = './Results'
    acceptable_pct_difference = 0.05
    sector = 'transportation'
    utils = TestingUtilities(sector, acceptable_pct_difference)
    # directory = 
    level_of_aggregation = None
    lmdi_model = 'multiplicative'
    base_year = 1985
    end_year = 2018
    comm = TransportationIndicators(directory, output_directory, 
                                    level_of_aggregation, 
                                    lmdi_model, 
                                    base_year, end_year)

    def 