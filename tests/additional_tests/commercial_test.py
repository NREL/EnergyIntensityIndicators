import pandas as pd
from sklearn import linear_model
import os


from EnergyIntensityIndicators.commercial import CommercialIndicators
from tests.utilities import TestingUtilities

class TestCommercial:

    pnnl_directory = './tests/Indicators_Spreadsheets_2020'
    output_directory = './Results'
    acceptable_pct_difference = 0.05
    sector = 'commercial'
    utils = TestingUtilities(sector, acceptable_pct_difference)
    # directory = 
    # output_directory = 
    level_of_aggregation = None
    lmdi_model = 'multiplicative'
    base_year = 1985
    end_year = 2018
    comm = CommercialIndicators(directory, output_directory, 
                               level_of_aggregation, 
                               lmdi_model, 
                               base_year, end_year)

    def test_collect_input_data():

    
    def test_adjusted_supplier_data(self):
        
        pnnl_data = pd.read_csv('./')
        eii_data = comm.adjusted_supplier_data()

        pct_diff_bool = TestLMDI().pct_diff(pnnl, eii, 
                                            self.acceptable_pct_difference, 
                                            self.sector)
        assert pct_diff_bool

    def test_get_saus():

    def test_dod_compare_old():

    def test_dodge_adjustment_ratios():
        # test_dodge_dataframe = 
        # start_year = 
        # stop_year = 
        # adjust_years = 
        # late = 

    def test_west_inflation():

    def test_hist_stat():

    def test_hist_stat_adj():

    def test_dodge_revised():

    def test_dodge_to_cbecs():

    def test_nems_logistic():

    def test_solve_logistic():

    def test_activity():

    def test_fuel_electricity_consumption():

    def test_get_seds():

    def test_collect_weather():

