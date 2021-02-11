import pandas as pd
from sklearn import linear_model
import os


from EnergyIntensityIndicators.residential import ResidentialIndicators
from tests.utilities import TestingUtilities

class TestResidential:

    pnnl_directory = './tests/Indicators_Spreadsheets_2020'
    output_directory = './Results'
    acceptable_pct_difference = 0.05
    sector = 'residential'
    utils = TestingUtilities(sector, acceptable_pct_difference)
    directory = 
    output_directory = 
    level_of_aggregation = None
    lmdi_model = 'multiplicative'
    base_year = 1985
    end_year = 2018
    res = ResidentialIndicators(directory, output_directory, 
                                level_of_aggregation, 
                                lmdi_model, 
                                base_year, end_year)

    def test_get_seds(self):
        pnnl_fuels = pd.read_csv('./')
        pnnl_elec = pd.read_csv('./')
        eii_fuels, eii_elec = res.get_seds()
        
        df_pairs_list = [(eii_fuels, pnnl_fuels), (eii_elec, pnnl_elec)]
        pct_diff_bools = self.utils.pct_diff_bools_list(df_pairs_list)
        
        assert all(pct_diff_bool)

    def test_fuel_electricity_consumption(self):
        """This method selects region column from df, not sure it
        needs to be separately tested
        """
        pass

    def test_get_floorspace(self):
        pnnl_occupied_housing_units = pd.read_csv('./')
        pnnl_floorspace_square_feet = pd.read_csv('./')
        pnnl_household_size_square_feet_per_hu = pd.read_csv('./')
        
        eii_data = res.get_floorspace()
        eii_ohu = eii_data['occupied_housing_units']
        eii_fsf = eii_data['floorspace_square_feet']
        eii_hssfphu = eii_data['household_size_square_feet_per_hu']
        
        df_pairs_list = [(eii_ohu, pnnl_occupied_housing_units), 
                         (eii_fsf, pnnl_floorspace_square_feet), 
                         (eii_hssfphu, household_size_square_feet_per_hu)]

        pct_diff_bools = self.utils.pct_diff_bools_list(df_pairs_list)
        
        assert all(pct_diff_bool)

    def test_activity(self):
        pnnl_activity = pd.read_csv('./')

        test_floorspace = 
        eii_activity = res.activity(test_floorspace)

        pct_difference_bool = TestLMDI().pct_diff(pnnl_activity, eii_activity, 
                                             self.acceptable_pct_difference, 
                                             self.sector)
        assert pct_difference_bool
    
    def test_collect_weather(self):
        pnnl_weather = pd.read_csv('./')
        
        test_energy_dict = 
        test_nominal_energy_intensity = 
        eii_weather = res.collect_weather()

        pct_difference_bool = TestLMDI().pct_diff(pnnl_weather, eii_weather, 
                                             self.acceptable_pct_difference, 
                                             self.sector)
        assert pct_difference_bool