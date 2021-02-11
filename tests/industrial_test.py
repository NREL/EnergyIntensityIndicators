import pandas as pd
from sklearn import linear_model
import os

from EnergyIntensityIndicators.industry import IndustrialIndicators
from EnergyIntensityIndicators.Industry.manufacturing import Manufacturing
from EnergyIntensityIndicators.Industry.nonmanufacuturing import NonManufacturing
from tests.lmdi_test import TestLMDI
from tests.utilities import TestingUtilities

class TestIndustry:

    pnnl_directory = './tests/Indicators_Spreadsheets_2020'
    output_directory = './Results'
    acceptable_pct_difference = 0.05
    sector = 'industrial'
    utils = TestingUtilities(self.sector, self.acceptable_pct_difference)  

    

class TestNonManufacturing:

    pnnl_directory = './tests/Indicators_Spreadsheets_2020'
    output_directory = './Results'
    acceptable_pct_difference = 0.05
    sector = 'industrial'
    utils = TestingUtilities(sector, acceptable_pct_difference)  


    def test_indicators_nonman_2018_bea(self):
        pnnl_va = pd.read_csv('./')
        pnnl_go = pd.read_csv('./')
        eii_va, eii_go = NonManufacturing().indicators_nonman_2018_bea()
        
        df_pairs_list = [(eii_va, pnnl_va), (eii_go, pnnl_go)]
        bools_list = self.utils.pct_diff_bools_list(df_pairs_list)
        assert all(bools_list)


    @staticmethod
    def test_get_econ_census():
        """Not currently used (this could be a problem)
        """
        ...
    
    @staticmethod
    def test_petroleum_prices():
        """Not currently used (this could be a problem)
        """
        ...

    def test_construction_raw_data(self):
        ...
        pnnl_elec = pd.read_csv('./')
        pnnl_fuels = pd.read_csv('./')
        eii_elec, eii_fuels = NonManufacturing().construction_raw_data()

        df_pairs_list = [(eii_elec, pnnl_elec), (eii_fuels, pnnl_fuels)]
        bools_list = self.utils.pct_diff_bools_list(df_pairs_list)
        assert all(bools_list)

    
    def test_construction(self):
        ...
        pnnl_elec = pd.read_csv('./')
        pnnl_fuels = pd.read_csv('./')
        pnnl_go = pd.read_csv('./')
        pnnl_va = pd.read_csv('./')

        eii_data_dict = NonManufacturing().construction()
        eii_elec = eii_data_dict['energy']['elec']
        eii_fuels = eii_data_dict['energy']['fuels'] 
        eii_go = eii_data_dict['activity']['gross_output'] 
        eii_va = eii_data_dict['activity']['value_added'] 
        
        df_pairs_list = [(eii_elec, pnnl_elec), (eii_fuels, pnnl_fuels),
                         (eii_go, pnnl_go), (eii_va, pnnl_va)]

        bools_list = self.utils.pct_diff_bools_list(df_pairs_list)

        assert all(bools_list)

    def test_agriculture(self):
        ...
        pnnl_elec = pd.read_csv('./')
        pnnl_fuels = pd.read_csv('./')
        pnnl_go = pd.read_csv('./')
        pnnl_va = pd.read_csv('./')

        eii_data_dict = NonManufacturing().agriculture()
        eii_elec = eii_data_dict['energy']['elec']
        eii_fuels = eii_data_dict['energy']['fuels'] 
        eii_go = eii_data_dict['activity']['gross_output'] 
        eii_va = eii_data_dict['activity']['value_added'] 
        
        df_pairs_list = [(eii_elec, pnnl_elec), (eii_fuels, pnnl_fuels),
                         (eii_go, pnnl_go), (eii_va, pnnl_va)]

        bools_list = self.utils.pct_diff_bools_list(df_pairs_list)
        
        assert all(bools_list)
    
    def test_aggregate_mining_data():
        ...
        pnnl_data = pd.read_csv('./')

        test_mining_data = pd.read_csv('./')
        eii_data = NonManufacturing().aggregate_mining_data(test_mining_data, 
                                                            allfos=False)

        pct_difference_bool = TestLMDI().pct_diff(pnnl_data, eii_data, 
                                                  self.acceptable_pct_difference, 
                                                  self.sector)
        assert pct_difference_bool


    def test_build_mining_output():
        ...
    
    def test_crude_petroleum_natgas():
        ...
    
    def test_coal_mining():
        ...

    def test_metal_mining():
        ...

    def test_nonmetallic_mineral_mining():
        ...

    def test_other_mining():
        ...

    def test_drilling_and_mining_support():
        ...

    def test_mining_fuels_adjust():
        ...

    def test_price_ratios():
        ...
    
    def test_calculate_physical_units():
        ...

    def test_aggregate_sector_estimates():
        ...

    def test_mining_data_1987_2017():
        ...

    def test_mining_sector_estimates():
        ...

    def test_mining():
        ...
    
    def test_nonmanufacturing_data():
        ...


class TestManufacturing:

    pnnl_directory = './tests/Indicators_Spreadsheets_2020'
    output_directory = './Results'
    acceptable_pct_difference = 0.05
    sector = 'industrial'
    utils = TestingUtilities(sector, acceptable_pct_difference)  

    def test_mecs_fuel():
        ...

    def test_get_historical_mecs():
        ...

    def test_manufacturing_prices():
        ...

    def test_import_mecs_fuel():
        ...

    def test_call_census_data():
        ...

    def test_call_activity_data():
        ...

    def test_import_mecs_electricity():
        ...

    def test_mecs_annual_fuel():
        ...

    def test_calc_quantity_shares():
        ...

    def test_quantity_shares_1985_1998():
        ...

    def test_expenditure_ratios_revised():
        ...

    def test_expend_ratios_revised_85_97():
        ...

    def test_aggregate_naics():
        ...

    def test_quantity_shares_1998_forward():
        ...

    def test_mecs_data_sic():
        ...

    def test_interpolate_mecs():
        ...

    def test_pre_1998_quantities():
        ...

    def test_quantities_1998_forward():
        ...

    def test_final_quantities_asm_85():
        ...

    def test_get_manufacturing_fuels():
        ...

    def test_manufacturing_energy():
        ...
    
    def test_manufacturing():
        ...

    