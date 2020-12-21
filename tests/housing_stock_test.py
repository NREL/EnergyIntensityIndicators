import pytest
import unittest
import pandas as pd

from EnergyIntensityIndicators.Residential.residential_floorspace import ResidentialFloorspace


class TestClass:
    directory = 'C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020'
    
    pytest.mark.parametrize('housing_type', ['single_family', 'multifamily', 'manufatured_housing'])
    def test_housing_stock_inputs(self, housing_type):
        """Test that the input data match PNNL's input data for the housing stock model 
        """        
        pass

    pytest.mark.parametrize('housing_type', ['single_family', 'multifamily', 'manufatured_housing'])
    def test_housing_stock(self, housing_type):
        data = ResidentialFloorspace()
        if housing_type == 'single_family':

            housing_stock = data.get_housing_stock_sf()
            sheetname_ = 'Total_stock_SF'
            cols = 'AX'
        elif housing_type == 'multifamily':
            all_stock_mf = []
            pub_total_mf = []
            housing_stock == data.get_housing_stock_mf(all_stock_mf, pub_total_mf)
            sheetname_ = 'Total_stock_MF'
            cols = 'AX'
        else: 
            all_stock_mh = []
            pub_total_mh = []
            housing_stock == data.get_housing_stock_mh(all_stock_mh, pub_total_mh)
            sheetname_ = 'Total_stock_MH'
            cols = 'AX'
        
        pnnl_occupied = pd.read_excel(f'{self.directory}/AHS_summary_results_051720.xlsx', sheet_name=sheetname_, usecols=cols, header=13, skiprows=12)

        assert all(housing_stock == pnnl_occupied)

    pytest.mark.parametrize('housing_type', ['single_family', 'multifamily', 'manufatured_housing'])
    def test_housing_size_model(self, housing_type):
        data = ResidentialFloorspace()
        if housing_type == 'single_family':
            housing_stock = data.get_housing_size_sf()
            sheetname_ = 'Total_stock_SF'
            cols = 'BX'
        elif housing_type == 'multifamily':
            housing_stock == data.get_housing_size_mf()
            sheetname_ = 'Total_stock_MF'
            cols = 'CT'
        else: 
            housing_stock == data.model_average_housing_unit_size_mh()
            sheetname_ = 'Total_stock_MH'
            cols = 'CD'
        
        pnnl_housing_size = pd.read_excel(f'{self.directory}/AHS_summary_results_051720.xlsx', sheet_name=sheetname_, usecols=cols, header=13, skiprows=12)

        assert all(housing_size == pnnl_housing_size)


    pytest.mark.parametrize('housing_type', ['single_family', 'multifamily', 'manufatured_housing'])
    def test_avg_housing_size_inputs(self, housing_type):
            """Test that the input data match PNNL's input data for the average housing size model
            """
            pass
