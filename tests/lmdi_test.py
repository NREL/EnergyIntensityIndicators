import pytest
import unittest
import pandas as pd
import os
import glob
import numpy as np

# os.chdir('.')
# print(os.getcwd())
from EnergyIntensityIndicators.LMDI import CalculateLMDI
from EnergyIntensityIndicators.commercial import CommercialIndicators
from EnergyIntensityIndicators.residential import ResidentialIndicators
from EnergyIntensityIndicators.transportation import TransportationIndicators
from EnergyIntensityIndicators.industry import IndustrialIndicators
from EnergyIntensityIndicators.electricity import ElectricityIndicators

class TestLMDI:
    sector_modules = {'residential': ResidentialIndicators,
                      'commercial': CommercialIndicators,
                      'transportation': TransportationIndicators, 
                      'industrial': IndustrialIndicators,
                      'electricity': ElectricityIndicators}

    pnnl_directory = 'C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020'
    output_directory = 'C:/Users/irabidea/Desktop/LMDI_Results'

    def eii_output_factory(self, sector):
        module_ = self.sector_modules[sector](self.pnnl_directory, 
                                             self.output_directory, 
                                             level_of_aggregation=None,  
                                             base_year=1985, end_year=2018)
        return module_

    @staticmethod
    def pnnl_melt(data):
        # module_ = self.sector_modules[sector](self.pnnl_directory, 
        #                                       self.output_directory, 
        #                                       level_of_aggregation=None,  
        #                                       base_year=1985, end_year=2018)
        # categories = module_.sub_categories_list

        # for key, value in categories.items():
        #     if isinstance(value, dict):

        #         for k, v in value.items():
        id_vars = ['Sector', 'Nest level', 'Unit', 'Data Type', 'Energy Type', 'Year']

        value_vars = set(data.columns).difference(set(id_vars))

        data_melt = pd.melt(data, id_vars=id_vars, value_vars=value_vars, var_name='Category', 
                            value_name='Value') #col_level=
        return data_melt
        
    
    def get_pnnl_input(self, sector, dtype):
        # files = glob.glob("C:/Users/irabidea/Desktop/pnnl_csvs/*.csv")
        files = os.listdir(f'C:/Users/irabidea/Desktop/pnnl_csvs/{sector}/{dtype}/')
        files = [f for f in files if f.endswith('.csv')]
        dfs = []
        for f in files: 
            try:    
                df = pd.read_csv(f'C:/Users/irabidea/Desktop/pnnl_csvs/{sector}/{dtype}/{f}')
                df = self.pnnl_melt(df)
                dfs.append(df)
            except Exception as e:
                print(f'{f} failed with error {e} for {dtype}, \n print {df.columns}')
        df = pd.concat(dfs)
        return df

    def get_pnnl_data(self, sector):
        energy_activity = self.get_pnnl_input(sector, 'input_data')
        components = self.get_pnnl_input(sector, 'components')
        results = self.get_pnnl_input(sector, 'results')
        data = {'results': results, 'input_data': energy_activity, 'components': components}
        return data
    
    @pytest.mark.parametrize('sector', ['transportation']) # 'residential', 'commercial', 'industrial', 'electricity', 
    def test_build_nest(self, sector, acceptable_pct_difference=0.05):
        """testing the results of LMDI.build_nest against a csv (to be compiled) of PNNL data.

        - Assertion should be in terms of a % difference from the PNNL data.
        - Test should be parameterized to loop through and test all sectors.

        output of build_nest:
        data_dict = {'energy': energy_data, 'activity': activity_data, 'level_total': level_name}

        results_dict[f'{level_name}'] = data_dict 
        """
        eii = self.eii_output_factory(sector)
        raw_eii_data = eii.collect_data()
        data = eii.collect_energy_data(raw_eii_data)

        results_dict = dict()
        level_of_aggregation = 'All_Freight'

        categories = eii.deep_get(eii.sub_categories_list, level_of_aggregation)
        print('eii.sub_categories_list : \n', eii.sub_categories_list)
        print('categories:\n', categories)

        eii_output = eii.build_nest(data, categories, results_dict, breakout=False, level=1, level1_name=level_of_aggregation, level_name=None)
        
        pnnl_data_raw = self.get_pnnl_data(sector)['input_data']

        new_data = []
        pnnl_data = []
        for results_dict in eii_output:
            print('results dict:\n', results_dict)
            for key, value in results_dict.keys():
                try:
                    eii_activity = value['activity'][[key]]
                    eii_activity = eii_activity.rename(columns={key: 'value'})
                    eii_activity['datatype'] = 'activity'
                    new_data.append(eii_activity)

                    eii_energy = value['energy'][[key]]            
                    eii_activity = eii_energy.rename(columns={key: 'value'})

                    eii_energy['datatype'] = 'energy'
                    new_data.append(eii_energy)

                    pnnl_data_raw[pnnl_data_raw['Category'] == key]
                except Exception as e:
                    print(f'{key} had error {e}')
                    continue

        eii = pd.concat(new_data, axis=0)
        pnnl = pd.concat(pnnl_data, axis=0)
        difference = np.abs(eii.subtract(pnnl)).divide(pnnl)
        
        assert all(difference < acceptable_pct_difference)      

    # @pytest.mark.parametrize('sector', ['transportation']) # 'residential', 'commercial', 'industrial', 'electricity', 
    # def test_prepare_lmdi_inputs(self, sector):
    #     """`LMDI.prepare_lmdi_inputs to test original PNNL data (compiled for #13 ) against 
    #     PNNL results for energy_input_data, energy_shares, and log_ratios

    #     -Test should be parameterized to loop through all sectors.
    #     """        
    #     eii = self.eii_output_factory(sector)

    #     eii_output = eii.prepare_lmdi_inputs()
    #     pnnl_output = []

    #     assert all(eii_output == pnnl_output)

    # @pytest.mark.parametrize('sector', ['transportation']) # 'residential', 'commercial', 'industrial', 'electricity', 
    # def test_multiplicative_lmdi_log_mean_divisia_weights(self, sector):
    #     """Multiplicative test should use original PNNL data (compiled for #13)
    #     Test should be parametrized to loop through all sectors.
    #     """                
    #     eii = self.eii_output_factory(sector)

    #     eii_output = eii.log_mean_divisia_weights()
    #     pnnl_output = []
    #     assert all(eii_output == pnnl_output)

    # @pytest.mark.parametrize('sector', ['transportation']) # 'residential', 'commercial', 'industrial', 'electricity', 
    # def test_additive_lmdi_log_mean_divisia_weights(self, sector):
    #     """Additive test should use "fake" data whose results are know
    #     Test should be parametrized to loop through all sectors.
    #     """        
    #     eii = self.eii_output_factory(sector)

    #     eii_output = eii.log_mean_divisia_weights()
    #     constructed_data = []
    #     assert all(eii_output == constructed_data)

    # @pytest.mark.parametrize('sector', ['transportation']) # 'residential', 'commercial', 'industrial', 'electricity', 
    # def test_calc_asi(self, sector):
    #     """Write test_calc_ASI to test LMDI class.

    #     - Test both additive and multiplicative forms
    #     - Test all sectors
    #     """   
    #     eii = self.eii_output_factory(sector)
     
    #     eii_output = eii.calc_ASI()
    #     pnnl_output = []
    #     assert all(eii_output == pnnl_output)


if __name__ == '__main__':
    test = TestLMDI()
    data = test.get_pnnl_data('transportation')
