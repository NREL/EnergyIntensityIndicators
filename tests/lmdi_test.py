import pytest
import unittest
import pandas as pd
import os
import glob
import numpy as np

from EnergyIntensityIndicators.LMDI import CalculateLMDI
from EnergyIntensityIndicators.commercial import CommercialIndicators
from EnergyIntensityIndicators.residential import ResidentialIndicators
from EnergyIntensityIndicators.transportation import TransportationIndicators
from EnergyIntensityIndicators.industry import IndustrialIndicators
from EnergyIntensityIndicators.electricity import ElectricityIndicators
from EnergyIntensityIndicators.multiplicative_lmdi import MultiplicativeLMDI
from EnergyIntensityIndicators.additive_lmdi import AdditiveLMDI

class TestLMDI:
    sector_modules = {'residential': ResidentialIndicators,
                      'commercial': CommercialIndicators,
                      'transportation': TransportationIndicators, 
                      'industrial': IndustrialIndicators,
                      'electricity': ElectricityIndicators}

    pnnl_directory = 'C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020'
    output_directory = 'C:/Users/irabidea/Desktop/LMDI_Results'

    def eii_output_factory(self, sector):
        """Method to call the sector module
        """        
        module_ = self.sector_modules[sector](self.pnnl_directory, 
                                             self.output_directory, 
                                             level_of_aggregation=None,  
                                             base_year=1985, end_year=2018)
        return module_

    @staticmethod
    def pnnl_melt(data):
        """Method to format PNNL data to match EII format
        """        

        id_vars = ['Sector', 'Nest level', 'Unit', 'Data Type', 'Energy Type', 'Year']

        value_vars = set(data.columns).difference(set(id_vars))

        data_melt = pd.melt(data, id_vars=id_vars, value_vars=value_vars).rename(columns={'variable': 'Category', 'value': 'Value'}) #col_level= # , var_name='Category', 
                           # value_name='Value'

        return data_melt
        
    def get_pnnl_input(self, sector, dtype):
        """Method to read in all PNNL csvs for dtype

        Args:
            dtype (str): which data category (i.e. input_data, components, results)
                         to call

        Returns:
            df : all PNNL data for dtype
        """        
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
        df = pd.concat(dfs, axis=0, ignore_index=True)
        return df

    def get_pnnl_data(self, sector):
        """[summary]

        Args:
            sector ([type]): [description]

        Returns:
            [type]: [description]
        """        
        energy_activity = self.get_pnnl_input(sector, 'input_data')

        nested_data = dict()
        for level_ in energy_activity['Nest level'].unique():
            to_nest = energy_activity[energy_activity['Nest level'] == level_]
            nested_ = self.nest_(to_nest)
            nested_data[level_] = nested_

        components = self.get_pnnl_input(sector, 'components')

        results = self.get_pnnl_input(sector, 'results')

        data = {'results': results, 'input_data': energy_activity, 'components': components}
        return data

    @staticmethod
    def nest_(input_data):
        
        input_data['Year'] = input_data['Year'].astype('int')

        activity = input_data[input_data['Data Type'] == 'Activity']
        activity_data = dict()
        for activity_type in activity['Unit']:
            a_df =  activity[activity['Unit'] == activity_type]
            a_df = a_df.pivot(index='Year', columns='Category', values='Value')
            activity_data[activity_type] = a_df

        energy = input_data[input_data['Data Type'] == 'Energy']
        energy_data = dict()
        for energy_type in input_data['Energy Type']:
            e_df = energy[energy['Energy Type'] == energy_type]
            e_df = e_df.pivot(index='Year', columns='Category', values='Value')
            energy_data[energy_type] = e_df

        try:
            weather = input_data[input_data['Data Type'] == 'weather']
            weather = weather.pivot(index='Year', columns='Category', values='Value')

        except Exception as e: 
            print(f'Exception: {e}')
            weather = None

        return {'activity': activity_data, 'energy': energy_data, 'weather': weather}

    def input_data(self, sector):
        eii = self.eii_output_factory(sector)
        raw_eii_data = eii.collect_data()
        print('raw_eii_data', raw_eii_data)
        level_of_aggregation_ = 'All_Freight.Pipeline'
        eii, final_results = eii.get_nested_lmdi(level_of_aggregation_, 
                                                 raw_eii_data, 
                                                 calculate_lmdi=False,
                                                 breakout=False,
                                                 save_breakout=False)
        # eii = pd.concat(new_data, axis=0, ignore_index=True)
        print('eii: \n', eii)

        pnnl_data_raw = self.get_pnnl_data(sector)['input_data']
        
        level_of_aggregation = level_of_aggregation_.split(".")
        level1_name = level_of_aggregation[-1]
        print('level1_name', level1_name)
        print("pnnl_data_raw['Nest level']:", pnnl_data_raw['Nest level'].unique())
        pnnl_data_raw = pnnl_data_raw.replace({'Pipelines': 'Pipeline', 
                                               'Freight Total': 'All_Freight', 
                                               'Deliv': 'deliv'})
        pnnl_data_ = dict()
        print('eii keys:', eii.keys())
        for energy_type in pnnl_data_raw['Energy Type'].unique():
            energy_type_dict = dict()
            for col in  eii[energy_type]['energy'].columns:
                try:
                    pnnl = pnnl_data_raw[pnnl_data_raw['Nest level'] == col]
                except KeyError:
                    print('error there')
                    pass
                for d_type in pnnl['Data Type'].unique():
                    try:
                        pnnl_sub = pnnl[pnnl['Data Type'] == d_type].pivot(index='Year', columns='Category', values='Value')
                    except KeyError:
                        print('error here')
                        pass
                    print('pnnl_sub:\n', pnnl_sub)
                    energy_type_dict[d_type] = pnnl_sub


            pnnl_data_[energy_type] = energy_type_dict
        

        print('pnnl:', pnnl_data_)

        return eii, pnnl_data_
    
    @pytest.mark.parametrize('sector', ['transportation']) # 'residential', 'commercial', 'industrial', 'electricity', 
    def test_build_nest(self, sector, acceptable_pct_difference=0.05):
        """testing the results of LMDI.build_nest against a csv (to be compiled) of PNNL data.

        - Assertion should be in terms of a % difference from the PNNL data.
        - Test should be parameterized to loop through and test all sectors.

        output of build_nest:
        data_dict = {'energy': energy_data, 'activity': activity_data, 'level_total': level_name}

        results_dict[f'{level_name}'] = data_dict 
        """
        eii, pnnl = self.input_data(sector)
        for energy_type, energy_dict in pnnl.items():
            for data_type, data_dict in energy_dict.items():
                for cat, pnnl_df in data_dict.items():
                    eii_df = eii[energy_type][data_type][cat]
                    difference = np.abs(eii_df.subtract(pnnl_df)).divide(pnnl_df)
                    
                    assert all(difference < acceptable_pct_difference)      
                
    # @pytest.mark.parametrize('sector', ['transportation']) # 'residential', 'commercial', 'industrial', 'electricity', 
    # def test_prepare_lmdi_inputs(self, sector):
    #     """`LMDI.prepare_lmdi_inputs to test original PNNL data (compiled for #13 ) against 
    #     PNNL results for energy_input_data, energy_shares, and log_ratios

    #     -Test should be parameterized to loop through all sectors.
    #     """        
    #     eii = self.eii_output_factory(sector)
    #     eii = self. (sector)

    #     for e_, e_df in eii_energy.items():
    #         eii_output = eii.prepare_lmdi_inputs(energy_input_data=e_df, activity_input_data=eii_activity, 
    #                                              total_label=) 
    #     pnnl_output = []

    #     assert all(eii_output == pnnl_output)

    # @pytest.mark.parametrize('sector', ['transportation']) # 'residential', 'commercial', 'industrial', 'electricity', 
    # def test_multiplicative_lmdi_log_mean_divisia_weights(self, sector):
    #     """Multiplicative test should use original PNNL data (compiled for #13)
    #     Test should be parametrized to loop through all sectors.
    #     """                
    #     eii = self.eii_output_factory(sector)
    #     model_ = MultiplicativeLMDI(energy_data, energy_shares, 1985, 2017, total_label, lmdi_type)
    #     eii_output = model_.log_mean_divisia_weights()
    #     pnnl_data_raw = self.get_pnnl_data(sector)['results']

    #     weights_cols = [cat for cat in pnnl_data_raw['Category'].unique() if cat.endswith('eights')]
    #     pnnl_weights = pnnl_data_raw[pnnl_data_raw['Category'].isin(weights_cols)]
    #     assert all(eii_output == pnnl_weights)

    # @pytest.mark.parametrize('sector', ['transportation']) # 'residential', 'commercial', 'industrial', 'electricity', 
    # def test_additive_lmdi_log_mean_divisia_weights(self, sector):
    #     """Additive test should use "fake" data whose results are know
    #     Test should be parametrized to loop through all sectors.
    #     """        
    #     model_ = AdditiveLMDI(energy_data, energy_shares, 1985, 2017, total_label, lmdi_type)
    #     eii_output = model_.log_mean_divisia_weights()

    #     constructed_data = []
    #     assert all(eii_output == constructed_data)

    # @pytest.mark.parametrize('sector', ['transportation']) # 'residential', 'commercial', 'industrial', 'electricity', 
    # def test_calc_asi(self, sector):
    #     """Write test_calc_ASI to test LMDI class.

    #     - Test both additive and multiplicative forms
    #     - Test all sectors
    #     """   
    #     eii = self.eii_output_factory(sector)
     
    #     eii_output = eii.calc_ASI('multiplicative', weather_data, weights, log_ratios)
    #     pnnl_data_raw = self.get_pnnl_data(sector)['components']
    #     assert all(eii_output == pnnl_output)


if __name__ == '__main__':
    test = TestLMDI()
    data = test.get_pnnl_data('transportation')
    print(data)
