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
        data_melt = data_melt.replace('#DIV/0!', np.nan)
        data_melt['Value'] = data_melt['Value'].astype(float)
        data_melt['Year'] = data_melt['Year'].astype('int')

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
                df = df.dropna(axis=1, how='all')
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

    def input_data(self, sector, level_of_aggregation_='All_Freight.Pipeline'):
        eii = self.eii_output_factory(sector)
        raw_eii_data = eii.collect_data()
        eii, final_results = eii.get_nested_lmdi(level_of_aggregation_, 
                                                 raw_eii_data, 
                                                 calculate_lmdi=False,
                                                 breakout=False,
                                                 save_breakout=False)
        # eii = pd.concat(new_data, axis=0, ignore_index=True)

        pnnl_data_raw = self.get_pnnl_data(sector)['input_data']
        
        level_of_aggregation = level_of_aggregation_.split(".")
        level1_name = level_of_aggregation[-1]
        print('level1_name', level1_name)
        print("pnnl_data_raw['Nest level']:", pnnl_data_raw['Nest level'].unique())
        pnnl_data_raw = pnnl_data_raw.replace({'Pipelines': 'Pipeline', 
                                               'Freight Total': 'All_Freight', 
                                               'Deliv': 'deliv'})
        pnnl_data_ = dict()
        for energy_type in pnnl_data_raw['Energy Type'].unique():
            energy_type_dict = dict()
            if isinstance(energy_type, float):
                continue
            elif energy_type.lower() in eii.keys():
                for col in  eii[energy_type.lower()]['energy'].columns:
                    try:
                        pnnl = pnnl_data_raw[pnnl_data_raw['Nest level'] == col]
                    except KeyError:
                        print('error there')
                        continue
                    for d_type in pnnl['Data Type'].unique():
                        try:
                            pnnl_sub = pnnl[pnnl['Data Type'] == d_type].pivot(index='Year', columns='Category', values='Value')
                            pnnl_sub.index = pnnl_sub.index.astype(int)
                            pnnl_sub = pnnl_sub.rename_axis(col, axis='columns')
                            pnnl_sub = pnnl_sub.fillna(np.nan)
                        except KeyError:
                            print('error here')
                            continue
                        energy_type_dict[d_type.lower()] = pnnl_sub
            else:
                continue

            pnnl_data_[energy_type.lower()] = energy_type_dict
        
        return eii, pnnl_data_
    
    # @pytest.mark.parametrize('sector', ['transportation']) # , 'residential', 'commercial', 'industrial', 'electricity'
    # def test_build_nest(self, sector, acceptable_pct_difference=0.05):
    #     """testing the results of LMDI.build_nest against a csv (to be compiled) of PNNL data.

    #     - Assertion should be in terms of a % difference from the PNNL data.
    #     - Test should be parameterized to loop through and test all sectors.

    #     output of build_nest:
    #     data_dict = {'energy': energy_data, 'activity': activity_data, 'level_total': level_name}

    #     results_dict[f'{level_name}'] = data_dict 
    #     """
    #     eii, pnnl = self.input_data(sector)
    #     for energy_type, energy_dict in pnnl.items():
    #         for data_type, data_dict in energy_dict.items():
    #             for cat, pnnl_df in data_dict.items():
    #                 pnnl_df.index = pnnl_df.index.astype(int)
    #                 eii_df = eii[energy_type][data_type][[cat]]

    #                 acceptable_bool = self.pct_diff(pnnl_df, eii_df, acceptable_pct_difference, sector)

    #                 assert all(acceptable_bool)      
                
    # @pytest.mark.parametrize('sector', ['transportation']) # 'residential', 'commercial', 'industrial', 'electricity', 
    # def test_prepare_lmdi_inputs(self, sector, acceptable_pct_diff=0.5):
    #     """`LMDI.prepare_lmdi_inputs to test original PNNL data (compiled for #13 ) against 
    #     PNNL results for energy_input_data, energy_shares, and log_ratios

    #     -Test should be parameterized to loop through all sectors.

    #     prepare_lmdi_inputs returns the following:
    #         log_ratios = {'activity': log_ratio_activity, 
    #                       'structure': log_ratio_structure, 
    #                       'intensity': log_ratio_intensity}
    #     """        
    #     eii = self.eii_output_factory(sector)
    #     pnnl_output = self.get_pnnl_input(sector, 'intermediate')

    #     print('pnnl_output:\n', pnnl_output)

    #     eii_data, pnnl = self.input_data(sector, level_of_aggregation_='All_Freight.Pipeline')
    #     print('pnnl:', pnnl)
    #     for e_, e_dict in pnnl.items():
    #         activity_data = e_dict['activity']

    #         print('activity_data.columns:', activity_data.columns)
    #         energy_data = e_dict['energy']
    #         print('energy_data.columns:', energy_data.columns)
    #         try:
    #             total_label = activity_data.columns.name
    #         except ValueError:
    #             print(f'Error: activity data of type {type(activity_data)} \
    #                     with columns: {activity_data.columns}')
    #             continue
            
    #         activity_dict = dict()
    #         if isinstance(activity_data, pd.DataFrame):
    #             activity_dict[total_label] = activity_data

    #         print('activity_data:', activity_data)
    #         print('energy_data:', energy_data)
    #         if total_label in activity_data.columns and total_label in energy_data.columns:
    #             energy_data, energy_shares, eii_log_ratios = eii.prepare_lmdi_inputs(energy_input_data=energy_data, 
    #                                                                                  activity_input_data=activity_dict, 
    #                                                                                  total_label=total_label)
    #             print('eii_output:\n', eii_log_ratios) 
    #             print('pnnl_output_ pre manipulation:\n', pnnl_output)
    #             pnnl_output = pnnl_output.replace({'Pipelines': 'Pipeline', 
    #                                                  'Freight Total': 'All_Freight', 
    #                                                  'Deliv': 'deliv'})
    #             print("total_label in pnnl_output['Nest level']", total_label in pnnl_output['Nest level'])
    #             print("e_ in pnnl_output['Energy Type']", e_ in pnnl_output['Nest level'])

    #             print("pnnl_output['Energy Type'].unique()]", pnnl_output['Energy Type'].unique())
    #             print("pnnl_output['Nest level'].unique()]", pnnl_output['Nest level'].unique())
    #             print("pnnl_output['Category'].unique()", pnnl_output['Category'].unique())

    #             pnnl_output_ = pnnl_output[(pnnl_output['Energy Type'] == e_) & (pnnl_output['Nest level'] == total_label)]
    #             print('pnnl_output_ here:\n', pnnl_output_)

    #             pnnl_component_data = dict()
    #             for d_type in pnnl_output_['Data Type'].unique():
    #                 pnnl_df = pnnl_output_[pnnl_output_['Data Type'] == d_type][['Year', 'Category', 'Value']]
    #                 pnnl_df = pnnl_df.pivot(index='Year', columns='Category', values='Value').dropna(axis=1, how='all')
    #                 pnnl_df.columns.name = total_label
    #                 print('dtype:', d_type)
    #                 print('pnnl_df:\n', pnnl_df)
    #                 if d_type == 'Log Changes Intensity':
    #                     pnnl_component_data['intensity'] = pnnl_df
    #                 elif d_type == 'Log Changes Activity':
    #                     pnnl_component_data['activity'] = pnnl_df
    #                 elif d_type == 'Log Changes Structure':
    #                     pnnl_component_data['structure'] = pnnl_df
    #             print('pnnl_component_data:\n', pnnl_component_data)
    #             eii_test_data = {k: eii_log_ratios[k] for k in pnnl_component_data.keys()}
    #             print('eii_test_data:\n', eii_test_data)
    #             if eii_test_data != eii_log_ratios:
    #                 print('PNNL missed components')
    #             bools_list = [self.pct_diff(pnnl_component_data[k], eii_test_data[k], acceptable_pct_diff, sector) for k in pnnl_component_data.keys()]
    #             assert all(bools_list)
    #         else:
    #             print(f'Missing {total_label}, with {activity_data.columns} activity columns and \
    #                     {energy_data.columns} energy columns')
    #             continue

    def test_log_mean_divisia(self, sector='transportation'):
        eii = self.eii_output_factory(sector)
        x = 0.5913
        y = 0.5650
        L = eii.logarithmic_average(x, y)
        pnnl_result = 0.578
        assert round(L, 3) == pnnl_result
    
    def test_calculate_log_changes(self, sector='transportation', acceptable_pct_difference=0.05):
        eii = self.eii_output_factory(sector)
        
        input_data = [[1.2759, 0.9869],
                      [1.2650, 0.9743],
                      [1.2579, 0.9910],
                      [1.2634, 0.9915],
                      [1.2396, 0.9906]]


        input_df = pd.DataFrame(input_data, 
                                     index=[1970, 1971, 1972, 1973, 1974], 
                                     columns=['All_Passenger', 'All_Freight'])

        log_ratio_df = eii.calculate_log_changes(input_df)
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
        # assert log_ratio_df.equals(comparison_df)
        assert self.pct_diff(comparison_df, log_ratio_df, acceptable_pct_difference, sector='transportation')

    def calc_component(self, sector):
        eii = self.eii_output_factory(sector)

        log_ratio_component = [[np.nan, np.nan],
                               [-0.0086, -0.0129],
                               [-0.0056, 0.0170],
                               [0.0044, 0.0005],
                               [-0.0190, -0.0009]]
        log_ratio_component = pd.DataFrame(log_ratio_component, 
                                           index=[1970, 1971, 1972, 1973, 1974], 
                                           columns=['All_Passenger', 'All_Freight'])
        
        weights = [[0.3911, 0.6089],
                   [0.7602, 0.2398],
                   [0.7610, 0.2390],
                   [0.7596, 0.2404],
                   [0.7563, 0.2437]]

        weights = pd.DataFrame(weights, 
                               index=[1970, 1971, 1972, 1973, 1974], 
                               columns=['All_Passenger', 'All_Freight'])

        component = eii.calc_component(log_ratio_component, weights)
        component = component.apply(lambda col: np.exp(col), axis=1)

        comparison_output = [[np.nan], 
                             [0.9904],
                             [0.9998],
                             [1.0034],
                             [0.9855]]
        comparison_output = pd.DataFrame(comparison_output, 
                                         index=[1970, 1971, 1972, 1973, 1974], 
                                         columns=['Intensity Index'])
        print('component:\n', component)
        print('comparison_output:\n', comparison_output)
        assert component.equals(comparison_output)

    def test_compute_index(self):
        eii = MultiplicativeLMDI()
        
        results = [[0.9705, 1.0386, 1.0037], 
                   [0.9957, 1.0329, 1.0054],
                   [0.9982, 1.0145, 1.0052],
                   [1.0076, 1.0165, 1.0066],
                   [0.9814, 1.0412, 1.0016]]

        results = pd.DataFrame(results, 
                               index=[1983, 1984, 1985, 1986, 1987], 
                               columns=['Intensity Index', 'Activity Index', 'Structure Index'])
        
        for col in results.columns:
            results[col] = eii.compute_index(results[col], 1985)
            results[col] = results[col].astype(float).round(4)

        comparison_output = [[1.0062, 0.9543, 0.9895],
                             [1.0018, 0.9857, 0.9948],
                             [1.0000, 1.0000, 1.0000],
                             [1.0076, 1.0165, 1.0066],
                             [0.9889, 1.0584, 1.0082]]
        
        comparison_output = pd.DataFrame(comparison_output, 
                                         index=[1983, 1984, 1985, 1986, 1987], 
                                         columns=['Intensity Index', 'Activity Index', 'Structure Index'])
        print('results_:\n', results)
        print('comparison_output:\n', comparison_output)
        # assert results.equals(comparison_output)
        assert self.pct_diff(comparison_output, results, acceptable_pct_difference=0.05, sector='transportation')
    
    # def test_multiplicative_decomposition(self, sector='transportation'):
    #     eii = MultiplicativeLMDI()

    #     test_asi = [[1.0062, 0.9543, 0.9895],
    #                 [1.0018, 0.9857, 0.9948],
    #                 [1.0000, 1.0000, 1.0000],
    #                 [1.0076, 1.0165, 1.0066],
    #                 [0.9889, 1.0584, 1.0082]]
        
    #     test_asi = pd.DataFrame(test_asi, 
    #                             index=[1983, 1984, 1985, 1986, 1987], 
    #                             columns=['Intensity Index', 'Activity Index', 'Structure Index (lower level)'])

    #     test_weights = 
    #     test_log_ratios = 
    #     weather_data = None
    #     model = 'multiplicative'
    #     components = self.calc_ASI(model, weather_data, weights, test_log_ratios)
    #     results = eii.decomposition(components)
    #     results = results[['effect']].round(4)

    #     comparison_output = [[0.9502],
    #                          [0.9824],
    #                          [1.0000],
    #                          [1.0310],
    #                          [1.0553]]
        
    #     comparison_output = pd.DataFrame(comparison_output, 
    #                                      index=[1983, 1984, 1985, 1986, 1987], 
    #                                      columns=['effect'])

    # def test_lower_level_structure(self, sector='transportation', acceptable_pct_diff=0.05):
    #     eii = self.eii_output_factory(sector)
    #     final_fmt_results =
    #     categories = 
    #     lower_level_results = eii.calc_lower_level(categories, final_fmt_results, e_type='deliv')

    #     comparison_df = 
    #     acceptable_bool = self.pct_diff(comparison_df, lower_level_results, acceptable_pct_diff, sector)

    def test_shift(self, sector='transportation', acceptable_pct_difference=0.05):
        eii = self.eii_output_factory(sector)
        pnnl_data = [[0.5433, 0.1449], [0.5479, 0.1402], [0.5650, 0.1367]]
        energy_shares = pd.DataFrame(pnnl_data, index=[1970, 1971, 1972], columns=['Highway', 'Rail']) 

        log_mean_weights = pd.DataFrame(index=energy_shares.index)
        print("log_mean_divisia_weights energy shares:", energy_shares)
        for col in energy_shares.columns: 
            print(f'log_mean_divisia_weights col: {col}')
            energy_shares[f"{col}_shift"] = energy_shares[col].shift(periods=1, axis='index', fill_value=0)
            print('energy shares with shift:\n', energy_shares)
            # apply generally not preferred for row-wise operations but?
            log_mean_weights[f'log_mean_weights_{col}'] = energy_shares.apply(lambda row: \
                                                          eii.logarithmic_average(row[col], row[f"{col}_shift"]), axis=1)
        print('log_mean_weights:\n', log_mean_weights)
        log_mean_weights = log_mean_weights.loc[1971:, :]
        print('log_mean_weights:\n', log_mean_weights)
        log_mean_weights = log_mean_weights.round(4)
        print('log_mean_weights:\n', log_mean_weights)

        pnnl_results = [[0.5456, 0.1425], [0.5564, 0.1385]]
        pnnl_df = pd.DataFrame(pnnl_results, index=[1971, 1972], columns=['log_mean_weights_Highway', 'log_mean_weights_Rail'])
        print('pnnl_df:\n', pnnl_df)
        acceptable_bool = self.pct_diff(pnnl_df, log_mean_weights, acceptable_pct_difference, sector)
        assert acceptable_bool
        
    def test_normalize_weights(self, sector='transportation', acceptable_pct_difference=0.05):
        # eii = self.eii_output_factory(sector)
        pnnl_results = [[0.5456, 0.1425, 0.0436, 0.0556, 0.2126],
                        [0.5564, 0.1385, 0.0449, 0.0525, 0.2076]]
        log_mean_weights = pd.DataFrame(pnnl_results, index=[1971, 1972], columns=['log_mean_weights_Highway', 'log_mean_weights_Rail', 'log_mean_weights_Air', 'log_mean_weights_Waterborne', 'log_mean_weights_Pipeline'])
        sum_log_mean_shares = log_mean_weights.sum(axis=1)
        test_total = pd.Series([[0.9999], [0.9999]], index=[1971, 1972])
        print('sum_log_mean_shares:\n', sum_log_mean_shares)
        print('sum_log_mean_shares == test_total', sum_log_mean_shares.equals(test_total))
        log_mean_weights_normalized = log_mean_weights.divide(sum_log_mean_shares.values.reshape(len(sum_log_mean_shares), 1))
        log_mean_weights_normalized = log_mean_weights_normalized.round(4)
        print('log_mean_weights_normalized:\n', log_mean_weights_normalized)

        pnnl_normalized = [[0.5456, 0.1425, 0.0436, 0.0556, 0.2126],
                           [0.5565, 0.1385, 0.0449, 0.0525, 0.2076]]
        pnnl_normalized_df = pd.DataFrame(pnnl_normalized, index=[1971, 1972], columns=['log_mean_weights_Highway', 'log_mean_weights_Rail', 'log_mean_weights_Air', 'log_mean_weights_Waterborne', 'log_mean_weights_Pipeline'])
        print('pnnl_normalized_df:\n', pnnl_normalized_df)
        acceptable_bool = self.pct_diff(pnnl_normalized_df, log_mean_weights_normalized, acceptable_pct_difference, sector=sector)
        assert acceptable_bool

    # @pytest.mark.parametrize('sector', ['transportation']) # 'residential', 'commercial', 'industrial', 'electricity', 
    # def test_multiplicative_lmdi_log_mean_divisia_weights(self, sector, acceptable_pct_difference=0.05):
    #     """Multiplicative test should use original PNNL data (compiled for #13)
    #     Test should be parametrized to loop through all sectors.
    #     """                
    #     eii = self.eii_output_factory(sector)

    #     pnnl_data = self.get_pnnl_input(sector, 'intermediate')
    #     eii_, pnnl_data_ = self.input_data(sector, level_of_aggregation_='All_Freight.Pipeline')

    #     bools_list = []

    #     for e_type in pnnl_data['Energy Type'].unique():

    #         for level_ in pnnl_data['Nest level'].unique():
    #             energy_data = pnnl_data_[e_type]['energy']
    #             energy_shares = pnnl_data[(pnnl_data['Energy Type'] == e_type) & (pnnl_data['Data Type'] == 'Energy Shares') & (pnnl_data['Nest level'] == level_)]
    #             energy_shares = energy_shares[['Year', 'Category', 'Value']]
    #             energy_shares['Value'] = energy_shares['Value'].astype(float)
    #             energy_shares = energy_shares.pivot(index='Year', columns='Category', values='Value')
    #             print('energy_share columns (should just be pipeline):', energy_shares.columns)

    #             energy_shares = energy_shares.dropna(axis=1, how='all')
    #             total_label = level_
    #             model_ = MultiplicativeLMDI(energy_data, energy_shares, 1985, 2017, total_label)
    #             eii_output = model_.log_mean_divisia_weights()
    #             print('eii_output log mean divisia weights: \n', eii_output.head())

    #             # pnnl_data_raw = self.get_pnnl_data(sector)['results']
    #             # weights_cols = [cat for cat in pnnl_data_raw['Category'].unique() if cat.endswith('eights')]
    #             # pnnl_weights = pnnl_data_raw[pnnl_data_raw['Category'].isin(weights_cols)]
    #             print('pnnl_data:\n', pnnl_data)
    #             pnnl_weights = pnnl_data[(pnnl_data['Energy Type'] == e_type) & (pnnl_data['Data Type'] == 'Log Mean Divisia Weights (normalized)') & (pnnl_data['Nest level'] == level_)]
    #             pnnl_weights = pnnl_weights[['Year', 'Category', 'Value']]

    #             print('pnnl_weights:\n', pnnl_weights)

    #             pnnl_weights_ = pnnl_weights.pivot(index='Year', columns='Category', values='Value')
    #             pnnl_weights_.columns.name = None
    #             print('pnnl_weights_:\n', pnnl_weights_)

    #             pnnl_weights_ = pnnl_weights_.dropna(axis=1, how='all')
    #             pnnl_weights_ = pnnl_weights_.rename(columns={col: f'log_mean_weights_{col}' for col in pnnl_weights_.columns})
    #             print('pnnl_weights_:\n', pnnl_weights_)

    #             acceptable_bool = self.pct_diff(pnnl_weights_, eii_output, acceptable_pct_difference, sector)
   
    #             bools_list.append(acceptable_bool)
    #             print('bools_list:', bools_list)    

    #     assert all(bools_list) #         assert all(all(bools_list))


    # @pytest.mark.parametrize('sector', ['transportation']) # 'residential', 'commercial', 'industrial', 'electricity', 
    # def test_additive_lmdi_log_mean_divisia_weights(self, sector):
    #     """Additive test should use "fake" data whose results are know
    #     Test should be parametrized to loop through all sectors.
    #     """

    #     eii = self.eii_output_factory(sector)

    #     pnnl_data = self.get_pnnl_input(sector, 'intermediate')
    #     eii, pnnl_data_ = self.input_data(sector)

    #     bools_list = []

    #     for e_type in pnnl_data['Energy Type'].unique():

    #         for level_ in pnnl_data['Nest level'].unique():
    #             energy_data = pnnl_data_[e_type]['energy'].head(5)
    #             energy_shares = pnnl_data[pnnl_data['Data Type'] == 'Energy Shares']
    #             energy_shares = energy_shares[['Year', 'Category', 'Value']]
    #             energy_shares['Value'] = energy_shares['Value'].astype(float)
    #             energy_shares = energy_shares.pivot(index='Year', columns='Category', values='Value')
    #             energy_shares = energy_shares.dropna(axis=1, how='all').head(5)
    #             total_label = level_
    #             lmdi_type = 'LMDI-I'      
    #             model_ = AdditiveLMDI(energy_data, energy_shares, 1985, 2017, total_label, lmdi_type)
    #             eii_output = model_.log_mean_divisia_weights()



    #             eii = self.eii_output_factory(sector)
    #             pnnl_data = [[1980.1, 528.1], 
    #                          [2072.5, 530.2],
    #                          [2290.9, 554.4]]
    #             energy_data = pd.DataFrame(pnnl_data, index=[1970, 1971, 1972], columns=['Highway', 'Rail']) 

    #             log_mean_weights = pd.DataFrame(index=energy_data.index)
    #             print("log_mean_divisia_weights energy shares:", energy_data)
    #             for col in energy_data.columns: 
    #                 print(f'log_mean_divisia_weights col: {col}')
    #                 energy_data[f"{col}_shift"] = energy_data[col].shift(periods=1, axis='index', fill_value=0)
    #                 print('energy shares with shift:\n', energy_data)
    #                 # apply generally not preferred for row-wise operations but?
    #                 log_mean_weights[f'log_mean_weights_{col}'] = energy_data.apply(lambda row: \
    #                                                               eii.logarithmic_average(row[col], row[f"{col}_shift"]), axis=1)
    #             print('log_mean_weights:\n', log_mean_weights)
    #             log_mean_weights = log_mean_weights.loc[1971:, :]
    #             log_mean_weights = log_mean_weights.round(4)
    #             print('log_mean_weights:\n', log_mean_weights)

    #             print('energy_data:\n', energy_data)
    #             print('energy_shares:\n', energy_shares)
    #             print('eii_output log mean divisia weights: \n', eii_output.head())
    #             one = eii.logarithmic_average(2072.5, 1980.1)
    #             two = eii.logarithmic_average(2290.9, 2072.5)
    #             three = eii.logarithmic_average(530.2, 528.1)
    #             four = eii.logarithmic_average(554.4, 530.2)
    #             constructed_data = [[one, three], [two, four]]

    #             constructed_df = pd.DataFrame(constructed_data, columns=eii_output.columns)
    #             bools_list.append(eii_output.equal(constructed_data))

    #     assert all(bools_list)

    def pct_diff(self, pnnl_data, eii_data, acceptable_pct_difference, sector):
        eii = self.eii_output_factory(sector)
        pnnl_data, eii_data = eii.ensure_same_indices(pnnl_data, eii_data)
        diff_df = pnnl_data.subtract(eii_data)
        diff_df_abs = np.absolute(diff_df)
        pct_diff = np.absolute(diff_df_abs.divide(pnnl_data))
        compare_df = pct_diff.fillna(0).apply(lambda col: col<=acceptable_pct_difference, axis=1)

        print('compare df:\n', compare_df)
        print('diff_df: ', diff_df)
        print('\npct_diff:\n', pct_diff)
        print('(pct_diff <= acceptable_pct_difference).all():', (pct_diff <= acceptable_pct_difference).all().all())
        return compare_df.all(axis=None)

    # @pytest.mark.parametrize('sector', ['transportation']) # 'residential', 'commercial', 'industrial', 'electricity', 
    # def test_calc_asi(self, sector, acceptable_pct_difference=0.05):
    #     """Write test_calc_ASI to test LMDI class.

    #     - Test both additive and multiplicative forms
    #     - Test all sectors
    #     """   

    #     eii = self.eii_output_factory(sector)

    #     pnnl_data = self.get_pnnl_input(sector, 'intermediate')

    #     pnnl_output = self.get_pnnl_data(sector)
    #     pnnl_output = pnnl_output['results']

    #     model = 'multiplicative'

    #     bools_list = []

    #     for e_type in pnnl_data['Energy Type'].unique():

    #         for level_ in pnnl_data['Nest level'].unique():

            
    #             if 'Weather' in pnnl_data['Energy Type']:
    #                 weather_data = pnnl_data[pnnl_data['Energy Type'] == 'Weather']
    #             else:
    #                 weather_data = None

    #             log_mean_divisia_weights_normalized = pnnl_data[pnnl_data['Data Type'] == 'Log Mean Divisia Weights (normalized)'][['Year', 'Category', 'Value']].pivot(index='Year', columns='Category', values='Value').dropna(axis=1, how='all')
    #             log_ratio_activity = pnnl_data[pnnl_data['Data Type'] == 'Log Changes Activity'][['Year', 'Category', 'Value']].pivot(index='Year', columns='Category', values='Value').dropna(axis=1, how='all')
    #             log_ratio_structure = pnnl_data[pnnl_data['Data Type'] == 'Log Changes Lower-level Structure'][['Year', 'Category', 'Value']].pivot(index='Year', columns='Category', values='Value').dropna(axis=1, how='all')
    #             log_ratio_intensity = pnnl_data[pnnl_data['Data Type'] == 'Log Changes Intensity'][['Year', 'Category', 'Value']].pivot(index='Year', columns='Category', values='Value').dropna(axis=1, how='all')

    #             log_ratios = {'activity': log_ratio_activity, 
    #                           'structure': log_ratio_structure, 
    #                           'intensity': log_ratio_intensity}
    #             print('log ratios:\n', log_ratios)
    #             print('log ratios type:\n', type(log_ratios['activity']))

    #             print("log_mean_divisia_weights_normalized: \n", log_mean_divisia_weights_normalized)

    #             print("log_mean_divisia_weights_normalized type: \n", type(log_mean_divisia_weights_normalized))

    #             eii_output = eii.calc_ASI(model, weather_data, log_mean_divisia_weights_normalized, 
    #                                       log_ratios)
    #             print('eii_output calc asi:\n', eii_output)
    #             pnnl_output_ = pnnl_output[(pnnl_output['Energy Type'] == e_type) & (pnnl_output['Nest level'] == level_)]
    #             pnnl_output_ = pnnl_output_[pnnl_output_['Category'].isin(['Structure: Lower level (**)', 'Component Intensity Index', 'Weighted Activity Index'])]
    #             pnnl_output_ = pnnl_output_.pivot(index='Year', columns='Category', values='Value')
    #             print('pnnl_output calc asi:\n', pnnl_output_)
    #             acceptable_bool = self.pct_diff(pnnl_output_, eii_output, acceptable_pct_difference, sector)
    #             bools_list.append(acceptable_bool)
        
    #     assert all(bools_list)  #         assert all(all(bools_list))


if __name__ == '__main__':
    test = TestLMDI()
    data = test.get_pnnl_data('transportation')
    print(data)

