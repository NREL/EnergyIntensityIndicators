"""Overview, summary of work from pnnl, highlight results with multiplicative and additive figures, """

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
# from EnergyIntensityIndicators.utilities.dataframe_utilities \
#     import DFUtilities as df_utils
from EnergyIntensityIndicators.utilities.testing_utilties \
    import TestingUtilities
from EnergyIntensityIndicators.utilities import lmdi_utilities
from EnergyIntensityIndicators import REPODIR
from EnergyIntensityIndicators.utilities import loggers


logger = loggers.get_logger()


class TestLMDI:
    sector_modules = {'residential': ResidentialIndicators,
                      'commercial': CommercialIndicators,
                      'transportation': TransportationIndicators,
                      'industrial': IndustrialIndicators,
                      'electricity': ElectricityIndicators}

    pnnl_directory = os.path.join(REPODIR, 'tests/Indicators_Spreadsheets_2020')
    output_directory = os.path.join(REPODIR, 'tests/Results')
    utils = TestingUtilities()

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
        data_melt = pd.melt(
            data, id_vars=id_vars, value_vars=value_vars).rename(
                columns={'variable': 'Category', 'value': 'Value'})  #col_level= # , var_name='Category',
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
        files = os.listdir(os.path.join(REPODIR, f'tests/pnnl_csvs/{sector}/{dtype}/'))
        files = [f for f in files if f.endswith('.csv')]
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(os.path.join(REPODIR, f'tests/pnnl_csvs/{sector}/{dtype}/{f}'))
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

        return {'activity': activity_data,
                'energy': energy_data,
                'weather': weather}

    def input_data(self, sector, level_of_aggregation_='All_Transportation.All_Freight.Pipeline'):
        eii = self.eii_output_factory(sector)
        raw_eii_data = eii.collect_data()
        eii, final_results = eii.get_nested_lmdi(level_of_aggregation_,
                                                 raw_eii_data,
                                                 calculate_lmdi=False,
                                                 breakout=False,
                                                 lmdi_type='II')
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
                for col in eii[energy_type.lower()]['energy'].columns:
                    try:
                        pnnl = \
                            pnnl_data_raw[pnnl_data_raw['Nest level'] == col]
                    except KeyError:
                        print('error there')
                        continue
                    for d_type in pnnl['Data Type'].unique():
                        try:
                            pnnl_sub = \
                                pnnl[pnnl['Data Type'] == d_type].pivot(
                                    index='Year', columns='Category',
                                    values='Value')
                            pnnl_sub.index = pnnl_sub.index.astype(int)
                            pnnl_sub = \
                                pnnl_sub.rename_axis(col, axis='columns')
                            pnnl_sub = pnnl_sub.fillna(np.nan)
                        except KeyError:
                            print('error here')
                            continue
                        energy_type_dict[d_type.lower()] = pnnl_sub
            else:
                continue

            pnnl_data_[energy_type.lower()] = energy_type_dict

        return eii, pnnl_data_

    @pytest.mark.parametrize(
        ('sector'),
        ((['transportation']))
    ) # , 'residential', 'commercial', 'industrial', 'electricity'
    def test_build_nest(self, sector):
        """testing the results of LMDI.build_nest against a csv
           (to be compiled) of PNNL data.

        - Assertion should be in terms of a % difference from the PNNL data.
        - Test should be parameterized to loop through and test all sectors.

        output of build_nest:
        data_dict = {'energy': energy_data,
                     'activity': activity_data,
                     'level_total': level_name}

        results_dict[f'{level_name}'] = data_dict
        """
        eii, pnnl = self.input_data(sector)
        for energy_type, energy_dict in pnnl.items():
            for data_type, data_dict in energy_dict.items():
                for cat, pnnl_df in data_dict.items():
                    pnnl_df.index = pnnl_df.index.astype(int)
                    eii_df = eii[energy_type][data_type][[cat]]

                    acceptable_bool = self.utils.pct_diff(pnnl_df, eii_df)

                    assert all(acceptable_bool)

    # 'residential', 'commercial', 'industrial', 'electricity',
    @pytest.mark.parametrize('sector', ['transportation'])
    def test_prepare_lmdi_inputs(self, sector, acceptable_pct_diff=0.5):
        """`LMDI.prepare_lmdi_inputs to test original PNNL data
            (compiled for #13 ) against PNNL results for energy_input_data,
            energy_shares, and log_ratios

        -Test should be parameterized to loop through all sectors.

        prepare_lmdi_inputs returns the following:
            log_ratios = {'activity': log_ratio_activity,
                          'structure': log_ratio_structure,
                          'intensity': log_ratio_intensity}
        """
        eii = self.eii_output_factory(sector)
        pnnl_output = self.get_pnnl_input(sector, 'intermediate')

        print('pnnl_output:\n', pnnl_output)

        eii_data, pnnl = \
              self.input_data(sector,
                              level_of_aggregation_='All_Freight.Pipeline')
        print('pnnl:', pnnl)
        for e_, e_dict in pnnl.items():
            activity_data = e_dict['activity']

            print('activity_data.columns:', activity_data.columns)
            energy_data = e_dict['energy']
            print('energy_data.columns:', energy_data.columns)
            try:
                total_label = activity_data.columns.name
            except ValueError:
                print(f'Error: activity data of type {type(activity_data)} \
                        with columns: {activity_data.columns}')
                continue

            activity_dict = dict()
            if isinstance(activity_data, pd.DataFrame):
                activity_dict[total_label] = activity_data

            print('activity_data:', activity_data)
            print('energy_data:', energy_data)
            if total_label in activity_data.columns and \
                  total_label in energy_data.columns:
                energy_data, energy_shares, eii_log_ratios = \
                      eii.prepare_lmdi_inputs(energy_input_data=energy_data,
                                              activity_input_data=activity_dict,
                                              total_label=total_label)
                print('eii_output:\n', eii_log_ratios)
                print('pnnl_output_ pre manipulation:\n', pnnl_output)
                pnnl_output = \
                  pnnl_output.replace({'Pipelines': 'Pipeline',
                                       'Freight Total': 'All_Freight',
                                       'Deliv': 'deliv'})
                print("total_label in pnnl_output['Nest level']", total_label in pnnl_output['Nest level'])
                print("e_ in pnnl_output['Energy Type']", e_ in pnnl_output['Nest level'])

                print("pnnl_output['Energy Type'].unique()]", pnnl_output['Energy Type'].unique())
                print("pnnl_output['Nest level'].unique()]", pnnl_output['Nest level'].unique())
                print("pnnl_output['Category'].unique()", pnnl_output['Category'].unique())

                pnnl_output_ = pnnl_output[(pnnl_output['Energy Type'] == e_) & (pnnl_output['Nest level'] == total_label)]
                print('pnnl_output_ here:\n', pnnl_output_)

                pnnl_component_data = dict()
                for d_type in pnnl_output_['Data Type'].unique():
                    pnnl_df = \
                        pnnl_output_[pnnl_output_['Data Type'] == \
                            d_type][['Year', 'Category', 'Value']]
                    pnnl_df = pnnl_df.pivot(index='Year',
                                            columns='Category',
                                            values='Value').dropna(
                                                axis=1, how='all')
                    pnnl_df.columns.name = total_label
                    print('dtype:', d_type)
                    print('pnnl_df:\n', pnnl_df)
                    if d_type == 'Log Changes Intensity':
                        pnnl_component_data['intensity'] = pnnl_df
                    elif d_type == 'Log Changes Activity':
                        pnnl_component_data['activity'] = pnnl_df
                    elif d_type == 'Log Changes Structure':
                        pnnl_component_data['structure'] = pnnl_df
                print('pnnl_component_data:\n', pnnl_component_data)
                eii_test_data = {k: eii_log_ratios[k] for k in pnnl_component_data.keys()}
                print('eii_test_data:\n', eii_test_data)
                if eii_test_data != eii_log_ratios:
                    print('PNNL missed components')
                bools_list = [self.utils.pct_diff(pnnl_component_data[k], eii_test_data[k]) for k in pnnl_component_data.keys()]
                assert all(bools_list)
            else:
                print(f'Missing {total_label}, with {activity_data.columns} activity columns and \
                        {energy_data.columns} energy columns')
                continue

    def test_calc_sum_product(self, sector='transportation'):
        eii = self.eii_output_factory(sector)

        log_ratio_component = [[np.nan, np.nan],
                               [-0.0086, -0.0129],
                               [-0.0056, 0.0170],
                               [0.0044, 0.0005],
                               [-0.0190, -0.0009]]
        log_ratio_component = pd.DataFrame(log_ratio_component,
                                           index=[1970, 1971, 1972, 1973, 1974],
                                           columns=['All_Passenger',
                                                    'All_Freight'])

        weights = [[0.3911, 0.6089],
                   [0.7602, 0.2398],
                   [0.7610, 0.2390],
                   [0.7596, 0.2404],
                   [0.7563, 0.2437]]

        weights = pd.DataFrame(weights,
                               index=[1970, 1971, 1972, 1973, 1974],
                               columns=['All_Passenger', 'All_Freight'])

        component = eii.sum_product(log_ratio_component, weights,
                                    name='Intensity Index')
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
        # assert component.equals(comparison_output)
        assert self.utils.pct_diff(comparison_output, component)

    def test_compute_index1(self):
        """Data is from Total_Transportation 1983-1987"""
        eii = MultiplicativeLMDI(output_directory='./Results')

        results = [[0.9705, 1.0386, 1.0037],
                   [0.9957, 1.0329, 1.0054],
                   [0.9982, 1.0145, 1.0052],
                   [1.0076, 1.0165, 1.0066],
                   [0.9814, 1.0412, 1.0016]]

        results = pd.DataFrame(results,
                               index=[1983, 1984, 1985, 1986, 1987],
                               columns=['Intensity Index',
                                        'Activity Index',
                                        'Structure Index'])

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
                                         columns=['Intensity Index',
                                                  'Activity Index',
                                                  'Structure Index'])
        print('results_:\n', results)
        print('comparison_output:\n', comparison_output)
        # assert results.equals(comparison_output)
        assert self.utils.pct_diff(comparison_output, results)

    def test_compute_index2(self):
        """Data is from Total_Transportation 1970-1975"""
        eii = MultiplicativeLMDI(output_directory=None)

        results = [[np.nan, 1.1301, np.nan],
                   [0.9904, 1.0460, 1.0107],
                   [0.9998, 1.0621, 1.0068],
                   [1.0034, 1.0370, 1.0039],
                   [0.9855, 0.9809, 0.9996],
                   [0.9981, 1.0050, 1.0085],
                   [0.9858, 1.0593, 1.0070],
                   [0.9791, 1.0444, 1.0064],
                   [0.9886, 1.0691, 0.9922],
                   [0.9883, 1.0045, 0.9977],
                   [0.9682, 0.9980, 0.9859],
                   [0.9975, 1.0034, 0.9972],
                   [0.9759, 0.9972, 1.0117],
                   [0.9705, 1.0386, 1.0037],
                   [0.9957, 1.0329, 1.0054],
                   [0.9982, 1.0145, 1.0052],
                   [1.0076, 1.0165, 1.0066],
                   [0.9814, 1.0412, 1.0016]]

        results = pd.DataFrame(results,
                               index=[1970, 1971, 1972, 1973,
                                      1974, 1975, 1976, 1977,
                                      1978, 1979, 1980, 1981,
                                      1982, 1983, 1984, 1985,
                                      1986, 1987],
                               columns=['Intensity Index',
                                        'Activity Index',
                                        'Structure Index'])

        for col in results.columns:
            results[col] = eii.compute_index(results[col], 1985)
            results[col] = results[col].astype(float).round(4)

        comparison_output = [[1.1935, 0.6819, 0.9594],
                             [1.1821, 0.7133, 0.9696],
                             [1.1818, 0.7576, 0.9762],
                             [1.1859, 0.7856, 0.9800],
                             [1.1687, 0.7707, 0.9796],
                             [1.1665, 0.7745, 0.9879],
                             [1.1499, 0.8204, 0.9948],
                             [1.1258, 0.8568, 1.0013],
                             [1.1130, 0.9160, 0.9934],
                             [1.1000, 0.9202, 0.9912],
                             [1.0650, 0.9183, 0.9772],
                             [1.0623, 0.9214, 0.9745],
                             [1.0368, 0.9188, 0.9859],
                             [1.0062, 0.9543, 0.9895],
                             [1.0018, 0.9857, 0.9948],
                             [1.0000, 1.0000, 1.0000],
                             [1.0076, 1.0165, 1.0066],
                             [0.9889, 1.0584, 1.0082]]

        comparison_output = pd.DataFrame(comparison_output,
                                         index=[1970, 1971, 1972, 1973,
                                                1974, 1975, 1976, 1977,
                                                1978, 1979, 1980, 1981,
                                                1982, 1983, 1984, 1985,
                                                1986, 1987],
                                         columns=['Intensity Index',
                                                  'Activity Index',
                                                  'Structure Index'])
        print('results_:\n', results)
        print('comparison_output:\n', comparison_output)
        # assert results.equals(comparison_output)
        assert self.utils.pct_diff(comparison_output, results)

    def test_multiplicative_decomposition(self, sector='transportation'):
        mult = MultiplicativeLMDI(output_directory='./Results')
        eii = self.eii_output_factory(sector)

        test_weights = [[0.7258, 0.2742],
                        [0.7276, 0.2724],
                        [0.7307, 0.2693],
                        [0.7372, 0.2628],
                        [0.7380, 0.2620]]
        test_weights = pd.DataFrame(test_weights,
                                    index=[1983, 1984, 1985, 1986, 1987],
                                    columns=['All Passenger', 'All Freight'])

        log_change_intensity = [[-0.0172, -0.0637],
                                [-0.0016, -0.0118],
                                [0.0024, -0.0134],
                                [0.0114, -0.0029],
                                [-0.0180, -0.0211]]
        log_change_intensity = pd.DataFrame(log_change_intensity,
                                            index=[1983, 1984, 1985,
                                                   1986, 1987],
                                            columns=['All Passenger',
                                                     'All Freight'])

        log_change_activity = [[0.0353, 0.0447],
                               [0.0289, 0.0416],
                               [0.0245, -0.0129],
                               [0.0239, -0.0049],
                               [0.0360, 0.0528]]
        log_change_activity = pd.DataFrame(log_change_activity,
                                           index=[1983, 1984, 1985,
                                                  1986, 1987],
                                           columns=['All Passenger',
                                                    'All Freight'])

        log_change_lower_level_structure = [[0.0010, 0.0109],
                                            [0.0034, 0.0106],
                                            [0.0019, 0.0139],
                                            [0.0018, 0.0199],
                                            [0.0008, 0.0040]]
        log_change_lower_level_structure = \
            pd.DataFrame(log_change_lower_level_structure,
                         index=[1983, 1984, 1985, 1986, 1987],
                         columns=['All Passenger', 'All Freight'])

        log_change_structure = [[1, 1],
                                [1, 1],
                                [1, 1],
                                [1, 1],
                                [1, 1]]

        log_change_structure = pd.DataFrame(log_change_structure,
                                            index=[1983, 1984, 1985,
                                                   1986, 1987],
                                            columns=['All Passenger',
                                                     'All Freight'])

        test_log_ratios = {'intensity': log_change_intensity,
                           'activity': log_change_activity,
                           'structure': log_change_structure,
                           'lower_level_structure': log_change_lower_level_structure}

        test_asi = [[1.0062, 0.9543, 0.9895],
                    [1.0018, 0.9857, 0.9948],
                    [1.0000, 1.0000, 1.0000],
                    [1.0076, 1.0165, 1.0066],
                    [0.9889, 1.0584, 1.0082]]

        test_asi = pd.DataFrame(test_asi,
                                index=[1983, 1984, 1985,
                                       1986, 1987],
                                columns=['Intensity Index',
                                         'Activity Index',
                                         'Structure Index (lower level)'])

        model = 'multiplicative'
        components = eii.calc_ASI(model, test_weights,
                                  test_log_ratios,
                                  total_label='Transportation')
        print('test_asi:\n', test_asi)
        print('components:\n', components)
        results = mult.decomposition(components)
        print('results:\n', results)
        results = results[['effect']].round(4)

        comparison_output = [[0.9502],
                             [0.9824],
                             [1.0000],
                             [1.0310],
                             [1.0553]]

        comparison_output = pd.DataFrame(comparison_output,
                                         index=[1983, 1984, 1985,
                                                1986, 1987],
                                         columns=['effect'])

    # def test_lower_level_structure(self, sector='transportation',
    #                                acceptable_pct_diff=0.05):
    #     eii = self.eii_output_factory(sector)
    #     final_fmt_results =
    #     categories =
    #     lower_level_results = eii.calc_lower_level(categories,
    #                                                final_fmt_results,
    #                                                e_type='deliv')

    #     comparison_df =
    #     acceptable_bool = self.utils.pct_diff(comparison_df,
    #                                           lower_level_results)

    def test_shift(self):
        pnnl_data = [[0.5433, 0.1449], [0.5479, 0.1402], [0.5650, 0.1367]]
        energy_shares = pd.DataFrame(pnnl_data, index=[1970, 1971, 1972],
                                     columns=['Highway', 'Rail'])

        log_mean_weights = pd.DataFrame(index=energy_shares.index)
        print("log_mean_divisia_weights energy shares:", energy_shares)
        for col in energy_shares.columns:
            print(f'log_mean_divisia_weights col: {col}')
            energy_shares[f"{col}_shift"] = \
                energy_shares[col].shift(periods=1,
                                         axis='index',
                                         fill_value=0)
            print('energy shares with shift:\n', energy_shares)
            # apply generally not preferred for row-wise operations but?
            log_mean_weights[f'log_mean_weights_{col}'] = \
                energy_shares.apply(lambda row:
                                    lmdi_utilities.logarithmic_average(row[col],
                                                            row[f"{col}_shift"]),
                                    axis=1)
        print('log_mean_weights:\n', log_mean_weights)
        log_mean_weights = log_mean_weights.loc[1971:, :]
        print('log_mean_weights:\n', log_mean_weights)
        log_mean_weights = log_mean_weights.round(4)
        print('log_mean_weights:\n', log_mean_weights)

        pnnl_results = [[0.5456, 0.1425], [0.5564, 0.1385]]
        pnnl_df = pd.DataFrame(pnnl_results, index=[1971, 1972],
                               columns=['log_mean_weights_Highway',
                               'log_mean_weights_Rail'])
        print('pnnl_df:\n', pnnl_df)
        acceptable_bool = self.utils.pct_diff(pnnl_df, log_mean_weights)
        assert acceptable_bool

    def test_normalize_weights(self, sector='transportation'):
        # eii = self.eii_output_factory(sector)
        pnnl_results = [[0.5456, 0.1425, 0.0436, 0.0556, 0.2126],
                        [0.5564, 0.1385, 0.0449, 0.0525, 0.2076]]
        log_mean_weights = \
            pd.DataFrame(pnnl_results, index=[1971, 1972],
                         columns=['log_mean_weights_Highway',
                                  'log_mean_weights_Rail',
                                  'log_mean_weights_Air',
                                  'log_mean_weights_Waterborne',
                                  'log_mean_weights_Pipeline'])

        sum_log_mean_shares = log_mean_weights.sum(axis=1)
        test_total = pd.Series([[0.9999], [0.9999]], index=[1971, 1972])
        print('sum_log_mean_shares:\n', sum_log_mean_shares)
        print('sum_log_mean_shares == test_total', sum_log_mean_shares.equals(test_total))
        log_mean_weights_normalized = \
            log_mean_weights.divide(
                sum_log_mean_shares.values.reshape(len(sum_log_mean_shares), 1))
        log_mean_weights_normalized = log_mean_weights_normalized.round(4)
        print('log_mean_weights_normalized:\n', log_mean_weights_normalized)

        pnnl_normalized = [[0.5456, 0.1425, 0.0436, 0.0556, 0.2126],
                           [0.5565, 0.1385, 0.0449, 0.0525, 0.2076]]
        pnnl_normalized_df = pd.DataFrame(pnnl_normalized, index=[1971, 1972],
                                          columns=['log_mean_weights_Highway',
                                                   'log_mean_weights_Rail',
                                                   'log_mean_weights_Air',
                                                   'log_mean_weights_Waterborne',
                                                   'log_mean_weights_Pipeline'])

        print('pnnl_normalized_df:\n', pnnl_normalized_df)
        acceptable_bool = self.utils.pct_diff(pnnl_normalized_df,
                                              log_mean_weights_normalized)
        assert acceptable_bool

    @pytest.mark.parametrize('sector', ['transportation']) # 'residential', 'commercial', 'industrial', 'electricity',
    def test_multiplicative_lmdi_log_mean_divisia_weights(self, sector):
        """Multiplicative test should use original PNNL data (compiled for #13)
        Test should be parametrized to loop through all sectors.
        """

        pnnl_data = self.get_pnnl_input(sector, 'intermediate')
        eii_, pnnl_data_ = self.input_data(sector, level_of_aggregation_='All_Freight.Pipeline')

        bools_list = []

        for e_type in pnnl_data['Energy Type'].unique():

            for level_ in pnnl_data['Nest level'].unique():
                energy_data = pnnl_data_[e_type]['energy']
                energy_shares = pnnl_data[(pnnl_data['Energy Type'] == e_type) & (pnnl_data['Data Type'] == 'Energy Shares') & (pnnl_data['Nest level'] == level_)]
                energy_shares = energy_shares[['Year', 'Category', 'Value']]
                energy_shares['Value'] = energy_shares['Value'].astype(float)
                energy_shares = energy_shares.pivot(index='Year', columns='Category', values='Value')
                print('energy_share columns (should just be pipeline):', energy_shares.columns)

                energy_shares = energy_shares.dropna(axis=1, how='all')
                total_label = level_
                model_ = MultiplicativeLMDI(energy_data, energy_shares, 1985, 2017, total_label)
                eii_output = model_.log_mean_divisia_weights()
                print('eii_output log mean divisia weights: \n', eii_output.head())

                # pnnl_data_raw = self.get_pnnl_data(sector)['results']
                # weights_cols = [cat for cat in pnnl_data_raw['Category'].unique() if cat.endswith('eights')]
                # pnnl_weights = pnnl_data_raw[pnnl_data_raw['Category'].isin(weights_cols)]
                print('pnnl_data:\n', pnnl_data)
                pnnl_weights = pnnl_data[(pnnl_data['Energy Type'] == e_type) & (pnnl_data['Data Type'] == 'Log Mean Divisia Weights (normalized)') & (pnnl_data['Nest level'] == level_)]
                pnnl_weights = pnnl_weights[['Year', 'Category', 'Value']]

                print('pnnl_weights:\n', pnnl_weights)

                pnnl_weights_ = pnnl_weights.pivot(index='Year', columns='Category', values='Value')
                pnnl_weights_.columns.name = None
                print('pnnl_weights_:\n', pnnl_weights_)

                pnnl_weights_ = pnnl_weights_.dropna(axis=1, how='all')
                pnnl_weights_ = pnnl_weights_.rename(columns={col: f'log_mean_weights_{col}' for col in pnnl_weights_.columns})
                print('pnnl_weights_:\n', pnnl_weights_)

                acceptable_bool = self.utils.pct_diff(pnnl_weights_, eii_output)

                bools_list.append(acceptable_bool)
                print('bools_list:', bools_list)

        assert all(bools_list)  # assert all(all(bools_list))


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
    #                                                               lmdi_utilities.logarithmic_average(row[col], row[f"{col}_shift"]), axis=1)
    #             print('log_mean_weights:\n', log_mean_weights)
    #             log_mean_weights = log_mean_weights.loc[1971:, :]
    #             log_mean_weights = log_mean_weights.round(4)
    #             print('log_mean_weights:\n', log_mean_weights)

    #             print('energy_data:\n', energy_data)
    #             print('energy_shares:\n', energy_shares)
    #             print('eii_output log mean divisia weights: \n', eii_output.head())
    #             one = lmdi_utilities.logarithmic_average(2072.5, 1980.1)
    #             two = lmdi_utilities.logarithmic_average(2290.9, 2072.5)
    #             three = lmdi_utilities.logarithmic_average(530.2, 528.1)
    #             four = lmdi_utilities.logarithmic_average(554.4, 530.2)
    #             constructed_data = [[one, three], [two, four]]

    #             constructed_df = pd.DataFrame(constructed_data, columns=eii_output.columns)
    #             bools_list.append(eii_output.equal(constructed_data))

    #     assert all(bools_list)


    def get_eii_asi(self, sector):
        pass

    def test_calc_asi(self, sector='transportation'):
        """Write test_calc_ASI to test LMDI class.

        - Test both additive and multiplicative forms
        - Test all sectors
        """

        eii = self.eii_output_factory(sector)

        pnnl_data = self.get_pnnl_input(sector, 'intermediate')

        pnnl_output = self.get_pnnl_data(sector)
        pnnl_output = pnnl_output['results']

        model = 'multiplicative'

        bools_list = []

        for e_type in pnnl_data['Energy Type'].unique():

            for level_ in pnnl_data['Nest level'].unique():

                if 'Weather' in pnnl_data['Energy Type']:
                    weather_data = \
                        pnnl_data[pnnl_data['Energy Type'] == 'Weather']
                else:
                    weather_data = None

                log_mean_divisia_weights_normalized = \
                    pnnl_data[
                        pnnl_data['Data Type'] == 'Log Mean Divisia Weights (normalized)'][['Year',
                                                                                            'Category',
                                                                                            'Value']].pivot(index='Year',
                                                                                                            columns='Category',
                                                                                                            values='Value').dropna(
                                                                                                                axis=1, how='all')
                log_ratio_activity = \
                    pnnl_data[pnnl_data['Data Type'] == 'Log Changes Activity'][['Year',
                                                                                 'Category',
                                                                                 'Value']].pivot(index='Year',
                                                                                                columns='Category',
                                                                                                values='Value').dropna(
                                                                                                    axis=1, how='all')
                log_ratio_structure = \
                    pnnl_data[pnnl_data['Data Type'] == 'Log Changes Lower-level Structure'][['Year',
                                                                                              'Category',
                                                                                              'Value']].pivot(index='Year',
                                                                                                              columns='Category',
                                                                                                              values='Value')
                                                                                                              #.dropna(
                                                                                                                #  axis=1, how='all')
                log_ratio_intensity = \
                    pnnl_data[pnnl_data['Data Type'] == 'Log Changes Intensity'][['Year',
                                                                                  'Category',
                                                                                  'Value']].pivot(index='Year',
                                                                                                  columns='Category',
                                                                                                  values='Value').dropna(
                                                                                                      axis=1, how='all')

                log_ratios = {'activity': log_ratio_activity,
                              'structure': log_ratio_structure,
                              'intensity': log_ratio_intensity}

                print('log ratios:\n', log_ratios)
                print('log ratios type:\n', type(log_ratios['activity']))

                print("log_mean_divisia_weights_normalized: \n", log_mean_divisia_weights_normalized)

                print("log_mean_divisia_weights_normalized type: \n", type(log_mean_divisia_weights_normalized))

                eii_output = eii.calc_ASI(model, log_mean_divisia_weights_normalized,
                                          log_ratios, total_label=None)

                print('eii_output calc asi:\n', eii_output)
                pnnl_output_ = pnnl_output[(pnnl_output['Energy Type'] == e_type) & (pnnl_output['Nest level'] == level_)]
                pnnl_output_ = pnnl_output_[pnnl_output_['Category'].isin(['Structure: Lower level (**)',
                                                                           'Component Intensity Index',
                                                                           'Weighted Activity Index'])]
                pnnl_output_ = pnnl_output_.pivot(index='Year', columns='Category', values='Value')
                print('pnnl_output calc asi:\n', pnnl_output_)
                acceptable_bool = self.utils.pct_diff(pnnl_output_, eii_output)
                bools_list.append(acceptable_bool)

        assert all(bools_list)  #         assert all(all(bools_list))


    def test_components(self, sector='transportation'):
        """Write test_calc_ASI to test LMDI class.

        - Test both additive and multiplicative forms
        - Test all sectors
        """

        pnnl_eii_match = {'Passenger Highway': 'Highway',
                          'Freight Total': 'All_Freight',
                          "Component Intensity          Index": 'Index',
                          'Product: Activity x Structure x Intensity': 'Effect',
                          'Structure: Lower level': "Structure: Next lower level",
                          'Activity (passenger-miles)': 'Activity',
                          'Pipelines': 'Pipeline'}

        eii = self.eii_output_factory(sector)

        pnnl_data = self.get_pnnl_input(sector, 'intermediate')

        pnnl_output = self.get_pnnl_data(sector)
        pnnl_output = pnnl_output['results']
        print('pnnl_output columns:\n', pnnl_output.columns)
        for p, e in pnnl_eii_match.items():
            pnnl_output = pnnl_output.replace(p, e)

        output_directory = os.path.join(REPODIR, 'tests/Results/')
        print('os.getcwd:', os.getcwd())
        eii_results_data = \
            pd.read_csv(
                f'{output_directory}transportation_results2.csv').rename(
                    columns={'@timeseries|Year':
                             'Year'}).set_index('Year')

        eii_results_data = eii_results_data.replace('Highway', 'Freight Trucks')
        levels = eii_results_data['lower_level'].unique()
        print('pnnl levels:', pnnl_output['Nest level'].unique())
        model = 'multiplicative'

        bools_list = []

        for e_type in pnnl_output['Energy Type'].unique():

            for level_ in levels:
                print('level_:', level_)

                print('e_type:', e_type)
                if level_ == np.nan:
                    continue

                eii_data = eii_results_data[(eii_results_data['lower_level'] == level_) & (eii_results_data['@filter|EnergyType'] == e_type.lower()) & (eii_results_data['@filter|Model'] == model.capitalize())]
                eii_effect = eii_data[['@filter|Measure|Effect']].rename(columns={'@filter|Measure|Effect': 'Effect'})
                eii_intensity = eii_data[['@filter|Measure|Intensity']].rename(columns={'@filter|Measure|Intensity': 'Intensity'})
                eii_structure = eii_data[['@filter|Measure|Structure']].rename(columns={'@filter|Measure|Structure': 'Structure'})
                eii_lower_level_structure = eii_data[['lower_level_structure']].rename(columns={'lower_level_structure': 'Structure: Next lower level'})
                eii_activity = eii_data[['@filter|Measure|Activity']].rename(columns={'@filter|Measure|Activity': 'Activity'})

                data_ = pnnl_output[(pnnl_output['Nest level'] == level_) & (pnnl_output['Energy Type'] == e_type) & (pnnl_output['Sector'] == 'transportation')]
                print('categories:', data_['Category'].unique())

                pnnl_intensity = data_[data_['Category'] == 'Intensity'][['Year', 'Category', 'Value']].pivot(index='Year', columns='Category', values='Value').dropna(axis=1, how='all')
                pnnl_intensity.columns.name = None

                pnnl_activity = data_[data_['Category'] == 'Activity'][['Year', 'Category', 'Value']].pivot(index='Year', columns='Category', values='Value').dropna(axis=1, how='all')
                pnnl_activity.columns.name = None

                try:
                    pnnl_lower_level_structure = data_[data_['Category'] == 'Structure: Next lower level'][['Year', 'Category', 'Value']].pivot(index='Year', columns='Category', values='Value').dropna(axis=1, how='all')
                    pnnl_lower_level_structure.columns.name = None
                    if not pnnl_lower_level_structure.empty:
                        acceptable_bool_lower_level_structure = self.utils.pct_diff(pnnl_lower_level_structure, eii_lower_level_structure)
                        print(f"{level_} lower level structure is {acceptable_bool_lower_level_structure}")
                        bools_list.append(acceptable_bool_lower_level_structure)

                except KeyError:
                    pass

                pnnl_structure = \
                    data_[data_['Category'] == 'Structure'][['Year',
                                                             'Category',
                                                             'Value']].pivot(index='Year',
                                                                             columns='Category',
                                                                             values='Value').dropna(
                                                                                 axis=1, how='all')
                pnnl_structure.columns.name = None

                pnnl_effect = \
                    data_[data_['Category'] == 'Effect'][['Year',
                                                          'Category',
                                                          'Value']].pivot(index='Year',
                                                                          columns='Category',
                                                                          values='Value').dropna(
                                                                              axis=1, how='all')
                pnnl_effect.columns.name = None

                acceptable_bool_effect = self.utils.pct_diff(pnnl_effect, eii_effect)
                print(f"{level_} effect is {acceptable_bool_effect}")
                bools_list.append(acceptable_bool_effect)

                acceptable_bool_intensity = self.utils.pct_diff(pnnl_intensity, eii_intensity)
                print(f"{level_} intensity is {acceptable_bool_intensity}")
                bools_list.append(acceptable_bool_intensity)

                acceptable_bool_activity = self.utils.pct_diff(pnnl_activity, eii_activity)
                print(f"{level_} activity is {acceptable_bool_activity}")
                bools_list.append(acceptable_bool_activity)

                acceptable_bool_structure = self.utils.pct_diff(eii_structure, eii_structure)
                print(f"{level_} structure is {acceptable_bool_structure}")
                bools_list.append(acceptable_bool_structure)

        assert all(bools_list)


if __name__ == '__main__':
    test = TestLMDI()
    data = test.test_build_nest('transportation')
    #print(data)

