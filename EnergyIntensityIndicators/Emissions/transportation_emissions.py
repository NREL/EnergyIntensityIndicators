
import pandas as pd
import numpy as np
import os

from EnergyIntensityIndicators.transportation import TransportationIndicators
from EnergyIntensityIndicators.LMDI import CalculateLMDI
# from EnergyIntensityIndicators.economy_wide import EconomyWide
from EnergyIntensityIndicators.pull_eia_api import GetEIAData
from EnergyIntensityIndicators.utilities.dataframe_utilities \
    import DFUtilities as df_utils
from EnergyIntensityIndicators.utilities.standard_interpolation \
    import standard_interpolation
from EnergyIntensityIndicators.lmdi_gen import GeneralLMDI
from EnergyIntensityIndicators.Emissions.co2_emissions \
    import SEDSEmissionsData, CO2EmissionsDecomposition


class TransportationEmssions(CO2EmissionsDecomposition):
    def __init__(self, directory, output_directory, level_of_aggregation):
        fname = 'transportation_emissions'
        config_path = f'C:/Users/irabidea/Desktop/yamls/{fname}.yaml'

        self.sub_categories_list = \
            {'All_Transportation':
                {'All_Passenger':
                    {'Highway':
                        {'Passenger Cars and Trucks':
                            {'Passenger Car – SWB Vehicles':
                                {'Passenger Car': None,
                                 'SWB Vehicles': None},
                             'Light Trucks – LWB Vehicles':
                                {'Light Trucks': None,
                                 'LWB Vehicles': None},
                             'Motorcycles': None},
                         'Buses':
                            {'Urban Bus': None,
                             'Intercity Bus': None,
                             'School Bus': None},
                         'Paratransit':
                            None},
                     'Rail':
                        {'Urban Rail':
                            {'Commuter Rail': None,
                             'Heavy Rail': None,
                             'Light Rail': None},
                         'Intercity Rail': None},
                     'Air':
                        {'Commercial Carriers': None,
                         'General Aviation': None}},
                 'All_Freight':
                    {'Highway':
                        {'Single-Unit Truck': None,
                         'Combination Truck': None},
                     'Rail': None,
                     'Air': None,
                     'Waterborne': None,
                     'Pipeline':
                        {'Oil Pipeline': None,
                         'Natural Gas Pipeline': None}}}}

        super().__init__(directory, output_directory,
                         sector='Transportation',
                         config_path=config_path,
                         level_of_aggregation=level_of_aggregation,
                         categories_dict=self.sub_categories_list)
        self.transport_data = \
            TransportationIndicators(directory=directory,
                                     output_directory=output_directory,
                                     level_of_aggregation=level_of_aggregation,
                                     lmdi_model=self.lmdi_models,
                                     base_year=self.base_year,
                                     end_year=self.end_year)

    @staticmethod
    def transportation_data():
        """[summary]

        Returns:
            [type]: [description]
        """
        tedb_18 = \
            pd.read_excel(
                "https://tedb.ornl.gov/wp-content/uploads/2021/02/Table2_07_01312021.xlsx",
                skiprows=9, skipfooter=10, index_col=0, usecols='B:J')
        tedb_18 = tedb_18.rename(columns={'Electricityb': 'Electricity',
                                          'Totalc': 'Total'})
        tedb_18.index = tedb_18.index.str.strip()
        tedb_18 = tedb_18.reset_index()
        print(tedb_18)
        categories = ['HIGHWAY', 'TOTAL HWY & NONHWYc',
                      'Air', 'Rail', 'Pipeline', 'Water']  # 'NONHIGHWAY',
        conditions = [(tedb_18['index'] == r) for r in categories]
        tedb_18.loc[:, 'Category'] = np.select(conditions, categories)
        tedb_18.loc[:, 'Category'] = \
            tedb_18['Category'].replace(to_replace='0',
                                        value=np.nan).fillna(method='ffill')
        tedb_18 = tedb_18[~tedb_18['index'].isin(categories)]
        tedb_18 = tedb_18.rename(columns={'index':
                                          'Mode',
                                          ' Residual fuel oil':
                                          'Residual fuel oil'})
        print(tedb_18.columns)
        tedb_fuel_types = ['Gasoline', 'Diesel fuel', 'Liquefied petroleum gas',
                           'Jet fuel',  'Residual fuel oil', 'Natural gas',
                           'Electricity', 'Total']

        tedb_fuel = pd.melt(tedb_18, id_vars=['Category', 'Mode'],
                            value_vars=tedb_fuel_types, var_name='Fuel Type')
        tedb_fuel.loc[:, 'Year'] = 2018
        tedb_fuel = tedb_fuel.replace({'HIGHWAY': 'Highway',
                                       'Water': 'Waterborne'})
        print(tedb_fuel)
        historical_fuel_consump = \
            pd.read_excel(
                './EnergyIntensityIndicators/Transportation/Data/FuelConsump.xlsx',
                skipfooter=196, skiprows=2, usecols='A:BQ')
        historical_fuel_consump = historical_fuel_consump.fillna(np.nan)
        historical_fuel_consump.loc[0:2, :] = \
            historical_fuel_consump.loc[0:2, :].ffill(axis=1)
        historical_fuel_consump.loc[0, 'Unnamed: 0'] = 'Category'
        historical_fuel_consump.loc[1, 'Unnamed: 0'] = 'Mode'

        historical_fuel_consump = \
            historical_fuel_consump.set_index('Unnamed: 0')

        historical_fuel_consump = historical_fuel_consump.transpose()
        historical_fuel_consump = \
            historical_fuel_consump.rename(columns={'Year': 'Fuel Type'})
        historical_fuel_consump = \
            historical_fuel_consump.reset_index().drop('index', axis=1)

        year_cols = \
            [c for c in historical_fuel_consump.columns if isinstance(c, int)]
        fuel = pd.melt(historical_fuel_consump, id_vars=['Category', 'Mode',
                                                         'Fuel Type'],
                       value_vars=year_cols)

        fuel = fuel.rename(columns={'Unnamed: 0': 'Year'})
        fuel = fuel[(fuel['Fuel Type'] != 'Year') &
                    (fuel['Mode'] != 'Not Used')]

        transport_fuel = pd.concat([fuel, tedb_fuel], axis=0)
        transport_fuel = transport_fuel.replace('Passenger Car ', 'Passenger Car')
        return transport_fuel

    def transport(self, data):
        sector_ = 'All_Transportation'
        all_data = data[sector_]
        energy_data = self.transportation_data()
        print('energy_data:\n', energy_data)

        all_data_dict = dict()

        for cargo, cargo_data in self.sub_categories_list[sector_].items():  # Passenger/Freight
            cargo_dict = dict()

            for category, category_data in cargo_data.items():  # Highway/Rail/etc.
                category_dict = dict()
                if category_data is None:
                    category_data_ = \
                        all_data[cargo][category]
                    mode_group_dict = \
                        self.wrap_data(
                            energy_data, category_data_,
                            category, category)

                elif isinstance(category_data, dict):
                    for mode_group, mode_group_data in category_data.items(): # 'Passenger Cars and Trucks'/Urban Rail etc
                        mode_group_dict = dict()
                        if mode_group_data is None:
                            mode_group_data_ = \
                                all_data[cargo][category][mode_group]
                            mode_group_dict = \
                                self.wrap_data(
                                    energy_data, mode_group_data_,
                                    mode_group, category)

                        elif isinstance(mode_group_data, dict):
                            for mode, mode_data in mode_group_data.items(): #'Passenger Car – SWB Vehicles', 'Motorcycles'
                                mode_dict = dict()
                                if mode_data is None:
                                    mode_data_ = \
                                        all_data[cargo][category][mode_group][mode]
                                    mode_dict = \
                                        self.wrap_data(
                                            energy_data, mode_data_,
                                            mode, category)

                                elif isinstance(mode_data, dict):
                                    for sub_mode, sub_mode_d in mode_data.items():
                                        if sub_mode_d is None:
                                            sub_mode_data = \
                                                all_data[cargo][category][mode_group][mode][sub_mode]
                                            sub_mode_dict = \
                                                self.wrap_data(
                                                    energy_data, sub_mode_data,
                                                    sub_mode, category)
                                            mode_dict[sub_mode] = sub_mode_dict

                        mode_group_dict[mode] = mode_dict
                    category_dict[mode_group] = mode_group_dict
                cargo_dict[category] = category_dict
            all_data_dict[cargo] = cargo_dict
        transportation_data = {sector_: all_data_dict}
        return transportation_data

    def wrap_data(self, energy_data, activity_data,
                  mode, category):
        wrapped_data = dict()
        wrapped_data['A_i_k'] = activity_data['activity']

        energy_data = \
            energy_data[
                (energy_data['Mode'] == mode) &
                (energy_data['Category'] == category)]

        energy = \
            energy_data.pivot(index='Year',
                              columns='Fuel Type',
                              values='value')

        emissions, energy = \
            self.calculate_emissions(energy,
                                     emissions_type='CO2 Factor',
                                     datasource='TEDB')

        wrapped_data['E_i_j_k'] = energy
        wrapped_data['C_i_j_k'] = emissions
        return wrapped_data

    def main(self):
        energy_data = self.transportation_data()
        print('energy_data:\n', energy_data)
        energy_decomp_data = \
            self.transport_data.collect_data(
                )
        transportation_data = self.transport(energy_decomp_data)
        exit()
        return transportation_data


if __name__ == '__main__':
    directory = './EnergyIntensityIndicators/Data'
    output_directory = './Results'

    module_ = TransportationEmssions
    level = 'All_Transportation'

    s = module_(directory, output_directory,
                level_of_aggregation=level)
    s_data = s.main()
    results = s.calc_lmdi(breakout=True,
                          calculate_lmdi=True,
                          data_dict=s_data)
    print('s_data:\n', s_data)
    print('results:\n', results)
