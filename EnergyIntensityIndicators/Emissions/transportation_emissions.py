
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
                         level_of_aggregation=level_of_aggregation,
                         fname=fname,
                         categories_dict=self.sub_categories_list)
        self.transport = \
            TransportationIndicators(directory=directory,
                                     output_directory=output_directory,
                                     level_of_aggregation=level_of_aggregation,
                                     lmdi_model=self.model,
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

        return transport_fuel

    def main(self):
        energy_data = self.transportation_data()
        print('transport energy_data:\n', energy_data)
        print('transport energy_data fuels:\n',
              energy_data['Fuel Type'].unique())

        energy_decomp_data = self.transport.collect_data()


        emissions = \
            self.calculate_emissions(energy_data,
                                     emissions_type='CO2 Factor',
                                     datasource='TEDB')
        return {'energy': energy_data,
                'emissions': emissions}


if __name__ == '__main__':
    directory = './EnergyIntensityIndicators/Data'
    output_directory = './Results'

    module_dict = {'transport': TransportationEmssions}
    levels = {'transport': 'All_Transportation'}
    results = dict()
    for sector, module_ in module_dict.items():
        print('sector:', sector)
        s = module_(directory, output_directory,
                    level_of_aggregation=levels[sector])
        s_data = s.main()
        results = s.calc_lmdi(breakout=True,
                              calculate_lmdi=True,
                              data_dict=s_data)
        print('s_data:\n', s_data)
        print('results:\n', results)

        results[sector] = s_data
