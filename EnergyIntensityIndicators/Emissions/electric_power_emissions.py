
import pandas as pd
import numpy as np
import os

from EnergyIntensityIndicators.electricity import ElectricityIndicators
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


class ElectricPowerEmissions(CO2EmissionsDecomposition):
    """Class to decompose changes in Emissions from the electric
    power sector
    """
    def __init__(self, directory, output_directory, level_of_aggregation):
        self.directory = directory
        self.output_directory = output_directory
        self.level_of_aggregation = level_of_aggregation
        fname = 'electric_power_sector_emissions'
        fossil_fuels = {'Coal': None,
                        'Petroleum': None,
                        'Natural Gas': None,
                        'Other Gasses': None}
        wood_waste = {'Wood': None,
                       'Waste': None}
        self.sub_categories_list = \
            {'Elec Generation Total':
                {'Elec Power Sector':
                    {'Electricity Only':
                        {'Fossil Fuels':
                            fossil_fuels,
                         'Nuclear': None,
                         'Hydroelectric': None,
                         'Renewable':
                            {'Wood': None,
                             'Waste': None,
                             'Geothermal': None,
                             'Solar': None,
                             'Wind': None}},
                     'Combined Heat & Power':
                        {'Fossil Fuels':
                            fossil_fuels,
                         'Renewable':
                            wood_waste}},
                 'Commercial Sector': None,
                 'Industrial Sector': None},
             'All CHP':
                {'Elec Power Sector':
                    {'Combined Heat & Power':
                        {'Fossil Fuels':
                            fossil_fuels,
                         'Renewable':
                            wood_waste,
                         'Other': None}},
                 'Commercial Sector':
                    {'Combined Heat & Power':
                        {'Fossil Fuels':
                            fossil_fuels,
                         'Hydroelectric': None,
                         'Renewable':
                            wood_waste,
                         'Other': None}},
                 'Industrial Sector':
                    {'Combined Heat & Power':
                        {'Fossil Fuels':
                            fossil_fuels,
                         'Hydroelectric': None,
                         'Renewable':
                            wood_waste,
                         'Other': None}}}}

        super().__init__(self.directory,
                         self.output_directory,
                         sector='Electric Power',
                         level_of_aggregation=self.level_of_aggregation,
                         fname=fname,
                         categories_dict=self.sub_categories_list)
        self.elec_data = \
            ElectricityIndicators(directory=self.directory,
                                  output_directory=self.output_directory,
                                  level_of_aggregation='Electric Power',
                                  end_year=2018).collect_data()

    def electric_power_co2(self):
        """[summary]

        Returns:
            [type]: [description]
        """

        elec_gen_total = \
            self.elec_data['Elec Generation Total']
        elec_power_sector = elec_gen_total['Elec Power Sector']
        elec_only = elec_power_sector['Electricity Only']

        comm_sector = elec_gen_total['Commercial Sector']
        ind_sector = elec_gen_total['Industrial Sector']

        all_chp = self.elec_data['All CHP']
        chp_elec_power = all_chp['Elec Power Sector']['Combined Heat & Power']
        chp_comm = all_chp['Commercial Sector']['Combined Heat & Power']
        chp_ind = all_chp['Industrial Sector']['Combined Heat & Power']

        data_cats = [elec_only, comm_sector,
                     ind_sector, chp_elec_power,
                     chp_comm, chp_ind]

        emissions_data = dict()
        for d in data_cats:
            print('d:\n', d)
            for c in d.keys():
                print('c:', c)
                try:
                    activity = d[c]['activity']
                except KeyError:
                    print('d[c] keys', d[c].keys())
                print('activity:\n', activity)
                try:
                    energy = d[c]['energy']
                except KeyError:
                    print('d[c] keys', d[c].keys())
                print('energy:\n', energy)
                for e, e_df in energy.items():
                    print('e:', e)
                    # d = d.rename(columns=self.electric_power_sector())
                    print('e_df.columns:', e_df.columns)
                    no_emissions = ['Solar', 'Wind',
                                    'Nuclear', 'Geothermal',
                                    'Hydroelectric']
                    rename_ = dict()
                    for type_ in no_emissions:
                        cols = {c: type_ for c in e_df.columns if type_ in c}
                        rename_.update(cols)

                    e_df = e_df.rename(columns=rename_)

                    print('e_df:\n', e_df)
                    d_emissions = \
                        self.calculate_emissions(e_df,
                                                 emissions_type='CO2 Factor',
                                                 datasource='eia_elec')
                    print('d_emissions:\n', d_emissions)

                    emissions_data[c] = d_emissions
        print('emissions_data:\n', emissions_data)
        return emissions_data

    def main(self):

        return self.electric_power_co2()


if __name__ == '__main__':
    directory = './EnergyIntensityIndicators/Data'
    output_directory = './Results'

    module_dict = {'elec': ElectricPowerEmissions}
    levels = {'elec': 'Elec Generation Total'}
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
