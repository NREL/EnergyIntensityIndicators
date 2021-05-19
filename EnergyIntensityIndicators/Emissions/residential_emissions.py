
import pandas as pd
import numpy as np
import os

from EnergyIntensityIndicators.LMDI import CalculateLMDI
# from EnergyIntensityIndicators.economy_wide import EconomyWide
from EnergyIntensityIndicators.pull_eia_api import GetEIAData
from EnergyIntensityIndicators.utilities.dataframe_utilities \
    import DFUtilities as df_utils
from EnergyIntensityIndicators.utilities.standard_interpolation \
    import standard_interpolation
from EnergyIntensityIndicators.residential \
    import ResidentialIndicators
from EnergyIntensityIndicators.lmdi_gen import GeneralLMDI
from EnergyIntensityIndicators.Emissions.co2_emissions \
    import SEDSEmissionsData, CO2EmissionsDecomposition


class ResidentialEmissions(SEDSEmissionsData):
    def __init__(self, directory, output_directory,
                 level_of_aggregation='National'):
        if level_of_aggregation == 'National':
            fname = 'residential_all_emissions'
        else:
            fname = 'residential_regional'
        config_path = f'C:/Users/irabidea/Desktop/yamls/{fname}.yaml'

        housing_types = \
            {'Single-Family': None,
             'Multi-Family': None,
             'Manufactured-Homes': None}

        self.sub_categories_list = \
            {'National':
                {'Northeast':
                    housing_types,
                 'Midwest':
                    housing_types,
                 'South':
                    housing_types,
                 'West':
                    housing_types}}

        super().__init__(directory, output_directory,
                         sector='Residential',
                         fname=config_path,
                         categories_dict=self.sub_categories_list,
                         level_of_aggregation=level_of_aggregation)

        self.res = \
            ResidentialIndicators(directory='./EnergyIntensityIndicators/Data',
                                  output_directory='./Results',
                                  level_of_aggregation=level_of_aggregation,
                                  lmdi_model=self.lmdi_models,
                                  end_year=self.end_year,
                                  base_year=self.base_year)

    def main(self):

        res_data = self.res.collect_data()['National']
        all_data = dict()

        energy_data = self.seds_energy_data(sector='residential')
        for r in res_data.keys():
            print('region:', r)
            r_activity = res_data[r]['activity']

            r_weather_factors = \
                res_data[r]['weather_factors']['elec']
            print('r_weather_factors:\n', r_weather_factors)

            r_energy = energy_data[r]
            r_energy = r_energy.drop('Census Region', axis=1)

            print('r_energy:\n', r_energy)
            print('r_energy:\n', r_energy.info())
            # exit()
            r_emissions, r_energy = \
                self.calculate_emissions(r_energy,
                                         emissions_type='CO2 Factor',
                                         datasource='SEDS')
            print('r_emissions:\n', r_emissions)
            print('r_activity:\n', r_activity)
            print('r_activity keys:\n', r_activity.keys())
            # activity_types = ['occupied_housing_units',
            #                   'floorspace_square_feet',
            #                   'household_size_square_feet_per_hu']
            # exit()
            r_data = {'E_i_j_k': r_energy,
                      'A_i_k': r_activity['occupied_housing_units'],
                      'C_i_j_k': r_emissions,
                      'WF_i': r_weather_factors}
            all_data[r] = r_data
            print('all_data:\n', all_data)
        data_dict = {'National': all_data}
        return data_dict


if __name__ == '__main__':
    directory = './EnergyIntensityIndicators/Data'
    output_directory = './Results'

    module_ = ResidentialEmissions
    level = 'National'

    s = module_(directory, output_directory,
                level_of_aggregation=level)

    s_data = s.main()
    results = s.calc_lmdi(breakout=True,
                          calculate_lmdi=True,
                          data_dict=s_data)
    print('s_data:\n', s_data)
    print('results:\n', results)
