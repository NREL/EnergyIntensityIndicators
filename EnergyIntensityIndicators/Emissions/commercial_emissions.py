
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
from EnergyIntensityIndicators.commercial \
    import CommercialIndicators
from EnergyIntensityIndicators.lmdi_gen import GeneralLMDI
from EnergyIntensityIndicators.Emissions.co2_emissions \
    import SEDSEmissionsData, CO2EmissionsDecomposition
from EnergyIntensityIndicators import DATADIR



class CommercialEmissions(SEDSEmissionsData):
    """Class to decompose changes in Emissions from the Commercial
    Sector
    """
    def __init__(self, directory, output_directory,
                 level_of_aggregation='Commercial_Total'):
        fname = os.path.join(DATADIR, 'yamls/commercial_total.yaml')
        self.level_of_aggregation = level_of_aggregation

        self.sub_categories_list = {'Commercial_Total': None}
        super().__init__(directory, output_directory,
                         sector='Commercial',
                         fname=fname,
                         categories_dict=self.sub_categories_list,
                         level_of_aggregation=self.level_of_aggregation)
        self.comm = \
            CommercialIndicators(
                directory='./EnergyIntensityIndicators/Data',
                output_directory='./Results',
                level_of_aggregation=self.level_of_aggregation,
                lmdi_model=self.lmdi_models,
                end_year=self.end_year,
                base_year=self.base_year)

    def main(self):
        """Collect data from energy decomposition,
        calculate emissions and return ammended dictionary

        Returns:
            data_dict (dict): data for emissions decomposition
        """

        comm_data = self.comm.collect_data()['Commercial_Total']
        weather_data = comm_data['weather_factors']

        energy_data = self.seds_energy_data(sector='commercial')['US']
        energy_data = energy_data.drop('Census Region', axis=1)

        emissions_data, energy_data = \
            self.calculate_emissions(energy_data,
                                     emissions_type='CO2 Factor',
                                     datasource='SEDS')

        activity = comm_data['activity']
        activity.index = activity.index.astype(int)
        activity_dict = {activity.columns[0]: activity}

        weather_factors = \
            self.collect_weather_data(comm_data['energy'],
                                      activity_dict,
                                      weather_data,
                                      total_label='Commercial_Total',
                                      weather_activity='floorspace_bsf',
                                      sector='Commercial')
        data_dict = {'Commercial_Total':
                        {'E_j': energy_data,
                         'C_j': emissions_data,
                         'WF': weather_factors,
                         'A': activity,
                         'total_label': 'Commerical'}}
        return data_dict


if __name__ == '__main__':
    directory = './EnergyIntensityIndicators/Data'
    output_directory = './Results'

    module_ = CommercialEmissions
    level = 'Commercial_Total'

    s = module_(directory, output_directory,
                level_of_aggregation=level)

    s_data = s.main()
    # results = s.calc_lmdi(breakout=True,
    #                       calculate_lmdi=True,
    #                       data_dict=s_data)
