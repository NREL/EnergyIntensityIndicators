
import pandas as pd
import numpy as np
import os

from pandas.core.algorithms import isin

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
from EnergyIntensityIndicators import DATADIR


class ElectricPowerEmissions(CO2EmissionsDecomposition):
    """Class to decompose changes in Emissions from the electric
    power sector
    """
    def __init__(self, directory, output_directory, level_of_aggregation):
        self.directory = directory
        self.output_directory = output_directory
        self.level_of_aggregation = level_of_aggregation
        fname = 'electric_power_sector_emissions'
        config_path = os.path.join(DATADIR, f'yamls/{fname}.yaml')
        fossil_fuels = {'Coal': None,
                        'Petroleum': None,
                        'Natural Gas': None,
                        'Other Gasses': None}
        renewables = {'Wood': None,
                      'Waste': None,
                      'Geothermal': None,
                      'Solar': None,
                      'Wind': None}
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
                            renewables},
                     'Combined Heat & Power':
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

        self.chp_cats = \
            {'All CHP':
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
                         config_path=config_path,
                         categories_dict=self.sub_categories_list)
        self.elec_data = \
            ElectricityIndicators(directory=self.directory,
                                  output_directory=self.output_directory,
                                  level_of_aggregation='Electric Power',
                                  end_year=2018).collect_data()

    def process_e_data(self, data_dict):
        """Clean level data and calculate emissions by fuel type
        from energy consumption by fuel type

        Args:
            data_dict dict): Dictionary of data from energy decomposition
                             (with 'activity' and 'energy' keys)

        Raises:
            TypeError: data_dict must by dictionary. If this
                       error is raised, it is likely that the
                       parsing of the energy decomposition data
                       in main() is not structured properly

        Returns:
            emissions_data (dict): Data for level with activity ('A_i_k'),
                                   energy ('E_i_k') and emissions ('C_i_k')
                                   keys (and respective DataFrames as values)
        """
        if isinstance(data_dict, dict):

            activity = data_dict['activity']
            if 'Year' in activity.columns:
                activity = activity.set_index('Year')
                if isinstance(activity.index[0], float):
                    activity.index = activity.index * 1000
                activity.index = activity.index.astype(int)
            activity = self.electric_epa_mapping(activity)

            energy = data_dict['energy']['primary']
            if 'Year' in energy.columns:
                if isinstance(energy.index[0], float):
                    energy.index = energy.index * 1000
                energy = energy.set_index('Year')
                energy.index = energy.index.astype(int)
        else:
            raise TypeError('data_dict is not dictionary')

        no_emissions = ['Solar', 'Wind',
                        'Nuclear', 'Geothermal',
                        'Hydroelectric']
        rename_ = dict()

        for type_ in no_emissions:
            cols = {c: type_ for c in energy.columns if type_ in c}
            rename_.update(cols)

        energy = energy.rename(columns=rename_)

        d_emissions, d_energy = \
            self.calculate_emissions(energy,
                                     emissions_type='CO2 Factor',
                                     datasource='eia_elec')

        emissions_data = {'E_i_j': d_energy,
                          'A_i': activity,
                          'C_i_j': d_emissions}

        return emissions_data

    def main(self):
        """Collect data from energy decomposition,
        calculate emissions and return ammended dictionary

        Returns:
            all_emissions_data (dict): data for emissions decomposition
        """

        elec_gen_total = \
            self.elec_data['Elec Generation Total']

        all_data_dict = dict()
        categories = self.sub_categories_list['Elec Generation Total']
        for sector, sector_dict in categories.items(): # electric power sector, Commercial, Industrial
            sector_data = dict()
            for gen_cat, gen_cat_dict in sector_dict.items(): ## elec only/chp
                get_cat_d = dict()
                if isinstance(gen_cat_dict, dict):
                    for gen_type, gen_data in gen_cat_dict.items(): # Fossil fuels, nuclear etc
                        category_dict = dict()
                        if isinstance(gen_data, dict):
                            for fuel_category, category_data in gen_data.items():  # wood, waste, etc
                                if isinstance(category_data, dict):
                                    raise ValueError('category data is dictionary')
                                else:  # category_data is None
                                    data = elec_gen_total[sector][gen_cat][gen_type][fuel_category]
                                    data = self.process_e_data(data)
                                    category_dict[fuel_category] = data

                            get_cat_d[gen_type] = category_dict
                        else:  #  gen_data is None
                            data = elec_gen_total[sector][gen_cat][gen_type]
                            data = self.process_e_data(data)
                            get_cat_d[gen_type] = data

                sector_data[gen_cat] = get_cat_d
            all_data_dict[sector] = sector_data

        all_emissions_data = {'Elec Generation Total': all_data_dict}
        return all_emissions_data


if __name__ == '__main__':
    directory = './EnergyIntensityIndicators/Data'
    output_directory = './Results'

    module_ = ElectricPowerEmissions
    level = 'Elec Generation Total'

    s = module_(directory, output_directory,
                level_of_aggregation=level)
    s_data = s.main()
    results = s.calc_lmdi(breakout=True,
                          calculate_lmdi=True,
                          data_dict=s_data)
    print('s_data:\n', s_data)
    print('results:\n', results)
