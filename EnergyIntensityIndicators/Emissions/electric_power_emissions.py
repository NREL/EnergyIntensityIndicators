
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
        config_path = f'C:/Users/irabidea/Desktop/yamls/{fname}.yaml'
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
                         config_path=config_path,
                         categories_dict=self.sub_categories_list)
        self.elec_data = \
            ElectricityIndicators(directory=self.directory,
                                  output_directory=self.output_directory,
                                  level_of_aggregation='Electric Power',
                                  end_year=2018).collect_data()

    def process_e_data(self, data_dict):
        
        if isinstance(data_dict, dict):
            print('data_dict keys:', data_dict.keys())
            activity = data_dict['activity']
            activity = self.electric_epa_mapping(activity)
            energy = data_dict['energy']['primary']
            print('energy_:\n', energy)
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
        print('d_emissions:\n', d_emissions)
        print('d_energy:\n', d_energy)

        emissions_data = {'E_i_j': d_energy,
                          'A_i': activity,
                          'C_i_j': d_emissions}

        print('emissions_data:\n', emissions_data)
        return emissions_data

    def electric_power_co2(self):
        """[summary]

        Returns:
            [type]: [description]
        """

        elec_gen_total = \
            self.elec_data

        all_data_dict = dict()
        print("self.sub_categories_list keys:", self.sub_categories_list.keys())
        for sub, sub_dict in self.sub_categories_list.items(): # electric power sector, all_chp
            print('sub:', sub)
            print('sub_dict:', sub_dict)
            sub_data = dict()
            for gen_cat, gen_cat_dict in sub_dict.items():
                if isinstance(gen_cat_dict, dict):
                    for gen_type, gen_data in gen_cat_dict.items(): ## elec only, commercial, industrial
                        gen_dict = dict()
                        if isinstance(gen_data, dict):
                            for fuel_category, category_data in gen_data.items(): # Fossil fuels, nuclear etc
                                category_dict = dict()
                                if isinstance(category_data, dict):
                                    type_dict = dict()
                                    for fuel_type, type_data in category_data.items(): # wood, waste, etc
                                        if isinstance(type_data, dict):
                                            raise TypeError('Type data should be None')
                                        elif type_data is None:
                                            data = elec_gen_total[sub][gen_cat][gen_type][fuel_category][fuel_type]
                                            print('type_data:', type_data)
                                            data = self.process_e_data(data)
                                        type_dict[fuel_type] = data
                                    category_dict[fuel_category] = type_dict
                                elif category_data is None:
                                    data = elec_gen_total[sub][gen_cat][gen_type][fuel_category]
                                    print('category_data:', category_data)

                                    data = self.process_e_data(data)
                                    category_dict[fuel_category] = data
                            gen_dict[gen_type] = category_dict
                        elif gen_data is None:
                            data = elec_gen_total[sub][gen_cat][gen_type]
                            print('gen_data:', gen_data)

                            data = self.process_e_data(data)
                            gen_dict[gen_type] = data

                    sub_data[gen_cat] = gen_dict

                elif sub_dict is None:
                    data = elec_gen_total[sub]
                    data = self.process_e_data(data)
                    sub_data[gen_cat] = data

            all_data_dict[sub] = sub_data

        return all_data_dict

    def main(self):
        emissions_data = self.electric_power_co2()
        print('emissions_data:\n', emissions_data)
        print('emissions_data keys:\n', emissions_data.keys())
        for k in emissions_data.keys():
            print(f'emissions_data k keys for k {k}:\n', emissions_data[k].keys())
        return emissions_data


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
