
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
        """[summary]

        Args:
            data_dict ([type]): [description]

        Raises:
            TypeError: [description]

        Returns:
            [type]: [description]
        """
        if isinstance(data_dict, dict):
            print('data_dict keys:', data_dict.keys())
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
        if d_energy.empty:
            print('d_energy:\n', d_energy)
            exit()

        else:
            if d_emissions.empty:
                print('d_emissions:', d_emissions)
                exit()

        if activity.empty:
            print('activity:\n', activity)
            exit()

        return emissions_data
    
    def check_path(self, dict_):
        """[summary]

        Args:
            dict_ ([type]): [description]
        """
        paths = list(self.gen.get_paths(self.sub_categories_list))
        print('paths:', paths)
        paths_sorted = sorted(paths, key=len, reverse=True)
        print('paths_sorted:', paths_sorted)

        raw_data_paths = list(self.gen.get_paths(dict_))
        print('raw_data_paths paths:', raw_data_paths)
        raw_data_paths_sorted = sorted(raw_data_paths, key=len, reverse=True)
        # raw_data_paths_sorted = [p[:-1] for p in raw_data_paths_sorted]
        print('\n \n \n')
        print('raw_data_paths_sorted:', raw_data_paths_sorted)
        print('\n \n \n')
        for p in raw_data_paths_sorted:
            print('p:', p)
            print('\n')
        print('\n \n \n')
        missing_paths_raw = [p for p in paths_sorted if p not in raw_data_paths_sorted]
        print('missing_paths_raw:\n', missing_paths_raw)
        missing_paths_paths = [p for p in raw_data_paths_sorted if p not in paths_sorted]
        print('missing_paths_paths:\n', missing_paths_paths)
        exit()

    def test_nest(self, d):
        """[summary]

        Args:
            d ([type]): [description]
        """
        paths = list(self.gen.get_paths(d))
        variable = 'activity'
        end_paths = [p for p in paths if p[-1] is 'activity'] # or p[-1] is 'deliv']
        end_paths = sorted(end_paths, key=len, reverse=True)
        for p in end_paths:
            # data = self.gen.dict_iter(d, p, variable)
            print('p:', p[:-1])
            # if data.empty:
            #     print('data:\n', data)
        exit()

    def electric_power_co2(self):
        """[summary]

        Returns:
            [type]: [description]
        """

        elec_gen_total = \
            self.elec_data['Elec Generation Total']

        all_data_dict = dict()
        print("self.sub_categories_list keys:", self.sub_categories_list.keys())
        categories = self.sub_categories_list['Elec Generation Total']
        print('categories.keys()')
        for sector, sector_dict in categories.items(): # electric power sector, Commercial, Industrial
            print('sector:', sector)
            # print('sector_dict:', sector_dict)
            sector_data = dict()
            for gen_cat, gen_cat_dict in sector_dict.items(): ## elec only/chp
                print('  gen_cat:', gen_cat)
                # print('gen_cat_dict:', gen_cat_dict)
                get_cat_d = dict()
                if isinstance(gen_cat_dict, dict):
                    for gen_type, gen_data in gen_cat_dict.items(): # Fossil fuels, nuclear etc
                        print('    gen_type:', gen_type)
                        category_dict = dict()
                        if isinstance(gen_data, dict):
                            for fuel_category, category_data in gen_data.items():  # wood, waste, etc
                                print('      fuel_category:', fuel_category)
                                if isinstance(category_data, dict):
                                    raise ValueError('category data is dictionary')
                                else:  # category_data is None
                                    print('sector:', sector)
                                    print('gen_cat:', gen_cat)
                                    print('gen_type:', gen_type)
                                    print('fuel_category:', fuel_category)
                                    print(' elec_gen_total[sector] keys:', elec_gen_total[sector].keys())
                                    data = elec_gen_total[sector][gen_cat][gen_type][fuel_category]
                                    print('data:\n', data)
                                    data = self.process_e_data(data)
                                    category_dict[fuel_category] = data

                            get_cat_d[gen_type] = category_dict
                        else:  #  gen_data is None
                            print('sector:', sector)
                            print('gen_cat:', gen_cat)
                            print('gen_type:', gen_type)
                            data = elec_gen_total[sector][gen_cat][gen_type]
                            print('data:\n', data)
                            data = self.process_e_data(data)
                            get_cat_d[gen_type] = data

                sector_data[gen_cat] = get_cat_d
            all_data_dict[sector] = sector_data
        # for k in all_data_dict.keys():
        #     print('sector', k)
        #     for j in all_data_dict[k].keys():
        #         print('  subcats', j)
        #         for m in all_data_dict[k][j].keys():
        #             print('    fuel category:', m)
        #             if isinstance(all_data_dict[k][j][m], dict):
        #                 for f in all_data_dict[k][j][m].keys():
        #                     print('      fuel type:', f)

        # # self.test_nest(all_data_dict)
        # exit()
        all_emissions_data = {'Elec Generation Total': all_data_dict}
        return all_emissions_data

    def main(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        emissions_data = self.electric_power_co2()
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
