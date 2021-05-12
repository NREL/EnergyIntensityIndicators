
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
from EnergyIntensityIndicators.Emissions.noncombustion \
    import NonCombustion
from EnergyIntensityIndicators.industry \
    import IndustrialIndicators
from EnergyIntensityIndicators.lmdi_gen import GeneralLMDI
from EnergyIntensityIndicators.Emissions.co2_emissions \
    import SEDSEmissionsData, CO2EmissionsDecomposition


class IndustrialEmissions(CO2EmissionsDecomposition):
    def __init__(self, directory, output_directory, level_of_aggregation):
        if level_of_aggregation == 'Manufacturing':
            fname = 'combustion_noncombustion_test'
        elif level_of_aggregation == 'NonManufacturing':
            fname = 'combustion_noncombustion_test'
        elif level_of_aggregation == 'Industry':
            fname = 'industrial_emissions'
        self.sub_categories_list = \
            {'Industry':
                {'Manufacturing':
                    {'Food and beverage and tobacco products': None,
                     'Textile mills and textile product mills': None,
                     'Apparel and leather and allied products': None,
                     'Wood products': None,
                     'Paper products': None,
                     'Printing and related support activities': None,
                     'Petroleum and coal products': None,
                     'Chemical products':
                        {'noncombustion':
                            {'Petrochemical Production': None,
                             'Titanium Dioxide Production': None,
                             'Nitric Acid Production': None,
                             'Phosphoric Acid Production': None,
                             'Adipic Acid Production': None,
                             'Ammonia Production': None,
                             'Carbide Production and Consumption': None,
                             'Soda Ash Production': None,
                             'N2O from Product Uses': None,
                             'Urea Consumption for NonAgricultural Purposes':
                                None,
                             'Caprolactam, Glyoxal, and Glyoxylic Acid Production':
                                None},
                         'combustion': None},
                     'Plastics and rubber products': None,
                     'Nonmetallic mineral products':
                        {'noncombustion':
                            {'Cement Production': None,
                             'Glass Production': None,
                             'Lime Production': None,
                             'Other Process Uses of Carbonates': None,
                             'Carbon Dioxide Consumption': None},
                         'combustion': None},
                     'Primary metals':
                        {'noncombustion':
                            {'Lead Production': None,
                             'Zinc Production': None,
                             'Aluminum Production': None},
                         'combustion': None},
                     'Fabricated metal products':
                        {'noncombustion':
                            {'Ferroalloy Production': None,
                             'Metallurgical coke': None,
                             'Iron and Steel': None},
                         'combustion': None},
                     'Machinery': None,
                     'Computer and electronic products': None,
                     'Electrical equipment, appliances, and components': None,
                     'Motor vehicles, bodies and trailers, and parts': None,
                     'Furniture and related products': None,
                     'Miscellaneous manufacturing': None},
                 'Nonmanufacturing':
                    {'Agriculture, Forestry & Fishing':
                        {'noncombustion':
                            {'Urea Fertilization': None,
                             'Agricultural Soil Management': None,
                             'Manure Management': None,
                             'Enteric Fermentation': None,
                             'Liming': None},
                         'combustion': None},
                     'Mining':
                        {'Petroleum and Natural Gas':
                            {'combustion': None},
                         'Other Mining':
                            {'noncombustion':
                                {'Coal Mining': None},
                             'combustion': None},
                         'Support Activities':
                            {'combustion': None}},
                     'Construction':
                        {'combustion': None},
                     'Waste':
                        {'noncombustion':
                            {'Landfills': None,
                             'Composting': None}},
                     'Energy':
                        {'noncombustion':
                            {'Stationary Combustion': None,
                             'Non-Energy Use of Fuels': None}}}}}

        super().__init__(directory, output_directory,
                         sector='Industry',
                         level_of_aggregation=level_of_aggregation,
                         fname=fname,
                         categories_dict=self.sub_categories_list)

    @staticmethod
    def energy_data():
        data_dir = './EnergyIntensityIndicators/Industry/Data/'
        construction_elec_fuels = \
            pd.read_csv(
                f'{data_dir}construction_elec_fuels.csv').set_index('Year')
        agriculture = \
            pd.read_excel(
                f'{data_dir}miranowski_data.xlsx',
                sheet_name='Ag Cons by Use', skiprows=4, skipfooter=9,
                usecols='A:F', index_col=0,
                names=['Year', 'Gasoline', 'Diesel', 'LP Gas',
                       'Natural Gas', 'Electricity'])
        # Mining
        mining = \
            pd.read_csv(
                f'{data_dir}mining_energy.csv')
        print('mining:\n', mining)
        mining = mining.fillna(np.nan)
        mining = mining.dropna(how='all', axis=1)
        mining = mining[mining['NAICS'].notnull()]
        mining = mining.astype({'Year': int,
                                'NAICS': int})
        all_mining = []
        for n in mining['NAICS'].unique():
            mining_naics = mining[mining['NAICS'] == n]
            mining_naics = mining_naics.drop('NAICS', axis=1)
            mining_naics = mining_naics.set_index(['Year'])
            mining_naics = \
                mining_naics.apply(
                    lambda col: pd.to_numeric(col, errors='coerce'), axis=1)
            print('mining:\n', mining)

            for c in mining_naics.columns:
                mining_naics = \
                    standard_interpolation(mining_naics,
                                           name_to_interp=c,
                                           axis=1)
            mining_naics['NAICS'] = n
            all_mining.append(mining_naics)
        all_mining = pd.concat(all_mining, axis=0)

        manufacturing = pd.read_csv(
            f'{data_dir}mecs_table42.csv')
        print('manufacturing:\n', manufacturing)
        manufacturing = manufacturing.dropna(how='all', axis=1)
        manufacturing = manufacturing.fillna(np.nan)
        manufacturing = manufacturing[manufacturing['NAICS'].notnull()]
        manufacturing = manufacturing.astype({'Year': int,
                                              'NAICS': int})
        all_manufacturing = []
        for n in manufacturing['NAICS'].unique():
            manufacturing_naics = manufacturing[manufacturing['NAICS'] == n]
            manufacturing_naics = manufacturing_naics.drop('NAICS', axis=1)
            manufacturing_naics = manufacturing_naics.set_index(['Year'])
            manufacturing_naics = \
                manufacturing_naics.apply(
                    lambda col: pd.to_numeric(col, errors='coerce'), axis=1)

            print('manufacturing_naics:\n', manufacturing_naics)
            for c in manufacturing_naics.columns:
                manufacturing_naics = \
                    standard_interpolation(manufacturing_naics,
                                           name_to_interp=c,
                                           axis=1)
            manufacturing_naics['NAICS'] = n
            all_manufacturing.append(manufacturing_naics)

        all_manufacturing = pd.concat(all_manufacturing, axis=0)
        return {'Manufacturing': all_manufacturing,
                'NonManufacturing':
                    {'Mining': all_mining,
                     'Construction': construction_elec_fuels,
                     'Agriculture, Forestry & Fishing': agriculture}}

    def collect_manufacturing_data(self, energy_data, noncombustion_data,
                                   manufacturing, combustion_activity):
        man = self.sub_categories_list['Industry']['Manufacturing']
        combustion_activity_m = combustion_activity['Manufacturing']
        manufacturing_dict = dict()
        for naics in man.keys():
            if man[naics]:
                combustion_energy_data = \
                    energy_data[energy_data['NAICS'] == naics]
                combustion_activity_naics = \
                    combustion_activity_m[naics]
                naics_dict = dict()
                noncombustion_activity = []
                noncombustion_emissions = []
                naics_emissions = \
                    self.calculate_emissions(combustion_energy_data,
                                             emissions_type='CO2 Factor',
                                             datasource='MECS')
                naics_dict['combustion'] = {'E_i_j': combustion_energy_data,
                                            'A_i_k': combustion_activity_naics,
                                            'C_i_j_k': naics_emissions}

            elif not man[naics]:
                continue
            else:
                for sub_category in man[naics]['noncombustion'].keys():
                    noncombustion_cat_data = noncombustion_data[sub_category]
                    c_ = noncombustion_cat_data['emissions']
                    a_ = noncombustion_cat_data['activity']

                    c_ = \
                        df_utils.create_total_column(c_, sub_category)
                    c_ = c_[[sub_category]]
                    noncombustion_emissions.append(c_)
                    a_ = \
                        df_utils.create_total_column(a_, sub_category)
                    a_ = a_[[sub_category]]
                    noncombustion_activity.append(a_)

                noncombustion_activity = \
                    df_utils.merge_df_list(noncombustion_activity)
                noncombustion_emissions = \
                    df_utils.merge_df_list(noncombustion_emissions)

                naics_dict['noncombustion'] = \
                    {'A_i_k': noncombustion_activity,
                     'C_i_j_k': noncombustion_emissions}
                manufacturing_dict[naics] = naics_dict

    def collect_nonmanufacturing_data(self, energy_data, combustion_activity,
                                      nonman_data, noncombustion_data):
        cats = self.sub_categories_list['Industry']['Manufacturing']

        nonmanufacturing_dict = dict()
        for subcategory in cats.keys():
            subcategory_dict = dict()
            noncombustion_activity = []
            noncombustion_emissions = []

            if subcategory in \
                    ['Agriculture, Forestry & Fishing', 'Construction']:
                sub_energy_data_combustion = energy_data[subcategory]
                sub_activity_data_combustion = \
                   combustion_activity[subcategory]['activity']

                sub_emissions_data_combustion = \
                    self.calculate_emissions(sub_energy_data_combustion,
                                             emissions_type='CO2 Factor',
                                             datasource='MECS')
                subcategory_dict['combustion'] = \
                    {'A_i_k': sub_activity_data_combustion,
                     'E_i_k_j': sub_energy_data_combustion,
                     'C_i_j_k': sub_emissions_data_combustion}

                sub_data_noncombustion = \
                    nonman_data[subcategory]['noncombustion']

                # s_data = cats[subcategory]
                noncombustion_activity, noncombustion_emissions = \
                    self.handle_noncombustion(sub_data_noncombustion,
                                              noncombustion_data,
                                              subcategory)

                subcategory_dict['noncombustion'] = \
                    {'A_i_k': noncombustion_activity,
                     'C_i_j_k': noncombustion_emissions}

            elif subcategory == 'Mining':
                mining_dict = dict()
                s_data = cats[subcategory]
                for lower in s_data.keys():
                    if lower == 'Other Mining':
                        other_mining_dict = dict()

                        noncombustion = s_data[lower]['noncombustion']
                        noncombustion_activity = noncombustion['activity']
                        noncombustion_emissions = noncombustion['emissions']
                        other_mining_dict['noncombustion'] = \
                            {'A_i_k': noncombustion_activity,
                             'C_i_j_k': noncombustion_emissions}

                        combustion_activity = \
                            combustion_activity[subcategory][lower]['activity']
                        combustion_energy = \
                            energy_data[subcategory][[lower]]
                        combustion_emissions = \
                            self.calculate_emissions(
                                combustion_energy,
                                emissions_type='CO2 Factor',
                                datasource='MECS')
                        other_mining_dict['combustion'] = \
                            {'A_i_k': combustion_activity,
                             'C_i_j_k': combustion_emissions,
                             'E_i_j_k': combustion_energy}

                        mining_dict[lower] = other_mining_dict
                    else:
                        mining_combustion_activity = \
                            combustion_activity[subcategory][lower]['activity']
                        mining_combustion_energy = \
                            energy_data[subcategory][[lower]]
                        mining_combustion_emissions = \
                            self.calculate_emissions(
                                                mining_combustion_energy,
                                                emissions_type='CO2 Factor',
                                                datasource='MECS')
                        mining_dict[lower] = \
                            {'combustion':
                                {'A_i_k': mining_combustion_activity,
                                 'C_i_j_k': mining_combustion_emissions,
                                 'E_i_j_k': mining_combustion_energy}}
            else:
                s_data = cats[subcategory]
                noncombustion_activity, noncombustion_emissions = \
                    self.handle_noncombustion(s_data,
                                              noncombustion_data,
                                              subcategory)
                if noncombustion_activity is not None and \
                        noncombustion_emissions is not None:
                    subcategory_dict['noncombustion'] = \
                        {'A_i_k': noncombustion_activity,
                         'C_i_j_k': noncombustion_emissions}
                else:
                    pass

            nonmanufacturing_dict[subcategory] = subcategory_dict

    @staticmethod
    def handle_noncombustion(s_data, noncombustion_data,
                             sub_category):
        noncombustion_activity = []
        noncombustion_emissions = []
        if s_data:
            for s in s_data['noncombustion'].keys():
                print('s:', s)
                noncombustion_cat_data = noncombustion_data[s]
                e_ = noncombustion_cat_data['emissions']
                a_ = noncombustion_cat_data['activity']
                e_ = e_.drop('Total', axis=1, errors='ignore')
                if e_.empty:
                    return None, None
                e_ = \
                    df_utils().create_total_column(e_, s)
                e_ = e_[[s]]
                noncombustion_emissions.append(e_)
                print('a_:\n', a_)
                if a_.empty:
                    return None, None
                a_ = a_.drop('Total', axis=1, errors='ignore')
                a_ = \
                    df_utils().create_total_column(a_, s)
                print('a_:\n', a_)
                a_ = a_[[s]]
                noncombustion_activity.append(a_)

            noncombustion_activity = \
                df_utils().merge_df_list(noncombustion_activity)
            noncombustion_emissions = \
                df_utils().merge_df_list(noncombustion_emissions)
        else:
            if sub_category in noncombustion_data:
                noncombustion_cat_data = noncombustion_data[sub_category]
                noncombustion_emissions = noncombustion_cat_data['emissions']
                noncombustion_activity = noncombustion_cat_data['activity']
            else:
                return None, None

        return noncombustion_activity, noncombustion_emissions

    # @staticmethod
    # def handle_combustion():

    def main(self):
        noncombustion_data = NonCombustion().main()

        combustion = \
            IndustrialIndicators(directory='./EnergyIntensityIndicators/Data',
                                 output_directory='./Results',
                                 level_of_aggregation='Industry',
                                 lmdi_model=self.model,
                                 end_year=self.end_year,
                                 base_year=self.base_year)

        combustion_activity = combustion.collect_data()['Industry']
        manufacturing_combustion = combustion_activity['Manufacturing']
        nonmanufacturing_combustion = combustion_activity['Nonmanufacturing']

        energy_data = self.energy_data()

        manufacturing_data = \
            self.collect_manufacturing_data(energy_data['Manufacturing'],
                                            noncombustion_data,
                                            manufacturing_combustion,
                                            combustion_activity)
        nonmanufacturing_data = \
            self.collect_nonmanufacturing_data(energy_data['NonManufacturing'],
                                               combustion_activity,
                                               nonmanufacturing_combustion,
                                               noncombustion_data)

        data = {'Industry':
                  {'Nonmanufacturing':
                      nonmanufacturing_data,
                   'Manufacturing':
                      manufacturing_data}}
        return data


if __name__ == '__main__':
    directory = './EnergyIntensityIndicators/Data'
    output_directory = './Results'

    module_dict = {'industry': IndustrialEmissions}
    levels = {'industry': 'Industry'}
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
