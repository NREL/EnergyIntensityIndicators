
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
            fname = 'C:/Users/irabidea/Desktop/yamls/combustion_noncombustion_test.yaml'
        elif level_of_aggregation == 'NonManufacturing':
            fname = 'C:/Users/irabidea/Desktop/yamls/combustion_noncombustion_test.yaml'
        elif level_of_aggregation == 'Industry':
            fname = 'C:/Users/irabidea/Desktop/yamls/total_industrial_emissions.yaml'
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
        self.naics_labels = \
            {'311-312': 'Food and beverage and tobacco products',
             '313-314': 'Textile mills and textile product mills',
             '315-316': 'Apparel and leather and allied products',
             '321': 'Wood products',
             '322': 'Paper products',
             '323': 'Printing and related support activities',
             '324': 'Petroleum and coal products',
             '325': 'Chemical products',
             '326': 'Plastics and rubber products',
             '327': 'Nonmetallic mineral products',
             '331': 'Primary metals',
             '332': 'Fabricated metal products',
             '333': 'Machinery',
             '334': 'Computer and electronic products',
             '335': 'Electrical equipment, appliances, and components',
             '336': 'Motor vehicles, bodies and trailers, and parts',
             '337': 'Furniture and related products',
             '339': 'Miscellaneous manufacturing'}

        super().__init__(directory, output_directory,
                         sector='Industry',
                         level_of_aggregation=level_of_aggregation,
                         config_path=fname,
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
                manufacturing_naics_interp = \
                    standard_interpolation(manufacturing_naics,
                                           name_to_interp=c,
                                           axis=1)
            manufacturing_naics_interp['NAICS'] = n
            all_manufacturing.append(manufacturing_naics_interp)

        all_manufacturing = pd.concat(all_manufacturing, axis=0)
        all_manufacturing = all_manufacturing.drop('Industry', axis=1)

        return {'Manufacturing': all_manufacturing,
                'NonManufacturing':
                    {'Mining': all_mining,
                     'Construction': construction_elec_fuels,
                     'Agriculture, Forestry & Fishing': agriculture}}

    def collect_manufacturing_data(self, energy_data, noncombustion_data,
                                   manufacturing):
        man = self.sub_categories_list['Industry']['Manufacturing']
        manufacturing_dict = dict()
        labels_naics = \
            dict((value, key) for key, value in self.naics_labels.items())
        for label in man.keys():
            naics = labels_naics[label]
            naics_dict = dict()
            if '-' in naics:
                naics_list = naics.split('-')
                naics_list = [int(n) for n in naics_list]
                n_energy_data = \
                    energy_data[energy_data['NAICS'].isin(naics_list)]
                combustion_energy_data = n_energy_data.groupby('Year').sum()
            else:
                combustion_energy_data = \
                    energy_data[energy_data['NAICS'] == int(naics)]
                if combustion_energy_data.empty:
                    print('energy_data:\n', energy_data)
                    raise ValueError(f'energy_data missing naics code {naics}')

            combustion_energy_data = \
                combustion_energy_data.drop(
                    'NAICS', axis=1, errors='ignore')
            combustion_activity_naics = \
                manufacturing[label]['activity']
            gross_output = combustion_activity_naics['gross_output']
            value_added = combustion_activity_naics['value_added']
            naics_emissions, combustion_energy_data = \
                self.calculate_emissions(combustion_energy_data,
                                         emissions_type='CO2 Factor',
                                         datasource='MECS')
            naics_dict['combustion'] = {'E_i_j': combustion_energy_data,
                                        'A_i_k': gross_output,
                                        'V_i_k': value_added,
                                        'C_i_j_k': naics_emissions}
            if man[label] is not None:
                noncombustion_activity, noncombustion_emissions = \
                    self.handle_noncombustion(
                        s_data=man[label],
                        noncombustion_data=noncombustion_data,
                        sub_category=label)

                naics_dict['noncombustion'] = \
                    {'A_i_k': noncombustion_activity,
                        'C_i_j_k': noncombustion_emissions}

            print('label:', label)
            print('label:\n', naics_dict)

            manufacturing_dict[label] = naics_dict

        return manufacturing_dict

    def collect_nonmanufacturing_data(self, energy_data, nonman_data,
                                      noncombustion_data):
        cats = self.sub_categories_list['Industry']['Nonmanufacturing']

        nonmanufacturing_dict = dict()
        for subcategory in cats.keys():
            subcategory_dict = dict()
            noncombustion_activity = []
            noncombustion_emissions = []

            if subcategory in \
                    ['Agriculture, Forestry & Fishing', 'Construction']:
                sub_energy_data_combustion = energy_data[subcategory]
                sub_activity_data_combustion = \
                    nonman_data[subcategory]['activity']
                value_added = sub_activity_data_combustion['value_added']
                gross_output = sub_activity_data_combustion['gross_output']

                sub_emissions_data_combustion, sub_energy_data_combustion = \
                    self.calculate_emissions(sub_energy_data_combustion,
                                             emissions_type='CO2 Factor',
                                             datasource='MECS')
                subcategory_dict['combustion'] = \
                    {'A_i_k': gross_output,
                     'V_i_k': value_added,
                     'E_i_j_k': sub_energy_data_combustion,
                     'C_i_j_k': sub_emissions_data_combustion}

                if subcategory == 'Agriculture, Forestry & Fishing':
                    sub_data_noncombustion = \
                        cats[subcategory]
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

                        s = s_data[lower]
                        noncombustion_activity, noncombustion_emissions = \
                            self.handle_noncombustion(s,
                                                      noncombustion_data,
                                                      lower)

                        other_mining_dict['noncombustion'] = \
                            {'A_i_k': noncombustion_activity,
                             'C_i_j_k': noncombustion_emissions}

                        combustion_activity = \
                            nonman_data[subcategory][lower]['activity']
                        combustion_energy = \
                            energy_data[subcategory]
                        combustion_energy = \
                            combustion_energy.drop(['Total Fuel', 'NAICS'],
                                                   axis=1, errors='ignore')
                        combustion_emissions, combustion_energy = \
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
                            nonman_data[subcategory][lower]['activity']
                        mining_combustion_energy = \
                            energy_data[subcategory]
                        mining_combustion_energy = \
                            mining_combustion_energy.drop(
                                ['Total Fuel', 'NAICS'],
                                 axis=1, errors='ignore')
                        mining_combustion_emissions, \
                            mining_combustion_energy = \
                                self.calculate_emissions(
                                                    mining_combustion_energy,
                                                    emissions_type='CO2 Factor',
                                                    datasource='MECS')
                        mining_dict[lower] = \
                            {'combustion':
                                {'A_i_k': mining_combustion_activity,
                                 'C_i_j_k': mining_combustion_emissions,
                                 'E_i_j_k': mining_combustion_energy}}

                    subcategory_dict[subcategory] = mining_dict

            else:
                s_data = cats[subcategory]
                noncombustion_activity, noncombustion_emissions = \
                    self.handle_noncombustion(s_data,
                                              noncombustion_data,
                                              subcategory)

                subcategory_dict['noncombustion'] = \
                    {'A_i_k': noncombustion_activity,
                        'C_i_j_k': noncombustion_emissions}
            nonmanufacturing_dict[subcategory] = subcategory_dict

        return nonmanufacturing_dict

    @staticmethod
    def handle_noncombustion(s_data, noncombustion_data,
                             sub_category):

        if s_data:
            noncombustion_activity = []
            noncombustion_emissions = []

            for s in s_data['noncombustion'].keys():
                print('s:', s)
                noncombustion_cat_data = noncombustion_data[s]

                e_ = noncombustion_cat_data['emissions']
                if isinstance(e_, list):
                    e_ = df_utils().merge_df_list(e_)
                e_ = e_.drop('Total', axis=1, errors='ignore')
                e_ = df_utils().create_total_column(e_, s)
                e_ = e_[[s]]
                noncombustion_emissions.append(e_)

                a_ = noncombustion_cat_data['activity']
                if isinstance(a_, list):
                    a_ = df_utils().merge_df_list(a_)
                print('a_:\n', a_)
                a_ = a_.drop('Total', axis=1, errors='ignore')
                a_ = df_utils().create_total_column(a_, s)
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
                raise KeyError(
                    'noncombustion_cat_data missing emissions or activity ' +
                    f'for subcategory {sub_category}')

        return noncombustion_activity, noncombustion_emissions

    def main(self):
        noncombustion_data = NonCombustion().main()

        combustion = \
            IndustrialIndicators(directory='./EnergyIntensityIndicators/Data',
                                 output_directory='./Results',
                                 level_of_aggregation='Industry',
                                 lmdi_model=self.lmdi_models,
                                 end_year=self.end_year,
                                 base_year=self.base_year)

        combustion_data = combustion.collect_data()['Industry']

        manufacturing_combustion = combustion_data['Manufacturing']
        nonmanufacturing_combustion = combustion_data['Nonmanufacturing']

        energy_data = self.energy_data()
        manufacturing_energy = energy_data['Manufacturing']
        nonmanufacturing_energy = energy_data['NonManufacturing']
        nonmanufacturing_data = \
            self.collect_nonmanufacturing_data(
                energy_data=nonmanufacturing_energy,
                nonman_data=nonmanufacturing_combustion,
                noncombustion_data=noncombustion_data)

        manufacturing_data = \
            self.collect_manufacturing_data(
                energy_data=manufacturing_energy,
                noncombustion_data=noncombustion_data,
                manufacturing=manufacturing_combustion)

        data = {'Industry':
                  {'Nonmanufacturing':
                      nonmanufacturing_data,
                   'Manufacturing':
                      manufacturing_data}}
        return data


if __name__ == '__main__':
    directory = './EnergyIntensityIndicators/Data'
    output_directory = './Results'

    module_ = IndustrialEmissions
    level = 'Industry'

    s = module_(directory, output_directory,
                level_of_aggregation=level)
    s_data = s.main()
    results = s.calc_lmdi(breakout=True,
                          calculate_lmdi=True,
                          data_dict=s_data)
    print('s_data:\n', s_data)
    print('results:\n', results)
