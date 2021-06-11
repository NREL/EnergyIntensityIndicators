
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
from EnergyIntensityIndicators.Industry.manufacturing \
    import Manufacturing


class IndustrialEmissions(CO2EmissionsDecomposition):
    """Class to decompose changes in Emissions
    from combustion and noncombustion sources
    in the Industrial Sector of the US Economy
    """
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
                    {'Food and beverage and tobacco products': {'combustion': None},
                     'Textile mills and textile product mills': {'combustion': None},
                     'Apparel and leather and allied products': {'combustion': None},
                     'Wood products': {'combustion': None},
                     'Paper products': {'combustion': None},
                     'Printing and related support activities': {'combustion': None},
                     'Petroleum and coal products': {'combustion': None},
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
                     'Plastics and rubber products': {'combustion': None},
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
                     'Machinery': {'combustion': None},
                     'Computer and electronic products': {'combustion': None},
                     'Electrical equipment, appliances, and components': {'combustion': None},
                     'Motor vehicles, bodies and trailers, and parts': {'combustion': None},
                     'Furniture and related products': {'combustion': None},
                     'Miscellaneous manufacturing': {'combustion': None}},
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

    def energy_data(self):
        """Collect energy consumtion by fuel type data for
        the Industrial Sector (organized by subcategory)

        Returns:
            data (dict): Nested dictionary containing energy
                         consumption by fuel type dataframes
        """
        all_manufacturing = self.manufacturing_energy_data()

        data_dir = './EnergyIntensityIndicators/Industry/Data/'
        construction_elec_fuels = \
            pd.read_csv(
                f'{data_dir}construction_elec_fuels.csv').set_index('Year')
        agriculture = \
            pd.read_excel(
                f'{data_dir}miranowski_data.xlsx',
                sheet_name='Ag Cons by Use', skiprows=4, skipfooter=50,
                usecols='A:F', index_col=0,
                names=['Year', 'Gasoline', 'Diesel', 'LP Gas',
                       'Natural Gas', 'Electricity'])

        # Mining
        mining = \
            pd.read_csv(
                f'{data_dir}mining_energy.csv')
        mining = mining.fillna(np.nan)
        mining = mining.dropna(how='all', axis=1)
        mining = mining[mining['NAICS'].notnull()]
        mining = mining.astype({'Year': int,
                                'NAICS': int})

        all_mining = []
        for n in mining['NAICS'].unique():
            mining_naics = mining[mining['NAICS'] == n]
            mining_naics = mining_naics.drop('NAICS', axis=1)
            mining_naics = mining_naics.set_index('Year')
            mining_naics = \
                mining_naics.apply(
                    lambda col: pd.to_numeric(col, errors='coerce'), axis=1)

            for c in mining_naics.columns:
                mining_naics = \
                    standard_interpolation(mining_naics,
                                           name_to_interp=c,
                                           axis=1)
                mining_naics['NAICS 4 Digit'] = int(str(n)[:4])
                all_mining.append(mining_naics)

        all_mining = pd.concat(all_mining, axis=0)
        all_mining = all_mining.reset_index()
        all_mining = all_mining.groupby(['Year', 'NAICS 4 Digit']).sum()

        all_mining = all_mining.reset_index()

        industry_names = \
            {2111: 'Petroleum and Natural Gas',
             2121: 'Coal Mining',
             2122: 'Metal Ore Mining',
             2123: 'Nonmetallic Mineral Mining and Quarrying',
             2131: 'Support Activities'}

        all_mining_data = dict()
        other_mining_data = []
        for number, name in industry_names.items():
            mining_df = all_mining[all_mining['NAICS 4 Digit'] == number]
            mining_df = mining_df.drop(['Total Fuel', 'NAICS 4 Digit'],
                                       axis=1,
                                       errors='ignore')
            if number in [2121, 2122, 2123]:
                other_mining_data.append(mining_df)
            else:
                mining_df = mining_df.set_index('Year')
                all_mining_data[name] = mining_df

        other_mining_data = pd.concat(other_mining_data, axis=0)
        other_mining_data = other_mining_data.groupby('Year').sum()
        all_mining_data['Other Mining'] = other_mining_data

        data = {'Manufacturing': all_manufacturing,
                'NonManufacturing':
                    {'Mining': all_mining_data,
                     'Construction': construction_elec_fuels,
                     'Agriculture, Forestry & Fishing': agriculture}}
        return data

    @staticmethod
    def agricultural_extrapolate(energy_df, gross_output):
        """Extend Miranowski data by fuel type (which ends in 2002)
        using the PNNL method based on energy intensity

        Args:
            energy_df (pd.DataFrame): Energy data for the agricultural
                                      sector
            gross_output (pd.DataFrame): Gross output data for the agricultural
                                         sector
        Returns:
            results (pd.DataFrame): Energy data for the agricultural
                                    sector extrapolated to end of gross
                                    output index
        """
        go_col = 'Agriculture, Forestry & Fishing'
        dfs = []
        for c in energy_df.columns:
            fuel_df = energy_df[[c]]
            fuel_df = fuel_df.merge(gross_output,
                                    left_index=True,
                                    right_index=True,
                                    how='outer')

            fuel_df['fuel_intensity'] = fuel_df[c].divide(
                gross_output[go_col] * 0.001, axis='index')

            fuel_df[c] = fuel_df['fuel_intensity'].multiply(
                gross_output[go_col] * 0.001, axis='index').ffill()
            dfs.append(fuel_df[c])

        results = pd.concat(dfs, axis=1)
        return results

    def manufacturing_energy_data(self):
        """Collect Manufacturing energy consumption
        by fuel type

        Returns:
            all_manufacturing (pd.DataFrame): Energy Consumption
                                              by fuel type and NAICS
                                              code
        """
        __, industrial_btu = \
            Manufacturing(naics_digits=3).mecs_data_by_year()
        industrial_btu = \
            industrial_btu[
                industrial_btu['region'] == 'Total United States']
        manufacturing = \
            industrial_btu.drop('region', axis=1, errors='ignore')
        manufacturing = \
            industrial_btu.drop('Total', axis=1, errors='ignore')
        manufacturing = manufacturing.dropna(how='all', axis=1)
        manufacturing = manufacturing.fillna(np.nan)
        manufacturing = manufacturing[
            (manufacturing['NAICS'].notnull()) & (manufacturing['NAICS'] != 'Total')
            & (manufacturing['NAICS'] != 'RSE Column Factors:')]

        all_manufacturing = []
        for n in manufacturing['NAICS'].unique():
            manufacturing_naics = manufacturing[manufacturing['NAICS'] == n]
            manufacturing_naics = manufacturing_naics.drop('NAICS', axis=1)
            manufacturing_naics = manufacturing_naics.set_index('Year')

            manufacturing_naics = \
                manufacturing_naics.apply(
                    lambda col: pd.to_numeric(col, errors='coerce'), axis=1)
            manufacturing_naics.index = manufacturing_naics.index.astype(int)
            manufacturing_naics = manufacturing_naics.sort_index()
            for c in manufacturing_naics.columns:
                manufacturing_naics = \
                    standard_interpolation(manufacturing_naics,
                                           name_to_interp=c,
                                           axis=1)

            manufacturing_naics['NAICS'] = n
            all_manufacturing.append(manufacturing_naics)
        all_manufacturing = pd.concat(all_manufacturing, axis=0)

        return all_manufacturing

    def collect_manufacturing_data(self, energy_data, noncombustion_data,
                                   manufacturing):
        """CCollect noncombustion data and
        data from energy decomposition,
        calculate emissions and return ammended
        dictionary for the Manufacturing Sector

        Args:
            energy_data (pd.DataFrame): Energy data for the manufacturing sector
                                        by fuel type and NAICS
            noncombustion_data (dict): Nested dictionary containing noncombustion
                                       activity and emissions data
            manufacturing (dict): Manufacturing data from energy decomposition

        Raises:
            ValueError: NAICS code missing from energy_data

        Returns:
            manufacturing_dict (dict): Nested dictionary containing all manufacturing
                                       combustion (with energy by fuel type) and
                                       noncombustion data
        """
        man = self.sub_categories_list['Industry']['Manufacturing']
        manufacturing_dict = dict()
        labels_naics = \
            dict((value, key) for key, value in self.naics_labels.items())
        for label in man.keys():
            naics = labels_naics[label]
            print('naics:\n', naics)
            naics_dict = dict()
            if '-' in naics:
                naics_list = naics.split('-')
                naics_list = [int(n) for n in naics_list]
                n_energy_data = \
                    energy_data[energy_data['NAICS'].isin(naics_list)]
                n_energy_data = n_energy_data.reset_index()
                combustion_energy_data = n_energy_data.groupby('Year').sum()
            else:
                combustion_energy_data = \
                    energy_data[energy_data['NAICS'] == int(naics)]

                if combustion_energy_data.empty:
                    print('energy_data:\n', energy_data)
                    raise ValueError(f'energy_data missing naics code {naics}')

            combustion_energy_data = \
                combustion_energy_data.drop(
                    ['NAICS', 'index'], axis=1, errors='ignore')
            combustion_activity_naics = \
                manufacturing[label]['activity']
            gross_output = combustion_activity_naics['gross_output']
            value_added = combustion_activity_naics['value_added']
            naics_emissions, combustion_energy_data = \
                self.calculate_emissions(combustion_energy_data,
                                         emissions_type='CO2 Factor',
                                         datasource='MECS')

            naics_dict['combustion'] = {'E_i_j_k': combustion_energy_data,
                                        'A_i_k': gross_output,
                                        'V_i_k': value_added,
                                        'C_i_j_k': naics_emissions}
            if 'noncombustion' in man[label]:
                noncombustion_activity, noncombustion_emissions = \
                    self.handle_noncombustion(
                        s_data=man[label],
                        noncombustion_data=noncombustion_data,
                        sub_category=label)

                naics_dict['noncombustion'] = \
                    {'A_i_k': noncombustion_activity,
                        'C_i_j_k': noncombustion_emissions}

            manufacturing_dict[label] = naics_dict

        return manufacturing_dict

    def collect_nonmanufacturing_data(self, energy_data, nonman_data,
                                      noncombustion_data):
        """Collect noncombustion data and
        data from energy decomposition,
        calculate emissions and return ammended
        dictionary for the Non-Manufacturing Sector

        Args:
            energy_data (dict): Nested dictionary of NonManufacturing
                                energy consumption by fuel type data
            nonman_data (dict): Nested dictionary of NonManufacturing
                                energy decomposition input data
            noncombustion_data (dict): Nested dictionary containing
                                       noncombustion activity and
                                       emissions data

        Returns:
            nonmanufacturing_dict (dict): Nested dictionary containing all
                                          nonmanufacturing combustion
                                          (with energy by fuel type) and
                                          noncombustion data
        """
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
                if subcategory == 'Agriculture, Forestry & Fishing':
                    sub_energy_data_combustion = \
                        self.agricultural_extrapolate(
                            sub_energy_data_combustion,
                            gross_output)
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

                nonmanufacturing_dict[subcategory] = subcategory_dict

            elif subcategory == 'Mining':
                mining_dict = dict()
                s_data = cats[subcategory]
                combustion_energy = \
                    energy_data[subcategory]
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
                        gross_output = combustion_activity['gross_output']
                        value_added = combustion_activity['value_added']

                        combustion_energy = combustion_energy[lower]

                        combustion_emissions, combustion_energy = \
                            self.calculate_emissions(
                                combustion_energy,
                                emissions_type='CO2 Factor',
                                datasource='MECS')

                        other_mining_dict['combustion'] = \
                            {'A_i_k': gross_output,
                             'V_i_k': value_added,
                             'C_i_j_k': combustion_emissions,
                             'E_i_j_k': combustion_energy}

                        mining_dict[lower] = other_mining_dict
                    else:
                        mining_combustion_activity = \
                            nonman_data[subcategory][lower]['activity']
                        mining_gross_output = \
                            mining_combustion_activity['gross_output']
                        mining_value_added = \
                            mining_combustion_activity['value_added']
                        mining_combustion_energy = \
                            energy_data[subcategory][lower]
                        mining_combustion_emissions, \
                            mining_combustion_energy = \
                                self.calculate_emissions(
                                                    mining_combustion_energy,
                                                    emissions_type='CO2 Factor',
                                                    datasource='MECS')
                        mining_dict[lower] = \
                            {'combustion':
                                {'A_i_k': mining_gross_output,
                                 'V_i_k': mining_value_added,
                                 'C_i_j_k': mining_combustion_emissions,
                                 'E_i_j_k': mining_combustion_energy}}

                    nonmanufacturing_dict[subcategory] = mining_dict

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
        """Merge noncombustion data into the sub_category level

        Args:
            s_data (dict): categories below subcategory
            noncombustion_data (dict): Nested dictionary.
                                       Keys are subcategories,
                                       inner dictionary keys are
                                       'activity' and 'emissions'
                                       with respective dataframes
                                       as values.
            sub_category (str): Subcategory to collect

        Raises:
            KeyError: noncombustion data missing emissions or
                      activity data

        Returns:
            noncombustion_activity (pd.DataFrame): sub-subcategory data
                                                   merged into one
                                                   subcategory activity df
            noncombustion_emissions (pd.DataFrame): sub-subcategory data
                                                   merged into one
                                                   subcategory emissions df
        """
        if s_data:
            noncombustion_activity = []
            noncombustion_emissions = []

            for s in s_data['noncombustion'].keys():
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
        """Collect noncombustion data and
        data from energy decomposition,
        calculate emissions and return ammended
        dictionary

        Returns:
            data (dict): data for emissions decomposition
        """
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
