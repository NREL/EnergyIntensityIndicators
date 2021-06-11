import pandas as pd
import numpy as np
from datetime import datetime
from functools import reduce

from EnergyIntensityIndicators.pull_bea_api import BEA_api
from EnergyIntensityIndicators.pull_eia_api import GetEIAData
from EnergyIntensityIndicators.utilities.dataframe_utilities \
    import DFUtilities as df_utils
from EnergyIntensityIndicators.Industry.asm_price_fit import Mfg_prices
from EnergyIntensityIndicators.utilities.standard_interpolation \
     import standard_interpolation


class Manufacturing:

    def __init__(self, naics_digits=3):
        self.eia = GetEIAData('Industry')
        self.end_year = datetime.now().year
        self.naics_digits = naics_digits

    def mecs_data_by_year(self):
        """[summary]

        Returns:
            mecs [type]: [description]
            industrial_btu [type]: [description]
        """        
        # Energy Consumption as a Fuel
        # Table 3.1 : By Mfg. Industry & Region (physical units)
        # Table 3.2 : By Mfg. Industry & Region (trillion Btu)
        # Table 3.5 : Byproducts in Fuel Consumption by Mfg.
        #             Industry & Region (trillion Btu)
        mecs_data = {2018:
                        {'table_3_1':
                            {'endpoint': 'Table3_1.xlsx',
                             'skiprows': 9,
                             'skip_footer': 20},
                         'table_3_2':
                            {'endpoint': 'Table3_2.xlsx',
                             'skiprows': 9,
                             'skip_footer': 14},
                         'table_3_5':
                            {'endpoint': 'Table3_5.xlsx',
                             'skiprows': 9,
                             'skip_footer': 12},
                         'table_4_2':
                            {'endpoint': 'Table4_2.xlsx',
                             'skiprows': 0,
                             'skip_footer': 13}},
                     2014:
                        {'table_3_1':
                            {'endpoint': 'table3_1.xlsx',
                             'skiprows': 9,
                             'skip_footer': 20},
                         'table_3_2':
                            {'endpoint': 'table3_2.xlsx',
                             'skiprows': 9,
                             'skip_footer': 14},
                         'table_3_5':
                            {'endpoint': 'table3_5.xlsx',
                             'skiprows': 9,
                             'skip_footer': 12},
                         'table_4_2':
                            {'endpoint': 'table4_2.xlsx',
                             'skiprows': 0,
                             'skip_footer': 13}},
                     2010:
                        {'table_3_1':
                            {'endpoint': 'Table3_1.xls',
                             'skiprows': 9,
                             'skip_footer': 47},
                         'table_3_2':
                            {'endpoint': 'Table3_2.xls',
                             'skiprows': 8,
                             'skip_footer': 47},
                         'table_3_5':
                            {'endpoint': 'Table3_5.xls',
                             'skiprows': 9,
                             'skip_footer': 29},
                         'table_4_2':
                            {'endpoint': 'Table4_2.xls',
                             'skiprows': 0,
                             'skip_footer': 34}},
                     2006:
                        {'table_3_1':
                            {'endpoint': 'Table3_1.xls',
                             'skiprows': 10,
                             'skip_footer': 49},
                         'table_3_2':
                            {'endpoint': 'Table3_2.xls',
                             'skiprows': 9,
                             'skip_footer': 49},
                         'table_3_5':
                            {'endpoint': 'Table3_5.xls',
                             'skiprows': 10,
                             'skip_footer': 31},
                         'table_4_2':
                            {'endpoint': 'Table4_2.xls',
                             'skiprows': 0,
                             'skip_footer': 36}},
                     2002:
                        {'table_3_1':
                            {'endpoint': 'Table3.1_02.xls',
                             'skiprows': 7,
                             'skip_footer': 49},
                         'table_3_2':
                            {'endpoint': 'Table3.2_02.xls',
                             'skiprows': 6,
                             'skip_footer': 49},
                         'table_3_5':
                            {'endpoint': 'Table3.5_02.xls',
                             'skiprows': 7,
                             'skip_footer': 55},
                         'table_4_2':
                            {'endpoint': 'Table4.2_02.xls',
                             'skiprows': 0,
                             'skip_footer': 36}},
                     1998:
                        {'table_3_1':
                            {'endpoint': 'd98n3_1.xls',
                             'skiprows': 7,
                             'skip_footer': 53},  # Fuel Consumption, 1998; Level: National and Regional Data; Row: NAICS Codes; Column: Energy Sources; Unit: Physical Units or Btu
                         'table_3_2':
                            {'endpoint': 'd98n3_2.xls',
                             'skiprows': 6,
                             'skip_footer': 53},  # Fuel Consumption, 1998; Level: National and Regional Data; Row: NAICS Codes; Column: Energy Sources; Unit: Trillion Btu
                         'table_3_5':
                            {'endpoint': 'd98n5_1.xls',
                             'skiprows': 7,
                             'skip_footer': 59},  # Selected Byproducts in Fuel Consumption, 1998; Level: National Data and Regional Totals; Row: NAICS Codes; Column: Energy Sources; Unit: Trillion Btu
                         'table_4_2':
                            {'endpoint': 'd98n4_2.xls',
                             'skiprows': 0,
                             'skip_footer': 40}},
                     1994:
                        {'table_3_1':
                            {'endpoint': 'm94_04a.xls',
                             'skiprows': 6,
                             'skip_footer': 34},
                         'table_3_2':
                            {'endpoint': 'm94_04b.xls',
                             'skiprows': 5,
                             'skip_footer': 34},
                         'table_3_5':
                            {'endpoint': 'm94_06.xls',
                             'skiprows': 7,
                             'skip_footer': 21},
                         'table_4_2':
                            {'endpoint': 'm94_05b.xls',
                             'skiprows': 0,
                             'skip_footer': 31}},
                     1991:
                        {'table_3_1':
                            {'endpoint': 'mecs04a.xls',
                             'skiprows': 6,
                             'skip_footer': 36},
                         'table_3_2':
                            {'endpoint': 'mecs04b.xls',
                             'skiprows': 5,
                             'skip_footer': 38},
                         'table_3_5':
                            None,
                         'table_4_2':
                            {'endpoint': 'mecs05b.xls',
                             'skiprows': 0,
                             'skip_footer': 34}},
                     1985:
                        {'table_3_1': None,
                         'table_3_2': None,
                         'table_3_5': None,
                         'table_4_2': None}}

        all_3_1 = []
        all_3_2 = []
        all_3_5 = []
        all_4_2 = []

        sic_3_1 = []
        sic_3_2 = []
        sic_3_5 = []
        sic_4_2 = []

        for year, table_dict in mecs_data.items():
            if year < 1998: 
                sic = True
            else:
                sic = False

            for t, t_dict in table_dict.items():
                if t_dict:
                    endpoint = t_dict['endpoint'] 
                    general_url = f'https://www.eia.gov/consumption/manufacturing/data/{year}/xls/{endpoint}'
                    general_df = pd.read_excel(general_url, index_col=0)
                    col_labels = general_df.loc[:'Code(a)'].tail(4)

                    index_label = \
                        general_df.iloc[general_df.index.get_loc('Code(a)')-1].name

                    col_labels = col_labels.apply(lambda c: c.str.cat(sep=' '), axis=0)
                    col_labels = col_labels.apply(lambda s: s.strip())
                    col_labels = \
                        col_labels.to_frame(name=index_label).transpose()

                    df = pd.read_excel(general_url, skiprows=t_dict['skiprows'],
                                       skipfooter=t_dict['skip_footer'], 
                                       index_col=0)
                    df = df.iloc[df.index.get_loc('Code(a)')+1:]

                    df.columns = col_labels.loc[index_label, :]
                    df.columns.name = None

                    df = df.dropna(axis=1, how='all')

                    df = \
                        df.rename(
                            columns={'Total (trillion Btu)':
                                        'Total',
                                     'Industry Group and Industry':
                                        'Subsector and Industry',
                                     'Industry Groups and Industry':
                                        'Subsector and Industry',
                                     'LPG and NGL(e) (million bbl)':
                                        'HGL (excluding natural gasoline)(e) (million bbl)',
                                     'LPG and NGL(e)':
                                        'HGL (excluding natural gasoline)(e)'})
                    df = df.drop(['RSE Row Factors', ''], axis=1, errors='ignore')

                    if t == 'table_3_1':
                        mecs_3_1 = \
                            self.clean_industrial_data(df,
                                                       table_3_1=True,
                                                       sic=sic)
                        mecs_3_1['Year'] = year

                        if sic:
                            sic_3_1.append(mecs_3_1)
                        else:
                            all_3_1.append(mecs_3_1)

                    elif t == 'table_3_5':
                        mecs_3_5 = self.clean_industrial_data(df, sic=sic)
                        mecs_3_5['Year'] = year
                        if sic:
                            sic_3_5.append(mecs_3_5)
                        else:
                            all_3_5.append(mecs_3_5)

                    elif t == 'table_3_2':
                        mecs_3_2 = self.clean_industrial_data(df, sic=sic)

                        mecs_3_2['Year'] = year

                        if sic:
                            sic_3_2.append(mecs_3_2)
                        else:
                            all_3_2.append(mecs_3_2)

                    elif t == 'table_4_2':
                        mecs_4_2 = self.clean_industrial_data(df, sic=sic)
                        mecs_4_2['Year'] = year

                        if sic:
                            sic_4_2.append(mecs_4_2)
                        else:
                            all_4_2.append(mecs_4_2)

        x_walk = self.mecs_sic_crosswalk()

        all_3_1 = pd.concat(all_3_1, axis=0).reset_index().drop(
            'Subsector and Industry', axis=1, errors='ignore')
        sic_3_1 = pd.concat(sic_3_1, axis=0).reset_index()
        sic_naics_3_1 = self.naics_to_sic(sic_3_1, x_walk)
        thousand_to_million = \
            ['Residual Fuel Oil (1000 bbls)',
             'Distillate Fuel Oil(c) (1000 bbls)',
             'LPG (1000 bbls)',
             'Coal (1000 short tons)',
             'Coke and Breeze (1000 short tons)']
        sic_naics_3_1.loc[:, thousand_to_million] = \
            sic_naics_3_1.loc[:, thousand_to_million].multiply(0.001)
        rename_unit_change = \
            {'Residual Fuel Oil (1000 bbls)': 'Residual Fuel Oil (million bbl)',
             'Distillate Fuel Oil(c) (1000 bbls)': 'Distillate Fuel Oil(c) (million bbl)',
             'Coal (1000 short tons)': 'Coal (million short tons)',
             'Coke and Breeze (1000 short tons)': 'Coke and Breeze (million short tons)',
             'Other(e) (trillion Btu)': 'Other(f) (trillion Btu)',
             'LPG (1000 bbls)': 'HGL (excluding natural gasoline)(e) (million bbl)'}
        sic_naics_3_1 = sic_naics_3_1.rename(columns=rename_unit_change)
        table_3_1 = pd.concat([sic_naics_3_1, all_3_1], axis=0)

        all_3_2 = pd.concat(all_3_2, axis=0).reset_index().drop(
            'Subsector and Industry', axis=1, errors='ignore')
        sic_3_2 = pd.concat(sic_3_2, axis=0).reset_index()
        sic_naics_3_2 = self.naics_to_sic(sic_3_2, x_walk)
        sic_naics_3_2 = \
            sic_naics_3_2.rename(
                columns={'LPG': 'HGL (excluding natural gasoline)(e)',
                         'Other(e)': 'Other(f)'})
        table_3_2 = pd.concat([sic_naics_3_2, all_3_2], axis=0)

        all_3_5 = pd.concat(all_3_5, axis=0).reset_index().drop(
            'Subsector and Industry', axis=1, errors='ignore')
        sic_3_5 = pd.concat(sic_3_5, axis=0).reset_index()
        sic_naics_3_5 = self.naics_to_sic(sic_3_5, x_walk)
        sic_naics_3_5 = \
            sic_naics_3_5.rename(
                columns={'Pulping Liquor':
                            'Pulping Liquor or Black Liquor',
                         'Waste Oils/Tars And Waste Materials': 
                            'Waste Oils/Tars and Waste Materials'})
        table_3_5 = pd.concat([sic_naics_3_5, all_3_5], axis=0)

        all_4_2 = pd.concat(all_4_2, axis=0).reset_index().drop(
            'Subsector and Industry', axis=1, errors='ignore')
        sic_4_2 = pd.concat(sic_4_2, axis=0).reset_index()
        sic_naics_4_2 = self.naics_to_sic(sic_4_2, x_walk)
        table_4_2 = pd.concat([sic_naics_4_2, all_4_2], axis=0)

        table_3_2_other = table_3_2[['Year', 'region', 'NAICS',
                                     'Other(f)']]

        table_3_5 = table_3_5.merge(table_3_2_other, on=['Year', 'region', 'NAICS'],
                                    how='inner')

        table_3_5['steam'] = table_3_5['Other(f)'].subtract(table_3_5['Total'])
        table_3_5 = table_3_5.drop(['Total', 'Other(f)'], axis=1)
        industrial_btu = table_3_5.merge(table_3_2, on=['Year', 'region', 'NAICS'],
                                         how='outer')
        industrial_btu = industrial_btu.drop('Other(f)', axis=1)

        table_3_1 = self.set_n_naics_digits(table_3_1)
        table_3_2 = self.set_n_naics_digits(table_3_2)
        table_3_5 = self.set_n_naics_digits(table_3_5)
        table_4_2 = self.set_n_naics_digits(table_4_2)
        industrial_btu = self.set_n_naics_digits(industrial_btu)

        mecs = {'3_1': table_3_1, '3_2': table_3_2,
                '3_5': table_3_5, '4_2': table_4_2}
        return mecs, industrial_btu

    def set_n_naics_digits(self, table):
        """[summary]

        Args:
            table ([type]): [description]

        Returns:
            table (pd.DataFrame): [description]
        """
        table = table[table['NAICS'] != 'Total']

        table['NAICS'] = \
            table['NAICS'].astype(str).str[:self.naics_digits]

        table = table[(table['NAICS'] != 'nan') & (table['NAICS'] != 'RSE')]

        table['NAICS'] = table['NAICS'].astype(int)

        cols = [i for i in table.columns if i not in ['Year', 'region', 'NAICS']]
        for col in cols:
            table[col] = pd.to_numeric(table[col])

        table = \
            table.groupby(by=['Year', 'region', 'NAICS']).sum()

        table = table.reset_index()
        return table

    def clean_industrial_data(self, raw_data, table_3_1=False, sic=False):
        """[summary]

        Args:
            raw_data ([type]): [description]
            table_3_1 (bool, optional): [description]. Defaults to False.
            sic (bool, optional): [description]. Defaults to False.

        Returns:
           raw_data [type]: [description]
        """
        if sic:
            code = 'SIC'
        else:
            code = 'NAICS'

        if table_3_1:
            raw_data.index = raw_data.index.str.strip()

        else:
            raw_data.index = raw_data.index.fillna(np.nan)
            raw_data.index = raw_data.index.str.strip()
            raw_data.index.name = code

        raw_data = raw_data.reset_index()
        regions = ['Total United States', 'Northeast Census Region',
                   'Midwest Census Region', 'South Census Region',
                   'West Census Region']
        conditions = [(raw_data['Total'] == r) for r in regions]
        raw_data['region'] = np.select(conditions, regions)
        raw_data['region'] = raw_data['region'].replace(to_replace='0',
                                                        value=np.nan).fillna(method='ffill')
        raw_data = raw_data[~raw_data['Total'].isin(regions)]
        raw_data = raw_data.dropna(axis=0, how='all')

        raw_data[code] = raw_data[code].fillna(
            raw_data['Subsector and Industry'])
        raw_data = raw_data.set_index(
            ['region', code, 'Subsector and Industry'])
        raw_data = \
            raw_data.applymap(
                lambda x: x.strip() if isinstance(x, str) else x)
        raw_data = raw_data.replace({'*': 0.25, 'Q': np.nan,
                                     'D': np.nan, 'W': np.nan})

        rename_dict = {'Electricity(b)': 'Net Electricity',
                       'Fuel Oil': 'Residual Fuel Oil',
                       'Fuel Oil(c)': 'Distillate Fuel Oil',
                       'Gas(d)': 'Natural Gas',
                       'natural gasoline)(e)': 'HGL (excluding natural gasoline)',
                       'and Breeze': 'Coke Coal and Breeze',
                       'Oven Gases': 'Blast Furnace/Coke Oven Gases',
                        'Gas': 'Waste Gas',
                        'Coke': 'Petroleum Coke',
                        'Black Liquor': 'Pulping Liquor or Black Liquor',
                        'Bark': 'Wood Chips, Bark',
                        'Materials': 'Waste Oils/Tars and Waste Materials'}
        raw_data = raw_data.rename(columns=rename_dict)

        return raw_data

    @staticmethod
    def mecs_sic_crosswalk(
            data_dir='./EnergyIntensityIndicators/Industry/Data/'):
        """[summary]

        Returns:
           cw [pd.DataFrame]: [description]
        """
        #  Use crosswalk 1987 SIC to 1997 NAICS from
        #  https://www.census.gov/eos/www/naics/concordances/concordances.html
        # cw: 'https://www.census.gov/eos/www/naics/concordances/1987_SIC_to_1997_NAICS.xls'
        cw = pd.read_excel(f'{data_dir}1987_SIC_to_1997_NAICS.xlsx')
        cw = cw.astype(int, errors='ignore')
        cw = cw[['SIC', '1997 NAICS']]
        cw = cw.drop_duplicates()
        cw['SIC Count'] = cw.groupby('SIC')['SIC'].transform('count')
        cw['SIC Allocation Ratio'] = cw['SIC Count'].apply(lambda x: 1/x)
        print('cw:\n', cw)
        return cw

    def create_historical_mecs_31_32(self):
        """[summary]

        Returns:
           historical_mecs_31_32 [type]: [description]
           mecs_fuel [type]: [description]
        """
        mecs = self.mecs_data_by_year()
        mecs_3_1 = mecs['3_1'][['Year',
                                'region',
                                'NAICS',
                                'Net Electricity(b) (million kWh)']]
        mecs_3_2 = mecs['3_2'][['Year',
                                'region',
                                'NAICS',
                                'Total',
                                'Net Electricity(b)']]
        historical_mecs_31_32 = mecs_3_1.merge(mecs_3_2,
                                               on=['Year',
                                                   'region',
                                                   'NAICS'],
                                               how='outer')
        mecs_fuel = mecs_3_2.copy()
        mecs_fuel['Fuel'] = \
            mecs_fuel['Total'].subtract(mecs_fuel['Net Electricity(b)'])
        mecs_fuel = mecs_fuel.drop(['Total', 'Net Electricity(b)'], axis=1)
        return historical_mecs_31_32, mecs_fuel

    def naics_to_sic(self, sic_data, cw):
        """[summary]

        Args:
            sic_data ([type]): [description]
            cw ([type]): [description]
        """
        sic_data = sic_data[~sic_data['SIC'].isnull()]
        sic_data = sic_data[sic_data['SIC'].str.isnumeric()]
        sic_data['SIC'] = sic_data['SIC'].astype(int)
        cw = cw[pd.notna(cw['1997 NAICS'])]

        cw['1997 NAICS'] = cw['1997 NAICS'].astype(int)

        sic_data = sic_data.merge(cw, on='SIC', how='inner')

        sic_data = sic_data.rename(columns={'1997 NAICS': 'NAICS'})
        sic_data = \
            sic_data.drop(['SIC', 'Subsector and Industry'],
                          axis=1, errors='ignore')
        data_cols = [c for c in sic_data.columns if c not in ['Year', 'region',
                                                              'NAICS', 'SIC Count',
                                                              'SIC Allocation Ratio']]

        sic_data.loc[:, data_cols] = \
            sic_data[data_cols].multiply(sic_data['SIC Allocation Ratio'], axis='index')
        sic_data = sic_data.drop(['SIC Count', 'SIC Allocation Ratio'], axis=1)
        sic_data['1st NAICS'] = sic_data['NAICS'].astype(str).str[:1]
        sic_data = sic_data[sic_data['1st NAICS'] == '3']
        sic_data = sic_data.drop('1st NAICS', axis=1)
        print('sic_data:\n', sic_data)
        return sic_data

    # def industrial_sector_data(self):
    #     mecs_3_1 = pd.read_excel('https://www.eia.gov/consumption/manufacturing/data/2018/xls/Table3_1.xlsx', skiprows=9, index_col=0).dropna(axis=0, how='all') # By Manufacturing Industry and Region (physical units)
    #     mecs_3_1 = mecs_3_1.drop('Code(a)', axis=0)
    #     mecs_3_1 = mecs_3_1.rename(columns={' ': 'Subsector and Industry'})
    #     mecs_3_1 = self.clean_industrial_data(mecs_3_1, table_3_1=True)


    #     mecs_3_2 = pd.read_excel('https://www.eia.gov/consumption/manufacturing/data/2018/xls/Table3_2.xlsx', skiprows=10, index_col=0).dropna(axis=0, how='all') # By Manufacturing Industry and Region (trillion Btu)
    #     mecs_3_2 = self.clean_industrial_data(mecs_3_2)
    #     rename_dict_3_1 = {'Electricity(b)': 'Net Electricity',
    #                    'Fuel Oil': 'Residual Fuel Oil',
    #                    'Fuel Oil(c)': 'Distillate Fuel Oil',
    #                    'Gas(d)': 'Natural Gas',
    #                    'natural gasoline)(e)': 'HGL (excluding natural gasoline)',
    #                    'and Breeze': 'Coke Coal and Breeze'}
    #     mecs_3_2 = mecs_3_2.rename(columns=rename_dict_3_1)

    #     mecs_3_5 = pd.read_excel('https://www.eia.gov/consumption/manufacturing/data/2018/xls/Table3_5.xlsx', skiprows=10, index_col=0).dropna(axis=0, how='all') # Byproducts in Fuel Consumption By Manufacturing Industry and Region
    #                 # (trillion Btu)
    #     rename_dict_3_5 = {'Oven Gases': 'Blast Furnace/Coke Oven Gases',
    #                        'Gas': 'Waste Gas',
    #                        'Coke': 'Petroleum Coke',
    #                        'Black Liquor': 'Pulping Liquor or Black Liquor',
    #                        'Bark': 'Wood Chips, Bark',
    #                        'Materials': 'Waste Oils/Tars and Waste Materials'}
    #     mecs_3_5 = mecs_3_5.rename(columns=rename_dict_3_5)
    #     mecs_3_5 = self.clean_industrial_data(mecs_3_5)

    #     mecs_3_2_other = mecs_3_2[['Other(f)']]

    #     mecs_3_5 = mecs_3_5.merge(mecs_3_2_other, left_index=True, right_index=True, how='inner')

    #     mecs_3_5['steam'] = mecs_3_5['Total'].subtract(mecs_3_5['Other(f)'])
    #     mecs_3_5 = mecs_3_5.drop(['Total', 'Other(f)'], axis=1)

    #     industrial_btu = mecs_3_5.merge(mecs_3_2, left_index=True, right_index=True, how='outer')
    #     print('industrial_btu:\n', industrial_btu)
    #     print('industrial_btu cols:\n', industrial_btu.columns)

    #     return industrial_btu

    # def industrial_sector_energy(self):
    #     """TODO: do further processing to bridge Btu energy data with
    #     physical units used for emissions factors
    #     """
    #     industrial_data_btu = self.industrial_sector_data() # This is not in physical units!!
    #     industrial_renamed = self.mecs_epa_mapping(industrial_data_btu)
    #     return industrial_renamed

    def manufacturing_prices(self):
        """Call ASM API method from Asm class in get_census_data.py
        Specify three-digit NAICS Codes

        Returns:
            asm_price_data [type]: [description]
        """
        fuel_types = ['Gas', 'Coal', 'Distillate', 'Residual',
                      'LPG', 'Coke', 'Other']
        asm_cols = {'Gas': "Gas $/MBTU",
                    'Coal': 'Pre-2013 Price Estimate $/MMBtu',
                    'Distillate': "Distilate $/MBTU",
                    'Residual': "Residual $/MBTU",
                    'LPG': 'LPG (Use Propane Price) cents/gal',
                    'Coke': "Anthracite $/MBTU",
                    'Other': "Bituminous $/MBTU"}

        naics = [311, 312, 313, 314, 315, 316, 321, 322, 323, 324,
                 325, 326, 327, 331, 332, 333, 334, 335, 336, 337, 339]

        asm_price_data = []
        for f in fuel_types:
            try:
                predicted_fuel_price = \
                    Mfg_prices().main(latest_year=self.end_year,
                                      fuel_type=f, naics=naics,
                                      asm_col_map=asm_cols)
                predicted_fuel_price['fuel_type'] = f
                predicted_fuel_price = predicted_fuel_price.reset_index()              
            except Exception as e:
                print(f'fuel type {f} failed with error {e}')
                continue
            asm_price_data.append(predicted_fuel_price)

        asm_price_data = pd.concat(asm_price_data, axis=0)
        print('asm_price_data:\n', asm_price_data)
        asm_price_data['Year'] = asm_price_data['Year'].astype(int)
        asm_price_data = pd.melt(asm_price_data, id_vars=['Year', 'fuel_type'],
                                 value_vars=naics,
                                 var_name='NAICS', value_name='Price')
        print('asm_price_data:\n', asm_price_data)

        asm_price_data['NAICS'] = asm_price_data['NAICS'].astype(int)
        return asm_price_data

    def calc_quantity_shares(self, region='Total United States'):
        # From ASMdata_010220.xlsx[Quantity_shares_revised]
        """
        For a given MECS year, take NAICS by fuel (TBtu),
        calculate sum, then calcuate quantity shares

        Returns: 
            quantity_shares [DataFrame]: 
        """

        mecs_data, __ = self.mecs_data_by_year()
        mecs42_df = mecs_data['4_2']
        print('mecs42_df:\n', mecs42_df)
        print('mecs42_df cols', mecs42_df.columns)
        print("mecs42_df['region].unique()", mecs42_df['region'].unique())
        mecs42_df = mecs42_df[(mecs42_df['region'] == region) 
                              & ~(mecs42_df['NAICS'] == 'RSE Column Factors:')]
        mecs42_df = mecs42_df.drop('region', axis=1)
        mecs42_df = mecs42_df.set_index(['NAICS', 'Year'])
        print('mecs42_df:\n', mecs42_df)

        quantity_shares = df_utils().calculate_shares(mecs42_df,
                                                      total_label='Total')
        quantity_shares = quantity_shares.reset_index()
         
        rename_dict = {'Electricity(b)': 'Electricity',
                       'Residual Fuel Oil': 'Residual',
                       'Distillate Fuel Oil(c)': 'Distillate',
                       'Natural Gas(d)': 'Gas',
                       'HGL (excluding natural gasoline)(e)': 'HGL',
                       'Coke and Breeze': 'Coke',
                       'Other(f)': 'Other'}

        quantity_shares = quantity_shares.rename(columns=rename_dict)

        return quantity_shares

    @staticmethod
    def interpolate_mecs(mecs_data, col_name, reindex=None):
        """Interpolate MECS data where NAICS column gives
        dataframe a third dimension (thus table must be pivoted)

        Args:
            mecs_data (pd.DataFrame): MECS data by NAICS Code and other
                                      dimension
            col_name (str): Column in mecs_data containing values to
                            interpolate
            reindex (list-like, optional): New index to give mecs_data.
                                           Defaults to None.

        Returns:
            mecs_data (DataFrame): Interpolated MECS data
        """
        if 'Year' not in mecs_data.columns:
            mecs_data = mecs_data.reset_index()

        mecs_data = mecs_data.pivot(index='Year',
                                    columns='NAICS',
                                    values=col_name)

        if reindex is not None:
            mecs_data = mecs_data.reindex(reindex)
        for c in mecs_data.columns:
            mecs_data = \
                standard_interpolation(mecs_data,
                                       name_to_interp=c,
                                       axis=1)  # from mixed sources

        mecs_data = pd.melt(mecs_data.reset_index(), id_vars='Year',
                            var_name='NAICS', value_name=col_name)
        return mecs_data

    def quantity_shares_1998_forward(self):
        """[summary]

        Returns:
           composite_price [DataFrame]: [description]
        """
        mecs_years_prices_and_interpolations = self.manufacturing_prices()

        mecs_data_qty_shares = self.calc_quantity_shares()
        fuel_types = ['Gas', 'Coal', 'Distillate', 'Residual',
                      'LPG', 'Coke', 'Other', 'HGL', 'Electricity']               

        fuel_quanity_shares = []
        # interpolate mecs_data_qty_shares data (has 3 dimensions: fuel type, year, naics)
        print('mecs_data_qty_shares:\n', mecs_data_qty_shares)
        for fuel_type in fuel_types:
            if fuel_type in mecs_data_qty_shares.columns:
                fuel_type_data = []
                fuel_df = mecs_data_qty_shares[['Year', 'NAICS', fuel_type]]
                for n in fuel_df['NAICS'].unique():
                    fuel_naics = fuel_df[fuel_df['NAICS'] == n]
                    print('fuel_naics:\n', fuel_naics)
                    fuel_naics = \
                        self.interpolate_mecs(
                            fuel_naics,
                            fuel_type,
                            reindex=mecs_years_prices_and_interpolations['Year'].unique())
                    fuel_type_data.append(fuel_naics)
                
            df = pd.concat(fuel_type_data, axis=0)
            fuel_quanity_shares.append(df)
        fuel_quanity_shares = reduce(lambda df1,df2: df1.merge(df2, how='outer', 
                                     on=['Year', 'NAICS']), fuel_quanity_shares)
        fuel_quanity_shares = fuel_quanity_shares.set_index(['Year', 'NAICS'])

        # mecs_years_prices_and_interpolations = mecs_years_prices_and_interpolations.set_index('Year')
        mecs_prices = []
        for year in mecs_years_prices_and_interpolations['Year'].unique():
            prices_df = mecs_years_prices_and_interpolations[mecs_years_prices_and_interpolations['Year'] == year]
            prices_df = prices_df.pivot(index='NAICS',
                                        columns='fuel_type',
                                        values='Price')
            prices_df['Year'] = year                            
            mecs_prices.append(prices_df)

        mecs_prices_df = pd.concat(mecs_prices, axis=0)
        mecs_prices_df = mecs_prices_df.reset_index()
        mecs_prices_df = mecs_prices_df.set_index(['Year', 'NAICS'])


        mecs_prices_df['composite_price_calc'] = mecs_prices_df.multiply(fuel_quanity_shares,
                                                       fill_value=1,
                                                       axis=1).sum(skipna=True,
                                                                   axis=1)
        composite_price_calc = mecs_prices_df[['composite_price_calc']]


        return composite_price_calc.reset_index()
    
    def expenditure_ratios_revised(self, asm_data):
        """[summary]

        Args:
            asm_data ([type]): [description]

        Returns:
           mecs_based_expenditure [DataFrame]: [description]
        """
        # E from MECS_prices_122419.xlsx[MECS_data]/AN and NAICS3D/J (also called EXPFUEL)
        mecs = \
            pd.read_csv(
                './EnergyIntensityIndicators/Industry/Data/mecs_calc_purchased_fuels_historical.csv')

        mecs['Year'] = mecs['Year'].astype(int)

        dataset = mecs.merge(asm_data, how='outer', on=['Year', 'NAICS']).set_index('Year')
        dataset.index = dataset.index.astype(int)
        dataset['NAICS'] = dataset['NAICS'].astype(int)

        dataset['mecs_asm_ratio'] = \
            dataset['Calc. Cost of Fuels'].divide(
                dataset['EXPFUEL'], axis='index').multiply(1000) # G

        dataset_ = \
            self.interpolate_mecs(
                dataset, col_name='mecs_asm_ratio').rename(
                    columns={'mecs_asm_ratio': 'mecs_asm_ratio_interp'})
        interpolated_ratios_filler = \
            pd.read_csv(
                './EnergyIntensityIndicators/Industry/Data/mecs_asm_interpolated_ratio.csv')
        dataset_ = \
            dataset_.merge(interpolated_ratios_filler,
                           how='outer',
                           on=['Year', 'NAICS'])
        dataset_['mecs_asm_ratio_interp'] = \
            dataset_['mecs_asm_ratio_interp'].fillna(
                dataset_['interpolated_ratio'])
        dataset_ = dataset_.drop('interpolated_ratio', axis=1)

        dataset = dataset_.merge(dataset,
                                  how='outer',
                                  on=['Year', 'NAICS']).set_index('Year')       

        dataset['mecs_based_expenditure'] = dataset['Calc. Cost of Fuels'].multiply(1000) # I depends on MECS year/not
        dataset['fill_values'] = dataset['EXPFUEL'].multiply(dataset['mecs_asm_ratio_interp'], axis='index')
        dataset['mecs_based_expenditure'] = dataset['mecs_based_expenditure'].fillna(dataset['fill_values'])

        mecs_based_expenditure = dataset.reset_index()[['Year', 'NAICS', 'mecs_based_expenditure']]
        return mecs_based_expenditure

    def quantities_1998_forward(self, NAICS3D): 
        """[summary]

        Args:
            NAICS3D ([type]): [description]

        Returns:
           dataset [DataFrame]: [description]
        """
        quantity_shares_1998_forward = self.quantity_shares_1998_forward() # MECSPrices122419[Quantity shares 1998 forward]
        asm_data = NAICS3D.reset_index()[['Year', 'NAICS', 'EXPFUEL']]
        NAICS3D = NAICS3D.rename(columns={'column_av': 'ratio_fuel_to_offsite'})
        ratio_fuel_to_offsite = NAICS3D.reset_index()[['Year',
                                                       'NAICS',
                                                       'ratio_fuel_to_offsite']]

        mecs_based_expenditure = self.expenditure_ratios_revised(asm_data)
        mecs_based_expenditure['Year'] = mecs_based_expenditure['Year'].astype(int)
        dataset = mecs_based_expenditure.merge(quantity_shares_1998_forward, 
                                               how='outer', on=['NAICS', 'Year'])
        dataset = dataset.merge(ratio_fuel_to_offsite, how='outer', on=['NAICS', 'Year'])

        dataset = dataset[dataset['Year'] >= 1998]
        dataset = dataset.rename(columns={'Composite Price': 'composite_price'})
        return dataset
    
    @staticmethod
    def quantity_shares_1985_1998():
        """Take the sum of the product of quantity shares and the interpolated
        prices (calcualted using asm_price_fit.py), by 3-digit NAICS for a
        given MECS year.

        Returns:
            composite_price (pd.DataFrame): [description]
        """
        composite_price = \
            pd.read_csv(
                './EnergyIntensityIndicators/Industry/Data/' +
                'historical_composite_prices.csv').rename(
                    columns={'Composite Price ': 'composite_price'})

        return composite_price

    def expend_ratios_revised_85_97(self):
        """[summary]

        Returns:
           mecs_based_expenditure_hist [DataFrame]: [description]
        """
        mecs_based_expenditure_hist = \
            pd.read_csv(
                './EnergyIntensityIndicators/Industry/Data/Expend_ratios_revised_1985-97.csv')  
        return mecs_based_expenditure_hist
    
    @staticmethod
    def mecs_data_sic():
        """[summary]

        Returns:
           mecs_data_sic [DataFrame]: [description]
        """
        # from [MECS_prices_101116b.xlsx]MECS_data_SIC BA
        mecs_data_sic = \
            pd.read_csv(
                './EnergyIntensityIndicators/Industry/Data/MECS_data_SIC.csv')
        mecs_data_sic = mecs_data_sic[mecs_data_sic['Year'].notnull()]
        mecs_data_sic['Year'] = mecs_data_sic['Year'].astype(int)
        mecs_data_sic['NAICS'] = mecs_data_sic['NAICS'].astype(int)

        return mecs_data_sic

    def pre_1998_quantities(self):
        """[summary]

        Returns:
           dataset [DataFrame]: [description]
        """        
        # from quantity_shares_revised CW --> '[MECS_prices_122419.xlsx]Quantity Shares_1985-1998'!
        dollar_per_mmbtu = self.quantity_shares_1985_1998()
        dollar_per_mmbtu['Year'] = dollar_per_mmbtu['Year'].astype(int)
        dollar_per_mmbtu = dollar_per_mmbtu.rename(columns={'Composite Price': 'composite_price'})

        # from Expend_ratios_revised_1985-97 and Expend_ratios_revised
        mecs_based_expenditure = self.expend_ratios_revised_85_97()

        mecs_data_sic = self.mecs_data_sic()

        mecs_data_sic = mecs_data_sic[mecs_data_sic['Variable'] == 'Scale Factor'].drop('Variable', axis=1)
        mecs_data_sic = mecs_data_sic.rename(columns={'Value': 'ratio_fuel_to_offsite'})
        mecs_data_sic = mecs_data_sic.replace({'NAICS': {331: 328, 332: 329}})

        dataset = dollar_per_mmbtu.merge(mecs_based_expenditure, how='outer', on=['NAICS', 'Year'])
        
        mecs_sic_merge = dataset.merge(mecs_data_sic, how='outer', on=['NAICS', 'Year'])

        mecs_sic_merge = self.interpolate_mecs(mecs_sic_merge, col_name='ratio_fuel_to_offsite')

        dataset = dataset.merge(mecs_sic_merge, how='outer', on=['NAICS', 'Year'])

        dataset = dataset[dataset['Year'] <= 1997]
        dataset = dataset.rename(columns={' Expenditure': 'mecs_based_expenditure'})

        return dataset
    
    @staticmethod
    def aggregate_naics(df, values):
        """[summary]

        Args:
            df ([type]): [description]
            values ([type]): [description]

        Returns:
            df (pd.DataFrame): [description]
        """        
        df = df.pivot(index='NAICS', columns='Year', values=values)
        df.index = df.index.astype(int)

        df.loc['311-312', :] = df.loc[[311, 312], :].sum(axis=0)
        df.loc['313-314', :] = df.loc[[313, 314], :].sum(axis=0)
        df.loc['315-316', :] = df.loc[[315, 316], :].sum(axis=0)
        df = df.drop(index=[313, 314, 315, 316, 311, 312])
        df.index = df.index.astype(str)

        return df

    def final_quantities_asm_85(self):
        """Between-MECS-year interpolations are made in MECS_Annual_Fuel1
        and MECS_Annual_Fuel2 tabs in Ind_hap3 spreadsheet.
        Interpolations are also based on estimates developed in
        ASMdata_010220.xlsx[3DNAICS], which ultimately tie back to MECS fuel data
        from Table 4.2 and Table 3.2

        Note: NAICS codes are replaced in quantities_1998 forward because PNNL
              does this. Not sure why.

        Returns:
            final_quantities_asm_85_agg (pd.DataFrame): [description]
        """
        NAICS3D = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/3DNAICS.csv').set_index('Year')

        pre_1998_quantities = self.pre_1998_quantities()
        pre_1998_quantities['NAICS'] = pre_1998_quantities['NAICS'].astype(int)
        pre_1998_quantities = pre_1998_quantities.replace({'NAICS': {331: 328, 332: 329}})

        quantities_1998_forward = self.quantities_1998_forward(NAICS3D)
        quantities_1998_forward = quantities_1998_forward.replace({'NAICS': {331: 328, 332: 329}})
        print('quantities_1998_forward:\n', quantities_1998_forward)

        quantities_1998_forward = \
            quantities_1998_forward.rename(
                columns={c: 'composite_price' for c in quantities_1998_forward.columns if 'rice' in c})

        quantities = pd.concat([pre_1998_quantities, quantities_1998_forward], axis=0, sort=True)
        quantities = quantities.sort_values(by='Year')

        quantities['jan_2020_estimate'] = \
            quantities['mecs_based_expenditure'].divide(
                quantities['composite_price'], axis='index').multiply(0.001)
        
        # ASMdata_010330.xlsx , Final_quant_elec_w_ASM_87'
        quantities['final_quantities_asm_85'] = \
            quantities['jan_2020_estimate'].multiply(
                quantities['ratio_fuel_to_offsite'], axis='index')

        final_quantities_asm_85 = \
            quantities[pd.notnull(quantities['final_quantities_asm_85'])]
        final_quantities_asm_85 = \
            final_quantities_asm_85[~final_quantities_asm_85[['NAICS', 'Year']].duplicated()]

        final_quantities_asm_85_agg = \
            self.aggregate_naics(final_quantities_asm_85, values='final_quantities_asm_85')
        return final_quantities_asm_85_agg
    
    @staticmethod
    def import_mecs_electricity(asm):
        """ ### NOT SURE IF ASM or MECS ELECTRICITY DATA ARE USED ###
        Imports MECS data on electricityuse by 3-digit NAICS code.
        In the future,these values will need to be manually downloaded from
        Table 3.2 and added to csv.

        Args:
            asm ([type]): [description]

        Returns:
            electricity_consumption (pd.DataFrame): [description]
        """
        # import a CSV file of historical MECS electricity use from Table 3.2
        
        # elechap3b, all historical
        mecs_elec = pd.read_csv(
                    './EnergyIntensityIndicators/Industry/Data/elechap3b.csv')
        mecs_elec = mecs_elec.replace({'NAICS': {'331': '328', '332': '329'}})
        mecs_elec = mecs_elec.set_index('NAICS')
        mecs_elec = mecs_elec.rename(columns={col: int(col) for col
                                     in mecs_elec.columns})

        link_ratio_df = asm[[1987]].merge(mecs_elec[[1987]], how='outer', 
                                          left_index=True, right_index=True)
        # Ind_hap3_122219.xlsx[ASM_Annual_Elec_1970on]
        link_ratio = link_ratio_df['1987_x'].divide(link_ratio_df['1987_y'], 
                                                    axis='index').fillna(1)
        link_ratio = pd.DataFrame(pd.np.tile(link_ratio.values.reshape(
                                  len(link_ratio), 1), 
                                  (1, mecs_elec.shape[1])),
                                  index=link_ratio_df.index,
                                  columns=mecs_elec.columns)
        nea_based_data_linked = mecs_elec.multiply(link_ratio)
        nea_drop = [c for c in nea_based_data_linked.columns if c >= 1987]
        nea_based_data_linked = nea_based_data_linked.drop(nea_drop, axis=1)
        asm_drop = [c for c in asm.columns if c < 1987]
        asm = asm.drop(asm_drop, axis=1)

        electricity_consumption = nea_based_data_linked.merge(asm,
                                                              how='outer',
                                                              left_index=True,
                                                              right_index=True)
        electricity_consumption = electricity_consumption.transpose()
        electricity_consumption.index = \
            electricity_consumption.index.astype(int)
        return electricity_consumption
    
    # Corresponds to data in MECS_Fuel tab in Indhap3 spreadsheet, which
    # is connected to the MECS_Annual_Fuel1 and MECS_Annual_Fuel2
    # tabs in the same spreadsheet.
    def import_mecs_fuel(self):
        """Imports MECS data on fuel use by 3-digit NAICS code. In the future,
        these values will need to be manually downloaded from Table 3.2
        and added to csv.

        Returns:
            fallhap3b (pd.DataFrame): MECS data on fuel use by 3-digit NAICS code
        """
        fallhap3b = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/fallhap3b.csv')
        fallhap3b = fallhap3b.set_index('NAICS')
        return fallhap3b
    
    def get_historical_mecs(self):
        """Read in historical MECS csv, format (as in e.g. Coal (MECS) Prices)

        Returns:
            historical_mecs (pd.DataFrame): Historical MECS data (3 Digit NAICS)
        """
        # NAICS ARE ALREADY AGGREGATED
        historical_mecs = \
            pd.read_csv('./EnergyIntensityIndicators/Industry/Data/MECS_Fuel_Historical.csv')
        return historical_mecs

    def mecs_fuel(self, asm_elec_data, historical_mecs):
        """[summary]

        Args:
            asm_elec_data ([type]): [description]
            historical_mecs ([type]): [description]

        Returns:
            mecs_fuel (pd.DataFrame): [description]
        """        
        historical_mecs_31_32 = \
            pd.read_csv('./EnergyIntensityIndicators/Industry/Data/historical_mecs_31_32.csv')
        mecs_fuel = historical_mecs.set_index('NAICS')
        mecs_fuel = mecs_fuel.rename(columns={c: int(c) for c in mecs_fuel.columns})
        mecs_fuel = mecs_fuel.reset_index()
        mecs_fuel = pd.melt(mecs_fuel, id_vars='NAICS', value_name='Value', var_name='Year')

        mecs_fuel = self.interpolate_mecs(mecs_fuel, col_name='Value')
        mecs_fuel = mecs_fuel.pivot(index='NAICS', columns='Year', values='Value')
        mecs_fuel = mecs_fuel.rename(columns={c: int(c) for c in mecs_fuel.columns})

        return mecs_fuel

    def get_manufacturing_fuels(self, electricity_data):
        """[summary]

        Args:
            electricity_data ([type]): [description]

        Returns:
            fuels_consumption (pd.DataFrame): [description]
        """        
        # Ind_hap3_122219.xlsx[ASM_Annual_Fuel3_1970on]
        electricity_data = electricity_data.transpose()

        fuels_nea = self.import_mecs_fuel()  # fallhap3
        fuels_nea = fuels_nea.rename(columns={col: int(col) for col in
                                              fuels_nea.columns})

        mecs_fuel = self.get_historical_mecs()
        
        mecs_interpolated_data = self.mecs_fuel(electricity_data,
                                                historical_mecs=mecs_fuel)

        link_ratio_df = mecs_interpolated_data[[1985]].merge(fuels_nea[[1985]],
                                                             how='outer',
                                                             left_index=True,
                                                             right_index=True)
        link_ratio = link_ratio_df['1985_x'].divide(link_ratio_df['1985_y'],
                                                    axis='index').fillna(1)
        link_ratio = pd.DataFrame(pd.np.tile(link_ratio.values.reshape(len(link_ratio), 1),
                                  (1, fuels_nea.shape[1])),
                                  index=link_ratio_df.index,
                                  columns=fuels_nea.columns)

        nea_adjusted = fuels_nea.multiply(link_ratio)

        nea_drop = [c for c in nea_adjusted.columns if c >= 1985]
        nea_adjusted = nea_adjusted.drop(nea_drop, axis=1)
        mecs_drop = [c for c in mecs_interpolated_data.columns if c < 1985]
        mecs_interpolated_data = mecs_interpolated_data.drop(mecs_drop, axis=1)

        fuels_consumption = nea_adjusted.merge(mecs_interpolated_data, 
                                               how='outer', left_index=True, 
                                               right_index=True)

        fuels_consumption = fuels_consumption.transpose()

        fuels_consumption.index = fuels_consumption.index.astype(int)

        return fuels_consumption
    
    def manufacturing_energy(self):
        """Collect electricity and fuels for the manufacturing sector
        """        
        asm = self.final_quantities_asm_85()

        electricity_consumption = self.import_mecs_electricity(asm)
        electricity_consumption = electricity_consumption.rename(columns={'328': '331', '329': '332'})

        fuels_consumption = self.get_manufacturing_fuels(asm)
        # Transfered to industrial_indicators_060220.xlsx[Manufacturing_Energy_Data]
        return electricity_consumption, fuels_consumption

    # Data used in industrial_indicators[Manufacturing]
    def call_activity_data(self):
        """Call BEA API for gross ouput and value add by 3-digit NAICS.

        Returns:
            va_quant_index [type]: [description]
            go_quant_index [type]: [description]
        """
        va_quant_index, go_quant_index = \
            BEA_api(years=list(range(1949, 2018))).chain_qty_indexes()
        return va_quant_index, go_quant_index

    def manufacturing(self):
        """Main datasource is the Manufacturing Energy Consumption
        Survey (MECS), conducted by the EIA since 1985 (supplemented
        for non-MECS years by estimates derived from the Annual Survey
        of Manufactures (ASM) and the Economic Census (EC)
        conducted every five years)
        https://www.eia.gov/consumption/manufacturing/data/2014/
        https://www.eia.gov/consumption/manufacturing/data/2014/#r4

        Returns:
            data_dict (dict): [description]
        """
        va_quant_index, go_quant_index = self.call_activity_data()

        electricity_consumption, fuels_consumption = self.manufacturing_energy()

        rename_dict = {'311-312': 'Food and beverage and tobacco products', 
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

        electricity_consumption = electricity_consumption.rename(columns=rename_dict)
        fuels_consumption = fuels_consumption.rename(columns=rename_dict)
        
        data_dict = dict()
        for value in rename_dict.values():
            elec = electricity_consumption[[value]]
            fuels = fuels_consumption[[value]]
            go = go_quant_index[[value]]
            va = va_quant_index[[value]]

            sub_data_dict = {'energy': {'elec': elec, 
                                        'fuels': fuels}, 
                            'activity': {'gross_output': go,
                                        'value_added': va}}
            data_dict[value] = sub_data_dict
            
        return data_dict   

if __name__ == '__main__':
    asm = Manufacturing().manufacturing()
    print('asm:\n', asm)