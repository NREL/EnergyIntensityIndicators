import pandas as pd
import os
import numpy as np
import zipfile
import requests
import io
import glob
from datetime import datetime
from functools import reduce

from EnergyIntensityIndicators.electricity import ElectricityIndicators

from EnergyIntensityIndicators.pull_eia_api import GetEIAData
from EnergyIntensityIndicators.utilites import dataframe_utilities as df_utils
from EnergyIntensityIndicators.Industry.asm_price_fit import Mfg_prices
from EnergyIntensityIndicators.utilites.standard_interpolation \
     import standard_interpolation


class ManufacturingSectors:

    def __init__(self):
        self.eia = GetEIAData('Industry')
        self.end_year = datetime.now().year

    def mecs_data_by_year(self):
        # Energy Consumption as a Fuel
        # Table 3.1 : By Mfg. Industry & Region (physical units)
        # Table 3.2 : By Mfg. Industry & Region (trillion Btu)
        # Table 3.5 : Byproducts in Fuel Consumption by Mfg. Industry & Region (trillion Btu)
        mecs_data = {2018: 
                        {'table_3_1': {'endpoint': 'Table3_1.xlsx', 'skiprows': 9, 'skip_footer': 20}, 
                         'table_3_2': {'endpoint': 'Table3_2.xlsx', 'skiprows': 9, 'skip_footer': 14}, 
                         'table_3_5': {'endpoint': 'Table3_5.xlsx', 'skiprows': 9, 'skip_footer': 12},
                         'table_4_2': {'endpoint': 'Table4_2.xlsx', 'skiprows': 0, 'skip_footer': 13}},
                     2014: 
                        {'table_3_1': {'endpoint': 'table3_1.xlsx', 'skiprows': 9, 'skip_footer': 20}, 
                         'table_3_2': {'endpoint': 'table3_2.xlsx', 'skiprows': 9, 'skip_footer': 14}, 
                         'table_3_5': {'endpoint': 'table3_5.xlsx', 'skiprows': 9, 'skip_footer': 12},
                         'table_4_2': {'endpoint': 'table4_2.xlsx', 'skiprows': 0, 'skip_footer': 13}},
                     2010: 
                        {'table_3_1': {'endpoint': 'Table3_1.xls', 'skiprows': 9, 'skip_footer': 47}, 
                         'table_3_2': {'endpoint': 'Table3_2.xls', 'skiprows': 8, 'skip_footer': 47}, 
                         'table_3_5': {'endpoint': 'Table3_5.xls', 'skiprows': 9, 'skip_footer': 29},
                         'table_4_2': {'endpoint': 'Table4_2.xls', 'skiprows': 0, 'skip_footer': 34}},
                     2006: 
                        {'table_3_1': {'endpoint': 'Table3_1.xls', 'skiprows': 10, 'skip_footer': 49}, 
                         'table_3_2': {'endpoint': 'Table3_2.xls', 'skiprows': 9, 'skip_footer': 49}, 
                         'table_3_5': {'endpoint': 'Table3_5.xls', 'skiprows': 10, 'skip_footer': 31},
                         'table_4_2': {'endpoint': 'Table4_2.xls', 'skiprows': 0, 'skip_footer': 36}},
                     2002: 
                        {'table_3_1': {'endpoint': 'Table3.1_02.xls', 'skiprows': 7, 'skip_footer': 49}, 
                         'table_3_2': {'endpoint': 'Table3.2_02.xls', 'skiprows': 6, 'skip_footer': 49}, 
                         'table_3_5': {'endpoint': 'Table3.5_02.xls', 'skiprows': 7, 'skip_footer': 55},
                         'table_4_2': {'endpoint': 'Table4.2_02.xls', 'skiprows': 0, 'skip_footer': 36}},
                     1998: 
                        {'table_3_1': {'endpoint': 'd98n3_1.xls', 'skiprows': 7, 'skip_footer': 53},  # Fuel Consumption, 1998; Level: National and Regional Data; Row: NAICS Codes; Column: Energy Sources; Unit: Physical Units or Btu
                         'table_3_2': {'endpoint': 'd98n3_2.xls', 'skiprows': 6, 'skip_footer': 53},  # Fuel Consumption, 1998; Level: National and Regional Data; Row: NAICS Codes; Column: Energy Sources; Unit: Trillion Btu
                         'table_3_5': {'endpoint': 'd98n5_1.xls', 'skiprows': 7, 'skip_footer': 59},  # Selected Byproducts in Fuel Consumption, 1998; Level: National Data and Regional Totals; Row: NAICS Codes; Column: Energy Sources; Unit: Trillion Btu
                         'table_4_2': {'endpoint': 'd98n4_2.xls', 'skiprows': 0, 'skip_footer': 40}}, 
                     1994: 
                        {'table_3_1': {'endpoint': 'm94_04a.xls', 'skiprows': 6, 'skip_footer': 34}, 
                         'table_3_2': {'endpoint': 'm94_04b.xls', 'skiprows': 5, 'skip_footer': 34}, 
                         'table_3_5': {'endpoint': 'm94_06.xls', 'skiprows': 7, 'skip_footer': 21},
                         'table_4_2': {'endpoint': 'm94_05b.xls', 'skiprows': 0, 'skip_footer': 31}},
                     1991: 
                        {'table_3_1': {'endpoint': 'mecs04a.xls', 'skiprows': 6, 'skip_footer': 36}, 
                         'table_3_2': {'endpoint': 'mecs04b.xls', 'skiprows': 5, 'skip_footer': 38}, 
                         'table_3_5': None,
                         'table_4_2': {'endpoint': 'mecs05b.xls', 'skiprows': 0, 'skip_footer': 34}},
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
            print('year:', year)
            print('type year:', type(year))
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

                    index_label = general_df.iloc[general_df.index.get_loc('Code(a)')-1].name

                    col_labels = col_labels.apply(lambda c: c.str.cat(sep=' '), axis=0)
                    col_labels = col_labels.apply(lambda s: s.strip())
                    col_labels = col_labels.to_frame(name=index_label).transpose()

                    df = pd.read_excel(general_url, skiprows=t_dict['skiprows'],
                                       skipfooter=t_dict['skip_footer'], 
                                       index_col=0)
                    df = df.iloc[df.index.get_loc('Code(a)')+1:]

                    df.columns = col_labels.loc[index_label, :]
                    df.columns.name = None

                    df = df.dropna(axis=1, how='all')

                    df = df.rename(columns={'Total (trillion Btu)': 'Total',
                                            'Industry Group and Industry': 'Subsector and Industry',
                                            'Industry Groups and Industry': 'Subsector and Industry',
                                            'LPG and NGL(e) (million bbl)': 'HGL (excluding natural gasoline)(e) (million bbl)',
                                            'LPG and NGL(e)': 'HGL (excluding natural gasoline)(e)'})
                    df = df.drop(['RSE Row Factors', ''], axis=1, errors='ignore')

                    if t == 'table_3_1':
                        mecs_3_1 = self.clean_industrial_data(df, table_3_1=True, sic=sic)
                        mecs_3_1['Year'] = year
                        # print('mecs_3_1:\n', mecs_3_1)
                        # print('mecs_3_1 cols:\n', mecs_3_1.columns)

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
                        print('mecs_3_2:\n', mecs_3_2)

                        if sic:
                            sic_3_2.append(mecs_3_2)
                        else:
                            all_3_2.append(mecs_3_2)
                    
                    elif t == 'table_4_2':
                        mecs_4_2 = self.clean_industrial_data(df, sic=sic)
                        # print('table_4_2:\n', mecs_4_2)

                        if sic:
                            sic_4_2.append(mecs_4_2)
                        else:
                            all_4_2.append(mecs_4_2)

        all_3_1 = pd.concat(all_3_1, axis=0).reset_index()
        all_3_1 = all_3_1.set_index(['Year', 'region', 'NAICS', 'Subsector and Industry'])
        sic_3_1 = pd.concat(sic_3_1, axis=0).reset_index()
        # sic_3_1 = sic_3_1.set_index(['Year', 'region', 'SIC', 'Subsector and Industry'])

        all_3_2 = pd.concat(all_3_2, axis=0).reset_index()
        # all_3_2 = all_3_2.set_index(['Year', 'region', 'NAICS', 'Subsector and Industry'])
        sic_3_2 = pd.concat(sic_3_2, axis=0).reset_index()
        # sic_3_2 = sic_3_2.set_index(['Year', 'region', 'SIC', 'Subsector and Industry'])

        all_3_5 = pd.concat(all_3_5, axis=0).reset_index()
        # all_3_5 = all_3_5.set_index(['Year', 'region', 'NAICS', 'Subsector and Industry'])
        sic_3_5 = pd.concat(sic_3_5, axis=0).reset_index()
        # sic_3_5 = sic_3_5.set_index(['Year', 'region', 'SIC', 'Subsector and Industry'])

        all_4_2 = pd.concat(all_4_2, axis=0).reset_index()
        # all_4_2 = all_4_2.set_index(['Year', 'region', 'NAICS', 'Subsector and Industry'])
        sic_4_2 = pd.concat(sic_4_2, axis=0).reset_index()
        # sic_4_2 = sic_4_2.set_index(['Year', 'region', 'SIC', 'Subsector and Industry'])
        print('sic_3_1:\n', sic_3_1)
        print('sic_3_2:\n', sic_3_2)
        print('all_3_1:\n', all_3_1)

        print('all_3_2:\n', all_3_2)
        exit()
        all_3_2 = all_3_2[['Year', 'region', 'NAICS', 'Subsector and Industry', 'Other(f)']]

        print('all_3_5:\n', all_3_5)
        print('all_3_5 cols:\n', all_3_5.columns)

        all_3_5 = all_3_5.merge(all_3_2, on=['Year', 'region', 'NAICS',
                                             'Subsector and Industry'],
                                how='inner')

        print("all_3_5['Total']:\n", all_3_5['Total'])
        all_3_5['steam'] = all_3_5['Total'].subtract(all_3_5['Other(f)'])
        all_3_5 = all_3_5.drop(['Total', 'Other(f)'], axis=1)

        industrial_btu = all_3_5.merge(all_3_2, on=['Year', 'region', 'NAICS', 'Subsector and Industry'], how='outer')

        print('industrial_btu:\n', industrial_btu)
        x_walk = self.mecs_sic_crosswalk()
        sic_naics_3_1 = self.naics_to_sic(sic_3_1 , x_walk)
        sic_naics_3_2 = self.naics_to_sic(sic_3_2 , x_walk)
        sic_naics_3_5 = self.naics_to_sic(sic_3_5, x_walk)
        sic_naics_4_2 = self.naics_to_sic(sic_4_2 , x_walk)
        exit()
        print('industrial nan:\n', industrial_btu[pd.isna(industrial_btu['Waste Gas'])])
        mecs = {'NAICS': {'3_1': all_3_1, '3_2': all_3_2,
                          '3_5': all_3_5, '4_2': all_4_2},
                'SIC': {'3_1': sic_naics_3_1, '3_2': sic_naics_3_2,
                        '3_5': sic_naics_3_5, '4_2': sic_naics_4_2}}
        return mecs, industrial_btu
    
    @staticmethod
    def clean_industrial_data(raw_data, table_3_1=False, sic=False):

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

        raw_data[code] = raw_data[code].fillna(raw_data['Subsector and Industry'])
        raw_data = raw_data.set_index(['region', code, 'Subsector and Industry'])
        raw_data = raw_data.replace({'*': 0.25, 'Q': np.nan, 'D': np.nan, 'W': np.nan})
        return raw_data

    @staticmethod
    def mecs_sic_crosswalk():
        #  Use crosswalk 1987 SIC to 1997 NAICS from 
        #  https://www.census.gov/eos/www/naics/concordances/concordances.html
        cw = pd.read_excel('https://www.census.gov/eos/www/naics/concordances/1987_SIC_to_1997_NAICS.xls')
        cw = cw.astype(int, errors='ignore')
        cw = cw[['SIC', '1997 NAICS']]
        print('cw:\n', cw)
        return cw

    def create_historical_mecs_31_32(self):
        mecs = self.mecs_data_by_year()
        mecs_3_1 = mecs['NAICS']['3_1'][['Year', 
                                         'region', 
                                         'NAICS', 
                                         'Subsector and Industry',
                                         'Net Electricity(b) (million kWh)']]
        mecs_3_2 = mecs['NAICS']['3_2'][['Year', 
                                         'region', 
                                         'NAICS', 
                                         'Subsector and Industry',
                                         'Total',
                                         'Net Electricity(b)']]
        historical_mecs_31_32 = mecs_3_1.merge(mecs_3_2, 
                                               on=['Year', 'region', 
                                                   'NAICS', 'Subsector and Industry'], 
                                               how='outer')
        mecs_fuel = mecs_3_2.copy()
        mecs_fuel['Fuel'] = mecs_fuel['Total'].subtract(mecs_fuel['Net Electricity(b)'])
        mecs_fuel = mecs_fuel.drop(['Total', 'Net Electricity(b)'], axis=1)
        return historical_mecs_31_32, mecs_fuel

    @staticmethod
    def naics_to_sic(sic_data, cw):
        sic_data = sic_data[~sic_data['SIC'].isnull()]
        sic_data = sic_data[sic_data['SIC'].str.isnumeric()]
        sic_data['SIC'] = sic_data['SIC'].astype(int)
        # sic_data = sic_data[~isinstance(sic_data['SIC'], str)]
        # sic_data = dic_data.drop('RSE Column Factors:', axis=0)
        print("sic_data['SIC']:\n", sic_data['SIC'])
        print("cw['SIC']:\n", cw['SIC'])
        sic_data = sic_data.merge(cw, on='SIC', how='left')
        print('sic_data:\n', sic_data)
        # sic_data = sic_data.drop('SIC', axis=1)
        sic_data = sic_data.rename(columns={'1997 NAICS': 'NAICS'})
        print('sic_data rename:\n', sic_data)
        pass

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

    def industrial_sector_energy(self):
        """TODO: do further processing to bridge Btu energy data with 
        physical units used for emissions factors
        """        
        industrial_data_btu = self.industrial_sector_data() # This is not in physical units!!
        industrial_renamed = self.mecs_epa_mapping(industrial_data_btu) 
        return industrial_renamed

    def manufacturing_prices(self):
        """Call ASM API method from Asm class in get_census_data.py
        Specify three-digit NAICS Codes
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
            predicted_fuel_price = Mfg_prices().main(latest_year=self.end_year,
                                                     fuel_type=f, naics=naics,
                                                     asm_col_map=asm_cols)
            asm_price_data.append(predicted_fuel_price)

        asm_price_data = pd.concat(asm_price_data, axis=1)
        asm_price_data['Year'] = asm_price_data['Year'].astype(int)
        asm_price_data['NAICS'] = asm_price_data['NAICS'].astype(int)

        return asm_price_data

    def calc_quantity_shares(self):
        # From ASMdata_010220.xlsx[Quantity_shares_revised]
        """
        For a given MECS year, take NAICS by fuel (TBtu),
        calculate sum, then calcuate quantity shares
        """

        mecs_data, industrial_btu = self.mecs_data_by_year()
        mecs42_df = mecs_data['NAICS']['4_2']
        
        quantity_shares = df_utils.calculate_shares(mecs42_df, total_label=['  Calc. Total'])
        return quantity_shares

    @staticmethod
    def interpolate_mecs(mecs_data, col_name, reindex=None):
        if 'Year' not in mecs_data.columns:
            mecs_data = mecs_data.reset_index()
        mecs_data = mecs_data.pivot(index='Year', columns='NAICS', 
                                    values=col_name)
        if reindex is not None:
            mecs_data = mecs_data.reindex(reindex)
        for c in mecs_data.columns:
            mecs_data = standard_interpolation(mecs_data, name_to_interp=c,
                                               axis=1)  # from mixed sources
        
        mecs_data = pd.melt(mecs_data.reset_index(), id_vars='Year',
                            var_name='NAICS', value_name=col_name)
        return mecs_data

    def quantity_shares_1998_forward(self):
        mecs_years_prices_and_interpolations = self.manufacturing_prices()

        mecs_data_qty_shares = self.calc_quantity_shares()
        mecs_cols = mecs_data_qty_shares.columns
        mecs_data_qty_shares = mecs_data_qty_shares.reset_index()
        fuel_quanity_shares = []
        # interpolate mecs_data_qty_shares data (has 3 dimensions: fuel type, year, naics)
        for fuel_type in mecs_cols:
            fuel_df = mecs_data_qty_shares[['Year', 'NAICS', fuel_type]]
            fuel_df = self.interpolate_mecs(fuel_df, fuel_type, 
                                            reindex=mecs_years_prices_and_interpolations['Year'].unique())
            fuel_quanity_shares.append(fuel_df)

        fuel_quanity_shares = reduce(lambda df1,df2: df1.merge(df2, how='outer', 
                                     on=['Year', 'NAICS']), fuel_quanity_shares)
        fuel_quanity_shares = fuel_quanity_shares.set_index('Year')

        mecs_years_prices_and_interpolations = mecs_years_prices_and_interpolations.set_index('Year')

        # composite_price = mecs_years_prices_and_interpolations.multiply(fuel_quanity_shares, axis='index').sum(axis=1)
        composite_price = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/current_composite_price.csv')
        composite_price['NAICS'] = composite_price['NAICS'].astype(int)
        composite_price['Year'] = composite_price['Year'].astype(int)
        
        return composite_price
    
    def expenditure_ratios_revised(self, asm_data):
        mecs = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/mecs_calc_purchased_fuels_historical.csv') # E from MECS_prices_122419.xlsx[MECS_data]/AN and NAICS3D/J (also called EXPFUEL)

        mecs['Year'] = mecs['Year'].astype(int)

        dataset = mecs.merge(asm_data, how='outer', on=['Year', 'NAICS']).set_index('Year')        
        dataset.index = dataset.index.astype(int)
        dataset['NAICS'] = dataset['NAICS'].astype(int)

        dataset['mecs_asm_ratio'] = dataset['Calc. Cost of Fuels'].divide(dataset['EXPFUEL'], axis='index').multiply(1000) # G

        dataset_ = self.interpolate_mecs(dataset, col_name='mecs_asm_ratio').rename(columns={'mecs_asm_ratio': 'mecs_asm_ratio_interp'})
        interpolated_ratios_filler = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/mecs_asm_interpolated_ratio.csv')
        dataset_ = dataset_.merge(interpolated_ratios_filler, how='outer', on=['Year', 'NAICS'])
        dataset_['mecs_asm_ratio_interp'] = dataset_['mecs_asm_ratio_interp'].fillna(dataset_['interpolated_ratio'])
        dataset_ = dataset_.drop('interpolated_ratio', axis=1)

        dataset =  dataset_.merge(dataset, how='outer', on=['Year', 'NAICS']).set_index('Year')       

        dataset['mecs_based_expenditure'] = dataset['Calc. Cost of Fuels'].multiply(1000) # I depends on MECS year/not
        dataset['fill_values'] = dataset['EXPFUEL'].multiply(dataset['mecs_asm_ratio_interp'], axis='index')
        dataset['mecs_based_expenditure'] = dataset['mecs_based_expenditure'].fillna(dataset['fill_values'])

        mecs_based_expenditure = dataset.reset_index()[['Year', 'NAICS', 'mecs_based_expenditure']]
        return mecs_based_expenditure

    def quantities_1998_forward(self, NAICS3D): 
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
        """
        Take the sum of the product of quantity shares and the interpolated
        prices (calcualted using asm_price_fit.py), by 3-digit NAICS for a
        given MECS year.
        """
        composite_price = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/historical_composite_prices.csv').rename(columns={'Composite Price ': 'composite_price'})

        return composite_price

    def expend_ratios_revised_85_97(self):

        mecs_based_expenditure_hist = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/Expend_ratios_revised_1985-97.csv')  
        return mecs_based_expenditure_hist
    
    @staticmethod
    def mecs_data_sic():
        mecs_data_sic = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/MECS_data_SIC.csv') # from [MECS_prices_101116b.xlsx]MECS_data_SIC BA
        mecs_data_sic = mecs_data_sic[mecs_data_sic['Year'].notnull()]
        mecs_data_sic['Year'] = mecs_data_sic['Year'].astype(int)
        mecs_data_sic['NAICS'] = mecs_data_sic['NAICS'].astype(int)

        return mecs_data_sic

    def pre_1998_quantities(self):
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
    
    def final_quantities_asm_85(self):
        """
        Between-MECS-year interpolations are made in MECS_Annual_Fuel1
        and MECS_Annual_Fuel2 tabs in Ind_hap3 spreadsheet.
        Interpolations are also based on estimates developed in
        ASMdata_010220.xlsx[3DNAICS], which ultimately tie back to MECS fuel data
        from Table 4.2 and Table 3.2
        """
        NAICS3D = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/3DNAICS.csv').set_index('Year')

        pre_1998_quantities = self.pre_1998_quantities()
        pre_1998_quantities['NAICS'] = pre_1998_quantities['NAICS'].astype(int)
        pre_1998_quantities = pre_1998_quantities.replace({'NAICS': {331: 328, 332: 329}})

        quantities_1998_forward = self.quantities_1998_forward(NAICS3D)
        quantities_1998_forward = quantities_1998_forward.replace({'NAICS': {331: 328, 332: 329}})

        quantities_1998_forward = quantities_1998_forward.rename(columns={c: 'composite_price' for c in quantities_1998_forward.columns if 'rice' in c})

        quantities = pd.concat([pre_1998_quantities, quantities_1998_forward], axis=0, sort=True)
        quantities = quantities.sort_values(by='Year')

        quantities['jan_2020_estimate'] = quantities['mecs_based_expenditure'].divide(quantities['composite_price'], axis='index').multiply(0.001)

        quantities['final_quantities_asm_85'] = quantities['jan_2020_estimate'].multiply(quantities['ratio_fuel_to_offsite'], axis='index') # ASMdata_010330.xlsx , Final_quant_elec_w_ASM_87'

        final_quantities_asm_85 = quantities[pd.notnull(quantities['final_quantities_asm_85'])]
        final_quantities_asm_85 = final_quantities_asm_85[~final_quantities_asm_85[['NAICS', 'Year']].duplicated()]
        final_quantities_asm_85_agg = self.aggregate_naics(final_quantities_asm_85, values='final_quantities_asm_85')
        return final_quantities_asm_85_agg

if __name__ == '__main__':
    asm = ManufacturingSectors().final_quantities_asm_85()
    print('asm:\n', asm)