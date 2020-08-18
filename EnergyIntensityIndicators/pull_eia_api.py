import os
import json
import requests
import pandas as pd
import numpy as np
from functools import reduce 
# from eiapy import Category
# from eiapy import Series
# from eiapy import MultiSeries

class GetEIAData:
    def __init__(self):
        # self.id_ = id_
        pass

    @staticmethod
    def eia_api(id_, id_type='category'):
        """[summary]

        Args:
            id_ ([type]): [description]
            id_type (str, optional): [description]. Defaults to 'category'.

        Returns:
            [type]: [description]
        """          
        api_key = os.environ.get("EIA_API_Key")

        if id_type == 'category':
            eia_data = GetEIAData.get_category(api_key, id_)
        elif id_type == 'series':
            eia_data = GetEIAData.get_series(api_key, id_)
        else:
            eia_data = None
            print('Error: neither series nor category given')
        return eia_data
    
    @staticmethod
    def get_category(api_key, id_):
            api_call = f'http://api.eia.gov/category/?api_key={api_key}&category_id={id_}'
            r = requests.get(api_call)
            data = r.json()
            eia_childseries = data['category']['childseries']
            eia_series_ids = [i['series_id'] for i in eia_childseries]
            eia_data = [GetEIAData.get_series(api_key, s) for s in eia_series_ids]
            all_category = reduce(lambda x, y: pd.merge(x, y, on ='Date'), eia_data)
            all_category = all_category.set_index('Date')
            return all_category

    @staticmethod
    def get_series(api_key, id_):
        api_call = f'http://api.eia.gov/series/?api_key={api_key}&series_id={id_}'
        r = requests.get(api_call)
        eia_data = r.json()
        date_column_name = str(eia_data['series'][0]['f'])
        data_column_name =  str(eia_data['series'][0]['name']) + ', ' + str(eia_data['series'][0]['units'])
        eia_df = pd.DataFrame.from_dict(eia_data['series'][0]['data'])
        eia_df = eia_df.rename(columns={0: date_column_name, 1: data_column_name})
        if date_column_name == 'M':
            eia_df['Date'] = pd.to_datetime(eia_df['M'], format='%Y%m')
            eia_df = eia_df.drop('M', axis='columns')
        elif date_column_name == 'A':
            eia_df['Date'] = pd.to_datetime(eia_df['A'], format='%Y')
            eia_df = eia_df.drop('A', axis='columns')
        else:
            print('No date column')
            pass
        return eia_df

    def get_seds(self, sector):
        """Used for commercial (ESCCB and TNCCB) and residential (ESCRB and TNRCB)
        
        """    
        consumption_all_btu = pd.read_csv('./EnergyIntensityIndicators/use_all_btu.csv')  # Commercial: '40210 , residential : '40209 
                                                                                          # 1960 through 2017 SEDS Data, MSN refers to fuel type
        state_to_census_region = pd.read_csv('./state_to_census_region.csv')
        consumption_census_region = consumption_all_btu.merge(state_to_census_region, left_on='State', right_on='USPC', how='left')

        years = list(range(1960, 2018))
        years = [str(year) for year in years]

        if sector == 'residential':
            pivotted = pd.pivot_table(consumption_census_region, index=years, columns='MSN', aggfunc='sum')  # , values='Census Region'
            pivotted['Grand Total'] = pivotted.sum()
            pivotted = pivotted[['ESRCB', 'TNRCB']]
            pt_results = 
        elif sector == 'commercial':
            pivotted_by_census_region = pd.pivot_table(consumption_census_region, index=years, columns='MSN', aggfunc='sum')
            esccb_by_census_region = pivotted_by_census_region.reset_index()
            tnccb_by_census_region = pivotted_by_census_region.reset_index()
            ESCCB_by_region = pd.pivot_table(pivotted_by_census_region['ESCCB'], index=years , columns='Census Region', aggfunc='sum')
            TNCCB_by_region = pd.pivot_table(pivotted_by_census_region['TNCCB'], index=years , columns='Census Region', aggfunc='sum')
            pt_results = 
        else:
            pt_results = None
        
        return pt_results

    def get_weather_data(self):
        """Tables 1.9 and 1.10 in the Monthly Energy Review"""
        # cdd_by_division = 
        # hdd_by_division = 
        pass
    
    def get_seds():
        """https://www.eia.gov/state/seds/seds-data-complete.php?sid=US
        """    
        pass

eia_data_cat = GetEIAData.eia_api(id_='711272', id_type='category')
print(eia_data_cat)
# eia_data = GetEIAData.eia_api(id_='TOTAL.NGACPUS.M', id_type='series')
# print(eia_data)
print('done')
