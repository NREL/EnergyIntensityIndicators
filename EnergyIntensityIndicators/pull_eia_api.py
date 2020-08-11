import os
import json
import requests
import pandas as pd
import numpy as np
# from eiapy import Category
# from eiapy import Series
# from eiapy import MultiSeries

class GetEIAData:
    def __init__(self):
        # self.id_ = id_
        pass

    @staticmethod
    def eia_api(id_, category=True, series=False):
        """[summary]

        Args:
            id_ (int): [description]
            category (bool, optional): [description]. Defaults to True.
            series (bool, optional): [description]. Defaults to False.

        Returns:
            eia_data [type]: [description]
        """            
        print('current directory', os.getcwd())
        api_key = os.environ.get("EIA_API_Key")
        EIA_KEY = os.environ['EIA_KEY']
        print(api_key)
        print(EIA_KEY)
        print(api_key==EIA_KEY)

        # api_call = f'http://api.eia.gov/category/?api_key={api_key}&category_id={id_}'
        # r = requests.get(api_call)
        # data = r.json()
        # eia_childseries = data['category']['childseries']
        # eia_series_ids = [i['series_id'] for i in eia_childseries]
        # print(eia_series_ids)
        # eia_data = MultiSeries(eia_series_ids).get_data(all_data=True)
        # print(eia_data)

        # print(data['category'])

        # # api_call2 = f'http://api.eia.gov/category/?api_key={EIA_KEY}&category_id={id_}'
        # # r2 = requests.get(api_call2)
        # # data2 = r2.json()
        # # print('//////////////////////////////', data['category'])
        # if category:
        #     eia_category = Category(id_)
        #     print(type(eia_category.get_info()))
        #     print(eia_category.get_info())
        #     # eia_childseries = eia_category['category']['childseries']
        #     # eia_series_ids = [i['series_id'] for i in eia_childseries]
        #     # eia_data = MultiSeries(eia_series_ids)
        #     print('retrieved category')
        # elif series:
        #     eia_data = Series(id_)
        # else:
        #     eia_data = None
        #     print('No data')
        

        # return eia_data
        return None

    def get_seds(self):
        """Used for commercial (ESCCB and TNCCB) and residential (ESCRB and TNRCB)
        
        """    

        consumption_all_btu = pd.read_csv('./EnergyIntensityIndicators/use_all_btu.csv')  # Commercial: '40210 , residential : '40209

        years = list(range(1960, 2018))
        years = [str(year) for year in years]
        consumption_all_btu = pd.read_csv('../use_all_btu.csv') # 1960 through 2017 SEDS Data, MSN refers to fuel type
        state_to_census_region = pd.read_csv('./state_to_census_region.csv')
        consumption_census_region = consumption_all_btu.merge(state_to_census_region, left_on='State', right_on='USPC', how='left')
        # print(consumption_all_btu.head())
        
        pivotted = pd.pivot_table(consumption_census_region, index=years, columns='MSN', aggfunc='sum')  # , values='Census Region'
        pivotted['Grand Total'] = pivotted.sum()
        pivotted = pivotted[['ESRCB', 'TNRCB']]
        print(pivotted)

        pivotted_by_census_region = pd.pivot_table(consumption_census_region, index=years, columns='MSN', aggfunc='sum')
        # print('pivotted_by_census_region', pivotted_by_census_region.head())
        esccb_by_census_region = pivotted_by_census_region.reset_index()
        # print('esccb_by_census_region', esccb_by_census_region)
        tnccb_by_census_region = pivotted_by_census_region.reset_index()
        ESCCB_by_region = pd.pivot_table(pivotted_by_census_region['ESCCB'], index=years , columns='Census Region', aggfunc='sum')
        TNCCB_by_region = pd.pivot_table(pivotted_by_census_region['TNCCB'], index=years , columns='Census Region', aggfunc='sum')
        # PRINT(TNCCB_by_region)
        return None

    def get_weather_data(self):
        """Tables 1.9 and 1.10 in the Monthly Energy Review"""
        # cdd_by_division = 
        # hdd_by_division = 
        pass


# eia_data = GetEIAData.eia_api(id_='711272')
# print(eia_data)
# print('done')
print('current directory', os.getcwd())
api_key = os.environ.get("EIA_API_Key")
EIA_KEY = os.environ['EIA_KEY']
print(api_key)
print(EIA_KEY)
print(api_key==EIA_KEY)