import os
import json
import requests
import pandas as pd
import numpy as np

class GetEIAData:
    print('current directory', os.getcwd())
    api_key = os.environ.get("EIA_API_Key")

    api_category_ids = {'EIA Datsets': 371, 'seds_consumption_all': 40204, 'seds_consumption_residential': 40209}
    category = 'seds_consumption_all'
    call  = f'http://api.eia.gov/category/?api_key={api_key}&category_id={api_category_ids[category]}'

    r = requests.get(call)
    files = r.json()
    print(files)

    call2 = f'http://api.eia.gov/series/?api_key={api_key}&series_id=SEDS.PATCB.AL.A'
    r = requests.get(call2)
    j = r.json()
    # df = pd.DataFrame.from_dict(j)
    print(j)
    # print(df)

    consumption_all_btu = pd.read_csv('./EnergyIntensityIndicators/use_all_btu.csv')
    print(consumption_all_btu.head())

    def get_seds():
        """Used for commercial (ESCCB and TNCCB) and residential (ESCRB and TNRCB)
        
        """    
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
    def get_weather_data():
        """Tables 1.9 and 1.10 in the Monthly Energy Review"""
        cdd_by_division = 
        hdd_by_division = 