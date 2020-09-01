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
    def __init__(self, sector):
        self.sector = sector
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

    def get_seds(self):
        """Used for commercial (ESCCB and TNCCB) and residential (ESCRB and TNRCB)
        './EnergyIntensityIndicators/use_all_btu.csv'
           https://www.eia.gov/state/seds/seds-data-complete.php?sid=US
        """    
        consumption_all_btu = pd.read_csv('https://www.eia.gov/state/seds/sep_use/total/csv/use_all_btu.csv')  # Commercial: '40210 , residential : '40209 
                                                                                          # 1960 through 2017 SEDS Data, MSN refers to fuel type
        state_to_census_region = pd.read_csv('./state_to_census_region.csv')
        consumption_census_region = consumption_all_btu.merge(state_to_census_region, left_on='State', right_on='USPC', how='left')

        years = list(range(1960, 2018))
        years = [str(year) for year in years]

        pivotted_by_census_region = pd.pivot_table(consumption_census_region, index=years, columns='MSN', aggfunc='sum').reset_index()  # , values='Census Region'
        pivotted_by_census_region['Grand Total'] = pivotted_by_census_region.sum()


        if self.sector == 'residential':
            pivotted_by_census_region = pivotted_by_census_region[['ESRCB', 'TNRCB']]

            ESRCB_by_region = pd.pivot_table(pivotted_by_census_region['ESRCB'], index=years , columns='Census Region', aggfunc='sum')
            TNRCB_by_region = pd.pivot_table(pivotted_by_census_region['TNRCB'], index=years , columns='Census Region', aggfunc='sum')
            
            elec_to_indicators = ESRCB_by_region.multiply(0.001)
            elec_to_indicators['US'] = elec_to_indicators.sum(1)

            total_primary = TNRCB_by_region.subtract(ESRCB_by_region)
            total_primary_to_indicators = total_primary.multiply(0.001)
            total_primary_to_indicators['US'] = total_primary_to_indicators.sum(1)

        elif self.sector == 'commercial':
            pivotted_by_census_region = pivotted_by_census_region[['ESCCB', 'TNCCB']]

            ESCCB_by_region = pd.pivot_table(pivotted_by_census_region['ESCCB'], index=years , columns='Census Region', aggfunc='sum')
            TNCCB_by_region = pd.pivot_table(pivotted_by_census_region['TNCCB'], index=years , columns='Census Region', aggfunc='sum')

            elec_to_indicators = ESCCB_by_region.multiply(0.001)
            elec_to_indicators['US'] = elec_to_indicators.sum(1)

            total_primary = TNCCB_by_region.subtract(ESCCB_by_region)
            total_primary_to_indicators = total_primary.multiply(0.001)
            total_primary_to_indicators['US'] = total_primary_to_indicators.sum(1)

        else:
            return None
        
        return total_primary_to_indicators, elec_to_indicators

    def national_calibration(self):
        """Calibrate SEDS energy consumption data to most recent data from the Annual or Monthly Energy Review

        TODO: 
        The whole point of this is to reconcile the AER and MER data, so they shouldn't be the same API endpoint
        """
        if self.sector == 'residential':
            AER11_table2_1b_update = GetEIAData.eia_api(id_='711250')
            AnnualData_MER_22_Dec2019 = GetEIAData.eia_api(id_='711250')         
            electricity_df = pd.DataFrame()
            electricity_df['AER 11 (Billion Btu)'] = # Column S
            electricity_df['MER, 12/19 (Trillion Btu)'] = # Column K
            electricity_df['SEDS (10/18) (Trillion Btu)'] =  # Column G
            electricity_df['Ratio MER/SEDS'] = electricity_df['MER, 12/19 (Trillion Btu)'].div(electricity_df['SEDS (10/18) (Trillion Btu)'])
            electricity_df['Final Est. (Trillion Btu)'] = electricity_df['SEDS (10/18) (Trillion Btu)'].multiply(electricity_df['Ratio MER/SEDS'])
            # If SEDS is 0, replace with MER

            fuels_df['AER 11 (Billion Btu)'] = # Column Q
            fuels_df['MER, 12/19 (Trillion Btu)'] =  # Column J
            fuels_df['SEDS (10/18) (Trillion Btu)'] =  # Column N
            fuels_df['Ratio MER/SEDS'] = fuels_df['MER, 12/19 (Trillion Btu)'].div(fuels_df['SEDS (10/18) (Trillion Btu)'])
            fuels_df['Final Est. (Trillion Btu)'] = fuels_df['SEDS (10/18) (Trillion Btu)'].multiply(fuels_df['Ratio MER/SEDS'])
            # If SEDS is 0, replace with MER

            # Not sure if these are needed
            recs_millions =  # RECS (millions) column AF
            recs_btu_hh = electricity_df['SEDS (10/18) (Trillion Btu)'].add(fuels_df['SEDS (10/18) (Trillion Btu)']).div(recs_millions)  # How do order of operations work here ?? (should be add and then divide)
            calibrated_hh = # National column N
            aer_btu_hh =  electricity_df['MER, 12/19 (Trillion Btu)'].add(fuels_df['MER, 12/19 (Trillion Btu)']).div(calibrated_hh)  # How do order of operations work here ?? (should be add and then divide)
        
        elif self.sector === 'commercial':
            AER11_Table21C_Update =  GetEIAData.eia_api(id_='711251')
            mer_data23_Dec_2019 = GetEIAData.eia_api(id_='711251')
            electricity_df = pd.DataFrame()
            electricity_df['AER 11 (Billion Btu)'] = # Column W
            electricity_df['MER, 12/19 (Trillion Btu)'] = # Column M
            electricity_df['SEDS (01/20) (Trillion Btu)'] =  # Column G
            electricity_df['Ratio MER/SEDS'] = electricity_df['MER, 12/19 (Trillion Btu)'].div(electricity_df['SEDS (01/20) (Trillion Btu)']])
            electricity_df['Final Est. (Trillion Btu)'] = electricity_df['SEDS (01/20) (Trillion Btu)']].multiply(electricity_df['Ratio MER/SEDS'])

            fuels_df = pd.DataFrame()
            fuels_df['AER 11 (Billion Btu)'] = # Column U
            fuels_df['MER, 12/19 (Trillion Btu)'] =  # Column L
            fuels_df['SEDS (01/20) (Trillion Btu)'] =  # Column N
            fuels_df['Ratio MER/SEDS'] = fuels_df['MER, 12/19 (Trillion Btu)'].div(fuels_df['SEDS (01/20) (Trillion Btu)'])
            fuels_df['Final Est. (Trillion Btu)'] = fuels_df['SEDS (01/20) (Trillion Btu)'].multiply(fuels_df['Ratio MER/SEDS'])
            # If SEDS is 0, replace with MER
    
        else: 
            pass

        national_calibration = electricity_df.merge(fuels_df, on='year', how='outer')
        return national_calibration

    def conversion_factors(self, include_utility_sector_efficiency_in_total_energy_intensity=True):
        """can streamline this function 

        Returns:
            [type]: [description]
        """        
                                              
        if self.sector == 'residential':
            datasource =  GetEIAData.eia_api(id_='711250')  # AnnualData_MER_22_Dec2019

        elif self.sector == 'commercial': 
            datasource = GetEIAData.eia_api(id_='711251') # mer_data23_Dec_2019
                                  
        elif self.sector == 'industrial': 
            datasource =  GetEIAData.eia_api(id_='711252') # MER_Nov19_Table24

        else: # Electricity and Tranportation don't use conversion factors
            return None
        
        sector_name = self.sector.capitalize()
        conversion_factors_df = datasource[['Annual Total', f'Electricity Retail Sales to the {sector_name} Sector (Trillion Btu)',
                                                                f'{sector_name} Sector Electrical System Energy Losses (Trillion Btu)']]
        conversion_factors_df = conversion_factors_df.rename(columns={'Annual Total: 
                                                                            'year', 
                                                                        f'Electricity Retail Sales to the {sector_name} Sector (Trillion Btu)':
                                                                            'electricity_retail_sales', 
                                                                        f'{sector_name} Sector Electrical System Energy Losses (Trillion Btu)':
                                                                            'electrical_system_energy_losses'})       

        conversion_factors_df['Losses/Sales'] = conversion_factors_df['electrical_system_energy_losses'].div(conversion_factors_df['electricity_retail_sales'])  
        conversion_factors_df['source-site conversion factor'] = conversion_factors_df['Losses/Sales'].add(1)
        base_year_source_site_conversion_factor = conversion_factors_df[conversion_factors_df['year'] == base_year]['source-site conversion factor'].values()[0]
        conversion_factors_df['conversion factor index'] = conversion_factors_df['source-site conversion factor'].div(base_year_source_site_conversion_factor)
        
        if include_utility_sector_efficiency_in_total_energy_intensity:
            conversion_factors_df['utility efficiency adjustment factor'] = conversion_factors_df['conversion factor index']
            conversion_factors_df['selected site-source conversion factor'] = conversion_factors_df['source-site conversion factor']
        else: 
            conversion_factors_df['utility efficiency adjustment factor'] = 1
            conversion_factors_df['selected site-source conversion factor'] = base_year_source_site_conversion_factor

        return conversion_factors_df['selected site-source conversion factor']


eia_data_cat = GetEIAData.eia_api(id_='711250', id_type='category')
# GetEIAData.eia_api(id_='711250')
print(eia_data_cat)
# eia_data = GetEIAData.eia_api(id_='TOTAL.NGACPUS.M', id_type='series')
# print(eia_data)
print('done')
