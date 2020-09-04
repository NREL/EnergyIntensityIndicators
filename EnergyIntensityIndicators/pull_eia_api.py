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

    def eia_api(self, id_, id_type='category'):
        """[summary]

        Args:
            id_ ([type]): [description]
            id_type (str, optional): [description]. Defaults to 'category'.

        Returns:
            [type]: [description]
        """          
        api_key = os.environ.get("EIA_API_Key")

        if id_type == 'category':
            eia_data = self.get_category(api_key, id_)
        elif id_type == 'series':
            eia_data = self.get_series(api_key, id_)
        else:
            eia_data = None
            print('Error: neither series nor category given')
        return eia_data
    
    def get_category(self, api_key, id_):
            api_call = f'http://api.eia.gov/category/?api_key={api_key}&category_id={id_}'
            r = requests.get(api_call)
            data = r.json()
            eia_childseries = data['category']['childseries']
            eia_series_ids = [i['series_id'] for i in eia_childseries]
            eia_data = [self.get_series(api_key, s) for s in eia_series_ids]
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
        elif date_column_name == 'A' | date_column_name == 'Year':
            eia_df['Date'] = pd.to_datetime(eia_df['A'], format='%Y')
            eia_df = eia_df.drop('A', axis='columns')
        else:
            print('No date column')
            pass
        return eia_df

    def get_seds(self):
        """Load and format energy consumption data
        Used for commercial (ESCCB and TNCCB) and residential (ESCRB and TNRCB)
        './EnergyIntensityIndicators/use_all_btu.csv'
           https://www.eia.gov/state/seds/seds-data-complete.php?sid=US
        """    
        consumption_all_btu = pd.read_csv('https://www.eia.gov/state/seds/sep_use/total/csv/use_all_btu.csv')  # Commercial: '40210 , residential : '40209 
                                                                                          # 1960 through 2017 SEDS Data, MSN refers to fuel type
        state_to_census_region = pd.read_csv('./EnergyIntensityIndicators/state_to_census_region.csv')
        state_to_census_region = state_to_census_region.rename(columns={'USPC': 'State'})
        consumption_census_region = consumption_all_btu.merge(state_to_census_region, on='State', how='outer')


        years = list(range(1960, 2018))
        years = [str(year) for year in years]
        
        consumption_census_region = consumption_census_region[['Census Region', 'MSN'] + years]

        if self.sector == 'residential':
            consumption_census_region = consumption_census_region[consumption_census_region['MSN'].isin(['ESRCB', 'TNRCB'])]
            consumption_census_region = consumption_census_region.set_index(['MSN', 'Census Region'])

            consumption_census_region = consumption_census_region.stack().reset_index().rename(columns={'level_2': 'year', 0: 'value'})

            ESRCB_by_region = consumption_census_region[consumption_census_region['MSN'] == 'ESRCB'].drop('MSN', axis=1) 
            ESRCB_by_region = pd.pivot_table(ESRCB_by_region, index='year', columns='Census Region', values='value') 

            TNRCB_by_region = consumption_census_region[consumption_census_region['MSN'] == 'TNRCB'].drop('MSN', axis=1)
            TNRCB_by_region = pd.pivot_table(TNRCB_by_region, index='year', columns='Census Region', values='value')  

            elec_to_indicators = ESRCB_by_region[[1, 2, 3, 4]].multiply(0.001)
            elec_to_indicators['National'] = elec_to_indicators.sum(1)

            total_primary = TNRCB_by_region[[1, 2, 3, 4]].subtract(ESRCB_by_region[[1, 2, 3, 4]])
            total_fuels_to_indicators = total_primary.multiply(0.001)
            total_fuels_to_indicators['National'] = total_fuels_to_indicators.sum(1)

        elif self.sector == 'commercial':
            consumption_census_region = consumption_census_region[consumption_census_region['MSN'].isin(['ESCCB', 'TNCCB'])]
            consumption_census_region = consumption_census_region.set_index(['MSN', 'Census Region'])

            consumption_census_region = consumption_census_region.stack().reset_index().rename(columns={'level_2': 'year', 0: 'value'})

            ESCCB_by_region = consumption_census_region[consumption_census_region['MSN'] == 'ESCCB'].drop('MSN', axis=1) 
            ESCCB_by_region = pd.pivot_table(ESCCB_by_region, index='year', columns='Census Region', values='value') 

            TNCCB_by_region = consumption_census_region[consumption_census_region['MSN'] == 'TNCCB'].drop('MSN', axis=1)
            TNCCB_by_region = pd.pivot_table(TNCCB_by_region, index='year', columns='Census Region', values='value')

            elec_to_indicators = ESCCB_by_region[[1, 2, 3, 4]].multiply(0.001)
            elec_to_indicators['National'] = elec_to_indicators.sum(1)

            total_primary = TNCCB_by_region[[1, 2, 3, 4]].subtract(ESCCB_by_region[[1, 2, 3, 4]])
            total_fuels_to_indicators = total_primary.multiply(0.001)
            total_fuels_to_indicators['National'] = total_fuels_to_indicators.sum(1)

        else:
            return None
        
        return total_fuels_to_indicators, elec_to_indicators

    def national_calibration(self):
        """Calibrate SEDS energy consumption data to most recent data from the Annual or Monthly Energy Review

        TODO: 
        The whole point of this is to reconcile the AER and MER data, so they shouldn't be the same API endpoint
        """
        if self.sector == 'residential':
            AER11_table2_1b_update = pd.read_csv('https://www.eia.gov/totalenergy/data/browser/xls.php?tbl=T02.02') #  self.eia_api(id_='711250')
            AnnualData_MER_22_Dec2019 = pd.read_csv('https://www.eia.gov/totalenergy/data/browser/csv.php?tbl=T02.02') # self.eia_api(id_='711250')
            
              # .eia_api(id_=, id_type='category')
            electricity_retail_sales_residential_sector = self.eia_api(id_='TOTAL.ESRCBUS.A', id_type='series')
            total_primary_energy_consumed_residential_sector = self.eia_api(id_='TOTAL.TXRCBUS.A', id_type='series')

            fuels_census_region, electricity_census_region = self.get_seds()  
            electricity_df = pd.DataFrame()
            electricity_df['AER 11 (Billion Btu)'] = AER11_table2_1b_update['Electricity Retail Sales']  # Column S
            electricity_df['MER, 12/19 (Trillion Btu)'] =  electricity_retail_sales_residential_sector # AnnualData_MER_22_Dec2019['Electricity Retail Sales to the Residential Sector'] # Column K
            electricity_df['SEDS (10/18) (Trillion Btu)'] = electricity_census_region['National']  # Column G
            electricity_df['Ratio MER/SEDS'] = electricity_df['MER, 12/19 (Trillion Btu)'].div(electricity_df['SEDS (10/18) (Trillion Btu)'])
            electricity_df['Final Est. (Trillion Btu)'] = electricity_df['SEDS (10/18) (Trillion Btu)'].multiply(electricity_df['Ratio MER/SEDS'])
            # If SEDS is 0, replace with MER

            fuels_df['AER 11 (Billion Btu)'] = AER11_table2_1b_update['Total Primary'] # Column Q
            fuels_df['MER, 12/19 (Trillion Btu)'] =  total_primary_energy_consumed_residential_sector # AnnualData_MER_22_Dec2019['Total Primary Energy Consumed by the Residential Sector']# Column J
            fuels_df['SEDS (10/18) (Trillion Btu)'] = fuels_census_region['National'] # Column N
            fuels_df['Ratio MER/SEDS'] = fuels_df['MER, 12/19 (Trillion Btu)'].div(fuels_df['SEDS (10/18) (Trillion Btu)'])
            fuels_df['Final Est. (Trillion Btu)'] = fuels_df['SEDS (10/18) (Trillion Btu)'].multiply(fuels_df['Ratio MER/SEDS'])
            # If SEDS is 0, replace with MER

            # Not sure if these are needed
            recs_btu_hh = electricity_df['SEDS (10/18) (Trillion Btu)'].add(fuels_df['SEDS (10/18) (Trillion Btu)']).div(recs_millions)  # How do order of operations work here ?? (should be add and then divide)
            calibrated_hh = # National column N
            aer_btu_hh =  electricity_df['MER, 12/19 (Trillion Btu)'].add(fuels_df['MER, 12/19 (Trillion Btu)']).div(calibrated_hh)  # How do order of operations work here ?? (should be add and then divide)
        
        elif self.sector == 'commercial':
            electricity_retail_sales_commercial_sector = self.eia_api(id_='TOTAL.ESCCBUS.A', id_type='series')
            total_primary_energy_consumed_commercial_sector = self.eia_api(id_='TOTAL.TXCCBUS.A', id_type='series')


            AER11_Table21C_Update = pd.read_excel('https://www.eia.gov/totalenergy/data/browser/xls.php?tbl=T02.03')  # GetEIAData.eia_api(id_='711251')
            mer_data23_Dec_2019 = pd.read_csv()  # GetEIAData.eia_api(id_='711251')
            fuels_census_region, electricity_census_region = self.get_seds()
            electricity_df = pd.DataFrame()
            electricity_df['AER 11 (Billion Btu)'] = AER11_Table21C_Update['Electricity Retail Sales'] # Column W
            electricity_df['MER, 12/19 (Trillion Btu)'] = electricity_retail_sales_commercial_sector # mer_data23_Dec_2019['Electricty Retail Sales to the Commercial Sector'] # Column M
            electricity_df['SEDS (01/20) (Trillion Btu)'] =  electricity_census_region['National'] # Column G
            electricity_df['Ratio MER/SEDS'] = electricity_df['MER, 12/19 (Trillion Btu)'].div(electricity_df['SEDS (01/20) (Trillion Btu)']])
            electricity_df['Final Est. (Trillion Btu)'] = electricity_df['SEDS (01/20) (Trillion Btu)']].multiply(electricity_df['Ratio MER/SEDS'])

            fuels_df = pd.DataFrame()
            fuels_df['AER 11 (Billion Btu)'] = AER11_Table21C_Update['Total Primary'] # Column U
            fuels_df['MER, 12/19 (Trillion Btu)'] = total_primary_energy_consumed_commercial_sector # mer_data23_Dec_2019['Total Primary Energy Consumed by the Commercial Sector']  # Column L
            fuels_df['SEDS (01/20) (Trillion Btu)'] = fuels_census_region['National']  # Column N
            fuels_df['Ratio MER/SEDS'] = fuels_df['MER, 12/19 (Trillion Btu)'].div(fuels_df['SEDS (01/20) (Trillion Btu)'])
            fuels_df['Final Est. (Trillion Btu)'] = fuels_df['SEDS (01/20) (Trillion Btu)'].multiply(fuels_df['Ratio MER/SEDS'])
            # If SEDS is 0, replace with MER
    
        else: 
            pass

        national_calibration = electricity_df.merge(fuels_df, on='year', how='outer')
        return national_calibration

    def conversion_factors(self, include_utility_sector_efficiency_in_total_energy_intensity=True):
        """Not sure if this is correct class method use

        Returns:
            [type]: [description]
        """        
                                              
        if self.sector == 'residential':
            electricity_retail_sales = self.eia_api(id_='TOTAL.ESRCBUS.A', id_type='series') # electricity retail sales to the residential sector
            electrical_system_energy_losses = self.eia_api(id_='TOTAL.LORCBUS.A', id_type='series')  # Residential Sector Electrical System Energy Losses

        elif self.sector == 'commercial': 
            electricity_retail_sales = self.eia_api(id_='TOTAL.ESCCBUS.A', id_type='series') # electricity retail sales to the commercial sector
            electrical_system_energy_losses = self.eia_api(id_='TOTAL.LOCCBUS.A', id_type='series')  # Commercial Sector Electrical System Energy Losses
                                  
        elif self.sector == 'industrial': 
            electricity_retail_sales = self.eia_api(id_='TOTAL.ESICBUS.A', id_type='series') # electricity retail sales to the industrial sector
            electrical_system_energy_losses = self.eia_api(id_='TOTAL.LOICBUS.A', id_type='series') # Industrial Sector Electrical System Energy Losses

        else: # Electricity and Tranportation don't use conversion factors
            return None
        
        sector_name = self.sector.capitalize()
        conversion_factors_df = pd.DataFrame([electricity_retail_sales, electrical_system_energy_losses]).transpose().columns(['electricity_retail_sales', 'electrical_system_energy_losses'])  

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



eia_data_cat = GetEIAData('residential').eia_api(id_='TOTAL.ESRCBUS.A', id_type='series')
# GetEIAData.eia_api(id_='711250')
print(eia_data_cat.columns)
print(eia_data_cat)
# eia_data = GetEIAData.eia_api(id_='TOTAL.NGACPUS.M', id_type='series')
# print(eia_data)
print('done')
