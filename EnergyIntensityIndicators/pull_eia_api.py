import os
import json
import requests
import pandas as pd
import numpy as np
from functools import reduce 

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
        
        eia_data['Year'] = eia_data['Year'].apply(lambda y: y.strftime('%Y'))
        eia_data = eia_data.set_index('Year').sort_index(ascending=True)
        return eia_data
    
    def get_category(self, api_key, id_):
        """Collect categorical data from EIA API by merging data for all child series
        """        
        api_call = f'http://api.eia.gov/category/?api_key={api_key}&category_id={id_}'
        r = requests.get(api_call)
        data = r.json()
        eia_childseries = data['category']['childseries']
        eia_series_ids = [i['series_id'] for i in eia_childseries]
        eia_data = [self.get_series(api_key, s) for s in eia_series_ids]
        print('eia_data: \n', eia_data)
        all_category = reduce(lambda x, y: pd.merge(x, y, on ='Year'), eia_data)
        return all_category

    @staticmethod
    def get_series(api_key, id_):
        """Collect series data from EIA API, format in dataframe with year as index
        """        
        api_call = f'http://api.eia.gov/series/?api_key={api_key}&series_id={id_}'
        r = requests.get(api_call)
        eia_data = r.json()
        date_column_name = str(eia_data['series'][0]['f'])
        data_column_name =  str(eia_data['series'][0]['name']) + ', ' + str(eia_data['series'][0]['units'])
        eia_df = pd.DataFrame.from_dict(eia_data['series'][0]['data'])
        eia_df = eia_df.rename(columns={0: date_column_name, 1: data_column_name})
        if date_column_name == 'M':
            eia_df['Year'] = pd.to_datetime(eia_df['M'], format='%Y%m') # .dt.to_period('Y')
            eia_df = eia_df.drop('M', axis='columns')
        elif date_column_name == 'A' or date_column_name == 'Year':
            eia_df['Year'] = pd.to_datetime(eia_df['A'], format='%Y') #.dt.to_period('Y')
            eia_df = eia_df.drop('A', axis='columns')
        else:
            print('No year column')
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
        state_to_census_region = pd.read_csv('./Data/state_to_census_region.csv')
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
            AER11_table2_1b_update = pd.read_excel('https://www.eia.gov/totalenergy/data/browser/xls.php?tbl=T02.02', skiprows=10, header=0).drop(0, axis=0)
            AER11_table2_1b_update = AER11_table2_1b_update.replace({'Not Available': np.nan})
            AER11_table2_1b_update['Month'] = pd.to_datetime(AER11_table2_1b_update['Month'], format='%Y-%m-%d')
            AER11_table2_1b_update['Year'] = pd.DatetimeIndex(AER11_table2_1b_update['Month']).year
            AER11_table2_1b_update = AER11_table2_1b_update.groupby(by=['Year']).sum()  # add groupby(dropna=False) when that feature is released #  self.eia_api(id_='711250')
            AnnualData_MER_22_Dec2019 = pd.read_csv('https://www.eia.gov/totalenergy/data/browser/csv.php?tbl=T02.02') # self.eia_api(id_='711250')
            
              # .eia_api(id_=, id_type='category')
            electricity_retail_sales_residential_sector = self.eia_api(id_='TOTAL.ESRCBUS.A', id_type='series')
            # print('MER: \n', electricity_retail_sales_residential_sector)
            total_primary_energy_consumed_residential_sector = self.eia_api(id_='TOTAL.TXRCBUS.A', id_type='series')

            fuels_census_region, electricity_census_region = self.get_seds()  
            # print('SEDS: \n', fuels_census_region, electricity_census_region)
            electricity_df = pd.DataFrame()
            electricity_df['AER 11 (Billion Btu)'] = AER11_table2_1b_update['Electricity Retail Sales to the Residential Sector']  # Column S Electricity Retail Sales
            electricity_df['MER, 12/19 (Trillion Btu)'] =  electricity_retail_sales_residential_sector # AnnualData_MER_22_Dec2019['Electricity Retail Sales to the Residential Sector'] # Column K
            electricity_df['SEDS (10/18) (Trillion Btu)'] = electricity_census_region['National']  # Column G
            electricity_df['Ratio MER/SEDS'] = electricity_df['MER, 12/19 (Trillion Btu)'].div(electricity_df['SEDS (10/18) (Trillion Btu)'].values)
            electricity_df['Final Est. (Trillion Btu)'] = electricity_df['SEDS (10/18) (Trillion Btu)'].multiply(electricity_df['Ratio MER/SEDS'].values)
            # If SEDS is 0, replace with MER
            electricity_df['Final Est. (Trillion Btu)'] = electricity_df['Final Est. (Trillion Btu)'].fillna(electricity_df['MER, 12/19 (Trillion Btu)'])

            fuels_df = pd.DataFrame()
            fuels_df['AER 11 (Billion Btu)'] = AER11_table2_1b_update['Total Primary Energy Consumed by the Residential Sector'] # Column Q
            fuels_df['MER, 12/19 (Trillion Btu)'] =  total_primary_energy_consumed_residential_sector # AnnualData_MER_22_Dec2019['Total Primary Energy Consumed by the Residential Sector']# Column J
            fuels_df['SEDS (10/18) (Trillion Btu)'] = fuels_census_region['National'] # Column N
            fuels_df['Ratio MER/SEDS'] = fuels_df['MER, 12/19 (Trillion Btu)'].div(fuels_df['SEDS (10/18) (Trillion Btu)'].values)
            fuels_df['Final Est. (Trillion Btu)'] = fuels_df['SEDS (10/18) (Trillion Btu)'].multiply(fuels_df['Ratio MER/SEDS'].values)
            # If SEDS is 0, replace with MER
            fuels_df['Final Est. (Trillion Btu)'] = fuels_df['Final Est. (Trillion Btu)'].fillna(fuels_df['MER, 12/19 (Trillion Btu)'])
            # # Not sure if these are needed
            # recs_btu_hh = electricity_df['SEDS (10/18) (Trillion Btu)'].add(fuels_df['SEDS (10/18) (Trillion Btu)']).div(recs_millions)  # How do order of operations work here ?? (should be add and then divide)
            # # calibrated_hh = [0] # National column N
            # aer_btu_hh =  electricity_df['MER, 12/19 (Trillion Btu)'].add(fuels_df['MER, 12/19 (Trillion Btu)']).div(calibrated_hh)  # How do order of operations work here ?? (should be add and then divide)
        
        elif self.sector == 'commercial':
            electricity_retail_sales_commercial_sector = self.eia_api(id_='TOTAL.ESCCBUS.A', id_type='series')
            total_primary_energy_consumed_commercial_sector = self.eia_api(id_='TOTAL.TXCCBUS.A', id_type='series')
            # AER11_Table21C_Update = pd.read_excel('https://www.eia.gov/totalenergy/data/browser/xls.php?tbl=T02.03', header=10)  # GetEIAData.eia_api(id_='711251')
            # AER11_Table21C_Update = AER11_Table21C_Update.query('Month != "NaT"')
            AER11_Table21C_Update = pd.read_csv('https://www.eia.gov/totalenergy/data/browser/csv.php?tbl=T02.03')
            # AER11_Table21C_Update['YYYYMM'] = pd.to_datetime(AER11_Table21C_Update['YYYYMM'], format='%Y%m')
            AER11_Table21C_Update['YYYYMM'] = AER11_Table21C_Update['YYYYMM'].astype(str)
            AER11_Table21C_Update['Month'] = AER11_Table21C_Update['YYYYMM'].str[-2:]
            AER11_Table21C_Update['Year'] = AER11_Table21C_Update['YYYYMM'].str[:-2]
            AER11_Table21C_Update = AER11_Table21C_Update.query('Month == "13"')
            AER11_Table21C_Update = AER11_Table21C_Update.set_index('Year').sort_index(ascending=True)
            aer_retail_sales_tbtu = AER11_Table21C_Update[AER11_Table21C_Update['MSN']=='ESCCBUS']['Value'].astype(float)
            aer_retail_sales_bbtu = aer_retail_sales_tbtu.divide(1000)

            aer_total_primary_tbtu = AER11_Table21C_Update[AER11_Table21C_Update['MSN']=='TXCCBUS']['Value'].astype(float)
            aer_total_primary_bbtu = aer_total_primary_tbtu.divide(1000)

            mer_data23_Dec_2019 = self.eia_api(id_='711251')  # pd.read_csv()
            fuels_census_region, electricity_census_region = self.get_seds()

            electricity_df = pd.DataFrame()
            electricity_df['AER 11 (Billion Btu)'] = aer_retail_sales_bbtu # Column W
            electricity_df['MER, 12/19 (Trillion Btu)'] = electricity_retail_sales_commercial_sector # mer_data23_Dec_2019['Electricty Retail Sales to the Commercial Sector'] # Column M
            electricity_df['SEDS (01/20) (Trillion Btu)'] =  electricity_census_region['National'] # Column G
            electricity_df['SEDS (01/20) (Trillion Btu)']= electricity_df['SEDS (01/20) (Trillion Btu)'].fillna(electricity_df['MER, 12/19 (Trillion Btu)'])

            electricity_df['Ratio MER/SEDS'] = electricity_df['MER, 12/19 (Trillion Btu)'].div(electricity_df['SEDS (01/20) (Trillion Btu)'].values)
            electricity_df['Final Est. (Trillion Btu)'] = electricity_df['SEDS (01/20) (Trillion Btu)'].multiply(electricity_df['Ratio MER/SEDS'].values)

            fuels_df = pd.DataFrame()
            fuels_df['AER 11 (Billion Btu)'] = aer_total_primary_bbtu # Column U
            fuels_df['MER, 12/19 (Trillion Btu)'] = total_primary_energy_consumed_commercial_sector # mer_data23_Dec_2019['Total Primary Energy Consumed by the Commercial Sector']  # Column L
            fuels_df['SEDS (01/20) (Trillion Btu)'] = fuels_census_region['National']  # Column N
            fuels_df['SEDS (01/20) (Trillion Btu)']= fuels_df['SEDS (01/20) (Trillion Btu)'].fillna(fuels_df['MER, 12/19 (Trillion Btu)'])
            fuels_df['Ratio MER/SEDS'] = fuels_df['MER, 12/19 (Trillion Btu)'].div(fuels_df['SEDS (01/20) (Trillion Btu)'].values)
            fuels_df['Final Est. (Trillion Btu)'] = fuels_df['SEDS (01/20) (Trillion Btu)'].multiply(fuels_df['Ratio MER/SEDS'].values)
            # If SEDS is 0, replace with MER
            # fuels_df['Final Est. (Trillion Btu)'] = fuels_df['Final Est. (Trillion Btu)'].fillna(fuels_df['MER, 12/19 (Trillion Btu)'], inplace=True)

        else: 
            pass

        national_calibration = electricity_df.merge(fuels_df, left_index=True, right_index=True, how='outer', suffixes=['_elec','_fuels'])
        return national_calibration

    def conversion_factors(self, include_utility_sector_efficiency_in_total_energy_intensity=False):
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
        col_rename = {f'Electricity Retail Sales to the {sector_name} Sector, Annual, Trillion Btu': 'electricity_retail_sales', f'{sector_name} Sector Electrical System Energy Losses, Annual, Trillion Btu': 'electrical_system_energy_losses'}
        conversion_factors_df = electricity_retail_sales.merge(electrical_system_energy_losses, how='outer', on='Year').rename(columns=col_rename)
        conversion_factors_df['Losses/Sales'] = conversion_factors_df['electrical_system_energy_losses'].div(conversion_factors_df['electricity_retail_sales'])  
        conversion_factors_df['source-site conversion factor'] = conversion_factors_df['Losses/Sales'].add(1)
        base_year_source_site_conversion_factor = conversion_factors_df.loc['1985', ['source-site conversion factor']].values[0]
        conversion_factors_df['conversion factor index'] = conversion_factors_df['source-site conversion factor'].div(base_year_source_site_conversion_factor)
        if include_utility_sector_efficiency_in_total_energy_intensity:
            conversion_factors_df['utility efficiency adjustment factor'] = conversion_factors_df['conversion factor index']
            conversion_factors_df['selected site-source conversion factor'] = conversion_factors_df['source-site conversion factor']
        else: 
            conversion_factors_df['utility efficiency adjustment factor'] = 1
            conversion_factors_df['selected site-source conversion factor'] = base_year_source_site_conversion_factor

        return conversion_factors_df[['selected site-source conversion factor']]


