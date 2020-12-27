import pandas as pd
import os
import requests
import zipfile
from zipfile import ZipFile
import numpy as np
import scipy
from scipy.optimize import leastsq, least_squares, minimize
import matplotlib.pyplot as plt
from statistics import mean
import io
from functools import reduce



class ResidentialFloorspace:
    """Calculate regional and national estimates of residential
    floorspace based upon model to interpolate/smooth AHS estimates
    """    
    def __init__(self, end_year=2018):
        self.end_year = end_year 
        pass
    
    @staticmethod
    def update_ahs_data():
        """Spreadsheet equivalent: AHS_2017_extract
        Extract and process American Housing Survey (AHS, formerly Annual Housing Survey) 
        web: https://www.census.gov/programs-surveys/ahs/data.html
        ? https://www.census.gov/programs-surveys/ahs/data/2017/ahs-2017-public-use-file--puf-/ahs-2017-national-public-use-file--puf-.html 
        """    
        ahs_url_folder = 'http://www2.census.gov/programs-surveys/ahs/2017/AHS%202017%20National%20PUF%20v3.0%20CSV.zip?#' 

        if os.path.exists('C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/household.csv'):
            print('AHS data already ready')
            pass
        else: 
            r = requests.get(ahs_url_folder, stream=True)
            print('AHS data get successful:', r.ok)
            z = ZipFile(io.BytesIO(r.content))
            z.extract('household.csv', path='C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/')

        ahs_household_data = pd.read_csv('C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/household.csv')
        columns = ['JYRBUILT', 'WEIGHT', 'YRBUILT', 'DIVISION', 'BLD', 'UNITSIZE', 'VACANCY']
        extract_ahs = ahs_household_data[columns]
        extract_ahs = extract_ahs.replace(to_replace={"'": ""})
        extract_ahs['BLD'] = extract_ahs['BLD'].replace(to_replace="'", value='')
        extract_ahs['DIVISION'] = extract_ahs['DIVISION'].replace(to_replace="'", value='')
        extract_ahs['UNITSIZE'] = extract_ahs['UNITSIZE'].replace(to_replace="'", value='')
        extract_ahs['BLD'] = extract_ahs['BLD'].replace(to_replace="'", value='')

        vals_list = ["'04'", "'05'", "'06'", "'07'", "'08'", "'09'"]

        extract_ahs= extract_ahs[extract_ahs['BLD'].isin(vals_list)]
        housing_types = ['single_family', 'multifamily', 'manufactured_homes']
        housing_occupancy_types = [f'{h}_total' for h in housing_types] + [f'{h}_occupied' for h in housing_types]
        housing_type_number = dict(zip()) # match the numbers in ['04', '05', '06', '07', '08', '09'] to housing/occupancy type
        #  For the total number of units of each building type, the variable
        # s, the pivot table processing
        # generated six tables shown below the active pivot table in the first 16 rows of the worksheet. The six
        # tables are for Single Family (SF) units (Total and Occupied), Multi-Family (MF) units (Total and Occupied),
        # and Manufactured Home (Man.Homes) units (Total and Occupied). These tables were obtained by
        # copying the values from the active pivot table at the top of the worksheet, each table reflecting a
        # different choice for VACANCY and BLD. VACANCY is set to “All”. For occupied units, set the VACANCY variable to ‘-6’. For the building type,
        # single family units are obtained by selecting building code ‘02’ and ’03’ (corresponding to detached and
        # attached units). Multi-family units are selected by checking the boxes under BLD, ‘04’ through ‘09’).
        # Manufactured homes are selected with a check of the code ’01.’

        # Pivot Tables!!
        #  Rows correspond to various size categories columns correspond to census division which are then aggregated to census regions. 
        #  Only Categorical size data were included in the AHS surveys for 2015 and 2017. An effort to 
        # utilize the category data to estimate the average size change between 2015 and 2017 did not provide useful results. Processing of future AHS public use
        # files may exlcude any consideration of this variable. In the current tables, only the values for total units across all sizes were considered.
        pivot_census_division = pd.pivot_table(extract_ahs, values='WEIGHT' , index=['BLD', 'UNITSIZE'] , columns='DIVISION' , aggfunc='sum')  
        pivot_census_division['census_region_1'] = pivot_census_division["'1'"] + pivot_census_division["'2'"]   # alternative method: df['C'] = df.apply(lambda row: row['A'] + row['B'], axis=1)
        pivot_census_division['census_region_2'] = pivot_census_division["'3'"] + pivot_census_division["'4'"]
        pivot_census_division['census_region_3'] = pivot_census_division["'5'"] + pivot_census_division["'6'"] + pivot_census_division["'7'"]
        pivot_census_division['census_region_4'] = pivot_census_division["'8'"] + pivot_census_division["'9'"]
        pivot_census_division['total'] = pivot_census_division[['census_region_1', 'census_region_2', 'census_region_3', 'census_region_4']].sum(axis=1)
        # pivot_census_division = pivot_census_division.reset_index('BLD')
        # pivot_census_division = pivot_census_division.reindex(pivot_census_division.index + ['Grand Total'])
        # pivot_census_division.loc['Grand Total'] = pivot_census_division.sum(axis=0)

        return pivot_census_division        

    def get_ahs_tables(self):
        """Collect AHS historical tables"""
        historical_ahs = pd.read_csv('../AHS_Historical_Tables.csv')
        #  historical_ahs['Years'] = historical_ahs['Years'].astype(int)
        for year in list(historical_ahs['Years']):
            df = historical_ahs[historical_ahs['Years'] == year]
            regions = ['National', 'West', 'Northeast', 'South', 'Midwest']
            for region in regions: 
                columns_ = [c for c in list(range(df.shape[1])) if df.iloc[1, [c]].values == region]
                region_df = df[df.ix[:, columns_]] 
        pass

    @staticmethod
    def get_percent_remaining_surviving(factor, lifetime, gamma):
        """Calculate percent remaining surviving based (of a housing unit)
        """
        return 1 / (1 + (factor / lifetime) ** gamma)   

    @staticmethod
    def interpolate_with_avg(dataframe, columns, even=True):
        """Use average to interpolate dataframe

        Parameters:
            dataframe (df): dataframe with year index 
            columns (list): names of columns to interpolate with average
            even (bool): whether the years to fill are even (or odd if False)

        """ 
        if even:
            years_to_fill = [year for year in dataframe.index if year % 2 == 0]
        else:
            years_to_fill = [year for year in dataframe.index if year % 2 != 0]
        for c in columns:
           for y in years_to_fill:
                if y > min(dataframe.index) and y + 1 in dataframe.index:
                    value_before = dataframe.loc[y - 1, [c]].values[0]
                    value_after = dataframe.loc[y + 1, [c]].values[0]
                    value = mean([value_before, value_after])
                    dataframe.loc[y, [c]] = value
        return dataframe
                     
    @staticmethod
    def get_place_nsa_all():
        """Scrape placement data for manufactured homes"""
        years = list(range(1994, 2013 + 1))
        for year in years:
            url = f'http://www2.census.gov/programs-surveys/mhs/tables/{str(year)}/stplace{str(year)[-2:]}.xls'
            placement_df = pd.read_excel(url, index_col=0, skiprows=4, use_cols='B, F:H') # Placement units are thousands of units
        
    @staticmethod
    def housing_stock_model(year_array, new_comps_ann, pub_total_0, elasticity_of_retirements, coeffs, full_data=False, retirement=False):
        """Calculate housing stock using provided coefficients
        """          
        for index_, year_ in enumerate(year_array):
            if index_ == 0: 
                existing_stock = pub_total_0 + coeffs[0]
                predicted_retirement = 0
                predicted_retirement_series = np.array([predicted_retirement])
                new_units = 0
                existing_stock_series = np.array([existing_stock])
                predicted_total_stock_series = np.array([existing_stock])
            else:
                new_units = ((new_comps_ann[index_] + new_comps_ann[index_-1]) / 2 ) * coeffs[2] 
                adjusted_new_units = np.sign(new_units) * (np.abs(new_units)) ** elasticity_of_retirements
                existing_stock = predicted_total_stock_series[index_ - 1]
                predicted_retirement = (-1 * existing_stock) * adjusted_new_units * coeffs[1]
                predicted_total_stock = existing_stock + predicted_retirement + new_units
            
                predicted_retirement_series = np.vstack([predicted_retirement_series, predicted_retirement])
                existing_stock_series = np.vstack([existing_stock_series, existing_stock])
                predicted_total_stock_series = np.vstack([predicted_total_stock_series, predicted_total_stock])
        
        predicted_total_stock_series = predicted_total_stock_series.flatten()
        predicted_total_stock_series_skip = predicted_total_stock_series[0::2]
        
        if retirement:
            return predicted_retirement_series

        if full_data:
            return predicted_total_stock_series
        else:
            return predicted_total_stock_series_skip

    def residuals_avg_size_sf(self, coeffs, actual_size, input_data):
        """Calculate residuals, for use in optimization of single-family housing avg size model"""
        residuals = actual_size - self.model_average_housing_unit_size_sf(coeffs, input_data).values
        return residuals
    
    @staticmethod
    def model_average_housing_unit_size_sf(coeffs, df):
        """Model of single-family housing avg size
        """

        # year_array = np.array(df.index)
        # for y in year_array:
        #     if y == 1985: 

        df.loc[1997, ['BN']] = coeffs[0]
        df.loc[1985, ['BM']] = df.loc[1985, ['occupied_predicted']].values
        df.loc[1985, ['BL']] = df.loc[1985, ['BM']].values
        for y in df.index:
            if y > 1985: 
                df.loc[y, ['BM']] = df.loc[1985, ['BM']].add(sum(df.loc[1985 : y, ['occupied_predicted']].values))
                df.loc[y, ['BL']] = ((df.loc[y, ['post_1984_units']].values + df.loc[y - 1, ['post_1984_units']].values) * 0.5 * coeffs[2]) + df.loc[y, ['BM']].values

            df.loc[y, ['BN']] = df.loc[1997, ['BN']].add(coeffs[1]).multiply((y - 1997))

        df['total_sqft_pre_1985'] = df['BM'].multiply(df['BN'].values)  # BO
        df['total_sqft_post_1985'] = df['post_1984_units'].multiply(df['avg_size_post84_units']).multiply(coeffs[2]).multiply(1.15) # BQ
        df['predicted_avg_size'] = (df['total_sqft_pre_1985'].add(df['total_sqft_post_1985'].values)).divide(df['BL'].values)
        return df['predicted_avg_size']

    def get_housing_size_sf(self, df): 
        """Optimize prediction of average housing stock for single-family homes

        Args:
            df (dataframe): Dataframe containing column 'occupied_predicted' (predicted number housing units)
        """        
        # CSV for 3 average housing size columns
        if self.end_year > max(df.index) or not os.path.exists(f'./EnergyIntensityIndicators/Residential/resuts_{self.end_year}.csv'):

            df = df.reindex(columns=list(df.columns) + ['BI', 'post_1984_units', 'BM', 'BN', 'BL'])
            
            x0 = [1901.39732722429, 7.49900783768688, 1] # Top to bottom, skipping SSRD

            new_comps_ann = pd.read_excel('C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/AHS_summary_results_051720.xlsx', sheet_name='Comps Ann', usecols='A, C').rename(columns={'New Privately Owned Housing Units Completed': 'Years', 'Unnamed: 2': 'new_comps_ann'}).set_index('Years') #  skiprows=26, index_col=0, 
            new_comps_ann = new_comps_ann.loc[list(range(1968, self.end_year + 1)), :]
            new_comps_ann = new_comps_ann.rename(columns={0: 'new_comps_ann'}) # Comps Ann column C
            df = df.merge(new_comps_ann, left_index=True, right_index=True, how='left')

            cnh_avg_size = pd.read_excel('C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/SFTotalMedAvgSqFt.xlsx', sheet_name='data', index_col=0) #) #, usecols='G') , skipfooter=54, skiprows=7, header=8
            cnh_avg_size = cnh_avg_size.loc[list(range(1973, self.end_year + 1)), ['Unnamed: 6']].reset_index().rename(columns={'Median and Average Square Feet of Floor Area in New Single-Family Houses Completed1': 'Years', 
                                                                        'Unnamed: 6': 'cnh_avg_size'})
            cnh_avg_size = cnh_avg_size.set_index('Years')                                                                                                           
            df = df.merge(cnh_avg_size, left_index=True, right_index=True, how='left')

            actual_avg_size = pd.read_excel('C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/AHS_summary_results_051720.xlsx', sheet_name='Total_stock_SF', usecols='AW, BD').rename(columns={'Unnamed: 48': 'Years', 'Size Calc1': 'actual_avg_size'}).set_index('Years')  # from AHS Tables pd.read(csv?)
            df = df[df.index.notnull()]
            df = df.merge(actual_avg_size, left_index=True, right_index=True, how='left')

            df['BH'] = df['new_comps_ann'].multiply(df['cnh_avg_size'].values)
            for year in df.index:
                if year == min(df.index):
                    df.loc[year, ['BI']] = df.loc[year, ['BH']].values
                    df.loc[year, ['post_1984_units']] = df.loc[year, ['new_comps_ann']].values
                elif year == 1984:
                    df.loc[year, ['BI']] = 0
                    df.loc[year, ['post_1984_units']] = 0
                else: 
                    df.loc[year, ['BI']] = df.loc[year - 1, ['BI']].add(df.loc[year, ['BH']].values)
                    df.loc[year, ['post_1984_units']] =  df.loc[year - 1, ['post_1984_units']].add(df.loc[year, ['new_comps_ann']].values)  # BJ
            
            df['avg_size_post84_units'] = df['BI'].divide(df['post_1984_units'].values)  # BK
            
            if os.path.exists('./saved_coeffs.csv'):
                saved_coeffs = pd.read_csv('./saved_coeffs.csv') # structure: row for most recent year, cols are housing unit types
                                                                 # each entry is list of saved coefficients
                try:
                    x = saved_coeffs[(saved_coeffs['Years'] == max(df.index)) & (saved_coeffs['housing_type'] == 'sf')]
                    x = x.drop(['Years', 'housing_type'], axis=1).drop_duplicates().values[0]
                    print('sf x:', x)
                except Exception as e:
                    print('sf failed with error', e)
                    try:
                        x, flag = leastsq(self.residuals_avg_size_sf, x0, args=(df['actual_avg_size'], df), maxfev=1600)
                        print('flag sf:', flag)
                    except RuntimeWarning:
                        print('Warning, optimization timed out, using excel Solver historical coefficients') # should log
                        x = x0

            else:
                x = x0

            df_data = [[max(df.index), 'sf'] + list(x)]


            coeffs_df = pd.DataFrame(data=df_data, columns=['Years', 'housing_type', 'x1', 'x2', 'x3'], index=[0])
            coeffs_df.to_csv('./saved_coeffs.csv', mode='a', index=False, header=False)

            results = self.model_average_housing_unit_size_sf(x, df)

            return results

    def get_housing_size_mf(self, retirement_df, occupied_predicted_mf1):
        """Optimize prediction of average housing stock for multi-family homes

        Parameters:
            new_comps_ann (series): [description]
            coeffs (list): [CM$5, $CM$6, $CM$7]
            df (dataframe): Input data, df contains 'occupied_predicted_mf' and 'retirements' columns (from housing stock mf)

        """        
    #     # just_adjustment = 0.75
        df = retirement_df.reindex(columns=list(retirement_df.columns) + ['BJ', 'BM', 'CB', 'final', 'Predicted-II'])
        
        new_comps_ann = pd.read_excel('C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/AHS_summary_results_051720.xlsx', sheet_name='Comps Ann_2015', skiprows=10, 
                                      usecols='C, E', header=None).dropna().rename(columns={2: 'Years', 4: 'new_comps_ann'}).set_index('Years') # Comps Ann column C

        df = df.merge(new_comps_ann, left_index=True, right_index=True, how='left')

        ahs_table = pd.read_excel('C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/AHS_summary_results_051720.xlsx', sheet_name='Total_stock_MF', skiprows=32, 
                                   usecols='BZ:CI', header=1)[:3] # skipfooter=40,  .rename(columns={'2': 'Years', '4': ''})  # from ahs tables

        ahs_table = ahs_table.rename(columns={'Unnamed: 77': 'Years', '1970-1979': '1975', '1980-1989': '1985', '1990-1999': '1995', '2000 & later': '2005'})
        ahs_table = ahs_table.set_index('Years')
        ahs_table = ahs_table.reindex(list(ahs_table.index) + [2011])
        ahs_table.loc[2011, :] = ahs_table.mean(axis=0)

        actual_size = pd.read_excel('C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/AHS_summary_results_051720.xlsx', sheet_name='Total_stock_MF', skiprows=15, 
                                    usecols='AW, BC', header=None).rename(columns={48: 'Years', 54: 'actual_avg_size'}).set_index('Years')  # from ahs tables
        actual_size = actual_size[actual_size.index.notnull()]

        df = df.merge(actual_size, left_index=True, right_index=True, how='left')

        for i in df.index:
            if i == min(df.index):
                df.loc[i, ['BJ']] = df.loc[i, 'new_comps_ann']
            elif i == 1984:
                df.loc[i, ['BJ']] = 0
            else:
                df.loc[i, ['BJ']] = df.loc[i - 1, ['BJ']].add(df.loc[i, ['new_comps_ann']].values)

        df.loc[1985, ['BM']] = occupied_predicted_mf1
        for year in range(1986, max(df.index)):
            bm_value = occupied_predicted_mf1 + sum(df.loc[list(range(1985, year + 1)), ['predicted_retirement']].values)
            df.loc[year, ['BM']] = bm_value.values

        df['BP'] = df['BJ'].divide(df['BJ'].add(df['BM'].values).values)
        
        decades = list(range(1985, 2005 + 10, 10))

        for d_ in decades: 
            df.loc[d_, ['CB']] = ahs_table.loc[2011, [str(d_)]].values
        adjustment_dict = dict()
        for i, d in enumerate(decades):
            if d + 10 <= 2005:
                decade_after = decades[i + 1]
                decade_after_value = df.loc[decade_after, ['CB']]
                decade_value = df.loc[d, ['CB']]
                adjustment = (decade_after_value.subtract(decade_value.values)).divide(10).values[0]
                adjustment_dict[d] = adjustment
                for y_ in range(d + 1, decade_after):
                    df.loc[y_, ['CB']] = df.loc[y_ - 1, ['CB']] + adjustment

        df.loc[2006, ['CB']] = df.loc[2005, ['CB']].values + adjustment_dict[1995]
        df.loc[2007, ['CB']] = df.loc[2006, ['CB']].values + adjustment_dict[1995]
        df['CB'] = df['CB'].ffill()

        x0 = [567.081097939713, -22.7744777937053, 2.44216497607253]
        
        if os.path.exists('./saved_coeffs.csv'):
            saved_coeffs = pd.read_csv('./saved_coeffs.csv') # structure: row for most recent year, cols are housing unit types
                                                                # each entry is list of saved coefficients
            try:
                x = saved_coeffs[(saved_coeffs['Years'] == max(df.index)) & (saved_coeffs['housing_type'] == 'mf')]
                x = x.drop(['Years', 'housing_type'], axis=1).drop_duplicates().values[0]
            except Exception as e:
                print(f'mf failed with error {e}')
                try:
                    x, flag = leastsq(self.residuals_avg_size_mf, x0, args=(df['actual_avg_size'], df), maxfev=1600)
                    print('flag mf:', flag)
                except RuntimeWarning:
                    print('Warning, optimization timed out, using excel Solver historical coefficients') # should log
                    x = x0

        else:
            x = x0

        df_data = [[max(df.index), 'mf'] + list(x)]

        coeffs_df = pd.DataFrame(data=df_data, columns=['Years', 'housing_type', 'x1', 'x2', 'x3'])
        coeffs_df.to_csv('./saved_coeffs.csv', mode='a', index=False, header=False)

        results = self.average_housing_unit_size_mf(x, df)

        return results

    def residuals_avg_size_mf(self, coeffs, actual_size, input_data):
        """Residuals of avg housing size model for multi-family homes"""
        residuals = actual_size - self.average_housing_unit_size_mf(coeffs, input_data).values
        return residuals

    def average_housing_unit_size_mf(self, coeffs, df):
        """Model of avg housing size model for multi-family homes"""

        year_1997_value = coeffs[0]
        index_year = 1997

        df['Predicted size of pre-1985 stock'] = year_1997_value + coeffs[1] * (df.index - index_year) * 0.75
        df['CN'] = df['CB'] * coeffs[2]
        df['Post-85 shr'] = df['BP'].values 
        df['Ave Size Predicted'] = (1 - df['Post-85 shr']).multiply(df['Predicted size of pre-1985 stock'].values).add(df['Post-85 shr'].multiply(df['CN'].values)).values
        value_1999 = df.loc[1999, ['Ave Size Predicted']].values
        for i in df.index: 
            df.loc[i, ['Predicted-II']] = value_1999 + ((df.loc[i, ['Ave Size Predicted']].values - value_1999) * 0.75)

        df.loc[2010, ['Ave Size Predicted']] = 1027
        cq = mean(df.loc[list(range(2007, 2013 + 1)), ['actual_avg_size']].dropna().values.flatten().tolist())
        for y_ in list(range(2011, self.end_year + 1)):
            df.loc[y_, ['Ave Size Predicted']] = cq

        df.loc[list(range(1985, 1999 + 1)), ['final']] = df.loc[list(range(1985, 1999 + 1)), ['Predicted-II']].values
        df.loc[list(range(2000, max(df.index) + 1)), ['final']] = df.loc[list(range(2000, max(df.index) + 1)), ['Ave Size Predicted']].values

        return df['final']

    @staticmethod
    def model_average_housing_unit_size_mh():
        """Avg housing size model for manufactured homes

        recs_data from  D:\Supporting_Sheets_SRB\[Residential_Floor_Space_2013_SRB.xlsx]REC_Total_SF (2)
           ratio_80_70 from D:\Supporting_Sheets_SRB\[Residential_Floor_Space_2013_SRB.xlsx]RECS_Vintage_size
           ratio_80_74 from D:\Supporting_Sheets_SRB\[Residential_Floor_Space_2013_SRB.xlsx]RECS_Vintage_size
           recs_data[2015] = 1191.2 from RECS_4_adj
        """        
        recs_data = {1980: 826.087, 1984: 843.1373, 1987: 864.7059, 1990: 942.3077, 1993: 989.0909, 1997: 995.2381, 
                     2001: 1061.995, 2005: 1062.009, 2009: 1087} 
        ratio_80_70 = 0.838422
        ratio_80_74 = 0.944122
        recs_data[1970] = recs_data[1980] * ratio_80_70         
        recs_data[1974] = recs_data[1980] * ratio_80_74
        recs_data[2015] = 1191.2 

        manh_data = dict()
        increment_years = sorted(list(recs_data.keys()))
        for index, y_ in enumerate(increment_years):
            if index > 0:
                year_before = increment_years[index - 1]
                num_years = y_ - year_before
                difference = recs_data[y_] - recs_data[year_before]
                increment = difference / num_years
                for delta in range(num_years):
                    value = recs_data[year_before]  + delta * increment
                    year = year_before + delta
                    manh_data[year] = value
            manh_data[y_] = recs_data[y_]

        manh_data[2016] = 1196  # Assume smaller change - 5 sq. ft. per year,  rounded?
        manh_data[2017] = manh_data[2016] + 5 # Assume smaller change - 5 sq. ft. per year 
        manh_size_df = pd.DataFrame.from_dict(manh_data, orient='index', columns=['manh_size'])

        recs_based = manh_size_df.loc[1985:, :]
        return recs_based
    
    def residuals(self, coeffs, actual_stock, year_array, ca, pub_total, elasticity_of_retirements):
        """"Calculate squared resiudals from the housing stock model"""
        residuals = actual_stock - self.housing_stock_model(year_array, ca, pub_total, elasticity_of_retirements, coeffs, full_data=False)
        return residuals

    def sum_squared_residuals(self, coeffs, actual_stock, year_array, ca, pub_total, elasticity_of_retirements):
        """Calculate residuals for the housing stock model"""
        residuals_ = self.residuals(coeffs, actual_stock, year_array, ca, pub_total, elasticity_of_retirements)
        residuals_sq = [r**2 for r in residuals_]
        return sum(residuals_sq)

    def get_housing_stock_sf(self):
        """Get housing stock for single family units"""
        factor = 0.95
        use_columns = "C"
        actual_stock = [60607+4514, 61775+5496, 63587+5703, 63646+6156, 64283+6079, 66189+6213, 68109+6778, 70355+8027, 73427+8428, 74916+7227, 
        77703+7046, 80406+7135, 82472+7053, 82974+7768, 83392+7581, 92988, 94867]
        elasticity_of_retirements = 0.7
        pub_total = actual_stock[0]

        # url_ = 'https://www.census.gov/construction/nrc/xls/co_cust.xls'
        # new_comps_ann = pd.read_excel(url_) # completed
        
        new_comps_ann = pd.read_excel('C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/AHS_summary_results_051720.xlsx', sheet_name='Comps Ann', skiprows=26, 
                                usecols=use_columns, header=None).dropna()

        year_array = list(range(1985, 2019))

        # S is the actual housing stock (column X)
        # ca is the Comps Ann (column E)
        # Maybe use scipy.optimize.curve_fit?
        x0 = [-826, 0.00000982007075752705, 1]  # use Excel solver values as starting?
        ca = new_comps_ann.values

        x, flag = leastsq(self.residuals, x0, args=(actual_stock, year_array, ca, pub_total, elasticity_of_retirements), maxfev=1600)

        predicted_total_stock = self.housing_stock_model(year_array, ca, pub_total, elasticity_of_retirements, x, full_data=True)  

        occupied_published = [55076+4102, 56559+4820, 58242+4962, 57485+5442, 58918+5375, 60826+5545, 67951, 71499, 74434, 74026, 
                              76147, 77491, 79052, 80526, 80942, 83272, 85790]  # different_sources
        pub_total_occupied = np.subtract(actual_stock, occupied_published)
        total_vacancy_rate = np.divide(pub_total_occupied, actual_stock)
        
        total_vacancy_rate_all = []
        for index, rate in enumerate(list(total_vacancy_rate)): 
            if index == 0: 
                total_vacancy_rate_all.append(rate)
            elif index > 0: 
                previous_rate = total_vacancy_rate[index - 1]
                average_rate = (rate + previous_rate) / 2
                total_vacancy_rate_all.append(average_rate)
                total_vacancy_rate_all.append(rate)

        total_vacancy_rate_all.append(total_vacancy_rate_all[-1])  # 2018 just takes 2017 value, need to automate for future years

        occupation_rate = [1 - r for r in total_vacancy_rate_all]
        
        occupied_predicted = np.multiply(occupation_rate, predicted_total_stock)
        occupied_predicted_df = pd.DataFrame(list(zip(year_array, occupied_predicted)), columns=['Year', 'occupied_predicted']).set_index('Year')

        return actual_stock, occupied_published, occupied_predicted, occupied_predicted_df

    def get_housing_stock_mf(self, all_stock_sf, occupied_pub_sf):
        """Get housing stock for multi-family units
        
        Note: all_stock_sf has two values missing from end in the PNNL version?? (this keeps the pub_total_mf from being negative)

        """        
        factor = 0.96
        use_columns = "D:E"
        pub_total_values = [99931-6094, 102852-6688, 105661-6908, 104592-6983, 106611-7072, 109457-7647, 112357-8301, 115253-8433, 119117-8876, 120777-8971,
                        124377-8630, 128203-8705, 130112-8769, 132419-9049, 132832-8603, 33046, 34067]
        adjustment_factors = [1, 1, 1, 1.06, 1, 1, 1, 1.03, 1.04, 1, 1, 1, 1, 1, 1, 1, 1]
        elasticity_of_retirements = 0.8

        pub_total_years = list(range(1985, 1985 + 2 * len(pub_total_values), 2))       
        actual_sf_years = list(range(1985, 1985 + 2 * len(all_stock_sf), 2))

        actual_stock = np.subtract(pub_total_values, all_stock_sf)
        actual_stock_df = pd.DataFrame.from_dict({year: all_stock_sf[i] for i, year in enumerate(actual_sf_years)}, orient='index', columns=['all_stock_sf']).fillna(0)
        pub_total = pd.DataFrame(list(zip(pub_total_years, pub_total_values)), columns=['Year', 'pub_total']).set_index('Year')
        df = actual_stock_df.merge(pub_total, left_index=True, right_index=True, how='outer')

        df['pub_total_mf'] = df['pub_total'].subtract(df['all_stock_sf'].values)
        df['pub_total_mf'] = df['pub_total_mf'].multiply(adjustment_factors)

        year_array = list(range(1985, 1985 + 2 * len(pub_total_values), 1)) 
        year_df = pd.DataFrame(year_array, columns=['Year']).set_index('Year')
        df = df.merge(year_df, left_index=True, right_index=True, how='outer')

        # url_ = 'https://www.census.gov/construction/nrc/xls/co_cust.xls'
        # new_comps_ann = pd.read_excel(url_) # completed
        
        new_comps_ann_df = pd.read_excel('C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/AHS_summary_results_051720.xlsx', sheet_name='Comps Ann', skiprows=26, 
                                usecols=use_columns, header=None).dropna()
        new_comps_ann_df['New'] = new_comps_ann_df.sum(axis=1)
        new_comps_ann = new_comps_ann_df['New']

        occupied = [511+107+763+62, 487+192+701+104, 440+151+665+93, 444+259+692+66, 388+195+624+70, 482+197+624+77, 574+280+682+98, 481+269+670+79,
                    556+334+858+78, 675+252+882+82, 681+229+1026+92, 850+192+567+104, 820+198+1477+175, 803+252+1238+115] # 2670, for 2010
        occupied_years = list(range(1985, 1985 + 2 * len(occupied), 2))
        even_years = [o + 1 for o in occupied_years]

        occupied_df = pd.DataFrame(list(zip(occupied_years, occupied)), columns=['Year', 'occupied']).set_index('Year')
        
        occupied_pub_sf_years = list(range(1985, 1985 + 2 * len(occupied_pub_sf), 2)) 
        occupied_pub_sf_df = pd.DataFrame(list(zip(occupied_pub_sf_years, occupied_pub_sf)), columns=['Year', 'occupied_single_family']).set_index('Year')

        df = df.merge(occupied_df, left_index=True, right_index=True, how='outer')
        df = df.merge(occupied_pub_sf_df, left_index=True, right_index=True, how='outer')
        df['occupied_single_family'] = df['occupied_single_family'].fillna(0)
        df['all_stock_sf'] = df['all_stock_sf'].fillna(0)
  
        x0 = [-1171.75478590956, 0.0000292335584342192, 0.8]  # use Excel solver values as starting?
        ca = new_comps_ann.values
        actual_stock_calc = df[df['pub_total_mf'].notnull()]['pub_total_mf'].values.flatten()
        x, flag = leastsq(self.residuals, x0, args=(actual_stock_calc, year_array, ca, df['pub_total_mf'].values[0], elasticity_of_retirements), maxfev=1600)

        predicted_total_stock = self.housing_stock_model(year_array, ca, df['pub_total_mf'].values[0], elasticity_of_retirements, x, full_data=True)  
        predicted_retirement = self.housing_stock_model(year_array, ca, df['pub_total_mf'].values[0], elasticity_of_retirements, x, retirement=True)  

        df['predicted_retirement'] = predicted_retirement.flatten()
        R = [88425-4754, 90888-5267, 93683-5438, 93147-5630, 94724-5655, 97693-6164, 99487-6544, 102803-6785, 106261-7219, 105842-6854, 108871-6940, 110692-6919, 111806-6839, 114907-7190, 115852-6917, 28047, 28967] # Last two values (included here) are from AHS_2015_extract and AHS_2017_extract
        R = np.array(R) - np.array(occupied_pub_sf[:-2] + [0, 0])
        df_R = pd.DataFrame(list(zip(actual_sf_years, R)), columns=['Year', 'R']).set_index('Year')
        df = df.merge(df_R, left_index=True, right_index=True, how='outer')
        df['Y'] = df['pub_total_mf'].subtract(df['R'].values)
        df['Z'] = df['Y'].divide(df['pub_total_mf'].values)

        df = self.interpolate_with_avg(df, columns=['occupied', 'Z'])
        df.loc[2007, 'Z'] = df.loc[2007, 'Z'] - 0.005
        df['Z'] = df['Z'].ffill()
        predicted_unoccupied = predicted_total_stock - df['occupied'].values
        df['occupied_predicted'] = (1 - df['Z']) * predicted_total_stock 
        return df[['occupied_predicted']], df[['predicted_retirement']]

    def get_housing_stock_mh(self, all_stock_sf, occupied_pub_sf):
        """Get housing stock for manufactured housing units"""

        # new_comps_ann =   pd.read_excel('C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/placensa_all.xls', sheet_name='histplac', skiprows= , usecols=)# Added (place_nsa_all)
        # NEED TO PROCESS 
        new_comps_ann = pd.read_excel('C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/AHS_summary_results_051720.xlsx', sheet_name='place_nsa_all', skiprows=8, usecols="C", header=None).dropna()
        factor = 0.96
        elasticity_of_retirements = 0.5

        occupied_mh = [4754, 5267, 5438, 5630, 5655, 6164, 6544, 6785, 7219, 6854, 6940, 6919, 6839, 7190, 6917, 6901.7, 6724.4]
        pub_total_mh = [6094, 6688, 6908, 6983, 7072, 7647, 8301, 8433, 8876, 8971, 8630, 8705, 8769, 9049, 8603, 8686.5, 8396.7]
        L = [511+107+763+62, 487+192+701+104, 440+151+665+93, 444+259+692+66, 388+195+624+70, 482+197+624+77, 574+280+682+98, 
             481+269+670+79, 556+334+858+78, 675+252+882+82, 681+229+1026+92, 850+192+567+104, 820+198+1477+175, 803+252+1238+115]

        occupied_years = list(range(1985, 1985 + 2 * (len(occupied_mh)), 2))
        pub_total_years = list(range(1985, 1985 + 2 * (len(pub_total_mh)), 2))
        L_years = list(range(1985, 1985 + 2 * (len(L)), 2))
        all_stock_sf_years = list(range(1985, 1985 + 2 * (len(all_stock_sf)), 2))
        occupied_pub_sf_years = list(range(1985, 1985 + 2 * (len(occupied_pub_sf)), 2))

        occupied_mh_df = pd.DataFrame(list(zip(occupied_years, occupied_mh)), columns=['Year', 'occupied_mf']).set_index('Year')
        pub_total_mh_df = pd.DataFrame(list(zip(pub_total_years, pub_total_mh)), columns=['Year', 'pub_total_mh']).set_index('Year')
        L_df = pd.DataFrame(list(zip(L_years, L)), columns=['Year', 'L']).set_index('Year')
        all_stock_sf_df = pd.DataFrame(list(zip(all_stock_sf_years, all_stock_sf)), columns=['Year', 'all_stock_sf']).set_index('Year')
        occupied_pub_sf_df = pd.DataFrame(list(zip(occupied_pub_sf_years, occupied_pub_sf)), columns=['Year', 'occupied_pub_sf']).set_index('Year')

        df = occupied_mh_df.merge(pub_total_mh_df, left_index=True, right_index=True, how='outer')
        df = df.merge(L_df, left_index=True, right_index=True, how='outer')
        df = df.merge(all_stock_sf_df, left_index=True, right_index=True, how='outer')
        df = df.merge(occupied_pub_sf_df, left_index=True, right_index=True, how='outer')

        df['Y'] = df['pub_total_mh'].subtract(df['occupied_mf'])
        df['Z'] = df['Y'].divide(df['pub_total_mh'])

        year_array = list(range(1985, 1985 + 2 * len(pub_total_mh), 1)) 
        year_df = pd.DataFrame(year_array, columns=['Year']).set_index('Year')
        df = df.merge(year_df, left_index=True, right_index=True, how='outer')

        df = self.interpolate_with_avg(df, columns=['Z', 'L'], even=True)
        df['Z'] = df['Z'].ffill()
        df['all_stock_sf'] = df['all_stock_sf'].fillna(0)
        df['occupied_pub_sf'] = df['occupied_pub_sf'].fillna(0)

        year_array = list(range(1985, 2019))

        x0 = [216.913442843698, 0.00109247463281509, 1.05]  # use Excel solver values as starting?
        ca = new_comps_ann.values
        actual_stock_calc = df[df['pub_total_mh'].notnull()]['pub_total_mh'].values.flatten()

        x, flag = leastsq(self.residuals, x0, args=(actual_stock_calc, year_array, ca, df['pub_total_mh'].values[0], elasticity_of_retirements), maxfev=1600)

        predicted_total_stock = self.housing_stock_model(year_array, ca, df['pub_total_mh'].values[0], elasticity_of_retirements, x, full_data=True)  
        occupied_predicted = (1 - df['Z'].values) * predicted_total_stock
        occupied_predicted_df = pd.DataFrame(data={'Years': year_array, 'occupied_predicted_mh': occupied_predicted}).set_index('Years')
        return occupied_predicted, occupied_predicted_df

    def get_housing_stock(self):
        """Spreadsheet equivalent: Comps Ann, place_nsa_all
        Data Sources: 
            - Census Bureau Survey of  New Construction https://www.census.gov/construction/nrc/historical_data/index.html
            - Manufactured Housing Survey: Annual data for the most current years were not found on the Census Bureau website.
            Monthly data were downloaded for both total units and single (wide) units from the Census Bureau (in
            worksheets CIDR-1 and CIDR-single). The monthly data were aggregated to an annual basis for the years
            2014 through 2018 on these worksheets and the annual values appended to the existing data in the
            place_nsa_all worksheet. (Ideally, the place_nsa_all spreadsheet would be available from the Census
            Bureau for the most recent years but was not found as part of the 2020 update work.)
        Estimate regional housing and regional floorspace by housing type (single family, multifamily, manufactured homes)
        Data Sources: 
            - American Housing Survey (AHS) conducted by the Census Bureau to estimate aggregate floor space for three types
            of housing units: single-family (attached and detached), multi-family, and manufactured homes
        Spreadsheet Equivalents:
            - AHS_summary_results_date \Total_Stock_SF.xlsx (single family)
            - AHS_summary_results_date \Total_Stock_MF.xlsx (multifamily)
            - AHD_summary_results_date \Total_Stock_MH.xlsx (manufactured homes) 
        Methodology: 
            1. An estimated survival curve was first developed from vintage data over the 1999 through
            2009 AHS surveys.
            2. Curve was used along with reported new construction from the Characteristics of New
            Housing reports from the Census Bureau.
            3. The “stock adjustment model” was used to arrive at estimates of “Occupied Housing Units”
            at the national level.
        """
        actual_stock_sf, occupied_sf, occupied_predicted_sf, occupied_predicted_df_sf = self.get_housing_stock_sf()
        a = self.get_housing_size_sf(df=occupied_predicted_df_sf).rename(columns={'predicted_avg_size': 'avg_size_sqft_sf'})
        a = a.to_frame(name='avg_size_sqft_sf')

        occupied_predicted, occupied_predicted_df = self.get_housing_stock_mh(all_stock_sf=actual_stock_sf, occupied_pub_sf=occupied_sf)
        c = self.model_average_housing_unit_size_mh().rename(columns={'manh_size': 'avg_size_sqft_mh'})
        # c = c.to_frame(name='avg_size_sqft_mh')

        occupied_predicted_mf, predicted_retirements = self.get_housing_stock_mf(all_stock_sf=actual_stock_sf, occupied_pub_sf=occupied_sf)
        occupied_predicted_mf1 = occupied_predicted_mf.loc[min(occupied_predicted_mf.index)]
        occupied_predicted_mf = occupied_predicted_mf.rename(columns={'occupied_predicted': 'occupied_units_mf'})

        b = self.get_housing_size_mf(retirement_df=predicted_retirements, occupied_predicted_mf1=occupied_predicted_mf1).rename(columns={'final': 'avg_size_sqft_mf'}) # NEED TO FILL IN AHS TABLES
        b.name = 'avg_size_sqft_mf'
        occupied_predicted_df = occupied_predicted_df.rename(columns={'occupied_predicted_mh': 'occupied_units_mh'})
        occupied_predicted_df_sf = occupied_predicted_df_sf.rename(columns={'occupied_predicted': 'occupied_units_sf'})

        number_occupied_units_national = occupied_predicted_df_sf.merge(occupied_predicted_df, left_index=True, right_index=True, how='outer')
        number_occupied_units_national = number_occupied_units_national.merge(occupied_predicted_mf, left_index=True, right_index=True, how='outer')

        average_size_national = a.merge(c, left_index=True, right_index=True, how='outer')

        average_size_national = average_size_national.merge(b, left_index=True, right_index=True, how='outer')
        
        return number_occupied_units_national, average_size_national

    def final_floorspace_estimates(self):
        """Collect final floorspace estimates for the residential sector"""

        try: 
            os.chdir('./EnergyIntensityIndicators/Residential') # 
            cwd_changed = True
        except FileNotFoundError:
            try: 
                os.chdir('./Residential') # 
                cwd_changed = True
            except FileNotFoundError:
                cwd_changed = False
        
        number_occupied_units_national, average_size_national = self.get_housing_stock()
        number_occupied_units_national = number_occupied_units_national[['occupied_units_sf', 'occupied_units_mf', 'occupied_units_mh']]
        average_size_national  = average_size_national[['avg_size_sqft_sf', 'avg_size_sqft_mf', 'avg_size_sqft_mh']]

        regions = {'National': None, 'Northeast': 'NE', 'Midwest': 'MW', 'South': 'S', 'West': 'W'}
        final_results_total_floorspace_regions = dict()

        early_data = pd.read_csv('./historical_number_occupied_units.csv').set_index('Years')  # Not sure where these numbers come from, they're hardcoded in the residential indicators file
        number_occupied_units_national_early = early_data[early_data['Region']=='National'][['occupied_units_sf', 'occupied_units_mf', 'occupied_units_mh']]
        number_occupied_units_national_final = pd.concat([number_occupied_units_national_early, number_occupied_units_national], sort=True)
        number_occupied_units_national_final = number_occupied_units_national_final[number_occupied_units_national_final.index.notnull()]

        calculated_shares = pd.read_csv('./AHS_shares.csv').set_index('Year') # From AHS tables
        ratios_to_national_average_size = pd.read_csv('./AHS_size_shares.csv').set_index('Year') # From AHS tables
        ratios_to_national_average_size_mh = {'NE': 0.980, 'MW': 0.942, 'S': 1.022, 'W': 0.993}

        housing_types = ['SF', 'MF', 'MH']

        regional_units = {}
        regional_initial_sq_ft = {}
        avg_size_initial_regional = {}
        regional_estimates_all = {}
        final_results_total_floorspace_regions = {}

        for region in regions.keys(): 
            abbrev = regions[region]

            if abbrev:

                # Number of Units
                col_names = [f'{h_type}_{abbrev}' for h_type in housing_types]
                region_shares = calculated_shares[col_names]
            
                regional_estimates_late = number_occupied_units_national.multiply(region_shares.values)
                
                early_data_region = early_data[early_data['Region']==region]
                number_occupied_units_region_early = early_data_region[['occupied_units_sf', 'occupied_units_mf', 'occupied_units_mh']]
                regional_estimates = pd.concat([number_occupied_units_region_early, regional_estimates_late], sort=True)

                regional_estimates = regional_estimates[regional_estimates.index.notnull()]
                regional_units[region] = regional_estimates

                regional_estimates['Total'] = regional_estimates.sum(axis=1)
                regional_estimates_all[region] = regional_estimates

                shares_by_type = regional_estimates.drop('Total', axis=1).divide(regional_estimates['Total'].values.reshape(len(regional_estimates), 1))

                # Size of Units
                col_names_2 = [f'{h_type}_{abbrev}' for h_type in ['SF', 'MF']]
                region_size_shares = ratios_to_national_average_size[col_names_2]
                region_size_shares.loc[:, f'MH_{abbrev}'] = ratios_to_national_average_size_mh[abbrev]
                avg_size_initial = region_size_shares.multiply(average_size_national.values)
                avg_size_initial_regional[region] = avg_size_initial

                avg_size_region_early = early_data_region[['avg_size_sqft_sf', 'avg_size_sqft_mf', 'avg_size_sqft_mh']]

                total_sq_ft_initial = regional_estimates_late.multiply(avg_size_initial.values).multiply(0.000001)
                regional_initial_sq_ft[region] = total_sq_ft_initial
        
        regional_units_ne = regional_units['Northeast']
        regional_units_rest = [regional_units[r].values for r in ['Midwest', 'South', 'West']]
        initial_sum_regions_us = regional_units_ne.add(sum(regional_units_rest))
        number_occupied_units_national_final['Total'] = number_occupied_units_national_final.sum(axis=1)
        national_shares_by_type = number_occupied_units_national_final.drop('Total', axis=1).divide(number_occupied_units_national_final['Total'].values.reshape(len(number_occupied_units_national_final), 1))

        national_initial_sq_ft_ne = regional_initial_sq_ft['Northeast']
        national_initial_sq_ft_rest = [regional_initial_sq_ft[r].values for r in ['Midwest', 'South', 'West']]
        national_initial_sq_ft = national_initial_sq_ft_ne.add(sum(national_initial_sq_ft_rest))

        national_final_floorspace = number_occupied_units_national.multiply(average_size_national.values).multiply(0.000001)

        final_results_total_floorspace_regions['National'] = national_final_floorspace
        scale_factor = national_final_floorspace.divide(national_initial_sq_ft.values)

        avg_size_all_regions = {}
        for key, value in avg_size_initial_regional.items():
            avg_size_calibrated_late = value.multiply(scale_factor.values)
            abbrev = regions[key]
            avg_size_calibrated_late = avg_size_calibrated_late.rename(columns={f'SF_{abbrev}': 'avg_size_sqft_sf', f'MF_{abbrev}': 'avg_size_sqft_mf', f'MH_{abbrev}': 'avg_size_sqft_mh'})

            early_data_region = early_data[early_data['Region']==key]
            avg_size_region_early = early_data_region[['avg_size_sqft_sf', 'avg_size_sqft_mf', 'avg_size_sqft_mh']]

            avg_size_calibrated = pd.concat([avg_size_region_early, avg_size_calibrated_late], sort=True)
            avg_size_all_regions[key] = avg_size_calibrated

            final_results_total_floorspace = regional_estimates_all[key].drop('Total', axis=1).multiply(avg_size_calibrated.values).multiply(0.000001)
            final_results_total_floorspace_regions[key] = final_results_total_floorspace

        regional_estimates_all['National'] = number_occupied_units_national_final
        avg_size_all_regions['National'] = average_size_national
        if cwd_changed:
            os.chdir('..')
        return final_results_total_floorspace_regions, regional_estimates_all, avg_size_all_regions
        
    def main(self):
        data = ResidentialFloorspace()
        final_results_total_floorspace_regions, regional_estimates_all, avg_size_all_regions = data.final_floorspace_estimates()


if __name__ == '__main__':
    ResidentialFloorspace().main()

