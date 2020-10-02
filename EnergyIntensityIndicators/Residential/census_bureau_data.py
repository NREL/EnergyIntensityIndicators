import pandas as pd
import os
import requests
from zipfile import ZipFile
import numpy as np
import scipy
from scipy.optimize import leastsq, least_squares, minimize
import matplotlib.pyplot as plt
from statistics import mean


class GetCensusData:
    
    def __init__(self): 
        pass
    
    @staticmethod
    def update_ahs_data():
        """Spreadsheet equivalent: AHS_2017_extract
        Extract and process American Housing Survey (AHS, formerly Annual Housing Survey) 
        web: https://www.census.gov/programs-surveys/ahs/data.html
        ? https://www.census.gov/programs-surveys/ahs/data/2017/ahs-2017-public-use-file--puf-/ahs-2017-national-public-use-file--puf-.html 
        """    
        ahs_url_folder = 'http://www2.census.gov/programs-surveys/ahs/2017/AHS%202017%20National%20PUF%20v3.0%20CSV.zip?#' 
        
        if os.path.exists('./household.csv'):
            pass
        else: 
            with ZipFile(ahs_url_folder, 'r') as zipOjb:
                zipOjb.extact('household.csv')
       
        ahs_household_data = pd.read_csv('./household.csv')
        columns = ['JYRBUILT', 'WEIGHT', 'YRBUILT', 'DIVISION', 'BLD', 'UNITSIZE', 'VACANCY']
        extract_ahs = ahs_household_data[columns]
        extract_ahs= extract_ahs[extract_ahs['BLD'].isin(['04', '05', '06', '07', '08', '09'])]
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
        pivot_census_division['census_region_1'] = pivot_census_division['1'] + pivot_census_division['2']   # alternative method: df['C'] = df.apply(lambda row: row['A'] + row['B'], axis=1)
        pivot_census_division['census_region_2'] = pivot_census_division['3'] + pivot_census_division['4']
        pivot_census_division['census_region_3'] = pivot_census_division['5'] + pivot_census_division['6'] + pivot_census_division['7']
        pivot_census_division['census_region_4'] = pivot_census_division['8'] + pivot_census_division['8']
        pivot_census_division['total'] = pivot_census_division['census_region_1'] + pivot_census_division['census_region_2'] + pivot_census_division['census_region_3'] + pivot_census_division['census_region_4']
        pivot_census_division.loc['Grand Total', :]= pivot_census_division.sum(axis=0)

        return pivot_census_division          
    
    @staticmethod
    def get_percent_remaining_surviving(factor, lifetime, gamma):
        return 1 / (1 + (factor / lifetime) ** gamma)   

    @staticmethod
    def interpolate_with_avg(dataframe, columns, even=True):
        """[summary]

        Args:
            dataframe (df): dataframe with year index 
            columns (list): names of columns to interpolate with average
            even (bool): whether the years to fill are even (or odd if False)

        Returns:
            [type]: [description]
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
        years = list(range(1994, 2013 + 1))
        for year in years:
            url = f'http://www2.census.gov/programs-surveys/mhs/tables/{str(year)}/stplace{str(year)[-2:]}.xls'
            placement_df = pd.read_excel(url, index_col=0, skiprows=4, use_cols='B, F:H') # Placement units are thousands of units
    
    @staticmethod
    def housing_stock_model(year_array, new_comps_ann, pub_total_0, elasticity_of_retirements, coeffs, full_data=False, retirement=False):
        """[summary]

        Args:
            year_array (array): [description]
            new_comps_ann (array): New annual completed housing
            actual_stock (array): [description]

            coeffs (array): constant_adjustment, fraction_of_retirements, fixed_value

        Returns:
            [type]: [description]
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
    
    @staticmethod
    def model_average_housing_unit_size_sf(year_array, new_comps_ann, predicted_retirement, coeffs, x):
        """[summary]

        Args:
            year_array ([type]): [description]
            new_comps_ann ([type]): [description]
            coeffs ([type]): [description]
        """                
        new_comps_ann_adj = new_comps_ann * x[2]  # here x is the coeffs from occupied units
        cnh_avg_size = [0] # SFTotalMedAvgSqFt column G
        column_bh = new_comps_ann.multiply(cnh_avg_size)
        select_index = 24
        for index_, year_ in enumerate(year_array):
            if index_ == 0: 
                bi = column_bh[index_]
                column_bi = np.array([bi])

                post_1984_units = new_comps_ann[_index]
                post_1984_units_series = np.array([post_1984_units])

                pre_1985_stock = occupied_predicted[index_]
                pre_85_stock_series = np.array([pre_1985_stock])

                bl = pre_1985_stock
            else:
                bi =  column_bh[index_] + column_bi[index_ - 1]
                column_bi = np.vstack([column_bi, bi])

                post_1984_units = new_comps_ann[_index] + post_1984_units_series[index_ - 1]
                post_1984_units_series = np.vstack([post_1984_units_series, post_1984_units])

                pre_1985_stock = pre_85_stock_series[0] + sum(predicted_retirement[0:index_:])
                pre_85_stock_series = np.vstack([pre_85_stock_series, pre_1985_stock])

                bl = (post_1984_units + post_1984_units_series[index_ - 1]) * 0.5 * coeffs[2] + pre_1985_stock

            ave_size_post_84_units = bi / post_1984_units
            
            if index_ == select_index:
                predicted_size_pre_1985_stock = coeffs[0]
            else: 
                predicted_size_pre_1985_stock = coeffs[0] + coeffs[1] * (year_ - year_array[select_index])

            total_sq_feet_pre_1985 = pre_1985_stock *  predicted_size_pre_1985_stock
            bp = post_1984_units / (post_1984_units + pre_1985_stock)
            
            total_sq_feet_post_1985 = post_1984_units * ave_size_post_84_units * coeffs[2] * coeffs[4]

            predicted_ave_size = (total_sq_feet_pre_1985 + total_sq_feet_post_1985) / bl 

            if index_ == 0:
                predicted_ave_size_series = np.array([predicted_ave_size])
            else:
                predicted_ave_size_series = np.vstack([predicted_ave_size_series, predicted_ave_size])

        predicted_ave_size_series = predicted_ave_size_series.flatten()
        predicted_ave_size_series_skip = predicted_ave_size_series # SLICE
        
        df['predicted'] = (df['total_sqft_pre_1985'].add(df['total_sqft_post_1985'].values)).divde(df['BL'])
        return df['predicted']
    
    def get_housing_size_mf(self, retirement_df, occupied_predicted_mf1):
         """[summary]

        Args:
            new_comps_ann (series): [description]
            coeffs (list): [CM$5, $CM$6, $CM$7]
            df (dataframe): Input data, df contains 'occupied_predicted_mf' and 'retirements' columns (from housing stock mf)

        Returns:
            [type]: [description]
        """        
        just_adjustment = 0.75
        df = df.reindex(columns=df.columns + ['BJ', 'BM', 'CB', 'final'])
        df['BE'] = comps_ann

        ahs_table = 
        ahs_table = ahs_table.rename(columns={'1970-1979': '1975', '1980-1989': '1985', '1990-1999': '1995', '2000 & later': '2005'})
        ahs_table = ahs_table.reindex(index=ahs_table + [2011])
        ahs_table.loc[2011, :] = ahs_table.mean(axis=0)

        actual_size =  # from ahs tables

        for i in df.index:
            if i == min(df.index):
                df.loc[i, ['BJ']] = df.loc[i, 'BE']
            elif i == 1984:
                df.loc[i, ['BJ']] = 0
            else:
                df.loc[i, ['BJ']] = df.loc[i - 1, ['BJ']].add(df.loc[i, ['BE']].values)

        df.loc[1985, ['BM']] = occupied_predicted_mf1
        for year in range(1986, max(dataframe.index)):
            df.loc[year, ['BM']] = df.loc[1985:year, ['retirements']]

        df['BP'] = df['BJ'].divide(df['BJ'].add(df['BM'].values).values)
        
        decades = list(range(1985, 2005, 10))
        adjustment_dict = dict()
        for i, d in enumerate(decades):
            df.loc[d, ['CB']] = ahs_table.loc[2011, [str(d)]]
            decade_after = df.loc[decades[i + 1]
            adjustment = (decade_after, ['CB']].values - df.loc[d, ['CB']].values) / 10
            adjustment_dict[d] = adjustment
            for y_ in range(d + 1, decade_after - 1):
                df.loc[y_, ['CB']] = df.loc[y_ - 1, ['CB']] + adjustment
        df.loc[2006, ['CB']] = df.loc[2005, ['CB']].values + adjustment_dict[1995]
        df.loc[2007, ['CB']] = df.loc[2006, ['CB']].values + adjustment_dict[1995]
        df['CB'] = df['CB'].ffill()

        x0 = [567.081097939713, -22.7744777937053, 2.44216497607253]

        x, flag = leastsq(self.residuals_avg_size_mf, x0, args=(actual_size, df['BE'].values, df), maxfev=1600)
        results = self.average_housing_unit_size_mf(x, df)

        return results

    def residuals_avg_size_mf(self, coeffs, actual_size, input_data):
        residuals = actual_size - self.average_housing_unit_size_mf(coeffs, input_data)
        return residuals

    @staticmethod
    def average_housing_unit_size_mf(coeffs, df):
        year_1997_value = coeffs[0]
        index_year = 1997
        df['Predicted size of pre-1985 stock'] = year_1997_value + coeffs[1] * (df.index - index_year) * 0.75
        df['CN'] = df['CB'] * coeffs[2]
        df['Post-85 shr'] = df['BP'] 
        df['Ave Size Predicted'] = (1 - df['Post-85 shr'].values).multiply(df['Predicted size of pre-1985 stock']).add(df['Post-85 shr'].multiply(df['CN'].values))
        df.loc[1985:1999, ['final']] = df.loc[1985:1999, ['Predicted-II']]
        df.loc[2000:, ['final']] = df.loc[2000:, ['Ave Size Predicted']]
        return df['final'].values
        
    
    @staticmethod
    def model_average_housing_unit_size_mh():
        """recs_data from  D:\Supporting_Sheets_SRB\[Residential_Floor_Space_2013_SRB.xlsx]REC_Total_SF (2)
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

        recs_based = recs_manhome.loc[1985:, :]
        return recs_based
    
    def residuals(self, coeffs, actual_stock, year_array, ca, pub_total, elasticity_of_retirements):
        residuals = actual_stock - self.housing_stock_model(year_array, ca, pub_total, elasticity_of_retirements, coeffs, full_data=False)
        return residuals

    def sum_squared_residuals(self, coeffs, actual_stock, year_array, ca, pub_total, elasticity_of_retirements):
        residuals_ = self.residuals(coeffs, actual_stock, year_array, ca, pub_total, elasticity_of_retirements)
        residuals_sq = [r**2 for r in residuals_]
        return sum(residuals_sq)

    def get_housing_stock_sf(self):
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

        # result = least_squares(self.residuals, x0, meht)
        # x =  result.x
        # success = result.success
        # residuals = result.fun

        # = curve_fit(self.residuals, )

        # print('X0:', x0)
        # print('X: ', x)
        # print('flag: ', flag)
        # ssr_x0 = self.sum_squared_residuals(x0, actual_stock, year_array, ca, pub_total, elasticity_of_retirements)
        # ssr_x = self.sum_squared_residuals(x, actual_stock, year_array, ca, pub_total, elasticity_of_retirements)
        # print('ssr_x0:', ssr_x0)
        # print('ssr_x:', ssr_x)
        # print(ssr_x < ssr_x0)

        # predicted_total_stock_pnnl = self.housing_stock_model(year_array, ca, pub_total, elasticity_of_retirements, x0, full_data=True)  

        predicted_total_stock = self.housing_stock_model(year_array, ca, pub_total, elasticity_of_retirements, x, full_data=True)  
        # residuals = self.residuals(x0, actual_stock=actual_stock , year_array=year_array, ca=ca , pub_total=pub_total , elasticity_of_retirements=elasticity_of_retirements)
        # plt.style.use('seaborn-darkgrid')
        # palette = plt.get_cmap('Set1')
        # plt.plot(year_array[0::2], predicted_total_stock[0::2], marker='', color=palette(1), linewidth=1, alpha=0.9, label='SciPy Prediction')
        # plt.plot(year_array[0::2], predicted_total_stock_pnnl[0::2], marker='', color=palette(2), linewidth=1, alpha=0.9, label='Solver Prediction')
        # plt.plot(year_array[0::2], actual_stock, marker='', color=palette(3), linewidth=1, alpha=0.9, label='Actual Stock')
        # plt.title('A Comparison of Optimization Software Results', fontsize=12, fontweight=0)
        # plt.xlabel('Year')
        # plt.ylabel('Housing Stock')
        # plt.legend(loc=2, ncol=2)
        # plt.show()

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

        return actual_stock, occupied_published, occupied_predicted

    def get_housing_stock_mf(self, all_stock_sf, occupied_pub_sf):
        """Note: all_stock_sf has two values missing from end in the PNNL version?? (this keeps the pub_total_mf from being negative)

        Args:
            all_stock_sf ([type]): [description]

        Returns:
            [type]: [description]
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
        print('find coefficients:')
        actual_stock_calc = df[df['pub_total_mf'].notnull()]['pub_total_mf'].values.flatten()
        x, flag = leastsq(self.residuals, x0, args=(actual_stock_calc, year_array, ca, df['pub_total_mf'].values[0], elasticity_of_retirements), maxfev=1600)
        print('X: ', x)
        print('flag: ', flag)

        print('find predicted total stock:') 
        predicted_total_stock = self.housing_stock_model(year_array, ca, df['pub_total_mf'].values[0], elasticity_of_retirements, x, full_data=True)  
        predicted_retirement = self.housing_stock_model(year_array, ca, df['pub_total_mf'].values[0], elasticity_of_retirements, x, retirement=True)  

        df['predicted_retirement'] = predicted_retirement.flatten().values
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
        occupied_predicted = (1 - df['Z']) * predicted_total_stock 
        return occupied_predicted, df[['predicted_retirement']]

    def get_housing_stock_mh(self, all_stock_sf, occupied_pub_sf):

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
        print('X: ', x)
        print('flag: ', flag)

        predicted_total_stock = self.housing_stock_model(year_array, ca, df['pub_total_mh'].values[0], elasticity_of_retirements, x, full_data=True)  
        occupied_predicted = (1 - df['Z'].values) * predicted_total_stock
        return occupied_predicted

    @staticmethod
    def get_housing_stock(housing_type):
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
        return number_occupied_units_national, average_size_national

    # def final_floorspace_estimates():
        
    #     number_occupied_units_national, average_size_national = get_housing_stock()

    #     regions = ['National', 'Northeast', 'Midwest', 'South', 'West']
    #     final_results_total_floorspace_regions = dict()

    #     for region in regions: 
    #         calculated_shares_by_region = [0] # From AHS tables
    #         ratios_to_national_average_size = [0] # From AHS tables

    #         regional_estimates = number_occupied_units_national.multiply(calculated_shares_by_region)
    #         regional_estimates['Total'] = regional_estimates.sum(axis=1)
    #         shares_by_type = regional_estimates[['Single Family', 'Multi-Family', 'Man. Homes']].divide(regional_estimates['Total'])
            
    #         # Calibration Procedure
    #         sum_of_regions = 
    #         scale_factor = 
    #         final_check =
    #         average_size_all_housing_units =
    #         number_of_units = 

    #         total_square_feet =
    #         average_size = 
    #         ratios_national_average_size = 

    #         average_size_after_calibration = 


    #         final_results_total_floorspace = number_occupied_units.multiply(average_size).multiply(0.000001)
    #         final_results_total_floorspace['Total'] = final_results_total_floorspace.sum(axis=1)

    #         number_occupied_units['Total'] = number_occupied_units.sum(axis=1)
            
    #         final_results_total_floorspace_regions[region] = final_results_total_floorspace

    #     return final_results_total_floorspace_regions

    @staticmethod
    def weighted_floorspace():
        energy_types = ['Electricity', 'Fuels', 'Delivered', 'Source']
        constant_electricity_factor = 3.2

    @staticmethod
    def get_census_bureau_manufactured_housing_survey():
        """[summary]
        Annual data for the most current years were not found on the Census Bureau website.
        Monthly data were downloaded for both total units and single (wide) units from the Census Bureau (in
        worksheets CIDR-1 and CIDR-single). The monthly data were aggregated to an annual basis for the years
        2014 through 2018 on these worksheets and the annual values appended to the existing data in the
        place_nsa_all worksheet. (Ideally, the place_nsa_all spreadsheet would be available from the Census
        Bureau for the most recent years but was not found as part of the 2020 update work.)
        """
        pass    

    def main(self):
        data = GetCensusData()
        # actual_stock_sf, occupied_sf, occupied_predicted_sf = data.get_housing_stock_sf()
        # # a = data.model_average_housing_unit_size_sf()

        # z = data.get_housing_stock_mh(all_stock_sf=actual_stock_sf, occupied_pub_sf=occupied_sf)
        # c = data.model_average_housing_unit_size_mh()

        occupied_predited_mf, predicted_retirements = data.get_housing_stock_mf(all_stock_sf=actual_stock_sf, occupied_pub_sf=occupied_sf)
        occupied_predicted_mf1 = occupied_predited_mf[0]

        b = data.get_housing_size_mf(retirement_df=predicted_retirements, occupied_predicted_mf1=occupied_predicted_mf1)
        print(b)





if __name__ == '__main__':
    GetCensusData().main()







# for index_, year_ in enumerate(year_array):
#             if index_ == 0: 
#                 bi = column_bh[index_]
#                 column_bi = np.array([bi])

#                 post_1984_units = new_comps_ann[_index]
#                 post_1984_units_series = np.array([post_1984_units])

#                 pre_1985_stock = occupied_predicted[index_]
#                 pre_85_stock_series = np.array([pre_1985_stock])

#                 bl = pre_1985_stock
#             else:
#                 bi =  column_bh[index_] + column_bi[index_ - 1]
#                 column_bi = np.vstack([column_bi, bi])

#                 post_1984_units = new_comps_ann[_index] + post_1984_units_series[index_ - 1]
#                 post_1984_units_series = np.vstack([post_1984_units_series, post_1984_units])

#                 pre_1985_stock = pre_85_stock_series[0] + sum(predicted_retirement[0:index_:])
#                 pre_85_stock_series = np.vstack([pre_85_stock_series, pre_1985_stock])

#                 bl = (post_1984_units + post_1984_units_series[index_ - 1]) * 0.5 * coeffs[2] + pre_1985_stock

#             ave_size_post_84_units = bi / post_1984_units
            
#             if index_ == select_index:
#                 predicted_size_pre_1985_stock = coeffs[0]
#             else: 
#                 predicted_size_pre_1985_stock = coeffs[0] + coeffs[1] * (year_ - year_array[select_index])

#             total_sq_feet_pre_1985 = pre_1985_stock *  predicted_size_pre_1985_stock
#             bp = post_1984_units / (post_1984_units + pre_1985_stock)
            
#             total_sq_feet_post_1985 = post_1984_units * ave_size_post_84_units * coeffs[2] * coeffs[4]

#             predicted_ave_size = (total_sq_feet_pre_1985 + total_sq_feet_post_1985) / bl 

#             if index_ == 0:
#                 predicted_ave_size_series = np.array([predicted_ave_size])
#             else:
#                 predicted_ave_size_series = np.vstack([predicted_ave_size_series, predicted_ave_size])

#         predicted_ave_size_series = predicted_ave_size_series.flatten()
#         predicted_ave_size_series_skip = predicted_ave_size_series # SLICE

#         return predicted_ave_size_series_skip