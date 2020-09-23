import pandas as pd
import os
import requests
from zipfile import ZipFile
import numpy as np
import scipy
from scipy.optimize import leastsq, least_squares, minimize
import matplotlib.pyplot as plt


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
    def get_place_nsa_all():
        years = list(range(1994, 2013 + 1))
        for year in years:
            url = f'http://www2.census.gov/programs-surveys/mhs/tables/{str(year)}/stplace{str(year)[-2:]}.xls'
            placement_df = pd.read_excel(url, index_col=0, skiprows=4, use_cols='B, F:H') # Placement units are thousands of units
    
    @staticmethod
    def housing_stock_model(year_array, new_comps_ann, pub_total_0, elasticity_of_retirements, coeffs, full_data=False):
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
                new_units = 0
                existing_stock_series = np.array([existing_stock])
                predicted_total_stock_series = np.array([existing_stock])
            else:
                new_units = ((new_comps_ann[index_] + new_comps_ann[index_-1]) / 2 ) * coeffs[2] 
                adjusted_new_units = np.sign(new_units) * (np.abs(new_units)) ** elasticity_of_retirements
                existing_stock = predicted_total_stock_series[index_ - 1]
                predicted_retirement = (-1 * existing_stock) * adjusted_new_units * coeffs[1]
            
                predicted_total_stock = existing_stock + predicted_retirement + new_units
            
                existing_stock_series = np.vstack([existing_stock_series, existing_stock])
                predicted_total_stock_series = np.vstack([predicted_total_stock_series, predicted_total_stock])
        
        predicted_total_stock_series = predicted_total_stock_series.flatten()
        predicted_total_stock_series_skip = predicted_total_stock_series[0::2]

        if full_data:
            return predicted_total_stock_series
        else:
            return predicted_total_stock_series_skip
    
    @staticmethod
    def model_average_housing_unit_size_sf(year_array, new_comps_ann, predicted_retirement, coeffs):
        """[summary]

        Args:
            year_array ([type]): [description]
            new_comps_ann ([type]): [description]
            coeffs ([type]): [description]
        """                
        new_comps_ann_adj = new_comps_ann * x[2]
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

        return predicted_ave_size_series_skip

    @staticmethod
    def model_average_housing_unit_size_mf(year_array, new_comps_ann, predicted_retirement, coeffs):
        """[summary]

        Args:
            year_array ([type]): [description]
            new_comps_ann (series): [description]
            predicted_retirement ([type]): [description]
            coeffs ([type]): [description]

        Returns:
            [type]: [description]
        """        
        new_comps_ann_adj = new_comps_ann * x[2]
        new_comps_ann_multifamily = [0]

        cnh_avg_size = [0] # SFTotalMedAvgSqFt column I
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

        return predicted_ave_size_series_skip
    
    @staticmethod
    def model_average_housing_unit_size_mh():
        """Create average housing unit size dataframe for Manufactured housing

        Returns:
            [type]: [description]
        """        
        pre_1980_ratios = {1970: 0.838421513, 1974: 0.944122109}  # from 'D:\Supporting_Sheets_SRB\[Residential_Floor_Space_2013_SRB.xlsx]RECS_Vintage_size'
        recs_total =  {1980: 826.0869565, 1984: 843.1372549, 1987: 864.7058824, 1990: 942.3076923, 
                                   1993: 989.0909091, 1997: 995.2380952, 2001: 1061.995005, 2005: 1062.008978, 2009: 1087}  # from 'D:\Supporting_Sheets_SRB\[Residential_Floor_Space_2013_SRB.xlsx]REC_Total_SF (2)'
        for y in [1970, 1974]:
            recs_total[y] = recs_total[1980] * pre_1980_ratios[y]
        
        recs_total[2015] = 1191.2  # From 'RECS_4_adj'

        manh_size = dict()
        increment_years = [1970, 1974, 1980, 1984, 1987, 1990, 1993, 1997, 2001, 2005, 2009, 2015]
        
        for index, y_ in enumerate(increment_years):
            if index > 0:
                year_before = increment_years[index - 1]
                num_years = y_ - year_before
                difference = recs_total[y_] - recs_total[year_before]
                increment = difference / num_years
                for delta in range(num_years):
                    value = recs_total[year_before]  + delta * increment
                    year = year_before + delta
                    manh_size[year] = value
        
        manh_size[2016] = 1196  # Assume smaller change - 5 sq. ft. per year,  rounded?
        manh_size[2017] = manh_size[2016] + 5 # Assume smaller change - 5 sq. ft. per year 
        
        manh_size_df = pd.DataFrame(manh_size, columns=['year', 'manh_size']).set_index('year')
        return manh_size
    
    def residuals(self, coeffs, actual_stock, year_array, ca, pub_total, elasticity_of_retirements):
        residuals = actual_stock - self.housing_stock_model(year_array, ca, pub_total, elasticity_of_retirements, coeffs, full_data=False)
        # residuals_sq = np.square(residuals)
        # sum_residuals_sq = np.sum(residuals_sq)
        # return sum_residuals_sq
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

        print('X0:', x0)
        print('X: ', x)
        print('flag: ', flag)
        ssr_x0 = self.sum_squared_residuals(x0, actual_stock, year_array, ca, pub_total, elasticity_of_retirements)
        ssr_x = self.sum_squared_residuals(x, actual_stock, year_array, ca, pub_total, elasticity_of_retirements)
        print('ssr_x0:', ssr_x0)
        print('ssr_x:', ssr_x)
        print(ssr_x < ssr_x0)

        predicted_total_stock_pnnl = self.housing_stock_model(year_array, ca, pub_total, elasticity_of_retirements, x0, full_data=True)  

        predicted_total_stock = self.housing_stock_model(year_array, ca, pub_total, elasticity_of_retirements, x, full_data=True)  
        # residuals = self.residuals(x0, actual_stock=actual_stock , year_array=year_array, ca=ca , pub_total=pub_total , elasticity_of_retirements=elasticity_of_retirements)
        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set1')
        plt.plot(year_array[0::2], predicted_total_stock[0::2], marker='', color=palette(1), linewidth=1, alpha=0.9, label='SciPy Prediction')
        plt.plot(year_array[0::2], predicted_total_stock_pnnl[0::2], marker='', color=palette(2), linewidth=1, alpha=0.9, label='Solver Prediction')
        plt.plot(year_array[0::2], actual_stock, marker='', color=palette(3), linewidth=1, alpha=0.9, label='Actual Stock')
        plt.title('A Comparison of Optimization Software Results', fontsize=12, fontweight=0)
        plt.xlabel('Year')
        plt.ylabel('Housing Stock')
        plt.legend(loc=2, ncol=2)
        plt.show()

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

        return occupied_predicted

    def get_housing_stock_mf(self):
        factor = 0.96
        use_columns = "D:E"
        pub_total = [99931-6094, 102852-6688, 105661-6908, 104592-6983, 106611-7072, 109457-7647, 112357-8301, 115253-8433, 119117-8876, 120777-8971,
                        124377-8630, 128203-8705, 130112-8769, 132419-9049, 132832-8603, 33046, 34067]
        # all_stock_sf = [60607+4514, 61775+5496, 63587+5703, 63646+6156, 64283+6079, 66189+6213, 68109+6778, 70355+8027, 73427+8428, 4916+7227, 
        # 77703+7046, 80406+7135, 82472+7053, 82974+7768, 83392+7581, 92988, 94867]
        adjustment_factors = [1, 1, 1, 1.06, 1, 1, 1, 1.03, 1.04, 1, 1, 1, 1, 1, 1, 1, 1]

        predicted_total_stock_sf, all_stock_sf = self.get_housing_stock_sf()

        actual_stock = np.subtract(pub_total, all_stock_sf)
        actual_stock = np.multiply(actual_stock, adjustment_factors)
        elasticity_of_retirements = 0.8

        year_array = list(range(1985, 2019))

        # url_ = 'https://www.census.gov/construction/nrc/xls/co_cust.xls'
        # new_comps_ann = pd.read_excel(url_) # completed
        
        new_comps_ann_df = pd.read_excel('C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/AHS_summary_results_051720.xlsx', sheet_name='Comps Ann', skiprows=26, 
                                usecols=use_columns, header=None).dropna()
        new_comps_ann_df['New'] = new_comps_ann_df.sum(axis=1)
        new_comps_ann = new_comps_ann_df['New']

        all_single_family = actual_stock
        occupied_single_family = [55076+4102, 56559+4820, 58242+4962, 57485+5442, 58918+5375, 60826+5545, ] # column J Total_stock_SF, append prortion from AHS tables, etc
        households = [0] # National_Calibration'!E13
        year = [0]

        x0 = [-1171.75478590956, 0.0000292335584342192, 0.8]  # use Excel solver values as starting?
        ca = new_comps_ann.values
        x, flag = leastsq(self.residuals, x0, args=(actual_stock, year_array, ca, pub_total, elasticity_of_retirements), maxfev=1600)
        print('X: ', x)
        print('flag: ', flag)

        predicted_total_stock = self.housing_stock_model(year_array, ca, pub_total, elasticity_of_retirements, x, full_data=True)  

        return predicted_total_stock

    def get_housing_stock_mh(self):

        # new_comps_ann =   pd.read_excel('C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/placensa_all.xls', sheet_name='histplac', skiprows= , usecols=)# Added (place_nsa_all)
        # NEED TO PROCESS 
        new_comps_ann = pd.read_excel('C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/AHS_summary_results_051720.xlsx', sheet_name='place_nsa_all', skiprows=8, usecols="C", header=None).dropna()
        factor = 0.96
        elasticity_of_retirements = 0.5
        
        year_array = list(range(1985, 2019))


        x0 = [216.913442843698, 0.00109247463281509, 1.05]  # use Excel solver values as starting?
        ca = new_comps_ann.values
        x, flag = leastsq(self.residuals, x0, args=(actual_stock, year_array, ca, pub_total, elasticity_of_retirements), maxfev=1600)
        print('X: ', x)
        print('flag: ', flag)

        predicted_total_stock = self.housing_stock_model(year_array, ca, pub_total, elasticity_of_retirements, x, full_data=True)  

        return predicted_total_stock

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
        pass

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


data = GetCensusData()

x = data.get_housing_stock_sf()
# y = data.get_housing_stock_mf()
# z = data.get_housing_stock_mh()
print('sf results:', x)
# print('mf results:', y)
# print('mh results:', z)