"""
[summary]
"""
import pandas as pd
import os
import requests
from zipfile import ZipFile
import numpy as np
import scipy
from scipy import optimize

class GetCensusData:
    
    def __init__(self): 
        pass

    def update_ahs_data():
        """Spreadsheet equivalent: AHS_2017_extract
        Extract and process American Housing Survey (AHS, formerly Annual Housing Survey) 
        web: https://www.census.gov/programs-surveys/ahs/data.html
        ? https://www.census.gov/programs-surveys/ahs/data/2017/ahs-2017-public-use-file--puf-/ahs-2017-national-public-use-file--puf-.html 
        """    
        ahs_url_folder   'http://www2.census.gov/programs-surveys/ahs/2017/AHS%202017%20National%20PUF%20v3.0%20CSV.zip?#' 
        
        if os.path.exists('./household.csv'):
            pass
        else: 
            with ZipFile(ahs_url_folder, 'r') as zipOjb:
                zipObj.extact('household.csv')
       
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
        pivot_census_division['total'] = pivot_census_division['census_region_1'] + pivot_census_division['census_region_2'] + 
                                                pivot_census_division['census_region_3'] + pivot_census_division['census_region_4']
        pivot_census_division.loc['Grand Total', :]= pivot_census_division.sum(axis=0)

        return pivot_census_division          

    def get_percent_remaining_surviving(factor, lifetime, gamma):
        return 1 / (1 + (factor / lifetime) ** gamma)   

    def get_place_nsa_all():
        years = list(range(1994, 2013 + 1))
        for year in years:
            url = f'http://www2.census.gov/programs-surveys/mhs/tables/{str(year)}/stplace{str(year)[-2:]}.xls'
            placement_df = pd.read_excel(url, index_col=0, skiprows=4, use_cols='B, F:H') # Placement units are thousands of units

    def housing_stock_model(year_array, new_comps_ann, full_data=False, coeffs):
        """[summary]

        Args:
            year_array (array): [description]
            new_comps_ann (array): New annual completed housing
            actual_stock (array): [description]

            coeffs (array): constant_adjustment, fraction_of_retirements, fixed_value

        Returns:
            [type]: [description]
        """          

        elasticity_of_retirements = 0.7
        pub_total = [65121]

        for index_, year_ in enumerate(year_array):
            if index_ == 0: 
                existing_stock = pub_total[index_] + coeffs[0]
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
    
    def model_average_housing_unit_size_sf(year_array, new_comps_ann, predicted_retirement, coeffs):
        """[summary]

        Args:
            year_array ([type]): [description]
            new_comps_ann ([type]): [description]
            coeffs ([type]): [description]
        """                
        new_comps_ann_adj = new_comps_ann * x[2]
        cnh_avg_size = # SFTotalMedAvgSqFt column G
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
        new_comps_ann_multifamily = 

        cnh_avg_size = # SFTotalMedAvgSqFt column I
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
            url_ = 'https://www.census.gov/construction/nrc/xls/co_cust.xls'

            if housing_type == 'single_family' | housing_type == 'multi_family':
                housing_units_completed_or_placed = pd.read_excel(url_) # completed

            else:
                pass

            if housing_type == 'single_family':
                factor = 0.95
                use_columns = "C"
                actual_stock = [60607+4514, 61775+5496, 63587+5703, 63646+6156, 64283+6079, 66189+6213, 68109+6778, 70355+8027, 73427+8428, 4916+7227, 
                77703+7046, 80406+7135, 82472+7053, 82974+7768, 83392+7581, 92988, 94867]
                elasticity_of_retirements = 0.7

            elif housing_type == 'multifamily'
                factor = 0.96
                use_columns = "D:E"
                pub_total = [99931-6094, 102852-6688, 105661-6908, 104592-6983, 106611-7072, 109457-7647, 112357-8301, 115253-8433, 119117-8876, 120777-8971,
                             124377-8630, 128203-8705, 130112-8769, 132419-9049, 132832-8603, 33046, 34067]
                all_stock_sf = [60607+4514, 61775+5496, 63587+5703, 63646+6156, 64283+6079, 66189+6213, 68109+6778, 70355+8027, 73427+8428, 4916+7227, 
                77703+7046, 80406+7135, 82472+7053, 82974+7768, 83392+7581, 92988, 94867]
                actual_stock = pub_total - all_stock_sf
                actual_stock = actual_stock.multiply()
                elasticity_of_retirements = 0.8


            else: 
                housing_units_completed_or_placed =   # Added (place_nsa_all)
                columns = 'US Total'
                factor = 0.96
                elasticity_of_retirements = 0.5


            new_comps_ann = pd.read_excel('C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020/AHS_summary_results_051720.xlsx', sheet_name='Comps Ann', skiprows=26, 
                                        usecols=use_columns, header=None).dropna()
            print(new_comps_ann)
            year_array = list(range(1985, 2019))

            # predicted_total_stock_series = housing_stock_model(year_array, new_comps_ann.values, coeffs=x0)
            # print(predicted_total_stock_series)

            # S is the actual housing stock (column X)
            # ca is the Comps Ann (column E)
            # Maybe use scipy.optimize.curve_fit?
            def residuals(coeffs, actual_stock, year_array, ca):
                return actual_stock - housing_stock_model(year_array, ca, coeffs)

            # x0 = [-826, 9.8e-6, 1]  # use Excel solver values as starting?
            x0 = [-825, 9.7e-6, 0.9]
            ca = new_comps_ann.values
            x, flag = scipy.optimize.leastsq(residuals, x0, args=(actual_stock, year_array, ca), maxfev=1600)
            print('X: ', x)
            print('flag: ', flag)

            predicted_total_stock = housing_stock_model(year_array, ca, x)  # need the one that doesn't "skip"
            total_vacancy_rate = 
            occupied_predicted = (1 - total_vacancy_rate).multiply(predicted_total_stock)


                # Model for average housing unit size


                # elif housing_type == 'multi_family':
                all_single_family = actual_stock
                occupied_single_family =  # column J Total_stock_SF
                households =  # National_Calibration'!E13
                year = 
                

            return housing_units_completed

    def final_floorspace_estimates():
        
        number_occupied_units_national, average_size_national = get_housing_stock()

        regions = ['National', 'Northeast', 'Midwest', 'South', 'West']
        final_results_total_floorspace_regions = dict()

        for region in regions: 
            calculated_shares_by_region =  # From AHS tables
            ratios_to_national_average_size = # From AHS tables

            regional_estimates = number_occupied_units_national.multiply(calculated_shares_by_region)
            regional_estimates['Total'] = regional_estimates.sum(axis=1)
            shares_by_type = regional_estimates[['Single Family', 'Multi-Family', 'Man. Homes']].divide(regional_estimates['Total'])
            
            # Calibration Procedure
            sum_of_regions = 
            scale_factor = 
            final_check =
            average_size_all_housing_units =
            number_of_units = 

            total_square_feet =
            average_size = 
            ratios_national_average_size = 

            average_size_after_calibration = 


            final_results_total_floorspace = number_occupied_units.multiply(average_size).multiply(0.000001)
            final_results_total_floorspace['Total'] = final_results_total_floorspace.sum(axis=1)

            number_occupied_units['Total'] = number_occupied_units.sum(axis=1)
            
            final_results_total_floorspace_regions[region] = final_results_total_floorspace

        return final_results_total_floorspace_regions

    def weighted_floorspace():
        energy_types = ['Electricity', 'Fuels', 'Delivered', 'Source']
        constant_electricity_factor = 3.2

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





