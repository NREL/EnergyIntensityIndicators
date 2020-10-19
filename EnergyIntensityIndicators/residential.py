"""Overview and Assumptions:
A. Data on the number and average size of occupied housing units from the biennial American
Housing Survey were employed to generate many of the activity metrics for this sector.
B. Three types of residential housing units are distinguished: single-family, multi-family, and
manufactured homes.
C. Regional data from EIA’s State Energy Data System (SEDS) are employed to develop regional
intensity indicators.
D. Regression models at the regional level are used to adjust for year-to-year changes in weather.
E. Two separate data construction elements are required to generate the regional and national
estimates of energy intensity indicators for this sector.
    1. Regional time series of floor space for residential housing units in the U.S (census level).
    2. Weather adjustment for the four census regions.
"""
import pandas as pd
from sklearn import linear_model
from pull_eia_api import GetEIAData
from LMDI import CalculateLMDI
from Residential.residential_floorspace import ResidentialFloorspace
from weather_factors import WeatherFactors


class ResidentialIndicators(CalculateLMDI): 

    def __init__(self, directory, level_of_aggregation=None, lmdi_model='multiplicative', base_year=1985, end_year=2018):
        self.eia_res = GetEIAData('residential')
        self.sub_categories_list = {'Northeast': {'Single-Family': None, 'Multi-Family': None, 'Manufactured Homes': None}, 
                                    'Midwest': {'Single-Family': None, 'Multi-Family': None, 'Manufactured Homes': None},
                                    'South': {'Single-Family': None, 'Multi-Family': None, 'Manufactured Homes': None},
                                    'West': {'Single-Family': None, 'Multi-Family': None, 'Manufactured Homes': None}}
        self.national_calibration = self.eia_res.national_calibration()
        self.seds_census_region = self.eia_res.get_seds() # energy_consumtpion_data_regional
        self.ahs_Data = ResidentialFloorspace.update_ahs_data()
        self.conversion_factors = self.eia_res.conversion_factors()
        self.regions = ['Northeast', 'South', 'West', 'Midwest', 'National']
        self.base_year = base_year
        self.directory = directory
        self.end_year = end_year
        self.energy_types = ['elec', 'fuels', 'deliv', 'source']
        super().__init__(sector='residential', level_of_aggregation=level_of_aggregation, lmdi_models=lmdi_model, categories_dict=self.sub_categories_list, \
                    energy_types=self.energy_types, directory=directory, base_year=base_year, base_year_secondary=1996, charts_ending_year=2003)


        # self.AER11_table2_1b_update = GetEIAData.eia_api(id_='711250') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711250'
        # self.AnnualData_MER_22_Dec2019 = GetEIAData.eia_api(id_='711250') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711250' ?
        # self.RECS_intensity_data =   # '711250' for Residential Sector Energy Consumption
    
    def get_seds(self):
        census_regions = {4: 'West', 3: 'South', 2: 'Midwest', 1: 'Northeast'}
        total_fuels = self.seds_census_region[0].rename(columns=census_regions)
        elec = self.seds_census_region[1].rename(columns=census_regions)
        return total_fuels, elec

    def fuel_electricity_consumption(self, total_fuels, elec, region):
        """Combine Energy datasets into one Energy Consumption dataframe in Trillion Btu
        Data Source: EIA's State Energy Data System (SEDS)"""

        fuels_dataframe = total_fuels[[region]]
        elec_dataframe = elec[[region]]

        energy_data = {'elec': elec_dataframe, 'fuels': fuels_dataframe}
        return energy_data
    
    def get_floorspace(self):

        residential_data = ResidentialFloorspace(end_year=self.end_year)
        floorspace_square_feet, occupied_housing_units, household_size_square_feet_per_hu = residential_data.final_floorspace_estimates()

        final_floorspace_results = {'occupied_housing_units': occupied_housing_units, 'floorspace_square_feet': floorspace_square_feet, 
                                    'household_size_square_feet_per_hu': household_size_square_feet_per_hu}
        return final_floorspace_results


    def activity(self, floorspace, region):
        """Combine Energy datasets into one Energy Consumption Occupied Housing Units
        """ 
        region_activity = dict()
        for variable, data in floorspace.items():
            df = data[region]
            region_activity[variable] = df

        return region_activity
    
    def collect_weather(self, energy_dict, nominal_energy_intensity, energy_type, energy_df):
        weather = WeatherFactors(sector='residential', directory=self.directory, nominal_energy_intensity=nominal_energy_intensity)
        weather_factors = weather.get_weather(energy_dict, energy_type, energy_df, weather_adjust=False) # What should this return?? (e.g. weather factors or weather adjusted data, both?)
        return weather_factors

    def collect_data(self):
        total_fuels, elec = self.get_seds()
        floorspace = self.get_floorspace()

        all_data = dict()
        for r in self.sub_categories_list.keys(): 
            energy_data = self.fuel_electricity_consumption(total_fuels, elec, region=r)
            region_activity = self.activity(floorspace, r)

            weather_factors_by_e_type = dict()

            for e, e_df in energy_data.items():
                e_df = e_df.rename_axis(columns=None)
                floorspace = region_activity['floorspace_square_feet']
                total_floorspace = floorspace.sum(axis=1)
                nominal_energy_intensity = self.lmdi_multiplicative(activity_input_data=total_floorspace, energy_input_data=e_df, unit_conversion_factor=1, return_nominal_energy_intensity=True) # shouldn't rely on multiplicative?
                print('nominal_energy_intensity: \n', nominal_energy_intensity)
                print('energy type: \n', e)
                weather_factors = self.collect_weather(region=r, energy_dict=energy_data, nominal_energy_intensity=nominal_energy_intensity, energy_type=e, energy_df=e_df) # need to integrate this into the data passed to LMDI
                weather_factors_by_e_type[e] = weather_factors

            region_data = {'energy': energy_data, 'activity': region_activity, 'weather_factors': weather_factors_by_e_type} # HOW DOES WEATHER FIT IN?
        
        all_data[r] = region_data

        return all_data

    def main(self, breakout, calculate_lmdi):
        unit_conversion_factor = 1

        data_dict = self.collect_data()
        # print(data_dict)
        exit()
        results = self.get_nested_lmdi(level_of_aggregation=self.level_of_aggregation, breakout=breakout, calculate_lmdi=calculate_lmdi, raw_data=data_dict, account_for_weather=True)
        
        return results

    # def residential_total_lmdi_utiladj(self, _base_year=None):
    # """purpose
    #     Parameters
    #     ----------
        
    #     Returns
    #     -------
        
    # """
    #     if not base_year:
    #         _base_year = self.base_year
    #     else:
    #         _base_year = _base_year
        

    #     for key in self.sub_categories_list.keys():
    #         energy_input_data = self.fuel_electricity_consumption(key)
    #         activity_input_data = self.activity(key)

         
    #         energy_activity_data['nominal_energy_intensity_mmbtu_per_hu'] = 
    #         nominal_energy_intensity_kbtu_per_sf = 
    #         nominal_energy_intensity_kbtu_per_sf_weather_adjusted = 
    #         weather_factors_actual_by_30_year_normal = 
    #         energy_intensity_btu_per_sf_weather_adjusted_index = 


    #         energy_calc = super().lmdi_multiplicative(activity_input_data, energy_input_data, _base_year)


    #     pass


if __name__ == '__main__':
    indicators = ResidentialIndicators(directory='C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020', level_of_aggregation='All_Residential')
    indicators.main(breakout=False, calculate_lmdi=False)  


















# def regional_time_series_floor_space():
#         pass

#     def estimate_floorspace_occupied_housing_units_regional(self, ):
        
#         estimated_survival_curve = [0] # Estimate from vintage data over the 1999 through 2009 AHS surveys
#         new_housing = [0] # From Characteristics of New Housing reports from the Census Bureau
#         stock_adjustment_model = 
#         estimated_occupied_housing_units = [0] # from stock adjustment level
#         return estimated_occupied_housing_units

#     def estimate_floorspace_housing_unit_size_national(self, housing_type='single_family'):
#         """Single family and multi-family units use AHS data, combined with adjusted Characteristics of New Housing Data. Manufactured homes use RECS data
#         Data Sources: 
#             - American Housing Survey (AHS) conducted by the Census Bureau to estimate aggregate
#               floor space for three types of housing units: single-family (attached and detached), multi-family, and
#               manufactured homes. 
#             - RECS 
#             - Characteristics of New Housing
#         Spreadsheet Equivalents: 
#             - AHS_summary_results_date \Total_Stock_SF.xlsx (single family) – estimates for housing unit size are
#               to be found in this worksheet to the right of the estimates for the number of “single family” units.
#             - AHS_summary_results_date \Total_Stock_MF.xlsx (multifamily) - estimates for housing unit size are
#               to be found in this worksheet to the right of the estimates for the number of “multifamily” units.
#             - AHS_summary_results_date \Total_Stock_MH.xlsx (manufactured homes) - estimates for housing
#               unit size are to be found in this worksheet to the right of the estimates for the number of
#               “manufactured homes” units. 
#         Methodology: 
#         - Single family
#             1. Estimate the average size for existing units after 1985.
#             2. Estimates of the stock for units constructed prior to 1985, and for 1985 and subsequent
#                years, were made separately. 
#             3. The average size of new single-family homes to the existing housing stock was based upon
#                data from the Characteristics of New Housing (with a 15% upward adjustment to better
#                match the AHS data).
#         - Multifamily
#             The same procedure was followed for multi-family units to estimate average national unit size.
#         - Manufactured Homes
#             1. The estimates for manufactured home size from the AHS were deemed unsuitable for
#                inclusion in the time series estimates of residential floor space.
#             2. Instead, the size estimates for mobile homes from the various RECS were employed. While
#                the RECS had inconsistent methods of estimating square footage for single- and multi-family
#                housing units, that does not appear to be the case for mobile homes. 
#                 """
#         if housing_type == 'manufactured_homes':
#             size_estimates = 
#         else: 
#             average_size_post_1985 = 
#             stock_units_pre_1985 = 
#             stock_units_post_1985 = [0] # including 1985

#         housing_stock = ResidentialFloorspace.get_housing_stock(housing_type)

        

#     def estimate_floorspace_regional_shares_national_level_housing_units(self, ):
#         """Smooth out some of the implausible changes in the reported number of housing from one AHS to the next. The overall methodology is described more
#            generally in the comprehensive documentation report, Section A.1.2. The regional shares for the non-AHS years are computed via a simple average of the preceding (odd) year and subsequent (odd) year.
#         Data Source: AHS
#         Spreadsheet Equivalent: 
#             - AHS_Summary_Results.xlsx
#         Methodology: 
#             - The regional shares for the non-AHS years are computed via a simple average of the preceding (odd)
#               year and subsequent (odd) year."""
        

#         return final_floorspace_estimates

#     def estimate_final_floorspace_by_housing_type(self, ):
#         """Data Source: AHS
#         Spreadsheet Equivalent: AHS_Summary_Results_date \Final Floorspace Estimates.xlsx
#         Methodology:  
#             - The estimates of floor space are calculated by multiplying the number of housing
#               units times the average size per unit
#             - Use regional based estimates of floor space (as explained in the sections above) as control 
#               totals to which the regional estimates are calibrated. 
#             - """

#                     ahs_tables = 

#         national_calibration = self.national_calibration
#         comps_ann_2015 =  housing_units_completed
#         total_stock = 

#         weighted_floorspace = 
#         pass