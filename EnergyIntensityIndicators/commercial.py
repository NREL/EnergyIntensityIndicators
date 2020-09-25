import pandas as pd
import datetime as dt
from sklearn import linear_model
from weather_factors import WeatherFactors
import math
import statsmodels.api as sm
from pull_eia_api import GetEIAData
from scipy.optimize import curve_fit
from LMDI import LMDI

"""Overview and Assumptions: 
A. A national time series of floorspace for the commercial buildings in the US
B. Weather adjustment for the four census regions
C. Adjustment for the major reclassifications of customers by individual utilities. The reclassifications generally involve customers 
   whose electricity purchases were classified under an industrial rate structure being moved to commercial rate struture, and to a
   lesser degree, customers moving from a commercial rate to an industrial rate classification. These reclassfications can distort the 
   short term aggregate changes in commercial and industrial electricity consumption, as reported by EIA's Estimation of National Commercial 
   Floorspace

Data Sources: New construction is based on data from Dodge Data and Analytics, available from the published versions of the Statistical
              Abstract of the United States (SAUS)

Methodology: Perpetual inventory model, where estimates of new additions and removals are added to the previous year's estimate of stock
             to update the current year"""

class CommercialIndicators(LMDI):
    """
    Data Sources: 
    - New construction is based on data from Dodge Data and Analytics. Dodge data on new floor space additions is available 
    from the published versions of the Statistical Abstract of the United States (SAUS). The Most recent data is from the 2020 
    SAUS, Table 995 "Construction Contracts Started- Value of the Construction and Floor Space of Buildings by Class of Construction:
    2014 to 2018". 
    """    

    def __init__(self):
        self.sub_categories_list = {'Commercial_Total': None, 'Total_Commercial_LMDI_UtilAdj': None}
        self.eia_comm = GetEIAData('commercial')
        # self.cbecs = 
        # self.residential_housing_units = # Use regional estimates of residential housing units as interpolator, extrapolator via regression model

        # self.mer_data23_May_2016 = GetEIAData.eia_api(id_='711251')  # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711251'
        # self.mer_data23_Jan_2017 = GetEIAData.eia_api(id_='711251')   # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711251'
        # self.mer_data23_Dec_2019 =  GetEIAData.eia_api(id_='711251')  # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711251'
        # self.AER11_Table21C_Update = GetEIAData.eia_api(id_='711251')  # Estimates?

    def collect_data(self, dataset_name):
        datasets = {'national_calibration': self.eia_comm.national_calibration(), 'conversion_factors': self.eia_comm.conversion_factors(include_utility_sector_efficiency_in_total_energy_intensity=True), 
                    'SEDS_CensusRgn': self.eia_comm.get_seds(), 'mer_data_23': self.eia_comm.eia_api(id_='711251', id_type='category')}
        return datasets[dataset_name]

    @staticmethod
    def get_saus():
        """Get Data from the Statistical Abstract of the United States (SAUS)
        """        
        saus_2002 = pd.read_csv('./SAUS2002_table995.csv').set_index('Year')
        saus_1994 = {1980: 738, 1981: 787, 1982: 631, 1983: 716, 1984: 901, 1985: 1039, 1986: 960, 1987: 933, 
                    1988: 883, 1989: 867, 1990: 694, 1991: 477, 1992: 462, 1993: 479}
        saus_2001 = {1980: 738, 1981: None, 1982: None, 1983: None, 1984: None, 1985: 1039, 1986: None, 1987: None, 
                    1988: None, 1989: 867, 1990: 694, 1991: 476, 1992: 462, 1993: 481, 1994: 600, 1995: 700, 
                    1996: 723, 1997: 855, 1998: 1106, 1999: 1117, 2000: 1176}
        saus_merged = dict() 
        for (year, value) in saus_2001.items():
            if value == None: 
                set_value = saus_1994[year]
            else: 
                set_value = value
            saus_merged[year] = set_value

        return saus_2002, saus_merged

    def adjusted_supplier_data(self):
        """
        This worksheet adjusts some of commercial energy consumption data
        as reported in the Annual Energy Review.  These adjustments are 
        based upon state-by-state analysis of energy consumption in the 
        industrial and commercial sectors.  For electricity, there have been 
        a number of reclassifications by utilities since 1990 that has moved 
        sales from the industrial sector to the commercial sector. 

        The adjustment for electricity consumption is based upon a
        state-by-state examination of commercial and electricity 
        sales from 1990 through 2011.  This data is collected
        by EIA via Survey EIA-861.  Significant discontinuities
        in the sales data from one year to the next were removed.  
        In most cases, these adjustments caused industrial consumption
        to increase and commercial consumption to decrease.  The
        spreadsheet with these adjustments is Sectoral_reclassification5.xls  (10/25/2012).

        In 2009, there was a significant decline in commercial
        electricity sales in MA and a corresponding increase in industrial sales
        Assuming that industrial consumption would have
        fallen by 2% between 2008 and 2009, the adjustment
        to both the commercial (+) and industrial sectors (-) was
        estimated to be 7.61 TWh.  .
        The 7.61 TWh converted to Tbtu is 26.0.  This value is then added 
        to the negative 164.0 Tbtu in 2009 and subsequent years.  

        State Energy Data System (Jan. 2017) via National Calibration worksheet 

        """

        # 1949-1969 
        published_consumption_trillion_btu = self.eia_comm.eia_api(id_='TOTAL.ESCCBUS.A', id_type='series')  # Column W (electricity retail sales to the commercial sector) # for years 1949-69
        published_consumption_trillion_btu = published_consumption_trillion_btu.rename(columns={'Electricity Retail Sales to the Commercial Sector, Annual, Trillion Btu': 'published_consumption_trillion_btu'})
        # 1970-2018
        national_calibration = self.collect_data('national_calibration')
        print('replacement data:', national_calibration.loc['1970': '2018', ['Final Est. (Trillion Btu)_elec']])
        published_consumption_trillion_btu.loc['1970':, ['published_consumption_trillion_btu']] = national_calibration.loc['1970':, ['Final Est. (Trillion Btu)_elec']].values  # Column G (electricity final est) # for years 1970-2018
        print('published_consumption_trillion_btu:', published_consumption_trillion_btu)
        # 1977-1989

        years = list(range(1977, max(published_consumption_trillion_btu.index.astype(int)) + 1))
        years = [str(y) for y in years]
        # adjustment_to_commercial_trillion_btu_early = number_for_1990
        adjustment_to_commercial_trillion_btu = [9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340312799975,
                                                 9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340312799975,
                                                 9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340654000005, 
                                                 29.77918535999970, 10.21012680399960, 1.70263235599987, -40.63866012000020, -40.63865670799990, 
                                                 -117.72073870000000, -117.72073528800000, -117.72073187600000, -117.72072846400000, -162.61452790400100, 
                                                 -136.25241618800100, -108.91594645600000, -125.97594304400000, -125.97593963200100, -163.95020989600000,
                                                 -163.95020648400000, -163.95020307200000, -137.98708428968000, -137.98487966000100, -137.98487966000100, 
                                                 -137.98487966000100, -137.98487966000100, -137.98487966000100, -137.98487966000100, -137.98487966000100, 
                                                 -137.98487966000100, -137.98487966000100] # First value is for 1977 - 2018
        adjustment_df = pd.DataFrame([years, adjustment_to_commercial_trillion_btu]).transpose()
        adjustment_df.columns = ['Year', 'adjustment_to_commercial_trillion_btu']

        adjusted_supplier_data = adjustment_df.merge(published_consumption_trillion_btu, how='outer', on='Year')
        adjusted_supplier_data['adjustment_to_commercial_trillion_btu'] = adjusted_supplier_data['adjustment_to_commercial_trillion_btu'].fillna(0)
        adjusted_supplier_data = adjusted_supplier_data.set_index('Year')
        print(adjusted_supplier_data)
        adjusted_supplier_data['adjusted_consumption_trillion_btu'] = adjusted_supplier_data['adjustment_to_commercial_trillion_btu'].add(adjusted_supplier_data['published_consumption_trillion_btu'])
        print('adjusted_supplier_data:', adjusted_supplier_data)
        adjusted_supplier_data = adjusted_supplier_data.sort_index(ascending=True)
        return adjusted_supplier_data['adjusted_consumption_trillion_btu']

    def regional_intensity_aggregate(self):
        """, """
        pass

    @staticmethod
    def dod_compare_old():
        dod_old = pd.read_csv('./').set_index('Year')
        # dod_old['Misc'] = dod_old['Soc/Misc'].subtract(dod_old['Soc/Amuse'])
        # dod_old = dod_old.drop(columns='Soc/Misc')
        # dod_old['Total'] = dod_old.sum(axis=1)
        dod_old['Commercial'] = dod_old[['Retail', 'Auto R', 'Office', 'Warehouse']].sum(axis=1)

        dod_old_subset = dod_old.loc[list(range(1960, 1982)), [['Retail', 'Auto R', 'Office', 'Warehouse']]]
        dod_old_hotel = dod_old.loc[list(range(1980, 1990)), ['Commercial']]
        return dod_old_subset, dod_old_hotel 

    @staticmethod
    def dodge_adjustment_ratios(dodge_dataframe, start_year, stop_year):
        (1985, 1990) or (1960, 1970)
        year_indices = list(range(start_year, stop_year))
        revision_factor_commercial = sum(list(dodge_revised.loc[year_indices, ['Commercial']]))
        categories = ['Retail', 'Auto R', 'Office', 'Warehouse']
        revision_factors = []
        for category in categories: 
            revision_factor_cat = sum(list(dodge_revised.loc[year_indices, [category]])) / revision_factor_commercial
            revision_factors.append(revision_factor_cat)
        return revision_factors

    @staticmethod
    def west_inflation():
        # hist_stat column E
        # west inflation column Q
        # ornl_78 = 
        final_factors = dict()
        for year_ in list(range(1925, 1955 + 1)): 
            if year_ in ornl_years: 
                final_factors[year_] = ornl_78[year]
            else: 
                diff = []

    @staticmethod
    def dodge_revised():
        """Dodge Additions, adjusted for omission of West Census Region prior to 1956
        """       
        saus_2002, saus_merged = get_saus()
        dod_old_subset, dod_old_hotel = dod_compare_old()
        # west_inflation = 

        dodge_revised = pd.DataFrame().set_index('Year')
        dodge_revised.loc[list(range(1919, 1990)), ['Commercial, Incl Hotel']] = dodge_revised.loc[list(range(1919, 1990)), ['Commercial, Excl Hotel']].add(dodge_revised.loc[list(range(1919, 1990)), ['Hotel']])
        dodge_revised.loc[list(range(1990, 1998)), ['Commercial, Incl Hotel']] =  saus_2002.loc[list(range(1990, 1998)), ['Commercial']]
        # dodge_revised.loc[list(range(1998, 2018)), ['Commercial, Incl Hotel']] =  # hard coded

        revision_factors_60_69 = dodge_adjustment_ratios(dodge_revised, 1960, 1969 + 1)
        revision_factors_85_89 = dodge_adjustment_ratios(dodge_revised, 1985, 1989 + 1)

        dodge_revised.loc[list(range(1960, 1982)), [['Retail', 'Auto R', 'Office', 'Warehouse']]] = dod_old_subset 

        # dodge_revised.loc[list(range(1919, 1960)), ['Commercial, Excl Hotel']] =  # hist_stat_adj column Q
        dodge_revised.loc[list(range(1960, 1990)), ['Commercial, Excl Hotel']] =  dodge_revised.loc[list(range(1960, 1990)), [['Retail', 'Auto R', 'Office', 'Warehouse']]].sum(index=1)
        dodge_revised.loc[list(range(1990, 2019)), ['Commercial, Excl Hotel']] = np.nan

        dodge_revised.loc[list(range(1919, 1960)), [['Retail', 'Auto R', 'Office', 'Warehouse']]] = dodge_revised.loc[list(range(1919, 1960)), ['Commercial, Excl Hotel']].multiply(revision_factors_60_69) 
        dodge_revised.loc[list(range(1990, 2019)), [['Retail', 'Auto R', 'Office', 'Warehouse']]] = dodge_revised.loc[list(range(1919, 1960)), ['Commercial']].multiply(revision_factors_85_89) 

        hotel_80_89 = saus_merged.subtract(dod_old_hotel) 
        dodge_revised.loc[list(range(1980, 1990)), ['Hotel']] = hotel_80_89
        
        hotel_80_89_ratio = sum(hotel_80_89) / sum(dodge_revised.loc[list(range(1980, 1990)), ['Commercial, Excl Hotel']])
        dodge_revised.loc[list(range(1919, 1980)), ['Hotel']] = dodge_revised.loc[list(range(1919, 1980)), ['Commercial, Excl Hotel']].multiply(hotel_80_89_ratio)


        dodge_revised.loc[list(range(1990, 2019)), [['Retail', 'Auto R', 'Office', 'Warehouse', 'Hotel']]] = []


    @staticmethod
    def dodge_to_cbecs():
        """Redefine the Dodge building categories more along the lines of CBECS categories. Constant fractions of floor space are moved among categories. 

        Returns:
            dodge_to_cbecs (dataframe): redefined data
        """    
        # Key Assumptions: 
        education_floor_space_office = .10
        auto_repair_retail = .80
        retail_merc_service = .80  # remainder to food service and sales
        retail_merc_service_food_sales = .11
        retail_merc_service_food_service = .90
        education_assembly = .05
        education_misc = .05 # (laboratories)
        health_transfered_to_cbecs_health = .75 # 25% to lodging (nursing homes)
        misc_public_assembly = .10 # (passenger terminals)

        # dodge_revised = # dataframe
        
        dodge_to_cbecs = pd.dataframe(dodge_revised[['Year', 'Total', 'Religious', 'Warehouse']]).rename(columns={'Total': 'Dodge_Totals'}).set_index('index')

        dodge_to_cbecs['Office'] = dodge_revised['Office'] + education_floor_space_office * dodge_revised['Education']
        dodge_to_cbecs['Merc/Serv'] = retail_merc_service * (dodge_revised['Retail'] + auto_repair_retail * dodge_revised['Auto R'])
        dodge_to_cbecs['Food_Sales'] = retail_merc_service_food_sales * (dodge_revised['Retail'] + auto_repair_retail * dodge_revised['Auto R'])
        dodge_to_cbecs['Food_Serv'] = retail_merc_service_food_service * (dodge_revised['Retail'] + auto_repair_retail * dodge_revised['Auto R'])
        dodge_to_cbecs['Education'] = (1 - education_floor_space_office - education_assembly - education_misc) * dodge_revised['Education']
        dodge_to_cbecs['Health'] = health_transfered_to_cbecs_health * dodge_revised['Hospital']
        dodge_to_cbecs['Lodging'] = dodge_revised['Hotel']+ (1 - health_transfered_to_cbecs_health) * dodge_revised['Hospital']
        dodge_to_cbecs['Assembly'] = dodge_revised['Soc/Amus'] +  misc_public_assembly * dodge_revised['Misc'] + education_assembly * dodge_revised['Education']
        dodge_to_cbecs['Other'] = dodge_revised['Public'] + (1 - misc_public_assembly) * dodge_revised['Misc'] + (1 - auto_repair_retail) * dodge_revised['Auto R'] + education_misc * dodge_revised['Education']
        dodge_to_cbecs['Redefined_Totals'] = dodge_to_cbecs.sum(index=1)
        
        # dodge_to_cbecs = dodge_to_cbecs.drop()  # don't need totals?
        return dodge_to_cbecs

    def nems_logistic(self, dataframe, params):
        """[summary]

        Args:
            dataframe ([type]): [description]
            params (list): gamma, lifetime, 
        """    
        current_year = dt.datetime.now().year
        dataframe['age'] = dataframe['Year'].astype(int).subtract(current_year).multiply(-1)
        dataframe['remaining'] = ((dataframe['age'].divide(params[1]).add(1)).pow(params[0])).pow(-1)
        dataframe['inflate_fac'] = dataframe['remaining'].pow(-1)

        link_factors = pd.read_excel(f'{self.directory}/CO-EST_statepop2.xls', sheet_name='Stock', usecols='E', index_col='D', skiprows=158, header=4)
        column_t_0 = 0.1
        column_t_1 = 40
        column_t = link_factors.multiply(column_t_0)

        timing_wgts_current_yr = 0.4
        timing_wgts_lag_yr = 0.6
        benchmark_factor = 1

        vpip_estimates = []
        for index, value in enumerate(link_factors):
            if index == 0: 
                vpip_estimate = column_t[index]
            elif index == 14: ######YEAR??##### == 1920: 
                vpip_estimate = column_t[index]
            elif index > 0:
                vpip_estimate = (timing_wgts_current_yr * column_t[index] + timing_wgts_lag_yr * column_t[index - 1]) * benchmark_factor
            vpip_estimates.append(vpip_estimate)

        x_column_value = [_variable] * len(range(1990, 2021))
        db_estimates = [1.2] * len(range(1990, 2021))
        db_estimates2 = [1.25 - 0.01*d for d in list(range(1990, 2021))]

        post_1989_scaling_factor_key = db_estimates # Should choose this

        without_lags = column_t * post_1989_scaling_factor_key

        scaled_additions_estimate_a = vpip_estimates * _variable_2
        column_t_adjusted = 1.15 * column_t
        scaled_additions_estimate_b = []
        scaled_additions_estimate_c = []

        for index_, value in enumerate(column_t_adjusted):
            calc_value_b = np.dot(column_t_adjusted[:index_], dataframe.loc[:index_, ['remaining']])
            scaled_additions_estimate_b.append(calc_value_b)

            calc_value_c = np.dot(scaled_additions_estimate_a[:index_], dataframe.loc[:index_, ['remaining']])
            scaled_additions_estimate_c.append(calc_value_c)

        removal_chg = 1 # Not sure what this is about
        fractions = [0.3, 0.4, 0.4, 0.35, 0.35, 0.35, 0.35, 0.3, 0.3, 0.3]
        fraction_retained = [f * removal_chg for f in fractions]
        
        removals = []
        for i in range(len(scaled_additions_estimate_a)):
            if i > 0: 
                removals.append(scaled_additions_estimate_c[i] - scaled_additions_estimate_c[i - 1] - scaled_additions_estimate_a[i])
        
        adjusted_removals = fraction_retained * removals

        adjust_removals = []          
        for i in range(len(fraction_retained)):
            if i == 0: 
                adjustment = adjusted_removals[i]
            else: 
                adjustment = adjust_removals[i - 1] + adjusted_removals[i]
            adjust_removals.append(adjustment)
        

        return scaled_additions_estimate_c - adjust_removals
        
    def solve_logistic(self, dataframe):
        """Solve NES logistic parameters
        """    

        pnnl_coefficients = [3.92276415, 73.2238120168849]  # [gamma, lifetime]
        # popt, pcov = curve_fit(self.nems_logistic, xdata=dataframe[], ydata=dataframe[] , p0=pnnl_coefficients)
        # return popt 
        return pnnl_coefficients

    def activity(self):
        """Use logistic parameters to find predicted historical floorspace
        """ 
        dodge_to_cbecs = self.dodge_to_cbecs() # columns c-m starting with year 1920 (row 17)
        coeffs = self.solve_logistic(dodge_to_cbecs)
        historical_floorspace_late = self.nems_logistic(dodge_to_cbecs, coeffs)  # properly formatted?

        historical_floorspace_early = {1949: 27235.1487296062, 1950: 27788.6370796569, 1951: 28246.642791733, 1952: 28701.4989706012, 
                                                    1953: 29253.2282427217, 1954: 29913.8330998026, 1955: 30679.7157232176, 1956: 31512.6191323126,
                                                    1957: 32345.382764321, 1958: 33206.8483392728, 1959: 34088.6640247816}
        historical_floorspace_early = pd.DataFrame(historical_floorspace_early, columns=['Year', 'value'])

        historical_floorspace = pd.concat([historical_floorspace_early, historical_floorspace_late])
        historical_floorspace_billion_sq_feet = historical_floorspace.multiply(0.001)

        return historical_floorspace_billion_sq_feet

    def fuel_electricity_consumption(self):
        """Trillion Btu
        """
        year_range = list(range(1949, 1970))
        year_range = [str(y) for y in year_range]
        national_calibration = self.collect_data('national_calibration')
        total_primary_energy_consumption = self.eia_comm.eia_api(id_='TOTAL.TXCCBUS.A', id_type='series') # pre 1969: AER table 2.1c update column U 
        total_primary_energy_consumption = total_primary_energy_consumption.rename(columns={'Total Primary Energy Consumed by the Commercial Sector, Annual, Trillion Btu': 'total_primary'})
        # total_primary_energy_consumption = total_primary_energy_consumption[total_primary_energy_consumption.index.isin(year_range)]
        # total_primary_energy_consumption = total_primary_energy_consumption.multiply(0.001)

        fuels_dataframe = total_primary_energy_consumption.copy()
        print('fuels_dataframe', fuels_dataframe)
        replacement_data = national_calibration.loc['1970':, ['Final Est. (Trillion Btu)_fuels']]  # >= 1970: National Calibration Column 0
        print('replacement_data:', replacement_data)
        fuels_dataframe.loc['1970':, ['total_primary']] = replacement_data.values
        print(fuels_dataframe)
        elec_dataframe =  self.adjusted_supplier_data() 

        energy_data = {'elec': elec_dataframe, 'fuels': fuels_dataframe}
        return energy_data

    def main(self, lmdi_model='multiplicative'):
        # Activity: Floorspace_Estimates column U, B
        # Energy: Elec --> Adjusted Supplier Data Column D
        #         Fuels --> AER11 Table 2.1C_Update column U, National Calibration Column O
        energy_data = self.fuel_electricity_consumption()
        print('energy data:', energy_data)
        activity_data = self.activity()
        print('activity data:', activity_data)
        # results = self.call_lmdi(unit_conversion_factor=, adjust_for_weather=False, lmdi_model=lmdi_model)


if __name__ == '__main__':
    # CommercialIndicators().main()
    # y = CommercialIndicators().fuel_electricity_consumption()
    # print(y)
    x = CommercialIndicators().adjusted_supplier_data()
    print(x)

# CommercialIndicators().collect_data()



   




#     @staticmethod
#     def floorspace_estimates():
#         previous_year_stock = pd.read_excel('./')
#         new_construction = pd.read_excel('./')  # Floor space reported in million square feet
#         additions_completed_same_year = .4  # Fraction of new construction completed the same year construction began
#         additions_with_lagged_completion = .6  # Fraction of new construction completed the following year'
#         dodge_adjustment = 1.2  # Account for underreporting by Dodge (column AD in spreadsheet)

#         def survival_function():
#             """Non-linear regression model applied to the vintage data from the 1989 and 1999 CBECS
#             """
#             recessional_reductions_level_of_removals =  # 30 to 40% 
#             demolitions_before_2008 = 
#             demolitions_after_2008 =             
#             pass

#         pass

#     @staticmethod
#     def reclassification_electricity_sales():
#         """[summary]
#         """

#  def estimate_regional_floorspace_share(self,):
#     """assumed commercial floorspace in each region follows same trends as population or housing units"""

#     def estimate_regional_floorspace(self):
#         """[summary]
#         """        

#     def estimate_intensity_indexes_regional(self, parameter_list):
#         """Data Sources: Fuel Consumption and electricity consumption from SEDS, Shares of regional floorspace from CBECs
#         Purpose: used to produce weather adjustment facotrs"""
#                 """[summary]
#         -Regional etsimates of floorspace are derived by applying regional shares to the total national floor space that were 
#         estimatedwithin the spreadsheet historical_floorspace.xlsx
#         -The time series of regional shares are estimated in the worksheet "Regional Shares" (key data for this process are the 
#         shares of regional floor space reported in the various Commercial Building Energy Consumption Surveys (CBECS) or NBECS for 
#         years prior to 1986)
#         -To provide annual estimates of the shares, the assumption was made that commercial floor space in each region would 
#         generally follow the same trends as population or housing units. Here residential estimates of residential housing units
#         were used to reflect these overall trends. 
#         -For each region, a simple regression model was estimated between the regional housing unit share and the regional commmercial
#         building floor space share from the NBECS/CBECS. The regression employed both shares in log form. Based on the estimated 
#         coefficients from this regression, the annual housing unit values are used to predict the share of commercial floor space 
#         in each region. The normalized shares of floorspace by census region (normalized to sum to 1) are contained in Regional_Floorspace
#         columns E through H. 
#         -Regional floor space levels are calculated by multiplying the regional shares times the national estimate of floorspace  (taken 
#         from sheet called Commercial_Total)

#         """
        
#         total_national_floorspace = 

#         X = regional_housing_unit_share
#         Y = regional_commercial_building_floorspace_shares  # from historical CBECS

#         X = sm.add_constant(x)  # Add constant

#         model = sm.OLS(X, Y).fit
#         predictions = model.predict(x)

#         print(model.summary())

#         # reg = linear_model.LinearRegression()
#         # reg.fit(X, Y)
#         # coefficients = reg.coef_
#         # regional_shares_floorspace = dict()  # assumed commercial floorspace in each region follows same trends as population or housing units
#         # regional_floorspace = dict()
#         # regional_intensity_index = dict()
#         # for region in regions: 
#         #     regional_energy_consumption =  # consumption of electricity and fuels
#         #     predicted_share = reg.predict(annual_housing_unit_values)
#         #     regional_shares_floorspace[region] = predicted_share
#         #     region_floorspace =  predicted_share * total_national_floorspace
#         #     regional_floorspace[region] = region_floorspace
#         #     regional_intensity_index[region] = regional_energy_consumption / region_floorspace  

#         return None
          
#     def estimate_reclassification_electricity_sector_sales(self,):
#         """Data Source: Commercial electricity sales from EIA SEDS
#         Assumption: The significant changes (same magnitude but opposite directions in the same year) in state-level electricity slaes, typically
#                     showing up in the commercial and industrial sectors reflect reclassification of some customers form the industrial rate class
#                     to the commercial rate class or vice versa. 
#         Strategy: adjust the more recent data by adding or subtracting a constant vlaue, determined from the year in which the reclassification was 
#                   judged to have occured."""
#         state_level_adjustment = 
#         pass


