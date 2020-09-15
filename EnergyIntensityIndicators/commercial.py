import pandas as pd
import datetime as dt
from sklearn import linear_model
from .weather_factors import weather_factors
import math
import statsmodels.api as sm
from pull_eia_api import GetEIAData
# from outline import LMDI



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

class GetCommercialData:
    """
    Data Sources: 
    - New construction is based on data from Dodge Data and Analytics. Dodge data on new floor space additions is available 
    from the published versions of the Statistical Abstract of the United States (SAUS). The Most recent data is from the 2020 
    SAUS, Table 995 "Construction Contracts Started- Value of the Construction and Floor Space of Buildings by Class of Construction:
    2014 to 2018". 
    """    

    def __init__(self):
        pass

    def get_saus_table_995():
        pass
    
    @staticmethod
    def floorspace_estimates():
        previous_year_stock = pd.read_excel('./')
        new_construction = pd.read_excel('./')  # Floor space reported in million square feet
        additions_completed_same_year = .4  # Fraction of new construction completed the same year construction began
        additions_with_lagged_completion = .6  # Fraction of new construction completed the following year'
        dodge_adjustment = 1.2  # Account for underreporting by Dodge (column AD in spreadsheet)

        def survival_function():
            """Non-linear regression model applied to the vintage data from the 1989 and 1999 CBECS
            """
            recessional_reductions_level_of_removals =  # 30 to 40% 
            demolitions_before_2008 = 
            demolitions_after_2008 =             
            pass

        pass
    
    @staticmethod
    def fuel_and_electricity_consumption_census_region():
        """Data Source: EIA's State Energy Data System (SEDS)
        """
        eia_seds = pd.read_ # 1960-2017 SEDS data: https://www.eia.gov/state/seds/seds-data-complete.php?sid=US
        eia_pivot = pd.pivot_table(eia_seds, index=['year', 'region'], columns='MSN', aggfunc='sum')
        eia_pivot = eia_pivot[['ESCCB', 'TNCCB']]
        total_fuel_consumption = total_energy - electricity_sales 
        energy_consumtpion_data_regional = 
        approximate_intesity_time_series = 
        weather_adjustment_factors_regional = 
        energy_consumption_regional = 

    @staticmethod


    @staticmethod
    def reclassification_electricity_sales():
        """[summary]
        """






class CommercialIndicators(LMDI):
    
    def __init__(self, energy_data, activity_data, categories_list):
        super().__init__(energy_data, activity_data, categories_list)
         
        self.sub_categories_list = categories_list['commercial']
        self.conversion_factors = eia_comm.conversion_factors()
        self.cbecs = 
        self.residential_housing_units = # Use regional estimates of residential housing units as interpolator, extrapolator via regression model
        self.SEDS_CensusRgn = eia_comm.get_seds()
        self.national_calibration = eia_comm.national_calibration()
        self.Weather_Factors = WeatherFactors.weather_factors('commercial')


        # self.mer_data23_May_2016 = GetEIAData.eia_api(id_='711251')  # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711251'
        # self.mer_data23_Jan_2017 = GetEIAData.eia_api(id_='711251')   # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711251'
        # self.mer_data23_Dec_2019 =  GetEIAData.eia_api(id_='711251')  # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711251'
        # self.AER11_Table21C_Update = GetEIAData.eia_api(id_='711251')  # Estimates?
        self.mer_data_23 = eia_comm.eia_api(id_='711251', id_type='category')

    def estimate_regional_floorspace_share(self,):
    """assumed commercial floorspace in each region follows same trends as population or housing units"""

    def estimate_regional_floorspace(self):
        """[summary]
        """        

    def estimate_intensity_indexes_regional(self, parameter_list):
        """Data Sources: Fuel Consumption and electricity consumption from SEDS, Shares of regional floorspace from CBECs
        Purpose: used to produce weather adjustment facotrs"""
                """[summary]
        -Regional etsimates of floorspace are derived by applying regional shares to the total national floor space that were 
        estimatedwithin the spreadsheet historical_floorspace.xlsx
        -The time series of regional shares are estimated in the worksheet "Regional Shares" (key data for this process are the 
        shares of regional floor space reported in the various Commercial Building Energy Consumption Surveys (CBECS) or NBECS for 
        years prior to 1986)
        -To provide annual estimates of the shares, the assumption was made that commercial floor space in each region would 
        generally follow the same trends as population or housing units. Here residential estimates of residential housing units
        were used to reflect these overall trends. 
        -For each region, a simple regression model was estimated between the regional housing unit share and the regional commmercial
        building floor space share from the NBECS/CBECS. The regression employed both shares in log form. Based on the estimated 
        coefficients from this regression, the annual housing unit values are used to predict the share of commercial floor space 
        in each region. The normalized shares of floorspace by census region (normalized to sum to 1) are contained in Regional_Floorspace
        columns E through H. 
        -Regional floor space levels are calculated by multiplying the regional shares times the national estimate of floorspace  (taken 
        from sheet called Commercial_Total)

        """
        
        total_national_floorspace = 

        X = regional_housing_unit_share
        Y = regional_commercial_building_floorspace_shares  # from historical CBECS

        X = sm.add_constant(x)  # Add constant

        model = sm.OLS(X, Y).fit
        predictions = model.predict(x)

        print(model.summary())

        # reg = linear_model.LinearRegression()
        # reg.fit(X, Y)
        # coefficients = reg.coef_
        # regional_shares_floorspace = dict()  # assumed commercial floorspace in each region follows same trends as population or housing units
        # regional_floorspace = dict()
        # regional_intensity_index = dict()
        # for region in regions: 
        #     regional_energy_consumption =  # consumption of electricity and fuels
        #     predicted_share = reg.predict(annual_housing_unit_values)
        #     regional_shares_floorspace[region] = predicted_share
        #     region_floorspace =  predicted_share * total_national_floorspace
        #     regional_floorspace[region] = region_floorspace
        #     regional_intensity_index[region] = regional_energy_consumption / region_floorspace  

        return None

    def conversion_factors():
        """[summary]
        """        
        
    def estimate_reclassification_electricity_sector_sales(self,):
        """Data Source: Commercial electricity sales from EIA SEDS
        Assumption: The significant changes (same magnitude but opposite directions in the same year) in state-level electricity slaes, typically
                    showing up in the commercial and industrial sectors reflect reclassification of some customers form the industrial rate class
                    to the commercial rate class or vice versa. 
        Strategy: adjust the more recent data by adding or subtracting a constant vlaue, determined from the year in which the reclassification was 
                  judged to have occured."""
        state_level_adjustment = 
        pass

    def adjusted_supplier_data(self,):
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

        """"
        years = list(range(1949, 2019))
        # 1949-1969 
        published_consumption_trillion_btu = list(self.AER11_Table2.1C_Update[])  # Column W (electricity retail sales to the commercial sector) # for years 1949-69
        # 1970-2018
        published_consumption_trillion_btu.append(list(self.national_calibration[]))  # Column G (electricity final est) # for years 1970-2018
        # 1977-1989
        adjustment_to_commercial_trillion_btu_early = number_for_1990
        adjustment_to_commercial_trillion_btu = [9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340312799975
                                                 9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340312799975,
                                                 9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340654000005, 
                                                 29.77918535999970, 10.21012680399960, 1.70263235599987, -40.63866012000020, -40.63865670799990, 
                                                 -117.72073870000000, -117.72073528800000, -117.72073187600000, -117.72072846400000, -162.61452790400100, 
                                                 -136.25241618800100, -108.91594645600000, -125.97594304400000, -125.97593963200100, -163.95020989600000,
                                                 -163.95020648400000, -163.95020307200000, -137.98708428968000, -137.98487966000100, -137.98487966000100, 
                                                 -137.98487966000100, -137.98487966000100, -137.98487966000100, -137.98487966000100, -137.98487966000100, 
                                                 -137.98487966000100, -137.98487966000100] # First value is for 1977 - 2018
        adjusted_supplier_data = pd.DataFrame([years, adjustment_to_commercial_trillion_btu]).transpose()
        adjusted_supplier_data.columns = ['year', 'adjustment_to_commercial_trillion_btu']
        adjusted_consumption_trillion_btu = published_consumption_trillion_btu + adjustment_to_commercial_trillion_btu_early
        
        return adjusted_consumption_trillion_btu

    def national_calibration(self,):
        """"""
        pass

    def regional_intensity_aggregate(self,):
        """""""
        pass

    def energy_consumption():
        """Trillion Btu
        """
        sources: {}
        pass

    def activity():
        """Floor Space
        """                

@staticmethod
def dodge_revised():
    """Dodge Additions, adjusted for omission of West Census Region prior to 1956
    """    
    commercial_excl_hotel = []  # hist_stat_adj column Q
    saus_2002_commercial = {1990: 694, 1991: 476, 1192: 462, 1993: 481}
    commercial_incl_hotel
    dodge_revised = pd.DataFrame().set_index('Year')
    dodge_revised.loc[list(range(1919, 1990)), ['Commercial, Incl Hotel']] = dodge_revised.loc[list(range(1919, 1990)), ['Commercial, Excl Hotel']].add(dodge_revised.loc[list(range(1919, 1990)), ['Hotel']])
    dodge_revised.loc[list(range(1990, 1998)), ['Commercial, Incl Hotel']] =  # SAUS2002 column E
    dodge_revised.loc[list(range(1998, 2018)), ['Commercial, Incl Hotel']] =  

    revision_factor_commercial = sum(list(dodge_revised.loc[list(range(1985, 1990)), ['Commercial']]))
    revision_factor_retail = sum(list(dodge_revised.loc[list(range(1985, 1990)), ['Retail']])) / revision_factor_commercial
    revision_factor_auto_r = sum(list(dodge_revised.loc[list(range(1985, 1990)), ['Auto R']])) / revision_factor_commercial
    revision_factor_office = sum(list(dodge_revised.loc[list(range(1985, 1990)), ['Office']])) / revision_factor_commercial
    revision_factor_warehouse = sum(list(dodge_revised.loc[list(range(1985, 1990)), ['Warehouse']])) / revision_factor_commercial
    revision_factor_hotel = sum(list(dodge_revised.loc[list(range(1985, 1990)), ['Hotel']])) / revision_factor_commercial


    dodge_revised.loc[list(range(1919, 1960)), [['Retail', 'Auto R', 'Office', 'Warehouse']]] = 
    
    dodge_revised.loc[list(range(1960, 1982)), [['Retail', 'Auto R', 'Office', 'Warehouse']]] = # DODCompareOld columns B,D,C,E

    dodge_revised.loc[list(range(1919, 1980)), ['Hotel']] = 
    dodge_revised.loc[list(range(1980, 1990)), ['Hotel']] = # DODCompareOld column AB

    dodge_revised.loc[list(range(1990, 2019)), [['Retail', 'Auto R', 'Office', 'Warehouse', 'Hotel']]] = 


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

    dodge_revised = # dataframe
    
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
    dataframe['age'] = dataframe['year'].subtract(current_year).multiply(-1)
    dataframe['remaining'] = 1.divide(1.add(dataframe['age'].divide(params[1])).pow(params[0]))
    dataframe['inflate_fac'] = 1.divide(dataframe['remaining'])

    dodge_to_cbecs = self.dodge_to_cbecs() # columns c-m starting with year 1920 (row 17)

x0 = [3.92276415, 73.2238120168849]  # [gamma, lifetime]