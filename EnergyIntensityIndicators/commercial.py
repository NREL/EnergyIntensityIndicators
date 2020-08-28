import pandas as pd
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
        self.conversion_factors = GetEIAData.conversion_factors('commercial')
        self.cbecs = 
        self.residential_housing_units = # Use regional estimates of residential housing units as interpolator, extrapolator via regression model
        self.SEDS_CensusRgn = GetEIAData.get_seds(sector='commercial')
        self.national_calibration = GetEIAData.national_calibration(sector='commercial')
        self.Weather_Factors = WeatherFactors.weather_factors('commercial')


        # self.mer_data23_May_2016 = GetEIAData.eia_api(id_='711251')  # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711251'
        # self.mer_data23_Jan_2017 = GetEIAData.eia_api(id_='711251')   # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711251'
        # self.mer_data23_Dec_2019 =  GetEIAData.eia_api(id_='711251')  # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711251'
        # self.AER11_Table21C_Update = GetEIAData.eia_api(id_='711251')  # Estimates?
        self.mer_data_23 = GetEIAData.eia_api(id_='711251', id_type='category')

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
        # 1949-1969 
        published_consumption_trillion_btu = list(self.AER11_Table2.1C_Update[])  # Column W
        # 1970-2018
        published_consumption_trillion_btu.append(list(self.national_calibration[]))  # Column G
        # 1977-1989
        adjustment_to_commercial_trillion_btu_early = number_for_1990
        adjustment_to_commercial_trillion_btu_early # WHERE DOES THIS FORM FROM
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