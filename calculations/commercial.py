import pandas as pd
from sklearn import linear_model

"""Overview and Assumptions: 
A. A national time series of floorspace for the commercial buildings in the US
B. Weather adjustment for the four census regions
C. Adjustment for the major reclassifications of customers by individual utilities. The reclassifications generally involve customers whose electricity purchases were classified under an industrial
rate structure being moved to commercial rate struture, and to a lesser degree, customers moving from a commercial rate to an industrial rate classification. These reclassfications can distort the 
short term aggregate changes in commercial and industrial electricity consumption, as reported by EIA's Estimation of National Commercial Floorspace

Data Sources: New construction is based on data from Dodge Data and Analytics, available from the published versions of the Statistical Abstract of the United States (SAUS)

Methodology: Perpetual inventory model, where estimates of new additions and removals are added to the previous year's estimate of stock to update the current year"""


class CommercialIndicators:
    def __init__(self, ):
		self. =

    def load_data(self,):
        SEDS_CensusRgn = 
        mer_data23_May_2016 = 
        mer_data23_Jan_2017 = 
        mer_data23_Dec_2019 = 
        AER11_Table21C_Update = 
        CDD_by_Division18 = 
        HDD_by_Division18 =


    def estimate_fuel_electricity_consumption_regional(self,):
        """Data Source: EIA's State Energy Data System (SEDS)"""
        energy_consumtpion_data_regional = 
        approximate_intesity_time_series = 
        weather_adjustment_factors_regional = 
        energy_consumption_regional = 
        return None 


    def weather_factors(self, region, type='electricity', energy_type='delivered'):
        """Estimate a simple regression model to fit the regional intensity to a linear function of time (included squared and cubed values of time) and degree days. 
        -electricity model: constant term, heating degree day (HDD), cooling degree day (CDD), time, time-squared, and time-cubed
        -fuels model: contant term?, HDD, HDD*Time, Time, Time-squared and composite fuel price index (the composite fuel price index was developed as a weighted average of the national distillate
            fuel oil price index and a national average price for natural gas)
        """
        if type == 'electricity':
            X = 
        elif type == 'fuels':
            X = 
        if energy_type == 'delivered':
            weather_factor = (reported_electricity + fuels) / (weather_adjusted_electrity + weather_adjusted_fuels)
            return weather_factor
        else:
            reg = linear_model.LinearRegression()
            reg.fit(X, Y)
            coefficients = reg.coef_
            predicted_value_intensity_actualdd = reg.predict(X_actualdd)  # Predicted value of the intensity based on actual degree days
            predicted_value_intensity_ltaveragesdd = reg.predict(X_ltaveragesdd)  # Predicted value of the intensity based on the long-term averages of the degree days
            weather_factor = predicted_value_intensity_actualdd / predicted_value_intensity_ltaveragesdd 
            weather_normalized_intensity = actual_intensity / weather_factor
            return weather_factor, weather_normalized_intensity


    def estimate_regional_floorspace_share(self,):
    """assumed commercial floorspace in each region follows same trends as population or housing units"""


    def estimate_intensity_indexes_regional(self, parameter_list):
        """Data Sources: Fuel Consumption and electricity consumption from SEDS, Shares of regional floorspace from CBECs
        Purpose: used to produce weather adjustment facotrs"""
        total_national_floorspace = 

        reg = linear_model.LinearRegression()
        X = regional_housing_unit_share
        Y = regional_commercial_building_floorspace_shares  # from historical CBECS
        reg.fit(X, Y)
        coefficients = reg.coef_
        regional_shares_floorspace = dict()  # assumed commercial floorspace in each region follows same trends as population or housing units
        regional_floorspace = dict()
        regional_intensity_index = dict()
        for region in regions: 
            regional_energy_consumption =  # consumption of electricity and fuels
            predicted_share = reg.predict(annual_housing_unit_values)
            regional_shares_floorspace[region] = predicted_share
            region_floorspace =  predicted_share * total_national_floorspace
            regional_floorspace[region] = region_floorspace
            regional_intensity_index[region] = regional_energy_consumption / region_floorspace  
        return None


    def estimate_reclassification_electricity_sector_sales(self,):
        """Data Source: Commercial electricity sales from EIA SEDS
        Assumption: The significant changes (same magnitude but opposite directions in the same year) in state-level electricity slaes, typically showing up in the commercial and industrial secotrs
        reflect reclassification of some customers form the industrial rate class to the commercial rate class or vice versa. 
        Strategy: adjust the more recent data by adding or subtracting a constant vlaue, determined from the year in which the reclassification was judged to have occured."""
        state_level_adjustment = 
        pass


    def total_commercial_lmdi_utiladj(self,):
        """"""
        pass


    def commercial_total(self,):
        """"""
        pass


    def report_tables(self,):
        """"""
        pass


    def adjusted_supplier_data(self,):
        """""""
        pass


    def national_calibration(self,):
        """"""
        pass

    
    def regional_intensity_aggregate(self,):
        """""""
        pass
