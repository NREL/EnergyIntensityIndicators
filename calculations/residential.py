import pandas as pd
from sklearn import linear_model

class ResidentialIndicators:
    def __init__(self, ):
		self. =


    def load_data(self, ):
        AER11_table2_1b_update = 
        AnnualData_MER22_2015 = 
        AnnualData_MER22_2017 = 
        AnnualData_MER_22_Dec2019 = 
        RECS_intensity_data = 
        National_Calibration = 
        Weather_Factors = 
        CDD_by_Division18 = 
        HDD_by_Division18 = 


    def regional_time_series_floor_space(self, ):
        pass


    def estimate_fuel_electricity_consumption_regional(self, ):
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
        Weather factors are applied at the regional level to generate the weather-normalized intensity indexes for each of the four Census regions
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


    def estimate_floorspace_occupied_housing_units_regional(self, ):
        """Estimate regional housing and regional floorspace by housing type (single family, multifamily, manufactured homes)"""
        estimated_survival_curve =  # Estimate from vintage data over the 1999 through 2009 AHS surveys
        new_housing =  # From Characteristics of New Housing reports from the Census Bureau
        stock_adjustment_model = 
        estimated_occupied_housing_units =  # from stock adjustment level
        return estimated_occupied_housing_units


    def estimate_floorspace_housing_unit_size_national(self, housing_type='single_family'):
        """Single family and multi-family units use AHS data, combined with adjusted Characteristics of New Housing Data. Manufactured homes use RECS data"""
        if housing_type == 'manufactured_homes':
            size_estimates = 
        else: 
            average_size_post_1985 = 
            stock_units_pre_1985 = 
            stock_units_post_1985 =  # including 1985


    def estimate_floorspace_regional_shares_national_level_housing_units(self, ):
        """The regional shares for the non-AHS years are computed via a simple average of the preceding (odd) year and subsequent (odd) year.
        Data Source: AHS"""
        pass


    def estimate_final_floorspace_by_housing_type(self, ):
        """Data Source: AHS"""

        
    def residential_total_lmdi_utiladj(self, ):
        """"""
        pass


    def national(self, ):
        """"""
        pass


    def northeast(self, ):
        """"""
        pass


    def midwest(self, ):
        """"""
        pass


    def south(self, ):
        """"""
        pass


    def west(self, ):
        """"""
        pass


    def report_tables(self, ):
        """"""
        pass

    
    def report_graphs(self, ):
        """"""
        pass



