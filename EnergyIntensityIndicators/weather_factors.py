from sklearn import linear_model
import pandas as pd

class WeatherFactors:
    def __init__(self, region, energy_type, type):
        self.hdd_by_division = GetEIAData.eia_api(id_='1566347')
        self.cdd_by_division = GetEIAData.eia_api(id_='1566348')
        self.region = region
        self.energy_type = energy_type  # 'electricity' or 'fuels'
        self.type = type  # 'delivered' etc

    def adjust_data(self):
        adjustment_factor_electricity =  # Weights derived from 1995 CBECS
        adjustment_factor_fuels = 
        self.adjusted_hdd = weights * self.hdd_by_division
        self.adjusted_cdd = weights * self.cdd_by_division


    def weather_factors(self):
        """Estimate a simple regression model to fit the regional intensity to a linear function of time (included squared and cubed values of time) and degree days. 
        -electricity model: constant term, heating degree day (HDD), cooling degree day (CDD), time, time-squared, and time-cubed
        -fuels model: contant term?, HDD, HDD*Time, Time, Time-squared and composite fuel price index (the composite fuel price index was developed as a weighted average of the national distillate
            fuel oil price index and a national average price for natural gas)
        Weather factors are applied at the regional level to generate the weather-normalized intensity indexes for each of the four Census regions
        
        -The weather factors for delivered energy and source nergy are computed implicityl. For delivered energy, they are calculated
        as the sum of reported electricity and fuels divided by the sum of the weather-adjusted electricity and weather-adjusted fuels. 
        A similar procedure is followed for source energt. As such, the implied weather factors are a result of the process, not an independent
        variable that influences the values of intensity indexes for delivered energy and source energy. All of these computation occur within Commercial_Total worksheet.
        """
        if self.energy_type == 'electricity':
            X = 
        elif self.energy_type == 'fuels':
            X = 
        if self.type == 'delivered':
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

    def 

    