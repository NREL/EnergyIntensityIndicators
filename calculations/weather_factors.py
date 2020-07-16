from sklearn import linear_model
import pandas as pd

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