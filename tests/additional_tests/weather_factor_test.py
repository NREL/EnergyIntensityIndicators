import pandas as pd
from sklearn import linear_model
import os


from EnergyIntensityIndicators.weather_factors import WeatherFactors

class TestWeatherFactors:

    pnnl_directory = './tests/Indicators_Spreadsheets_2020'
    output_directory = './Results'
    acceptable_pct_difference = 0.05
    sector = 'commercial' # also for residential
    utils = TestingUtilities(sector, acceptable_pct_difference)

    # directory = 
    activity_data = None
    residential_floorspace = None
    nominal_energy_intensity = None
    end_year = 2018
    projections = False

    weather = WeatherFactors(sector=sector, directory=directory, 
                             activity_data=activity_data, 
                             residential_floorspace=residential_floorspace, 
                             nominal_energy_intensity=nominal_energy_intensity,
                             end_year=end_year, projections=projections)

    def test_adjust_data():

    def test_process_prices():

    def test_cbecs_1995_shares():

    def test_recs_1993_shares():
    
    def test_regional_shares():

    def test_gather_weights_data():

    def test_heating_cooling_degree_days():

    def test_heating_cooling_data():

    def test_estimate_regional_shares():

    def test_commercial_estimate_regional_floorspace():

    def test_commercial_regional_intensity_aggregate():

    def test_residential_regional_intensity_aggregate():

    def test_weather_factors():

    def test_national_method1_fixed_end_use_share_weights():

    def test_national_method2_regression_models():

    def test_adjust_for_weather():

    def test_get_weather():