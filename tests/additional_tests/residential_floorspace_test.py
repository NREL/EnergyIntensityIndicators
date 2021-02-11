import pandas as pd
import os

from EnergyIntensityIndicators.Residential.residential_floorspace import ResidentialFloorspace
from tests.utilities import TestingUtilities


class TestResidentialFloorspace:

    pnnl_directory = './tests/Indicators_Spreadsheets_2020'
    output_directory = './Results'
    acceptable_pct_difference = 0.05
    sector = 'residential' # also for commercial
    utils = TestingUtilities(sector, acceptable_pct_difference)

    end_year = 2018
    floorspace = ResidentialFloorspace(end_year=end_year)

    def test_update_ahs_data():

    def test_get_ahs_tables():

    def test_get_percent_remaining_surviving():

    def test_interpolate_with_avg():

    def test_get_place_nsa_all():

    def test_housing_stock_model():

    def test_model_average_housing_unit_size_sf():

    def test_get_housing_size_sf():

    def test_get_housing_size_mf():

    def test_residuals_avg_size_mf():

    def test_average_housing_unit_size_mf():

    def test_model_average_housing_unit_size_mh():

    def test_get_housing_stock_sf():

    def test_get_housing_stock_mf():

    def test_get_housing_stock_mh():

    def test_get_housing_stock():

    def test_final_floorspace_estimates():

    