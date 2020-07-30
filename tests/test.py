import pytest
from ElectricityIntensityIndicators.outline import LMDI
from ElectricityIntensityIndicators.residential.ahs import * 

class TestClass:

    def func(x):
        return x + 1




    def test_answer():
        assert func(3) == 5

    @pytest.mark.parametrize()
    def test_select_value():
        excel_value_pnnl = pd.read_excel('./')
        value = LMDI.select_value(dataframe, base_row, base_column)
        assert excel_value_pnnl == value

    @pytest.mark.parametrize("region", ['Northeast', 
    'Midwest', 'South', 'West'])
    def test_calculate_shares(region):
        excel_shares_pnnl = pd.read_excel('./')
        shares = LMDI.calculate_shares(dataset, categories_list)
        assert shares == excel_shares_pnnl

    @pytest.mark.parametrize()
    def test_log_changes():
        excel_log_changes_pnnl = pd.read_excel('./')
        log_changes = LMDI.calculate_log_changes(dataset)
        assert excel_log_changes_pnnl == log_changes

    @pytest.mark.parametrize()
    def test_index():
        excel_index_pnnl = pd.read_excel('./')
        index_chg, index, index_normalized = LMDI.compute_index(log_mean_divisia_weights, log_changes_activity_shares, categories_list)
        assert excel_index_pnnl == index

    @pytest.mark.parametrize()
    def test_log_changes_activity_shares():
        excel_log_changes_activity_shares_pnnl = pd.read_excel('./')
        log_changes_activity_shares = LMDI.calculate_log_changes_activity_shares(dataset, categories_list)
        assert excel_log_changes_activity_shares_pnnl == log_changes_activity_shares

    @pytest.mark.parametrize()
    def test_log_mean_weights():
        excel_log_mean_weights_pnnl = pd.read_excel('./')
        excel_log_mean_weights_normalize_pnnl = pd.read_excel('./')
        log_mean_divisia_weights, log_mean_divisia_weights_normalized = LMDI.calculate_log_mean_weights(dataset, categories_list)
        assert excel_log_mean_weights_pnnl == log_mean_divisia_weights & excel_log_mean_weights_normalize_pnnl == log_mean_divisia_weights_normalized

    @pytest.mark.parametrize()
    def test_adjust_for_weather():
        excel_weather_adjustment_pnnl = pd.read_excel('./')
        weather_adjusted_data = adjust_for_weather(data, weather_factors)
        assert excel_weather_adjustment_pnnl == weather_adjusted_data
    
    @pytest.mark.parametrize()
    def test_energy_intensity_nominal():
        excel_energy_intensity_nominal_pnnl = pd.read_excel('./')
        energy_intensity_nominal = calculate_energy_intensity_nominal(base_year, energy_consumption, activity, adjustment_factor=1)
        assert excel_energy_intensity_nominal_pnnl == energy_intensity_nominal
    
    @pytest.mark.parametrize()
    def test_activity_index():
        assert
    
    @pytest.mark.parametrize()
    def test_index_aggregate_intensity():
        assert
    
    @pytest.mark.parametrize()
    def test_lmdi():
        assert