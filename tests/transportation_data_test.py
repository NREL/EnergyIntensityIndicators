import pytest
import pandas as pd

from EnergyIntensityIndicators.transportation import TransportationIndicators

class TestTransportation:

    transportation = TransportationIndicators(directory= , output_directory= , 
                                              level_of_aggregation='All Transportation')
    data = pd.read_csv(f'{dir}/pnnl_output_data.csv')
    transportation_data = data[data['Sector'] == 'transportation']
    categories = transportation.categories_dict # level of aggregation

    pytest.mark.parametrize('loa', [categories.keys()])
    def test_collect_data(self, loa):
        assert None

    pytest.mark.parametrize('loa', [categories.keys()])
    def test_nesting(self, loa):
        assert None   

    pytest.mark.parametrize('loa', [categories.keys()])
    def test_aggregation(self, loa):
        self.transportation.get_nested_lmdi()
        assert None   

if __name__ == '__main__':
    pass