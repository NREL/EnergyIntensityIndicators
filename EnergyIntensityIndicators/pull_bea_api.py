
import pandas as pd
import requests


class BEA_api:
    """class for calling gross output and quantity index values"""

    def __init__(self):
        self.base_url = 'https://apps.bea.gov/api/data/'
        self.base_params = {'UserID': apik, 'method': 'GetData',  # apik is API key
                            'Industry': 'ALL', 'Frequency': 'A',
                            'Format': 'json'}

    def call_api(self, years, params):
        """ Method for calling BEA api for given range of years [min, max]"""

        r = requests.get(self.base_url, params=params)

        try:
            data = pd.DataFrame.from_records(
                r.json()['BEAAPI']['Results']['Data']
                )

        except KeyError:
            print('API Error:{}'.format(
                r.json()['BEAAPI']['Error']['ErrorDetail']['Description']
                ))

        else:
            return data

    def format_data(self, data):
        """format and pivot called data"""

        data['DataValue'] = data.DataValue.astype(float)

        pivoted_data = data.pivot_table(
            index=['IndustrYDescription', 'Industry'], values='DataValue',
            columns='Year'
            )

        pivoted_data.reset_index(inplace=True)

        return pivoted_data

    def get_gross_output(self, years):
        """ API call for gross output"""

        api_params = self.base_params.copy()
        api_params['Year'] = \
            [str(y)+',' for y in range(years[0], years[1]+1)]
        api_params['DataSetName'] = 'GDPbyIndustry'
        api_params['tableID'] = 1

        data = self.call_api(years, params=api_params)

        if data:
            data = self.format_data(data)
            return data

        else:
            return

    def get_quantity_indexes(self, years):
        """ API call for gross output quantity indexes"""

        api_params = self.base_params.copy()
        api_params['Year'] = \
            [str(y)+',' for y in range(years[0], years[1]+1)]
        api_params['DataSetName'] = 'GDPbyIndustry'
        api_params['tableID'] = 16

        data = self.call_api(years, params=api_params)

        if data:
            data = self.format_data(data)
            return data

        else:
            return
