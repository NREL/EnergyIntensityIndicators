
import pandas as pd
import requests
import os


class BEA_api:
    """class for calling gross output and quantity index values"""

    def __init__(self, years):
        self.base_url = 'https://apps.bea.gov/api/data/'
        apik  = os.getenv("BEA_API_Key")
        self.base_params = {'UserID': apik, 'method': 'GetData',  # apik is API key
                            'Industry': 'ALL', 'Frequency': 'A',
                            'Format': 'json'}
        self.tableID_dict = {'go_quant_index': 16, 'go_nominal': 15,
                             'va_nominal': 1, 'va_quant_index': 8}
        self.years = years

    def call_api(self, params):
        """
        Method for calling BEA api for given range of years [min, max]

        Parameters
        ----------
        years : list
            List of begnning year and ending year of data to retrieve from
            BEA API.

        params : dict
            Dictionary of DataSetName and tableID.

        Returns                                                                                                                                     
        -------
        data : dataframe
            Dataframe of results. Will print API error, if applicable.
        """

        api_params = self.base_params.copy()
        api_params['Year'] = \
            [str(y)+',' for y in range(self.years[0], self.years[1]+1)]

        for k, v in params.items():
            api_params[k] = v

        r = requests.get(self.base_url, params=api_params)

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

    def get_data(self, table_name):
        """
        Get data using the BEA API call.

        Parameters
        ----------
        years : list
            List of begnning year and ending year of data to retrieve from
            BEA API.

        table_name : str
            go_quant_index (Industry Gross Output, quantity indexes),
            go_nominal (Industry Gross Output, nominal),
            va_nominal (Industry Value Added, nominal),
            va_quant_index (Industry Value Added, quanitity indexes).
            Additional values can be added with their corresponding tableID
            number to tableID_dict

        Returns
        -------
        data : dataframe
            Dataframe of BEA data.

        """

        api_params = {'DataSetName': 'GDPbyIndustry',
                      'tableID': self.tableID_dict[table_name]}

        data = self.call_api(params=api_params)

        if data:
            data = self.format_data(data)
            return data

        else:
            return

    def import_historical(self):
        """Method for importing historical BEA data, saved as csv"""
        historical_va_quant_index = pd.read_csv('./Data/Chain_Type_Qty_Indexes_Value_Added_by_Industry.csv') # 2012 = 100 
        historical_va = pd.read_csv('./Data/Historical_VA.csv') 
        historical_data = {'historical_va': historical_va, 'historical_va_quant_index': historical_va_quant_index}    
        return historical_data

    def chain_qty_indexes(self):
        """Merge historical and api data, manipulate as in ChainQtyIndexes
        """
        historical_data = self.import_historical()

        go_nominal = self.get_data(table_name='go_nominal')
        historical_go = historical_data['historical_go'] 
        go_nominal_12 = go_nominal[2012]

        go_quant_index = self.get_data(table_name='go_quant_index')
        historical_go_qty_index = historical_data['historical_go_qty_index']
        go_quant_index = go_quant_index.merge(historical_go_qty_index, left_index=True, right_index=True, how='outer')


        transformed_go_quant_index = go_quant_index.multiply(go_nominal_12, index=1).multiply(0.01)
        # transformed_go_quant_index.loc['          Transportation equipment', :] = 

        transformed_go_quant_index = transformed_go_quant_index.transpose()

        # transformed_go_quant_index is further manipulated but I'm not sure exactly how it works
        transformed_go_quant_index = transformed_go_quant_index.divide(transformed_go_quant_index.loc[self.base_year, :], axis=0)


    
        va_nominal = self.get_data(table_name='va_nominal')
        historical_va = historical_data['historical_va'] 
        va_nominal_12 = va_nominal[2012]

        va_quant_index = self.get_data(table_name='va_quant_index')
        historical_va_qty_index = historical_data['historical_va_qty_index'] 
        va_quant_index = va_quant_index.merge(historical_va_qty_index, left_index=True, right_index=True, how='outer')

        transformed_va_quant_index = va_quant_index.multiply(va_nominal_12, index=1).multiply(.01)
        transformed_va_quant_index = transformed_va_quant_index.transpose()
        # transformed_va_quant_index is further manipulated but I'm not sure exactly how it works

        go_over_va = transformed_go_quant_index.divide(transformed_va_quant_index)
        return transformed_va_quant_index, transformed_go_quant_index


if __name__ == '__main__':
    data = BEA_api(years=list(range(1949, 2018))).get_data(table_name='go_nominal')
    print(data)

