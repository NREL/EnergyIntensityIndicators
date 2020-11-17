
import pandas as pd
import requests
import os


class BEA_api:
    """class for calling gross output and quantity index values"""

    def __init__(self, years):
        self.base_url = 'https://apps.bea.gov/api/data/'
        apik  = os.getenv("BEA_API_Key")
        apik = apik.replace("'", "")
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
        api_params['Year'] = "All" #\
            #[str(y) for y in range(self.years[0], self.years[1]+1)] # 

        for k, v in params.items():
            api_params[k] = v

        r = requests.get(self.base_url, params=api_params)
        print('r:\n', type(r.json()['BEAAPI']['Results']), type(r.json()['BEAAPI']['Results'][0]))

        try:
            data_json = r.json()['BEAAPI']['Results']
            data = [pd.DataFrame.from_records(
                data_json[i]['Data']
                ) for i in range(len(data_json))]

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
        print('data: \n', data)
        print('data: \n', data[0])
        print('data: \n', type(data[0]))


        if data:
            data = [self.format_data(data[i]) for i in range(len(data))]
            return data

        else:
            return

    def import_historical(self):
        """Method for importing historical BEA data, saved as csv
        Footnotes: 1. Consists of agriculture, forestry, fishing, and hunting; mining; construction; and manufacturing.
        2. Consists of utilities; wholesale trade; retail trade; transportation and warehousing; information; finance, insurance, real estate, rental, and leasing; professional and business services; educational services, health care, and social assistance; arts, entertainment, recreation, accommodation, and food services; and other services, except government.
        3. Consists of computer and electronic product manufacturing (excluding navigational, measuring, electromedical, and control instruments manufacturing); software publishers; broadcasting and telecommunications; data processing, hosting and related services; internet publishing and broadcasting and web search portals; and computer systems design and related services.
        """
        historical_va_quant_index = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/Chain_Type_Qty_Indexes_Value_Added_by_Industry.csv').set_index('Industry') # 2012 = 100 
        historical_va = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/Historical_VA.csv').set_index('Industry') 
        
        historical_go = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/Historical_GO.csv').set_index('Industry')  
        historical_go_quant_index = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/historical_go_qty_index.csv').set_index('Industry') # 2012 = 100 
        
        historical_data = {'historical_va': historical_va, 'historical_va_quant_index': historical_va_quant_index,
                           'historical_go': historical_go, 'historical_go_quant_index': historical_go_quant_index}    
        return historical_data

    def chain_qty_indexes(self):
        """Merge historical and api data, manipulate as in ChainQtyIndexes
        """
        historical_data = self.import_historical()
        print('historical_data: \n', historical_data)


        go_nominal = self.get_data(table_name='go_nominal')
        if len(go_nominal) == 1:
            go_nominal = go_nominal[0]
            print('go_nominal: \n', go_nominal)
        else: 
            raise TypeError(f'list of go_nominal of len {len(go_nominal)}')

        print('len go_nominal: \n', len(go_nominal))
        historical_go = historical_data['historical_go'] 
        go_nominal_12 = go_nominal[go_nominal["Year"] == 2012][['DataValue', 'IndustrYDescription', 'Industry']].transpose()
        print('go_nominal_12: \n', go_nominal_12)
        go_quant_index = self.get_data(table_name='go_quant_index')
        min_api_year = min([int(y) for y in go_quant_index.columns])

        historical_go_qty_index = historical_data['historical_go_qty_index']
        historical_go_qty_index = historical_go_qty_index.loc[:, : min_api_year]

        go_quant_index = go_quant_index.merge(historical_go_qty_index, left_index=True, right_index=True, how='outer')
        print('go_quant_index: \n', go_quant_index)

        transformed_go_quant_index = go_quant_index.multiply(go_nominal_12, index=1).multiply(0.01)
        # transformed_go_quant_index.loc['          Transportation equipment', :] = 

        transformed_go_quant_index = transformed_go_quant_index.transpose()

        # transformed_go_quant_index is further manipulated but I'm not sure exactly how it works
        transformed_go_quant_index = transformed_go_quant_index.divide(transformed_go_quant_index.loc[self.base_year, :], axis=0)


    
        va_nominal = self.get_data(table_name='va_nominal')
        print('len va_nominal: \n', len(va_nominal))

        historical_va = historical_data['historical_va'] 
        va_nominal_12 = va_nominal[va_nominal["Year"] == 2012][['DataValue', 'IndustrYDescription', 'Industry']].transpose()
        print('va_nominal_12:\n', va_nominal_12)
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

