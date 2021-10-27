
import pandas as pd
import requests
import os
import numpy as np


class BEA_api:
    """class for calling gross output and quantity index values"""

    def __init__(self, years):
        self.base_url = 'https://apps.bea.gov/api/data/'
        apik = os.getenv("BEA_API_Key")
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
        historical_va_quant_index = pd.read_csv(
            './EnergyIntensityIndicators/Industry/Data/Chain_Type_Qty_Indexes_Value_Added_by_Industry.csv'
            ).set_index('Industry') # 2012 = 100 
        historical_va = pd.read_csv(
            './EnergyIntensityIndicators/Industry/Data/Historical_VA.csv'
            ).set_index('Industry') 

        historical_go = pd.read_csv(
            './EnergyIntensityIndicators/Industry/Data/Historical_GO.csv'
            ).set_index('Industry')  
        historical_go_quant_index = pd.read_csv(
            './EnergyIntensityIndicators/Industry/Data/historical_go_qty_index.csv'
            ).set_index('Industry')  # 2012 = 100 

        historical_data = {
            'historical_va': historical_va,
            'historical_va_quant_index': historical_va_quant_index,
            'historical_go': historical_go,
            'historical_go_quant_index': historical_go_quant_index
            }    
        return historical_data

    @staticmethod
    def laspeyres_quantity(nominal_data, quantity_index):
        """Calculate Laspeyres quantity"""
        index = quantity_index.shift().divide(quantity_index).multiply(nominal_data)
        laspeyres = index.sum(axis=1).divide(nominal_data.sum(axis=1))
        return laspeyres

    @staticmethod
    def pasche_quantity(nominal_data, quantity_index):
        """Calculate Pasche quantity"""
        index = (quantity_index.divide(quantity_index.shift())).multiply(nominal_data)
        pasche = nominal_data.sum(axis=1).divide(index.sum(axis=1))
        return pasche

    @staticmethod
    def adjust_transportation(qty_index, nominal_data):
        """Method to adjust transportation data in Industrial Sector"""
        nominal = nominal_data.loc[:, ['Motor Vehicles', 'Other Transportation Equipment']].multiply(1000)
        nominal_T = nominal.transpose()
        quantity_index = qty_index.loc[['Motor Vehicles', 'Other Transportation Equipment'], :]
        quantity_index_T = quantity_index.transpose()

        laspeyres_quantity = self.laspeyres_quantity(nominal_T, quantity_index_T)
        pasche_quantity = self.pasche_quantity(nominal_T, quantity_index_T)

        laspeyres_quantity['Chained_Laspeyres'] = np.sqrt(laspeyres_quantity.multiply(pasche_quantity))

        chained_laspeyres = laspeyres_quantity[['Chained_Laspeyres']].fillna(1)
        chained_laspeyres = chained_laspeyres.reindex(columns=chained_laspeyres.columns + ['Raw_Index', 'Index_2012=100'])
        for i in chained_laspeyres.index():
            if i == min(chained_laspeyres.index()):
                raw_index = 1
            else: 
                raw_index = chained_laspeyres.loc[i, ['Chained_Laspeyres']].multiply(chained_laspeyres.loc[i - 1, 'Raw_Index'])
            chained_laspeyres.loc[i, 'Raw_Index'] = raw_index
        
        chained_laspeyres['Index_2012=100'] = chained_laspeyres[['Raw_Index']].divide(chained_laspeyres.loc[2012, 'Raw_Index']).multiply(100)
        
        gross_output_T.loc[:, 'Total'] = gross_output_T.sum(axis=1)

        transportation_line = chained_laspeyres[['Raw_Index']].multiply(gross_output_T.loc[2012, 'Total'].values * 0.01)
        transportation_line = transportation_line.transpose()
        return transportation_line

    @staticmethod
    def merge_historical(api_data, historical):
        """Merge all historical data into one dataframe"""
        if len(api_data) == 1:
            api_data = api_data[0]
        else: 
            raise TypeError(f'list of go_nominal of len {len(api_data)}')
        

        api_data = api_data.set_index(['IndustrYDescription']).drop('Industry', axis=1)

        min_api_year = min([int(y) for y in api_data.columns])
        historical = historical.loc[:, : str(min_api_year - 1)]
        data = historical.merge(api_data, left_index=True, right_index=True, how='inner')
        data = data.transpose()
        data.index.name = 'Year'
        data.index =data.index.astype(int)
        # data = data.merge(label_to_naics, left_index=True, right_index=True, how='inner')
        return data
    
    def transform_data(self, nominal_historical, nominal_from_api, 
                       qty_index_historical, qty_index_from_api):
        """Format data"""

        nominal = self.merge_historical(nominal_from_api, nominal_historical)
        qty_index = self.merge_historical(qty_index_from_api, qty_index_historical)

        # nominal_12 = nominal[nominal["Year"] == 2012][['DataValue', 'IndustrYDescription', 'Industry']].transpose()
        cols = [q for q in qty_index.columns if q in nominal.columns]
        nominal_12  = nominal.loc[2012, cols]
        transformed_quant_index = qty_index.multiply(np.tile(nominal_12.values.transpose(), (len(qty_index), 1))).multiply(.01)

        # transformed_quant_index = transformed_quant_index.transpose()

        # transportation_qty_index = transformed_quant_index.loc[:, 'Transportation equipment']
        # transformed_quant_index.loc[:, 'Transportation equipment'] = self.adjust_transportation(transportation_qty_index, nominal)

        return transformed_quant_index

    def collect_va(self, historical_data):
        """Method to collect value added data"""
        va_nominal = self.get_data(table_name='va_nominal')
        historical_va = historical_data['historical_va'] 

        va_quant_index = self.get_data(table_name='va_quant_index')
        historical_va_qty_index = historical_data['historical_va_quant_index'] 
        
        transformed_va_quant_index = self.transform_data(historical_va, va_nominal, 
                                                         historical_va_qty_index, va_quant_index)

        return transformed_va_quant_index

    def collect_go(self, historical_data):
        """Method to collect gross output data"""
        go_nominal = self.get_data(table_name='go_nominal')
        historical_go = historical_data['historical_go'] 

        go_quant_index = self.get_data(table_name='go_quant_index')
        historical_go_qty_index = historical_data['historical_go_quant_index']

        transformed_go_quant_index = self.transform_data(historical_go,
                                                         go_nominal,
                                                         historical_go_qty_index,
                                                         go_quant_index)
        return transformed_go_quant_index
    

    def chain_qty_indexes(self):
        """Merge historical and api data, manipulate as in ChainQtyIndexes
        """
        historical_data = self.import_historical()

        go_quant_index = self.collect_va(historical_data)

        va_quant_index = self.collect_go(historical_data)

        return va_quant_index, go_quant_index


if __name__ == '__main__':
    data = BEA_api(years=list(range(1949, 2018))).get_data(table_name='go_nominal')

