import pandas as pd
from sklearn import linear_model


class ElectricityIndicators(LMDI):

    def __init__(self, energy_data, activity_data, categories_list):
        super().__init__(energy_data, activity_data, categories_list)
        self.sub_categories_list = categories_list['electricity']

    def load_data():
        Table21f10 = GetEIAData.eia_api(id_='711254') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711254'
        Table21f11 = GetEIAData.eia_api(id_='711254') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711254'
        Table82a10 = None  # blank ?
        Table82a11 = GetEIAData.eia_api(id_='3') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=3'
        Table82b10 = GetEIAData.eia_api(id_='21')  #  'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=21'
        Table82b11 = GetEIAData.eia_api(id_='21')  # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=21'
        Table82c11 = GetEIAData.eia_api(id_='1736765') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=1736765' ?
        Table82d10 = GetEIAData.eia_api(id_='711282') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711282'
        Table82d11 = GetEIAData.eia_api(id_='711282') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711282'
        Table82d11_2012_and_later = GetEIAData.eia_api(id_='1017') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=1017'
        Table83d_03 = GetEIAData.eia_api(id_='711284') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711284' ?
        Table84b10 = GetEIAData.eia_api(id_='711284') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711284' ?
        Table84b11 = GetEIAData.eia_api(id_='711284') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711284' ?
        Table84c11_industrial = GetEIAData.eia_api(id_='456') 
        Table84c11_commercial = GetEIAData.eia_api(id_='453') 
        Table85c10 = GetEIAData.eia_api(id_='379')  # ?
        Table84c10_industrial = GetEIAData.eia_api(id_='456') 
        Table84c10_commercial = GetEIAData.eia_api(id_='453') 
        Table85c11 = GetEIAData.eia_api(id_='379')  # ?
        Table86b09 = GetEIAData.eia_api(id_='463')  # ?
        Table86b10 = GetEIAData.eia_api(id_='463') # ?
        Table86b11 = GetEIAData.eia_api(id_='463') # ?
        MER_T72b_1013_AnnualData = GetEIAData.eia_api(id_='21')  #'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=21'
        MER_T72c_1013_AnnualData = GetEIAData.eia_api(id_='2') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=2'
        
   

    def reconcile():
        """Does this need to be done or can download Btu directly instead of converting physical units to Btu
        """        

