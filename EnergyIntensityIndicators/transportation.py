import pandas as pd
from sklearn import linear_model

class TransportationIndicators(LMDI):

    def __init__(self, energy_data, activity_data, categories_list):
        super().__init__(energy_data, activity_data, categories_list)
        self.sub_categories_list = categories_list['transportation']

    def load_data(self, parameter_list):
        mer_table25_dec_2019 = GetEIAData.eia_api(id_='711272') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711272'
        mer_table_43_old = GetEIAData.eia_api(id_='711272') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711272'
        mer_table_4.3_nov2019 = GetEIAData.eia_api(id_='711272') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711272'
        aer_2010_table_65 = GetEIAData.eia_api(id_='711272') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711272'

        pass

    def water_freight_regression(self, ):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        X =
        Y = 
        reg = linear_model.LinearRegression()
        reg.fit(X, Y)
        coefficients = reg.coef_
        predicted_value = reg.predict(X_test)  # Predicted value of the intensity based on actual degree days

    def detailed_data_school_buses(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def detailed_data_intercity_buses(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def fuel_heat_content(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def fuel_consump(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def adjust_truck_freight(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def freight_based_energy_use(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def freight_based_activity(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def passenger_based_energy_use(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def passenger_based_activity(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def compare_aggregates(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def mpg_check(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass


