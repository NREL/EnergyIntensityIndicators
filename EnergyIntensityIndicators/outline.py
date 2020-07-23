import pandas as pd
import numpy as np
from sklearn import linear_model

class LMDI:

    sectors = {'residential': {'Northeast': {'Single-Family', 'Multi-Family', 'Manufactured Homes'}, 
                               'Midwest': {'Single-Family', 'Multi-Family', 'Manufactured Homes'},
                               'South': {'Single-Family', 'Multi-Family', 'Manufactured Homes'},
                               'West': {'Single-Family', 'Multi-Family', 'Manufactured Homes'}},
              'industrial': {'Manufacturing': {'Food, Beverages, & Tobacco', 'Textile Mills and Products', 
                                               'Apparel & Leather', 'Wood Products', 'Paper',
                                               'Printing & Allied Support', 'Petroleum & Coal Products', 'Chemicals',
                                               'Plastics & Rubber Products', 'Nonmetallic Mineral Products', 'Primary Metals',
                                               'Fabricated Metal Products', 'Machinery', 'Computer & Electronic Products',
                                               'Electical Equip. & Appliances', 'Transportation Equipment',
                                               'Furniture & Related Products', 'Miscellaneous'},
                             'Nonmanufacturing': {'Agriculture, Forestry & Fishing', 'Mining', 'Construction'}}, 
              'commercial': {'Commercial_Total', 'Total_Commercial_LMDI_UtilAdj'}, 
              'transportation': {'All_Passenger':
                                    {'Highway': 
                                        {'Passenger Cars and Trucks': 
                                            {'Passenger Car – SWB Vehicles', 'Light Trucks – LWB Vehicles', 'Motorcycles'}, 
                                        'Buses': 
                                            {'Urban Bus', 'Intercity Bus', 'School Bus'}, 
                                        'Paratransit', 
                                        'Personal vehicles-aggregate': 
                                            {'Passenger Car', 'Light Truck', 'Motorcycle'}}, 
                                    'Rail': 
                                        {'Urban Rail': 
                                            {'Commuter Rail', 'Heavy Rail', 'Light Rail'}, 
                                        'Intercity Rail'}, 
                                    'Air': {'Commercial Carriers', 'General Aviation'}}, 
                                'All Freight': 
                                    {'Highway': 
                                        {'Freight-Trucks': 
                                            {'Single-Unit Truck', 'Combination Truck'}}, 
                                    'Rail', 
                                    'Air', 
                                    'Waterborne',
                                    'Pipeline': 
                                        {'Oil Pipeline', 'Natural Gas Pipeline'}}}, 
              'electricity': {'Elec Generation Total': 
                                {'Elec Power Sector': 
                                    {'Electricity Only':
                                        {'Fossil Fuels': 
                                            {'Coal', 'Petroleum', 'Natural Gas', 'Other Gasses'},
                                         'Nuclear', 
                                         'Hydro Electric', 
                                         'Renewable':
                                            {'Wood', 'Waste', 'Geothermal', 'Solar', 'Wind'}},
                                     'Combined Heat & Power': 
                                        {'Fossil Fuels'
                                            {'Coal', 'Petroleum', 'Natural Gas', 'Other Gasses'},
                                         'Renewable':
                                            {'Wood', 'Waste'}}}, 
                                'Commercial Sector', 
                                'Industrial Sector'},
                              'All CHP':
                                {'Elec Power Sector': 
                                    {'Combined Heat & Power':
                                        {'Fossil Fuels':
                                            {'Coal', 'Petroleum', 'Natural Gas', 'Other Gasses'},
                                        'Renewable':
                                            {'Wood', 'Waste'},
                                        'Other'}},
                                    
                                'Commercial Sector':
                                    {'Combined Heat & Power':
                                        {'Fossil Fuels':
                                            {'Coal', 'Petroleum', 'Natural Gas', 'Other Gasses'},
                                        'Hydroelectric',
                                        'Renewable':
                                            {'Wood', 'Waste'},
                                        'Other'}},, 
                                'Industrial Sector':
                                    {'Combined Heat & Power':
                                        {'Fossil Fuels':
                                            {'Coal', 'Petroleum', 'Natural Gas', 'Other Gasses'},
                                        'Hydroelectric',
                                        'Renewable':
                                            {'Wood', 'Waste'},
                                        'Other'}}}}}

    """Base class for LMDI"""
    index_base_year_primary = 1985
	index_base_year_secondary = 1996  # not used
	charts_starting_year = 1985
	charts_ending_year = 2003 

	def __init__(self, df, categories_list):
        """
        Parameters
        ----------
        df: dataframe
        description
        categories_list: list
        description
        """
		self.dataset = df 
        self.categories_list = categories_list

    def load_energy_data():
        pass
    
    @staticmethod
    def calculate_energy_shares(dataset, categories_list):
        """"sum row, calculate each type of energy as percentage of total
        Parameters
        ----------
        dataset: dataframe
            energy data
        
        Returns
        -------
        energy_shares: dataframe
            contains shares of each energy category relative to total energy 
        """
        energy_consumption_total = dataset[[categories_list]].sum(axis=1, skipna=True)
        # for category in categories_list:
        #     dataset[f'{category}_energy_shares'] = dataset.apply(lamba x: x[category] / x['Total'])

        energy_shares = dataset.divide(energy_consumption_total)
        return energy_shares

    @ staticmethod
    def calculate_log_mean_weights(dataset, categories_list):
        """purpose
           Parameters
           ----------
           dataset: dataframe
                Description
            categories_list: list
                Description
                
           Returns
           -------

        """
        for i in self.categories_list:
            for i in self.dataset.index: 
                if i >= 1:
                    self.dataset.loc[i, f'{category}_log_mean_divisia_weights'] = self.dataset.loc[i, f'{category}_energy_shares'] - self.dataset.loc[i-1, f'{category}_energy_shares'] / (self.dataset.loc[i, f'{category}_energy_shares'] / (math.log(self.dataset.loc[i-1, f'{category}_energy_shares'])))
                else:
                    self.dataset.loc[i, f'{category}_log_mean_divisia_weights'] = 0
        self.dataset['Log-Mean Divisia Weights Total'] = self.dataset[[self.categories_list]].sum(axis=0, skipna=True)
        return self.dataset


    @staticmethod
    def calculate_log_mean_weights_normalized(dataset, categories_list):
        """Normalize the log-mean divsia weights by the total log-mean divisia for each year 
           Parameters
           ----------
           
           Returns
           -------
           log_mean_divisia_weights_normalized: dataframe
                description
        """
        # for category in self.categories_list:
        #     for i in self.dataset.index:
        #         if i >= 1:
        #             self.dataset.loc[i, f'{category}_log_mean_divisia_weights_normalized'] = self.dataset.loc[i, f'{category}_log_mean_divisia_weights'] / 


        log_mean_divisia_weights_total = dataset[[categories_list]].sum(axis=1, skipna=True)
        log_mean_divisia_weights_normalized = dataset.divide(log_mean_divisia_weights_total)
        return log_mean_divisia_weights_normalized

    ##################################################
    def load_activity_data(self):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    @staticmethod
    def calculate_activity_shares(dataset, categories_list):
        """sum row, calculate each as percentage of total
           Parameters
           ----------
           dataset: dataframe 
                description
            categories_list: list
           Returns
           -------
           activity_shares: dataframe
                Description
        """
        # self.dataset['Activity_Total'] = self.dataset[[self.categories_list]].sum(axis=0, skipna=True)
        # for category in self.categories_list:
        #     self.dataset[f'{category}_energy_shares'] = self.dataset.apply(lamba x: x[category] / x['Total'])
        # return self.dataset

        activity_total = dataset[[categories_list]].sum(axis=1, skipna=True)
        activity_shares = dataset.divide(activity_total)
        return activity_shares
        
    @staticmethod
    def calculate_log_changes_activity_shares(dataset, categories_list):
        """purpose
           Parameters
           ----------
           df_name: str

           df: dataframe
           Returns
           -------
           log_changes: dataframe
                description
        """
        # for i in df.index: 
        #     for category in self.categories_list:
        #         df.at[i, f'{categories_list}_df_name_activities_share'] = df.loc
        
        change = dataset[['categories_list']].diff()
        log_ratio = np.log(dataset[['categories_list']] / dataset[['categories_list']].shift())
        log_changes = change.divide(log_ratio)
        return log_changes

    @staticmethod
    def compute_structure_index(log_mean_divisia_weights, log_changes_activity_shares, categories_list):
        """purpose
           Parameters
           ----------
           log_mean_divisia_weights: dataframe
                log-mean divisia weights normalized

            log_changes_activity_shares: dataframe

            categories_list: list

           Returns
           -------
           
        """
        index_chg = (log_mean_divisia_weights.multiply(log_changes_activity_shares)).sum(axis=1)
        index = (index * index_chg.shift()).ffill()  # first value should be set to 1? 
        divide_by_this = 
        index_normalized = index / divide_by_this # 1985=1

    ##################################################

    @staticmethod
    def adjust_for_weather(data, weather_factors):
        """purpose
           Parameters
           ----------
           data: dataframe
                dataset to adjust by weather
            weather_factors: array?
                description
            Returns
            -------
            weather_adjusted_data: dataframe ? 
        """
        weather_adjusted_data = data / weather_factors
        return weather_adjusted_data

    @staticmethod
    def calculate_energy_intensity_nominal(base_year, energy_consumption, activity, adjustment_factor=1):
        """purpose
           Parameters
           ----------
           base_year: int?

           energy_consumption: dataframe
                Energy Consumption (Trillion Btu)
            activity: dataframe
                Unit of activity data depends on sector
            adjustment_factor: int
                Depending on sector and subsector, used to adjust unit of activity data
           Returns
           -------
           Nominal energy intensity
        """

        energy_intensity_nominal = energy_consumption.divide((activity * adjustment_factor))
        return energy_intensity_nominal

    def calculate_log_changes_intensity(self, ):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def compute_intensity_index(self, ):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    ##################################################
    def activity_index(self, ):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def index_of_aggregate_intensity(self, ):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def structure_index(self, ):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def component_intensity_index(self, ):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def report_tables(self, ):
        """Create tables for report
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def report_graphs(self, ):
        """Create graphs for report
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def data_visualization(self,):
        """Format data for proper visualization
           Parameters
           ----------
           
           Returns
           -------
           
        """

    class LMDIMultiplicative:

    class LMDIAdditive:


    