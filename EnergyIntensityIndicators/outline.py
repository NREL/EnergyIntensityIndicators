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
                                        'Paratransit':
                                            {}, 
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

	def __init__(self, energy_data, activity_data, categories_list):
        """
        Parameters
        ----------
        energy_data: dataframe
            description
        activity_data: dataframe
            description
        categories_list: list
            description
        """
		self.energy_data = energy_data
        self.activity_data = activity_data 
        self.categories_list = categories_list
    
    @staticmethod
    def select_value(dataframe, base_row, base_column):
        return dataframe.iloc[base_row, base_column].values()
        
    @staticmethod
    def calculate_shares(dataset, categories_list):
        """"sum row, calculate each type of energy as percentage of total
        Parameters
        ----------
        dataset: dataframe
            energy data
        
        Returns
        -------
        shares: dataframe
            contains shares of each energy category relative to total energy 
        """
        consumption_total = dataset[[categories_list]].sum(axis=1, skipna=True)
        shares = dataset.divide(consumption_total)
        return shares

    @staticmethod
    def calculate_log_changes(dataset):
        """Calculate the log changes to intensity
           Parameters
           ----------
           dataset: dataframe

           Returns
           -------
           log_ratio: dataframe

        """
        log_ratio = np.log(dataset[['categories_list']].divide(dataset[['categories_list']].shift()))

        return log_ratio

    @staticmethod
    def compute_index(log_mean_divisia_weights, log_changes_activity_shares, categories_list):
        """[summary]

        Args:
            log_mean_divisia_weights ([type]): [description]
            log_changes_activity_shares ([type]): [description]
            categories_list ([type]): [description]

        Returns:
            [type]: [description]
        """                     
        index_chg = (log_mean_divisia_weights.multiply(log_changes_activity_shares)).sum(axis=1)
        index = (index_chg * index_chg.shift()).ffill().fillna(1)  # first value should be set to 1? 
        index_normalized = index / select_value(dataframe=, base_row=base_row, base_column=1) # 1985=1

        return index_chg, index, index_normalized 

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
        change = dataset[['categories_list']].diff()
        log_ratio = np.log(dataset[['categories_list']] / dataset[['categories_list']].shift())
        log_changes = change.divide(log_ratio)
        return log_changes
    
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

        change = dataset[['categories_list']].diff()
        log_ratio = np.log(dataset[['categories_list']] / dataset[['categories_list']].shift())
        log_mean_divisia_weights = change.divide(log_ratio)
        log_mean_divisia_weights_total = dataset[[categories_list]].sum(axis=1, skipna=True)
        log_mean_divisia_weights_normalized = log_mean_divisia_weights.divide(log_mean_divisia_weights_total)

        return log_mean_divisia_weights, log_mean_divisia_weights_normalized

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






    ##################################################
    
    # Energy
    energy_shares = calculate_shares(energy_data, categories)
    log_mean_divisia_weights_energy, log_mean_divisia_weights_normalized_energy = calculate_log_mean_weights(energy_shares, categories)
    log_changes_intensity = calculate_log_changes(energy_intensity_index)
    index_chg_energy, index_energy, index_normalized_energy = compute_index(log_mean_divisia_weights_normalized_energy, log_changes_intensity, categories)

    # Activity
    activity_shares = calculate_shares(activity_data, categories)
    log_changes_activity_shares = calculate_log_changes(activity_shares)
    index_chg_activity, index_activity, index_normalized_activity = compute_index(log_mean_divisia_weights_normalized_energy, log_changes_activity_shares, categories)    


    ##################################################

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

    def calculate_lmdi(self):
        """[summary]
        """        
        self.
        intensity_index = compute_index()
        structure_index = compute_index(log_mean_divisia_weights, log_changes_activity_shares, categories_list)
        component_intensity_index = 
        return
























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


    