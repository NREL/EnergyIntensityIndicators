import pandas as pd
import numpy as np
from sklearn import linear_model

class LMDI:
    """Base class for LMDI"""

	def __init__(self, categories_list, energy_data, activity_data, energy_types, base_year=1985, base_year_secondary=1996, charts_ending_year=2003):
        """
        Parameters
        ----------
        energy_data: dictionary of dataframes
            Energy input data, keys are the energy_type
        activity_data: dataframe
            Activity input data
        categories_list: list
            Sector or subsector categories over which to calculate LMDI
        """
		self.energy_data = energy_data
        self.activity_data = activity_data 
        self.categories_list = categories_list
        self.index_base_year_primary = base_year
        self.index_base_year_secondary = base_year_secondary  # not used
        self.charts_starting_year = base_year
        self.charts_ending_year = charts_ending_year
        self.energy_types = 
        
    def get_elec(self):
        delivered_electricity = self.energy_data['elec']
        delivered_electricity = delivered_electricity.set_index('year')
        delivered_electricity['Total'] = delivered_electricity.sum(axis=1)
        delivered_electricity['Energy_Type'] = 'Electricity'
        return delivered_electricity

    def get_fuels(self):
        fuels = self.energy_data['fuels']
        fuels = fuels.set_index('year')
        fuels['Total'] = fuels.sum(axis=1)
        fuels['Energy_Type'] = 'Fuels'
        return fuels

    def get_deliv(self, delivered_electricity, fuels):
        delivered = delivered_electricity.add(fuels)
        delivered['Energy_Type'] = 'Delivered'
        return delivered

    def get_source(self, delivered_electricity):
        conversion_factors = GetEIAData(self.sector).conversion_factors()

        source_electricity = delivered_electricity.multiply(conversion_factors) # Column A
        total_source = source_electricity.add(fuels)     
        total_source['Energy_Type'] = 'Source'
    
    def get_source_adj(self, delivered_electricity):
        conversion_factors = GetEIAData(self.sector).conversion_factors(include_utility_sector_efficiency_in_total_energy_intensity=True)

        source_electricity_adj = delivered_electricity.multiply(conversion_factors) # Column M
        source_adj = source_electricity_adj.add(fuels)
        source_adj['Energy_Type'] = 'Source_Adj'

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
        consumption_total = dataset[categories_list].sum(axis=1, skipna=True)
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
        log_ratio = np.log(dataset[categories_list].divide(dataset[categories_list].shift()))

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
        change = dataset[categories_list].diff()
        log_ratio = np.log(dataset[categories_list] / dataset[categories_list].shift())
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

        change = dataset[categories_list].diff()
        log_ratio = np.log(dataset[categories_list] / dataset[categories_list].shift())
        log_mean_divisia_weights = change.divide(log_ratio)
        log_mean_divisia_weights_total = dataset[[categories_list]].sum(axis=1, skipna=True)
        log_mean_divisia_weights_normalized = log_mean_divisia_weights.divide(log_mean_divisia_weights_total)

        return log_mean_divisia_weights, log_mean_divisia_weights_normalized

    def adjust_for_weather(self, data, energy_type, type, region):
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
        weather_factors = WeatherFactors(region, energy_type, type, sector=self.sector)
        weather_adjusted_data = data / weather_factors
        return weather_adjusted_data

    def lmdi_multiplicative(self, activity_input_data, energy_input_data, unit_conversion_factor=1):
        energy_shares = self.calculate_shares(self.energy_data, self.categories)
        log_mean_divisia_weights_energy, log_mean_divisia_weights_normalized_energy = self.calculate_log_mean_weights(energy_shares, self.categories)
        
        nominal_energy_intensity = self.energy_input_data.divide(self.activity_input_data).multiply(unit_conversion_factor)
        log_changes_intensity = self.calculate_log_changes(nominal_energy_intensity)

        activity_shares = self.calculate_shares(self.activity_data, self.categories)
        log_changes_activity_shares = self.calculate_log_changes_activity_shares(activity_shares)

        index_chg_energy, index_energy, index_normalized_energy = self.compute_index(log_mean_divisia_weights_normalized_energy, log_changes_intensity, self.categories)
        
        index_chg_activity, index_activity, index_normalized_activity = self.compute_index(log_mean_divisia_weights_normalized_energy, log_changes_activity_shares, self.categories)  

        # Final Indexes 
        activity_index = self.activity_data['Total'].divide(self.activity_data.loc[self.base_year, 'Total'])
        index_of_aggregate_intensity = nominal_energy_intensity['Total'].divide(nominal_energy_intensity.loc[self.base_year, 'Total'])
        structure_fuel_mix = index_normalized_activity
        component_intensity_index = index_normalized_energy
        product = activity_index.multiply(structure_fuel_mix).multiply(component_intensity_index)
        actual_energy_use = activity_index.multiply(index_of_aggregate_intensity)

        return activity_index, index_of_aggregate_intensity, structure_fuel_mix, component_intensity_index, product, actual_energy_use

    def lmdi_additive(self, activity_input_data, energy_input_data):
        pass

    def collect_energy_data(self):
        energy_data_by_type = dict()

        funcs = {'elec': self.get_elec(), 
                 'fuels': self.get_fuels(), 
                 'deliv': self.get_deliv(energy_data_by_type['elec'], energy_data_by_type['fuels']), 
                 'source': self.get_source(energy_data_by_type['elec']), 
                 'source_adj': self.get_source_adj(energy_data_by_type['elec'])}
        
        for e_type in energy_types:
            e_type_df = funcs[e_type]
            energy_data_by_type[e_type] = e_type_df
        
        return energy_data_by_type

    def call_lmdi(self, unit_conversion_factor, adjust_for_weather=False, lmdi_model='multiplicative'):
        
        energy_data_by_type = self.collect_energy_data()
        
        if adjust_for_weather: 
            for type, energy_dataframe in energy_data_by_type.items():
                region = 
                weather_adj_energy = adjust_for_weather(energy_dataframe, type, region) 
        
        if adjust_for_weather:

        multiplicative_results = []
        additive_results = []

        if lmdi_model == 'multiplicative':
            for type, energy_dataframe in energy_data_by_type.items():
                results = self.lmdi_multiplicative(self.activity_data, energy_dataframe, unit_conversion_factor)
                multiplicative_results.append(results)
        elif lmdi_model == 'additive': 
                        for type, energy_dataframe in energy_data_by_type.items():
                results = self.lmdi_additive(self.activity_data, energy_dataframe, unit_conversion_factor)
                additive_results.aapend(results)
        
        return multiplicative_results, additive_results

    def data_visualization(self,):
            """Format data for proper visualization
            
            The following data types have been proposed (an ellipsis ... indicates an optional parameter):

                @filter|Category1|...Category2|...|Label#units

                A list of options that can be grouped by 1 or more categories.
                @weight|Category1|...Category2|...|Label#units

                A weighted value to use with a matching filter (must match filter label and categories).
                @scenario|Label

                A list of options that are completely separate from each other, i.e. they will not be seen on the same chart at the same time.
                The options come from the unique values in the scenario column.
                @timeseries|Label

                A list of options that can be used to make a time series, e.g. a list of years.
                @geography|Label

                A list of geography names, e.g. states, counties, cities, that can be used in charts or a choropleth map.
                @geoid

                The column values are geography IDs that can be used in a choropleth map.
                @latlong

                Latitude and longitude coordinates
                        

            Parameters
            ----------
            
            Returns
            csv
            
            """



    