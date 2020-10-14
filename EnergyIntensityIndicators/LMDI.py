import pandas as pd
import numpy as np
from sklearn import linear_model
from pull_eia_api import GetEIAData
from functools import reduce

class CalculateLMDI:
    """Base class for LMDI"""
    def __init__(self, sector, level_of_aggregation, lmdi_models, categories_dict, energy_types, directory, energy_data=None, activity_data=None, base_year=1985, base_year_secondary=1996, charts_ending_year=2003):
        """
        Parameters
        ----------
        energy_data: dictionary of dataframes
            Energy input data, keys are the energy_type
        activity_data: dataframe
            Activity input data
        categories_dict: dict
            nested dictionary providing relationships between various levels of aggregation
        level_of_aggregation: str
            path in categories_dict to desired level of aggregation
                e.g. 'All_Freight.Pipeline' calculates the LMDI for Pipelines, a subcategory of All_Freight

        """
        self.directory = directory
        self.sector = sector
        self.energy_data = energy_data
        self.activity_data = activity_data 
        self.level_of_aggregation = level_of_aggregation
        self.categories_dict = categories_dict
        self.index_base_year_primary = base_year
        self.index_base_year_secondary = base_year_secondary  # not used
        self.charts_starting_year = base_year
        self.charts_ending_year = charts_ending_year
        self.energy_types = energy_types  # could use energy_data.keys but need 'elec' and 'fuels' to come before the others
        self.lmdi_models = lmdi_models

    def get_elec(self):
        elec = self.energy_data['elec']
        elec['Energy_Type'] = 'Electricity'
        print('Collected elec data')
        return elec

    def get_fuels(self):
        fuels = self.energy_data['fuels']
        fuels['Energy_Type'] = 'Fuels'
        print('Collected fuels data')
        return fuels

    def get_deliv(self, elec, fuels):
        delivered = elec.add(fuels.values)
        delivered['Energy_Type'] = 'Delivered'
        print('Calculated deliv data')
        return delivered

    def get_source(self, elec, fuels):
        conversion_factors = GetEIAData(self.sector).conversion_factors()
        source_electricity = elec[['adjusted_consumption_trillion_btu', 'Total']].multiply(conversion_factors.values) # Column A
        total_source = source_electricity.add(fuels[['adjusted_consumption_trillion_btu', 'Total']].values)     
        total_source['Energy_Type'] = 'Source'
        print('Calculated source data')
        return total_source
    
    def get_source_adj(self, elec, fuels):
        conversion_factors = GetEIAData(self.sector).conversion_factors(include_utility_sector_efficiency_in_total_energy_intensity=True)
        source_electricity_adj = elec[['adjusted_consumption_trillion_btu', 'Total']].multiply(conversion_factors.values) # Column M
        source_adj = source_electricity_adj.add(fuels[['adjusted_consumption_trillion_btu', 'Total']].values)
        source_adj['Energy_Type'] = 'Source_Adj'
        print('Calculated source_adj data')
        return source_adj
    
    def collect_energy_data(self):
        energy_data_by_type = dict()

        funcs = {'elec': self.get_elec, 
                 'fuels': self.get_fuels, 
                 'deliv': self.get_deliv, 
                 'source': self.get_source, 
                 'source_adj': self.get_source_adj}
        
        if len(self.energy_types) == 1:
            energy_data_by_type[self.energy_types[0]]
            return energy_data_by_type

        for e_type in self.energy_types:
            if e_type in ['deliv', 'source', 'source_adj']:
                elec = energy_data_by_type['elec']
                elec['Total'] = elec.sum(axis=1)
                fuels = energy_data_by_type['fuels']
                fuels['Total'] = fuels.sum(axis=1)
                e_type_df = funcs[e_type](elec, fuels)
            else:
                e_type_df = funcs[e_type]()
            energy_data_by_type[e_type] = e_type_df
            print(energy_data_by_type)
    
        return energy_data_by_type

    def collect_data(self):
        # energy_data_by_type = self.collect_energy_data()
        # activity_data = self.activity_data()
        # data_dict = {'energy': energy_data_by_type, 'activity': activity_data}
        passenger_based_energy_use = pd.read_csv('./Transportation/passenger_based_energy_use.csv').set_index('Year')
        passenger_based_activity = pd.read_csv('./Transportation/passenger_based_activity.csv').set_index('Year')
        freight_based_energy_use = pd.read_csv('./Transportation/freight_based_energy_use.csv').set_index('Year')
        freight_based_activity = pd.read_csv('./Transportation/freight_based_activity.csv').set_index('Year')

        data_dict = {'All_Passenger': {'energy': {'deliv': passenger_based_energy_use}, 'activity': passenger_based_activity}, 
                     'All_Freight': {'energy': {'deliv': freight_based_energy_use}, 'activity': freight_based_activity}}
        return data_dict

    @staticmethod
    def deep_get(dictionary, keys, default=None):
        return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

    def build_nest(self, data, select_categories, results_dict, breakout, level, level1_name, level_name=None):
        cat_columns = []
        for key, value in select_categories.items():
            if type(value) is dict:
                level +=  1
                yield from self.build_nest(data=data, select_categories=value, results_dict=results_dict, \
                                                breakout=breakout, level=level, level1_name=level1_name, \
                                                level_name=key)
            else:
                if key not in data['activity'].columns:
                    print(f'Warning: {key} column not in activity data')
                    yield None
                for e in self.energy_types:
                    if key not in data['energy'][e].columns:
                        print(f'Warning: {key} column not in {e} data')
                        yield None
                else:
                    cat_columns.append(key)
        
        activity_data = data['activity'][cat_columns]
        energy_data = dict()

        for e in self.energy_types:
            e_data = data['energy'][e][cat_columns]
            if not level_name:
                level_name = level1_name
            else:
                activity_data[level_name] = activity_data.sum(axis=1).values
                e_data[level_name] = e_data.sum(axis=1).values

            energy_data[e] = e_data

        data_dict = {'energy': energy_data, 'activity': activity_data}

        results_dict[f'{level_name}'] = data_dict 
        yield results_dict

    def get_nested_lmdi(self, level_of_aggregation, calculate_lmdi=False, breakout=False):
        """
        docstring
        """
        categories = self.deep_get(self.categories_dict, level_of_aggregation)
        level_of_aggregation = level_of_aggregation.split(".")
        level1_name = level_of_aggregation[-1]

        print('categories', categories)
        
        print('level_of_aggregation', level_of_aggregation)
        print('level1_name', level1_name)
        
        data = self.collect_data()

        if self.sector == 'transportation': 
            df_type_ = level_of_aggregation[0] 
            data = data[df_type_]

        results_dict = dict()
        for results_dict in self.build_nest(data=data, select_categories=categories, results_dict=results_dict, level=1, level1_name=level1_name, breakout=breakout):
            if breakout:
                energy_df = results_dict['energy']
                activity_df = results_dict['activity']

                category_lmdi = self.call_lmdi(energy_df, activity_df, lmdi_models=self.lmdi_model, unit_conversion_factor=1)  # what should happen with this?

        total_results_by_energy_type = dict()
        for e in self.energy_types:
            total_activty_df = results_dict[level1_name]['activity']
            total_energy_df = results_dict[level1_name]['energy'][e]

            for key, value in categories.items():
                if type(value) is dict: 
                    total_activty_df[key] = results_dict[key]['activity'][key].values
                    total_energy_df[key] = results_dict[key]['energy'][e][key].values
            
            total_activty_df[level1_name] = total_activty_df.sum(axis=1).values
            total_energy_df[level1_name] = total_energy_df.sum(axis=1).values
            
            if calculate_lmdi:
                final_results = self.call_lmdi(total_energy_df, total_activty_df, lmdi_models=self.lmdi_models, unit_conversion_factor=1)
                total_results_by_energy_type[e] = final_results

            else:
                total_results_by_energy_type[e] = {'activity': total_activty_df, 'energy': total_energy_df}
        print(total_results_by_energy_type)
        return total_results_by_energy_type

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
        log_ratio = np.log(dataset.divide(dataset.shift()))

        return log_ratio

    def compute_index(self, log_mean_divisia_weights, log_changes_activity_shares, categories_list):
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
        index_normalized = index / self.select_value(dataframe=index, base_row=self.index_base_year_primary, base_column=1) # 1985=1

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


    def lmdi_multiplicative(self, activity_input_data, energy_input_data, unit_conversion_factor=1):
        energy_shares = self.calculate_shares(energy_input_data, self.categories)
        log_mean_divisia_weights_energy, log_mean_divisia_weights_normalized_energy = self.calculate_log_mean_weights(energy_shares, self.categories)
        
        nominal_energy_intensity = energy_input_data.divide(self.activity_input_data).multiply(unit_conversion_factor)
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
        return None

    def call_lmdi(self, energy_data, activity_data, lmdi_models, unit_conversion_factor):
        
        if 'multiplicative' in lmdi_models:
            multiplicative_results = self.lmdi_multiplicative(activity_data, energy_data, unit_conversion_factor)
        else: 
            multiplicative_results = None

        if 'additive' in lmdi_models: 
            # additive_results = self.lmdi_additive(activity_data, energy_data, unit_conversion_factor)
            additive_results = None
        else:
            additive_results = None

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



    