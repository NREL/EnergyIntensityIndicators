import pandas as pd
import numpy as np
from sklearn import linear_model
from functools import reduce
import os
from datetime import date
import matplotlib.pyplot as plt
import seaborn
import plotly.graph_objects as go
import plotly.express as px

from EnergyIntensityIndicators.pull_eia_api import GetEIAData


class CalculateLMDI:
    """Base class for LMDI"""
    def __init__(self, sector, level_of_aggregation, lmdi_models, categories_dict, energy_types, \
                 directory, output_directory, base_year=1985):
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
        self.output_directory = output_directory
        self.sector = sector
        self.level_of_aggregation = level_of_aggregation
        self.categories_dict = categories_dict
        self.base_year = base_year
        self.energy_types = energy_types  # could use energy_data.keys but need 'elec' and 'fuels' to come before the others
        self.lmdi_models = lmdi_models

    @staticmethod
    def ensure_same_indices(df1, df2):
        """Returns two dataframes with the same indices
        purpose: enable dataframe operations such as multiply and divide between the two dfs
        """        
        df1.index = df1.index.astype(int)
        df2.index = df2.index.astype(int)

        intersection_ = df1.index.intersection(df2.index)

        if len(intersection_) == 0: 
            raise ValueError('DataFrames do not contain any shared years')
        
        if isinstance(df1, pd.Series): 
            df1_new = df1.loc[intersection_]
        else:
            df1_new = df1.loc[intersection_, :]

        if isinstance(df2, pd.Series): 
            df2_new = df2.loc[intersection_]
        else:
            df2_new = df2.loc[intersection_, :]


        return df1_new, df2_new

    def get_elec(self, elec):
        """Add 'Energy_Type' column to electricity dataframe
        """        
        elec['Energy_Type'] = 'Electricity'
        print('Collected elec data')
        return elec

    def get_fuels(self, fuels):
        """Add 'Energy_Type' column to fuels dataframe
        """      
        fuels['Energy_Type'] = 'Fuels'
        print('Collected fuels data')
        return fuels

    def get_deliv(self, elec, fuels):
        """Calculate delivered energy by adding electricity and fuels then add 'Energy_Type' 
        column to the resulting delivered energy dataframe
        """      
        delivered = elec.add(fuels.values)
        delivered['Energy_Type'] = 'Delivered'
        print('Calculated deliv data')
        return delivered

    def get_source(self, elec, fuels):
        """Call conversion factors method from GetEIAData, calculate source energy from 
        conversion_factors, electricity and fuels dataframe, then add 'Energy-Type' column 
        to the resulting source energy dataframe
        """        
        conversion_factors = GetEIAData(self.sector).conversion_factors()
        print('conversion_factors: \n', conversion_factors)
        conversion_factors, elec = self.ensure_same_indices(conversion_factors, elec)
        source_electricity = elec.drop('Energy_Type', axis=1).multiply(conversion_factors.values) # Column A
        total_source = source_electricity.add(fuels.drop('Energy_Type', axis=1).values)     
        total_source['Energy_Type'] = 'Source'
        print('Calculated source data')
        return total_source
    
    def get_source_adj(self, elec, fuels):
        """Call conversion factors method from GetEIAData, calculate source adjusted energy from 
        conversion_factors, electricity and fuels dataframe, then add 'Energy-Type' column 
        to the resulting source adjusted energy dataframe
        """        
        conversion_factors = GetEIAData(self.sector).conversion_factors(include_utility_sector_efficiency=True)
        print('conversion_factors source adj: \n', conversion_factors)
                
        conversion_factors, elec = self.ensure_same_indices(conversion_factors, elec)

        source_electricity_adj = elec.drop('Energy_Type', axis=1).multiply(conversion_factors.values) # Column M
        source_adj = source_electricity_adj.add(fuels.drop('Energy_Type', axis=1).values)
        source_adj['Energy_Type'] = 'Source_Adj'
        print('Calculated source_adj data')
        return source_adj
    
    def calculate_energy_data(self, e_type, energy_data):
        """Calculate 'deliv', 'source', and 'source_adj' data from 
        'fuels' and 'elec' dataframes contained in the energy_data dictionary. 
        """

        funcs = {'elec': self.get_elec, 
                 'fuels': self.get_fuels, 
                 'deliv': self.get_deliv, 
                 'source': self.get_source, 
                 'source_adj': self.get_source_adj}

        if e_type in ['deliv', 'source', 'source_adj']:
            elec = energy_data['elec']
            elec['Total'] = elec.sum(axis=1)
            fuels = energy_data['fuels']
            fuels['Total'] = fuels.sum(axis=1)
            e_type_df = funcs[e_type](elec, fuels)
        elif e_type in ['elec', 'fuels']:
            data = energy_data[e_type]
            e_type_df = funcs[e_type](data)
        else:
            raise KeyError(f'{type} not in ["elec", "fuels", "deliv", "source", "source_adj"], user must define \
                               provide {type} data')
    
        return e_type_df

    def collect_energy_data(self, data): 
        """Calculate energy data for energy types in self.energy_types for which data is not provided

        Returns:
            [type]: [description]

        Example data: 
            passenger_based_energy_use = pd.read_csv('./Transportation/passenger_based_energy_use.csv').set_index('Year')
            passenger_based_activity = pd.read_csv('./Transportation/passenger_based_activity.csv').set_index('Year')
            freight_based_energy_use = pd.read_csv('./Transportation/freight_based_energy_use.csv').set_index('Year')
            freight_based_activity = pd.read_csv('./Transportation/freight_based_activity.csv').set_index('Year')

            data_dict = {'All_Passenger': {'energy': {'deliv': passenger_based_energy_use}, 
                                           'activity': passenger_based_activity}, 
                        'All_Freight': {'energy': {'deliv': freight_based_energy_use}, 
                                        'activity': freight_based_activity}}
        """         
        data_dict_gen = dict()
        for key in data:
            energy_data = data[key]['energy']
            activity_data = data[key]['activity']

            provided_energy_data = list(energy_data.keys())

            if set(provided_energy_data) == set(self.energy_types):
                energy_data_by_type = energy_data
            elif 'elec' in energy_data and 'fuels' in energy_data:
                energy_data_by_type = dict()
                for type in self.energy_types:
                    try: 
                        e_type_df = self.calculate_energy_data(type, energy_data)
                        energy_data_by_type[type] = e_type_df
                    except KeyError as err:
                        print(err.args) 
            else: 
                raise ValueError('Warning: energy data dict not well defined')


            data_dict = {'energy': energy_data_by_type, 'activity': activity_data}
            data_dict_gen[key] = data_dict
        

        return data_dict_gen

    @staticmethod
    def deep_get(dictionary, keys, default=None):
        return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

    def build_nest(self, data, select_categories, results_dict, breakout, level, level1_name, level_name=None):
        cat_columns = []
        print('select_categories:', select_categories)
        print('data: \n', data.keys())
        for key, value in select_categories.items():
            if type(value) is dict:
                level +=  1
                yield from self.build_nest(data=data, select_categories=value, results_dict=results_dict, \
                                                breakout=breakout, level=level, level1_name=level1_name, \
                                                level_name=key)
            else:
                if type(data['activity']) is dict:
                    for activity_type, a_df in data['activity'].items():
                        if key not in a_df.columns:
                            print(f'Warning: {key} column not in activity data')
                            yield None
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

        if isinstance(data['activity'], dict):
            activity_data = dict()
            energy_data = dict()

            for activity_type, a_df in data['activity'].items():
                a_data = a_df[cat_columns]
                new_col_names = {c: f'{activity_type}_{c}' for c in cat_columns}
                a_data = a_data.rename(columns=new_col_names)
                for e in self.energy_types:
                    e_data = data['energy'][e][cat_columns]
                    e_data, a_data = self.ensure_same_indices(e_data, a_data)

                    if not level_name:
                        level_name = level1_name
                    else:
                        a_d
                        a_data[level_name] = a_data.sum(axis=1).values
                        e_data[level_name] = e_data.sum(axis=1).values

                    energy_data[e] = e_data
                    activity_data[activity_type] = a_data

        elif isinstance(data['activity'], pd.DataFrame):
            activity_data = data['activity'][cat_columns]

            energy_data = dict()
            for e in self.energy_types:
                e_data = data['energy'][e][cat_columns]
                e_data, activity_data = self.ensure_same_indices(e_data, activity_data)

                if not level_name:
                    level_name = level1_name
                else:
                    activity_data[level_name] = activity_data.sum(axis=1).values
                    e_data[level_name] = e_data.sum(axis=1).values

                energy_data[e] = e_data

        data_dict = {'energy': energy_data, 'activity': activity_data, 'level_total': level_name}

        results_dict[f'{level_name}'] = data_dict 
        yield results_dict

    @staticmethod
    def create_total_column(df, total_label):
        df[total_label] = df.sum(axis=1).values
        return df 

    def calculate_breakout_lmdi(self, raw_results, final_results_list, level_of_aggregation, weather_data, save_breakout):
        """If breakout=True, calculate LMDI for each lower aggregation level contained in raw_results.

        Args:
            raw_results (dictionary): Built "nest" of dictionaries containing input data for LMDI calculations
            final_results_list (list): list to which calculate_breakout_lmdi appends LMDI results

        Returns:
            final_results_list [list]: list of LMDI results dataframes
        
        TODO: Lower level Total structure (product of each structure index for multiplicative) and component 
        intensity index (index of aggregate intensity divided by total strucutre) need to be passed to higher level
        """        
        for key in raw_results.keys():
            level_total = raw_results[key]['level_total']

            if level_of_aggregation[-1] == level_total:
                loa = [self.sector.capitalize()] + level_of_aggregation
            else:
                loa = [self.sector.capitalize()] + level_of_aggregation + [level_total]

            energy = raw_results[key]['energy']
            activity_ = raw_results[key]['activity']

            if isinstance(energy, pd.DataFrame) and isinstance(activity_, pd.DataFrame):
                energy = self.create_total_column(energy, level_total)
                activity_ = self.create_total_column(activity_, level_total)
                category_lmdi = self.call_lmdi(energy_df, activity_, level_total, lmdi_models=self.lmdi_models, \
                                               unit_conversion_factor=1, weather_data=weather_data, \
                                               save_results=save_breakout, loa=loa) 
                # Make sure this case only happens when there is one type
                category_lmdi["@filter|Energy_Type"] = self.energy_types[0] 
                final_results_list.append(category_lmdi)

            elif isinstance(energy, dict) and isinstance(activity_, pd.DataFrame):
                for e_type, energy_df in energy.items():
                    energy_df = self.create_total_column(energy_df, level_total)
                    activity_ = self.create_total_column(activity_, level_total)

                    category_lmdi = self.call_lmdi(energy_df, activity_, level_total, lmdi_models=self.lmdi_models, \
                                                   unit_conversion_factor=1, weather_data=weather_data, \
                                                   save_results=save_breakout, loa=loa, energy_type=e_type) 
                    category_lmdi["@filter|Energy_Type"] = e_type

                    final_results_list.append(category_lmdi)

            elif isinstance(energy, pd.DataFrame) and isinstance(activity_, dict):
                energy[level_total] = energy.sum(axis=1).values

                activity_ = {a_type: self.create_total_column(a_df, level_total) for (a_type, a_df) in activity_.items()}

                category_lmdi = self.call_lmdi(energy_df, activity_, level_total, lmdi_models=self.lmdi_models, \
                                               unit_conversion_factor=1, weather_data=weather_data, \
                                               save_results=save_breakout, loa=loa, energy_type=e_type) 
                category_lmdi["@filter|Energy_Type"] = e_type

                final_results_list.append(category_lmdi)

            elif isinstance(energy, dict) and isinstance(activity_, dict):
                activity_ = {a_type: self.create_total_column(a_df, level_total) for (a_type, a_df) in activity_.items()}

                for e_type, energy_df in energy.items():
                    energy_df = self.create_total_column(energy_df, level_total)
                    category_lmdi = self.call_lmdi(energy_df, activity_, level_total, lmdi_models=self.lmdi_models, \
                                                   unit_conversion_factor=1, weather_data=weather_data,\
                                                   save_results=save_breakout, loa=loa, energy_type=e_type) 
                    category_lmdi["@filter|Energy_Type"] = e_type

                final_results_list.append(category_lmdi)

        return final_results_list


    def get_nested_lmdi(self, level_of_aggregation, raw_data, calculate_lmdi=False, breakout=False, \
                        save_breakout=False, weather_data=None):
        """
        docstring

        TODO: 
            - Build in weather capabilities
        """
        final_fmt_results = []

        categories = self.deep_get(self.categories_dict, level_of_aggregation)
        level_of_aggregation = level_of_aggregation.split(".")
        level1_name = level_of_aggregation[-1]

        data = self.collect_energy_data(raw_data)

        if self.sector == 'transportation': 
            df_type_ = level_of_aggregation[0] 
            data = data[df_type_]

        results_dict = dict()
        for results_dict in self.build_nest(data=data, select_categories=categories, results_dict=results_dict, \
                                            level=1, level1_name=level1_name, breakout=breakout):
            if results_dict:
                if breakout:
                    self.calculate_breakout_lmdi(results_dict, final_fmt_results, level_of_aggregation, \
                                                 weather_data, save_breakout)
                    

        total_results_by_energy_type = dict()
        for e in self.energy_types:
            total_activty_ = results_dict[level1_name]['activity']
            total_energy_df = results_dict[level1_name]['energy'][e]

            if isinstance(total_activty_, dict):
                total_activty_ = {a_type: self.create_total_column(a_df, level_total) for \
                                  (a_type, a_df) in total_activty_.items()}

                pass
            elif isinstance(total_activty_, pd.DataFrame):
                total_activty_df = total_activty_
                for key, value in categories.items():
                    if isinstance(value, dict): 
                        total_activty_df[key] = results_dict[key]['activity'][key].values
                        total_energy_df[key] = results_dict[key]['energy'][e][key].values
                
                total_activty_df = self.create_total_column(total_activty_df, level1_name)
                total_energy_df = self.create_total_column(total_activty_df, level1_name)
                
                if calculate_lmdi:
                    loa = [self.sector.capitalize()] + level_of_aggregation
                    final_results = self.call_lmdi(total_energy_df, total_activty_df, total_label=level1_name, \
                                                   lmdi_models=self.lmdi_models, unit_conversion_factor=1, \
                                                   weather_data=weather_data, save_results=True, \
                                                   loa=loa, energy_type=e)
                    final_results["@filter|Energy_Type"] = e

                    final_fmt_results.append(final_results)

                    total_results_by_energy_type[e] = final_results

                else:
                    total_results_by_energy_type[e] = {'activity': total_activty_df, 'energy': total_energy_df}
        if len(final_fmt_results) > 0: 
            final_results = pd.concat(final_fmt_results, axis=0, ignore_index=True, join='outer')
        return total_results_by_energy_type, final_results

    @staticmethod
    def select_value(dataframe, base_row, base_column):
        return dataframe.iloc[base_row, base_column].values()
        
    @staticmethod
    def calculate_shares(dataset, total_label):
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
        shares = dataset.drop(total_label, axis=1).divide(dataset[total_label].values.reshape(len(dataset[total_label]), 1))
        return shares

    @staticmethod
    def calculate_log_changes(dataset):
        """Calculate the log changes
           Parameters
           ----------
           dataset: dataframe

           Returns
           -------
           log_ratio: dataframe

        """
        log_ratio = np.log(dataset.divide(dataset.shift().values, axis='columns'))

        return log_ratio

    def compute_index(self, effect):
        """
        """                     
        index = (effect * effect.shift()).ffill() #.fillna(1)  # first value should be set to 1? 
        index_normalized = index / index.loc[self.base_year] # 1985=1

        return index, index_normalized 

    @staticmethod
    def logarithmic_average(x, y):
        """The logarithmic average of two positive numbers x and y
        """        
        if x > 0 and y > 0:
            if x != y:
                difference = x - y
                log_difference = np.log(x) - np.log(y)
                L = difference / log_difference
            else:
                L = x
        else: 
            L = np.nan

        return L


    def log_mean_weights_multiplicative(self, energy_data, energy_shares, total_label):
        """Calculate log mean weights where T = t, 0 = t-1

        Multiplicative model uses the LMDI-II model because 'the weights...sum[] to unity, a 
        desirable property in index construction.' (Ang, B.W., 2015. LMDI decomposition approach: A guide for 
                                        implementation. Energy Policy 86, 233-238.).
        """

        log_mean_weights = pd.DataFrame(index=energy_data.index)

        for col in energy_shares.columns: 
            energy_shares[f"{col}_shift"] = energy_shares[col].shift(periods=1, axis='index', fill_value=0)
            
            # apply generally not preferred for row-wise operations but?
            log_mean_weights[f'log_mean_weights_{col}'] = energy_shares.apply(lambda row: \
                                                          self.logarithmic_average(row[col], row[f"{col}_shift"]), axis=1) 

        sum_log_mean_shares = log_mean_weights.sum(axis=1)
        log_mean_weights_normalized = log_mean_weights.divide(sum_log_mean_shares.values.reshape(len(sum_log_mean_shares), 1))
        return log_mean_weights_normalized
    
    def log_mean_weights_additive(self, energy_data, energy_shares, total_label, lmdi_type='LMDI-I'):
        """Calculate log mean weights for the additive model where T=t, 0 = t - 1

        Args:
            energy_data (dataframe): energy consumption data
            energy_shares (dataframe): Shares of total energy for each category in level of aggregation
            total_label (str): Name of aggregation of categories in level of aggregation
            lmdi_type (str, optional): 'LMDI-I' or 'LMDI-II'. Defaults to 'LMDI-I' because it is 'consistent in aggregation and perfect 
                                        in decomposition at the subcategory level' (Ang, B.W., 2015. LMDI decomposition approach: A guide for 
                                        implementation. Energy Policy 86, 233-238.).
        """        

        log_mean_shares_labels = [f"log_mean_shares_{col}" for col in energy_shares.columns]
        log_mean_weights = pd.DataFrame(index=energy_data.index)

        for col in energy_shares.columns: 
            energy_data[f"{col}_shift"] = energy_data[col].shift(periods=1, axis='index', fill_value=0)

            # apply generally not preferred for row-wise operations but?
            log_mean_values = energy_data[[col, f"{col}_shift"]].apply(lambda row: 
                                                                self.logarithmic_average(row[col],
                                                                 row[f"{col}_shift"]), axis=1) 

            energy_shares[f"{col}_shift"] = energy_shares[col].shift(periods=1, axis='index', fill_value=0)
             # apply generally not preferred for row-wise operations but?
            log_mean_shares = energy_shares[[col, f"{col}_shift"]].apply(lambda row: 
                                                                   self.logarithmic_average(row[col], \
                                                                        row[f"{col}_shift"]), axis=1)
            energy_shares[f"log_mean_shares_{col}"] = log_mean_shares

            log_mean_weights[f'log_mean_weights_{col}'] = log_mean_shares * log_mean_values
        
        if lmdi_type == 'LMDI-I':
            return log_mean_values
        elif lmdi_type == 'LMDI-II':
            sum_log_mean_shares = energy_shares[log_mean_shares_labels].sum(axis=1)
            log_mean_weights_normalized = log_mean_weights.divide(sum_log_mean_shares.values.reshape(len(sum_log_mean_shares), 1))

            log_mean_weights_normalized = log_mean_weights_normalized.drop([c for c in log_mean_weights_normalized.columns \
                                                                            if not c.startswith('log_mean_weights_')], axis=1)
            return log_mean_weights_normalized

    @staticmethod
    def nominal_energy_intensity(energy_input_data, activity_input_data):
        if isinstance(activity_input_data, pd.DataFrame):
            activity_width = activity_input_data.shape[1]
        elif isinstance(activity_input_data, pd.Series):
            activity_width = 1

        nominal_energy_intensity = energy_input_data.divide(activity_input_data.values.reshape(len(activity_input_data), \
                                                                                                activity_width)) 
                                                                                    #.multiply(unit_conversion_factor)
        return nominal_energy_intensity



    def lmdi(self, model, activity_input_data, energy_input_data, weather_data, lmdi_type=None, total_label=None, unit_conversion_factor=1,\
             return_nominal_energy_intensity=False):
        """Calculate the LMDI

        TODO: 
            - Account for weather factors when 

        Args:
            activity_input_data (dataframe or dictionary of dataframes): Activity input data for LMDI calculations
            energy_input_data (dataframe): Energy input data for LMDI calculations
            total_label (str): Name of the level of the level of aggregation representing the total of the current level. 
                               E.g. If categories are "Northeast", "South", etc, the total_label is "National"
            unit_conversion_factor (int, optional): [description]. Defaults to 1.
            return_nominal_energy_intensity (bool, optional): If True, returns nominal energy intensity and does 
            not calculate LMDI. Defaults to False.

        Returns:
            [type]: [description]
        """
        print('energy_input_data:', energy_input_data)
        energy_input_data, activity_input_data = self.ensure_same_indices(energy_input_data, activity_input_data)
        energy_shares = self.calculate_shares(energy_input_data, total_label)

        if isinstance(activity_input_data, dict):
            nominal_energy_intensity = {activity: self.nominal_energy_intensity(energy_input_data, activity_df) \
                                                                                for (activity, activity_df)\
                                                                                in activity_input_data.items()}
            if return_nominal_energy_intensity:
                return nominal_energy_intensity
            activity_shares = {activity: self.calculate_shares(activity_df, total_label) for \
                                (activity, activity_df) in activity_input_data.items()}
            log_ratio_structure = []
            for activity, activity_shares in activity_shares.items():
                # ln(ST_i/S0_i) --> S_i= Q_i / Q,  S_i is the activity share of sector i
                log_ratio_structure_activity = self.calculate_log_changes(activity_shares).rename(columns={col: \
                                                                                            f'{activity}_{col}' \
                                                                                            for col in \
                                                                                            activity_shares.columns}) 
                log_ratio_structure.append(log_ratio_structure_activity)
            log_ratio_structure = pd.concat(log_ratio_structure, axis=0, ignore_index=True, join='outer')

        else: 
            nominal_energy_intensity = self.nominal_energy_intensity(energy_input_data, activity_input_data)
            if return_nominal_energy_intensity:
                return nominal_energy_intensity
            activity_shares = self.calculate_shares(activity_input_data, total_label)
            # ln(ST_i/S0_i) --> S_i= Q_i / Q,  S_i is the activity share of sector i
            log_ratio_structure = self.calculate_log_changes(activity_shares) 

        # E is the total energy consumption in industry, Q is the total industrial activitiy level
        # ln(IT_i/I0_i) --> I_i = E_i / E,  I_i is the energy intensity of sector i
        log_ratio_intensity = self.calculate_log_changes(energy_shares) 
        # ln(QT/Q0)  --> Q = Q,  Q is the total insutrial activity level
        log_ratio_activity = self.calculate_log_changes(activity_input_data[[total_label]])  

        if model == 'multiplicative':
            log_mean_divisia_weights_normalized = self.log_mean_weights_multiplicative(energy_input_data, \
                                                                                       energy_shares, total_label)
        elif model == 'additive':
            log_mean_divisia_weights_normalized = self.log_mean_weights_additive(energy_input_data, \
                                                                                 energy_shares, total_label, lmdi_type=lmdi_type)
            cols_to_drop1 = [col for col in energy_shares.columns if col.startswith('log_mean_shares_')]
            energy_shares = energy_shares.drop(cols_to_drop1, axis=1)

        cols_to_drop = [col for col in energy_shares.columns if col.endswith('_shift')]
        energy_shares = energy_shares.drop(cols_to_drop, axis=1)

        activity = (log_mean_divisia_weights_normalized.multiply(log_ratio_activity, axis='columns')).sum(axis=1)

        intensity = (log_mean_divisia_weights_normalized.multiply(log_ratio_intensity, axis='columns')).sum(axis=1)

        structure = (log_mean_divisia_weights_normalized.multiply(log_ratio_structure.values, axis='columns')).sum(axis=1)

        if weather_data: 
            if weather_data.shape[1] == 1:
                if model == 'multiplicative': 
                    structure_weather = weather_data.divide(weather_data.loc[self.base_year, :]) 
                elif model == 'additive': 
                    pass
            elif weather_data.shape[1] > 1:
                structure_weather = self.calculate_log_changes(weather_data)
                structure_weather = (log_mean_divisia_weights_normalized.multiply(structure_weather, axis='columns')).sum(axis=1)
                
                if model == 'multiplicative': 
                    # CALCULATE INDEX IF MULTIPLICATIVE, WHAT IF ADDITIVE?
                    pass
                elif model == 'additive': 
                    pass
                
                structure['structure_weather'] = structure_weather

        if model == 'multiplicative':
            activity = np.exp(activity)
            structure = np.exp(structure)
            intensity = np.exp(intensity)

        results = pd.DataFrame.from_dict(data={'activity': activity, 'structure': structure, 
                                               'intensity': intensity}, orient='columns')

        if model == 'multiplicative':
            results['effect'] = results.product(axis=1)

        elif model == 'additive':
            results['effect'] = results.sum(axis=1)
        
        print('results: \n', results)
        return results

    def aggregate_additive(self, results_df, energy_input_data, total_label):
        df = results_df.loc[self.base_year + 1: , :]
        df = df.sum(axis=0)
        df['initial_energy'] = energy_input_data.loc[self.base_year, total_label]
        df['final_energy'] = energy_input_data.loc[max(results_df.index), total_label]
        return df

    def call_lmdi(self, energy_data, activity_data, total_label, lmdi_models, unit_conversion_factor,\
                  weather_data, save_results, lmdi_type=None, loa=None, energy_type=None):
        results = dict()

        if 'multiplicative' in lmdi_models:
            multiplicative_results = self.lmdi('multiplicative', activity_data, energy_data, weather_data, total_label,  \
                                               unit_conversion_factor)
            results['multiplicative'] = multiplicative_results
            
        if 'additive' in lmdi_models: 
            additive_results = self.lmdi('additive', activity_data, energy_data, weather_data, lmdi_type, total_label,  \
                                         unit_conversion_factor)
            results['additive'] = additive_results

        if save_results:
            fmt_loa = [l.replace(" ", "_") for l in loa]
            for model, result in results.items():
                if model == 'additive':
                    df = self.aggregate_additive(result, energy_data, total_label)
                    final_year = max(result.index)
                    self.waterfall_chart(df, final_year, loa, model, 'activity', 'structure', 'intensity')
                elif model == 'multiplicative':
                    self.lineplot(result, loa, model, energy_type, 'activity', 'structure', 'intensity', 'effect') # path, 
                formatted_data = self.data_visualization(result, fmt_loa)
                formatted_data['@filter|Model'] = model.capitalize()

        return formatted_data

    def data_visualization(self, data, loa):
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
        
        Example Data Schema:
            +--------------+---------+------------------+----------------------------+-----------------------------+-----------------------------+-----------------------------+
            | "@Geography" | "@Year" | "@filter|Sector" | "@filter|Measure|Activity" | "@filter|Measure|Structure" | "@filter|Measure|Intensity" | "@filter|Measure|Weight"    |
            +==============+=========+==================+============================+=============================+=============================+=============================+
            | National     | 2000    | A                | 0                          |             0               |                 0           |             0               |
            +--------------+---------+------------------+----------------------------+-----------------------------+-----------------------------+-----------------------------+
            | National     | 2000    | B                |              0             |                 0           |                 0           |          0                  |
            +--------------+---------+------------------+----------------------------+-----------------------------+-----------------------------+-----------------------------+
            | National     | 2010    | A                |         0.8123             |           .6931             |         -0.1823             |        86.56                |
            +--------------+---------+------------------+----------------------------+-----------------------------+-----------------------------+-----------------------------+
            | National     | 2010    | B                |     0.8123                 |         -0.287              |        -0.287               |        33.07                |
            +--------------+---------+------------------+----------------------------+-----------------------------+-----------------------------+-----------------------------+

        Parameters
        ----------
        
        Returns
        csv
        
        """
        # output formatted csv and/or figure (summary lineplot, etc like website) formatted table 
        # (summary tables on website), default: do all
        # scenario: additive/mult
        # filter: level?
        data = data.reset_index()
        data = data.rename(columns={'Year': '@timeseries|Year', 'activity': "@filter|Measure|Activity", 
                                    'effect': "@filter|Measure|Effect", 'intensity': "@filter|Measure|Intensity", 
                                    'structure': "@filter|Measure|Structure"})
        data = data[['@timeseries|Year', "@filter|Measure|Activity", "@filter|Measure|Structure", "@filter|Measure|Effect", \
                     "@filter|Measure|Intensity"]]
        for i, l in enumerate(loa):
            label = f"@filter|Subsector_Level_{i + 1}"
            print('label, l:', label, l)
            data[label] = l

        return data
    
    def waterfall_chart(self, data, final_year, loa, model, *x_data):
        print('data: \n', data)
        print('data.ravel() : \n', data.ravel())
        figure_labels = []
        loa = [l.replace("_", " ") for l in loa]
        title = f"Change {self.base_year}-{final_year} {' '.join(loa)} {model.capitalize()}"
        x_data = ['initial_energy'] + list(x_data) + ['final_energy']
        y_data = data.ravel()
        x_labels = [x.replace("_", " ").capitalize() for x in x_data]
        
        # for example: ["relative", "relative", "total", "relative", "relative", "total"]
        measure =  ['relative'] * len(list(x_labels)) 
        fig = go.Figure(go.Waterfall(name="Change", orientation="v", measure=measure, x=x_labels, 
                                     textposition="outside", text=figure_labels, y=y_data, 
                                     connector={"line":{"color":"rgb(63, 63, 63)"}}))
                                      #  color_discrete_sequence=px.colors.qualitative.Vivid,

        fig.update_layout(title=title, showlegend = True)

        fig.show()
        # fig.save(f"{path}/{title}.png")
        
    
    @staticmethod
    def lineplot(data, loa, model, energy_type, *lines_to_plot): # path
        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set2')

        for i, l in enumerate(lines_to_plot):
            label_ = l.replace("_", " ").capitalize()
            plt.plot(data.index, data[l], marker='', color=palette(i), linewidth=1, alpha=0.9, label=label_)
        
        loa = [l_.replace("_", " ") for l_ in loa]
        loa = " /".join(loa)
        title = loa + f" {model.capitalize()}" + f" {energy_type.capitalize()}" 
        plt.title(title, fontsize=12, fontweight=0)
        plt.xlabel('Year')
        # plt.ylabel('')
        plt.legend(loc=2, ncol=2)
        plt.show()
        # plt.save(f"{path}/{title}.png")
    
    @staticmethod
    def main():
        print('main')

if __name__ == '__main__':
    pass



    