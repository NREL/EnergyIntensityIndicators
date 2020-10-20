import pandas as pd
import numpy as np
from sklearn import linear_model
from pull_eia_api import GetEIAData
from functools import reduce
import os
from datetime import date
import matplotlib.pyplot as plt
import seaborn
import plotly.graph_objects as go

class CalculateLMDI:
    """Base class for LMDI"""
    def __init__(self, sector, level_of_aggregation, lmdi_models, categories_dict, energy_types, directory, output_directory, base_year=1985, base_year_secondary=1996, charts_ending_year=2003):
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
        self.index_base_year_secondary = base_year_secondary  # not used
        self.charts_starting_year = base_year
        self.charts_ending_year = charts_ending_year
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
        elec['Energy_Type'] = 'Electricity'
        print('Collected elec data')
        return elec

    def get_fuels(self, fuels):
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
    
    def calculate_energy_data(self, e_type, energy_data):

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

            data_dict = {'All_Passenger': {'energy': {'deliv': passenger_based_energy_use}, 'activity': passenger_based_activity}, 
                        'All_Freight': {'energy': {'deliv': freight_based_energy_use}, 'activity': freight_based_activity}}
        """         
        data_dict_gen = dict()
        for key in data:
            energy_data = data[key]['energy']
            activity_data = data[key]['activity']

            provided_energy_data = {e: energy_data[e] for e in self.energy_types}

            if provided_energy_data == energy_data:
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

    def get_nested_lmdi(self, level_of_aggregation, raw_data, calculate_lmdi=False, breakout=False, save_breakout=False, account_for_weather=False):
        """
        docstring

        TODO: 
            - Build in weather capabilities
            - Allow for multiple activity dataframes (needed for the Residential sector)
        """

        categories = self.deep_get(self.categories_dict, level_of_aggregation)
        level_of_aggregation = level_of_aggregation.split(".")
        level1_name = level_of_aggregation[-1]

        print('categories', categories)
        
        print('level_of_aggregation', level_of_aggregation)
        print('level1_name', level1_name)
        
        data = self.collect_energy_data(raw_data)

        if self.sector == 'transportation': 
            df_type_ = level_of_aggregation[0] 
            data = data[df_type_]

        results_dict = dict()
        lmdi_dict = dict()
        for results_dict in self.build_nest(data=data, select_categories=categories, results_dict=results_dict, \
                                            level=1, level1_name=level1_name, breakout=breakout):
            if breakout:
                for key in results_dict.keys():
                    level_total = results_dict[key]['level_total']
                    print('level total: \n', level_total)

                    energy = results_dict[key]['energy']
                    activity_ = results_dict[key]['activity']

                    if isinstance(energy, pd.DataFrame) and isinstance(activity_, pd.DataFrame):
                        energy[level_total] = energy.sum(axis=1).values
                        activity_[level_total] = activity_.sum(axis=1).values
                        category_lmdi = self.call_lmdi(energy_df, activity_, level_total, lmdi_models=self.lmdi_models, \
                                                       unit_conversion_factor=1, account_for_weather=account_for_weather, save_results=save_breakout, loa=level_of_aggregation)  # what should happen with this?
                    elif isinstance(energy, dict) and isinstance(activity_, pd.DataFrame):
                        for e_type, energy_df in energy.items():
                            energy_df[level_total] = energy_df.sum(axis=1).values
                            activity_[level_total] = activity_.sum(axis=1).values
                            category_lmdi = self.call_lmdi(energy_df, activity_, level_total, lmdi_models=self.lmdi_models, \
                                                           unit_conversion_factor=1, account_for_weather=account_for_weather, save_results=save_breakout, loa=level_of_aggregation)  # what should happen with this?
                    elif isinstance(energy, pd.DataFrame) and isinstance(activity_, dict):
                        pass # How to handle??
                    elif isinstance(energy, dict) and isinstance(activity_, dict):
                        for e_type, energy_df in energy.items():
                            pass # How to handle??
                    else:
                        pass


        total_results_by_energy_type = dict()
        for e in self.energy_types:
            total_activty_ = results_dict[level1_name]['activity']
            total_energy_df = results_dict[level1_name]['energy'][e]

            if isinstance(total_activty_, dict):
                # HOW TO DO THIS??
                pass
            elif isinstance(total_activty_, pd.DataFrame):
                total_activty_df = total_activty_
                for key, value in categories.items():
                    if isinstance(value, dict): 
                        total_activty_df[key] = results_dict[key]['activity'][key].values
                        total_energy_df[key] = results_dict[key]['energy'][e][key].values
                
                total_activty_df[level1_name] = total_activty_df.sum(axis=1).values
                total_energy_df[level1_name] = total_energy_df.sum(axis=1).values
                
                if calculate_lmdi:
                    final_results = self.call_lmdi(total_energy_df, total_activty_df, total_label=level1_name, \
                                                   lmdi_models=self.lmdi_models, unit_conversion_factor=1, \
                                                   account_for_weather=account_for_weather, save_results=True, loa=level_of_aggregation)
                    total_results_by_energy_type[e] = final_results

                else:
                    total_results_by_energy_type[e] = {'activity': total_activty_df, 'energy': total_energy_df}
        return total_results_by_energy_type

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

    def compute_index(self, log_mean_divisia_weights, log_changes_activity_shares):
        """[summary]

        Args:
            log_mean_divisia_weights ([type]): [description]
            log_changes_activity_shares ([type]): [description]

        Returns:
            [type]: [description]
        """                     
        index_chg = np.exp((log_mean_divisia_weights.multiply(log_changes_activity_shares)).sum(axis=1))
        index = (index_chg * index_chg.shift()).ffill().fillna(1)  # first value should be set to 1? 
        index_normalized = index / index.loc[self.base_year] # 1985=1

        return index_chg, index, index_normalized 

    @staticmethod
    def calculate_log_changes_activity_shares(dataset, total_label):
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
        change = dataset.diff()
        log_ratio = np.log(dataset.divide(dataset.shift().values))
        log_changes = change.divide(log_ratio)
        return log_changes
    
    @ staticmethod
    def calculate_log_mean_weights(dataset):
        """purpose
           Parameters
           ----------
           dataset: dataframe
                Description
            total_label: list
                Description
                
           Returns
           -------
        TODO: Verify that this is the desired logarithmic average 
        """

        change = dataset.diff()
        log_ratio = np.log(dataset.divide(dataset.shift().values))
        log_mean_divisia_weights = change.divide(log_ratio)
        log_mean_divisia_weights_total = log_mean_divisia_weights.sum(axis=1) 
        log_mean_divisia_weights_normalized = log_mean_divisia_weights.divide(log_mean_divisia_weights_total.values.reshape(len(log_mean_divisia_weights_total), 1))

        return log_mean_divisia_weights, log_mean_divisia_weights_normalized


    def lmdi_multiplicative(self, activity_input_data, energy_input_data, total_label=None, unit_conversion_factor=1, return_nominal_energy_intensity=False):
        """Calculate the Multiplicative LMDI

        TODO: 
            - Account for weather factors when 

        Args:
            activity_input_data ([type]): [description]
            energy_input_data ([type]): [description]
            total_label ([type]): [description]
            unit_conversion_factor (int, optional): [description]. Defaults to 1.
            return_nominal_energy_intensity (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        energy_input_data, activity_input_data = self.ensure_same_indices(energy_input_data, activity_input_data)
        print('activity_input_data: \n', activity_input_data)
        if isinstance(activity_input_data, pd.DataFrame):
            activity_width = activity_input_data.shape[1]
        elif isinstance(activity_input_data, pd.Series):
            activity_width = 1
        nominal_energy_intensity = energy_input_data.divide(activity_input_data.values.reshape(len(activity_input_data), activity_width)) #.multiply(unit_conversion_factor)
        if return_nominal_energy_intensity:
            return nominal_energy_intensity
        log_changes_intensity = self.calculate_log_changes(nominal_energy_intensity)
        energy_shares = self.calculate_shares(energy_input_data, total_label)
        log_mean_divisia_weights_energy, log_mean_divisia_weights_normalized_energy = self.calculate_log_mean_weights(energy_shares, total_label)
        

        activity_shares = self.calculate_shares(activity_input_data, total_label)
        log_changes_activity_shares = self.calculate_log_changes_activity_shares(activity_shares, total_label)

        index_chg_energy, index_energy, index_normalized_energy = self.compute_index(log_mean_divisia_weights_normalized_energy, log_changes_intensity)
        
        index_chg_activity, index_activity, index_normalized_activity = self.compute_index(log_mean_divisia_weights_normalized_energy, log_changes_activity_shares)  

        # Final Indexes 
        activity_index = activity_input_data[[total_label]].divide(activity_input_data.loc[self.base_year, total_label])
        print(activity_index.columns)
        final_indices_df = activity_index.copy().rename(columns={0: "activity_index"})
        final_indices_df['index_of_aggregate_intensity'] = nominal_energy_intensity[[total_label]].divide(nominal_energy_intensity.loc[self.base_year, total_label])
        final_indices_df['structure_fuel_mix'] = index_normalized_activity
        final_indices_df['component_intensity_index'] = index_normalized_energy
        final_indices_df['product'] = activity_index.multiply(final_indices_df['structure_fuel_mix']).multiply(['component_intensity_index'])
        final_indices_df['actual_energy_use'] = activity_index.multiply(final_indices_df['index_of_aggregate_intensity'])
        return final_indices_df

    def lmdi_additive(self, activity_input_data, energy_input_data, total_label, unit_conversion_factor):
        L = calculate_log_mean_weights(dataset)
        d_E_act = 
        d_E_str = 
        d_E_ins = 

        result = d_E_act + d_E_str + d_E_ins 

        return None

    def call_lmdi(self, energy_data, activity_data, total_label, lmdi_models, unit_conversion_factor, account_for_weather, save_results, loa=None):
        results = dict()

        if account_for_weather: 
            pass
        if isinstance(activity_data, dict):
            ## HOW TO DO THIS??
            pass
        if 'multiplicative' in lmdi_models:
            multiplicative_results = self.lmdi_multiplicative(activity_data, energy_data, total_label,  unit_conversion_factor)
            print('multiplicative_results: \n', multiplicative_results)
            results['multiplicative'] = multiplicative_results
            
        if 'additive' in lmdi_models: 
            # additive_results = self.lmdi_additive(activity_data, energy_data, total_label, unit_conversion_factor)
            additive_results = None
            results['additive'] = additive_results


        if save_results:
            fmt_loa = [l.replace(" ", "_") for l in loa]
            path = f"{self.output_directory}/{'/'.join(fmt_loa)}"
            if not os.path.exists():
                os.mkdir(path)
            for model, result in results.items():
                f_name = '.'.join(fmt_loa) + model
                formatted_data = self.data_visualization(result)
                date = date.today().strftime("%m%d%y")
                formatted_data.to_csv(f'{path}/{f_name}_{date}.csv')

        return results

    def data_visualization(self, data):
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
        # output formatted csv and/or figure (summary lineplot, etc like website) formatted table 
        # (summary tables on website), default: do all
        label_ = 
        data = data.reset_index()
        data = data.rename(columns={'Year': f'@timeseries|{label_}'})
        return formatted_data
    
    @staticmethod
    def waterfall_chart():
        x_labels = []
        figure_labels = []
        title = ""
        measure = []  # for example: ["relative", "relative", "total", "relative", "relative", "total"]
        fig = go.Figure(go.Waterfall(name="", orientation="v", measure=measure, x=x_labels, 
                                     textposition="outside", text=figure_labels, y=values,
                                     connector = {"line":{"color":"rgb(63, 63, 63)"}},))

        fig.update_layout(title=title, showlegend = True)

        fig.show()
        fig.save(f"{path}/{title}.png")
        
    
    @staticmethod
    def lineplot(path, *lines_to_plot):
        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set1')
        for i, l in enumerate(lines_to_plot):
            plt.plot(x, y, marker='', color=palette(i), linewidth=1, alpha=0.9, label='')
        
        title = 
        plt.title(title, fontsize=12, fontweight=0)
        plt.xlabel('')
        plt.ylabel('')
        plt.legend(loc=2, ncol=2)
        plt.show()
        plt.save(f"{path}/{title}.png")





    