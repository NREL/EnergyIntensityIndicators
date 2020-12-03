# from abc import ABC, abstractmethod, abstractstaticmethod
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
from EnergyIntensityIndicators.multiplicative_lmdi import MultiplicativeLMDI
from EnergyIntensityIndicators.additive_lmdi import AdditiveLMDI


class LMDI():
    """Base class for LMDI"""

    LMDI_types = {'additive': AdditiveLMDI,
                   'multiplicative': MultiplicativeLMDI}

    def __init__(self, sector, lmdi_models, output_directory, base_year, end_year):
        self.sector = sector
        self.lmdi_models = lmdi_models
        self.output_directory = output_directory
        self.base_year = base_year
        self.end_year = end_year

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
        log_ratio_df = pd.DataFrame(data=log_ratio, index=dataset.index, columns=dataset.columns)
        return log_ratio_df

    @staticmethod
    def calc_component(log_ratio_component, weights):
        component = (weights.multiply(log_ratio_component.values, axis='columns')).sum(axis=1)
        return component

    def calc_ASI(self, model, weather_data, log_mean_divisia_weights_normalized, 
                 log_ratios):
        """Calculate activity, structure, and intensity 
        """        
        activity = self.calc_component(log_ratios['activity'], log_mean_divisia_weights_normalized)
        intensity = self.calc_component(log_ratios['intensity'], log_mean_divisia_weights_normalized)
        structure = self.calc_component(log_ratios['structure'], log_mean_divisia_weights_normalized)
        try:
            lower_level_structure = self.calc_component(log_ratios['lower_level_structure'], log_mean_divisia_weights_normalized)
            print('COMPONENT:\n', lower_level_structure)
        except KeyError:
            lower_level_structure = pd.DataFrame()

        if weather_data: 
            if weather_data.shape[1] == 1:
                structure_weather = weather_data.divide(weather_data.loc[self.base_year].values)
            elif weather_data.shape[1] > 1:
                structure_weather = self.calculate_log_changes(weather_data)
                structure_weather = (log_mean_divisia_weights_normalized.multiply(structure_weather, axis='columns')).sum(axis=1)

        if not lower_level_structure.empty:
            ASI = {'activity': activity, 'structure': structure, 
                    'intensity': intensity, 'lower_level_structure': lower_level_structure}
        else: 
            ASI = {'activity': activity, 'structure': structure, 
                   'intensity': intensity}

        print('FINAL ASI KEYS:', ASI.keys())
        return ASI

    def call_decomposition(self, energy_data, energy_shares, weather_data, 
                           log_ratios, total_label, lmdi_type, loa, save_results, 
                           energy_type):
        results_list = []
        if isinstance(self.lmdi_models, list):
            pass
        else:
            self.lmdi_models = [self.lmdi_models]

        for model in self.lmdi_models:
            if model == 'additive':
                lmdi_type = lmdi_type
            else:
                lmdi_type = None

            model_ = self.LMDI_types[model](energy_data, energy_shares, 
                                            self.base_year, self.end_year, total_label, lmdi_type)
            weights = model_.log_mean_divisia_weights()
        
            cols_to_drop_ = [col for col in weights.columns if col.endswith('_shift')]
            weights = weights.drop(cols_to_drop_, axis=1)

            components = self.calc_ASI(model, weather_data, weights, log_ratios)

            results = model_.decomposition(components)

            fmt_loa = [l.replace(" ", "_") for l in loa]
            formatted_data = self.data_visualization(results, fmt_loa)
            formatted_data['@filter|Model'] = model.capitalize()
            formatted_data['@filter|EnergyType'] = energy_type

            data_to_plot = formatted_data[formatted_data["@filter|Measure|BaseYear"] == self.base_year]

            model_.visualizations(data_to_plot, self.base_year, self.end_year, 
                                  loa, model, energy_type, 
                                  "@filter|Measure|Activity", "@filter|Measure|Intensity",
                                  "@filter|Measure|Structure", "@filter|Measure|Effect")

            results_list.append(formatted_data)
        
        final_results = pd.concat(results_list, axis=0)

        if save_results:
            final_results.to_csv(f'{self.output_directory}{self.sector}_{total_label}_decomposition.csv')

        return final_results
    
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

        # cols = ['@timeseries|Year', "@filter|Measure|Activity", "@filter|Measure|Intensity", \
        #              "@filter|Measure|Structure", "@filter|Measure|Effect"]

        # if "@filter|Measure|Activity" in data.columns:
        #     cols = cols.append("@filter|Measure|Activity")
        # print('cols:', cols)
        # data = data[cols]
        # for i, l in enumerate(loa):
        #     label = f"@filter|Subsector_Level_{i + 1}"
        #     print('label, l:', label, l)
        #     data[label] = l
        data["@filter|Sector"] = self.sector.capitalize()

        return data


class CalculateLMDI(LMDI):

    def __init__(self, sector, level_of_aggregation, lmdi_models, categories_dict, energy_types,
                 directory, output_directory, base_year=1985, end_year=2017):

        super().__init__(sector=sector, lmdi_models=lmdi_models, 
                         output_directory=output_directory, base_year=base_year, end_year=end_year)

        self.directory = directory
        self.output_directory = output_directory
        self.sector = sector
        self.level_of_aggregation = level_of_aggregation
        self.categories_dict = categories_dict
        self.base_year = base_year
        self.energy_types = energy_types  # could use energy_data.keys but need 'elec' and 'fuels' to come before the others

    @staticmethod
    def use_intersection(data, intersection_):
        
        if isinstance(data, pd.Series): 
            data_new = data.loc[intersection_]
        else:
            data_new = data.loc[intersection_, :]
            
        return data_new

    def ensure_same_indices(self, df1, df2):
        """Returns two dataframes with the same indices
        purpose: enable dataframe operations such as multiply and divide between the two dfs
        """        
        df1.index = df1.index.astype(int)
        df2.index = df2.index.astype(int)

        intersection_ = df1.index.intersection(df2.index)

        if len(intersection_) == 0: 
            raise ValueError('DataFrames do not contain any shared years')
        
        df1_new = self.use_intersection(df1, intersection_)
        df2_new = self.use_intersection(df2, intersection_)

        return df1_new, df2_new
    
    @staticmethod
    def get_elec(elec):
        """Add 'Energy_Type' column to electricity dataframe
        """        
        elec['Energy_Type'] = 'Electricity'
        print('Collected elec data')
        return elec

    @staticmethod
    def get_fuels(fuels):
        """Add 'Energy_Type' column to fuels dataframe
        """      
        fuels['Energy_Type'] = 'Fuels'
        print('Collected fuels data')
        return fuels

    @staticmethod
    def get_deliv(elec, fuels):
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
            # elec['Total'] = elec.sum(axis=1)
            fuels = energy_data['fuels']
            # fuels['Total'] = fuels.sum(axis=1)
            e_type_df = funcs[e_type](elec, fuels)
        elif e_type in ['elec', 'fuels']:
            data = energy_data[e_type]
            e_type_df = funcs[e_type](data)
        else:
            raise KeyError(f'{type} not in ["elec", "fuels", "deliv", "source", "source_adj"], user must define \
                               provide {type} data')
    
        return e_type_df

    def collect_energy_data(self, energy_data): 
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

        return energy_data_by_type
    
    @staticmethod
    def deep_get(dictionary, keys, default=None):
        return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

    @staticmethod
    def check_cols(dict_key, df, label):
        if dict_key not in df.columns:
            print(f'Warning: {dict_key} column not in {label} data')
            return False
        else:
            return True

    def build_col_list(self, data, key):
        if isinstance(data['activity'], pd.DataFrame):
            col_a = self.check_cols(key, data['activity'], label='activity')

        elif isinstance(data['activity'], dict):
            cols_a = []
            for a_type, a_df in data['activity'].items():
                col_ = self.check_cols(key, a_df, label=f'activity_{a_type}') 
                cols_a.append(col_)
            
            if False in cols_a:
                col_a = False
            else:
                col_a = True
        else:
            print(f"data['activity'] is type: {type(data['activity'])}")
            col_a = None

        for e in data['energy'].keys():
            print(f"{e}: data['energy'][e] {data['energy'][e]}")
            self.check_cols(key, data['energy'][e], label=f'energy_{e}')
        
        return col_a

    def agg_df(self, categories, results_dict, level_name, data_type, types):

        if not types:
            try:
                print("results_dict[list(results_dict.keys())[0]]keys", results_dict[list(results_dict.keys())[0]].keys())
                types = results_dict[list(results_dict.keys())[0]][data_type].keys()
            except KeyError:
                print("results_dict[key][data_type]:\n", results_dict[list(results_dict.keys())[0]][data_type])

        data_by_type = dict()
        for t in types:
            dfs = []
            for key, value in categories.items():
                if type(value) is dict: 
                    sub_data_ = results_dict[key][data_type][t]
                    data_ = self.create_total_column(sub_data_, level_name)[[level_name]]

                else:
                    data_ = results_dict[key][data_type][t]
                
                dfs.append(data_)

            df = reduce(lambda df1,df2: df1.merge(df2, how='outer', left_index=True, right_index=True), dfs)
            df = self.create_total_column(df, level_name)
            data_by_type[t] = df

        return data_by_type

    def build_nest(self, data, select_categories, results_dict, level1_name, level_name=None):
        if isinstance(select_categories, dict):
            for key, value in select_categories.items():
                print('BUILD NEST:', key)
                if type(value) is dict:
                    print(f'value for {key} is dictionary: with value:\n {value}')
                    yield from self.build_nest(data=data[key], select_categories=value, 
                                            results_dict=results_dict, level1_name=level1_name, 
                                            level_name=key)

                else: 
                    print(f'value for {key} is NOT dictionary: with value:\n {value}')
                    print('type data:', type(data))
                    if isinstance(data, dict):
                        print('data keys nest:', data.keys())
                    if 'activity' in data.keys() and 'energy' in data.keys():
                        col_a = self.build_col_list(data, key)
                        raw_energy_dict = data['energy']
                        activity_data = data['activity']
                    elif 'activity' in data[key].keys() and 'energy' in data[key].keys():
                        col_a = self.build_col_list(data[key], key)
                        raw_energy_dict = data[key]['energy']
                        activity_data = data[key]['activity']

                    if not level_name:
                        level_name = level1_name

                    activity_dict = dict()
                    if isinstance(activity_data, pd.DataFrame):
                        activity_dict[level_name] = activity_data

                    elif isinstance(activity_data, dict):
                        activity_dict = activity_data

                    energy = self.collect_energy_data(raw_energy_dict)
                    energy_data = dict()
                    for e in self.energy_types:
                        e_data = energy[e].drop('Energy_Type', errors='ignore', axis=1)
                        energy_data[e] = e_data
                    
                    data_dict = {'energy': energy_data, 'activity': activity_dict, 'level_total': level_name}

                    results_dict[f'{level_name}'] = data_dict 

        else:
            print('OOOPS')
            results_dict = results_dict
    
        yield results_dict

    def aggregate_levels(self, results_dict, categories):
        print('results dict:', results_dict)
        print('results dict keys:', results_dict.keys())

        agg_results = dict()
        levels = [sub['level_total'] for sub in results_dict.values()]
        for l in levels:
            results_dict_ = results_dict[l]
            agg_activity_results = self.agg_df(categories, results_dict_, l, 'activity', types=None)
            agg_energy_results = self.agg_df(categories, results_dict_, l, 'energy', types=self.energy_types)
            data_dict = {'activity': agg_activity_results, 'energy': agg_energy_results, 'level_total': l}

        agg_results[f'{l}'] = data_dict 
        return agg_results

    @staticmethod
    def create_total_column(df, total_label):
        df[total_label] = df.sum(axis=1).values
        return df 

    def calculate_breakout_lmdi(self, raw_results, final_results_list, level_of_aggregation, 
                                breakout, save_breakout, categories, lmdi_type):
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
            print('BREAK OUT KEY:', key)
            level_total = raw_results[key]['level_total']

            if level_total == level_of_aggregation[0]:
                print(f"level total {level_total} is the highest \
                        level of aggregation: {level_of_aggregation[0]}")
                continue
            else: 
                print(f"level total {level_total} is not equal to the highest \
                        level of aggregation: {level_of_aggregation[0]}")

            if level_of_aggregation[-1] == level_total:
                categories = self.deep_get(self.categories_dict, level_of_aggregation)
            else:
                categories = self.deep_get(self.categories_dict, '.'.join(level_of_aggregation) + f'.{key}')

            if not categories:
                print(f"{key} not in categories")
                continue
            
            energy = raw_results[key]['energy']
            activity_ = raw_results[key]['activity']

            # activity_ = {a_type: self.create_total_column(a_df, level_total) for (a_type, a_df) in activity_.items()}
            for a_type, a_df in activity_.items():
                if level_total not in a_df.columns:
                    raise KeyError(f'{level_total} not in {a_type} dataframe')
                    exit()

            if 'weather' in raw_results[key].keys():
                weather_data = raw_results[key]['weather']
            else:
                weather_data = None

            if level_of_aggregation[-1] == level_total:
                loa = [self.sector.capitalize()] + level_of_aggregation

            else:
                loa = [self.sector.capitalize()] + level_of_aggregation + [level_total]

            for e_type, energy_df in energy.items():
                if level_total not in energy_df.columns:
                    raise KeyError(f'{level_total} not in energy_df')
                    exit()

                lower_level_structure = self.calc_lower_level(categories, final_results_list, e_type)

                category_lmdi = self.call_lmdi(energy_df, activity_, 
                                               lower_level_structure, level_total,
                                               unit_conversion_factor=1, weather_data=weather_data, 
                                               save_results=save_breakout, 
                                               loa=loa, energy_type=e_type, lmdi_type=lmdi_type)
                structure_cols = [col for col in category_lmdi if 'Structure' in col]
                print('structure_cols:', structure_cols)
                category_lmdi['total_structure'] = category_lmdi[structure_cols].product(axis=1)
                category_lmdi["@filter|EnergyType"] = e_type
                category_lmdi['lower_level'] = level_total
                final_results_list.append(category_lmdi)
        return final_results_list

    def calc_lower_level(self, categories, final_fmt_results, e_type):
        if not final_fmt_results:
            return pd.DataFrame()
        else:
            print('CATEGORIES:\n', categories)
            final_fmt_results = pd.concat(final_fmt_results, axis=0)
            lower_level_structure_list = []
            for key, value in categories.items():
                lower_level = final_fmt_results[(final_fmt_results['lower_level'] == key) & (final_fmt_results["@filter|EnergyType"] == e_type) & (final_fmt_results["@filter|Measure|BaseYear"] == self.base_year) & (final_fmt_results["@filter|Model"] == 'Multiplicative')]
                lower_level = lower_level[['@timeseries|Year', 'total_structure']].set_index('@timeseries|Year')

                if not value:
                    print('KEY:', key)
                    lower_level_structure = pd.DataFrame(index=lower_level.index, columns=[f'lower_level_structure_{key}'])
                    lower_level_structure[f'lower_level_structure_{key}'] = 1
                elif type(value) is dict: 
                    try:
                        lower_level_structure = lower_level.rename(columns={'total_structure': f'lower_level_structure_{key}'})

                    except KeyError:
                        print(f"{key} dataframe does not contain total_structure column, \
                                columns are {lower_level.columns}")
                        continue
                lower_level_structure_list.append(lower_level_structure)

            if not lower_level_structure_list:
                return pd.DataFrame()
            else:
                lower_level_structure_df = reduce(lambda df1,df2: df1.merge(df2, how='outer', left_index=True, right_index=True), lower_level_structure_list)
                lower_level_structure_df = lower_level_structure_df.fillna(1)
                print('lower_level_structure_df:\n', lower_level_structure_df)
                return lower_level_structure_df

    def get_nested_lmdi(self, level_of_aggregation, raw_data, lmdi_type, calculate_lmdi=False, breakout=False,
                        save_breakout=False):
        """
        docstring

        TODO: 
            - Build in weather capabilities
        """
        categories = self.deep_get(self.categories_dict, level_of_aggregation)
        level_of_aggregation_ = level_of_aggregation.split(".")
        level1_name = level_of_aggregation_[-1]

        categories_pre_breakout = categories
        results_dict = dict()
        for results_dict in self.build_nest(data=raw_data, select_categories=categories, results_dict=results_dict,
                                            level1_name=level1_name):
            continue
        results_dict = self.aggregate_levels(results_dict, categories)
        print('results_dict.keys()', results_dict.keys())
        print('aGgg results:\n', results_dict)
        exit()
        final_fmt_results = []

        if breakout:
            final_fmt_results = self.calculate_breakout_lmdi(results_dict, final_fmt_results, level_of_aggregation_,
                                                             breakout, save_breakout, categories_pre_breakout, lmdi_type)

        total_activity_dfs = {}
        total_activty_dict = results_dict[level1_name]['activity']
        for activity_, total_activty_df in total_activty_dict.items():
            for key, value in categories.items():
                if type(value) is dict: 
                    total_activty_df[key] = results_dict[key]['activity'][key][key].values
            total_activty_df = total_activty_df.drop(level1_name, axis=1)
            total_activty_df[level1_name] = total_activty_df.sum(axis=1).values
            total_activity_dfs[activity_] = total_activty_df

        total_results_by_energy_type = dict()
        for e in self.energy_types:
            total_energy_df = results_dict[level1_name]['energy'][e]
            if 'weather' in results_dict[level1_name].keys():
                weather_data = results_dict[level1_name]['weather']
            else:
                weather_data = None

            for key, value in categories.items():
                if type(value) is dict: 
                    total_energy_df[key] = results_dict[key]['energy'][e][key].values
            total_energy_df = total_energy_df.drop(level1_name, axis=1)
            total_energy_df[level1_name] = total_energy_df.sum(axis=1).values

            lower_level_structure_df = self.calc_lower_level(categories, final_fmt_results, e)
            print('final lower level structure df:\n', lower_level_structure_df)

            if calculate_lmdi:
                loa = [self.sector.capitalize()] + level_of_aggregation_

                final_results = self.call_lmdi(total_energy_df, total_activity_dfs, 
                                               lower_level_structure_df, 
                                               total_label=level1_name,
                                               unit_conversion_factor=1,
                                               weather_data=weather_data, save_results=True, 
                                               loa=loa, energy_type=e, lmdi_type=lmdi_type)


                final_fmt_results.append(final_results)
                total_results_by_energy_type[e] = final_results

            else:
                total_results_by_energy_type[e] = {'activity': total_activity_dfs, 'energy': total_energy_df}

        if len(final_fmt_results) > 1: 
            final_results = pd.concat(final_fmt_results, axis=0, ignore_index=True, join='outer')
        else:
            final_results = final_fmt_results

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
    def logarithmic_average(x, y):
        """The logarithmic average of two positive numbers x and y
        """ 
        try:
            x = float(x)
            y = float(y)
        except TypeError:
            L = np.nan
            return L       
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

    def nominal_energy_intensity(self, energy_input_data, activity_input_data):
        energy_input_data, activity_input_data = self.ensure_same_indices(energy_input_data, activity_input_data)

        if isinstance(activity_input_data, pd.DataFrame):
            activity_width = activity_input_data.shape[1]
        elif isinstance(activity_input_data, pd.Series):
            activity_width = 1

        nominal_energy_intensity = energy_input_data.divide(activity_input_data.values.reshape(len(activity_input_data), \
                                                                                                activity_width)) 
                                                                                    #.multiply(unit_conversion_factor)
        return nominal_energy_intensity

    def prepare_lmdi_inputs(self, energy_input_data, activity_input_data, 
                            total_label, unit_conversion_factor=1):
        """Calculate the LMDI

        TODO: 
            - Account for weather factors when 

        Args:
            activity_input_data (dataframe or dictionary of dataframes): Activity input data for LMDI calculations
            energy_input_data (dataframe): Energy input data for LMDI calculations
            total_label (str): Name of the level of the level of aggregation representing the total of the current level. 
                                E.g. If categories are "Northeast", "South", etc, the total_label is "National"
            unit_conversion_factor (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """
 
        log_ratio_structure = []
        log_ratio_activity = []
        for activity, activty_data in activity_input_data.items():

            energy_input_data, activty_data = self.ensure_same_indices(energy_input_data, activty_data)
            activity_shares = self.calculate_shares(activty_data, total_label)

            # ln(ST_i/S0_i) --> S_i= Q_i / Q,  S_i is the activity share of sector i
            log_ratio_structure_activity = self.calculate_log_changes(activity_shares).rename(columns={col: 
                                                                                        f'{activity}_{col}' 
                                                                                        for col in 
                                                                                        activity_shares.columns}) 
            log_ratio_structure.append(log_ratio_structure_activity)

            # ln(QT/Q0)  --> Q = Q,  Q is the total industrial activity level
            log_ratio_activity_a = self.calculate_log_changes(activty_data[[total_label]])  
            log_ratio_activity.append(log_ratio_activity_a)

        if len(log_ratio_structure) > 1:
            log_ratio_structure = pd.concat(log_ratio_structure, axis=0, ignore_index=True, join='outer')
            log_ratio_activity = pd.concat(log_ratio_activity, axis=0, ignore_index=True, join='outer')
        else:
            log_ratio_structure = log_ratio_structure[0]
            log_ratio_activity = log_ratio_activity[0]

        energy_shares = self.calculate_shares(energy_input_data, total_label)

        # E is the total energy consumption in industry, Q is the total industrial activity level
        # ln(IT_i/I0_i) --> I_i = E_i / E,  I_i is the energy intensity of sector i
        log_ratio_intensity = self.calculate_log_changes(energy_shares) 

        log_ratios = {'activity': log_ratio_activity, 
                      'structure': log_ratio_structure, 
                      'intensity': log_ratio_intensity}

        return energy_input_data, energy_shares, log_ratios

    def call_lmdi(self, energy_input_data, activity_input_data, lower_level_structure, total_label, unit_conversion_factor,
                  weather_data, lmdi_type, save_results=False, loa=None, 
                  energy_type=None):
        
        energy_data, energy_shares, log_ratios = self.prepare_lmdi_inputs(energy_input_data, activity_input_data, 
                                                                          total_label, unit_conversion_factor=1)
        if not lower_level_structure.empty:
            log_ratios['lower_level_structure'] = lower_level_structure

        results = self.call_decomposition(energy_data, energy_shares, weather_data, 
                                          log_ratios, total_label, lmdi_type, loa, 
                                          save_results, energy_type)
        return results

        
if __name__ == '__main__':
    pass