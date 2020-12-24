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

    def __init__(self, sector, lmdi_models, output_directory, base_year,
                 end_year, primary_activity):
        self.sector = sector
        self.lmdi_models = lmdi_models
        self.output_directory = output_directory
        self.base_year = base_year
        self.end_year = end_year
        self.primary_activity = primary_activity

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
        change = dataset.divide(dataset.shift().values).astype(float)
        print('change:\n', change)
        print(change.dtypes)

        log_ratio = change.apply(lambda col: np.log(col), axis=1)

        log_ratio_df = pd.DataFrame(data=log_ratio, index=dataset.index, columns=dataset.columns)
        print("log_ratio_df:\n", log_ratio_df)

        return log_ratio_df

    @staticmethod
    def use_intersection(data, intersection_):
        """Select portion of dataframe where index is in intersection
        """

        if isinstance(data, pd.Series):
            data_new = data.loc[intersection_]
        else:
            data_new = data.loc[intersection_, :]

        return data_new

    def ensure_same_indices(self, df1, df2):
        """Returns two dataframes with the same indices
        purpose: enable dataframe operations such as multiply and divide between the two dfs
        """
        if df1.empty or df2.empty:
            raise ValueError('at least one dataframe is empty')

        df1.index = df1.index.astype(int)
        df1.index = df1.index.rename('Year')

        df2.index = df2.index.astype(int)
        df2.index = df2.index.rename('Year')

        intersection_ = df1.index.intersection(df2.index)

        if len(intersection_) == 0:
            raise ValueError('DataFrames do not contain any shared years')

        df1_new = self.use_intersection(df1, intersection_)
        df2_new = self.use_intersection(df2, intersection_)

        return df1_new, df2_new

    @staticmethod
    def sum_product(component_, weights, name):
        """Calculate the sum product of a log-ratio component and
        log mean divisia weights, rename column in the resulting dataframe
        """
        print('component:\n', component_)
        print('weights:\n', weights)
        if component_.shape[1] == 1:
            sum_product_ = component_.rename(columns={component_.columns[0]: name})
        else:
            sum_product_ = (component_.multiply(weights.values, axis='columns')).sum(axis=1)
            sum_product_ = sum_product_.to_frame(name=name)
        print('sum_product_:\n', sum_product_)
        return sum_product_

    def calc_component(self, log_ratio_component, weights, type_):
        """Calculate the component values from log_ratio components and log mean divisia weights
        """
        print('TYPE in calc component:', type_)
        if isinstance(log_ratio_component, pd.DataFrame):
            component = self.sum_product(log_ratio_component, weights, name=type_)
        elif isinstance(log_ratio_component, dict):
            comp_list = []
            if type_ == 'activity' and self.primary_activity:
                component = self.sum_product(log_ratio_component[self.primary_activity], weights, name=type_)
            elif type == 'intensity' and self.primary_activity:
                component = self.sum_product(log_ratio_component[self.primary_activity], weights, name=type_)

            else:
                for key, value in log_ratio_component.items():
                    if key == 'only_activity':
                        name_ = type_
                    else:
                        name_ = f'{key}_{type_}'
                    c = self.sum_product(value, weights, name=name_)
                    comp_list.append(c)
                if len(comp_list) > 1:
                    component = reduce(lambda df1,df2: df1.merge(df2, how='outer', left_index=True, right_index=True), comp_list)
                else:
                    component = comp_list[0]
        return component

    def calc_ASI(self, model, log_mean_divisia_weights_normalized,
                 log_ratios):
        """Collect activity, structure, and intensity components
        """
        activity = self.calc_component(log_ratios['activity'], log_mean_divisia_weights_normalized, type_='activity')
        intensity = self.calc_component(log_ratios['intensity'], log_mean_divisia_weights_normalized, type_='intensity')
        structure = self.calc_component(log_ratios['structure'], log_mean_divisia_weights_normalized, type_='structure')

        try:
            lower_level_structure = self.calc_component(log_ratios['lower_level_structure'], log_mean_divisia_weights_normalized, type_='lower_level_structure')
        except KeyError:
            lower_level_structure = pd.DataFrame()

        if self.primary_activity and log_ratios['activity'][self.primary_activity].shape[1] == 1:
            print('intensity:\n', intensity)
            print('structure:\n', structure)
            intensity = intensity[[f'{self.primary_activity}_intensity']].divide(structure.product(axis=1).values)

        ASI = {'activity': activity, 'structure': structure,
                'intensity': intensity}

        if not lower_level_structure.empty:
            ASI['lower_level_structure'] = lower_level_structure


        print('FINAL ASI KEYS:', ASI.keys())
        return ASI

    def call_decomposition(self, energy_data, energy_shares,
                           log_ratios, total_label, lmdi_type, loa,
                           energy_type):
        """Calculate Log Mean Divisia Index from input data"""
        results_list = []
        if isinstance(self.lmdi_models, list):
            pass
        else:
            self.lmdi_models = [self.lmdi_models]

        for model in self.lmdi_models:
            if model == 'additive':
                lmdi_type_ = lmdi_type
            else:
                lmdi_type_ = None
            print('energy data in call decomp:\n', energy_data)
            print('energy shares in call decomp:\n', energy_shares)
            model_ = self.LMDI_types[model](self.output_directory, energy_data, energy_shares,
                                            self.base_year, self.end_year, total_label, lmdi_type_)
            weights = model_.log_mean_divisia_weights()

            cols_to_drop_ = [col for col in weights.columns if col.endswith('_shift')]
            weights = weights.drop(cols_to_drop_, axis=1)

            print('log ratios:\n', log_ratios)
            components = self.calc_ASI(model, weights, log_ratios)

            results = model_.decomposition(components)

            fmt_loa = [l.replace(" ", "_") for l in loa]
            formatted_data, rename_dict = self.data_visualization(results, fmt_loa)
            formatted_data['@filter|Model'] = model.capitalize()
            formatted_data['@filter|EnergyType'] = energy_type

            data_to_plot = formatted_data[formatted_data["@filter|Measure|BaseYear"] == self.base_year]
            if '@timeseries|Year' not in data_to_plot.columns:
                data_to_plot = data_to_plot.rename(columns={'index': '@timeseries|Year'})
            model_.visualizations(data_to_plot, self.base_year, self.end_year,
                                  loa, model, energy_type, rename_dict)

            results_list.append(formatted_data)

        final_results = pd.concat(results_list, axis=0)
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
        -------
        csv

        """

        data = data.reset_index()
        print('data before rename:\n', data)

        rename_dict = dict()
        for col in data.columns:
            col_ = col.split('_')
            col_ = "".join([c.capitalize() for c in col_])
            if col == 'Year':
                col_ = '@timeseries|Year'
            elif col.startswith('@filter'):
                col_ = col
            else:
                col_ = "@filter|Measure|" + col_
            rename_dict[col] = col_
        print('rename dict:', rename_dict)
        data = data.rename(columns=rename_dict)

        data["@filter|Sector"] = self.sector.capitalize()
        print('data.columns', data.columns)
        return data, rename_dict


class CalculateLMDI(LMDI):

    def __init__(self, sector, level_of_aggregation, lmdi_models, categories_dict, energy_types,
                 directory, output_directory, primary_activity=None, base_year=1985,
                 end_year=2017, unit_conversion_factor=1):

        super().__init__(sector=sector, lmdi_models=lmdi_models, primary_activity=primary_activity,
                         output_directory=output_directory, base_year=base_year, end_year=end_year)

        self.directory = directory
        self.output_directory = output_directory
        self.sector = sector
        self.level_of_aggregation = level_of_aggregation
        self.categories_dict = categories_dict
        self.base_year = base_year
        self.energy_types = energy_types  # could use energy_data.keys but need 'elec' and 'fuels' to come before the others
        self.unit_conversion_factor = unit_conversion_factor

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
            fuels = energy_data['fuels']
            e_type_df = funcs[e_type](elec, fuels)
        elif e_type in ['elec', 'fuels']:
            data = energy_data[e_type]
            e_type_df = funcs[e_type](data)
        else:
            raise KeyError(f'{type} not in ["elec", "fuels", "deliv", "source", "source_adj"], user must define \
                               provide {type} data')

        return e_type_df

    def collect_energy_data(self, energy_data):
        """Calculate energy data for energy types in self.energy_types for which data is not provided.

        Examples
        --------
        >>>data_dict = {
            'All_Passenger': {
                'energy': {'deliv': passenger_based_energy_use},
                'activity': passenger_based_activity
                },
             'All_Freight': {
                'energy': {'deliv': freight_based_energy_use},
                'activity': freight_based_activity
                }
            }
        """

        provided_energy_data = list(energy_data.keys())

        if set(provided_energy_data) == set(self.energy_types):
            energy_data_by_type = energy_data
        elif 'elec' in energy_data and 'fuels' in energy_data:
            energy_data_by_type = dict()
            for type_ in self.energy_types:
                print('type_:', type_)
                try:
                    e_type_df = self.calculate_energy_data(type_, energy_data)
                    energy_data_by_type[type_] = e_type_df
                except KeyError as err:
                    print(err.args)
        else:
            raise ValueError('Warning: energy data dict not well defined')

        return energy_data_by_type

    @staticmethod
    def deep_get(dictionary, keys, default=None):
        """Get lower level portion of nested dictionary from path"""
        return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

    @staticmethod
    def check_cols(dict_key, df, label):
        """Check whether dataframe contains column matching dict_key"""
        print('DF in check cols:\n', df)
        if dict_key not in df.columns:
            print(f'Warning: {dict_key} column not in {label} data')
            return False
        else:
            return True

    def build_col_list(self, data, key):
        """Return bool indicating whether dataset contains data for given key"""
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

    def build_nest(self, data, select_categories, results_dict, level1_name, level_name=None):
        """Process and organize raw data"""
        print('Data at start of build nest:\n', data)
        print('Data keys at start of build nest:\n', data.keys())
        if isinstance(select_categories, dict):
            # level_energy_data = []
            # level_activity_data = []
            for key, value in select_categories.items():
                print('BUILD NEST:', key)
                print('level name:', level_name)
                if type(value) is dict:
                    print(f'value for {key} is dictionary: with value:\n {value}')
                    yield from self.build_nest(data=data[key], select_categories=value,
                                            results_dict=results_dict, level1_name=level1_name,
                                            level_name=key)

                else:
                    if not level_name:
                        level_name = level1_name

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

                    energy = self.collect_energy_data(raw_energy_dict)
                    energy_data = []
                    for e in self.energy_types:
                        print('e in build nest:\n', e)
                        e_data = energy[e].drop('Energy_Type', errors='ignore', axis=1)
                        e_data = e_data.apply(pd.to_numeric, errors='ignore', axis=1)

                        if 'Energy_Type' not in e_data.columns:
                            e_data['Energy_Type'] = e
                        energy_data.append(e_data)
                    energy_data = pd.concat(energy_data, axis=0)
                    print('energy data concat:\n', energy_data)

                    if isinstance(activity_data, pd.DataFrame):
                        print('activity data is dataframe')
                        activity_data['activity_type'] = 'only_activity'

                    elif isinstance(activity_data, dict):
                        print('activity data is dictionary')

                        activity_data_ = []
                        for a_type, a_df in activity_data.items():
                            a_df = a_df.apply(pd.to_numeric, errors='ignore', axis=1)
                            a_df.loc[:, 'activity_type'] = a_type
                            activity_data_.append(a_df)
                        activity_data = pd.concat(activity_data_, axis=0)

                    print('activity data pre drop:\n', activity_data)
                    activity_data = activity_data.drop('Total', axis=1, errors='ignore')
                    energy_data = energy_data.drop('Total', axis=1, errors='ignore')

                    print('energy_data:\n', energy_data)
                    print('activity_data:\n', activity_data)

                    if level_name in results_dict:
                        energy_data = self.merge_input_data([energy_data, results_dict[level_name]['energy']], 'Energy_Type')
                        activity_data = self.merge_input_data([activity_data, results_dict[level_name]['activity']], 'activity_type')

                    data_dict_ = {'energy': energy_data, 'activity': activity_data, 'level_total': level_name}
                    results_dict[level_name] = data_dict_

        else:
            results_dict = results_dict

        if not level_name:
            level_name = level1_name
        # print("results_dict[level_name]['energy'].columns:", results_dict[level_name]['energy'].columns)
        # print('select_categories.keys():', select_categories.keys())
        # if set(select_categories.keys()).issubset(results_dict[level_name]['energy'].columns):
        #     yield results_dict
        # else:
        if level_name in results_dict.keys():
            aggregate_activty = [results_dict[level_name]['activity']]
            print('aggregate_activity begin:\n', aggregate_activty)
            aggregate_energy = [results_dict[level_name]['energy']]
            print('aggregate_energy begin:\n', aggregate_energy)
        else:
            aggregate_activty = []
            aggregate_energy = []

        for key, value in select_categories.items():
            if key == np.nan:
                raise ValueError('SELECT CATEGORIES KEY IS NAN')

            print('key here:', key)
            print('results dict keys', results_dict.keys())
            print('results dict pre agg:\n', results_dict)
            try:
                lower_level_e = results_dict[key]['energy']
                lower_level_a = results_dict[key]['activity']
                if isinstance(value, dict):
                    base_col_a = 'activity_type'
                    if len(lower_level_a.columns.difference([base_col_a]).tolist()) > 1:
                        lower_level_a = self.create_total_column(lower_level_a, key)[[base_col_a, key]]

                    base_col_e = 'Energy_Type'
                    if len(lower_level_e.columns.difference([base_col_e]).tolist()) > 1:
                        lower_level_e = self.create_total_column(lower_level_e, key)[[base_col_e, key]]

                else:

                    if key in lower_level_e.columns:
                        lower_level_e = lower_level_e[['Energy_Type', key]]

                    if key in lower_level_a.columns:
                        lower_level_a = lower_level_a[['activity_type', key]]


                print('lower_level_e:\n', lower_level_e)
                print('lower_level_a:\n', lower_level_a)
                aggregate_activty.append(lower_level_a)
                aggregate_energy.append(lower_level_e)

            except KeyError:
                print(f'key: {key} failed on level : {level_name}')
                continue

            e_df = self.merge_input_data(aggregate_energy, 'Energy_Type')
            e_df = self.create_total_column(e_df, level_name)
            print('e_df:\n', e_df)
            agg_a_df = self.merge_input_data(aggregate_activty, 'activity_type')
            agg_a_df = self.create_total_column(agg_a_df, level_name)
            print('a_df:\n', agg_a_df)

            data_dict = {'energy': e_df, 'activity': agg_a_df, 'level_total': level_name}
            print('data_dict')
            results_dict[f'{level_name}'] = data_dict
            yield results_dict

    @staticmethod
    def int_index(df):
        """Ensure df index is Year of type int"""
        if 'Year' in df.columns:
            df = df.set_index('Year')
        else:
            df.index.name = 'Year'

        df.index = df.index.astype(int)
        return df

    def merge_input_data(self, list_dfs, second_index):
        """Merge dataframes of same variable type"""

        list_dfs = [self.int_index(l) for l in list_dfs]

        if np.array([list(df.columns) == list(list_dfs[0].columns) for df in list_dfs]).all():
            print('dataframes have the same columns')
            return list_dfs[0]
        else:
            list_dfs = [l.reset_index() for l in list_dfs]
            df = reduce(lambda df1,df2: df1.merge(df2[list(df2.columns.difference(df1.columns)) + \
                                                    ['Year', second_index]], how='outer', on=['Year', second_index]), list_dfs).set_index('Year')
            return df


    @staticmethod
    def create_total_column(df, total_label):
        """Create column from sum of all other columns, name with name of level of aggregation"""
        print('df dtypes:\n', df.dtypes)
        df_drop_str = df.select_dtypes(exclude='object')
        print("df_drop_str.columns:", df_drop_str.columns)
        if len(df_drop_str.columns.tolist()) > 1:
            df[total_label] = df.drop(total_label, axis=1, errors='ignore').sum(axis=1, numeric_only=True)
        elif len(df_drop_str.columns.tolist()) == 1:
            df[total_label] = df[df_drop_str.columns]
            print("df after duplicating column:\n", df)
        return df

    def order_categories(self, level_of_aggregation, raw_results):
        """Order categories so that lower levels are calculated prior to current level of aggregation.
        This ordering ensures that lower level structure is passed to higher level.
        """
        categories = self.deep_get(self.categories_dict, '.'.join(level_of_aggregation))
        categories_list = []
        print('raw_results.keys():', raw_results.keys())
        for key in raw_results.keys():
            if key in categories.keys():
                categories_list.append(key)

        for key in raw_results.keys():
            if key not in categories.keys():
                categories_list.append(key)
        return categories_list

    def calculate_breakout_lmdi(self, raw_results, final_results_list, level_of_aggregation,
                                breakout, categories, lmdi_type):
        """If breakout=True, calculate LMDI for each lower aggregation level contained in raw_results.

        Args:
            raw_results (dictionary): Built "nest" of dictionaries containing input data for LMDI calculations
            final_results_list (list): list to which calculate_breakout_lmdi appends LMDI results

        Returns:
            final_results_list [list]: list of LMDI results dataframes

        TODO: Lower level Total structure (product of each structure index for multiplicative) and component
        intensity index (index of aggregate intensity divided by total strucutre) need to be passed to higher level
        """
        categories_list = self.order_categories(level_of_aggregation, raw_results)
        print('CATEGORIES LIST:', categories_list)
        for key in categories_list:
            print('BREAK OUT KEY:', key)
            level_total = raw_results[key]['level_total']

            if level_of_aggregation[-1] == level_total:
                loa = [self.sector.capitalize()] + level_of_aggregation
                categories = self.deep_get(self.categories_dict, '.'.join(level_of_aggregation))

            else:
                loa = [self.sector.capitalize()] + level_of_aggregation + [level_total]
                categories = self.deep_get(self.categories_dict, '.'.join(level_of_aggregation) + f'.{key}')

            if not categories:
                print(f"{key} not in categories")
                continue
            print('CATEGORIES:\n', categories)
            activity_ = dict()
            total_activty_df = raw_results[key]['activity']

            for a_type in total_activty_df['activity_type'].unique():
                a_df = total_activty_df[total_activty_df['activity_type'] == a_type].drop('activity_type', axis=1)

                if level_total not in a_df.columns:
                    raise KeyError(f'{level_total} not in {a_type} dataframe')
                    exit()
                else:
                    activity_[a_type] = a_df


            total_energy_df = raw_results[key]['energy']
            for e_type in self.energy_types:
                energy_df = total_energy_df[total_energy_df['Energy_Type'] == e_type].drop('Energy_Type', axis=1)
                if level_total not in energy_df.columns:
                    raise KeyError(f'{level_total} not in energy_df')
                    exit()

                if 'weather' in raw_results[key].keys():
                    all_weather_data = raw_results[key]['weather']
                    weather_data = all_weather_data[all_weather_data['Energy_Type'] == e_type].drop('Energy_Type', axis=1)
                else:
                    weather_data = None

                lower_level_structure_df, lower_level_intensity_df = self.calc_lower_level(categories, final_results_list, e_type)

                category_lmdi = self.call_lmdi(energy_df, activity_,
                                               lower_level_structure_df, lower_level_intensity_df,
                                               level_total,
                                               unit_conversion_factor=1, weather_data=weather_data,
                                               loa=loa, energy_type=e_type, lmdi_type=lmdi_type)
                structure_cols = [col for col in category_lmdi if 'Structure' in col]
                print('structure_cols:', structure_cols)
                category_lmdi['total_structure'] = category_lmdi[structure_cols].product(axis=1)
                category_lmdi["@filter|EnergyType"] = e_type
                category_lmdi['lower_level'] = level_total
                final_results_list.append(category_lmdi)
        return final_results_list

    def calc_lower_level(self, categories, final_fmt_results, e_type):
        """Calculate decomposition for lower levels of aggregation
        """
        print('final_fmt_results in calc lower level:\n', final_fmt_results)
        if not final_fmt_results:
            return pd.DataFrame(), pd.DataFrame()
        else:
            print('CATEGORIES in lowr level:\n', categories.keys())
            final_fmt_results = pd.concat(final_fmt_results, axis=0)

            if 'lower_level' not in final_fmt_results.columns:
                return pd.DataFrame(), pd.DataFrame()

            lower_level_structure_list = []
            lower_level_intensity_list = []

            for key, value in categories.items():
                print('LOWER LEVEL for key:', key)
                print('final_fmt_results:\n', final_fmt_results)
                lower_level = final_fmt_results[(final_fmt_results['lower_level'] == key) & (final_fmt_results["@filter|EnergyType"] == e_type) & (final_fmt_results["@filter|Measure|BaseYear"] == self.base_year) & (final_fmt_results["@filter|Model"] == 'Multiplicative')]
                print('lower_level:\n', lower_level)

                if not value:
                    print('KEY:', key)
                    lower_level_structure = pd.DataFrame(index=lower_level.index, columns=[f'lower_level_structure_{key}'])
                    lower_level_structure[f'lower_level_structure_{key}'] = 1

                    lower_level_intensity = pd.DataFrame(index=lower_level.index, columns=[key])

                elif type(value) is dict:
                    try:
                        lower_level_structure = lower_level[['@timeseries|Year', 'total_structure']].set_index('@timeseries|Year')
                        print('lower_level 2:\n', lower_level_structure)
                        lower_level_structure = lower_level_structure.rename(columns={'total_structure': f'lower_level_structure_{key}'})

                    except KeyError:
                        print(f"{key} dataframe does not contain total_structure column, \
                                columns are {lower_level.columns}")
                        continue

                    try:
                        lower_level_intensity = lower_level[['@timeseries|Year', '@filter|Measure|Intensity']].set_index('@timeseries|Year')
                        lower_level_intensity = lower_level_intensity.rename(columns={'@filter|Measure|Intensity': key})
                    except KeyError:
                        print(f"{key} dataframe does not contain @filter|Measure|Intensity column, \
                                columns are {lower_level.columns}")
                        continue

                lower_level_structure_list.append(lower_level_structure)
                lower_level_intensity_list.append(lower_level_intensity)

            if not lower_level_structure_list:
                lower_level_structure_df = pd.DataFrame()
            else:
                lower_level_structure_df = reduce(lambda df1,df2: df1.merge(df2, how='outer', left_index=True, right_index=True), lower_level_structure_list)
                lower_level_structure_df = lower_level_structure_df.fillna(1)
                print('lower_level_structure_df:\n', lower_level_structure_df)

            if not lower_level_intensity_list:
                lower_level_intensity_df =  pd.DataFrame()
            else:
                lower_level_intensity_df = reduce(lambda df1,df2: df1.merge(df2, how='outer', left_index=True, right_index=True), lower_level_intensity_list)
            print('lower_level_intensity_df:\n', lower_level_intensity_df)
            return lower_level_structure_df, lower_level_intensity_df

    def get_nested_lmdi(self, level_of_aggregation, raw_data, lmdi_type, calculate_lmdi=False, breakout=False):
        """
        Collect LMDI decomposition according to user specifications
        Todo-- Build in weather capabilities
        """
        print('categories:', self.categories_dict)
        categories = self.deep_get(self.categories_dict, level_of_aggregation)

        if len(self.categories_dict) == 1 and not categories:
            categories = self.categories_dict

        print('categories:', categories)
        print('type categories:', categories)
        if isinstance(categories, dict):
            for k, v in categories.items():
                print('categories k:', k)
                print('categories v:', v)
                print("type(v):", type(v))

        data = reduce(lambda d, key: d.get(key, d) if isinstance(d, dict) else d, level_of_aggregation.split("."), raw_data)

        level_of_aggregation_ = level_of_aggregation.split(".")
        level1_name = level_of_aggregation_[-1]

        categories_pre_breakout = categories
        results_dict = dict()
        for results_dict in self.build_nest(data=data, select_categories=categories, results_dict=results_dict,
                                            level1_name=level1_name):
            continue

        print('results_dict.keys()', results_dict.keys())
        print('aGgg results:\n', results_dict)
        final_fmt_results = []

        if breakout:
            final_fmt_results = self.calculate_breakout_lmdi(results_dict, final_fmt_results, level_of_aggregation_,
                                                             breakout, categories_pre_breakout, lmdi_type)

        total_activity_dfs = dict()
        total_activty_df = results_dict[level1_name]['activity']
        for a_type in total_activty_df['activity_type'].unique():
            total_activity_dfs[a_type] = total_activty_df[total_activty_df['activity_type'] == a_type].drop('activity_type', axis=1)

        total_results_by_energy_type = dict()
        total_energy_df = results_dict[level1_name]['energy']
        print('total_energy_df:\n', total_energy_df)
        print("total_energy_df['Energy_Type'].unique()", total_energy_df['Energy_Type'].unique())
        for e in self.energy_types:
            print('energy type (e):', e)
            energy_data = total_energy_df[total_energy_df['Energy_Type'] == e].drop('Energy_Type', axis=1)

            if 'weather' in results_dict[level1_name].keys():
                weather_data = results_dict[level1_name]['weather']
            else:
                weather_data = None

            lower_level_structure_df, lower_level_intensity_df = self.calc_lower_level(categories, final_fmt_results, e)
            print('final lower level structure df:\n', lower_level_structure_df)

            if calculate_lmdi:
                loa = [self.sector.capitalize()] + level_of_aggregation_

                print('total_activity_dfs:\n', total_activity_dfs)
                print('energy_data\n', energy_data)

                final_results = self.call_lmdi(energy_data, total_activity_dfs,
                                               lower_level_structure_df,
                                               lower_level_intensity_df,
                                               total_label=level1_name,
                                               unit_conversion_factor=1,
                                               weather_data=weather_data,
                                               loa=loa, energy_type=e, lmdi_type=lmdi_type)


                final_fmt_results.append(final_results)
                total_results_by_energy_type[e] = final_results

            else:
                total_results_by_energy_type[e] = {'activity': total_activity_dfs, 'energy': total_energy_df, 'weather': weather_data}

        if len(final_fmt_results) > 1:
            final_results = pd.concat(final_fmt_results, axis=0, ignore_index=True, join='outer')
        else:
            final_results = final_fmt_results

        return total_results_by_energy_type, final_results

    @staticmethod
    def select_value(dataframe, base_row, base_column):
        """Select value from dataframe as in Excel's @index function"""
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
        dataset[total_label] = dataset[total_label].replace(0, np.nan)
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
        """Calculate nominal energy intensity (i.e. energy divided by activity)"""
        energy_input_data, activity_input_data = self.ensure_same_indices(energy_input_data, activity_input_data)

        energy_input_data = energy_input_data.T.drop_duplicates().T
        activity_input_data = activity_input_data.T.drop_duplicates().T

        if isinstance(activity_input_data, pd.DataFrame):
            activity_width = activity_input_data.shape[1]
        elif isinstance(activity_input_data, pd.Series):
            activity_width = 1

        energy_width = energy_input_data.shape[1]
        if energy_width == activity_width:
            nominal_energy_intensity = energy_input_data.divide(activity_input_data.values.reshape(len(activity_input_data), \
                                                                                                    activity_width)).multiply(self.unit_conversion_factor)
        elif energy_width == 1 and activity_width > 1:
            nominal_energy_intensity = np.divide(np.tile(energy_input_data.values, activity_width), activity_input_data.values)
            nominal_energy_intensity = pd.DataFrame(nominal_energy_intensity, index=energy_input_data.index, columns=activity_input_data.columns)
            nominal_energy_intensity = nominal_energy_intensity.multiply(self.unit_conversion_factor)

        return nominal_energy_intensity

    def prepare_lmdi_inputs(self, energy_input_data, activity_input_data, lower_level_intensity_df,
                            total_label, weather_data, unit_conversion_factor=1):
        """Calculate the LMDI inputs (collect log ratio components)

        Args:
            activity_input_data (dataframe or dictionary of dataframes): Activity input data for LMDI calculations
            energy_input_data (dataframe): Energy input data for LMDI calculations
            total_label (str): Name of the level of the level of aggregation representing the total of the current level.
                                E.g. If categories are "Northeast", "South", etc, the total_label is "National"
            unit_conversion_factor (int, optional): [description]. Defaults to 1.
        """

        log_ratio_structure = dict()
        log_ratio_activity = dict()
        log_ratio_intensity = dict()
        for activity, activity_data in activity_input_data.items():

            energy_input_data, activity_data = self.ensure_same_indices(energy_input_data, activity_data)
            activity_shares = self.calculate_shares(activity_data, total_label)

            # ln(ST_i/S0_i) --> S_i= Q_i / Q,  S_i is the activity share of sector i
            log_ratio_structure_activity = self.calculate_log_changes(activity_shares).rename(columns={col:
                                                                                        f'{activity}_{col}'
                                                                                        for col in
                                                                                        activity_shares.columns})
            log_ratio_structure[activity] = log_ratio_structure_activity

            # ln(QT/Q0)  --> Q = Q,  Q is the total industrial activity level
            log_ratio_activity_a = self.calculate_log_changes(activity_data[[total_label]])
            log_ratio_activity[activity] = log_ratio_activity_a

            # E is the total energy consumption in industry, Q is the total industrial activity level
            # ln(IT_i/I0_i) --> I_i = E_i / Q_i,  I_i is the energy intensity of sector i
            nom_intensity = self.nominal_energy_intensity(energy_input_data, activity_data).drop(total_label, axis=1, errors='ignore')
            print('nom_intensity_index:\n', nom_intensity)

            nom_intensity_base = nom_intensity.loc[self.base_year, :]
            intensity_index = nom_intensity.divide(np.tile(nom_intensity_base, (len(nom_intensity), 1)))
            print('intensity_index:\n', intensity_index)

            if not lower_level_intensity_df.empty:
                lower_level_intensity_df = lower_level_intensity_df.fillna(intensity_index)
                log_ratio_intensity_a = self.calculate_log_changes(lower_level_intensity_df)
                print('intensity is from lower level')

            else:
                log_ratio_intensity_a = self.calculate_log_changes(intensity_index)
                print('intensity is nominal intensity from current level')

            log_ratio_intensity[activity] = log_ratio_intensity_a

        if weather_data:
            log_changes_weather = self.calculate_log_changes(weather_data)
            log_ratio_structure['weather'] = log_changes_weather

        energy_shares = self.calculate_shares(energy_input_data, total_label)

        log_ratios = {'activity': log_ratio_activity,
                      'structure': log_ratio_structure,
                      'intensity': log_ratio_intensity}

        return energy_input_data, energy_shares, log_ratios

    def call_lmdi(self, energy_input_data, activity_input_data,
                  lower_level_structure, lower_level_intensity_df,
                  total_label, unit_conversion_factor,
                  weather_data, lmdi_type, loa=None,
                  energy_type=None):
        """Prepare LMDI inputs and pass them to call_decomposition method.

        Returns:
            results (dataframe): formatted LMDI results
        """
        energy_data, energy_shares, log_ratios = self.prepare_lmdi_inputs(energy_input_data, activity_input_data,
                                                                                           lower_level_intensity_df,
                                                                                           total_label, weather_data,
                                                                                           unit_conversion_factor=1)
        if not lower_level_structure.empty:
            print('lower_level_structure:\n', lower_level_structure)
            lower_level_structure = self.calculate_log_changes(lower_level_structure)
            log_ratios['lower_level_structure'] = lower_level_structure
        print('log_ratios:\n', log_ratios)
        results = self.call_decomposition(energy_data, energy_shares,
                                          log_ratios, total_label, lmdi_type, loa,
                                          energy_type)
        return results


if __name__ == '__main__':
    pass




# if len(log_ratio_structure) > 1:
#     log_ratio_structure = pd.concat(log_ratio_structure, axis=0, ignore_index=True, join='outer')
#     log_ratio_activity = pd.concat(log_ratio_activity, axis=0, ignore_index=True, join='outer')
#     log_ratio_intensity = pd.concat(log_ratio_intensity, axis=0, ignore_index=True, join='outer')

# else:
#     log_ratio_structure = log_ratio_structure[0]
#     log_ratio_activity = log_ratio_activity[0]
#     log_ratio_intensity = log_ratio_intensity[0]
