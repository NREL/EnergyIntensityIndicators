import pandas as pd
import numpy as np
from functools import reduce

from EnergyIntensityIndicators.pull_eia_api import GetEIAData
from EnergyIntensityIndicators.multiplicative_lmdi import MultiplicativeLMDI
from EnergyIntensityIndicators.additive_lmdi import AdditiveLMDI
from EnergyIntensityIndicators.utilites import dataframe_utilities as df_utils


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

    def sum_product(self, component_, weights, name):
        """Calculate the sum product of a log-ratio component and
        log mean divisia weights, rename column in the resulting dataframe
        """
        if component_.shape[1] == 1 or weights.empty:
            sum_product_ = component_.rename(columns={component_.columns[0]:
                                                      name})
        else:
            component_, weights = df_utils.ensure_same_indices(component_,
                                                               weights)
            sum_product_ = (component_.multiply(weights.values,
                                                axis='index')).sum(axis=1)
            sum_product_ = sum_product_.to_frame(name=name)
        return sum_product_

    def calc_component(self, log_ratio_component, weights, type_,
                       primary_activity):
        """Calculate the component values from log_ratio components
        and log mean divisia weights
        """

        if isinstance(log_ratio_component, pd.DataFrame):
            component = self.sum_product(log_ratio_component,
                                         weights, name=type_)
        elif isinstance(log_ratio_component, dict):
            comp_list = []

            if type_ == 'activity' and primary_activity is not None:
                component = self.sum_product(
                                        log_ratio_component[primary_activity],
                                        weights, name=type_)
            elif type_ == 'intensity' and primary_activity is not None:
                component = self.sum_product(
                                        log_ratio_component[primary_activity],
                                        weights, name=type_)

            else:
                for key, value in log_ratio_component.items():
                    if key == 'only_activity':
                        name_ = type_
                    else:
                        name_ = f'{key}_{type_}'
                    c = self.sum_product(value, weights, name=name_)
                    comp_list.append(c)
                if len(comp_list) > 1:
                    print('columns component df', [l_.columns
                                                   for l_ in comp_list])
                    component = df_utils.merge_df_list(comp_list)
                else:
                    component = comp_list[0]
        return component

    def calc_ASI(self, model, log_mean_divisia_weights_normalized,
                 log_ratios, total_label):
        """Collect activity, structure, and intensity components
        """
        if isinstance(self.primary_activity, dict):
            if total_label in self.primary_activity.keys():
                primary_activity = self.primary_activity[total_label]
            else:
                primary_activity = None
        elif isinstance(self.primary_activity, str):
            primary_activity = self.primary_activity
        else:
            primary_activity = None

        activity = self.calc_component(log_ratios['activity'],
                                       log_mean_divisia_weights_normalized,
                                       type_='activity',
                                       primary_activity=primary_activity)
        intensity = self.calc_component(log_ratios['intensity'],
                                        log_mean_divisia_weights_normalized,
                                        type_='intensity',
                                        primary_activity=primary_activity)
        structure = self.calc_component(log_ratios['structure'],
                                        log_mean_divisia_weights_normalized,
                                        type_='structure',
                                        primary_activity=primary_activity)

        try:
            lower_level_structure = self.calc_component(
                                        log_ratios['lower_level_structure'],
                                        log_mean_divisia_weights_normalized,
                                        type_='lower_level_structure',
                                        primary_activity=primary_activity)
        except KeyError:
            lower_level_structure = pd.DataFrame()

        if primary_activity is not None and log_ratios['activity'][
                                            primary_activity].shape[1] == 1:

            if model == 'additive':
                intensity = intensity.divide(structure.sum(axis=1),
                                             axis='index')

            elif model == 'multiplicative':
                intensity = intensity.divide(structure.product(axis=1),
                                             axis='index')
        ASI = {'activity': activity, 'structure': structure,
               'intensity': intensity}

        if not lower_level_structure.empty:
            ASI['lower_level_structure'] = lower_level_structure
        
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

            model_ = self.LMDI_types[model](self.output_directory, energy_data,
                                            energy_shares, self.base_year,
                                            self.end_year, total_label,
                                            lmdi_type_)
            weights = model_.log_mean_divisia_weights()
        
            cols_to_drop_ = [col for col in weights.columns if col.endswith('_shift')]
            weights = weights.drop(cols_to_drop_, axis=1)

            components = self.calc_ASI(model, weights, log_ratios, total_label)

            results = model_.decomposition(components)

            fmt_loa = [l.replace(" ", "_") for l in loa]
            results['@filter|Model'] = model.capitalize()
            results['@filter|EnergyType'] = energy_type
            data_to_plot, rename_dict = self.data_visualization(results, 
                                        fmt_loa, energy_type, total_label)

            if '@timeseries|Year' not in data_to_plot.columns:
                data_to_plot = data_to_plot.rename(columns={'index': '@timeseries|Year'})
            model_.visualizations(data_to_plot, self.base_year, self.end_year, 
                                  loa, model, energy_type, rename_dict)

            results_list.append(data_to_plot)
        
        final_results = pd.concat(results_list, axis=0, sort=False)

        return final_results
    
    def data_visualization(self, data, loa, energy_type, total_label):
        """Format data for proper visualization
        
        The following data types have been proposed (an ellipsis ... indicates 
        an optional parameter):

            @filter|Category1|...Category2|...|Label#units

            A list of options that can be grouped by 1 or more categories.
            @weight|Category1|...Category2|...|Label#units

            A weighted value to use with a matching filter (must match filter
            label and categories).
            @scenario|Label

            A list of options that are completely separate from each other, i.e.
            they will not be seen on the same chart at the same time.
            The options come from the unique values in the scenario column.
            @timeseries|Label

            A list of options that can be used to make a time series, e.g. a 
            list of years.
            @geography|Label

            A list of geography names, e.g. states, counties, cities, that can be
             used in charts or a choropleth map.
            @geoid

            The column values are geography IDs that can be used in a 
            choropleth map.
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

        data = data[(data["@filter|EnergyType"] == energy_type) &
                    (data["@filter|Measure|BaseYear"] == self.base_year)]
        if 'lower_level' in data.columns:
            data = data[data['lower_level'] == total_label]

        data.index.name = 'Year'
        data = data.reset_index()

        rename_dict = dict()
        for col in data.columns:
            col_ = col.split('_')
            col_ = "".join([c.capitalize() for c in col_])
            if col == 'Year':
                col_ = '@timeseries|Year'
            elif col.startswith('@filter'):
                col_ = col
            else:
                extensionsToCheck = ('Intensity', 'Structure', 'Activity',
                                     'Effect')
                if col_.endswith(extensionsToCheck):
                    col_ = "@filter|Measure|" + col_

            rename_dict[col] = col_
        data = data.rename(columns=rename_dict)

        cols_to_transfer = list(rename_dict.values())
        if 'total_structure' in data.columns:
            cols_to_transfer.append('total_structure')
        if 'lower_level' in data.columns:
            cols_to_transfer.append('lower_level')

        data = data[set(cols_to_transfer)]

        data["@filter|Sector"] = self.sector.capitalize()
        return data, rename_dict


class CalculateLMDI(LMDI):

    def __init__(self, sector, level_of_aggregation, lmdi_models,
                 categories_dict, energy_types, directory,
                 output_directory, primary_activity=None,
                 base_year=1985, end_year=2017,
                 unit_conversion_factor=1,
                 weather_activity=None):

        super().__init__(sector=sector, lmdi_models=lmdi_models,
                         primary_activity=primary_activity,
                         output_directory=output_directory,
                         base_year=base_year, end_year=end_year)

        self.directory = directory
        self.output_directory = output_directory
        self.sector = sector
        self.level_of_aggregation = level_of_aggregation
        self.categories_dict = categories_dict
        self.base_year = base_year
        self.energy_types = energy_types  # could use energy_data.keys
                                          # but need 'elec' and 'fuels'
                                          #  to come before the others
        self.unit_conversion_factor = unit_conversion_factor
        self.weather_activity = weather_activity

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
        """Calculate delivered energy by adding electricity and
        fuels then add 'Energy_Type' column to the resulting
        delivered energy dataframe
        """
        delivered = elec.add(fuels, axis='index')
        delivered['Energy_Type'] = 'Delivered'
        print('Calculated deliv data')
        return delivered

    def get_source(self, elec, fuels):
        """Call conversion factors method from GetEIAData,
        calculate source energy from conversion_factors,
        electricity and fuels dataframe, then add
        'Energy-Type' column to the resulting source
        energy dataframe
        """
        if self.sector == 'commercial':
            conversion_factors = GetEIAData(self.sector).conversion_factors(
                include_utility_sector_efficiency=True)
        else:
            conversion_factors = GetEIAData(self.sector).conversion_factors()

        if conversion_factors is None:
            return None

        conversion_factors.index = conversion_factors.index.astype(int)

        conversion_factors, elec = df_utils.ensure_same_indices(
                                            conversion_factors, elec)
        source_electricity = elec.drop('Energy_Type', 
                                       axis=1).multiply(conversion_factors,
                                                        axis='index')
                                                        # Column A
        total_source = source_electricity.add(fuels.drop('Energy_Type',
                                                         axis=1), axis='index')
        total_source['Energy_Type'] = 'Source'
        total_source = total_source.drop(
                                'selected site-source conversion factor',
                                axis=1, errors='ignore')
        print('Calculated source data')
        return total_source
    
    def get_source_adj(self, elec, fuels):
        """Call conversion factors method from GetEIAData,
        calculate source adjusted energy from conversion_factors,
        electricity and fuels dataframe, then add 'Energy-Type' column
        to the resulting source adjusted energy dataframe
        """        
        conversion_factors = GetEIAData(self.sector).conversion_factors(
                                        include_utility_sector_efficiency=True)
            
        if conversion_factors is None:
            return None

        conversion_factors.index = conversion_factors.index.astype(int)

        conversion_factors, elec = df_utils.ensure_same_indices(
                                                conversion_factors, elec)

        source_electricity_adj = elec.drop('Energy_Type',
                                           axis=1).multiply(conversion_factors,
                                                            axis='index')
                                                            # Column M
        source_adj = source_electricity_adj.add(fuels.drop('Energy_Type',
                                                axis=1), axis='index')
        source_adj['Energy_Type'] = 'Source_Adj'
        source_adj = source_adj.drop('selected site-source conversion factor',
                                     axis=1, errors='ignore')

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
            elec = elec.drop('electricity_weather_factor',
                             axis=1, errors='ignore')  # for weather factors
            fuels = energy_data['fuels']
            elec, fuels = df_utils.ensure_same_indices(elec, fuels)
            e_type_df = funcs[e_type](elec, fuels)

        elif e_type in ['elec', 'fuels']:
            data = energy_data[e_type]
            e_type_df = funcs[e_type](data)

        else:
            try:
                e_type_df = energy_data[e_type]
                e_type_df['Energy_Type'] = e_type
            except KeyError:
                return None

            print(f'{e_type} not in ["elec", "fuels", "deliv",\
                  "source", "source_adj"], user must define \
                   provide {e_type} data')
    
        return e_type_df

    def collect_energy_data(self, energy_data): 
        """Calculate energy data for energy types in
        self.energy_types for which data is not provided

        Example data: 

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
            for type_ in self.energy_types:
                try:
                    e_type_df = self.calculate_energy_data(type_, energy_data)
                    energy_data_by_type[type_] = e_type_df
                except KeyError as err:
                    print(err.args)
        else:
            energy_data_by_type = energy_data
            for e in self.energy_types:
                if e not in provided_energy_data:
                    energy_data_by_type[e] = None
                else:
                    try:
                        e_type_df = energy_data_by_type[e]
                        e_type_df['Energy_Type'] = e
                        energy_data_by_type[e] = e_type_df
                    except Exception as error:
                        print(f'Error {error} for energy type {e} in {self.sector}')
                        continue
                
        return energy_data_by_type
    
    @staticmethod
    def deep_get(dictionary, keys, default=None):
        """Get lower level portion of nested dictionary from path"""
        return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
                      keys.split("."), dictionary)

    @staticmethod
    def process_type_data(data_category, data, d_type_list):
        all_category_data = []
        for t in d_type_list:
            if data[t] is not None:
                t_data = data[t]
                t_data = t_data.drop(data_category, errors='ignore', axis=1)
                t_data = t_data.apply(pd.to_numeric, errors='ignore', axis=1)

                if data_category not in t_data.columns:
                    t_data[data_category] = t

                all_category_data.append(t_data)
            else:
                continue
        
        all_category_data = pd.concat(all_category_data, axis=0, sort=False)
        all_category_data = all_category_data.drop('Total',
                                                   errors='ignore',
                                                   axis=1)
        return all_category_data

    def process_results_dict(self, data, select_categories, results_dict,
                             level_name, key):

        if 'activity' in data.keys() and 'energy' in data.keys():
            level_data = data
        elif 'activity' in data[key].keys() and 'energy' in data[key].keys():
            level_data = data[key]
        else:
            return None

        # col_a = self.build_col_list(level_data, key)
        raw_energy_dict = level_data['energy']
        activity_data = level_data['activity']

        if 'weather_factors' in level_data.keys():
            weather_data_ = level_data['weather_factors']
            weather = True
        else:
            weather = False

        energy = self.collect_energy_data(raw_energy_dict)
        energy_data = self.process_type_data('Energy_Type', energy, 
                                             d_type_list=self.energy_types)

        if isinstance(activity_data, pd.DataFrame):
            activity_data['activity_type'] = 'only_activity'
        elif isinstance(activity_data, dict):
            activity_data = self.process_type_data('activity_type',
                                                   activity_data,
                                                   d_type_list=activity_data.keys())

        if level_name in results_dict:
            energy_data = self.merge_input_data([energy_data,
                                                results_dict[level_name]['energy']],
                                                'Energy_Type')
            activity_data = self.merge_input_data([activity_data,
                                                  results_dict[level_name]['activity']],
                                                  'activity_type')
        
        data_dict_ = {'energy': energy_data, 'activity': activity_data,
                      'level_total': level_name}

        if weather:
            data_dict_['weather_factors'] = weather_data_

        results_dict[level_name] = data_dict_

        return results_dict

    def nesting(self, level_name, results_dict, select_categories):
        """Aggregate data from lower level

        Args:
            level_name ([type]): [description]
            results_dict ([type]): [description]
            select_categories ([type]): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """

        if level_name in results_dict.keys():
            if 'activity' in results_dict[level_name].keys() and 'energy'\
                                         in results_dict[level_name].keys():
                aggregate_activity = [results_dict[level_name]['activity']]
                aggregate_energy = [results_dict[level_name]['energy']]

            else:
                aggregate_activity = []
                aggregate_energy = []

            if 'weather_factors' in results_dict[level_name].keys():
                weather = True
                aggregate_weather = results_dict[level_name]['weather_factors']
            else:
                weather = False
        else:
            aggregate_activity = []
            aggregate_energy = []
            weather = False

        for key, value in select_categories.items():
            if key == np.nan:
                raise ValueError('select_categories key is NAN')

            try:
                lower_level_e = results_dict[key]['energy']
                lower_level_a = results_dict[key]['activity']
                if 'weather_factors' in results_dict[key].keys():
                    lower_level_w = results_dict[key]['weather_factors']
                    for w_, w_data in lower_level_w.items():
                        if isinstance(w_data, pd.DataFrame):
                            lower_level_w = lower_level_w
                        elif isinstance(w_data, dict):
                            lower_level_w = results_dict[key]['weather_factors'][key]

                if isinstance(value, dict):
                    base_col_a = 'activity_type'
                    if len(lower_level_a.columns.difference([base_col_a]).tolist()) > 1:
                        lower_level_a = df_utils.create_total_column(lower_level_a, 
                                                                     key)[[base_col_a, key]]     

                    base_col_e = 'Energy_Type'
                    if len(lower_level_e.columns.difference([base_col_e]).tolist()) > 1:
                        lower_level_e = df_utils.create_total_column(lower_level_e,
                                                                     key)[[base_col_e, key]]

                else:

                    if key in lower_level_e.columns:
                        lower_level_e = lower_level_e[['Energy_Type', key]]

                    if key in lower_level_a.columns:
                        lower_level_a = lower_level_a[['activity_type', key]]

                aggregate_activity.append(lower_level_a)
                aggregate_energy.append(lower_level_e)

            except KeyError:
                print(f'lower level key: {key} failed on level : {level_name}')
                continue

        e_df = self.merge_input_data(aggregate_energy, 'Energy_Type')
        e_df = df_utils.create_total_column(e_df, level_name)

        agg_a_df = self.merge_input_data(aggregate_activity, 'activity_type')
        agg_a_df = df_utils.create_total_column(agg_a_df, level_name)

        data_dict = {'energy': e_df,
                     'activity': agg_a_df,
                     'level_total': level_name}
        if weather:
            data_dict['weather_factors'] = aggregate_weather

        results_dict[f'{level_name}'] = data_dict

        return results_dict

    def build_nest(self, data, select_categories, results_dict,
                   level1_name, level_name=None):
        """Process and organize raw data"""

        if isinstance(select_categories, dict):
            for key, value in select_categories.items():

                if isinstance(value, dict):
                    yield from self.build_nest(data=data[key],
                                               select_categories=value,
                                               results_dict=results_dict,
                                               level1_name=level1_name,
                                               level_name=key)

                else: 
                    if not level_name:
                        level_name = level1_name
                    results_dict = self.process_results_dict(data,
                                                             select_categories,
                                                             results_dict,
                                                             level_name, key)

        else:
            results_dict = results_dict

        if not level_name:
            level_name = level1_name

        results_dict = self.nesting(level_name, results_dict,
                                    select_categories)

        yield results_dict

    def merge_input_data(self, list_dfs, second_index):
        """Merge dataframes of same variable type"""

        list_dfs = [df_utils.int_index(l) for l in list_dfs]
        if len(list_dfs) == 1:
            return list_dfs[0]
        elif np.array([list(df.columns) == list(list_dfs[0].columns)
                       for df in list_dfs]).all():
            print('dataframes have the same columns')
            return list_dfs[0]
        else:
            list_dfs = [l.reset_index() for l in list_dfs]

            df = reduce(lambda df1, df2: df1.merge(df2[list(df2.columns.difference(df1.columns)) + \
                                                    ['Year', second_index]], how='outer', on=['Year', second_index]), list_dfs).set_index('Year')
            return df

    def order_categories(self, level_of_aggregation, raw_results):
        """Order categories so that lower levels are calculated prior
        to current level of aggregation. This ordering ensures that 
        lower level structure is passed to higher level.
        """
        if len(self.categories_dict) == 1:
            categories_list = list(self.categories_dict.keys())

        elif len(self.categories_dict) > 1:
            categories = self.deep_get(self.categories_dict, '.'.join(level_of_aggregation))
            categories_list = []
            for key in raw_results.keys():
                if key in categories.keys():
                    categories_list.append(key)
            for key in raw_results.keys():
                if key not in categories.keys():
                    categories_list.append(key)

        return categories_list

    def calculate_breakout_lmdi(self, raw_results, level_of_aggregation, 
                                breakout, categories, lmdi_type):
        """If breakout=True, calculate LMDI for each lower aggregation
        level contained in raw_results.

        Args:
            raw_results (dictionary): Built "nest" of dictionaries
                                      containing input data for LMDI
                                      calculations

        Returns:
            final_results_list [list]: list of LMDI results dataframes
        
        TODO: Lower level Total structure (product of each structure index
        for multiplicative) and component intensity index (index of aggregate
        intensity divided by total strucutre) need to be passed to higher level
        """        
        categories_list = self.order_categories(level_of_aggregation,
                                                raw_results)
        final_results_list = []

        for key in categories_list:
            level_total = raw_results[key]['level_total']

            if len(categories_list) == 1:
                level_total = categories_list[0]
                loa = [self.sector.capitalize()] + level_of_aggregation
                categories = self.categories_dict

            elif level_of_aggregation[-1] == level_total:
                loa = [self.sector.capitalize()] + level_of_aggregation
                categories = self.deep_get(self.categories_dict, '.'.join(level_of_aggregation))
            else:
                loa = [self.sector.capitalize()] + level_of_aggregation + [level_total]
                categories = self.deep_get(self.categories_dict, '.'.join(level_of_aggregation) + f'.{key}')

            if not categories:
                print(f"{key} not in categories")
                continue

            activity_ = dict()
            total_activty_df = raw_results[key]['activity']
            for a_type in total_activty_df['activity_type'].unique():
                a_df = total_activty_df[total_activty_df['activity_type'] == a_type].drop('activity_type', axis=1)
                if level_total not in a_df.columns:
                    raise KeyError(f'{level_total} not in {a_type} dataframe')
                activity_[a_type] = a_df

            total_energy_df = raw_results[key]['energy']
            for e_type in self.energy_types:
                try:
                    energy_df = total_energy_df[total_energy_df['Energy_Type'] == e_type].drop('Energy_Type', axis=1)
                except KeyError:
                    continue
                if level_total not in energy_df.columns:
                    raise KeyError(f'{level_total} not in energy_df')

                if 'weather_factors' in raw_results[key].keys():
                    weather_data = raw_results[key]['weather_factors']
                else:
                    weather_data = None

                lower_level_structure_df, lower_level_intensity_df = \
                    self.calc_lower_level(categories,
                                          final_results_list,
                                          e_type)

                category_lmdi = self.call_lmdi(energy_df, activity_,
                                               lower_level_structure_df,
                                               lower_level_intensity_df,
                                               level_total,
                                               unit_conversion_factor=1,
                                               weather_data=weather_data,
                                               loa=loa, energy_type=e_type,
                                               lmdi_type=lmdi_type)
                if category_lmdi is None:
                    continue
                else:
                    structure_cols = [col for col in category_lmdi if 'Structure' in col]
                    category_lmdi['total_structure'] = category_lmdi[structure_cols].product(axis=1)
                    category_lmdi["@filter|EnergyType"] = e_type
                    category_lmdi['lower_level'] = level_total
                    final_results_list.append(category_lmdi)
        
        if len(final_results_list) > 1:
            final_results = pd.concat(final_results_list,
                                      axis=0, ignore_index=True,
                                      join='outer', sort=False)
        elif len(final_results_list) == 0:
            raise ValueError('calculate_breakout_lmdi returned empty list')
        else:
            final_results = final_results_list[0]

        return final_results

    def calc_lower_level(self, categories, final_fmt_results, e_type):
        """Calculate decomposition for lower levels of aggregation
        """
        if not final_fmt_results:
            return pd.DataFrame(), pd.DataFrame()
        else:
            final_fmt_results = pd.concat(final_fmt_results,
                                          axis=0, sort=False)

            if 'lower_level' not in final_fmt_results.columns:
                return pd.DataFrame(), pd.DataFrame()

            lower_level_structure_list = []
            lower_level_intensity_list = []

            for key, value in categories.items():

                lower_level = final_fmt_results[(final_fmt_results['lower_level'] == key) & \
                                                (final_fmt_results["@filter|EnergyType"] == e_type) & \
                                                (final_fmt_results["@filter|Measure|BaseYear"] == self.base_year) & \
                                                (final_fmt_results["@filter|Model"] == 'Multiplicative')]

                if not value:
                    lower_level_structure = pd.DataFrame(index=lower_level.index, columns=[f'lower_level_structure_{key}'])
                    lower_level_structure[f'lower_level_structure_{key}'] = 1

                    lower_level_intensity = pd.DataFrame(index=lower_level.index, columns=[key])

                elif isinstance(value, dict):
                    try:
                        lower_level_structure = lower_level[['@timeseries|Year', 'total_structure']].set_index('@timeseries|Year')
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
                lower_level_structure_df = df_utils.merge_df_list(lower_level_structure_list)
                lower_level_structure_df = lower_level_structure_df.fillna(1)

            if not lower_level_intensity_list:
                lower_level_intensity_df =  pd.DataFrame()
            else:
                lower_level_intensity_df = df_utils.merge_df_list(lower_level_intensity_list)
            return lower_level_structure_df, lower_level_intensity_df

    def get_nested_lmdi(self, level_of_aggregation, raw_data,
                        lmdi_type, calculate_lmdi=False,
                        breakout=False):
        """
        Collect LMDI decomposition according to user specifications

        TODO: 
            - Build in weather capabilities
        """
        categories = self.deep_get(self.categories_dict, level_of_aggregation)
        
        if len(self.categories_dict) == 1 and not categories:
            categories = self.categories_dict

        data = reduce(lambda d, key: d.get(key, d) if isinstance(d, dict) else d, level_of_aggregation.split("."), raw_data)

        level_of_aggregation_ = level_of_aggregation.split(".")
        level1_name = level_of_aggregation_[-1]

        categories_pre_breakout = categories

        results_dict = dict()
        for results_dict in self.build_nest(data=data, select_categories=categories,
                                            results_dict=results_dict,
                                            level1_name=level1_name):
            continue

        final_results = self.calculate_breakout_lmdi(results_dict,
                                                     level_of_aggregation_,
                                                     breakout,
                                                     categories_pre_breakout,
                                                     lmdi_type)

        
        final_results.to_csv(f'{self.output_directory}/{self.sector}_{level1_name}_decomposition.csv', index=False)

        return final_results, final_results

    def nominal_energy_intensity(self, energy_input_data, activity_input_data):
        """Calculate nominal energy intensity (i.e. energy divided by activity)"""
        energy_input_data, activity_input_data = df_utils.ensure_same_indices(energy_input_data, activity_input_data)

        if isinstance(activity_input_data, pd.DataFrame):
            activity_width = activity_input_data.shape[1]

        elif isinstance(activity_input_data, pd.Series):
            activity_width = 1

        energy_width = energy_input_data.shape[1]

        if energy_width == activity_width:
            nominal_energy_intensity = energy_input_data.divide(activity_input_data, axis='index').multiply(self.unit_conversion_factor)
        elif energy_width == 1 and activity_width > 1:
            nominal_energy_intensity = np.divide(np.tile(energy_input_data.values, activity_width), activity_input_data.values)
            nominal_energy_intensity = pd.DataFrame(nominal_energy_intensity, index=energy_input_data.index, columns=activity_input_data.columns)
            nominal_energy_intensity = nominal_energy_intensity.multiply(self.unit_conversion_factor)

        return nominal_energy_intensity

    def prepare_lmdi_inputs(self, energy_type, energy_input_data, activity_input_data, lower_level_intensity_df,
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
        nom_intensity_dict = dict()

        for activity, activity_data in activity_input_data.items():

            energy_input_data, activity_data = df_utils.ensure_same_indices(energy_input_data, activity_data)

            # E is the total energy consumption in industry, Q is the total industrial activity level
            # ln(IT_i/I0_i) --> I_i = E_i / Q_i,  I_i is the energy intensity of sector i
            nom_intensity = self.nominal_energy_intensity(energy_input_data, activity_data).drop(total_label, axis=1, errors='ignore')
            nom_intensity_dict[activity] = nom_intensity

            activity_shares = df_utils.calculate_shares(activity_data, total_label)

            # ln(ST_i/S0_i) --> S_i= Q_i / Q,  S_i is the activity share of sector i
            log_ratio_structure_activity = df_utils.calculate_log_changes(activity_shares).rename(columns={col: 
                                                                                        f'{activity}_{col}' 
                                                                                        for col in 
                                                                                        activity_shares.columns}) 
            
            if activity != 'only_activity' or self.sector != 'commercial':                                          
                log_ratio_structure[activity] = log_ratio_structure_activity

            # ln(QT/Q0)  --> Q = Q,  Q is the total industrial activity level
            log_ratio_activity_a = df_utils.calculate_log_changes(activity_data[[total_label]])  
            log_ratio_activity[activity] = log_ratio_activity_a

            nom_intensity_base = nom_intensity.loc[self.base_year, :]
            intensity_index = nom_intensity.divide(np.tile(nom_intensity_base, (len(nom_intensity), 1)))

            if not lower_level_intensity_df.empty:
                lower_level_intensity_df = lower_level_intensity_df.fillna(intensity_index)
                log_ratio_intensity_a = df_utils.calculate_log_changes(lower_level_intensity_df)

            else:
                log_ratio_intensity_a = df_utils.calculate_log_changes(intensity_index)

            log_ratio_intensity[activity] = log_ratio_intensity_a

        if weather_data:
            if energy_type in ['elec', 'fuels']:
                weather_ = weather_data[energy_type]
            else:
                if 'only_activity' in nom_intensity_dict.keys():
                    nom_intensity = nom_intensity_dict['only_activity']
                else:
                    if isinstance(self.weather_activity, dict):
                        if total_label in self.weather_activity.keys():
                            nom_intensity = nom_intensity_dict[self.weather_activity[total_label]]
                    elif isinstance(self.weather_activity, str):
                        nom_intensity = nom_intensity_dict[self.weather_activity]

                weather_elec, nom_intensity = df_utils.ensure_same_indices(weather_data['elec'], nom_intensity)
                nom_intensity, weather_fuels = df_utils.ensure_same_indices(nom_intensity, weather_data['fuels'])

                weather_adj_dict = {'elec': nom_intensity.divide(weather_elec.values), 
                                    'fuels': nom_intensity.divide(weather_fuels.values)}

                nom_intensity_weather_adj_dict = self.collect_energy_data(weather_adj_dict)

                nom_intensity_weather_adj = nom_intensity_weather_adj_dict[energy_type]
                nom_intensity_weather_adj = nom_intensity_weather_adj.drop('Energy_Type', axis=1, errors='ignore')
                weather_ = nom_intensity.divide(nom_intensity_weather_adj.values)

            log_changes_weather = df_utils.calculate_log_changes(weather_)
            log_ratio_structure['weather'] = log_changes_weather

        energy_shares = df_utils.calculate_shares(energy_input_data, total_label)

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
        if energy_input_data.empty:
            return None

        energy_data, energy_shares, log_ratios = self.prepare_lmdi_inputs(energy_type, 
                                                                          energy_input_data, activity_input_data, 
                                                                          lower_level_intensity_df,
                                                                          total_label, weather_data,
                                                                          unit_conversion_factor=1)
        if not lower_level_structure.empty:
            lower_level_structure = df_utils.calculate_log_changes(lower_level_structure)
            log_ratios['lower_level_structure'] = lower_level_structure

        results = self.call_decomposition(energy_data, energy_shares, 
                                          log_ratios, total_label, lmdi_type, loa, 
                                          energy_type)
        return results

        
if __name__ == '__main__':
    pass


