

import yaml
import itertools
import logging
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import pandas as pd
import os

from EnergyIntensityIndicators.utilities.dataframe_utilities \
    import DFUtilities as df_utils
from EnergyIntensityIndicators.utilities import (lmdi_utilities,
                                                 loggers)

logger = loggers.init_logger(__name__)

logger = logging.getLogger(__name__)

class GeneralLMDI:
    """Class to decompose changes in a variable using model
    described in YAML file

    Example input (standard LMDI approach, Residential):

    {'variables': ['E_i', 'A_i'],
     'LHS_var': 'E_i',
     'decomposition': 'A*A_i/A*E_i/A_i',
     'terms': ['A', 'A_i/A', 'E_i/A_i']
     'model': 'multiplicative',
     'lmdi_type': 'II',
     'totals': {'A': 'sum(A_i)'},
     'subscripts': {'i': {'names':
                                 ['Northeast', 'Midwest', 'South', 'West'],
                           'count': 4}},
     'energy_types': ['source', 'deliv', 'elec', 'fuels']

     'base_year': 1990,
     'end_year': 2018}

    Note: terms may be different from the multiplied components of
    the decomposition (terms are the variables that are weighted by
    the log mean divisia weights in the final decomposition)
    """

    def __init__(self, config_path):
        """
        Args:
            directory (str): Path to folder containing YAML
                             files with LMDI input parameters
        """

        self.config_path = config_path
        self.read_yaml()

    def create_yaml(self):
        """Create YAML containing input data
        from dictionary
        """

        input_ = {'variables': ['E_i', 'A_i'],
                  'LHS_var': 'E_i',
                  'decomposition': 'A*A_i/A*E_i/A_i',
                  'terms': ['A', 'A_i/A', 'E_i/A_i'],
                  'model': 'multiplicative',
                  'lmdi_type': 'II',
                  'totals': {'A': 'sum(A_i)'},
                  'subscripts':
                  {'i':
                   {'names':
                    ['Northeast', 'Midwest', 'South', 'West'],
                    'count': 4}},
                  'energy_types': ['source', 'deliv', 'elec', 'fuels'],
                  'base_year': 1990,
                  'end_year': 2018}

        with open(self.config_path, 'w') as file:
            yaml.dump(input_, file)

    def read_yaml(self):
        """Read YAML containing input data, create attribute
        for each item in resulting dictionary

        Parameters:
            fname (str): YAML file containing input data
        """

        with open(self.config_path, 'r') as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            input_dict = yaml.load(file, Loader=yaml.FullLoader)
            logger.info(f'input_dict:\n {input_dict}')
            for k, v in input_dict.items():
                setattr(self, k, v)

    @staticmethod
    def test_expression(expression, lhs):
        """Verify expression provided properly simplifies

        Args:
            expression (Symbolic Expression): [description]
            lhs (Symbolic Variable): The LHS variable
                                     (variable to decompose)

        Returns:
            (bool): Whether or not the symbolic expression simplifies
                    to the LHS variable
        """

        if lhs == str(sp.simplify(expression)):
            logger.info('Decomposition expression simplifies properly')
        else:
            msg = ('Decomposition expression does not simplify '
                   'to LHS variable: '
                   f'{lhs} != {str(sp.simplify(expression))}')
            logger.error(msg)
            raise ValueError(msg)


    @staticmethod
    def check_eval_str(s):
        """From NREL rev.rev.utilities.utilities

        Check an eval() string for questionable code.
        Parameters
        ----------
        s : str
            String to be sent to eval(). This is most likely a math equation
            to be evaluated. It will be checked for questionable code like
            imports and dunder statements.
        """

        bad_strings = ('import', 'os.', 'sys.', '.__', '__.')
        for bad_s in bad_strings:
            if bad_s in s:
                msg = ('Will not eval() string which contains "{}": {} '
                       .format(bad_s, s))
                logger.error(msg)
                raise ValueError(msg)

    def multiplicative_weights(self, LHS, LHS_share):
        """Calculate log mean weights where T = t, 0 = t-1

        Args:
            LHS (pd.DataFrame): Data for the left hand side variable
                             of the decomposition equation
            LHS_share (pd.DataFrame): Shares of total LHS var for
                                       each category in level of
                                       aggregation total_label (str):
                                       Name of aggregation of categories
                                       in level of aggregation

        Multiplicative model uses the LMDI-II model because
        'the weights...sum[] to unity, a desirable property in
        index construction.' (Ang, B.W., 2015. LMDI decomposition
                               approach: A guide for implementation.
                               Energy Policy 86, 233-238.).

        Returns:
            log_mean_weights_normalized (pd.DataFrame): LMDI weights
        """

        if LHS_share.shape[1] == 1:
            return LHS_share
        else:
            log_mean_weights = pd.DataFrame(index=LHS.index)
            for col in LHS_share.columns:
                LHS_share[f"{col}_shift"] = LHS_share[col].shift(periods=1,
                                                                 axis='index',
                                                                 fill_value=0)
                # apply generally not preferred for row-wise operations but?
                log_mean_weights[f'log_mean_weights_{col}'] = \
                    LHS_share.apply(lambda row:
                                    lmdi_utilities.logarithmic_average(
                                       row[col], row[f"{col}_shift"]), axis=1)

            sum_log_mean_shares = log_mean_weights.sum(axis=1)
            log_mean_weights_normalized = \
                log_mean_weights.divide(
                    sum_log_mean_shares.values.reshape(
                            len(sum_log_mean_shares), 1))

            return log_mean_weights_normalized

    def compute_index(self, component, base_year_):
        """Compute index of components (indexing to chosen base_year_),
        replicating methodology in PNNL spreadsheets for the multiplicative
        model

        Args:
            component (pd.Series or pd.DataFrame): If Dataframe, needs to be n x 1
            base_year_ (int): [description]

        Returns:
            index (pd.DataFrame): Component data indexed to base_year_
        """

        component.index = component.index.astype(int)
        if isinstance(component, pd.DataFrame):
            component_col = component.columns[0]
            component = component[component_col]
        index = pd.DataFrame(index=component.index, columns=['index'])
        component = component.replace([np.inf, -np.inf], np.nan)
        component = component.fillna(1)

        for y in component.index:
            if y == min(component.index):
                index.loc[y, 'index'] = 1
            else:
                if component.loc[y] == np.nan:
                    index.loc[y, 'index'] = index.loc[y - 1, 'index']

                else:
                    index.loc[y, 'index'] = \
                        index.loc[y - 1, 'index'] * component.loc[y]

        index_normalized = index.divide(index.loc[base_year_])  # 1985=1
        return index_normalized

    def decomposition_multiplicative(self, terms_df):
        """Format component data, collect overall effect, return indexed
        dataframe of the results for the multiplicative LMDI model.

        Args:
            terms_df (pd.DataFrame): DataFrame with decomposed changes in
                                     LHS var

        Returns:
            results (pd.DataFrame): terms_df (exponential) with Effect column
        """

        results = terms_df.apply(lambda col: np.exp(col), axis=1)

        for col in results.columns:
            results[col] = self.compute_index(results[col], self.base_year)

        results['Effect'] = results.product(axis=1)

        return results

    def additive_weights(self, LHS, LHS_share):
        """Calculate log mean weights for the additive
        model where T=t, 0 = t - 1

        Args:
            LHS (pd.DataFrame): Data for the left hand side variable
                             of the decomposition equation
            LHS_share (pd.DataFrame): Shares of total LHS var for
                                       each category in level of
                                       aggregation total_label (str):
                                       Name of aggregation of categories
                                       in level of aggregation

        self.lmdi_type should be one of 'LMDI-I' or 'LMDI-II'.
        Standard choice is 'LMDI-I' because it is 'consistent in
        aggregation and perfect in decomposition at the subcategory
        level' (Ang, B.W., 2015. LMDI decomposition approach: A guide
        for implementation. Energy Policy 86, 233-238.).

        Returns:
            LMDI weights (pd.DataFrame)
        """

        if not self.lmdi_type:
            self.lmdi_type = 'LMDI-I'

        log_mean_shares_labels = [f"log_mean_shares_{col}" for
                                  col in LHS_share.columns]
        log_mean_weights = pd.DataFrame(index=LHS.index)
        log_mean_values_df = pd.DataFrame(index=LHS.index)
        LHS_data = LHS.copy()
        for col in LHS.columns:
            LHS_data[f"{col}_shift"] = LHS_data[col].shift(
                                        periods=1, axis='index', fill_value=0)

            # apply generally not preferred for row-wise operations but?
            log_mean_values = \
                LHS_data[[col, f"{col}_shift"]].apply(
                    lambda row: lmdi_utilities.logarithmic_average(
                        row[col], row[f"{col}_shift"]), axis=1)

            log_mean_values_df[col] = log_mean_values.values

            LHS_share[f"{col}_shift"] = LHS_share[col].shift(periods=1,
                                                             axis='index',
                                                             fill_value=0)
            # apply generally not preferred for row-wise operations but?
            log_mean_shares = \
                LHS_share[[col, f"{col}_shift"]].apply(
                    lambda row: lmdi_utilities.logarithmic_average(
                        row[col], row[f"{col}_shift"]), axis=1)

            LHS_share[f"log_mean_shares_{col}"] = log_mean_shares

            log_mean_weights[f'log_mean_weights_{col}'] = \
                log_mean_shares * log_mean_values

        cols_to_drop1 = \
            [col for col in LHS_share.columns if
             col.startswith('log_mean_shares_')]

        LHS_share = LHS_share.drop(cols_to_drop1, axis=1)

        cols_to_drop = \
            [col for col in LHS_share.columns if col.endswith('_shift')]
        LHS_share = LHS_share.drop(cols_to_drop, axis=1)

        cols_to_drop_ = [col for col in LHS.columns if col.endswith('_shift')]

        LHS_data = LHS_data.drop(cols_to_drop_, axis=1)

        if self.lmdi_type == 'LMDI-I':
            return log_mean_values_df

        elif self.lmdi_type == 'LMDI-II':
            sum_log_mean_shares = LHS_share[log_mean_shares_labels].sum(axis=1)
            log_mean_weights_normalized = \
                log_mean_weights.divide(
                    sum_log_mean_shares.values.reshape(
                        len(sum_log_mean_shares), 1))

            log_mean_weights_normalized = \
                log_mean_weights_normalized.drop(
                    [c for c in log_mean_weights_normalized.columns
                     if not c.startswith('log_mean_weights_')], axis=1)
            return log_mean_weights_normalized

        else:
            return log_mean_values_df

    def decomposition_additive(self, terms_df):
        """Format component data, collect overall effect,
        return aggregated dataframe of the results for
        the additive LMDI model.

        Calculate effect from changes to activity, structure,
        and intensity in the additive model
        """

        terms_df['Effect'] = terms_df.sum(axis=1)

        return terms_df

    @staticmethod
    def all_equal(iterator):
        """Create bool describing whether all
        items in an iterator are the same
        """

        return len(set(iterator)) <= 1

    @staticmethod
    def dict_iter(data_dict, path):
        """Iterate through dictionary using path, return resulting
        dataframe

        Args:
            data_dict (dict): raw data (all sector) containing
                              with nesting matching that of the
                              sub_categories dict up to (and sometimes
                              including) the innermost dictionary
                              (which then contains variable specific
                              keys and data)

            path (list): "path" (of keys) to dataframes in the data_dict

        Returns:
            data (pd.DataFrame): Data at the end of the path for variable
        """

        data = data_dict.copy()
        for k in path:
            data = data[k]
        return data

    def get_paths(self, d, current=[]):
        """Get list of 'paths' to all endpoints in dictionary

        Args:
            d (dict): Nested dictionary describing relationships
                      between all levels of aggregation
            current (list, optional): List containing path lists.
                                      Defaults to [].

        Yields:
            current (list): List of lists (each inner list containing
                            a path)
        """

        for a, b in d.items():
            yield current+[a]
            if isinstance(b, dict):
                yield from self.get_paths(b, current+[a])
            elif isinstance(b, list):
                for i in b:
                    yield from self.get_paths(i, current+[a])

    def collect_base_data(self, sub_categories, raw_data, variable):
        """Iterate through nested dictionary collecting dataframes
        for given variable

        Args:
            subscripts (list): Subscripts assigned to variable e.g. [i, k]
            raw_data (dict): Nested dictionary containing variable
                             keys and dataframes values in innermost
                             dictionary values. Outer nesting should match
                             sub_categories nesting.
            variable (str): variable (datatype) e.g. A_i_k

        Raises:
            ValueError: Throws error if base_data is not pd.DataFrame
            ValueError: Throws error if paths_dict is empty after build

        Returns:
            paths_dict (dict): Keys are paths to data
                               (e.g. 'National.Northeast.Single-Family')
                               and values are dataframes containing specified
                               data
        """
        paths_dict = dict()
        paths = list(self.get_paths(sub_categories))
        paths_sorted = sorted(paths, key=len, reverse=True)

        # logger.info(f'collect_base_data:\nsub_categories: {sub_categories}\npaths_sorted:{paths_sorted}')

        raw_data_paths = list(self.get_paths(raw_data))
        raw_data_paths_sorted = sorted(raw_data_paths, key=len, reverse=True)

        # logger.info(f'collect_base_data:\n, raw_data_paths_sorted:{raw_data_paths_sorted}')

        raw_data_paths_sorted = \
            [p for p in raw_data_paths_sorted if p[-1] == variable]

        for p in raw_data_paths_sorted:

            base_data = self.dict_iter(raw_data, p)

            if len(p) == 1:
                p = [self.total_label]
            elif len(p) > 1:
                p = p[:-1]
            if base_data is None:
                continue
            if isinstance(base_data, pd.DataFrame):
                base_data = base_data.loc[base_data.index.notnull()]
                base_data.index = base_data.index.astype(int)
                sub_dict = dict()
                if base_data.shape[1] > 1:
                    for c in base_data.columns:
                        sub_data = base_data[[c]]
                        path = p + [c]
                        if path in paths_sorted:
                            p_str = '.'.join(path)
                            sub_dict[p_str] = sub_data
                        else:
                            p_str = '.'.join(p)
                            paths_dict[p_str] = base_data

                    logger.info(f'sub_dict: {sub_dict}')
                    paths_dict.update(sub_dict)
                else:
                    p_str = '.'.join(p)
                    paths_dict[p_str] = base_data
            else:
                msg = ('base data is type', type(base_data))
                logger.error(msg)
                raise ValueError(msg)

        if len(paths_dict) == 0:
            msg = ('paths_dict is empty in collect_base_data')
            raise ValueError('paths_dict is empty')

        return paths_dict

    @staticmethod
    def create_len_dict(paths_dict):
        """Create dictionary with keys in paths_dict
        sorted by length where keys are the int length

        Args:
            paths_dict (dict): Keys are paths to data
                               (e.g. 'National.Northeast.Single-Family')
                               and values are dataframes containing specified
                               data

        Returns:
            len_dict (dict): Keys are len of paths, values are lists
                             of paths with that length e.g.
                             {3: ['National.Northeast.Single-Family'],
                              2: ['National.Northeast'],
                              1: ['National']}
            key_range (list): Lengths from len_dict
        """
        key_list = list(paths_dict.keys())
        len_dict = {k: len(k.split('.')) for k in key_list}
        key_list_split = [k.split('.') for k in key_list]
        order_keys = sorted(key_list_split, key=len, reverse=True)
        key_range = list(range(1, len(order_keys[0]) + 1))
        len_dict = dict()
        for j in key_range:
            len_list = []
            for l_ in order_keys:
                if len(l_) == j:
                    len_list.append('.'.join(l_))
            len_dict[j] = len_list

        return len_dict, key_range

    def aggregate_data(self, raw_data, subscripts,
                       variable, sub_categories,
                       lhs_data=None, lhs_sub_names=None):
        """Aggregate variable data from raw data for every level
        of aggregation in the sub_categories

        Args:
            raw_data (dict): Nested dictionary containing variable
                             keys and dataframes values in innermost
                             dictionary values. Outer nesting should match
                             sub_categories nesting.
            subscripts (list): Subscripts assigned to variable e.g. [i, k]
            variable (str): variable (datatype) e.g. A_i_k
            sub_categories (dict): Nested dictionary describing relationships
                                   between levels of aggregation in data
            lhs_data (dict, optional): Dictionary of dataframes of left hand
                                       side variable keys are 'paths'.
                                       Defaults to None.
            lhs_sub_names (dict, optional): keys are subscripts associated
                                            with the LHS variable, values
                                            are lists of (str) names
                                            associated with the subscript.
                                            Defaults to None.

        Returns:
            paths_dict (dict): Dictionary of variable data with paths as keys
                               and variable+path DataFrame as values
        """
        logger.info('Aggregating data with aggregate_data')
        paths_dict = self.collect_base_data(sub_categories, raw_data, variable)
        len_dict, key_range = self.create_len_dict(paths_dict)

        reverse_len = sorted(key_range, reverse=True)
        for n in reverse_len:
            n_lists = []
            paths = len_dict[n]
            if len(paths) > 1:
                for i, p in enumerate(paths):
                    p_list = p.split('.')
                    path_list_short = p_list[:-1]
                    p_short_data = [p]
                    other_p = paths[:i] + paths[(i+1):]
                    logger.info(f'other_p:\n {other_p}')
                    for k, j in enumerate(other_p):
                        logger.info(f'j: {j}')
                        other_p_short = j.split('.')[:-1]
                        if other_p_short == path_list_short:
                            p_short_data.append(j)
                    if n > 1:
                        higher_paths = len_dict[n-1]
                        if len(higher_paths) > 0:
                            for h in higher_paths:
                                h_list = h.split('.')
                                h_short = h_list[:-1]
                                if h_short == path_list_short:
                                    p_short_data.append(h)

                    if sorted(p_short_data) not in n_lists:
                        n_lists.append(sorted(p_short_data))

            level_data = self.group_data(n_lists, paths_dict,
                                         variable, lhs_data,
                                         lhs_sub_names)
            if n > 1:
                higher_keys = len_dict[n-1]
                for g in list(level_data.keys()):
                    higher_keys.append(g)
                len_dict[n-1] = higher_keys

            paths_dict.update(level_data)

        # print('paths_dict keys:', paths_dict.keys())
        # exit()
        return paths_dict

    def group_data(self, path_list, data_dict, variable,
                   lhs_data, lhs_sub_names):
        """[summary]

        Args:
            path_list (list): List of lists (of n length paths)
            data_dict (dict): Dictionary of variable data with paths as keys
                              and variable+path DataFrame as values
            variable (str): variable (e.g. A_i_k)
            lhs_data (dict, optional): Dictionary of dataframes of left hand
                                       side variable keys are 'paths'.
                                       Defaults to None.
            lhs_sub_names (dict, optional): keys are subscripts
                                            associated with the
                                            LHS variable, values
                                            are lists of (str)
                                            names associated
                                            with the subscript.
                                            Defaults to None.

        Raises:
            ValueError: Weighting data required LHS variable

        Returns:
            n_dict (dict): [description]
        """

        if variable.startswith('C') or variable.startswith('E'):
            keep_cols = True
        else:
            keep_cols = False

        n_dict = dict()

        for grouped_lists in path_list:
            grouped_lists = list(set(grouped_lists))
            all_level = []
            base_path = grouped_lists[0].split('.')
            if len(base_path) > 1:
                level_path = base_path[:-1]  # [self.total_label] +
                level_path = '.'.join(level_path)
            elif len(base_path) == 1:
                level_path = self.total_label

            for path in grouped_lists:
                logger.info(f'This is a path: {path}')
                key = path.split('.')[-1]
                data = data_dict[path]
                if data.empty:
                    continue
                if keep_cols:
                    lower_level_data = data
                else:
                    if lhs_data is not None:
                        try:
                            lhs_df = lhs_data[path]
                            weights = \
                                self.calculate_weights(lhs_df, key)
                            logger.info(f'lhs_df:\n {lhs_df}')
                            logger.info(f'type lhs_df:\n {type(lhs_df)}')
                        except Exception:
                            weights = None

                    else:
                        msg = ('LHS data not provided '
                               'to group data method')
                    # subscript = 'i'
                        logger.error(msg)
                        raise ValueError(msg)
                    # subscript = 'i'

                    # lower_level_data = \
                    #     self.aggregate_level_data(subscript,
                    #                               weights=weights,
                    #                               base_data=data,
                    #                               total_name=key)
                    if path in self.to_weight:
                        if variable in self.to_weight[path]:
                            weight_data = True

                        else:
                            weight_data = False

                    else:
                        weight_data = False

                    lower_level_data = \
                        self.aggregate_level_data(weight_data,
                                                  weights=weights,
                                                  base_data=data,
                                                  total_name=key)

                    if lower_level_data is None:
                        continue
                    if isinstance(lower_level_data, pd.Series):
                        lower_level_data = \
                            lower_level_data.to_frame(name=key)

                logger.info(f'lower_level_data:\n {lower_level_data}')
                all_level.append(lower_level_data)
            try:
                level_data = \
                    df_utils().merge_df_list(all_level, keep_cols)
            except Exception as e:
                logger.error('{} :\n, all_level:\n{}'.format(e, all_level))
                raise e

            n_dict[level_path] = level_data

        return n_dict

    def get_subscript_data(self, input_data, subscript_data, term_piece):
        """From variable subscripts, select desired data

        Args:
            input_data (dict): dictionary of dataframes
                               for selected variable.
                               keys are 'paths',
                               values are dataframes
            subscript_data (dict): dictionary with suscripts
                                   as keys, lists of names as
                                   values
            term_piece (str):  e.g. A_i_k

        Returns:
            term_df (pd.DataFrame): df of e.g. A_i_k data with
                                    i and k multiindex levels
        """

        subs = term_piece.split('_')  # list
        base_var = subs.pop(0)
        variable_data = input_data[base_var]
        new_paths = {k: f'total.{k}' for k in
                     variable_data.keys() if not k.startswith('total')}

        logger.info('new_paths:\n{}'.format(new_paths))
        for old, new in new_paths.items():
            variable_data[new] = variable_data.pop(old)
        logger.info('variable data keys:\n{}\n base_var:{}\n subs:{}'.format(
            variable_data.keys(), base_var, subs
            ))
        # print('base_var:', base_var)
        # print('subs:', subs)
        subscripts = subscript_data[base_var]  # dictionary
        base_path = 'total'
        term_piece_dfs = []
        subs_short = subs[:-1]

        if len(subs) == 0:
            if 'total' in variable_data.keys():
                path = base_path
                path_df = variable_data[path]
                logger.info('path_df of subs 0:\n{}'.format(path_df))
                # print('path_df subs 0:\n', path_df)
                term_piece_dfs.append(path_df)
            elif len(variable_data) == 1:
                path = list(variable_data.keys())[0]
                path_df = variable_data[path]
                cols = list(path_df.columns)
                if len(cols) == 1:
                    levels = [[base_path]]
                else:
                    levels = [[base_path]*len(cols)] + [cols]
                midx = pd.MultiIndex.from_arrays(levels)
                path_df.columns = midx
                return path_df

        base_path = base_path + '.' + self.total_label
        if len(subs) == 1:
            path = base_path
            path_df = variable_data[path]
            logger.info('path_df of subs 1:\n{}'.format(path_df))
            # print('path_df subs 1:\n', path_df)
            combo_list = base_path.split('.')
            cols = list(path_df.columns)
            levels = [[c]*len(cols) for c in combo_list] + [cols]
            midx = pd.MultiIndex.from_arrays(levels)
            path_df.columns = midx
            term_piece_dfs.append(path_df)

        elif len(subs) > 1:  # len(subs_short)
            p_names = [subscripts[p] for p in subs_short]  # list of lists of names
            logger.info('p_names: {}'.format(p_names))
            combinations = list(itertools.product(*p_names))
            logger.info('combinations:{}'.format(combinations))

            for combo in combinations:
                combo_list = base_path.split('.') + list(combo)
                logger.info(f'combo_list: {combo_list}')
                path_n_1 = '.'.join(combo_list[:-1])
                path = '.'.join(combo_list)
                logger.info(f'path: {path}')
                if path in variable_data:  # path_n_1
                    logger.info('path in variable data')
                    path_df = variable_data[path]  # path_n_1
                    logger.info(f'path_df subs > 1:\n {path_df}')
                elif path_n_1 in variable_data:
                    path_df = variable_data[path_n_1]

                if isinstance(path_df.columns, pd.MultiIndex):
                    pass
                else:
                    cols = list(path_df.columns)
                    levels = [[c]*len(cols) for c in combo_list] + [cols]   # combo should be combo_list
                    logger.info(f'levels: {levels}')
                    midx = pd.MultiIndex.from_arrays(levels)
                    path_df.columns = midx

                logger.info(f'path_df subs > 1 multi:\n {path_df}')
                logger.info(f'path_df.columns: {path_df.columns}')
                term_piece_dfs.append(path_df)

        # term_df = pd.concat(term_piece_dfs, axis=0)
        term_df = df_utils().merge_df_list(term_piece_dfs)
        logger.info(f'term_df:\n {term_df}')

        return term_df

    # def aggregate_level_data(self, subscript, weights, base_data, total_name):
    def aggregate_level_data(self,
                             weight_data,
                             weights,
                             base_data,
                             total_name):

        """Aggregate data for variable and level (e.g. region)

        Args:
            weight_data (bool): Whether or not to weight data
                                when summing (i.e. do the units
                                differ across columns to sum)
            weights (pd.DataFrame): LMDI weights
            base_data (pd.DataFrame): data to aggregate
            total_name (str): Name of aggregated data (column)

        Returns:
            total_col (pd.DataFrame): n x 1 df of aggregated data
                                      (sum or weighted average if
                                      column data units vary)
        """

        # units = self.subscripts[subscript]['names'].values()

        if total_name == 'Pipeline':
            print("THIS IS PIPELINE DATA\n Are there weights? {}\n{}".format(
                weight_data, weights)
                )
            base_data.to_csv('pipeline_data.csv')
        if weight_data:
            total_df = \
                df_utils().create_total_column(
                    base_data,
                    total_label=total_name)
            total_col = total_df[[total_name]]

        else:
            if weights is None:
                # raise ValueError('Weights not available at ' +
                #                  'level of aggregation')
                return None
            try:
                base_data, weights = \
                    df_utils().ensure_same_indices(base_data, weights)

                total_col = base_data.multiply(weights.values,
                                               axis=1).sum(axis=1)
            except ValueError:
                total_df = \
                    df_utils().create_total_column(
                        base_data,
                        total_label=total_name)
                total_col = total_df[[total_name]]
        return total_col

    def calculate_weights(self, lhs, name):
        """Calculate LMDI weights

        Args:
            lhs (pd.DataFrame): Dataframe containing data for the left
                                hand side variable of the decomposition
                                equation
            name (str): level name for use in aggregation
                        (not important, dropped)

        Returns:
            weights (pd.DataFrame): Log-Mean Divisia Weights (normalized)
        """
        logger.info(f'calculating weights.....\nName: {name}\nlhs: {lhs}')
        if isinstance(lhs, pd.MultiIndex):
            lhs_share = df_utils().calculate_shares(lhs)
        else:
            lhs_total = df_utils().create_total_column(lhs,
                                                       total_label=name)
            logger.info(f'lhs_total:\n{lhs_total}')
            lhs_share = df_utils().calculate_shares(lhs_total,
                                                    total_label=name)

        logger.info(f'lhs_share:\n{lhs_share}')

        if self.model == 'additive':
            weights = self.additive_weights(lhs, lhs_share)
        elif self.model == 'multiplicative':
            weights = self.multiplicative_weights(lhs, lhs_share)

        return weights

    def divide_multilevel(self, numerator, denominator,
                          shared_levels, lhs_data):
        """Divide term dataframes where they have multilevel index
        columns

        Args:
            numerator (pd.DataFrame): [description]
            denominator (pd.DataFrame): [description]
            shared_levels ([type]): [description]
            lhs_data ([type]): [description]

        Returns:
            [type]: [description]
        """
        logger.info(f'numerator:\n{numerator}')
        numerator_levels = numerator.columns.nlevels

        logger.info(f'denominator:\n{denominator}')

        highest_shared = sorted(shared_levels, reverse=True)[0]
        logger.info(f'highest_shared: {highest_shared}')
        if highest_shared == 0:
            column_tuples = [numerator.columns.get_level_values(0)[0]]
        else:
            column_tuples = [numerator.columns.get_level_values(i)
                             for i in range(highest_shared + 1)]
            column_tuples = list(set(list(zip(*column_tuples))))

        logger.info(f'column_tuples: {column_tuples}')
        grouped_n = numerator.groupby(level=shared_levels,
                                      axis=1)
        grouped_d = denominator.groupby(level=shared_levels,
                                        axis=1)

        results = []
        for u in column_tuples:
            logger.info(f'u: {u}')

            n = grouped_n.get_group(u)
            logger.info(f'n:\n{n}')
            if highest_shared > 0:
                to_drop = list(range(highest_shared + 1, numerator_levels))
                logger.info(f'to_drop: {to_drop}')
                n.columns = n.columns.droplevel(to_drop)
            if not isinstance(n.columns, pd.MultiIndex):
                midx = [list(n.columns)]
                n.columns = pd.MultiIndex.from_arrays(midx)

            logger.info(f'n post group:\n {n}')
            logger.info(f'isinstance(n.columns, pd.MultiIndex): {isinstance(n.columns, pd.MultiIndex)}')

            level_name = \
                pd.unique(n.columns.get_level_values(
                    highest_shared-1))[0]

            d = grouped_d.get_group(u)
            logger.info(f'n post group:\n {d}')
            logger.info(f'isinstance(n.columns, pd.MultiIndex): {isinstance(d.columns, pd.MultiIndex)}')
            try:
                ratio = n.divide(d, axis=1)
            except ValueError:
                ratio = n.divide(d.values, axis=1)

            logger.info(f'ratio:\n {ratio}')

            if isinstance(u, str):
                path = u
                if path not in lhs_data:
                    if self.total_label in lhs_data:
                        path = self.total_label
                    elif f'total.{self.total_label}' in lhs_data:
                        path = f'total.{self.total_label}'
            elif isinstance(u, tuple):
                path = '.'.join(list(u))
            lhs = lhs_data[path]
            print('lhs:\n', lhs)
            logger.info('Line 961: level_name == {}'.format(level_name))
            w = self.calculate_weights(lhs, level_name)
            print('w:\n', w.sum(axis=1))  # Weights sum should = 1
            if w.shape[1] == ratio.shape[1]:
                result = ratio.multiply(w, axis=1).sum(axis=1)
                result = self.decomposition_results(result)
                result = result[[level_name]]
            else:
                if ratio.shape[1] == 1:
                    result = ratio.divide(ratio.loc[self.base_year].values)
                else:
                    # print('ratio:\n', ratio)
                    ratio_levels = ratio.columns.nlevels - 1
                    # result = ratio.sum(axis=1, level=ratio_levels)
                    result = ratio.divide(ratio.loc[self.base_year].values)
                    # print('result:\n', result)

                    # raise ValueError('need to account for this case')

            results.append(result)

        results = pd.concat(results, axis=1)
        return results

    def process_terms(self, input_data, subscript_data,
                      weights, name, lhs_data):
        """From level data, calculate terms and weight them.

        Args:
            input_data (dict): Keys are base variables
                               (e.g. 'A' refers to all 'A',
                               'A_i', 'A_i_k', etc. variables),
                               values are dictionaries where keys
                               are paths (e.g. 'total.National.Northeast')
                               and values are dataframes with multilevel
                               index columns matching the path components
            subscript_data (dict): [description]
            weights (pd.DataFrame): LMDI weights for the level of aggregation
                                    name
            name (level_name): The total label/level of aggregation of interest

        Returns:
            results (pd.DataFrame): Activity, Structure, Intensity, etc.
                                    (results df should have a column
                                    containing results for each of these
                                    or, more generally, the components in
                                    self.term_labels)
        """

        terms = self.decomposition.split('*')
        parts = [t.split('/') for t in terms]
        parts = list(itertools.chain.from_iterable(parts))
        parts = list(set(parts))

        part_data_dict = {p: self.get_subscript_data(input_data,
                                                     subscript_data,
                                                     term_piece=p)
                          for p in parts}
        results = []
        for t in terms:
            # if '/' in t:
            parts = t.split('/')
            first_part = parts[0]
            first_df = part_data_dict[first_part]
            numerator = first_df.copy()
            logger.info(f'numerator:\n {numerator}')

            for i in range(1, len(parts)):
                denominator_part = parts[i]
                denominator = part_data_dict[denominator_part]
                logger.info(f'denominator:\n {denominator}')

                numerator, denominator = \
                    df_utils().ensure_same_indices(numerator, denominator)
                numerator_levels = numerator.columns.nlevels
                logger.info(f'numerator_levels: {numerator_levels}')
                try:
                    denominator_levels = denominator.columns.nlevels
                except ValueError:
                    denominator_levels = 0
                if denominator_levels == 1:
                    if list(denominator.columns)[0] == self.total_label:
                        levels = [['total'], [self.total_label]]
                        midx = pd.MultiIndex.from_arrays(levels)
                        denominator.columns = midx
                if numerator_levels == 1:
                    if list(numerator.columns)[0] == self.total_label:
                        levels = [['total'], [self.total_label]]
                        midx = pd.MultiIndex.from_arrays(levels)
                        numerator.columns = midx

                print('denominator_levels for {}:\n {}'.format(
                    t, denominator_levels
                    ))

                if numerator_levels > denominator_levels:
                    level_count = denominator_levels
                    group_ = True
                elif numerator_levels < denominator_levels:
                    level_count = numerator_levels
                    group_ = True
                elif numerator_levels == denominator_levels:
                    level_count = numerator_levels
                    group_ = False

                shared_levels = list(range(level_count))

                numerator.to_csv('C:/Users/cmcmilla/OneDrive - NREL/Documents - Energy Intensity Indicators/General/EnergyIntensityIndicators/yamls/numerator.csv')
                denominator.to_csv('C:/Users/cmcmilla/OneDrive - NREL/Documents - Energy Intensity Indicators/General/EnergyIntensityIndicators/yamls/denominator.csv')
                if group_:
                    logger.info(f'grouped numerator:\n {numerator.groupby(level=shared_levels,axis=1).sum()}')
                    numerator = self.divide_multilevel(numerator, denominator,
                                                       shared_levels, lhs_data)
                else:
                    # numeator_level_names = numerator.columns
                    # # numerator = \
                    # #     numerator.droplevel(level_count-1, axis=1).divide(
                    # #         denominator.droplevel(level_count-1, axis=1)
                    # #         )
                    # num_denom = numerator.join(denominator)
                    # if len(numerator.columns.levels[numerator_levels-2]) > 1:
                    #     numerator = pd.concat(
                    #         [num_denom.loc[:, (for l in numerator.columns.levels[numerator_levels-2]:

                    numerator = numerator.divide(denominator.values, axis=1)

                logger.info(f'numerator:\n {numerator}')
                if t == 'E_i_j/E_i':
                    exit()
            f = numerator.copy()
            # else:
                # f = input_data[t]
            logger.info(f'f:\n {f}')
            f_levels = f.columns.nlevels
            logger.info(f'f_levels: {f_levels}')

            if f.shape[1] > 1:
                if f.shape[1] == weights.shape[1]:
                    # if name in f.columns:
                    #     f = f.drop(name, axis=1, errors='ignore')
                    component = \
                        f.multiply(weights.values, axis=1).sum(axis=1)
                else:
                    if f.shape[1] > 1:
                        if isinstance(f.columns, pd.MultiIndex):
                            try:
                                if f_levels >= 2:
                                    component = \
                                        f.groupby(level=1, axis=1).sum(axis=1)
                                elif f_levels == 1:
                                    component = \
                                        f.groupby(level=0, axis=1).sum(axis=1)
                            except ValueError:
                                raise ValueError('f failed to groupby:\n', f)
                        else:
                            if name in f.columns:
                                f = f[[name]]
                            else:
                                f = df_utils().create_total_column(f, name)[[name]]
                            component = f
            else:
                component = f

            if isinstance(component, pd.Series):
                component = component.to_frame(name=t)

            logger.info(f'component:\n {component}')
            if component.shape[1] == 2 and name in component.columns:
                component = component.drop(name, axis=1, errors='ignore')
            component = component.rename(
                columns={list(component.columns)[0]: t})
            results.append(component)

        results = df_utils().merge_df_list(results)
        results = results.drop('Commercial_Total', axis=1, errors='ignore')
        results = results.rename(columns=self.term_labels)

        return results

    def nest_var_data(self, raw_data,
                      v, sub_categories,
                      lhs_data=None,
                      lhs_sub_names=None):
        """Collect data for each level of aggregation
        given variable

        Args:
            raw_data (dict): Nested dictionary containing
                             data for each variable in the
                             inner-most dictionaries
            v (str): variable (e.g. A_i_k)
            sub_categories (dict): Nested dictionary describing
                                   relationships between levels
                                   of aggregation in data
            lhs_data (dict, optional): Dictionary of dataframes of left
                                       hand side variable keys are 'paths'.
                                       Defaults to None.
            lhs_sub_names (dict, optional):  keys are subscripts associated
                                             with the LHS variable, values
                                             are lists of (str) names
                                             associated with the subscript.
                                            Defaults to None.

        Returns:
            v_data (dict): Dictionary containing paths as keys and
                           path+variable DataFrames as values
            sub_names (dict): Keys are subscripts (e.g. 'i'), values
                              are lists of name associated with the
                              subscript (e.g. ['Northeast', 'West',
                              'South', 'Midwest'])
        """

        subscripts = v.split('_')[1:]
        v_data = \
            self.aggregate_data(raw_data, subscripts,
                                v, sub_categories,
                                lhs_data, lhs_sub_names)

        sub_names = {s: self.subscripts[s]['names'].keys()
                     for s in subscripts}
        return v_data, sub_names

    def general_expr(self, raw_data, sub_categories):
        """Decompose changes in LHS variable

        Args:
            raw_data (dict): Dictionary containing
                             dataframes for each variable
                             and a the total label
            sub_categories (dict): Nested dictionary describing
                                   relationships between levels
                                   of aggregation in data
                                   e.g. {'National':
                                            {'Northeast': None,
                                             'West': None,
                                             'South': None,
                                             'Midwest': None}}
        Raises:
            ValueError: self.totals keys and values can only
                        contain one non-common subscript.
                        e.g. {'A': 'A_i'} works, {'A': 'A_i_k'}
                        will raise a ValueError
        Returns:
            results (dataframe): LMDI decomposition results
        """

        logger.info(f'gen expr attributes: {dir(self)}')
        self.check_eval_str(self.decomposition)

        for t in self.terms:
            self.check_eval_str(t)

        self.test_expression(self.decomposition, self.LHS_var)

        input_data = dict()
        all_subscripts = dict()

        vars_ = self.variables
        lhs_idx = vars_.index(self.LHS_var)
        lhs_ = vars_.pop(lhs_idx)
        # logger.info(f'Line 1219\nlhs_: {lhs_}')
        logger.info('Nesting data...')
        lhs_data, lhs_sub_names = \
            self.nest_var_data(raw_data,
                               lhs_, sub_categories)

        for v in vars_:
            var_name = v.split('_')[0]
            logger.info(f'Line 1226\nv: {v}')
            v_data, sub_names = \
                self.nest_var_data(raw_data,
                                   v, sub_categories,
                                   lhs_data, lhs_sub_names)

            input_data[var_name] = v_data
            all_subscripts[var_name] = sub_names

        name = self.total_label
        lhs_base_var = self.LHS_var.split('_')[0]
        input_data.update({lhs_base_var: lhs_data})
        all_subscripts.update({lhs_base_var: lhs_sub_names})

        logger.info(f'lhs_data.keys(): {lhs_data.keys()}')
        logger.info(f'name: {name}')
        lhs = lhs_data[name]

        weights = self.calculate_weights(lhs=lhs,
                                         name=name)

        totals = list(self.totals.keys())
        sorted_totals = sorted(totals, key=len, reverse=True)

        for total in sorted_totals:
            cols = self.totals[total]
            cols_subscript = cols.split('_')[1:]
            total_subscript = total.split('_')
            subscripts = [s for s in cols_subscript
                          if s not in total_subscript]

            if len(subscripts) == 1:
                subscript = subscripts[0]
            else:
                raise ValueError('Method not currently able to accomodate'
                                 'summing over multiple subscripts')

            sub_names = list(self.subscripts[subscript]['names'].keys())
            total_base_var = total.split('_')[0]
            base_data = input_data[total_base_var][name]  #[sub_names]
            total_col = self.aggregate_level_data(subscript, weights,
                                                  base_data=base_data,
                                                  total_name=name)

            var_data = input_data[total_base_var]
            var_data.update({'total': total_col})
            input_data[total_base_var] = var_data

        results = self.process_terms(input_data,
                                     all_subscripts,
                                     weights, name,
                                     lhs_data)
        logger.info(f'results:\n {results}')
        # exit()

        expression = self.decomposition_results(results)

        return expression

    def decomposition_results(self, results):
        """Calculate final decomposition results
        from decomposed components

        Args:
            results (pd.DataFrame): Activity, Structure, Intensity, etc.
                                    (results df should have a column
                                    containing results for each of these
                                    or, more generally, the components in
                                    self.term_labels)

        Returns:
            results (pd.DataFrame): results df processed appropriately for
                                    LMDI model type and with effect calculated
        """
        if self.model == 'additive':
            expression = self.decomposition_additive(results)
        elif self.model == 'multiplicative':
            results = df_utils().calculate_log_changes(results)
            expression = self.decomposition_multiplicative(results)
        return expression

    def prepare_for_viz(self, results_df):
        """Rename result columns for use in the OpenEI VizGen
        tool (https://vizgen.openei.org/)

        Args:
            results_df (DataFrame): Results of LMDI decomposition

        Returns:
            results_df (DataFrame): Results with VizGen appropriate
                                    headers
        """

        results_df["Base Year"] = self.base_year

        cols = list(results_df.columns)

        rename_dict = {c: f'@value|Category|{c}#Units' for c in cols}
        rename_dict['Base Year'] = '@scenario|Base Year'
        rename_dict['Year'] = '@timeseries|Year'

        results_df = results_df.reset_index()
        results_df = results_df.rename(columns=rename_dict)
        results_df['@filter|Sector'] = self.sector
        results_df['@filter|Sub-Sector'] = self.total_label
        results_df['@filter|Model'] = self.model
        results_df['@filter|LMDI Type'] = self.lmdi_type
        # results_df['@scenario|Energy Type'] = self.energy_types ?

        return results_df

    def spaghetti_plot(self, data, output_directory=None):
        """Visualize multiplicative LMDI results in a
        line plot
        """

        data = data[data.index >= self.base_year]

        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set2')

        for i, l in enumerate(data.columns):
            plt.plot(data.index, data[l], marker='',
                     color=palette(i), linewidth=1,
                     alpha=0.9, label=data[l].name)

        else:
            if self.LHS_var.startswith('E'):
                title = f"Change in Energy Use {self.total_label}"
            elif self.LHS_var.startswith('C'):
                title = f"Change in Emissions {self.total_label}"

        fig_name = self.total_label + str(self.base_year) + 'decomposition'

        plt.title(title, fontsize=12, fontweight=0)
        plt.xlabel('Year')
        plt.ylabel('Emissions MMT CO2 eq.')
        plt.legend(loc=2, ncol=2)

        if output_directory:
            try:
                plt.savefig(f"{output_directory}/{fig_name}.png")
            except FileNotFoundError:
                plt.savefig(f".{output_directory}/{fig_name}.png")
        plt.show()

    def main(self, input_data, sub_categories):
        """Calculate LMDI decomposition

        Args:
            input_data (dict): Dictionary containing dataframes
                               for each variable defined in the YAML
        """
        logging.basicConfig(filename='./EnergyIntensityIndicators/{}/{}.log'.format(self.sector, self.sector),
                            filemode='w', level=logging.DEBUG)


        results = self.general_expr(input_data, sub_categories)
        logger.info(f'results:\n {results}')
        # exit()
        if self.model == 'multiplicative':
            self.spaghetti_plot(data=results)

        formatted_results = self.prepare_for_viz(results)
        logger.info(f'formatted_results:\n {formatted_results}')

        return formatted_results

    @staticmethod
    def example_input_data():
        """Collect dictionary containing dataframes
        for each variable in the LMDI model
        """

        activity = \
            pd.read_csv('C:/Users/cmcmilla/OneDrive - NREL/Documents - Energy Intensity Indicators/General/EnergyIntensityIndicators/yamls/industrial_activity.csv').set_index('Year')
        energy = \
            pd.read_csv('C:/Users/cmcmilla/OneDrive - NREL/Documents - Energy Intensity Indicators/General/EnergyIntensityIndicators/yamls/industrial_energy.csv').set_index('Year')
        emissions = \
            pd.read_csv('C:/Users/cmcmilla/OneDrive - NREL/Documents - Energy Intensity Indicators/General/EnergyIntensityIndicators/yamls/industrial_energy.csv').set_index('Year')
        logger.info(f'energy cols: {energy.columns}')

        data = {'E_i_j': energy,
                'A_i': activity,
                'C_i_j': emissions,
                'total_label': 'NonManufacturing'}

        return data


if __name__ == '__main__':
    # Will need to update to a new directory in remote repo once code is finished.
    # C:\Users\cmcmilla\OneDrive - NREL\Documents - Energy Intensity Indicators\General\EnergyIntensityIndicators
    directory = 'C:/Users/cmcmilla/OneDrive - NREL/Documents - Energy Intensity Indicators/General/EnergyIntensityIndicators/yamls'
    gen_ = GeneralLMDI(directory)
    """fname (str): Name of YAML file containing
                         LMDI input parameters
    """
    fname = 'combustion_noncombustion_test'  # 'test1'
    gen_.read_yaml(fname)
    input_data = gen_.example_input_data()
    expression = gen_.main(input_data=input_data)
