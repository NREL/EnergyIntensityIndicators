
import sympy as sp
import numpy as np
import pandas as pd
import yaml
import itertools
import matplotlib.pyplot as plt

from EnergyIntensityIndicators.utilities.dataframe_utilities \
    import DFUtilities as df_utils
from EnergyIntensityIndicators.utilities import lmdi_utilities


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
            print('input_dict:\n', input_dict)
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
            print('Decomposition expression simplifies properly')
        else:
            raise ValueError(('Decomposition expression does not simplify '
                              'to LHS variable: '
                              f'{lhs} != {str(sp.simplify(expression))}'))

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
                raise ValueError('Will not eval() string which contains "{}": \
                                 {}'.format(bad_s, s))

    def multiplicative_weights(self, LHS, LHS_share):
        """Calculate log mean weights where T = t, 0 = t-1

        Multiplicative model uses the LMDI-II model because
        'the weights...sum[] to unity, a desirable property in
         index construction.' (Ang, B.W., 2015. LMDI decomposition
                               approach: A guide for implementation.
                               Energy Policy 86, 233-238.).
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
        """
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
            energy_data (dataframe): energy consumption data
            energy_shares (dataframe): Shares of total energy for
            each category in level of aggregation total_label (str):
            Name of aggregation of categories in level of aggregation
            lmdi_type (str, optional): 'LMDI-I' or 'LMDI-II'.

        Defaults to 'LMDI-I' because it is
        'consistent in aggregation and perfect
        in decomposition at the subcategory level'
        (Ang, B.W., 2015. LMDI decomposition approach: A guide for
        implementation. Energy Policy 86, 233-238.).
        """
        print(f'ADDITIVE LMDI TYPE: {self.lmdi_type}')
        if not self.lmdi_type:
            self.lmdi_type = 'LMDI-I'

        print(f'ADDITIVE LMDI TYPE: {self.lmdi_type}')

        log_mean_shares_labels = [f"log_mean_shares_{col}" for
                                  col in LHS_share.columns]
        log_mean_weights = pd.DataFrame(index=LHS.index)
        log_mean_values_df = pd.DataFrame(index=LHS.index)
        print('LHS:\n', LHS)
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
        print('LHS:\n', LHS_data)

        LHS_data = LHS_data.drop(cols_to_drop_, axis=1)
        print('LHS:\n', LHS_data)

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
        """[summary]

        Args:
            data_dict (dict): raw data (all sector) containing
                              with nesting matching that of the
                              sub_categories dict up to (and sometimes
                              including) the innermost dictionary
                              (which then contains variable specific
                              keys and data)

            path (list): "path" (of keys) to dataframes in the data_dict
            variable (str): Desired data variable e.g. A_i_k

        Returns:
            data [pd.DataFrame]: Data at the end of the path for variable
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
            current [list]: List of lists (each inner list containing
                            a path)
        """
        for a, b in d.items():
            yield current+[a]
            if isinstance(b, dict):
                yield from self.get_paths(b, current+[a])
            elif isinstance(b, list):
                for i in b:
                    yield from self.get_paths(i, current+[a])

    def aggregate_data(self, raw_data, subscripts,
                       variable, sub_categories,
                       lhs_data=None, lhs_sub_names=None):
        """[summary]

        Args:
            raw_data (dict): Nested dictionary containing variable
                             keys and dataframes values in innermost
                             dictionary values. Outer nesting should match
                             sub_categories nesting.
            subscripts (list): Subscripts assigned to variable e.g. [i, k]
            variable (str): variable (datatype) e.g. A_i_k
            sub_categories (dict): Nested dictionary describing relationships
                                   between levels of aggregation in data
            lhs_data (dict, optional): Dictionary of dataframes of left hand side
                                       variable keys are 'paths'. Defaults to None.
            lhs_sub_names (dict, optional): keys are subscripts associated with the
                                            LHS variable, values are lists of (str)
                                            names associated with the subscript.
                                            Defaults to None.

        Returns:
            paths_dict (dict): Dictionary of variable data with paths as keys
                               and variable+path DataFrame as values
        """
        paths_dict = dict()
        paths = list(self.get_paths(sub_categories))
        print('paths:', paths)
        paths_sorted = sorted(paths, key=len, reverse=True)
        print('paths_sorted:', paths_sorted)

        raw_data_paths = list(self.get_paths(raw_data))
        print('raw_data_paths paths:', raw_data_paths)
        raw_data_paths_sorted = sorted(raw_data_paths, key=len, reverse=True)
        print('raw_data_paths_sorted paths:', raw_data_paths_sorted)
        # print('raw_data:\n', raw_data)
        print('variable:', variable)

        raw_data_paths_sorted_short = [p[:-1] for p in raw_data_paths_sorted]
        print('raw_data_paths_sorted_short:', raw_data_paths_sorted_short)
        print('\n \n \n')
        missing_paths_raw = [p for p in paths_sorted if p not in raw_data_paths_sorted_short]
        print('missing_paths_raw:\n', missing_paths_raw)
        # if len(missing_paths_raw) > 0:
        #     raise ValueError(f'keys {missing_paths_raw} not in raw_data')
        missing_paths_paths = [p for p in raw_data_paths_sorted_short if p not in paths_sorted]
        print('missing_paths_paths:\n', missing_paths_paths)

        print('raw_data:\n', raw_data)
        raw_data_paths_sorted = \
            [p for p in raw_data_paths_sorted if p[-1] == variable]
        print('raw_data_paths_sorted paths:', raw_data_paths_sorted)

        for p in raw_data_paths_sorted:
            print('p:', p)
            base_data = self.dict_iter(raw_data, p)
            print('base_data:\n', base_data)
            if len(p) == 1:
                p = [self.total_label]
            elif len(p) > 1:
                p = p[:-1]
            if base_data is None:
                continue
            if isinstance(base_data, pd.DataFrame):
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

                    paths_dict.update(sub_dict)
                else:
                    p_str = '.'.join(p)
                    paths_dict[p_str] = base_data
            else:
                raise ValueError('base data is type', type(base_data))
        print('paths_dict:\n', paths_dict)
        print('paths_dict keys:\n', paths_dict.keys())
        if len(paths_dict) == 0:
            raise ValueError('paths dict is empty')
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
        print('len_dict:\n', len_list)

        reverse_len = sorted(key_range, reverse=True)
        for n in reverse_len:
            print('n:', n)
            n_lists = []
            paths = len_dict[n]
            if len(paths) > 1:
                for i, p in enumerate(paths):
                    p_list = p.split('.')
                    path_list_short = p_list[:-1]
                    p_short_data = [p]
                    other_p = paths[:i] + paths[(i+1):]
                    print('other_p:\n', other_p)
                    for k, j in enumerate(other_p):
                        print('j:', j)
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
            print('level_data.keys()):\n', level_data.keys())
            if n > 1:
                higher_keys = len_dict[n-1]
                print('higher_keys:\n', higher_keys)
                for g in list(level_data.keys()):
                    print('g:', g)
                    higher_keys.append(g)
                len_dict[n-1] = higher_keys

            paths_dict.update(level_data)

        return paths_dict

    def group_data(self, path_list, data_dict, variable,
                   lhs_data, lhs_sub_names):
        """[summary]

        Args:
            path_list ([type]): [description]
            data_dict ([type]): [description]
            variable (str): variable (e.g. A_i_k)
            lhs_data (dict, optional): Dictionary of dataframes of left hand side
                               variable keys are 'paths'. Defaults to None.
            lhs_sub_names (dict, optional): keys are subscripts associated with the
                                            LHS variable, values are lists of (str)
                                            names associated with the subscript.
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
            print('base_path:', base_path)
            print('self.total_label:', self.total_label)
            if len(base_path) > 1:
                level_path = base_path[:-1]  # [self.total_label] +
                level_path = '.'.join(level_path)
            elif len(base_path) == 1:
                level_path = self.total_label

            for path in grouped_lists:
                print('path:', path)
                key = path.split('.')[-1]
                data = data_dict[path]

                if keep_cols:
                    lower_level_data = data
                else:
                    if lhs_data is not None:
                        try:
                            lhs_df = lhs_data[path]
                            weights = \
                                self.calculate_weights(lhs_df, key)
                            print('lhs_df:\n', lhs_df)
                            print('type lhs_df:\n', type(lhs_df))
                        except KeyError:
                            weights = None

                    else:
                        raise ValueError('LHS data not provided ' +
                                         'to group data method')
                    subscript = 'i'
                    lower_level_data = \
                        self.aggregate_level_data(subscript,
                                                  weights=weights,
                                                  base_data=data,
                                                  total_name=key)
                all_level.append(lower_level_data)
            try:
                level_data = \
                    df_utils().merge_df_list(all_level, keep_cols)
            except Exception as e:
                print('all_level:\n', all_level)
                raise e
            print('level_path:\n', level_path)
            n_dict[level_path] = level_data
        # exit()
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
            term_df [pd.DataFrame]: df of e.g. A_i_k data with
                                    i and k multiindex levels
        """
        subs = term_piece.split('_')  # list
        base_var = subs.pop(0)
        variable_data = input_data[base_var]

        for key_path in variable_data.keys():
            print('key_path:\n', key_path)
            if not(key_path.startswith('total')):
                print('not done')
                new_label = f'total.{self.total_label}.{key_path}'
                if key_path.startswith(self.total_label):
                    print('not national')
                    new_label = f'total.{key_path}'
                    print('new_label:', new_label)
                if key_path == 'South.Multi-Family':
                    new_label = 'total.National.South.Multi-Family'
                elif key_path == 'South.Single-Family':
                    new_label = 'total.National.South.Single-Family'
                elif key_path == 'Midwest.Manufactured-Homes':
                    new_label = 'total.National.Midwest.Manufactured-Homes'
                elif key_path == 'Midwest.Multi-Family':
                    new_label = 'total.National.Midwest.Multi-Family'
                variable_data[new_label] = variable_data.pop(key_path)

        print('variable_data.keys()', variable_data.keys())

        print('variable data:\n', variable_data)
        print('variable data keys:\n', variable_data.keys())
        print('base_var:', base_var)
        print('subs:', subs)
        subscripts = subscript_data[base_var]  # dictionary
        base_path = 'total'
        term_piece_dfs = []
        subs_short = subs[:-1]

        if len(subs) == 0:
            if 'total' in variable_data.keys():
                path = base_path
                path_df = variable_data[path]
                print('path_df subs 0:\n', path_df)
                term_piece_dfs.append(path_df)
            elif len(variable_data) == 1:
                path = list(variable_data.keys())[0]
                path_df = variable_data[path]
                return path_df

        base_path = base_path + '.' + self.total_label
        if len(subs) == 1:
            path = base_path
            path_df = variable_data[path]
            print('path_df subs 1:\n', path_df)
            combo_list = base_path.split('.')
            cols = list(path_df.columns)
            levels = [[c]*len(cols) for c in combo_list] + [cols]
            midx = pd.MultiIndex.from_arrays(levels)
            path_df.columns = midx
            print('path_df subs 1 multi:\n', path_df)

            term_piece_dfs.append(path_df)

        elif len(subs) > 1:  # len(subs_short)
            p_names = [subscripts[p] for p in subs_short]  # list of lists of names
            print('p_names:', p_names)
            combinations = list(itertools.product(*p_names))
            print('combinations:', combinations)

            for combo in combinations:
                combo_list = base_path.split('.') + list(combo)
                print('combo_list:', combo_list)
                # path_n_1 = '.'.join(combo_list[:-1])
                path = '.'.join(combo_list)
                print('path:', path)
                if path in variable_data:  # path_n_1
                    print('path in variable data')
                    path_df = variable_data[path]  # path_n_1
                    print('path_df subs > 1:\n', path_df)

                    # labels = []
                    # for c in path_df.columns:
                    #     idx = combo_list + [c]
                    #     idx = tuple(idx)
                    #     labels.append(idx)

                    # midx = pd.MultiIndex.from_tuples(labels)
                    # path_df.columns = midx
                    cols = list(path_df.columns)
                    levels = [[c]*len(cols) for c in combo_list] + [cols]   # combo should be combo_list
                    print('levels', levels)
                    # labels = [[0]*path_df.shape[1] for c in list(combo)]
                    # labels = labels + [list(range(len(path_df.columns)))]
                    # midx = pd.MultiIndex(levels=levels, codes=labels)
                    midx = pd.MultiIndex.from_arrays(levels)
                    path_df.columns = midx
                    print('path_df subs > 1 multi:\n', path_df)

                    term_piece_dfs.append(path_df)

        # term_df = pd.concat(term_piece_dfs, axis=0)
        term_df = df_utils().merge_df_list(term_piece_dfs)
        print('term_df:\n', term_df)
        return term_df

    def aggregate_level_data(self, subscript, weights, base_data, total_name):
        """Aggregate data for variable and level (e.g. region)

        Args:
            subscript (str): e.g. i
            weights (pd.DataFrame): LMDI weights
            base_data (pd.DataFrame): data to aggregate
            total_name (str): Name of aggregated data (column)

        Returns:
            total_col [pd.DataFrame]: n x 1 df of aggregated data
                                     (sum or weighted average if
                                     column data units vary)
        """
        units = self.subscripts[subscript]['names'].values()

        if self.all_equal(units):
            total_df = \
                df_utils().create_total_column(
                    base_data,
                    total_label=total_name)
            total_col = total_df[[total_name]]

        else:
            if weights is None:
                raise ValueError('Weights not available at ' +
                                 'level of aggregation')
            try:
                base_data, weights = \
                    df_utils().ensure_same_indices(base_data, weights)
                print('base_data:\n', base_data)
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
            lhs (pd.DataFrame): Dataframe containing data for the left hand side
                                variable of the decomposition equation
            name (str): level name for use in aggregation (not important, dropped)

        Returns:
            weights (pd.DataFrame): Log-Mean Divisia Weights (normalized)
        """
        lhs_total = df_utils().create_total_column(lhs,
                                                   total_label=name)
        lhs_share = df_utils().calculate_shares(lhs_total,
                                                total_label=name)

        if self.model == 'additive':
            weights = self.additive_weights(lhs, lhs_share)
        elif self.model == 'multiplicative':
            weights = self.multiplicative_weights(lhs, lhs_share)

        return weights

    def process_terms(self, input_data, subscript_data, weights, name):
        """From level data, calculate terms and weight them.

        Args:
            input_data (dict): [description]
            subscript_data (dict): [description]
            weights (pd.DataFrame): [description]
            name (level_name): [description]

        Returns:
            results (pd.DataFrame): [description]
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
            print('numerator:\n', numerator)

            for i in range(1, len(parts)):
                denominator_part = parts[i]
                denominator = part_data_dict[denominator_part]
                print('denominator:\n', denominator)

                numerator, denominator = \
                    df_utils().ensure_same_indices(numerator, denominator)
                numerator_levels = numerator.columns.nlevels
                print('numerator_levels', numerator_levels)
                try:
                    denominator_levels = denominator.columns.nlevels
                except ValueError:
                    denominator_levels = 0
                print('denominator_levels', denominator_levels)

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

                print('level_count', level_count)

                print("grouped numerator:\n", numerator.groupby(level=shared_levels,
                                              axis=1).sum())
                numerator.to_csv('C:/Users/irabidea/Desktop/yamls/numerator.csv')
                denominator.to_csv('C:/Users/irabidea/Desktop/yamls/denominator.csv')
                if group_:
                    numerator = numerator.groupby(level=shared_levels,
                                                  axis=1).sum()
                    print('numerator grouped:\n', numerator)
                try:
                    numerator = numerator.divide(denominator, axis=1)
                except ValueError:
                    den = denominator.groupby(level=shared_levels,
                                              axis=1).sum()
                    numerator = numerator.divide(den, axis=1)
                print('divided:\n', numerator)
                if t == 'E_i_j/E_i':
                    exit()
            f = numerator.copy()
            # else:
                # f = input_data[t]
            print('f:\n', f)
            f_levels = f.columns.nlevels
            print('f_levels', f_levels)

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
 
            print('component:\n', component)
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
            lhs_data (dict, optional): Dictionary of dataframes of left hand side
                                       variable keys are 'paths'. Defaults to None.
            lhs_sub_names (dict, optional):  keys are subscripts associated with the
                                            LHS variable, values are lists of (str)
                                            names associated with the subscript.
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
        Raises:
            ValueError: [description]
        Returns:
            results (dataframe): LMDI decomposition results
        """
        print('gen expr attributes:', dir(self))
        self.check_eval_str(self.decomposition)

        for t in self.terms:
            self.check_eval_str(t)

        self.test_expression(self.decomposition, self.LHS_var)

        print('raw_data.keys()', raw_data.keys())
        input_data = dict()
        all_subscripts = dict()

        vars_ = self.variables
        lhs_idx = vars_.index(self.LHS_var)
        lhs_ = vars_.pop(lhs_idx)
        lhs_data, lhs_sub_names = \
            self.nest_var_data(raw_data,
                               lhs_, sub_categories)

        for v in vars_:
            var_name = v.split('_')[0]

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

        print('lhs_data.keys():', lhs_data.keys())
        print('name:', name)
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
                                     weights, name)
        print('results:\n', results)
        exit()

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
        # if output_directory:
        #     try:
        #         plt.savefig(f"{output_directory}/{fig_name}.png")
        #     except FileNotFoundError:
        #         plt.savefig(f".{output_directory}/{fig_name}.png")
        plt.show()

    def main(self, input_data, sub_categories):
        """Calculate LMDI decomposition

        Args:
            input_data (dict): Dictionary containing dataframes
                               for each variable defined in the YAML
        """
        results = self.general_expr(input_data, sub_categories)
        print('results:\n', results)
        exit()
        if self.model == 'multiplicative':
            self.spaghetti_plot(data=results)

        formatted_results = self.prepare_for_viz(results)
        print('formatted_results:\n', formatted_results)
        return formatted_results

    @staticmethod
    def example_input_data():
        """Collect dictionary containing dataframes
        for each variable in the LMDI model
        """
        activity = \
            pd.read_csv('C:/Users/irabidea/Desktop/yamls/industrial_activity.csv').set_index('Year')
        energy = \
            pd.read_csv('C:/Users/irabidea/Desktop/yamls/industrial_energy.csv').set_index('Year')
        emissions = \
            pd.read_csv('C:/Users/irabidea/Desktop/yamls/industrial_energy.csv').set_index('Year')
        print('energy cols:', energy.columns)

        data = {'E_i_j': energy,
                'A_i': activity,
                'C_i_j': emissions,
                'total_label': 'NonManufacturing'}
        return data


if __name__ == '__main__':
    directory = 'C:/Users/irabidea/Desktop/yamls/'
    symb = GeneralLMDI(directory)
    """fname (str): Name of YAML file containing
                         LMDI input parameters
    """
    fname = 'combustion_noncombustion_test'  # 'test1'
    symb.read_yaml(fname)
    input_data = symb.example_input_data()
    expression = symb.main(input_data=input_data)
    # subs_ = symb.eval_expression()
    # c = IndexedVersion(directory=directory).main(fname='test1')
