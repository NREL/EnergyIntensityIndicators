
import sympy as sp
import numpy as np
import pandas as pd
import yaml
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

    def build_nest(self, data, select_categories, results_dict,
                   previous_level, variable, level1_name, level_name=None):
        """Process and organize raw data"""

        if isinstance(select_categories, dict):
            level_results = dict()
            for key, value in select_categories.items():
                print('select_categories:\n', select_categories)
                print('data keys:\n', data.keys())

                if isinstance(value, dict):
                    yield from self.build_nest(data=data[key],
                                               select_categories=value,
                                               results_dict=results_dict,
                                               previous_level=level_name,
                                               variable=variable,
                                               level1_name=level1_name,
                                               level_name=key)
                elif value is None:

                    print('final_data:\n', data)
                    print('final_data keys:\n', data.keys())
                    try:
                        final_data = data[variable]
                    except KeyError:
                        final_data = data[key][variable]

                    print('final_data:\n', final_data)
                    if isinstance(final_data, pd.DataFrame):
                        level_results[key] = final_data
                    else:
                        raise ValueError(f'Final data is type {type(final_data)}')

        if len(level_results) == 0:
            yield results_dict
        elif len(level_results) == 1:
            level_data = list(level_results.values())[0]
        elif len(level_results) > 1:
            results_higher = \
                [df_utils().create_total_column(df, k)[[k]]
                 for k, df in level_results.items()]
            level_data = \
                df_utils().merge_df_list(results_higher)

        results_dict[level_name] = level_data

        yield results_dict

    # def nesting(select_categories, results_dict):
    #     for k, v in select_categories.items():
    #         level_data = dict()
    #         if isinstance(v, dict):
    #             for l_, lower_data in v.items():
    #                 if l_ in results_dict:
    #                     level_data[l_] = results_dict[l_]
    #                 else:
    #                     self.nesting(v, results_dict)
    #         if k in results_dict
    def nesting(self, data, select_categories, results_dict,
                previous_level, variable, level1_name, level_name=None):
        """Process and organize raw data"""

        if isinstance(select_categories, dict):
            results_higher = []
            for key, value in select_categories.items():
                print('select_categories:\n', select_categories)
                print('data keys:\n', data.keys())
                if isinstance(value, dict):
                    results_higher = \
                        [df_utils().create_total_column(results_dict[k], k)[[k]]
                            for k in value.keys() if k in results_dict]
                    if len(results_higher) > 1:
                        results_df_higher = \
                            df_utils().merge_df_list(results_higher)
                    elif len(results_dict) == 1:
                        results_df_higher = results_higher.copy()
                    else:
                        results_df_higher = None

                    if results_df_higher is not None:
                        if previous_level in results_dict:
                            existing_results = results_dict[previous_level]
                            results_dict[previous_level] = \
                                df_utils().merge_df_list([results_df_higher,
                                                         existing_results])
                        else:
                            results_dict[previous_level] = results_df_higher
                    try:
                        data = data[key]
                    except KeyError:
                        continue

                    yield from self.nesting(data=data,
                                            select_categories=value,
                                            results_dict=results_dict,
                                            previous_level=level_name,
                                            variable=variable,
                                            level1_name=level1_name,
                                            level_name=key)
                elif value is None:
                    yield results_dict

    def group_lower(self, i, subscripts, results_dict, categories):

        lower_names = \
            list(self.subscripts[subscripts[i+1]]['names'].keys())
        lower_names = \
            [l_ for l_ in lower_names if l_ in categories.keys()]
        lower = \
            [df_utils().create_total_column(results_dict[l], l)[[l]]
                for l in lower_names]
        lower_df = df_utils().merge_df_list(lower)

        return lower_df

    def dict_iter(self, dict_, after):
        for a in after:
            if isinstance(dict_, dict):
                if isinstance(dict_[a], dict):
                    r = dict_[a]
                    yield from self.dict_iter(dict_[a], a)
                else:
                    r = a
            else:
                r = a
            yield r

    def aggregate_data(self, raw_data, subscripts, variable, sub_categories):

        results_dict = dict()
        for results_dict in self.build_nest(data=raw_data,
                                            select_categories=sub_categories,
                                            results_dict=results_dict,
                                            previous_level=self.sector,
                                            variable=variable,
                                            level1_name=self.total_label):
            continue

        print('results_dict:\n', results_dict)
        subscripts_ = subscripts[:-1]
        subscripts_.reverse()
        lowest = subscripts[-1]
        second_highest = subscripts[1]
        highest = subscripts[0]
        for i, s in enumerate(subscripts_):
            names = list(self.subscripts[s]['names'].keys())
            for n in names:
                print('subcategories:\n', sub_categories)
                print('name:\n', n)
                after = subscripts_[i:]
                after.reverse()
                for cats in self.dict_iter(sub_categories, after):
                    continue

                categories = cats

                if s == lowest:
                    results_dict = results_dict
                elif i > subscripts_.index(highest) and \
                        i < subscripts_.index(lowest):
                    lower_df = self.group_lower(i, subscripts_,
                                                results_dict, categories)
                    results_dict[n] = lower_df

                elif s == highest:
                    highest_level = \
                        [df_utils().create_total_column(results_dict[n], n)[[n]]
                            for n in names]
                    highest_df = df_utils().merge_df_list(highest_level)
                    results_dict[self.total_label] = highest_df

        # final_input = {results_dict[level] for level in [self.total_label, second_highest]}
        return results_dict[self.total_label]

    @staticmethod
    def process_term(t, input_data):

        try:
            input_data['A'] = \
                input_data['A'][['floorspace_bsf']].rename(
                    columns={'floorspace_bsf': 'A'})
        except KeyError:
            pass

        try:
            input_data['WF'] = \
                input_data['WF'][['fuels_weather_factor']].rename(
                    columns={'fuels_weather_factor': 'WF'})
        except KeyError:
            pass

        if '/' in t:
            parts = t.split('/')
            first_df = parts[0]
            first_df = input_data[first_df]
            numerator = first_df.copy()

            for i in range(1, len(parts)):
                denominator = parts[i]
                denominator = input_data[denominator]
                numerator, denominator = \
                    df_utils().ensure_same_indices(numerator, denominator)
                print('numerator:\n', numerator)
                print('denominator:\n', denominator)
                numerator = numerator.divide(denominator.values, axis=0)

            f = numerator
        else:
            f = input_data[t]

        return f

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
        for v in self.variables:
            subscripts = v.split('_')[1:]
            if len(subscripts) > 1:
                v_data = \
                    self.aggregate_data(raw_data, subscripts,
                                        v, sub_categories)

                var_name = v.split('_')[0]
                input_data[var_name] = v_data

                sub_names = {s: self.subscripts[s]['names'].keys()
                             for s in subscripts}
                all_subscripts[var_name] = sub_names
            else:
                input_data.update({v: raw_data[v]})
        print('input_data:\n', input_data.keys())
        for k in input_data.keys():
            print('input_data key keys:\n', input_data[k].keys())

        name = self.total_label

        try:
            lhs = input_data[self.LHS_var.split('_')[0]][name]

        except KeyError:
            # 'Commercial_Total', 'National', 'Industry''Northeast'
            # print('input_data["National"].keys():', input_data["National"].keys())
            # print('input_data["Northeast"].keys():', input_data["Northeast"].keys())
            input_data = input_data[name]
            lhs = input_data[self.LHS_var]
            print("input_data[temp_label]:", input_data)

        print('lhs:\n', lhs)
        lhs_total = df_utils().create_total_column(lhs,
                                                   total_label=name)
        print('lhs_total:\n', lhs_total)
        lhs_share = df_utils().calculate_shares(lhs_total,
                                                total_label=name)
        print('lhs_share:\n', lhs_share)

        if self.model == 'additive':
            weights = self.additive_weights(lhs, lhs_share)
        elif self.model == 'multiplicative':
            weights = self.multiplicative_weights(lhs, lhs_share)

        totals = list(self.totals.keys())
        sorted_totals = sorted(totals, key=len, reverse=True)
        for total in sorted_totals:
            cols = self.totals[total]
            cols_subscript = cols.split('_')
            total_subscript = total.split('_')
            subscripts = [s for s in cols_subscript
                          if s not in total_subscript]

            if len(subscripts) == 1:
                subscript = subscripts[0]
            else:
                raise ValueError('Method not currently able to accomodate'
                                 'summing over multiple subscripts')

            units = self.subscripts[subscript]['names'].values()
            print('units:', units)
            sub_dfs = []
            sub_names = self.subscripts[subscript]['names'].keys()
            for s in sub_names:
                if self.all_equal(units):
                    print('total:\n', total)
                    total_base_var = total.split('_')[0]
                    print('total_base_var:\n', total_base_var)

                    total_df = \
                        df_utils().create_total_column(
                            input_data[total_base_var][s],
                            total_label=s)
                    print('total_df:\n', total_df)
                    total_col = total_df[[s]]
                    print('total_col:\n', total_col)

                else:
                    total_col = input_data[s].multiply(weights.values,
                                                        axis=1).sum(axis=1)
                sub_dfs.append(total_col)
            
            total_data = df_utils().merge_df_list(sub_dfs)
            input_data[total] = total_data

        results = pd.DataFrame(index=lhs.index)
        print('self.decomposition:', self.decomposition)

        results = []
        for t in self.decomposition.split('*'):

            f = self.process_term(t, input_data)

            if t in self.terms:
                print('t in terms!')
                print(f'f {t}:\n', f)
                print('f cols:', f.columns)
                print('weights cols:', weights.columns)
                if f.shape[1] > 1:
                    if f.shape[1] == weights.shape[1]:
                        if name in f.columns:
                            f = f.drop(name, axis=1, errors='ignore')
                        component = f.multiply(weights.values, axis=1).sum(axis=1)
                    elif f.shape[1] > 1:
                        if name in f.columns:
                            f = f[[name]]
                        else:
                            f = df_utils().create_total_column(f, name)[[name]]
                        component = f
                else:
                    component = f

                if isinstance(component, pd.Series):
                    component = component.to_frame(name=t)
                print(f'component {t}:\n', component)
            else:
                component = f
                print(f'{t} not in terms')

            print(f'component {t}:\n', component)
            print('component type', type(component))
            print('type t:', type(t))

            if component.shape[1] == 2 and name in component.columns:
                component = component.drop(name, axis=1, errors='ignore')
                print(f'component {t}:\n', component)

            results.append(component)

        results = df_utils().merge_df_list(results)
        results = results.drop('Commercial_Total', axis=1, errors='ignore')
        results = results.rename(columns=self.term_labels)
        print('results:\n', results)
        if self.model == 'additive':
            expression = self.decomposition_additive(results)
        elif self.model == 'multiplicative':
            results = df_utils().calculate_log_changes(results)
            expression = self.decomposition_multiplicative(results)

        print('expression:\n', expression)
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
