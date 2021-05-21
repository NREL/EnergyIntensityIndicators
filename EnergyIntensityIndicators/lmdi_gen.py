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
            raise ValueError(('Decomposition expression does not simplify'
                              'to LHS variable:'
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

    def general_expr(self, input_data):
        """Decompose changes in LHS variable

        Args:
            input_data (dict): Dictionary containing
                               dataframes for each variable
                               and a the total label

        Raises:
            ValueError: [description]

        Returns:
            results (dataframe): LMDI decomposition results
        """
        print('gen expr attributes:', dir(self))
        self.check_eval_str(self.decomposition)
        # if hasattr('GeneralLMDI', 'terms'):
        #     terms_ = True
        for t in self.terms:
            self.check_eval_str(t)
        # else:
        #     terms_ = False

        self.test_expression(self.decomposition, self.LHS_var)
        print('input_data.keys()', input_data.keys())
        try:
            lhs = input_data[self.LHS_var]
            name = input_data['total_label']
        except KeyError:
            temp_label = 'Industry'  # 'Commercial_Total', 'National', 'Industry''Northeast'
            # print('input_data["National"].keys():', input_data["National"].keys())
            # print('input_data["Northeast"].keys():', input_data["Northeast"].keys())

            input_data = input_data[temp_label]
            lhs = input_data[self.LHS_var]
            print("input_data[temp_label]:", input_data)
            try:
                name = input_data['total_label']
            except KeyError:
                name = temp_label

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
            if self.all_equal(units):
                total_df = \
                    df_utils().create_total_column(input_data[cols],
                                                   total_label=name)
                print('total_df:\n', total_df)
                total_col = total_df[[name]]
                print('total_col:\n', total_col)

            else:
                total_col = input_data[cols].multiply(weights.values,
                                                      axis=1).sum(axis=1)

            input_data[total] = total_col

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
                    if name in f.columns:
                        f = f.drop(name, axis=1, errors='ignore')
                    component = f.multiply(weights.values, axis=1).sum(axis=1)
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

    def main(self, input_data):
        """Calculate LMDI decomposition

        Args:
            input_data (dict): Dictionary containing dataframes
                               for each variable defined in the YAML
        """
        results = self.general_expr(input_data)
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
