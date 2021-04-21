import sympy as sp
import numpy as np
import pandas as pd
import yaml

from EnergyIntensityIndicators.utilites import dataframe_utilities as df_utils
from EnergyIntensityIndicators.utilites import lmdi_utilities


class GeneralLMDI:
    """Class to decompose changes in a variable using symbolic matrices

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
    def __init__(self, directory):
        self.directory = directory

    def create_yaml(self, fname):
        input_ = {'variables': ['E_i', 'A_i'],
                  'LHS_var': 'E_i',
                  'decomposition': 'A*A_i/A*E_i/A_i',
                  'terms': ['A', 'A_i/A', 'E_i/A_i'],
                  'model': 'multiplicative',
                  'lmdi_type': 'II',
                  'totals': {'A': 'sum(A_i)'},
                  'subscripts': {'i': {'names':
                                            ['Northeast', 'Midwest', 'South', 'West'],
                                    'count': 4}},
                  'energy_types': ['source', 'deliv', 'elec', 'fuels'],
                  'base_year': 1990,
                  'end_year': 2018}

        with open (f'{self.directory}/{fname}.yaml', 'w') as file:
            yaml.dump(input_, file)

    def read_yaml(self, fname):
        """Read yaml input data
        """
        with open(f'{self.directory}/{fname}.yaml', 'r') as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            input_dict = yaml.load(file, Loader=yaml.FullLoader)
            print('input_dict:\n', input_dict)
            for k, v in input_dict.items():
                setattr(GeneralLMDI, k, v)
    
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
            raise ValueError('Decomposition expression does not simplify to LHS variable')
    
    @staticmethod
    def check_eval_str(s):
        """From NREL rev.rev.utilities.utilities (properly import/cite?)
        
        Check an eval() string for questionable code.
        Parameters
        ----------
        s : str
            String to be sent to eval(). This is most likely a math equation to be
            evaluated. It will be checked for questionable code like imports and
            dunder statements.
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
        replicating methodology in PNNL spreadsheets for the multiplicative model
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

        results['effect'] = results.product(axis=1)

        results["@filter|Measure|BaseYear"] = self.base_year
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
        for col in LHS.columns:
            LHS[f"{col}_shift"] = LHS[col].shift(
                                      periods=1, axis='index', fill_value=0)

            # apply generally not preferred for row-wise operations but?
            log_mean_values = \
                LHS[[col, f"{col}_shift"]].apply(lambda row:
                                                 lmdi_utilities.logarithmic_average(row[col],
                                                 row[f"{col}_shift"]),
                                                 axis=1)

            log_mean_values_df[col] = log_mean_values.values

            LHS_share[f"{col}_shift"] = LHS_share[col].shift(periods=1, axis='index', fill_value=0)
            # apply generally not preferred for row-wise operations but?
            log_mean_shares = \
                LHS_share[[col, f"{col}_shift"]].apply(lambda row:
                                                       lmdi_utilities.logarithmic_average(row[col],
                                                       row[f"{col}_shift"]),
                                                       axis=1)

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
        LHS = LHS.drop(cols_to_drop_, axis=1)

        if self.lmdi_type == 'LMDI-I':
            return log_mean_values_df

        elif self.lmdi_type == 'LMDI-II':
            sum_log_mean_shares = LHS_share[log_mean_shares_labels].sum(axis=1)
            log_mean_weights_normalized = \
                log_mean_weights.divide(
                    sum_log_mean_shares.values.reshape(len(sum_log_mean_shares),
                                                       1))

            log_mean_weights_normalized = \
                log_mean_weights_normalized.drop(
                    [c for c in log_mean_weights_normalized.columns
                     if not c.startswith('log_mean_weights_')], axis=1)
            return log_mean_weights_normalized
            
        else:
            return log_mean_values_df

    def calculate_effect_additive(self, terms_df):
        """Calculate effect from changes to activity, structure, 
        and intensity in the additive model
        """

        terms_df['effect'] = terms_df.sum(axis=1)

        return terms_df

    def decomposition_additive(self, terms_df):
        """Format component data, collect overall effect,
        return aggregated dataframe of the results for
        the additive LMDI model.
        """
        # ASI.pop('lower_level_structure', None)

        df = self.calculate_effect_additive(terms_df)
        df = df.reset_index()
        if 'Year' not in df.columns:
            df = df.rename(columns={'index': 'Year'})

        return df

    def general_expr(self, input_data):

        self.check_eval_str(self.decomposition)
        for t in self.terms:
            self.check_eval_str(t)
    
        self.test_expression(self.decomposition, self.LHS_var)

        for total, cols in self.totals.items():
            name = input_data['total_label']
            total_df = \
                df_utils.create_total_column(input_data[cols],
                                             total_label=name)
            total_col = total_df[[name]]
            input_data[total] = total_col

        lhs = input_data[self.LHS_var]
        print('lhs:\n', lhs)
        lhs_total = df_utils.create_total_column(lhs,
                                                 total_label=name)
        print('lhs_total:\n', lhs_total)
        lhs_share = df_utils.calculate_shares(lhs_total,
                                              total_label=name)
        print('lhs_share:\n', lhs_share)

        if self.model == 'additive':
            weights = self.additive_weights(lhs, lhs_share)
        elif self.model == 'multiplicative':
            weights = self.multiplicative_weights(lhs, lhs_share)

        results = pd.DataFrame(index=lhs.index)

        for t in self.decomposition.split('*'):
            if '/' in t:
                parts = t.split('/')
                numerator = parts[0]
                numerator = input_data[numerator]
                denominator = parts[1]
                denominator = input_data[denominator]
                numerator, denominator = \
                    df_utils.ensure_same_indices(numerator, denominator)
                print('numerator:\n', numerator)
                print('denominator:\n', denominator)

                f = numerator.divide(denominator.values, axis=0)
            else:
                f = input_data[t]
            
            if t in self.terms:
                print(f'f {t}:\n', f)
                component = f.multiply(weights.values, axis=1).sum(axis=1)
                print(f'component {t}:\n', component)

            else:
                component = f

            results[t] = component
            print(f'component {t}:\n', component)

        results = results.rename(columns=self.term_labels)
        print('results:\n', results)
        if self.model == 'additive':
            expression = self.decomposition_additive(results)
        elif self.model == 'multiplicative':
            expression = self.decomposition_multiplicative(results)

        print('expression:\n', expression)
        return expression

    def eval_expression(self):
        """Substitute actual data into the symbolic
        LMDI expression to calculate results

        Returns:
            final_result (?): LMDI results ?

        TODO:
            Should return pandas dataframe containing
            the relative contributions of each term to the
            change in the LHS variable, with appropriate column
            labels and years as the index
        """
        activity = \
            pd.read_csv('C:/Users/irabidea/Desktop/yamls/industrial_activity.csv').set_index('Year')
        energy = \
            pd.read_csv('C:/Users/irabidea/Desktop/yamls/industrial_energy.csv').set_index('Year')
        print('energy cols:', energy.columns)
        data = {'A_i': activity, 'E_i': energy, 'total_label': 'NonManufacturing'}
        expression_dict = self.general_expr(data)

    def main(self, fname):
        self.read_yaml(fname)
        self.eval_expression()


if __name__ == '__main__':
    directory = 'C:/Users/irabidea/Desktop/yamls/'
    symb = GeneralLMDI(directory)
    symb.read_yaml(fname='test1')
    expression = symb.main(fname='test1')
    # subs_ = symb.eval_expression()
    # c = IndexedVersion(directory=directory).main(fname='test1')
