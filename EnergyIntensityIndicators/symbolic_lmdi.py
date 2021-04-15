import sympy as sp
import numpy as np
import pandas as pd
import yaml


class SymbolicLMDI:
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
                setattr(SymbolicLMDI, k, v)

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
        assert lhs == sp.simplify(expression)

    @staticmethod
    def create_symbolic_var(variable, num_years, num_columns):
        """Create m X n symbolic matrix (populated by 
        variable name) where m is the number of years and n is the 
        number of columns (subsectors)

        Args:
            variable (str): Variable name
            num_years (int): The number of years in the dataset
            num_columns (int): The number of columns in the dataset
                               (subsectors)

        Returns:
            variable [Symbolic Matrix]: Matrix version of string variable
        """
        variable = sp.MatrixSymbol(str(variable), num_years, num_columns)
        return variable

    def hadamard_division(self, numerator, denominator):
        """Perform element-wise division of two
        symbolic matrices
        
        Note: I haven't found this functionality in
        the sympy library though it may exist or not exist for a
        good reason? (e.g. mathematical properties)

        Args:
            numerator (Symbolic Matrix): dividend
            denominator ([type]): divisor

        Returns:
            result (Symbolic Matrix): Quotient
        """

        print('numerator shape:\n', numerator.shape)
        print('denominator shape pre power:\n', denominator.shape)

        denominator = sp.HadamardPower(denominator, -1).doit().as_explicit()
        print('denominator shape:\n', denominator.shape)

        if numerator.shape == denominator.shape:
            result = sp.HadamardPower(denominator, 
                                      numerator).doit().as_explicit()
        elif numerator.shape[0] == denominator.shape[1]:
            result = denominator * numerator
        elif numerator.shape[1] > denominator.shape[1] \
                and denominator.shape[1] == 1:
            denominator = self.transform_col_vector(denominator)
            print('denominator shape after transform:\n', denominator.shape)

            result = denominator * numerator

        return result
    
    @staticmethod
    def shift_matrices(matrix):
        """Create two matrices from input matrix 
        (append row of ones to the beginning of one and 
        the end of the other) in order to find the 
        change between rows (years)

        Args:
            matrix (Symbolic Matrix): any matrix

        Returns:
            shift_term (Symbolic Matrix): matrix with rows shifted down
            long_term (Symbolic Matrix): matrix with row appended to end
                                         (so that the dimensions will match
                                         shift term)

        TODO: Is a row of ones the best way to do this?

        """
        print('matrix shape', matrix.shape)
        n = matrix.shape[0]  # Length
        shift = -1
        ones_V = [sp.Matrix([0, 1])] + [1] * (n-np.abs(shift) - 1)

        subdiag = sp.diag(*ones_V)
        subdiag = subdiag.col_insert(subdiag.shape[1],
                                     sp.zeros(subdiag.shape[0], 1))
        sp.pprint(subdiag)
        shift_term = subdiag * matrix
        return shift_term, matrix

    def create_symbolic_term(self, numerator, denominator):
        """Create LMDI RHS term e.g. the log change of structure
        (Ai/A) in symbolic matrix terms

        Args:
            numerator (Symbolic Matrix): Dividend (e.g. Ai)
            denominator (Symbolic Matrix): Divisor (e.g. A)

        Returns:
            term (Symbolic Matrix): the log change of the RHS term
        """
        base_term = self.hadamard_division(numerator, denominator)
        shift_term, long_term = self.shift_matrices(base_term)

        # find change (divide every row by previous row)
        term = sp.matrix_multiply_elementwise(long_term, shift_term)
        term = term.applyfunc(sp.log)
        return term

    @staticmethod
    def weighted_term(weight, term):
        """Calculate components from dot product of weights and term

        Args:
            weight (MatrixSymbol): normalized log-mean divisia weights
            term (MatrixSymbol): The effect of the component (RHS) variable
                                 on the change in the LHS variable
        """
        weighted_term = sp.matrix_multiply_elementwise(weight, term).doit()
        ones_ = sp.ones(weighted_term.shape[1], 1)
        component = weighted_term * ones_
        return component

    def logarithmic_average(self, x, y):
        """[summary]

        Args:
            x ([type]): [description]
            y ([type]): [description]

        Returns:
            [type]: [description]
        """
        if x != y:

            negative_y = y * -1
            log_x = x.applyfunc(sp.log)
            log_y = y.applyfunc(sp.log)
            negative_log_y = log_y * -1

            numerator = sp.MatAdd(x, negative_y)
            print('numerator shape', numerator.shape)
            denominator = sp.MatAdd(log_x, negative_log_y)
            print('denominator shape', denominator.shape)

            logarithmic_average = self.hadamard_division(numerator,
                                                         denominator)

        elif x == y:
            logarithmic_average = x

        return logarithmic_average

    def multiplicative_weights(self, log_mean_matrix,
                               log_mean_share, log_mean_share_total,
                               log_mean_total):
        """Calculate log-mean divisia weights for the multiplicative model
        in symbolic terms

        Args:
            log_mean_matrix ([type]): [description]
            log_mean_share ([type]): [description]
            log_mean_share_total ([type]): [description]

        Returns:
            [type]: [description]
        """        """[summary]
        """
        if self.lmdi_type == 'I':
            weights = log_mean_matrix

        elif self.lmdi_type == 'II':
            numerator = sp.matrix_multiply_elementwise(log_mean_share, log_mean_matrix)
            log_mean_total = self.transform_col_vector(log_mean_total)
            weights = self.hadamard_division(numerator, log_mean_total)
        
        return weights

    def additive_weights(self, log_mean_matrix,
                         log_mean_matrix_total,
                         log_mean_share):
        """Calculate log-mean divisia weights for the additive model
        in symbolic terms

        Args:
            log_mean_matrix ([type]): [description]
            log_mean_matrix_total ([type]): [description]
        """

        if self.lmdi_type == 'I':
            weights = log_mean_matrix
        elif self.lmdi_type == 'II':
            numerator = sp.matrix_multiply_elementwise(log_mean_share, log_mean_matrix)
            weights = self.hadamard_division(numerator, log_mean_matrix_total)
        
        return weights

    @staticmethod
    def transform_col_vector(V):
        print(V.shape)
        ones_vector = sp.ones(1, V.shape[0])
        print(ones_vector.shape)

        V = V * ones_vector
        V = V.T
        return V
            
    def calc_weights(self, lhs_matrix, num_columns):
        """Calculate log-mean divisia weights

        Args:
            lhs_matrix (Symbolic Matrix): Matrix representing the LHS variable
                                          (the variable to decompose) symbolicly

        Returns:
            weights (Symbolic Matrix): The log-mean divisia weights
        """
        lhs_total = lhs_matrix * sp.ones(lhs_matrix.shape[1], 1)
        print('lhs_total:', lhs_total)
        lhs_total = self.transform_col_vector(lhs_total)
        print('lhs_total tiled:', lhs_total)

        sp.pprint(lhs_total)
        print(lhs_total.shape)
        sp.pprint(lhs_matrix)
        print(lhs_matrix.shape)
        # lhs_total = sp.HadamardPower(lhs_total, -1)
        lhs_share = lhs_total * lhs_matrix
        print(lhs_share.shape)

        shift_matrix, long_matrix = self.shift_matrices(lhs_matrix)
        shift_share, long_share = self.shift_matrices(lhs_share)

        log_mean_matrix = self.logarithmic_average(long_matrix,
                                                   shift_matrix)
        log_mean_matrix_total = log_mean_matrix * sp.ones(
                                            log_mean_matrix.shape[1], 1)

        log_mean_share = self.logarithmic_average(long_share,
                                                  shift_share)
        log_mean_share_total = log_mean_share * sp.ones(
                                            log_mean_share.shape[1], 1)

        if self.model == 'additive':
            weights = self.additive_weights(log_mean_matrix,
                                            log_mean_matrix_total,
                                            log_mean_share_total)
        elif self.model == 'multiplicative':
            weights = self.multiplicative_weights(log_mean_matrix,
                                                  log_mean_share,
                                                  log_mean_share_total,
                                                  log_mean_matrix_total)
        return weights

    def LMDI_expression(self):
        """Calculate the LMDI equation in
        symbolic Matrix terms

        Returns:
            expressions (Symbolic Matrix): Matrix where each element contains
                                           a symbolic expression representing
                                           the appropriate calculation of the 
                                           value

        TODO: 
            - describe the expression more accurately
        """
        print('self.variables:', self.variables)
        num_years = self.end_year - self.base_year
        num_columns = self.subscripts['i']['count']

        # variable_dict = {var:
        #                  self.create_symbolic_var(var,
        #                                           num_years,
        #                                           num_columns)
        #                  for var in self.variables}
        
        activity = pd.read_csv('C:/Users/irabidea/Desktop/yamls/residential_activity.csv', index_col=0)
        activity = activity.loc[self.base_year:self.end_year, :]
        activity = sp.Matrix(activity.values)
        # sp.pprint(activity)
        # print(type(activity))
        # exit()
        energy = pd.read_csv('C:/Users/irabidea/Desktop/yamls/residential_energy.csv', index_col=0)
        energy = energy.loc[self.base_year:self.end_year, :]
        energy = sp.Matrix(energy.values)

        variable_dict = {'A_i': activity, 'E_i': energy}
        print('input data A:\n', variable_dict['A_i'])
        for t, s in self.totals.items():
            to_sum = variable_dict[s]
            variable_dict[t] = to_sum * sp.ones(to_sum.shape[1], 1)

        lhs_matrix = variable_dict[self.LHS_var]
        # lhs_matrix = self.eval_expression(lhs_matrix)
        print('lhs_matrix done')
        # lhs_matrix = lhs_matrix.as_explicit()
        print('lhs_matrix term literal')
        sp.pprint(lhs_matrix)
        print('type(lhs_matrix):', type(lhs_matrix))

        weights = self.calc_weights(lhs_matrix, num_columns)
        sp.pprint(weights)
        print('type(weights):', type(weights))
        symbolic_terms = []

        for term in self.terms:
            print('term:', term)
            if '/' in term:
                parts = term.split('/')
                numerator = parts[0]
                numerator = variable_dict[numerator]
                denominator = parts[1]
                denominator = variable_dict[denominator]

                matrix_term = self.create_symbolic_term(numerator, denominator)

            else:
                matrix_term = variable_dict[term]
            
            # matrix_term = self.eval_expression(matrix_term)
            print('matrix_term done')
            # matrix_term = matrix_term.as_explicit()
            print('matrix term literal')
            weighted_term = self.weighted_term(weights, matrix_term)
            symbolic_terms.append(weighted_term)
        
        decomposition_pieces = self.decomposition.split('*')
        not_weighted = [t for t in decomposition_pieces
                        if t not in self.terms]

        for n in not_weighted:  # need to add capability for more complicated unweighted terms
            symbolic_terms.append(variable_dict[n])

        # symbolic_terms = [self.eval_expression(s) for s in symbolic_terms]

        # print('done')
        # symbolic_terms = [s.as_explicit() for s in symbolic_terms]
        # print('well done')
        print("symbolic_terms:\n", symbolic_terms)
        if self.model == 'additive':
            expression = sp.MatAdd(*symbolic_terms).doit().as_explicit()
        elif self.model == 'multiplicative':
            expression = sp.MatMul(*symbolic_terms).doit().as_explicit()
        print('expression done')
        # expression = expression.as_explicit()
        print('expression literal')
        sp.pprint(expression)
        return expression
    
    def eval_expression(self, expression):
        """Substitute actual data into the symbolic
        LMDI expression to calculate results

        Returns:
            final_result (Matrix): LMDI results ?
        
        TODO: 
            Should return pandas dataframe containing
            the relative contributions of each term to the
            change in the LHS variable, with appropriate column
            labels and years as the index
        """        
        activity = pd.read_csv('C:/Users/irabidea/Desktop/yamls/residential_activity.csv', index_col=0)
        activity = activity.loc[self.base_year:self.end_year, :]
        activity = sp.Matrix(activity.values)
        # sp.pprint(activity)
        # print(type(activity))
        # exit()
        energy = pd.read_csv('C:/Users/irabidea/Desktop/yamls/residential_energy.csv', index_col=0)
        energy = energy.loc[self.base_year:self.end_year, :]
        energy = sp.Matrix(energy.values)

        input_data = {'A_i': activity, 'E_i': energy}
        print('input data A:\n', input_data['A_i'])
        # expression = self.LMDI_expression()
        input_dict = {sp.symbols(v):
                      input_data[v] for v in self.variables}
        final_result = expression.subs(input_dict)   #.as_explicit()
        sp.pprint(final_result)
        return final_result


class IndexedVersion:
#  {'variables': ['E_i', 'A_i'],
#      'LHS_var': 'E_i',
#      'decomposition': 'A*A_i/A*E_i/A_i',
#      'terms': ['A', 'A_i/A', 'E_i/A_i']
#      'model': 'multiplicative',
#      'lmdi_type': 'II',
#      'totals': {'A': 'sum(A_i)'},
#      'subscripts': {'i': {'names':
#                                  ['Northeast', 'Midwest', 'South', 'West'],
#                            'count': 4}},
#      'energy_types': ['source', 'deliv', 'elec', 'fuels']

#      'base_year': 1990,
#      'end_year': 2018}

    def __init__(self, directory):
        self.directory = directory
    
    @staticmethod
    def logarithmic_average(x, y):
        if x == y:
            L = x
        else:
            L = sp.exp((x - y) / (sp.log(x) - sp.log(y)))

        return L

    def read_yaml(self, fname):
        """Read yaml input data
        """
        with open(f'{self.directory}/{fname}.yaml', 'r') as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            input_dict = yaml.load(file, Loader=yaml.FullLoader)
            print('input_dict:\n', input_dict)
            for k, v in input_dict.items():
                setattr(IndexedVersion, k, v)

    def weights(self, lhs, t, i, m):
        lhs_total = sp.Sum(lhs[t, i], (i, 1, m))
        lhs_total_shift = sp.Sum(lhs[sp.exp(t-1), i], (i, 1, m))

        lhs_share = sp.exp(lhs[t, i] / lhs_total)
        lhs_share_shift = sp.exp(lhs[sp.exp(t-1), i] / lhs_total_shift)
        
        log_average_total = self.logarithmic_average(lhs_total,
                                                     lhs_total_shift)
        sp.pprint(log_average_total)
        log_average = self.logarithmic_average(lhs[t, i], lhs[sp.exp(t-1), i])
        sp.pprint(log_average)

        log_average_shares = self.logarithmic_average(lhs_share,
                                                      lhs_share_shift)
        sp.pprint(log_average_shares)

        total_log_average_shares = sp.Sum(log_average_shares, (i, 1, m))

        if self.model == 'multiplicative':
            if self.lmdi_type == 'I':
                weights = sp.exp(log_average_total / log_average_total)
            elif self.lmdi_type == 'II':
                weights = sp.exp(log_average_shares / total_log_average_shares)

        elif self.model == 'additive':
            if self.lmdi_type == 'I':
                weights = log_average
            elif self.lmdi_type == 'II':
                weights = sp.exp((log_average_shares * log_average_total) /
                                 total_log_average_shares)

        return weights.doit()

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

    def process_var(self, subscripts, vars_, var):
        subs_ = [subscripts[i] for i in self.subscripts.keys()
                 if f'_{i}' in var]

        if var in self.totals.keys():
            total_subs = [i for i in self.subscripts.keys()
                          if f'_{i}' in var]
            other_subs = [i for i in self.subscripts.keys()
                          if f'_{i}' in var]
            diff = [i for i in total_subs if i
                    not in other_subs]
            for d in diff:
                m = self.subscripts[d]['count']
                var = sp.Sum(var, (d, 1, m))

        var = vars_[var[0]]
        return var

    def general_expr(self):

        # for i in self.subscripts.keys():
        #     i = sp.symbols(str(i), cls=sp.Idx)
        #     print('i', i)
        self.check_eval_str(self.decomposition)
        for t in self.terms:
            self.check_eval_str(t)

        vars_ = {str(v): sp.IndexedBase(str(v)) for v in self.variables}
        print(vars_)
        lhs = vars_[self.LHS_var]
        print(lhs)
        subscripts = {i: sp.symbols(str(i), cls=sp.Idx,
                      range=(1, self.subscripts[str(i)]['count'])) for i in
                      self.subscripts.keys()}

        counts = {i: self.subscripts[str(i)]['count'] for i in
                  self.subscripts.keys()}

        time = sp.symbols('t', cls=sp.Idx, range=(self.base_year,
                                                  self.end_year))

        weights = self.weights(lhs, time, subscripts['i'], counts['i'])
        print('weights:\n', weights)

        effect = 0
        results = dict()

        for t in self.terms:
            print('t:', t)
            if '/' in t:
                parts = t.split('/')
                numerator = parts[0]
                print('numerator:', numerator)

                subs_ = [subscripts[i] for i in self.subscripts.keys()
                         if f'_{i}' in numerator]
                print(subs_)
                print('numerator:', numerator)

                numerator = vars_[numerator[0]]

                print('numerator:', numerator)
                if len(subs_) > 0:
                    numerator = numerator[time, subs_[0]]
                else:
                    numerator = numerator[time]

                print('numerator:', numerator)
                print('numerator shape:', numerator.shape)

                base_denominator = parts[1]
                subs_d = [subscripts[i] for i in self.subscripts.keys()
                          if f'_{i}' in base_denominator]
                print(subs_d)
                denominator = vars_[base_denominator[0]]
                if len(subs_d) > 0:
                    denominator = denominator[time, subs_d[0]]
                else:
                    denominator = denominator[time]

                print('denominator:', denominator)
                print('denominator shape:', denominator.shape)

                f = sp.exp(numerator / denominator)
            else:
                f = vars_[t]

            print('f', f)
            component = sp.exp(f * weights)
            # sp.pprint(component)

            results[t] = component

            if self.model == 'additive':
                effect += component
            elif self.model == 'multiplicative':
                component = sp.functions.elementary.exponential.exp(component)
                sp.pprint(component)

                if effect == 0:
                    effect += 1
                effect *= component
        # sp.pprint(sp.exp(effect))
        results['effect'] = sp.exp(effect)
        return results

    def expr(self):
        
        E = sp.IndexedBase('E')
        A = sp.IndexedBase('A')
        i, t = sp.symbols('i t', cls=sp.Idx)

        t1 = A[t]
        t2 = A[t, i] / A[t]
        t3 = E[t, i] / A[t, i]

        lhs = E
        m = 4

        weights = self.weights(lhs, t, i, m)

        activity = t1 * weights
        structure = t2 * weights
        intensity = t3 * weights
        terms = [activity, structure, intensity]

        if self.model == 'multiplicative':
            effect = activity * structure * intensity
        
        elif self.model == 'additive':
            effect = activity + structure + intensity

        results = {str(t): t for t in terms}
        results['effect'] = effect

        for k in results.keys():
            print('k:', k)
            sp.pprint(results[k])
        return results
    
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
        activity = pd.read_csv('C:/Users/irabidea/Desktop/yamls/industrial_activity.csv')
        energy = pd.read_csv('C:/Users/irabidea/Desktop/yamls/industrial_energy.csv')
        data = {'A': activity, 'E': energy}

        expression_dict = self.general_expr()

        df = pd.DataFrame(index=activity.index,
                          columns=list(expression_dict.keys()))
        for name, expr in expression_dict.items():

            symbs_ = expr.free_symbols
            print('symbols:\n', symbs_)
            symbs_2 = expr.atoms(sp.IndexedBase)
            print('symbsols_2:\n', symbs_2)
            symbs_3 = expr.atoms(sp.Idx)
            print('symbsols_3:\n', symbs_3)
            f = sp.lambdify(tuple(list(symbs_2) + list(symbs_3)), expr, "numpy")
            print('tuple(symbs_2, symbs_3):\n',
                  tuple(list(symbs_2) + list(symbs_3)))
            
            # df[name] = f(energy, activity, t=1995)

            # print('answer:\n', answer)
            for year in df.index:
                df.at[year, name] = f(**{str(a): data[str(a)].ilo
                                         for a in expr.atoms()
                                         if isinstance(a, sp.Symbol)
                                         })


    def main(self, fname):
        self.read_yaml(fname)
        # print("dir(IndexedVersion):\n", dir(IndexedVersion))
        self.expr()
        # results = self.general_expr()
        # print('results:\n', results)
        # self.eval_expression()

if __name__ == '__main__':
    directory = 'C:/Users/irabidea/Desktop/yamls/'
    symb = SymbolicLMDI(directory)
    symb.read_yaml(fname='test1')
    expression = symb.LMDI_expression()
    # subs_ = symb.eval_expression()
    # c = IndexedVersion(directory=directory).main(fname='test1')

