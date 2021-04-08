import sympy as sp
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

    @staticmethod
    def hadamard_division(numerator, denominator):
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

        print('numerator:\n', numerator)

        denominator = sp.HadamardPower(denominator, -1).doit()
        print('denominator:\n', denominator)
        result = sp.HadamardProduct(numerator, denominator).doit()
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
        width = matrix.shape[1]
        length = matrix.shape[0]
        row = sp.ones(1, width)
        shift_term = matrix.row_insert(0, row)
        shift_term = sp.MatPow(shift_term, -1)
        long_term = matrix.row_insert(length, row)

        return shift_term, long_term

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
        term = sp.HadamardProduct(long_term, shift_term)
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
        weighted_term = sp.HadamardProduct(weight, term).doit()
        ones_ = sp.ones(weighted_term.shape[1], 1)
        component = sp.MatMult(weighted_term, ones_)
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

            negative_y = sp.HadamardProduct(y, -1)
            log_x = x.applyfunc(sp.log)
            log_y = y.applyfunc(sp.log)
            negative_log_y = sp.HadamardProduct(log_y, -1)

            numerator = sp.MatAdd(x, negative_y)
            denominator = sp.MatAdd(log_x, negative_log_y)

            logarithmic_average = self.hadamard_division(numerator,
                                                         denominator)

        elif x == y:
            logarithmic_average = x

        return logarithmic_average

    def multiplicative_weights(self, log_mean_matrix,
                               log_mean_share, log_mean_share_total):
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
            numerator = sp.HadamardProduct(log_mean_share, log_mean_matrix)
            weights = self.hadamard_division(numerator, log_mean_total)
        
        return weights

    def additive_weights(self, log_mean_matrix,
                         log_mean_matrix_total):
        """Calculate log-mean divisia weights for the additive model
        in symbolic terms

        Args:
            log_mean_matrix ([type]): [description]
            log_mean_matrix_total ([type]): [description]
        """

        if self.lmdi_type == 'I':
            weights = log_mean_matrix
        elif self.lmdi_type == 'II':
            numerator = sp.HadamardProduct(log_mean_share, log_mean_matrix)
            weights = self.hadamard_division(numerator, log_mean_matrix_total)

    @staticmethod
    def copyCol(N,V):
        M = sp.Matrix(V)
        for i in range(N):
            M = M.col_insert(1, V)
        return M
            
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
        lhs_total = self.copyCol(num_columns, lhs_total)
        print('lhs_total tiled:', lhs_total)

        lhs_share = self.hadamard_division(lhs_matrix, lhs_total)

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
                                            log_mean_matrix_total)
        elif self.model == 'multiplicative':
            weights = self.multiplicative_weights(log_mean_matrix,
                                                  log_mean_share,
                                                  log_mean_share_total)
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
        num_years = self.end_year - self.base_year
        num_columns = self.subscripts['i']['count']

        variable_dict = {var:
                         self.create_symbolic_var(var,
                                                  num_years,
                                                  num_columns)
                         for var in self.variables}

        lhs_matrix = variable_dict[self.LHS_var]
        weights = self.calc_weights(lhs_matrix, num_columns)

        symbolic_terms = []

        for term in self.terms:
            if '/' in term:
                parts = term.split('/')
                numerator = parts[0]
                denominator = parts[1]
                matrix_term = self.create_symbolic_term(numerator, denominator)

            else:
                matrix_term = variable_dict[term]
            
            weighted_term = self.weighted_term(weights, matrix_term)
            symbolic_terms.append(weighted_term)

        if self.model == 'additive':
            expression = sp.MatAdd(symbolic_terms)
        elif self.model == 'multiplicative':
            expression = sp.MatMult(symbolic_terms)
        
        return expression
    
    def eval_expression(self):
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
        expression = self.LMDI_expression()
        input_dict = {sp.symbols(v):
                      self.input_data[v] for v in self.variables}
        final_result = expression.subs(input_dict)
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
            L = (x - y) / (sp.log(x) - sp.log(y))

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
        lhs_total_shift = sp.Sum(lhs[t-1, i], (i, 1, m))

        lhs_share = lhs[t, i] / lhs_total
        lhs_share_shift = lhs[t-1, i] / lhs_total_shift
        
        log_average_total = self.logarithmic_average(lhs_total,
                                                     lhs_total_shift)
        sp.pprint(log_average_total)
        log_average = self.logarithmic_average(lhs[t, i], lhs[t-1, i])
        sp.pprint(log_average)

        log_average_shares = self.logarithmic_average(lhs_share,
                                                      lhs_share_shift)
        sp.pprint(log_average_shares)

        total_log_average_shares = sp.Sum(log_average_shares, (i, 1, m))

        if self.model == 'multiplicative':
            if self.lmdi_type == 'I':
                weights = log_average_total / log_average_total
            elif self.lmdi_type == 'II':
                weights = log_average_shares / total_log_average_shares

        elif self.model == 'additive':
            if self.lmdi_type == 'I':
                weights = log_average
            elif self.lmdi_type == 'II':
                weights = (log_average_shares * log_average_total) / \
                          total_log_average_shares

        return weights

    def general_expr(self):

        # for i in self.subscripts.keys():
        #     i = sp.symbols(str(i), cls=sp.Idx)
        #     print('i', i)

        vars_ = {str(v): sp.IndexedBase(str(v)) for v in self.variables}
        print(vars_)
        lhs = vars_[self.LHS_var]
        print(lhs)
        for i in self.subscripts.keys():
            i = sp.symbols(str(i), cls=sp.Idx)
            m = self.subscripts[str(i)]['count']
            print(i, m)
        
        t = sp.symbols('t', cls=sp.Idx)

        weights = self.weights(lhs, t, i, m)
        print('weights:\n', weights)

        for t in self.terms:
            print('t:', t)
            if '/' in t:
                parts = t.split('/')
                numerator = parts[0]
        #             for i in self.subscripts.keys():
                        
                print('numerator:', numerator)
                denominator = parts[1]
                print('denominator:', denominator)
                f = numerator / denominator
                # f = vars_[numerator] / vars_[denominator]
            else:
                # f = vars_[t]
                f = t

            print('f', f)

        #     # effect = t * weights
        #     result = effect.subs()

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

    def main(self, fname):
        self.read_yaml(fname)
        # print("dir(IndexedVersion):\n", dir(IndexedVersion))
        self.expr()
        self.general_expr()

if __name__ == '__main__':
    directory = 'C:/Users/irabidea/Desktop/yamls/'
    # symb = SymbolicLMDI(directory)
    # symb.read_yaml(fname='test1')
    # expression = symb.LMDI_expression()
    c = IndexedVersion(directory=directory).main(fname='test1')
