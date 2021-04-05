import sympy as sp
# import pandas as pd
# import yaml


class SymbolicLMDI:

    def __init__(self):
        self.lhs_variable = 'Ei'
        self.terms = ['A', 'Ai/A', 'Ei/Ai']
        self.subscripts = {'i': {'names':
                                 ['Northeast', 'Midwest', 'South', 'West'],
                           'count': 4}}
        # self.units = []
        # self.input_data = []
        self.energy_types = ['source', 'deliv', 'elec', 'fuels']
        self.model = 'multiplicative'
        self.base_year = 1990
        self.end_year = 2018
        self.variables = ['Ei', 'Ai']
        self.lmdi_type = 'II'
        

    # @staticmethod
    # def create_yaml():
    #     lhs_variable = 'E'
    #     terms = ['A', 'Ai/A', 'Ei/Ai']
    #     subscripts = {'i': {'names':
    #                         ['Northeast', 'Midwest', 'South', 'West'],
    #                         'count': 4}}
    #     # units = []
    #     # input_data = []
    #     energy_types = ['source', 'deliv', 'elec', 'fuels']
    #     model = 'multiplicative'
    #     base_year = 1990
    #     end_year = 2018
    #     variables = ['E', 'A']

    # @staticmethod
    # def read_yaml():
    #     """Read yaml input data
    #     """
    #     pass

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

        denominator = sp.HadamardPower(denominator, -1)
        print('denominator:\n', denominator)
        result = sp.HadamardProduct(numerator, denominator)
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

    def calc_weights(self, lhs_matrix):
        """Calculate log-mean divisia weights

        Args:
            lhs_matrix (Symbolic Matrix): Matrix representing the LHS variable
                                          (the variable to decompose) symbolically

        Returns:
            weights (Symbolic Matrix): The log-mean divisia weights
        """
        lhs_total = lhs_matrix * sp.ones(lhs_matrix.shape[1], 1)
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

        lhs_matrix = variable_dict[self.lhs_variable]
        weights = self.calc_weights(lhs_matrix)

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


if __name__ == '__main__':
    expression = SymbolicLMDI().LMDI_expression()
