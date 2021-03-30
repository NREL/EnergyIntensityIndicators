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
        """
        assert lhs == sp.simplify(expression)
    
    @staticmethod
    def create_symbolic_var(variable, num_years, num_columns):
        """[summary]
        """
        variable = sp.MatrixSymbol(str(variable), num_years, num_columns)
        return variable
    
    @staticmethod
    def create_symbolic_term(numerator, denominator):
        """[summary]
        """
        denominator = sp.MatPow(denominator, -1)
        base_term = sp.HadamardProduct(numerator, denominator)
        width = base_term.shape[1]
        length = base_term.shape[0]
        row = sp.ones(1, width)
        shift_term = base_term.row_insert(0, row)
        shift_term = sp.MatPow(shift_term, -1)
        long_term = base_term.row_insert(length, row)
        # find change (divide every row by previous row)
        term = sp.HadamardProduct(long_term, shift_term)
        term = term.applyfunc(sp.log)
        return term
    
    @staticmethod
    def weighted_term(weight, term):
        """Calculate components from dot product of weights and term

        Args:
            weight (MatrixSymbol): normalized log-mean divisia weights
            term (MatrixSymbol): [description]
        """
        weighted_term = sp.HadamardProduct(weight, term).doit()
        ones_ = sp.ones(weighted_term.shape[1], 1)
        component = sp.MatMult(weighted_term, ones_)
        return component

    @staticmethod
    logarithmic_average(x, y):
        """[summary]

        Returns:
            [type]: [description]
        """
        if x != y:
            numerator = 
            denominator = 
            
        elif x == y:
            return x
        pass

    def multiplicative_weights(self):
        """[summary]
        """
        if lmdi_type == 'I':
            
        elif lmdi_type == 'II':


    def additive_weights(self):
        """[summary]
        """
        if lmdi_type == 'I':
            
        elif lmdi_type == 'II':


    def calc_weights(self, model):
        """[summary]
        """
        if model == 'additive':
            weights = self.additive_weights()
        elif model == 'multiplicative':
            weights = self.multiplicative_weights()
        return weights

    def LMDI_expression(self):

        num_years = self.end_year - self.base_year
        num_columns = self.subscripts['i']['count']
        weights = self.calc_weights(self.model)

        variable_dict = {var:
                         self.create_symbolic_var(var,
                                                  num_years,
                                                  num_columns)
                         for var in self.variables}

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
        expression = self.LMDI_expression()
        input_dict = {sp.symbols(v):
                      self.input_data[v] for v in self.variables}
        final_result = expression.subs(input_dict)
        return final_result


if __name__ == '__main__':
    expression = SymbolicLMDI().LMDI_expression()
