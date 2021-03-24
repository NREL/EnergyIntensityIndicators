import sympy as sp
import pandas as pd
import yaml


class SymbolicLMDI:

    def __init__(self):
        pass

    @staticmethod
    def create_yaml():
        lhs_variable = 
        terms = []
        units = []
        input_data = []
        energy_types = []
        model = ''
        base_year = 
        end_year = 
        
    @staticmethod
    def read_yaml():
        """Read yaml input data
        """
        pass

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
        term = sp.HadamardProduct(numerator, denominator)
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
    def multiplicative_weights():
        """[summary]
        """
        pass

    @staticmethod
    def additive_weights():
        """[summary]
        """
        pass

    def calc_weights(self, model):
        """[summary]
        """
        if model == 'additive':
            weights = self.additive_weights()
        elif model == 'multiplicative':
            weights = self.multiplicative_weights()
        return weights

    def LMDI_expression(self):
        max_year = 
        min_year = 
        num_years = max_year - min_year
        num_columns = 
        model = 
        weights = self.calc_weights(model)

        variable_dict = {var:
                         self.create_symbolic_var(var,
                                                  num_years,
                                                  num_columns)
                         for var in variables}

        symbolic_terms = []

        for term in terms:
            numerator = 
            denominator = 
            term = self.create_symbolic_term(numerator, denominator)
            weighted_term = self.weighted_term(weight, term)
            symbolic_terms.append(weighted_term)

        if model == 'additive':
            expression = sp.MatAdd(symbolic_terms)
        elif model == 'multiplicative':
            expression = sp.MatMult(symbolic_terms)
        
        return expression, symbolic_terms
    
    def eval_expression(self):
        expression = self.LMDI_expression()
        input_dict = 
        final_result = expression.subs(input_dict)


if __name__ == '__main__':
