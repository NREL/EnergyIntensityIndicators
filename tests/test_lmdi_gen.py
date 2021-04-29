import sympy as sp
import numpy as np
import yaml
import pandas as pd

from EnergyIntensityIndicators.lmdi_gen import GeneralLMDI
from EnergyIntensityIndicators.utilites \
    import dataframe_utilities as df_utils
# from tests.utilites import TestingUtilities
import utilites
# import TestingUtilities


class TestLMDIGen:
    directory = 'C:/Users/irabidea/Desktop/yamls/'
    gen = GeneralLMDI(directory)
    utils = utilites.TestingUtilities(acceptable_pct_difference=0.05)

    @staticmethod
    def input_data():
        """Collect dictionary containing dataframes
        for each variable in the LMDI model
        """
        activity = \
            pd.read_csv(
                'C:/Users/irabidea/Desktop/yamls/residential_activity.csv')
        activity = activity.set_index('Year')
        energy = \
            pd.read_csv(
                'C:/Users/irabidea/Desktop/yamls/residential_energy.csv')
        energy = energy.set_index('Year')

        data = {'A_i': activity,
                'E_i': energy,
                'total_label': 'Residential'}
        return data

    def test_all_equal1(self):
        iterator = [1, 3, 5]
        calc = self.gen.all_equal(iterator)
        pnnl = False
        assert calc == pnnl

    def test_all_equal2(self):
        iterator = [1, 1, 1]
        calc = self.gen.all_equal(iterator)
        pnnl = True
        assert calc == pnnl

    def lhs_and_share(self):
        input_data = self.input_data()
        LHS = input_data['E_i']
        name = input_data['total_label']
        lhs_total = df_utils.create_total_column(LHS,
                                                 total_label=name)
        LHS_share = df_utils.calculate_shares(lhs_total,
                                              total_label=name)
        return LHS, LHS_share

    def pnnl_weights(self, model):
        if model == 'multiplicative':
            weights = \
                pd.read_csv(
                    f'{self.directory}/residential_source_weights_test.csv')
            weights = weights.set_index('Year')

        # elif model == 'additive':
        #     weights =
        return weights

    def test_multiplicative_weights(self):

        LHS, LHS_share = self.lhs_and_share()
        calc = self.gen.multiplicative_weights(LHS, LHS_share)
        pnnl = self.pnnl_weights(model='multiplicative')
        assert self.utils.pct_diff(calc, pnnl)

    def test_additive_weights(self, LHS, LHS_share):

        LHS, LHS_share = self.lhs_and_share()
        calc = self.gen.additive_weights(LHS, LHS_share)
        pnnl = self.pnnl_weights(model='additive')
        assert self.utils.pct_diff(calc, pnnl)

    def test_compute_index(self):
        data = \
            pd.read_csv(f'{self.directory}/residential_component_test.csv')
        data = data.set_index('Year')
        component = data[['Component']]
        base_year_ = 1985
        calc = self.gen.compute_index(component, base_year_)
        pnnl = data[['Index']]
        assert self.utils.pct_diff(calc, pnnl)

    # def test_decomposition_multiplicative(self, terms_df):
    #     calc = self.gen.decomposition_multiplicative(terms_df)
    #     pnnl =
    #     assert self.utils.pct_diff(calc, pnnl)

    # def test_decomposition_additive(self, terms_df):
    #     calc = self.gen.decomposition_additive(terms_df)
    #     pnnl =
    #     assert self.utils.pct_diff(calc, pnnl)

    # def test_general_expr():
    #     input_data = self.input_data()
    #     calc = gen.general_expr(input_data)
    #     pnnl =
    #     assert self.utils.pct_diff(calc, pnnl)


if __name__ == '__main__':
    print('running')
    pass
