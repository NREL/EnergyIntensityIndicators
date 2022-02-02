import pandas as pd
import numpy as np
import os

from EnergyIntensityIndicators.lmdi_gen import GeneralLMDI
from EnergyIntensityIndicators.utilities.dataframe_utilities \
    import DFUtilities as df_utils
from EnergyIntensityIndicators.utilities.testing_utilties \
    import TestingUtilities
from EnergyIntensityIndicators import DATADIR


class TestLMDIGen:
    directory = os.path.join(DATADIR, 'yamls/')
    gen = GeneralLMDI(directory)
    utils = TestingUtilities()

    @staticmethod
    def input_data():
        """Collect dictionary containing dataframes
        for each variable in the LMDI model
        """
        activity = \
            pd.read_csv(
                os.path.join(DATADIR, 'yamls/residential_activity.csv'),
                index_col=0)
        activity.index.name = 'Year'
        print('activity:\n', activity)
        energy = \
            pd.read_csv(
                os.path.join(DATADIR, 'yamls/residential_energy.csv'),
                index_col=0)
        energy.index.name = 'Year'
        print('energy:\n', energy)

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
        lhs_total = df_utils().create_total_column(LHS,
                                                   total_label=name)
        LHS_share = df_utils().calculate_shares(lhs_total,
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

    # def test_additive_weights(self):

    #     LHS, LHS_share = self.lhs_and_share()
    #     calc = self.gen.additive_weights(LHS, LHS_share)
    #     pnnl = self.pnnl_weights(model='additive')
    #     assert self.utils.pct_diff(calc, pnnl)

    # def test_compute_index(self):
    #     data = \
    #         pd.read_csv(f'{self.directory}/residential_component_test.csv')
    #     data = data.set_index('Year')
    #     component = data[['Component']]
    #     base_year_ = 1985
    #     calc = self.gen.compute_index(component, base_year_)
    #     pnnl = data[['Index']]
    #     assert self.utils.pct_diff(calc, pnnl)

    def test_compute_index1(self):
        """Data is from Total_Transportation 1983-1987"""
        eii = GeneralLMDI(directory=os.path.join(DATADIR, 'yamls/'))

        results = [[0.9705, 1.0386, 1.0037],
                   [0.9957, 1.0329, 1.0054],
                   [0.9982, 1.0145, 1.0052],
                   [1.0076, 1.0165, 1.0066],
                   [0.9814, 1.0412, 1.0016]]

        results = pd.DataFrame(results,
                               index=[1983, 1984, 1985, 1986, 1987],
                               columns=['Intensity Index', 'Activity Index',
                                        'Structure Index'])

        for col in results.columns:
            results[col] = eii.compute_index(results[col], 1985)
            results[col] = results[col].astype(float).round(4)

        comparison_output = [[1.0062, 0.9543, 0.9895],
                             [1.0018, 0.9857, 0.9948],
                             [1.0000, 1.0000, 1.0000],
                             [1.0076, 1.0165, 1.0066],
                             [0.9889, 1.0584, 1.0082]]

        comparison_output = pd.DataFrame(comparison_output,
                                         index=[1983, 1984, 1985,
                                                1986, 1987],
                                         columns=['Intensity Index',
                                                  'Activity Index',
                                                  'Structure Index'])
        print('results_:\n', results)
        print('comparison_output:\n', comparison_output)
        # assert results.equals(comparison_output)
        assert self.utils.pct_diff(comparison_output, results)

    def test_compute_index2(self):
        """Data is from Total_Transportation 1970-1975"""
        eii = GeneralLMDI(directory=os.path.join(DATADIR, 'yamls/'))

        results = [[np.nan, 1.1301, np.nan],
                   [0.9904, 1.0460, 1.0107],
                   [0.9998, 1.0621, 1.0068],
                   [1.0034, 1.0370, 1.0039],
                   [0.9855, 0.9809, 0.9996],
                   [0.9981, 1.0050, 1.0085],
                   [0.9858, 1.0593, 1.0070],
                   [0.9791, 1.0444, 1.0064],
                   [0.9886, 1.0691, 0.9922],
                   [0.9883, 1.0045, 0.9977],
                   [0.9682, 0.9980, 0.9859],
                   [0.9975, 1.0034, 0.9972],
                   [0.9759, 0.9972, 1.0117],
                   [0.9705, 1.0386, 1.0037],
                   [0.9957, 1.0329, 1.0054],
                   [0.9982, 1.0145, 1.0052],
                   [1.0076, 1.0165, 1.0066],
                   [0.9814, 1.0412, 1.0016]]

        results = pd.DataFrame(results,
                               index=[1970, 1971, 1972, 1973,
                                      1974, 1975, 1976, 1977,
                                      1978, 1979, 1980, 1981,
                                      1982, 1983, 1984, 1985,
                                      1986, 1987],
                               columns=['Intensity Index',
                                        'Activity Index',
                                        'Structure Index'])

        for col in results.columns:
            results[col] = eii.compute_index(results[col], 1985)
            results[col] = results[col].astype(float).round(4)

        comparison_output = [[1.1935, 0.6819, 0.9594],
                             [1.1821, 0.7133, 0.9696],
                             [1.1818, 0.7576, 0.9762],
                             [1.1859, 0.7856, 0.9800],
                             [1.1687, 0.7707, 0.9796],
                             [1.1665, 0.7745, 0.9879],
                             [1.1499, 0.8204, 0.9948],
                             [1.1258, 0.8568, 1.0013],
                             [1.1130, 0.9160, 0.9934],
                             [1.1000, 0.9202, 0.9912],
                             [1.0650, 0.9183, 0.9772],
                             [1.0623, 0.9214, 0.9745],
                             [1.0368, 0.9188, 0.9859],
                             [1.0062, 0.9543, 0.9895],
                             [1.0018, 0.9857, 0.9948],
                             [1.0000, 1.0000, 1.0000],
                             [1.0076, 1.0165, 1.0066],
                             [0.9889, 1.0584, 1.0082]]

        comparison_output = \
            pd.DataFrame(comparison_output,
                         index=[1970, 1971, 1972, 1973,
                                1974, 1975, 1976, 1977,
                                1978, 1979, 1980, 1981,
                                1982, 1983, 1984, 1985,
                                1986, 1987],
                         columns=['Intensity Index',
                                  'Activity Index',
                                  'Structure Index'])
        print('results_:\n', results)
        print('comparison_output:\n', comparison_output)
        # assert results.equals(comparison_output)
        assert self.utils.pct_diff(comparison_output, results)
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
