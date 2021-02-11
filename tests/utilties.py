from tests.lmdi_test import TestLMDI
   
    
class TestingUtilities:

    def __init__(self, sector, acceptable_pct_difference):
        self.sector = sector
        self.acceptable_pct_difference = acceptable_pct_difference

    def pct_diff_bools_list(self, df_pairs_list):
        """Given pairs (tuples) of eii, pnnl dataframes, return a list of bools indicating
        whether the percent different between the dataframes are within the acceptable range
        """
        bools_list = []
        for eii, pnnl in df_pairs_list:
            pct_difference = TestLMDI().pct_diff(pnnl, eii, 
                                                 self.acceptable_pct_difference, 
                                                 self.sector)
            bools_list.append(pct_difference)

        return bools_list