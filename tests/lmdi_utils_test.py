
from EnergyIntensityIndicators.utilites import lmdi_utilities


class TestLMDIUtilities:

    def test_log_mean_divisia(self, sector='transportation'):
        """Test for the lmdi_utilities logarithmic_average method

        Args:
            sector (str, optional): [description]. Defaults to 'transportation'.
        """
        x = 0.5913
        y = 0.5650
        L = lmdi_utilities.logarithmic_average(x, y)
        pnnl_result = 0.578
        assert round(L, 3) == pnnl_result