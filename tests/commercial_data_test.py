import pandas as pd 
import pytest

from EnergyIntensityIndicators.commercial import CommercialIndicators

class CommercialTest:
    """Test methods related to optimization models in the commercial sector
    """    
    pnnl_params = [3.92276415015621, 73.2238120168849] # [gamma, lifetime]

    def __init__(self, directory):
        self.directory = directory
        self.commercial = CommercialIndicators()

    def test_nems_logistic(self):
        """Use PNNL parameters to test whether the nems_logistic method replicates the historical floorspace data from PNNL
        """        
        pnnl_historical_floorspace = pd.read_excel(f'{self.directory}/Historical_Floorspace_021220.xlsb', sheet_name='NEMS_Logistic (current)', usecols='AB, AT', skiprows=335)
        historical_floorspace = self.commercial.nems_logistic( params=[3.92276415015621, 73.2238120168849])
        assert pnnl_historical_floorspace == historical_floorspace

    def test_nems_logistic_coeffs(self):
        """Test whether the run_model method replciates the paramaters found by PNNL
        """        
        solved_coeffs = self.commercial.solve_logistic()
        assert solved_coeffs == self.pnnl_params

        