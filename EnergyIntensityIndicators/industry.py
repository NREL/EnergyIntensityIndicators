import pandas as pd
from sklearn import linear_model

class IndustrialIndicators(LMDI):

    def __init__(self, energy_data, activity_data, categories_list):
        super().__init__(energy_data, activity_data, categories_list)
        self.sub_categories_list = categories_list['industry']


    def load_data(self, ):
        MER_Nov19_Table24 = 
        AER10_Table21d = 
        AER11_Table21d_MER0816 = 
        mer_dataT0204 =
        BEA

    def reconcile_physical_units(self, ):
        """Convert physical units to Btu. (Prior to 2005, the data on energy consumption fuels to produce electricity were supplied in physical units (e.g. mcf of natural gas, tons of coal, etc))
        Data Source: EIA's Annual Energy Review (AER)"""
        pass

    def industrial_total_lmdi_utiladj(self, ):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def total_industrial(self, ):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def manufacturing(self, ):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def nonmanufacturing(self, ):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def mining(self, ):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def conversion_factors(self, ):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass