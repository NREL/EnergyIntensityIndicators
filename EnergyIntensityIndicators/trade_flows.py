"""Goal: Incorporate embodied energy from imported goods in 
decomposition of the Industrial Sector. 

Required inputs:


Framework: 

Steps:


Desired outputs: results of decomposition in csv and visualizations
(as in the rest of EII) where imports are a sub-sector level?
"""

import urllib
import zipfile
from functools import reduce
import numpy as np
import pandas as pd
from sklearn import linear_model

from EnergyIntensityIndicators.industry import IndustrialIndicators


class TradeFlows(IndustrialIndicators):

    def __init__(self, directory, output_directory,
                 level_of_aggregation=None, 
                 lmdi_model='multiplicative', 
                 base_year=1985, end_year=2018, 
                 naics_digits=3):

        super().__init__(directory=directory, output_directory=output_directory,
                         level_of_aggregation=level_of_aggregation,
                         lmdi_model=lmdi_model, base_year=base_year, end_year=end_year,
                         naics_digits=naics_digits)
