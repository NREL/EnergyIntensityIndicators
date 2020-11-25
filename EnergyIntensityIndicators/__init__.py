"""The Energy Intensity Indicators Model
"""

from __future__ import print_function, division, absolute_import
import os 

from EnergyIntensityIndicators.Residential import residential_floorspace
from EnergyIntensityIndicators.Industry import (manufacturing, 
                                                nonmanufacuturing, asm_price_fit)
from EnergyIntensityIndicators import (industry, residential, commercial, transportation, 
                                       electricity, additive_lmdi, multiplicative_lmdi, 
                                       LMDI)


__author__  = 'Isabelle Rabideau'
__email__ = 'isabelle.rabideau@nrel.gov'

EIIDIR = os.path.dirname(os.path.realpath(__file__))
TESTDATADIR = os.path.join(os.path.dirname(EIIDIR), 'tests', 'data')