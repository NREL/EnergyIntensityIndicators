"""The Energy Intensity Indicators Model
"""

from __future__ import print_function, division, absolute_import
import os
import json

EIIDIR = os.path.dirname(os.path.realpath(__file__))
TESTDATADIR = os.path.join(os.path.dirname(EIIDIR), 'tests', 'data')
LOGDIR = os.path.join(os.path.dirname(EIIDIR), 'logs')
DATADIR = os.path.join(os.path.dirname(EIIDIR), 'Data')
RESULTSDIR = os.path.join(os.path.dirname(EIIDIR), 'RESULTS')

with open(os.path.join(EIIDIR, 'keys.json')) as f:
    keys = json.loads(f.read())

os.environ['EIA_API_Key'] = keys['EIA_API_Key']
os.environ['BEA_API_Key'] = keys['BEA_API_Key']

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

__author__  = 'Isabelle Rabideau'
__email__ = 'isabelle.rabideau@nrel.gov'

from EnergyIntensityIndicators.Residential import residential_floorspace
from EnergyIntensityIndicators.Industry import (manufacturing,
                                                nonmanufacuturing, asm_price_fit)
from EnergyIntensityIndicators import (industry, residential, commercial, transportation,
                                       electricity, additive_lmdi, multiplicative_lmdi,
                                       LMDI)
