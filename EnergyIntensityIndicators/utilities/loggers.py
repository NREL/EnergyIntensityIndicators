"""Logging utility from nrel-rex"""

import os

import rex
#from rex.utilities.loggers import *
from EnergyIntensityIndicators import LOGDIR

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

def init_logger(logger_name):
    log_file = os.path.join(LOGDIR, f'{logger_name}.log')
    return rex.utilities.loggers.init_logger(logger_name,
                                             log_file=log_file)