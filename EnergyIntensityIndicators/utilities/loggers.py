"""Logging utility from nrel-rex"""

from asyncio.log import logger
import os
import logging

from EnergyIntensityIndicators import LOGDIR

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

FORMAT = '%(levelname)s - %(asctime)s [%(filename)s:%(lineno)d] : %(message)s'
LOG_LEVEL = {'INFO': logging.INFO,
             'DEBUG': logging.DEBUG,
             'WARNING': logging.WARNING,
             'ERROR': logging.ERROR,
             'CRITICAL': logging.CRITICAL}

log_file = os.path.join(LOGDIR, 'eii_main.log')
logging.basicConfig(format=FORMAT, level=logging.DEBUG, filename=log_file)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter(FORMAT))
console.setLevel(logging.DEBUG)
logging.getLogger('').addHandler(console)


def get_logger():
    return logging.getLogger(__name__)
