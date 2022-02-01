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

logging.basicConfig(format=FORMAT, level=logging.INFO)


log_file = os.path.join(LOGDIR, 'eii_main.log')
file_log = logging.FileHandler(log_file)
file_log.setFormatter(logging.Formatter(FORMAT))
file_log.setLevel(logging.DEBUG)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter(FORMAT))
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logging.getLogger('').addHandler(file_log)


def get_logger():
    return logging.getLogger(__name__)
