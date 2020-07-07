
# Separate files/scripts:
# EIA API data call
# Base LMDI class and methods?
# Separate sector classes/methods inhereted from Base LMDI?
# Separate economy-wide class/methods file that rolls up all sector files.
# File for running everything (from commandline?): based on input of sector
# decomposing, base year, and form of LMDI (multiplicative or additive)
# File for output of data: output of final indicator data; output of line chart;
# output of results formatted for OpenEI data visualization

class LMDI:
    """

    """
    def __init__(self, base_year, ...):

        self.base_year = base_year

        self.activity = activity_data

        self.energy = energy_data

    @staticmethod
    def nominal_energy_intensity():
        """
        Calculate nominal energy intensity
        """

        return


    def energy_intensity_index(self):


class transportation(LMDI):

    def __init__(self, base_year, ...):

        LMDI.__init__(self, base_year, ...)
