"""Goal: Decompose CO2 Emissions by fuel and sector (historical and projected) 

Required inputs: CO2 Emissions data (EIA)
    Notes:
        -EIA API contains endpoints for CO2 emissions by state 
         (including total US), fuel and sector (including all)
            ~ Fuels (depend on sector?): 
                - All fuels
                - Coal
                - Natural Gas
                - Petroleum
        - Link: https://www.eia.gov/opendata/qb.php?category=2251604
        - AEO (projections) includes Energy-Related Carbon Dioxide Emissions by 
          Sector (including total), Source, and Region (including total US)
            ~ Sources (depend on sector): 
                - Coal
                - Natural Gas
                - Petroleum
                - Electricity
                - Total
Framework: 
    - Multiply fuel use by emissions factor? (using constant factors over time)
    - Replacing energy use with emissions in decomp formula
    - plot difference between EIA emissions and calculated emissions

Steps: 
    - Import collect_data methods from each sector specific class
    - collect emissions factor data (from EIA?)
    - multiply energy use data by emissions factor data, replace energy use data 
      with the product (emissions by energy type)
    - pass resulting data dictionaries to CalculateLMDI class

    Notes:
        ~ This flow follows similar process to EconomyWide, potentially inherit that class

Desired outputs: results of decomposition in csv and visualizations (as in the rest of EII)
        
"""
import pandas as pd
import os

# from EnergyIntensityIndicators.industry import IndustrialIndicators
# from EnergyIntensityIndicators.residential import ResidentialIndicators
# from EnergyIntensityIndicators.commercial import CommercialIndicators
# from EnergyIntensityIndicators.electricity import ElectricityIndicators
# from EnergyIntensityIndicators.transportation import TransportationIndicators
# from EnergyIntensityIndicators.LMDI import CalculateLMDI
from EnergyIntensityIndicators.economy_wide import EconomyWide
from EnergyIntensityIndicators.pull_eia_api import GetEIAData


class CO2EmissionsDecomposition(EconomyWide):
    """Class to decompose CO2 emissions by sector of the U.S. economy. 
    """
    def __init__(self, directory, output_directory, level_of_aggregation=None, 
                 lmdi_model='multiplicative', base_year=1985, end_year=2018): 
        
        super().__init__(directory=directory, output_directory=output_directory, 
                         level_of_aggregation=level_of_aggregation, 
                         lmdi_model=lmdi_model, base_year=base_year, end_year=end_year)

    def collect_emissions_factors(self, sector=None, energy_type=None, region=None):
        """Collect emissions factors from the EIA API (through GetEIAData). 
        If region is None, collect data for the U.S., if energy_type is None use total, 
        if sector is None use total

        Parameters:
            sector (str):
            energy_type (str): 
            region (str): 

        Returns: 
            emissions_factor (df, series or float):
        """
        pass

    @staticmethod
    def calculate_emissions(energy_data, emissions_factor):
        """Calculate emissions from the product of energy_data and 
        emissions_factor

        Parameters:
            energy_data (df): 
            emission_factor (df, series or float): 
        
        Returns: 
            emissions_data (df): 
        """
        emissions_data = energy_data.multiply(emissions_factor)
        return emissions_data

    def collect_emissions_data(self): 
        """[summary]

        Parameters:

        Returns: 
            emissions_data_dict (dict): Nested dictionary of all_data from EconomyWide 
                                        with energy data replaced with emissions data
                                        (with original dictionary keys remaining intact)
        TODO: 
            - Break sector_level_data into lowest levels of dict in order
              to manipulate energy data
            - replace energy_data with emissions_data in nested dictionary
        """
        all_data = self.collect_data() # This is currently dictionary of 
                                       # all data collected in EconomyWide.collect_data()
        
        for sector in all_data.keys():
            sector_level_data = all_data[sector]

            # sector_level_data is a complex nested dictionary, 
            # infrastructure to handle this is contained in CalculateLMDI
            energy_data = 
            energy_type = 
            region = 
            emission_factor = self.collect_emissions_factors(sector=sector, 
                                                             energy_type=energy_type, 
                                                             region=region)

            emissions_data = self.calculate_emission(energy_data, emission_factor)

            # replace energy_data in nested dictionary with emissions_data
        
        emissions_data_dict = 
        return emissions_data_dict

    
    def main(self, breakout, calculate_lmdi):
        """Calculate decomposition of CO2 emissions for the U.S. economy
        
        TODO: allow for different sectors to have different types of energy 
              and commercial and residential to have weather adjustment 
              (TODO carried over from EconomyWide)

        """        
        data_dict = self.collect_emissions_data()
        results_dict, formatted_results = self.get_nested_lmdi(level_of_aggregation=self.level_of_aggregation, 
                                                               breakout=breakout, calculate_lmdi=calculate_lmdi, 
                                                               raw_data=data_dict, lmdi_type='LMDI-I')
        return results_dict 

if __name__ == '__main__':
    indicators = EconomyWide(directory='./EnergyIntensityIndicators/Data', 
                             output_directory='./Results', level_of_aggregation=, 
                             end_year=2018, lmdi_model=['multiplicative', 'additive'])
    indicators.main(breakout=True, calculate_lmdi=True)


class EmissionsComparison(CO2EmissionsDecomposition):
    """Class to visualize the difference between emissions
    values calculated from energy data and emissions factors
    vs emissions values given by the EIA API
    """

    def __init__(self):
        pass

    def get_eia_emissions(self, sector=None, energy_type=None, region=None):
        """Collect emissions data from the EIA API (through GetEIAData). 
        If region is None, collect data for the U.S., if energy_type is None use total, 
        if sector is None use total

        Parameters:
            sector (str):
            energy_type (str): 
            region (str): 

        Returns: 
            emissions_factor (df, series or float):
        """
        pass

    def compare_values(self):
        sector = 
        energy_type = 
        region = 
        eia_data = self.get_eia_emissions(sector, energy_type, region)
        calc_data = # unclear how to extract these values from the nested dictionary
        pct_diff  = # extract perecent difference calculation currently in LMDITest 
                    # (should be moved to utilities)

