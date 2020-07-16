import pandas as pd
from sklearn import linear_model
from .weather_factors import weather_factors

class ResidentialIndicators:
    def __init__(self, ):
		self. =

    def load_data(self, ):
        AER11_table2_1b_update = 
        AnnualData_MER22_2015 = 
        AnnualData_MER22_2017 = 
        AnnualData_MER_22_Dec2019 = 
        RECS_intensity_data = 
        National_Calibration = 
        Weather_Factors = 
        CDD_by_Division18 = 
        HDD_by_Division18 = 

    def regional_time_series_floor_space(self, ):
        pass

    def estimate_fuel_electricity_consumption_regional(self, ):
        """Data Source: EIA's State Energy Data System (SEDS)"""
        energy_consumtpion_data_regional = 
        approximate_intesity_time_series = 
        weather_adjustment_factors_regional = 
        energy_consumption_regional = 
        return None 

    def estimate_floorspace_occupied_housing_units_regional(self, ):
        """Estimate regional housing and regional floorspace by housing type (single family, multifamily, manufactured homes)"""
        estimated_survival_curve =  # Estimate from vintage data over the 1999 through 2009 AHS surveys
        new_housing =  # From Characteristics of New Housing reports from the Census Bureau
        stock_adjustment_model = 
        estimated_occupied_housing_units =  # from stock adjustment level
        return estimated_occupied_housing_units

    def estimate_floorspace_housing_unit_size_national(self, housing_type='single_family'):
        """Single family and multi-family units use AHS data, combined with adjusted Characteristics of New Housing Data. Manufactured homes use RECS data"""
        if housing_type == 'manufactured_homes':
            size_estimates = 
        else: 
            average_size_post_1985 = 
            stock_units_pre_1985 = 
            stock_units_post_1985 =  # including 1985

    def estimate_floorspace_regional_shares_national_level_housing_units(self, ):
        """The regional shares for the non-AHS years are computed via a simple average of the preceding (odd) year and subsequent (odd) year.
        Data Source: AHS"""
        pass

    def estimate_final_floorspace_by_housing_type(self, ):
        """Data Source: AHS"""

    def residential_total_lmdi_utiladj(self, ):
        """"""
        pass

    def national(self, ):
        """"""
        pass

    def northeast(self, ):
        """"""
        pass

    def midwest(self, ):
        """"""
        pass

    def south(self, ):
        """"""
        pass

    def west(self, ):
        """"""
        pass

    def report_tables(self, ):
        """"""
        pass

    def report_graphs(self, ):
        """"""
        pass


categories_residential_national = ['Northeast', 'Midwest', 'South', 'West']
categories_residential_northeast = ['Single-Family', 'Multi-Family', 'Manufactured Homes']
categories_residential_midwest = ['Single-Family', 'Multi-Family', 'Manufactured Homes']
categories_residential_south = ['Single-Family', 'Multi-Family', 'Manufactured Homes']
categories_residential_west = ['Single-Family', 'Multi-Family', 'Manufactured Homes']
