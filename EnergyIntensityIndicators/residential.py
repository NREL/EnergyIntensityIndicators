"""Overview and Assumptions:
A. Data on the number and average size of occupied housing units from the biennial American
Housing Survey were employed to generate many of the activity metrics for this sector.
B. Three types of residential housing units are distinguished: single-family, multi-family, and
manufactured homes.
C. Regional data from EIA’s State Energy Data System (SEDS) are employed to develop regional
intensity indicators.
D. Regression models at the regional level are used to adjust for year-to-year changes in weather.
E. Two separate data construction elements are required to generate the regional and national
estimates of energy intensity indicators for this sector.
    1. Regional time series of floor space for residential housing units in the U.S (census level).
    2. Weather adjustment for the four census regions.
"""
import pandas as pd
from sklearn import linear_model

from EnergyIntensityIndicators.pull_eia_api import GetEIAData
from EnergyIntensityIndicators.LMDI import CalculateLMDI
from EnergyIntensityIndicators.Residential.residential_floorspace import ResidentialFloorspace
from EnergyIntensityIndicators.weather_factors import WeatherFactors


class ResidentialIndicators(CalculateLMDI): 
 
    def __init__(self, directory, output_directory, level_of_aggregation=None, lmdi_model='multiplicative', base_year=1985, end_year=2018):
        self.eia_res = GetEIAData('residential')
        self.sub_categories_list = {'National': {'Northeast': {'Single-Family': None, 'Multi-Family': None, 'Manufactured-Homes': None}, 
                                                 'Midwest': {'Single-Family': None, 'Multi-Family': None, 'Manufactured-Homes': None},
                                                 'South': {'Single-Family': None, 'Multi-Family': None, 'Manufactured-Homes': None},
                                                 'West': {'Single-Family': None, 'Multi-Family': None, 'Manufactured-Homes': None}}}
        self.national_calibration = self.eia_res.national_calibration()
        self.seds_census_region = self.eia_res.get_seds() # energy_consumtpion_data_regional
        self.ahs_Data = ResidentialFloorspace.update_ahs_data()
        self.regions = ['Northeast', 'South', 'West', 'Midwest', 'National']
        self.base_year = base_year
        self.directory = directory
        self.end_year = end_year
        self.energy_types = ['elec', 'fuels', 'deliv', 'source']
        super().__init__(sector='residential', level_of_aggregation=level_of_aggregation, lmdi_models=lmdi_model, categories_dict=self.sub_categories_list, 
                         energy_types=self.energy_types, directory=directory, output_directory=output_directory, primary_activity='occupied_housing_units',
                         base_year=base_year, end_year=end_year, weather_activity='floorspace_square_feet')


        # self.AER11_table2_1b_update = GetEIAData.eia_api(id_='711250') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711250'
        # self.AnnualData_MER_22_Dec2019 = GetEIAData.eia_api(id_='711250') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711250' ?
        # self.RECS_intensity_data =   # '711250' for Residential Sector Energy Consumption
    
    def get_seds(self):
        """Collect SEDS data"""
        census_regions = {4: 'West', 3: 'South', 2: 'Midwest', 1: 'Northeast'}
        total_fuels = self.seds_census_region[0].rename(columns=census_regions)
        elec = self.seds_census_region[1].rename(columns=census_regions)
        return total_fuels, elec

    def fuel_electricity_consumption(self, total_fuels, elec, region):
        """Combine Energy datasets into one Energy Consumption dataframe in Trillion Btu
        Data Source: EIA's State Energy Data System (SEDS)"""

        fuels_dataframe = total_fuels[[region]]
        elec_dataframe = elec[[region]]

        energy_data = {'elec': elec_dataframe, 'fuels': fuels_dataframe}
        return energy_data
    
    def get_floorspace(self):
        """Collect floorspace data for the Residential sector"""
        residential_data = ResidentialFloorspace(end_year=self.end_year)
        floorspace_square_feet, occupied_housing_units, household_size_square_feet_per_hu = residential_data.final_floorspace_estimates()

        final_floorspace_results = {'occupied_housing_units': occupied_housing_units, 'floorspace_square_feet': floorspace_square_feet, 
                                    'household_size_square_feet_per_hu': household_size_square_feet_per_hu}
        return final_floorspace_results


    def activity(self, floorspace):
        """Combine Energy datasets into one Energy Consumption Occupied Housing Units
        """ 
        all_activity = dict()
        for region in self.sub_categories_list['National'].keys():
            region_activity = dict()
            for variable, data in floorspace.items():
                df = data[region]
                if variable == 'household_size_square_feet_per_hu':
                    df = df.rename(columns={'avg_size_sqft_mf': 'Multi-Family', 'avg_size_sqft_mh': 'Manufactured-Homes', 'avg_size_sqft_sf': 'Single-Family'})
                else:
                    df = df.rename(columns={'occupied_units_mf': 'Multi-Family', 'occupied_units_mh': 'Manufactured-Homes', 'occupied_units_sf': 'Single-Family'})
                
                print(variable, df.columns)
                region_activity[variable] = df
            all_activity[region] = region_activity

        return all_activity
    
    def collect_weather(self, energy_dict, nominal_energy_intensity):
        """Collect weather data for the Residential Sector"""
        weather = WeatherFactors(sector='residential', directory=self.directory, nominal_energy_intensity=nominal_energy_intensity)
        weather_factors = weather.get_weather(energy_dict, weather_adjust=False) # What should this return?? (e.g. weather factors or weather adjusted data, both?)
        return weather_factors

    def collect_data(self):
        """Gather all input data for you in decomposition of 
        energy use for the Residential sector
        """
        total_fuels, elec = self.get_seds()
        floorspace = self.get_floorspace()
        activity = self.activity(floorspace)
        all_data = dict()
        nominal_energy_intensity_by_r = dict()
        for r in self.sub_categories_list['National'].keys(): 
            region_activity = activity[r]
    
            energy_data = self.fuel_electricity_consumption(total_fuels, elec, region=r)

            nominal_energy_intensity_by_e = dict()

            for e, e_df in energy_data.items():
                e_df = e_df.rename_axis(columns=None)
                floorspace = region_activity['floorspace_square_feet']
                total_floorspace = floorspace.sum(axis=1)
                nominal_energy_intensity = self.nominal_energy_intensity(energy_input_data=e_df, activity_input_data=total_floorspace) 

                nominal_energy_intensity_by_e[e] = nominal_energy_intensity 

            region_data = {'energy': energy_data, 'activity': region_activity}

            nominal_energy_intensity_by_r[r] = nominal_energy_intensity_by_e
            all_data[r] = region_data
                
        weather_factors = self.collect_weather(energy_dict=energy_data, nominal_energy_intensity=nominal_energy_intensity_by_r) # need to integrate this into the data passed to LMDI
        
        national_weather_dict = dict()
        for region, r_dict_ in all_data.items():
            weather_factors_by_e_type = dict()

            for e_ in r_dict_['energy'].keys():
                national_weather_dict[e_] = weather_factors[e_][[f'{e_}_weather_factor']]

                e_r_weather = weather_factors[e_][[f'{region.lower()}_weather_factor']]
                weather_factors_by_e_type[e_] = e_r_weather

            r_dict_['weather_factors'] = weather_factors_by_e_type
            all_data[region] = r_dict_
            
        # all_data['National'] = national_weather_dict
        print('all_data:\n', all_data)
        print('all_data:\n', all_data.keys())

        return all_data

    def main(self, breakout, calculate_lmdi):
        """Calculate decomposition for the Residential sector
        """
        unit_conversion_factor = 1
        data_dict = self.collect_data()

        results_dict, formatted_results = self.get_nested_lmdi(level_of_aggregation=self.level_of_aggregation, 
                                                               breakout=breakout, calculate_lmdi=calculate_lmdi, 
                                                               raw_data=data_dict, lmdi_type='LMDI-I')

        return results_dict


if __name__ == '__main__':
    indicators = ResidentialIndicators(directory='C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020', 
                                       output_directory='../Results', level_of_aggregation='National', 
                                       lmdi_model=['multiplicative', 'additive'], end_year=2017)
    indicators.main(breakout=True, calculate_lmdi=False)  

