"""
Sector level Energy Intensity Indicators
"""
import pandas as pd

# General Inputs
# Base Year for Indexes 
primary = 1985  # for transportation, commercial, 
secondary = 1996  # for transportation, commercial,  
starting_year_for_charts = 1985  # for transportation, commercial, 
if transportation: 
	ending_year_for_charts = 2003
elif commercial:
	ending_year_for_charts  = 2000
	include_electricity_utility_efficiency = True
	if  include_electricity_utility_efficiency:
		include = 1
	else: 
		include  = 0
	base_row_numbers


class IntensityIndicators:
	"""Base class for Sector Level and Economy-wide Intensity Indicators"""
	# index_base_year_primary = 1985
	# index_base_year_secondary = 1996  # not used
	# charts_starting_year = 1985
	# charts_ending_year = 2003 

	def __init__(self, df, ):
		self.dataset = df 


	def energy_consumption(self, ):
		"""
		Calculate energy shares
		Units TBtu
		"""
		self.total_us = 
		self.delivered_electricity = 
		self.source_electricity = 
		self.total_source = 
		self.source_deliveredelectricityratio = 


	def activity(self, ):
		"""millionpm_milliontm"""
		self.source = 


	def nominal_energy_intensity(self, ):
		"""btu_per_pm"""


	def weather_factors(self, ):
		""""""
		self.weather_factors_fuels = 
		self.weather_factors_electricity = 


	def structure_index(self,):
		""""""


	def energy_intensity_index(self, ):
		""""""


	def structure_index(self, ):
		""""""


	def final_indexes(self):  # 1985 = 1.0
		""""""
		# Decomposition of Source Energy
		self.weighted_activity_index = 
		self.index_of_aggregrate_intensity = 
		self.structure = 1 # Passenger vs Freight
		self.structure_lower_level = 
		self.component_intensity_index = 
		self.product_activity_structure_intensity = weighted_activity_index *  # Product: Activity x
													 # Structure X Intensity
		self.actual_energy_use = 
		self.total_structure = 

		# Decomposition with Delivered Intensity
		self.delivered_intensity_index = 
		self.structure_weather_delivered = 
		self.electrification_effect = 
		self.structure_electric_generation_efficiency = 
		self.aggregate_source_energy_intensity = 


	def energy_shares(self, fuels, source_electricity):
		""""""


	def log_mean_divisia_shares(self, fuels, source_electricity):
		""""""


	def log_mean_weights(self, fuels, source_electricity):
		""""""


	def log_mean_divisia_weights_normalized(self, fuels, source_electricity):
		""""""


	def log_changes_intensity(self, fuels, delivered_electricity):
		""""""


	def log_changes_total_delivered(self,):
		""""""


	def energy_share_delivered(self, fuels, electricity):
		""""""


	def log_changes_shares(self, fuels, electricity):
		""""""


	def log_changes_weather(self, fuels, delivered_electricity):
		""""""


	def log_changes_source_to_site(self,):
		""""""


	def activity_index(self, fuels, source_electricity):
		""""""


	def log_changes_activity(self, fuels, source_electricity):
		""""""


	def source_intensity(self, ):
		""""""


	def weather_adjustment(self, ):
		""""""


	def electric_power_sector(self,):
		""""""


	def total_delivered(self, ):
		"""check purposes only"""


	def electrification_effect(self,):
		""""""


def sum_row():
	""""Sum sub-categories of type"""

	

categories_total_transportation = ['All Passenger', 'All Freight']
categories_passenger_total = ['Highway', 'Rail', 'Air']
categories_passenger_highway = ['Passenger Cars and Trucks', 'Buses', 'Paratransit']
categories_personal_vehicles = ['Passenger Car – SWB Vehicles', 'Light Trucks – LWB Vehicles', 'Motorcycles']
categories_cars_and_swb_vehicles = ['Passenger Car', 'SWB Vehicles']
categories_light_trucks_and_lwb = ['Light Trucks', 'LWB Vehicles']
categories_buses = ['Urban Bus', 'Intercity Bus', 'School Bus']
categories_passenger_air = ['Commercial Carriers', 'General Aviation']
categories_passenger_rail = ['Urban Rail', 'Intercity Rail']
categories_commuter_rail = ['Commuter Rail', 'Heavy Rail', 'Light Rail']
categories_freight_total = ['Highway', 'Rail', 'Air', 'Waterborne', 'Pipeline']
categories_freight_trucks = ['Single-Unit Truck', 'Combination Truck']
categories_pipelines = ['Oil Pipeline', 'Natural Gas Pipeline']
categories_personal_vehicles_aggregate = ['Passenger Car', 'Light Truck', 'Motorcycles']

categories_residential_national = ['Northeast', 'Midwest', 'South', 'West']
categories_residential_northeast = ['Single-Family', 'Multi-Family', 'Manufactured Homes']
categories_residential_midwest = ['Single-Family', 'Multi-Family', 'Manufactured Homes']
categories_residential_south = ['Single-Family', 'Multi-Family', 'Manufactured Homes']
categories_residential_west = ['Single-Family', 'Multi-Family', 'Manufactured Homes']

categories_electricity = ['Elec Generation Total', 'All CHP']
categories_elec_generation = ['Elec Power Sector', 'Commercial Sector', 'Industrial Sector']
categories_elec_power_sector = ['Electricity Only', 'Combined Heat & Power']
categories_electricity_only = ['Fossil Fuels', 'Nuclear', 'Hydro Electric', 'Renewable']
categories_fossil_fuels = ['Coal', 'Petroleum', 'Natural Gas', 'Other Gasses']
categories_renewable = ['Wood', 'Waste', 'Geothermal', 'Solar', 'Wind']
categories_combined_heat_power = ['Fossil Fuels', 'Renewable']
categories_chp_renewable = ['Wood', 'Waste']
categories_all_chp = ['Elec Power Sector', 'Industrial Secotor']
categories_sub_all_chp = ['Fossil Fuels', 'Renewable', 'Other']


categories_economywide_all_sectors = ['Residential', 'Commercial', 'Industrial', 'Transportation']  # ‘Elec Power’

commercial = IntensityIndicators(commercial_df)
electricity = IntensityIndicators(electricity_df)
industry = IntensityIndicators(industry_df)
transportation = IntensityIndicators(transportation_df)





