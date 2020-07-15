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


# Total Transportation


class IntensityIndicators:
	"""Does this"""
	index_base_year_primary = 1985
	index_base_year_secondary = 1996  # not used
	charts_starting_year = 1985
	charts_ending_year = 2003 

	def __init__(self, df, ):
		self.dataset = df 
	def sum_row():
		""""Sum sub-categories of type"""
	def energy_consumption():
		"""
		Calculate energy shares
		Units TBtu
		"""
		self.total_us = 
		self.delivered_electricity = 
		self.source_electricity = 
		self.total_source = 
		self.source_deliveredelectricityratio = 

	def activity():
		"""millionpm_milliontm"""
		self.source = 
	def nominal_energy_intensity():
		"""btu_per_pm"""
	def weather_factors():
		""""""
		self.weather_factors_fuels = 
		self.weather_factors_electricity = 
	def structure_index():
		""""""
	def energy_intensity_index():
		""""""
	def structure_index():
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

	def energy_shares(fuels, source_electricity):
		""""""
	def log_mean_divisia_shares(fuels, source_electricity):
		""""""

	def log_mean_weights(fuels, source_electricity):
		""""""
	def log_mean_divisia_weights_normalized(fuels, source_electricity):
		""""""
	def log_changes_intensity(fuels, delivered_electricity):
		""""""
	def log_changes_total_delivered():
		""""""
	def energy_share_delivered(fuels, electricity):
		""""""
	def log_changes_shares(fuels, electricity):
		""""""
	def log_changes_weather(fuels, delivered_electricity):
		""""""
	def log_changes_source_to_site():
		""""""
	def activity_index(fuels, source_electricity):
		""""""
	def log_changes_activity(fuels, source_electricity):
		""""""
	def source_intensity():
		""""""
	def weather_adjustment():
		""""""
	def electric_power_sector():
		""""""
	def total_delivered():
		"""check purposes only"""
	def electrification_effect():
		""""""
	

transportation = IntensityIndicators(transportation_df)
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

categories_economywide_all_sectors = ['Residential', 'Commercial', 'Industrial', 'Transportation']  # ‘Elec Power’
commercial = IntensityIndicators(commercial_df)
electricity = IntensityIndicators(electricity_df)
industry = IntensityIndicators(industry_df)




