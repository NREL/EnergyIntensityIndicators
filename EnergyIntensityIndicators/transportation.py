import pandas as pd
from sklearn import linear_model
import zipfile
import numpy as np
import urllib

from EnergyIntensityIndicators.LMDI import CalculateLMDI
from EnergyIntensityIndicators.pull_eia_api import GetEIAData


class TransportationIndicators(CalculateLMDI):
    """Class to calculate Energy Intensity indicators for the U.S. Transportation Sector
    """    
    def __init__(self, directory, output_directory, level_of_aggregation, lmdi_model='multiplicative', tedb_date='04302020', base_year=1985, end_year=2018):
        self.transit_eia = GetEIAData('transportation')
        self.mer_table25_dec_2019 = self.transit_eia.eia_api(id_='711272', id_type='category') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711272'
        self.mer_table_43_nov2019 = self.transit_eia.eia_api(id_='711272', id_type='category') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711272'
        self.aer_2010_table_65 = self.transit_eia.eia_api(id_='711272', id_type='category') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711272'
        self.tedb_date = tedb_date
        self.energy_types = ['deliv']
        self.sub_categories_list = {'All_Transportation': {'All_Passenger':
                                                                {'Highway': 
                                                                    {'Passenger Cars and Trucks': 
                                                                        {'Passenger Car – SWB Vehicles': 
                                                                            {'Passenger Car': None, 'SWB Vehicles': None},
                                                                        'Light Trucks – LWB Vehicles': 
                                                                            {'Light Trucks': None, 'LWB Vehicles': None},
                                                                        'Motorcycles': None}, 
                                                                    'Buses': 
                                                                        {'Urban Bus': None, 'Intercity Bus': None, 'School Bus': None}, 
                                                                    'Paratransit':
                                                                        None}, 
                                                                'Rail': 
                                                                    {'Urban Rail': 
                                                                        {'Commuter Rail': None, 'Heavy Rail': None, 'Light Rail': None}, 
                                                                    'Intercity Rail': None}, 
                                                                'Air': {'Commercial Carriers': None, 'General Aviation': None}}, 
                                                            'All_Freight': 
                                                                {'Highway': 
                                                                        {'Single-Unit Truck': None, 'Combination Truck': None}, 
                                                                'Rail': None, 
                                                                'Air': None, 
                                                                'Waterborne': None,
                                                                'Pipeline': 
                                                                    {'Oil Pipeline': None, 'Natural Gas Pipeline': None}}}}
        super().__init__(sector='transportation', level_of_aggregation=level_of_aggregation, lmdi_models=lmdi_model, categories_dict=self.sub_categories_list, 
                         energy_types=self.energy_types, directory=directory, output_directory=output_directory, base_year=base_year, end_year=end_year,
                         unit_conversion_factor=1000000)

    def import_tedb_data(self, table_number, skip_footer=None, skiprows=None, sheet_name=None, usecols=None, index_col=None):
        try:
            file_url = f'https://tedb.ornl.gov/wp-content/uploads/2020/04/Table{table_number}_{self.tedb_date}.xlsx'
            xls = pd.read_excel(file_url, skipfooter=skip_footer, skiprows=skiprows, sheet_name=sheet_name, usecols=usecols, index_col=index_col)
            return xls
        except urllib.error.HTTPError:
            print('error with table', table_number)
            return 

    def adjust_truck_freight(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
        DOES THIS WORK ? dataframes
        """
        gross_output = pd.read_excel('./EnergyIntensityIndicators/Industry/Data/BLS_BEA_Data.xlsx', sheet_name='BLS_Data_011920', index_col=0)
        gross_output = gross_output.transpose()  # Note:  Gross output in million 2005 dollars from BLS database for their employment projections input-output model, 
                                    # PNNL spreadsheet: BLS_output_data.xlsx in folder BLS_Industry_Data)
        vehicle_miles_fhwa_tvm1 = pd.read_excel('https://www.fhwa.dot.gov/policyinformation/statistics/2018/xls/vm1.xlsx', header=4, index_col=0) 
        old_methodology_2007_extrapolated = gross_output.iloc[2007] / gross_output.iloc[2006] * vehicle_miles_fhwa_tvm1.iloc[2006, :]
        old_series_scaled_to_new = vehicle_miles_fhwa_tvm1 * vehicle_miles_fhwa_tvm1.iloc[2007, :] / old_methodology_2007_extrapolated  
        return old_series_scaled_to_new

    def passenger_based_activity(self):
        """Time series for the activity measures for passenger transportation
           1970-76: Oak Ridge National Laboratory. 1993. Transportation Energy Data Book, Edition 13. ORNL-6743. Oak Ridge, Tennessee, Table 3.30, p. 3-46.
           1977-2017 American Public Transportation Association. 2019 Public Transportation Fact Book. Appendix A, Table 3, Pt. A
           Note:  Transit bus does not include trolley bus because APTA did not not report electricity consumption for trolley buses separately for all years. 
           https://www.apta.com/research-technical-resources/transit-statistics/public-transportation-fact-book/

            Note: Series not continuous for transit bus between 2006 and 2007; estimation procedures changed for bus systems outside urban areas
            Note: Series not continuous between 1983 and 1984.
        """

        # Historical data from PNNL

        passenger_based_activity = pd.read_csv('./EnergyIntensityIndicators/Transportation/passenger_based_activity.csv').set_index('Year').rename(columns={'Motor-cycles': 'Motorcycles', 
                                                                                                                                                            'Light Truck': 'Light Trucks',
                                                                                                                                                            'Transit Bus': 'Urban Bus', 
                                                                                                                                                            'Para-Transit': 'Paratransit'})

         # Passenger cars and light trucks
        fha_table_vm1 = pd.read_excel('https://www.fhwa.dot.gov/policyinformation/statistics/2018/xls/vm1.xlsx', header=4, index_col=0) 
        # Bus / Transit
        apta_table3 = pd.read_excel('https://www.apta.com/wp-content/uploads/2020-APTA-Fact-Book-Appendix-A.xlsx', sheet_name='3', skiprows=7, skipfooter=42, index_col=0) # 1997-2017 apta_table3
         
        # # Bus / Intercity
        # see revised_intercity_bus_estimates

        # # Bus / School 
        # see revised_intercity_bus_estimates

        # Paratransit
        paratransit_activity_1984_2017 = pd.read_excel('https://www.apta.com/wp-content/uploads/2020-APTA-Fact-Book-Appendix-A.xlsx', sheet_name='3', skiprows=7, skipfooter=42, usecols='A,G', index_col=0) # 1997-2017 apta_table3
        
        # Commercial Air Carrier

        commercial_air_carrier = self.import_tedb_data(table_number='10_02')

        # Urban Rail (Commuter)
        urban_rail_commuter = pd.read_excel('https://www.apta.com/wp-content/uploads/2020-APTA-Fact-Book-Appendix-A.xlsx', sheet_name='3', skiprows=7, skipfooter=42, index_col=0,
                                             use_cols='A,L') # 1997-2017 apta_table3
        # Urban Rail (Light, Heavy)
        urban_rail_light_heavy = pd.read_excel('https://www.apta.com/wp-content/uploads/2020-APTA-Fact-Book-Appendix-A.xlsx', sheet_name='3', skiprows=7, skipfooter=42, usecols='A,O,P', index_col=0) # 1997-2017 apta_table3

        # Intercity Rail (Amtrak) Note: Amtrak data is compiled by fiscal year rather than calendar year
        intercity_rail_2017 =  pd.read_excel('https://www.bts.gov/sites/bts.dot.gov/files/table_01_40_091820.xlsx', skiprows=1, skipfooter=25, index_col=0)
        intercity_rail_2017 = intercity_rail_2017.transpose()

        intercity_rail_2017 = intercity_rail_2017[['Intercity/Amtraki']] # not sure how the i superscript shows up

                                 # 'Bureau of Transportation Statistics, National Transportation Statistics, Table 1-40: U.S. Passenger-Miles 

        return passenger_based_activity

    def passenger_based_energy_use(self):
        """Calculate the fuel consumption in Btu for passenger transportation
        """        
        passenger_based_energy_use = pd.read_csv('./EnergyIntensityIndicators/Transportation/passenger_based_energy_use.csv').set_index('Year').rename(columns={'Light Truck': 'Light Trucks', 
                                                                                                                                                            'Para-transit': 'Paratransit',
                                                                                                                                                            'Intercity Rail (Amtrak)': 'Intercity Rail'})

        passenger_based_energy_use['Paratransit'] = passenger_based_energy_use['Paratransit'].fillna(0.000000001)
        
        return passenger_based_energy_use

    def fuel_heat_content(self, gross=True):
        """Assumed Btu content of the various types of petroleum products. This dataframe is not time variant (no need to update it with subsquent indicator updates)
        Parameters
            gross (bool): if True use gross fuel heat content, if False use net
        """

        fuel_heat_content = pd.read_csv('./EnergyIntensityIndicators/Transportation/Data/fuel_heat_content.csv')
        if gross:
            fuel_heat_content_df = fuel_heat_content[['Fuel', 'Gross Content Fmt']]
        else:
            fuel_heat_content_df = fuel_heat_content[['Fuel', 'Net Content']]

        return fuel_heat_content_df

    def fuel_consump(self, parameter_list):
        """Time series of fuel consumption for the major transportation subsectors. Data are generally in 
        millions gallons or barrels of petroleum.
           Parameters
           ----------
           
           Returns
           -------
           
        """
        fha_table_vm1 = pd.read_excel('https://www.fhwa.dot.gov/policyinformation/statistics/2018/xls/vm1.xlsx', header=4, index_col=0)  # 2007-2017 FHA Highway Statistics Table VM-1
       
        apta_table59 = pd.read_excel('https://www.apta.com/wp-content/uploads/2020-APTA-Fact-Book-Appendix-A.xlsx', sheet_name='59', skiprows=2, skipfooter=24, index_col=0) # 1997-2017 apta_table3
        apta_table60 = pd.read_excel('https://www.apta.com/wp-content/uploads/2020-APTA-Fact-Book-Appendix-A.xlsx', sheet_name='60', skiprows=2, skipfooter=24, index_col=0) # 1997-2017 apta_table3

        swb_vehciles_all_fuel = fha_table_vm1
        motorcycles_all_fuel_1970_2017 = self.import_tedb_data(table_number='A_02')  # Alternative: 2017 data from Highway Statistics, Table VM-1.
        lwb_vehicles_all_fuel = fha_table_vm1 # 2007-2017 FHA Highway Statistics Table VM-1

        bus_urban_diesel = apta_table59 #[] #Table 59 in Appendix A, Bus Fuel Consumption (downloaded Excel file)							from: https://www.apta.com/research-technical-resources/transit-statistics/public-transportation-fact-book/							
							#Excel file:  2019-APTA-Fact-Book-Appendix-A.xlsx							

        bus_urban_cng = apta_table59 #[] #same as bus_urban_diesel
        bus_urban_lng = apta_table59 #[] #same as bus_urban_diesel
        bus_urban_other = apta_table59 #[] #same as bus_urban_diesel
        bus_urban_lpg = apta_table59 #[] #same as bus_urban_diesel
        paratransit_diesel = apta_table60 #[]						# http://www.apta.com/resources/statistics/Pages/transitstats.aspx								
        #                     https://www.apta.com/research-technical-resources/transit-statistics/public-transportation-fact-book/										Table 60 in Appendix A, Demand Response Fuel Consumption					

        general_aviation = self.import_tedb_data(table_number='A_08')
        commercial_air_carrier = self.import_tedb_data(table_number='A_09')


        rail_commuter = self.import_tedb_data(table_number='A_13') 
        rail_urban = self.import_tedb_data(table_number='A_15') 
        class_I_freight_distillate_fuel_oil = self.import_tedb_data(table_number='A_13') 

        pipeline_natural_gas_million_cu_ft = self.import_tedb_data(table_number='A_12')

        fuel_consump_df = []

        return fuel_consump_df

    def freight_based_energy_use(self, freight_based_activity):
        """Calculate the fuel consumption in Btu for freight transportation
        
        Need FuelConsump, Fuel Heat Content
        """        
        # Historical data from PNNL
        freight_based_energy_use = pd.read_csv('./EnergyIntensityIndicators/Transportation/freight_based_energy_use.csv').set_index('Year')
        freight_based_energy_use['Oil Pipeline'] = freight_based_activity['Oil Pipeline'].multiply(0.000001)

        natural_gas_delivered_to_end_users = self.aer_2010_table_65 # Column AH, million cu. ft.
        natural_gas_delivered_lease_plant_pipeline_fuel = self.mer_table_43_nov2019 # Column M - column D - column I
        natural_gas_delivered_lease_plant_pipeline_fuel.at[0] = 0.000022395
        natural_gas_consumption_million_tons = natural_gas_delivered_to_end_users.multiply(natural_gas_delivered_lease_plant_pipeline_fuel, axis=1)
        avg_length_natural_gas_shipment_miles = 620
        freight_based_energy_use['Natural Gas Pipeline'] = natural_gas_consumption_million_tons.multiply(avg_length_natural_gas_shipment_miles)

        return freight_based_energy_use

    def freight_based_activity(self):
        """Time series for the activity measures for passenger transportation
        """        
        # Historical data, from PNNL
        freight_based_activity = pd.read_csv('./EnergyIntensityIndicators/Transportation/freight_based_activity.csv').set_index('Year')
        freight_based_activity['Single-Unit Truck'] = freight_based_activity['Single-Unit Truck'].multiply(3)
        freight_based_activity['Oil Pipeline'] = freight_based_activity['Oil Pipeline'].multiply(0.000001)

        class_1_rail = pd.read_excel('https://www.bts.gov/sites/bts.dot.gov/files/table_04_25_112219.xlsx', skiprows=1, skipfooter=9, index_col=0) # USDOT, Bureau of Transportation Statistics			https://www.bts.gov/content/energy-intensity-class-i-railroad-freight-service					Table 4-25
        class_1_rail = class_1_rail.transpose()
        class_1_rail = class_1_rail[['Revenue freight ton-miles (millions)']]
        air_carrier = self.import_tedb_data(table_number='09_02')
        waterborne_vessles = self.import_tedb_data(table_number='10_05')
        gas_pipeline = self.mer_table_43_nov2019

        return freight_based_activity

    def water_freight_regression(self, intensity, actual_dd):
        X = np.log(intensity['intensity'])
        Y = intensity['Year']
        reg = linear_model.LinearRegression()
        reg.fit(X, Y)
        coefficients = reg.coef_
        predicted_value = reg.predict(actual_dd)  # Predicted value of the intensity based on actual degree days
        return predicted_value

    def detailed_data_school_buses(self, parameter_list):
        """"Adjusted miles - use number of buses times interpolated values for annual miles per bus.  
        For years ending 0 and 5, use published data							
        Thus, adjusted values match in years ending 0 and 5 with Column 4, and for 1994.  
        After 1994, values taken directly from 							
        School Bus Fleet Magazine estimates, adjusted for missing values							
        """
        avg_load_factor = 23
        adjusted_vehicle_miles = pd.read_csv('./EnergyIntensityIndicators/Transportation/Data/adjusted_data_school_bus.csv')
        revised_pass_miles = adjusted_vehicle_miles.multiply(avg_load_factor * 0.001)
        return revised_pass_miles

    def collect_data(self):
        """Method to collect freight and passenger energy and activity data. 
        This method should be adjusted to call dataframe building methods rather than reading csvs, when those are ready
        
        Gather Energy Data to use in LMDI Calculation TBtu and Activity Data to use in LMDI Calculation passenger-miles [P-M], ton-miles [T-M]
        Returns:
            [type]: [description]
        """        
        passenger_based_energy_use = self.passenger_based_energy_use()
        passenger_based_activity = self.passenger_based_activity()
        freight_based_activity = self.freight_based_activity()
        freight_based_energy_use = self.freight_based_energy_use(freight_based_activity)

        data_dict = {'All_Passenger':
                        {'Highway': 
                            {'Passenger Cars and Trucks': 
                                {'Passenger Car – SWB Vehicles': {'energy': {'deliv': passenger_based_energy_use[['Passenger Car', 'SWB Vehicles']]}, 'activity': passenger_based_activity[['Passenger Car', 'SWB Vehicles']]},
                                'Light Trucks – LWB Vehicles': {'energy': {'deliv': passenger_based_energy_use[['Light Trucks', 'LWB Vehicles']]}, 'activity': passenger_based_activity[['Light Trucks', 'LWB Vehicles']]},
                                'Motorcycles': {'energy': {'deliv': passenger_based_energy_use[['Motorcycles']]}, 'activity': passenger_based_activity[['Motorcycles']]}}, 
                            'Buses': {'energy': {'deliv': passenger_based_energy_use[['Urban Bus', 'Intercity Bus', 'School Bus']]}, 'activity': passenger_based_activity[['Urban Bus', 'Intercity Bus', 'School Bus']]}, 
                            'Paratransit':
                                {'energy': {'deliv': passenger_based_energy_use[['Paratransit']]}, 'activity': passenger_based_activity[['Paratransit']]}}, 
                        'Rail': 
                            # {'Urban Rail': {'energy': {'deliv': passenger_based_energy_use[['Commuter Rail', 'Heavy Rail', 'Light Rail']]}, 'activity': passenger_based_activity[['Commuter Rail', 'Heavy Rail', 'Light Rail']]}, 
                            {'Urban Rail': {'energy': {'deliv': passenger_based_energy_use[['Urban Rail (Hvy, Lt, Commuter)']]}, 'activity': passenger_based_activity[['Urban Rail (Hvy, Lt, Commuter)']]}, 

                            'Intercity Rail': {'energy': {'deliv': passenger_based_energy_use[['Intercity Rail']]}, 'activity': passenger_based_activity[['Intercity Rail']]}}, 
                        # 'Air': {'energy': {'deliv': passenger_based_energy_use[['Commercial Carriers', 'General Aviation']]}, 'activity': passenger_based_activity[['Commercial Carriers', 'General Aviation']]}}, 
                        'Air': {'energy': {'deliv': passenger_based_energy_use[['Carrier']]}, 'activity': passenger_based_activity[['Domestic Carriers (passenger-miles, millions)']]}}, 

                    'All_Freight': 
                        {'Highway': 
                                {'energy': {'deliv': freight_based_energy_use[['Single-Unit Truck', 'Combination Truck']]}, 'activity': freight_based_activity[['Single-Unit Truck', 'Combination Truck']]}, 
                        'Rail': {'energy': {'deliv': freight_based_energy_use[['Rail']]}, 'activity': freight_based_activity[['Rail']]}, 
                        'Air': {'energy': {'deliv': freight_based_energy_use[['Air']]}, 'activity': freight_based_activity[['Air']]}, 
                        'Waterborne': {'energy': {'deliv': freight_based_energy_use[['Waterborne']]}, 'activity': freight_based_activity[['Waterborne']]},
                        'Pipeline': {'energy': {'deliv': freight_based_energy_use[['Oil Pipeline', 'Natural Gas Pipeline']]}, 'activity': freight_based_activity[['Oil Pipeline', 'Natural Gas Pipeline']]}}}
        data_dict = {'All_Transportation': data_dict}
        return data_dict
       
    def main(self, breakout, calculate_lmdi): # base_year=None, 
        """Decompose Energy use for the Transportation sector
        """        

        data_dict = self.collect_data()

        if self.level_of_aggregation == 'personal_vehicles_aggregate':
            categories = {'Passenger Car': None, 'Light Truck': None, 'Motorcycles': None}
            results = None
        else: 
            results_dict, results = self.get_nested_lmdi(level_of_aggregation=self.level_of_aggregation, 
                                                         breakout=breakout, calculate_lmdi=calculate_lmdi, 
                                                         raw_data=data_dict, lmdi_type='LMDI-I')
        
        return results_dict

    def compare_aggregates(self, parameter_list):
        """Compare aggregates from MER and model
        """
        total_fuel_tbtu_published_mer = self.mer_table25_dec_2019['Total Energy Consumed by the Transportation Sector']  # j
        sum_of_modes = self.total_transportation['Energy_Consumption_Total']  # F
        difference = sum_of_modes - total_fuel_tbtu_published_mer
        pct_difference = difference / total_fuel_tbtu_published_mer
        return pct_difference

if __name__ == '__main__': 
    indicators = TransportationIndicators(directory='./EnergyIntensityIndicators/Data', 
                                          output_directory='./Results', 
                                          level_of_aggregation='All_Transportation', lmdi_model=['multiplicative', 'additive'],
                                          base_year=1985, end_year=2015) #  
    indicators.main(breakout=True, calculate_lmdi=True)


