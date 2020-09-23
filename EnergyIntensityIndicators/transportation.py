import pandas as pd
from sklearn import linear_model
import zipfile
from outline import LMDI


# Passenger cars, short wheelbase vehicles
total_fuel_gallons =  # TEDB-37 Table 4.1
total_energy_tbtu =  # TEDB-37 Table A.18
vehicle_miles =  # TEDB-37 Table 4.1
passenger_miles = # TEDB-37 Table A.18

#Light trucks, long wheelbase vehicles
total_fuel_gallons = # TEDB-37 Table 4.2
total_energy_tbtu = # TEDB-37 Table A.5
vehicle_miles =  # TEDB-37 Table 4.2
passenger_miles = # TEDB-37 Table A.19

# Motorcycles
total_fuel_gallons = # TEDB-37 Table A.2
total_energy_tbtu = # Assume all motorcycle fuel is gasoline
vehicle_miles =  # Bureau of Transportation Statistics Table 1-35 with some interpolation prior to 1990
passenger_miles = vehcile_miles * 1.1 # Assumed load factor fo 1.1 (for all years) X vehicle miles

# Transit Buses (is this the same as urban buses?)
total_fuel_gallons = # American Public Transit Association (APTA), Public Transportation Fact Book
                        # https://www.apta.com/research-technical-resources/transit-statistics/public-transportation-fact-book/
total_energy = # Apply conversion factors (Btu/gallon). Note: APTA reports GNG in diesel equivalent
                # Biodiesel conversion factor from AER 2008, p. 373
passenger_miles = # APTA

# Intercity Buses
total_fuel_gallons = # American Bus Association (ABA) with earlier data in TEDB-19 and later from ABA. 
                        # Fuel use was estimated on basis of separate estimates of average MPG and vehicle miles.
total_energy_tbtu = # All fuel for intercity buses assumed to be diesel
passenger_miles = # BTS Annual Report published in 1993. Table 6 for estimates from 1970 trhough 1991. For 1991-1997
                    # TEDB-23, Table 5.23. For 1998-2006, estimates of vehiclemiles interpolated between 1997 value
                    #  from TEDB-19 and ABA 2007 value. Average passenger load factors interpolated over same period.
                    # Passenger-miles computed as load factor x vehicle miles. Most recent estimates from ABA reports. 

# School buses
total_fuel_gallons = # TEDB-37 Table A.4 for years 1970-1994 at typically five-year intervals. Data after 1994 extrapolated 
                        # by vehicle-miles (not used). Estimates of fuel use interpolated for years other than ending in 0 or 5, 
                        # based upon estimates of vehicle-miles and average fuel economy. The approach was used over the period 
                        # 1970-1994
total_energy_tbtu = # TEDB-37, Table A.4. Fuel assumed to be 90% diesel, 10% gasoline
passenger_miles = # 1980-1994: TEDB, various issues for vehicle-miles. No estimates for recent years. 1995-2011: 1994 estimate 
                    #  extrapolated from U.S. route mileage for public school transportation, used by permission from School Bus Fleet 
                    # magazine, Fact Books. Passenger-miles calculated from assumed constant pupil load of 23

# Commuter rail 
fuel_use_kwh_gallons =  # APTA Public Transportation Fact Book
energy_use_tbtu =  # Conversion to TBtu with electricity and diesel energy conversion factors (electricity conversion at 
                    # 10,339 Btu/kWh from TEDB
passeneger_miles =  # APTA source as above for fuel use. Passenger-miles in Table 3 of Appendix A of APTA publication.

# Transit Rail (Heavy and Light Rail)
fuel_use_kwh_gallons =  # same source as commuter rail (APTA, Fact Book), Tables 38 and 39 in Appendix A
energy_use_tbtu =  # conversion to tbtu as for commuter rail
passenger_miles =  # APTA source as avober for fuel use. Passenger-miles in Table 3 of Appendix

# Intercity Rail (Amtrak)
fuel_use_kwh_gallons =  # 1994-2016" TEDB-37 Table A.16; prior data extrapolated from TEDB-30, Table 9.10
energy_use_tbtu =  # Conversion to TBtu as for commuter rail 
passenger_miles =  # TEDB-37, Table 9.10

# Air Carriers
fuel_use_gallons =  # TEDB-37, Table A.9
energy_use =  # TEDB-37 Table 9.2 includes all energy for domestic and international operations, (Table 2.12 includes only domestic air 
                # service). 1) All fuel is assumed to be jet fuel. 2) Allocation between passenger and freight transportation based on ton-
                # miles. 1 pass-mile = 0.1 ton-mile, as per BTS. Passenger percentage of ton-mile ~ 83% in 2017
passgenger_miles =  # TEDB-37 Table 9.2 and previous TEDB's for years before 1985

# General Aviation
fuel_use_gallons =  # TEDB-37, Table A.10
energy_use_tbtu =  # Conversion to Btu with factors for aviation fuel and jet fuel (120,2 and  135.0 kBtu/gallon, respectively.)
passenger_miles =  # 1970-2001: Eno Transportation Foundation, Transportation in America 2001, 19th Edition, p.45 Passenger-miles after 2001
                    # extrapolated by total energy use. 

# Single-Unit Trucks
fuel_use_gallons =  # Pre-2007 revised to match current FHWA methodology, see text and Table A.13.
energy_use_tbtu =  # Total fuel allocated among gasoline, gasohol, and diesel. Source: TEDB-37, Table A.6  Implausible estimate from
                    # 1982 Truck Inventory and Use survey (TIUS) showing diesel at only 60% of truck fuel. Changed to remain above 
                    # 80% in years proximate to 1982
vehicle_miles =  # Pre-2007 revised to match current FHWA methodology, see text and Table A.13 in this report..
ton_miles =  # Estimate based upon assumed average load of 3 tons per truck (x vehiclemiles)

# Combination Trucks
fuel_use_gallons =  # Pre-2007 revised to match current FHWA methodology, see text and Table A.13.
energy_use_tbtu =  # Total fuel allocated among gasoline, gasohol, and diesel. Source: TEDB-37, Table A.6
ton_miles =  # BTS-NTS Table 1-49 BTS publishes ton-miles for intercity truck (1990-2003) from most recent edition of Transportation
                # in America, published by Eno Transportation Foundation in 2007. Column to right shows data used to extrapolate 
                # before 1990 and after 2003

# Rail
fuel_use_gallons =  # TEDB-37, Table A.13
energy_use_tbtu =  # Convert to TBtu with conversion factors for diesel fuel (matches TEDB-37 estimates)
ton_miles =  # TEDB-37, Table 9.8       

# Air Carriers
fuel_use_gallons =  # TEDB-37, Table A.9, Reports fuel use for both domestic and international operations
energy_use_freight =  # 1)All fuel is assumed to be jet fuel 2) Allocation between passenger and freight transportation based on tonmiles.
                        # 1 pass-mile = 0.1 ton-mile, as per BTS. Passenger percentage ~ 70% in 2011.
ton_miles =  # BTS, Airline Data and Statistics: revenue ton-miles, http://www.bts.gov/xml/air_traffic/src/index.xml#CustomizeTable Data 
                # for both domestic and international operations


# Waterborne
fuel_use_gallons =  # TEDB-37, Table A.10 (Used for pre-1997 regression, see text)
energy_use_tbtu =  # TEDB-32, Table 2.15 presents improved estimates of intensity (Btu/ton-mile). Intensities used with domestic
                    # ton-miles to estimate historical energy consumption.
ton_miles =  # TEDB-37, Table 9.5

# Natural Gas Pipelines
fuel_use_cubic_feet_kwh =  # TEDB-37, Table A.12, Reports natural gas and electricity. Gas from EIA, electricity is estimated, see
                            # note in TEDB EIA, Annual Energy Review 2011, Table 6.5 (Natural gas used as fuel in delivery to customers)
                            #  More recent data from Monthly Energy Review, Table 4.3
energy_tbtu =  # Conversion to energy units with factor of 1,031 Btu/cu.ft for gas, and 10,339 Btu/kWh for electricity
ton_miles = # Natural gas converted to tons using methane density of 0.0448 lb/cu.ft. Average length of travel for natural 
            # gas assumed to be 620 miles

# Oil Pipelines: No energy intensity indicator, as there are no historical series for both energy use and ton-miles of oil 
# transported through pipelines





       
        # self.transit_bus = 'READ NOTE ^'
        # self.paratransit = "1984:20172019 Public Transportation Fact Book, Appendix A, Table 3, Pt. A		
        # 					" https://www.apta.com/research-technical-resources/transit-statistics/public-transportation-fact-book/
        #                     " Note: Data prior to 1984 was not collected by APTA ""




class TransportationIndicators(LMDI):

    def __init__(self, tedb_date='04302020'):
        self.transit_eia = GetEIAData('transportation')
        self.sub_categories_list = {'All_Passenger':
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
                                        {'Freight-Trucks': 
                                            {'Single-Unit Truck': None, 'Combination Truck': None}}, 
                                    'Rail': None, 
                                    'Air': None, 
                                    'Waterborne': None,
                                    'Pipeline': 
                                        {'Oil Pipeline': None, 'Natural Gas Pipeline': None}}}
        self.mer_table25_dec_2019 = self.transit_eia.eia_api(id_='711272') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711272'
        self.mer_table_43_nov2019 = self.transit_eia.eia_api(id_='711272') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711272'
        self.aer_2010_table_65 = self.transit_eia.eia_api(id_='711272') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711272'
        self.tedb_date = tedb_date

        self.transportation_data = {'Passenger Car – SWB Vehicles': {'total_fuel': 
                                                                {'unit': 'gallons', 'source': 'TEDB', 'table_number': '4_01', 'header_starts': 8, 'column_name': 'Fuel use'} # Table4_01_{date}
                                                            'total_energy': 
                                                                {'unit': 'tbtu', 'source': 'TEDB', 'table_number': 'A_18'}
                                                            'vehicle_miles': 
                                                                {'unit': 'miles', 'source': 'TEDB', 'table_number': '4_01', 'header_starts': 8, 'column_name': }
                                                            'passenger_miles': 
                                                                {'unit': 'miles', 'source': 'TEDB', 'table_number': 'A_18'}},
                                    'Light Trucks – LWB Vehicles': {'total_fuel': 
                                                                            {'unit': 'gallons', 'source': 'TEDB', 'table_number': '4_02', 'header_starts': 7}
                                                                        'total_energy': 
                                                                            {'unit': 'tbtu', 'source': 'TEDB', 'table_number': 'A_05'}
                                                                        'vehicle_miles': 
                                                                            {'unit': 'miles', 'source': 'TEDB', 'table_number': '4_02'}
                                                                        'passenger_miles':
                                                                            {'unit': 'miles', 'source': 'TEDB', 'table_number': 'A_19'}},
                                    'Motorcycles': {'total_fuel': 
                                                            {'unit': 'gallons', 'source': 'TEDB', 'table_number': 'A_02'}
                                                        'total_energy': 
                                                            {'unit': 'tbtu', 'source': None, 'table_number': None, 'Assumptions': 'Assume all motorcycle fuel is gasoline'}
                                                        'vehicle_miles': 
                                                            {'unit': 'miles', 'source': 'BTS', 'table_number': '1-35'}  #  with some interpolation prior to 1990'
                                                        'passenger_miles':
                                                            {'unit': 'miles', 'source': None, 'table_number': None, 'Assumptions': vehicle_miles * 1.1}},
                                    'Urban Bus': {'total_fuel': 
                                                        {'unit': 'gallons', 'source': 'APTA Public Transportation Fact Book'}
                                                'total_energy': 
                                                    {'unit': '', 'source': None, 'table_number': None, 'Assumption': 'Apply conversion factors (Btu/gallon)'}
                                                'passenger_miles':
                                                    {'unit': 'miles', 'source': 'APTA'}},
                                    'Intercity Bus': {'total_fuel': 
                                                        {'unit': 'gallons', 'source': 'Earlier data from TEDB-19 later from ABA'}
                                                        'total_energy': 
                                                        {'unit': 'tbtu', 'source': None, 'Assumptions': 'All fuel assumed to be diesel'}
                                                        'passenger_miles':
                                                        {'unit': 'miles', 'source': 'Multiple'}},
                                    'School Bus': {'total_fuel': 
                                                        {'unit': 'gallons', 'source': 'TEDB', 'table_number': 'A_04'}
                                                    'total_energy': 
                                                        {'unit': 'tbtu', 'source': 'TEDB', 'table_number': 'A_04', 'Assumptions': 'Fuel assumed to be 90% diesel, 10% gasoline'}
                                                    'passenger_miles':
                                                        {'unit': 'miles', 'source': 'Multiple'}},
                                    'Commuter Rail': {'total_fuel': 
                                                        {'unit': 'kwh_gallons', 'source': 'APTA Public Transportation Fact Book'}
                                                        'total_energy': 
                                                        {'unit': 'tbtu', 'source': 'TEDB', 'Assumptions': 'Conversion to TBtu with electricity and diesel energy conversion factors (electricity conversion at 10,339 Btu/kWh from TEDB'}
                                                        'passenger_miles':
                                                        {'unit': 'miles', 'source': 'APTA', 'table_number': ' Table 3 of Appendix A'}},
                                    'Transit Rail (Heavy and Light Rail)': {'total_fuel': 
                                                                                {'unit': 'kwh_gallons', 'source': '(APTA, Fact Book)' , 'table_number': 'Tables 38 and 39 in Appendix A'}
                                                                            'total_energy': 
                                                                                {'unit': 'tbtu', 'source': None, 'table_number': None, 'Assumptions': 'Conversion to TBtu as for commuter rail'}
                                                                            'passenger_miles':
                                                                                {'unit': 'miles', 'source': 'APTA', 'table_number': 'source as avober for fuel use. Passenger-miles in Table 3 of Appendix'}},
                                    'Intercity Rail': {'total_fuel': 
                                                            {'unit': 'kwh_gallons', 'source': '1994-2016" TEDB-37 Table A.16; prior data extrapolated from TEDB-30, Table 9.10'}
                                                        'total_energy': 
                                                            {'unit': 'tbtu', 'source': 'Conversion to TBtu as for commuter rail'}
                                                        'passenger_miles':
                                                            {'unit': 'miles', 'source': 'APTA source as avober for fuel use. Passenger-miles in Table 3 of Appendix'}},
                                    'Commercial Carriers': {'total_fuel': 
                                                                {'unit': 'gallons', 'source': 'TEDB', 'table_number': 'A_09'}, 
                                                            'total_energy':
                                                                {'unit': 'tbtu', 'source': 'Conversion to Btu with factors for aviation fuel and jet fuel (120,2 and  135.0 kBtu/gallon, respectively.)'},
                                                            'passenger_miles':
                                                                {'unit': 'miles', 'source': '1970-2001: Eno Transportation Foundation, Transportation in America 2001, 19th Edition, p.45 Passenger-miles after 2001 extrapolated by total energy use'}},
                                    'General Aviation': {'total_fuel': 
                                                                {'unit': 'gallons', 'source': 'TEDB', 'table_number': 'A_10'}, 
                                                            'total_energy':
                                                                {'unit': 'tbtu', 'source': 'Conversion to Btu with factors for aviation fuel and jet fuel (120,2 and  135.0 kBtu/gallon, respectively.)'},
                                                            'passenger_miles':
                                                                {'unit': 'miles', 'source': ' 1970-2001: Eno Transportation Foundation, Transportation in America 2001, 19th Edition, p.45 Passenger-miles after 2001 extrapolated by total energy use.'}},
                                //////     'Single-Unit Truck': {'total_fuel': 
                                                                {'unit': 'gallons', 'source': 'TEDB', 'table_number': 'A_09'}, 
                                                            'total_energy':
                                                                {'unit': 'tbtu', 'source': 'Conversion to Btu with factors for aviation fuel and jet fuel (120,2 and  135.0 kBtu/gallon, respectively.)'},
                                                            'vehicle_miles':
                                                                {'unit': 'miles', 'source': ' 1970-2001: Eno Transportation Foundation, Transportation in America 2001, 19th Edition, p.45 Passenger-miles after 2001 extrapolated by total energy use.'},
                                                            'ton_miles': {'unit': , 'source': }},
                                    'Combination Truck': {'total_fuel': 
                                                                {'unit': 'gallons', 'source': 'TEDB', 'A_10'}, 
                                                            'total_energy':
                                                                {'unit': 'tbtu', 'source': 'Conversion to Btu with factors for aviation fuel and jet fuel (120,2 and  135.0 kBtu/gallon, respectively.)'},
                                                            'vehicle_miles':
                                                                {'unit': 'miles', 'source': ' 1970-2001: Eno Transportation Foundation, Transportation in America 2001, 19th Edition, p.45 Passenger-miles after 2001 extrapolated by total energy use.'}},
                                    'Rail': {'total_fuel': 
                                                                {'unit': 'gallons', 'source': 'TEDB', 'A_10'}, 
                                                            'total_energy':
                                                                {'unit': 'tbtu', 'source': 'Conversion to Btu with factors for aviation fuel and jet fuel (120,2 and  135.0 kBtu/gallon, respectively.)'},
                                                            'vehicle_miles':
                                                                {'unit': 'miles', 'source': ' 1970-2001: Eno Transportation Foundation, Transportation in America 2001, 19th Edition, p.45 Passenger-miles after 2001 extrapolated by total energy use.'}},
                                    'Air Carriers': {'total_fuel': 
                                                                {'unit': 'gallons', 'source': 'TEDB', 'A_10'}, 
                                                            'total_energy':
                                                                {'unit': 'tbtu', 'source': 'Conversion to Btu with factors for aviation fuel and jet fuel (120,2 and  135.0 kBtu/gallon, respectively.)'},
                                                            'vehicle_miles':
                                                                {'unit': 'miles', 'source': ' 1970-2001: Eno Transportation Foundation, Transportation in America 2001, 19th Edition, p.45 Passenger-miles after 2001 extrapolated by total energy use.'}},
                                    'Waterborne': {'total_fuel': 
                                                                {'unit': 'gallons', 'source': 'TEDB', 'A_10'}, 
                                                            'total_energy':
                                                                {'unit': 'tbtu', 'source': 'Conversion to Btu with factors for aviation fuel and jet fuel (120,2 and  135.0 kBtu/gallon, respectively.)'},
                                                            'vehicle_miles':
                                                                {'unit': 'miles', 'source': ' 1970-2001: Eno Transportation Foundation, Transportation in America 2001, 19th Edition, p.45 Passenger-miles after 2001 extrapolated by total energy use.'}},
                                    'Natural Gas Pipelines': {'total_fuel': 
                                                                {'unit': 'gallons', 'source': 'TEDB', 'A_10'}, 
                                                            'total_energy':
                                                                {'unit': 'tbtu', 'source': 'Conversion to Btu with factors for aviation fuel and jet fuel (120,2 and  135.0 kBtu/gallon, respectively.)'},
                                                            'vehicle_miles':
                                                                {'unit': 'miles', 'source': ' 1970-2001: Eno Transportation Foundation, Transportation in America 2001, 19th Edition, p.45 Passenger-miles after 2001 extrapolated by total energy use.'}},}

    def import_tedb_data(self, table_number):
        file_url = f'https://tedb.ornl.gov/wp-content/uploads/2020/04/Table{table_number}_{self.tedb_date}.xlsx'
        xls = pd.read_excel(file_url)  # , sheetname=None, header=11
        return xls
    
    def get_data_from_nested_dict(self, nested_dictionary, subcategory_data_sources):
        for key, value in nested_dictionary.items():
            if type(value) is dict:
                get_data_from_nested_dict(value)
            elif value is None:
                data_sources = self.transportation_data[key]
                subcategory_data_sources[key] = data_sources
                return key, data_sources

    def load_data(self):
        energy_mode_data = dict()
        fuel_mode_data = dict()
        vehcile_miles_mode_data = dict()

        for mode_name, mode_dict in self.transportation_data.items():  # could be more efficient with pandas?
            for variable_name, variable_dict in mode_dict.items():
                if variable_name == 'total_fuel':

                elif variable_name == 'total_energy':
                
                elif variable_name == 'vehicle_miles'
                if variable_dict['source'] == 'TEDB':
                    mode_source_df = import_tedb_data(variable_dict['table_number'])

        get_data_from_nested_dict(self.sub_categories_list, subcategory_data_sources)


        pass
    

                
    def fuel_heat_content(self, parameter_list):
            """Assumed Btu content of the various types of petroleum products. This dataframe is not time variant (no need to update it with subsquent indicator updates)
            Parameters
            ----------
            
            Returns
            -------
            
            """

            pass

    def fuel_consump(self, parameter_list):
        """Time series of fuel consumption for the major transportation subsectors. Data are generally in 
        millions gallons or barrels of petroleum.
           Parameters
           ----------
           
           Returns
           -------
           
        """


        swb_vehciles_all_fuel =  # 2007-2017 FHA Highway Statistics Table VM-1
        motorcycles_all_fuel_1949_1969 = # Based upon data published in Table 4_11 of Bureau of Transportation Statistics (BTS), fuel consumption is based assumption of 50 miles per gallon
        motorcycles_all_fuel_1970_2017 = import_tedb_data(table_number='A_02')  # Alternative: 2017 data from Highway Statistics, Table VM-1.
        light_trucks_all_fuel = 
        lwb_vehicles_all_fuel = 

        bus_urban_diesel = 
        bus_urban_cng = 
        bus_urban_gasoline = 
        bus_urban_lng = 
        bus_urban_bio_diesel = 
        bus_urban_other = 
        bus_urban_lpg = 
        paratransit = 
        bus_school = 
        bus_school_million_bbl = 
        bus_intercity =  # Not used
        waterborne = 
        air = 
        rail_intercity_diesel_million_gallons = 
        rail_intercity_diesel_electricity_gwhrs = 
        rail_intercity_total_energy_tbtu =
        rail_intercity_total_energy_tbtu_old =
        rail_intercity_total_energy_tbtu_adjusted = 

        rail_commuter_diesel = 
        rail_commuter_electricity_gwhrs = 
        rail_heavy_electricity_gwhrs = 
        rail_light_electricity_gwhrs = 
        class_I_freight_distillate_fuel_oil =

        pipeline_natural_gas_million_cu_ft = 
        pipeline_natutral_gas_electricity_million_kwh = 



        pass

    def adjust_truck_freight(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
        DOES THIS WORK ? dataframes
        """
        gross_output = bls_data[] # Note:  Gross output in million 2005 dollars from BLS database for their employment projections input-output model, 
                    # PNNL spreadsheet: BLS_output_data.xlsx in folder BLS_Industry_Data)
        vehicle_miles_fhwa_tvm1 =  
        old_methodology_2007_extrapolated = gross_output.iloc[2007] / gross_output.iloc[2006] * vehicle_miles_fhwa_tvm1.iloc[2006, :]
        old_series_scaled_to_new = vehicle_miles_fhwa_tvm1 * vehicle_miles_fhwa_tvm1.iloc[2007, :] / old_methodology_2007_extrapolated  


        **come back**
        pass

    def passenger_based_energy_use(self):
        """Calculate the fuel consumption in Btu for passenger transportation
        """        
        all_passenger_categories = self.sub_categories_list['All_Passenger']
        for passenger_category in all_passenger_categories.keys():
            for transportation_mode in passenger_category.keys():
                if transportation_mode == 'Passenger Car – SWB Vehicles':
                    # Passenger Car and SWB Vehicles have separate data sources, later aggregated?

                elif transportation_mode == 'Light Trucks – LWB Vehicles':
                    # Light Trucks and LWB Vehicles have separate data sources, later aggregated?
                else: 
        urban_rail_categories = list(all_passenger_categories['Rail']['Urban Rail'].values()])
        passenger_based_energy_use_df['Urban Rail (Hvy, Lt, Commuter)'] = passenger_based_energy_use_df[urban_rail_categories].sum(1)
        pass
    
    def passenger_based_activity_data(self):
         # Passenger cars and light trucks
        fha_table_vm201a = pd.read_excel('https://www.fhwa.dot.gov/ohim/summary95/vm201a.xlw', sheetname='UnknownSheet4', header=11) # Passenger car and light truck VMT 1981-95
        fha_table_vm1 = pd.read_excel('https://www.fhwa.dot.gov/policyinformation/statistics/2018/xls/vm1.xlsx', header=4) 
        fha_table_vm1_1996  = pd.read_excel('https://www.fhwa.dot.gov/ohim/1996/vm1.xlw')
        fha_table_vm1_1997 = pd.read_excel('https://www.fhwa.dot.gov/ohim/hs97/xls/vm1.xls', sheetname='1997 VM1', header=6)
        fha_table_vm1_1998 = pd.read_excel('https://www.fhwa.dot.gov/policyinformation/statistics/1998/xls/vm1.xls', sheetname='Pub 98 VM1', header=6)
        fha_table_vm1_1999 = pd.read_excel('https://www.fhwa.dot.gov/ohim/hs99/excel/vm1.xls', sheetname='1999 VM1', header=3)
        fha_table_vm1_2000 = pd.read_excel('https://www.fhwa.dot.gov/ohim/hs00/xls/vm1.xls', sheetname='Sheet1', header=4)
        fha_table_vm1_2001 = pd.read_excel('https://www.fhwa.dot.gov/ohim/hs01/xls/vm1.xls', sheetname='Sheet1', header=4)
        vmt2002_2006_passenger_car_light_truck = pd.read_excel('https://www.bts.gov/sites/bts.dot.gov/files/table_01_35_013020.xlsx', header=1) # Note, this link has changed since usage in the spreadsheets
        vmt2007_2008_passenger_car = import_tedb_data(table_number='4_01')
        vmt2007_2008_light_truck = import_tedb_data(table_number='4_02')

        load_factors_1966_2011 = 

        # Short Wheelbase and Long Wheelbase Vehicles 
        2007-2017 FHA VM1
        fha_table_vm1_2014 = pd.read_excel('https://www.fhwa.dot.gov/policyinformation/statistics/2014/xls/vm1.xlsx', sheetname='final VM-1' , header=4)

        # Motorcycles
        1970-2010 = vmt2002_2006_passenger_car_light_truck
        fha_tables 

        # Bus / Transit
        # 1970-76: Oak Ridge National Laboratory. 1993. Transportation Energy Data Book, Edition 13. ORNL-6743. Oak Ridge, Tennessee, Table 3.30, p. 3-46.
        apta_table3 = pd.read_excel('https://www.apta.com/wp-content/uploads/2020-APTA-Fact-Book-Appendix-A.xlsx', sheetname='3', header=2) # 1997-2017
         
        # Bus / Intercity
        see revised_intercity_bus_estimates

        # Bus / School 
        see revised_intercity_bus_estimates

        # Paratransit
        paratransit_activity_1984_2017 = apta_table3
        
        # Commercial Air Carrier
        commercial_air_carrier_activity_1970_1974 = # TEDB 13 Table 6.2 page 6-7
        commercial_air_carrier_activity_1975_1984 = # TEDB 19 Table 12.1 page 12-2 (is the 2 a typo?)
        commercial_air_carrier_activity_1985_1999 = # TEDB 21 Table 12.1 page 12-2 (is the 2 a typo?)
        commercial_air_carrier_activity_2000_2018 = import_tedb_data(table_number='10_02')
        domestic_passenger_miles_2011 = # from Table 2.12 in 2013 TEDB

        # Urban Rail (Commuter)
        urban_rail_commuter_activity_1970_1976 = # Transportation Energy Data Book, Edition 13. ORNL-6743. Oak Ridge, Tennessee, Table 6.13, p. 6-31.
        urban_rail_commuter_activity_1977_2017 = apta_table3
        # Urban Rail (Light, Heavy)
        urban_rail_light_heavy_1970_1977 = import_tedb_data(table_number='10_10')
        urban_rail_light_heavy_1977_2017 = apta_table3

        # Intercity Rail (Amtrak) Note: Amtrak data is compiled by fiscal year rather than calendar year
        intercity_rail_1970 = None  # Amtrak not established until May 1971. Data for 1970 assumes same passenger activity as 1971, primarily included
                                    # to fill out all transportation-related time series back to 1970
        intercity_rail_1971_2016 = 
        intercity_rail_2017 =   # 'Bureau of Transportation Statistics, National Transportation Statistics, Table 1-40: U.S. Passenger-Miles

        # General Aviation 
        !!!!
        return passenger_based_activity_input_data

    def passenger_based_activity(self):
        """ Time series for the activity measures for passenger transportation
         

        # Highway
        self.intercity_buses = 'Detailed  data_Intercity buses' # Column E

        1970-76: Oak Ridge National Laboratory. 1993. Transportation Energy Data Book, Edition 13. ORNL-6743. Oak Ridge, Tennessee, Table 3.30, p. 3-46.
           1977-2017 American Public Transportation Association. 2019 Public Transportation Fact Book. Appendix A, Table 3, Pt. A
           Note:  Transit bus does not include trolley bus because APTA did not not report electricity consumption for trolley buses separately for all years. 
           https://www.apta.com/research-technical-resources/transit-statistics/public-transportation-fact-book/

            Note: Series not continuous for transit bus between 2006 and 2007; estimation procedures changed for bus systems outside urban areas
            Note: Series not continuous between 1983 and 1984.

         Note: Transit Bus == Urban Bus 
        This method is horribly structured. 
        """        


        all_passenger_categories = self.sub_categories_list['All_Passenger']
        for passenger_category in all_passenger_categories.keys():
            for transportation_mode in all_passenger_categories[passenger_category].keys():
                if transportation_mode == 'Passenger Car – SWB Vehicles':
                    # Passenger Car and SWB Vehicles have separate data sources, later aggregated?
                elif transportation_mode == 'Light Trucks – LWB Vehicles':
                    # Light Trucks and LWB Vehicles have separate data sources, later aggregated?
                else: 
        urban_rail_categories = list(all_passenger_categories['Rail']['Urban Rail'].values()])
        passenger_based_energy_use_df['Urban Rail (Hvy, Lt, Commuter)'] = passenger_based_energy_use_df[urban_rail_categories].sum(1)
        
        pass

    def freight_based_energy_use(self):
        """Calculate the fuel consumption in Btu for freight transportation
        
        Need FuelConsump, Fuel Heat Content
        """        
        all_freight_categories = self.sub_categories_list['All_Freight']
        for freight_category in all_freight_categories.keys():
            for freight_mode in freight_category.keys():



        pass

    def freight_based_activity(self):
        """Time series for the activity measures for passenger transportation
        """        
        all_freight_categories = self.sub_categories_list['All_Freight']
        for freight_category in all_freight_categories.keys():
            for freight_mode in freight_category.keys():

                if freight_category == 'Waterborne':
                    domestic_vessel = 
                    international_vessel_in_us_waters = 
                    total_commerce_us_waters = domestic_vessel + international_vessel_in_us_waters
                
                elif freight_category == 'Highway':
                    highway_published = 
                    
                    if freight_mode == 'Combination Truck':
                        freight_based_energy_use_df['Old Series from 2001 Eno, Trans. In America'] = 

                        # 1950-1989: 
                        freight_based_energy_use_df['Combination Truck, adjusted extrapolated'] = # 1990 value for this column divided by 1990 value for 'Old Series from 2001 Eno, Trans. In America' * contemporary year from old series
                        # 1990-2003
                        freight_based_energy_use_df['Combination Truck, adjusted extrapolated'] = 
                        # 2004-2017
                        vehicle_miles_combination_trucks_adjusted =  # from adjust truck freight column K

                        freight_based_energy_use_df['Combination Truck, adjusted extrapolated'] = # contemporary value of vehicle_miles_combination_trucks_adjusted divided by 2003 value of vehicle_miles_combination_trucks_adjusted times  2003 value of this
                    
                    elif freight_mode == 'Single-Unit Truck':
                        # 1970-2006 
                        freight_based_energy_use_df['Single-Unit Truck (million vehicle-miles), adjusted'] =  # Adjust_truck_Freight Column J
                        # 2007-2017
                        freight_based_energy_use_df['Single-Unit Truck (million vehicle-miles), adjusted'] = highway_published['Single-Unit Truck (million vehicle-miles)']


                if freight_mode == 'Natural Gas Pipeline':
                    natrual_gas_delivered_to_end_users = self.table65_AER2010 # Column AH, million cu. ft.
                    natural_gas_delivered_lease_plant_pipeline_fuel = self.MER_Table43_Nov2019 # Column M - column D - column I
                    natural_gas_delivered_lease_plant_pipeline_fuel.at[0] = 0.000022395
                    natural_gas_consumption_million_tons = natrual_gas_delivered_to_end_users * natural_gas_delivered_lease_plant_pipeline_fuel[0]
                    avg_length_natural_gas_shipment_miles = 620
                    freight_based_energy_use_df[freight_mode] = natural_gas_consumption_million_tons * 620
                
                


        pass

    def water_freight_regression(self, ):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        intensity = 
        X = math.log(intensity)
        Y = years
        reg = linear_model.LinearRegression()
        reg.fit(X, Y)
        coefficients = reg.coef_
        predicted_value = reg.predict(X_test)  # Predicted value of the intensity based on actual degree days

    def detailed_data_school_buses(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """


        pass

    def detailed_data_intercity_buses(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        passenger_miles =
        revised_passenger_miles =
        number_of_buses = 
        number_of_buses_old =  # Not used
        vehicle_miles_commercial =  # Not used
        vehicle_miles_intercity = 
        implied_load_factor = 
        energy_use_tedb_19_32 =  # Not used
        implied_mpg =  # Not used
        blended_mpg_miles_per_gallon =  #
        revised_energy_use = 

        number_of_motorcoaches = 
        number_of_motorcoaches['Ratio'] = number_of_motorcoaches['US'] / number_of_motorcoaches['Total N.A.']
        pass

    def mpg_check(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def energy_consumption(self):
        """Gather Energy Data to use in LMDI Calculation TBtu
        """        
        pass

    def activity():
        """Gather Activity Data to use in LMDI Calculation passenger-miles [P-M], ton-miles [T-M]
        """        
        pass
    
    def transportation_lmdi(self, _base_year=None):
        """potentially refactor later
    

        Args:
            _base_year ([type], optional): [description]. Defaults to None.
        """        
        if not base_year: 
            _base_year = self.base_year
        else: 
            _base_year = _base_year

        # Pipelines
        pipeline_cats = sub_categories_list['All_Freight']['Pipeline'].keys()
        pipelines_lmdi  = call_lmdi(pipeline_cats)

        # Freight-Trucks
        freight_truck_cats = sub_categories_list['All_Freight']['Highway']['Freight-Trucks'].keys()
        freight_trucks_lmdi = call_lmdi(freight_truck_cats)

        # Freight_Total
        freight_total_cats = sub_categories_list['All_Freight'].keys()
        freight_total_lmdi = call_lmdi(freight_total_cats)

        # Urban_Rail
        urban_rail_cats = sub_categories_list['All_Passenger']['Rail']['Urban Rail'].keys()
        urban_rail_lmdi = call_lmdi(urban_rail_cats)

        # Passenger Rail
        passenger_rail_cats = sub_categories_list['All_Passenger']['Rail'].keys()
        passenger_rail_lmdi = call_lmdi(passenger_rail_cats)

        # Passenger Air
        sub_categories_list['All_Passenger']['Air'].keys()

        # Buses
        sub_categories_list['All_Passenger']['Highway']['Buses'].keys()

        # Trucks and LWB
        sub_categories_list['All_Passenger']['Highway']['Light Trucks – LWB Vehicles'].keys()

        # Cars and SWB Vehicles
        sub_categories_list['All_Passenger']['Highway']['Passenger Car – SWB Vehicles'].keys()

        # Personal Passenger Vehicles
        sub_categories_list['All_Passenger']['Highway']['Passenger Cars and Trucks'].keys()

        # Passenger-Highway
        sub_categories_list['All_Passenger']['Highway'].keys()

        # Passenger_Total
        sub_categories_list['All_Passenger'].keys()

        # Personal vehicles - aggregate
        personal_vehicles_aggregate_cats = ['Passenger Car', 'Light Truck', 'Motorcycles']
        personal_vehicles_aggregate_lmdi = call_lmdi(personal_vehicles_aggregate_cats)

        # Total_Transportation
        sub_categories_list.keys()
        pass
        

    def compare_aggregates(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        total_fuel_tbtu_published_mer = self.mer_table_25_dec_2019['Total Energy Consumed by the Transportation Sector']  # j
        sum_of_modes = self.total_transportation['Energy_Consumption_Total']  # F
        difference = sum_of_modes - total_fuel_tbtu_published_mer
        pct_difference = difference / total_fuel_tbtu_published_mer
        return pct_difference




