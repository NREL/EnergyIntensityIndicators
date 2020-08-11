import pandas as pd
from sklearn import linear_model
import zipfile
# from outline import LMDI

class TranportationData:

    def __init__(self):
        self.transportation_data = {'Passenger Car – SWB Vehicles': {'total_fuel': 
                                                                        {'unit': 'gallons', 'source': 'TEDB', 'table_number': '4_01'} # Table4_01_{date}
                                                                 'total_energy': 
                                                                        {'unit': 'tbtu', 'source': 'TEDB', 'table_number': 'A_18'}
                                                                 'vehicle_miles': 
                                                                        {'unit': 'miles', 'source': 'TEDB', 'table_number': '4_01'}
                                                                 'passenger_miles': 
                                                                        {'unit': 'miles', 'source': 'TEDB', 'table_number': 'A_18'}},
                               'Light Trucks – LWB Vehicles': {'total_fuel': 
                                                                        {'unit': 'gallons', 'source': 'TEDB', 'table_number': '4_02'}
                                                                 'total_energy': 
                                                                        {'unit': 'tbtu', 'source': 'TEDB', 'table_number': 'A_05'}
                                                                 'vehicle_miles': 
                                                                        {'unit': 'miles', 'source': 'TEDB', 'table_number': '4_02'}
                                                                 'passenger_miles':
                                                                        {'unit': 'miles', 'source': 'TEDB', 'table_number': 'A_19'}},
                                'Motorcycles': {'total_fuel': 
                                                        {'unit': 'gallons', 'source': 'TEDB', 'table_number': 'A_02'}
                                                    'total_energy': 
                                                        {'unit': 'tbtu', 'source': '', 'table_number': '', 'Assumptions': 'Assume all motorcycle fuel is gasoline'}
                                                    'vehicle_miles': 
                                                        {'unit': 'miles', 'source': 'BTS', 'table_number': '1-35'}  #  with some interpolation prior to 1990'
                                                    'passenger_miles':
                                                        {'unit': 'miles', 'source': '', 'table_number': '', 'Assumptions': vehicle_miles * 1.1}},
                                'Urban Bus': {'total_fuel': 
                                                    {'unit': 'gallons', 'source': 'APTA Public Transportation Fact Book'}
                                            'total_energy': 
                                                {'unit': '', 'source': '', 'table_number': '', 'Assumption': 'Apply conversion factors (Btu/gallon)'}
                                            'passenger_miles':
                                                {'unit': 'miles', 'source': 'APTA'}},
                                'Intercity Bus': {'total_fuel': 
                                                    {'unit': 'gallons', 'source': 'Earlier data from TEDB-19 later from ABA'}
                                                 'total_energy': 
                                                    {'unit': 'tbtu', 'source': '', 'Assumptions': 'All fuel assumed to be diesel'}
                                                 'passenger_miles':
                                                    {'unit': 'miles', 'source': 'Multiple'}},
                                'School Bus': {'total_fuel': 
                                                    {'unit': 'gallons', 'source': 'TEDB', 'table_number': 'Table A.4'}
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
                                                                            {'unit': 'tbtu', 'source': '', 'table_number': '', 'Assumptions': 'Conversion to TBtu as for commuter rail'}
                                                                        'passenger_miles':
                                                                            {'unit': 'miles', 'source': 'APTA', 'table_number': 'source as avober for fuel use. Passenger-miles in Table 3 of Appendix'}},
                                'Intercity Rail': {'total_fuel': 
                                                        {'unit': 'kwh_gallons', 'source': '1994-2016" TEDB-37 Table A.16; prior data extrapolated from TEDB-30, Table 9.10'}
                                                   'total_energy': 
                                                        {'unit': 'tbtu', 'source': 'Conversion to TBtu as for commuter rail'}
                                                   'passenger_miles':
                                                        {'unit': 'miles', 'source': 'APTA source as avober for fuel use. Passenger-miles in Table 3 of Appendix'}},
                                'Commercial Carriers': {'total_fuel': 
                                                            {'unit': 'gallons', 'source': 'TEDB-37, Table A.9'}, 
                                                        'total_energy':
                                                            {'unit': 'tbtu', 'source': 'Conversion to Btu with factors for aviation fuel and jet fuel (120,2 and  135.0 kBtu/gallon, respectively.)'},
                                                        'passenger_miles':
                                                            {'unit': 'miles', 'source': '1970-2001: Eno Transportation Foundation, Transportation in America 2001, 19th Edition, p.45 Passenger-miles after 2001 extrapolated by total energy use'}},
                                'General Aviation': {'total_fuel': 
                                                            {'unit': 'gallons', 'source': 'TEDB-37, Table A.10'}, 
                                                     'total_energy':
                                                            {'unit': 'tbtu', 'source': 'Conversion to Btu with factors for aviation fuel and jet fuel (120,2 and  135.0 kBtu/gallon, respectively.)'},
                                                     'passenger_miles':
                                                            {'unit': 'miles', 'source': ' 1970-2001: Eno Transportation Foundation, Transportation in America 2001, 19th Edition, p.45 Passenger-miles after 2001 extrapolated by total energy use.'}},
                           //////     'Single-Unit Truck': {'total_fuel': 
                                                            {'unit': 'gallons', 'source': 'TEDB-37, Table A.9'}, 
                                                     'total_energy':
                                                            {'unit': 'tbtu', 'source': 'Conversion to Btu with factors for aviation fuel and jet fuel (120,2 and  135.0 kBtu/gallon, respectively.)'},
                                                     'vehicle_miles':
                                                            {'unit': 'miles', 'source': ' 1970-2001: Eno Transportation Foundation, Transportation in America 2001, 19th Edition, p.45 Passenger-miles after 2001 extrapolated by total energy use.'},
                                                      'ton_miles': {'unit': , 'source': }},
                                'Combination Truck': {'total_fuel': 
                                                            {'unit': 'gallons', 'source': 'TEDB-37, Table A.10'}, 
                                                     'total_energy':
                                                            {'unit': 'tbtu', 'source': 'Conversion to Btu with factors for aviation fuel and jet fuel (120,2 and  135.0 kBtu/gallon, respectively.)'},
                                                     'vehicle_miles':
                                                            {'unit': 'miles', 'source': ' 1970-2001: Eno Transportation Foundation, Transportation in America 2001, 19th Edition, p.45 Passenger-miles after 2001 extrapolated by total energy use.'}}  ,                      }

    @staticmethod
    def import_tedb_data(sub_mode_dict):
        date = '04302020'
        file_url = f'https://tedb.ornl.gov/wp-content/uploads/2020/04/Table{sub_mode_dict['table_number']}_{date}.xlsx'
        xls = pd.read_excel(file_url)  # , sheetname=None, header=11
        return xls


    tedbpath = 'c:\irabidea\Downloads\TEDB_38.1_Spreadsheets_06242020\'
    
    r = requests.get(tedb_url)
    with ZipFile.open(r, mode=r) as zip_ref:
        print(zip_ref)
        # call import_tedb_data


    # Dictionary with table_name: dataframe
    def load_data(self)
        tedb_dict = dict()
        for mode_dict in self.transportation_data[transportation_mode].keys():  # could be more efficient with pandas?
            mode_data = dict()
            for sub_mode_dict in mode_dict.keys:
                if sub_mode_dict['source'] == 'TEDB':
                    mode_source_df = import_tedb_data(sub_mode_dict)
                    mode_data[sub_mode_dict]

        pass
    
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



    def fuel_consump():
        """Time series of fuel consumption for the major transportation subsectors. Data are generally in 
        millions gallons or barrels of petroleum.
        """        
        pass

    def fuel_heat_content():
        """Assumed Btu content of the various types of petroleum products
        """        
        pass

    def passenger_based_energy_use():
        """Calculate the fuel consumption in Btu for passenger transportation
        """        
        pass

    def freight_based_energy_use():
        """Calculate the fuel consumption in Btu for freight transportation
        """        
        pass

    def passenger_based_activity(data): 
        """Time series for the activity measures for passenger transportation
        """        
        pass

    def freight_based_activity(data):
        """Time series for the activity measures for passenger transportation

        Args:
            data ([type]): [description]
        """        

class TransportationIndicators(LMDI):

    def __init__(self, energy_data, activity_data, categories_list):
        super().__init__(energy_data, activity_data, categories_list)
        self.sub_categories_list = categories_list['transportation']

    def load_data(self, parameter_list):
        mer_table25_dec_2019 = GetEIAData.eia_api(id_='711272') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711272'
        mer_table_43_old = GetEIAData.eia_api(id_='711272') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711272'
        mer_table_4.3_nov2019 = GetEIAData.eia_api(id_='711272') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711272'
        aer_2010_table_65 = GetEIAData.eia_api(id_='711272') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711272'

        pass

    def water_freight_regression(self, ):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        X =
        Y = 
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
        pass

    def fuel_heat_content(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def fuel_consump(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def adjust_truck_freight(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def freight_based_energy_use(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def freight_based_activity(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def passenger_based_energy_use(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def passenger_based_activity(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def compare_aggregates(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def mpg_check(self, parameter_list):
        """purpose
           Parameters
           ----------
           
           Returns
           -------
           
        """
        pass

    def energy_consumption():
        """TBtu
        """        
        pass

    def activity():
        """passenger-miles [P-M], ton-miles [T-M]
        """        
        pass




