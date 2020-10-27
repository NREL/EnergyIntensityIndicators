import pandas as pd
from sklearn import linear_model
from functools import reduce
import requests 
import win32com.client as win32

from EnergyIntensityIndicators.LMDI import CalculateLMDI
from EnergyIntensityIndicators.pull_eia_api import GetEIAData


class ElectricityIndicators(CalculateLMDI):

    def __init__(self, directory, output_directory, level_of_aggregation, lmdi_model=['multiplicative'], base_year=1985):
        self.sub_categories_list = {'Elec Generation Total': 
                                        {'Elec Power Sector': 
                                            {'Electricity Only':
                                                {'Fossil Fuels': 
                                                    {'Coal': None, 'Petroleum': None, 'Natural Gas': None, 'Other Gasses': None},
                                                'Nuclear': None, 
                                                'Hydro Electric': None, 
                                                'Renewable':
                                                    {'Wood': None, 'Waste': None, 'Geothermal': None, 'Solar': None, 'Wind': None}},
                                            'Combined Heat & Power': 
                                                {'Fossil Fuels':
                                                    {'Coal': None, 'Petroleum': None, 'Natural Gas': None, 'Other Gasses': None},
                                                'Renewable':
                                                    {'Wood': None, 'Waste': None}}}, 
                                        'Commercial Sector': None, 
                                        'Industrial Sector': None},
                                    'All CHP':
                                        {'Elec Power Sector': 
                                            {'Combined Heat & Power':
                                                {'Fossil Fuels':
                                                    {'Coal': None, 'Petroleum': None, 'Natural Gas': None, 'Other Gasses': None},
                                                'Renewable':
                                                    {'Wood': None, 'Waste': None},
                                                'Other': None}},
                                        'Commercial Sector':
                                            {'Combined Heat & Power':
                                                {'Fossil Fuels':
                                                    {'Coal': None, 'Petroleum': None, 'Natural Gas': None, 'Other Gasses': None},
                                                'Hydroelectric': None,
                                                'Renewable':
                                                    {'Wood': None, 'Waste': None},
                                                'Other': None}}, 
                                        'Industrial Sector':
                                            {'Combined Heat & Power':
                                                {'Fossil Fuels':
                                                    {'Coal': None, 'Petroleum': None, 'Natural Gas': None, 'Other Gasses': None},
                                                'Hydroelectric': None,
                                                'Renewable':
                                                    {'Wood': None, 'Waste': None},
                                                'Other': None}}}}
        self.elec_power_eia = GetEIAData(sector='electricity')
        # self.Table21f = pd.read_excel('https://www.eia.gov/totalenergy/data/browser/xls.php?tbl=T02.06') #self.elec_power_eia.eia_api(id_='711254') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711254'
        # self.Table82a = self.elec_power_eia.eia_api(id_='3') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=3'
        # self.Table82b = self.elec_power_eia.eia_api(id_='21')  # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=21'
        # self.Table82c = self.elec_power_eia.eia_api(id_='1736765') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=1736765' ?
        # self.Table82d = self.elec_power_eia.eia_api(id_='711282') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711282'
        # # self.Table82d_2012_and_later = self.elec_power_eia.eia_api(id_='1017') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=1017'
        # self.Table83d_03 = self.elec_power_eia.eia_api(id_='711284') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711284' ?
        # self.Table84b = self.elec_power_eia.eia_api(id_='711284') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711284' ?
        # self.Table85c = self.elec_power_eia.eia_api(id_='379')  # ?
        # self.Table86b = self.elec_power_eia.eia_api(id_='463') # ?
        # self.MER_T72b_1013_AnnualData = self.elec_power_eia.eia_api(id_='21')  #'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=21'
        # self.MER_T72c_1013_AnnualData = self.elec_power_eia.eia_api(id_='2') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=2'
        self.energy_types = ['primary']
        self.Table84c_url = r"https://www.eia.gov/totalenergy/data/annual/xls/stb0804c.xls" # elec_power_eia.eia_api(id_='456') 
        self.Table82d_url = r"https://www.eia.gov/totalenergy/data/browser/xls.php?tbl=T07.02C"
        self.Table85c_url = r'https://www.eia.gov/totalenergy/data/annual/xls/stb0805c.xls'
        self.Table82c_url = r'https://www.eia.gov/totalenergy/data/annual/xls/stb0802c.xls'
        super().__init__(sector='electric', level_of_aggregation=level_of_aggregation, lmdi_models=lmdi_model, categories_dict=self.sub_categories_list, \
                         energy_types=self.energy_types, directory=directory, output_directory=output_directory, base_year=base_year)
    @staticmethod
    def get_eia_aer():
        """Prior to 2012, the data for the indicators were taken directly from tables published (and downloaded in 
        Excel format) from EIA's Annual Energy Review. 
        """  
        pass      
    
    @staticmethod
    def get_reconciles():
        """ The EIA Annual Energy Review data (pre 2012) for energy consumption to produce
        electricity were generally supplied in physical units only (e.g., mcf of natural gas, tons of coal, etc.) The
        values needed to be converted to Btu, and still be consistent with aggregate energy consumption for
        this sector as published by EIA. For each major fossil fuel, a separate worksheet was developed; these
        worksheets are identified with the suffix “reconcile.” Thus, the worksheet “NatGas Reconcile” seeks to
        produce an estimate of the Btu consumption of natural gas used to generate electricity. Similar
        worksheets were developed for coal, petroleum, and other fuels. 

        ///   Does this need to be done or can download Btu directly instead of converting physical units to Btu --> no     
        """        
        pass


    def process_utility_level_data(self):
        """The indicators for electricity are derived entirely from data collected by EIA. Since 2012, the indicators
        are based entirely upon te EIA-923 survey
        """        
        eia_923_schedules = pd.read_excel('./')

        net_generation = pd.pivot_table(eia_923_schedules, columns='EIA Sector Number', index='AERFuel Type Code', aggfunc='sum')  # page A-71,
                                                                                                    # 'Net Generation' lower right-hand quadrant?
        net_generation['Grand_Total'] = net_generation.sum(axis=1, skipna=True) # Should have 18 rows labeled by type of fuel and seven columns 
                                                                                    # plus one for 'Grand Total'. Note: rows is not an arg of pd.pivot_table
        elec_btu_consumption = pd.pivot_table(eia_923_schedules, columns='EIA Sector Number', index='AERFuel Type Code', aggfunc='sum')  # page A-71,
                                                                                                    # 'Elec Fuel ConsumptionMMBTU' lower right-hand quadrant?
        elec_btu_consumption['Grand_Total'] = elec_btu_consumption.sum(axis=1, skipna=True) # Should have 18 rows labeled by type of fuel and seven columns 
                                                                                    # plus one for 'Grand Total'
        previous_years_net_gen = pd.read_excel('./')
        previous_yeas_elec_btu_consumption = pd.read_excel('./')
        master_net_gen = previous_years_net_gen.concat(net_generation)
        maseter_elec_btu_consumption = previous_yeas_elec_btu_consumption.concat(elec_btu_consumption)
        # Aggregate data?? page A-72 fpr net generation and elec btu consumption
        return None

    @staticmethod
    def reconcile(total, elec_gen, elec_only_plants, chp_elec, assumed_conv_factor, chp_heat):
        """total: df, Btu
        elec_gen: df, Btu 
        elec_only_plants: df, Short tons
        chp_elec: df, Short tons
        assumed_conv_factor: float, MMBtu/Ton
        chp_heat  
        """
        difference = total.subtract(elec_gen) # Btu
        implied_conversion_factor = total.divide(elec_only_plants).multiply(1000)  # MMBtu/Ton
        elec_only_billionbtu = elec_only_plants.multiply(assumed_conv_factor).multiply(1000) # Billion Btu
        chp_elec_billionbtu = chp_elec.multiply(assumed_conv_factor).multiply(0.001) # Billion Btu 
        chp_heat_billionbtu = chp_heat.multiply(assumed_conv_factor).multiply(0.001) # Billion Btu
        total_fuel = elec_only_billionbtu.add(chp_elec_billionbtu).add(chp_heat_billionbtu)

        # Cross Check
        total_short_tons = elec_only_plants + chp_elec + chp_heat # Short Tons
        implied_conversion_factor_cross = total.divide(total_short_tons).multiply(1000)  # MMBtu/Ton
        implied_conversion_factor_revised = elec_gen.divide(chp_elec.add(elec_only_plants)).multiply(1000) # MMBtu/Ton

        chp_plants_fuel = implied_conversion_factor_revised.multiply(chp_elec).multiply(0.000001)  # Trillion Btu
        elec_only_fuel =  elec_gen.multiply(.001).subtract(chp_plants_fuel)  # Trillion Btu
        resulting_total = chp_plants_fuel.add(elec_only_fuel)
        return chp_plants_fuel, elec_only_fuel

    def coal_reconcile(self):
        energy_consumption_coal = self.elec_power_eia.eia_api(id_='TOTAL.CLEIBUS.A', id_type='series')# Table21f11 column b
        consumption_for_electricity_generation_coal = self.elec_power_eia.eia_api(id_='TOTAL.CLEIBUS.A', id_type='series')# Table84b11 column b

        consumption_combustible_fuels_electricity_generation_coal = self.elec_power_eia.eia_api(id_='TOTAL.CLL1PUS.A', id_type='series')# Table85c11 column B SHOULD BE separated Elec-only/CHP

        consumption_combustible_fuels_useful_thermal_output_coal = self.elec_power_eia.eia_api(id_='TOTAL.CLEIPUS.A', id_type='series')# Table86b11 column B

        assumed_conversion_factor = 20.9
        total = energy_consumption_coal
        elec_gen = consumption_for_electricity_generation_coal
        elec_only_plants = consumption_combustible_fuels_electricity_generation_coal # should be separate part?  
        chp_elec= consumption_combustible_fuels_electricity_generation_coal # should be separate part?  
        assumed_conv_factor = assumed_conversion_factor
        chp_heat = consumption_combustible_fuels_useful_thermal_output_coal    
        # eia-923 pivot table 

        difference = total.subtract(elec_gen) # Btu
        implied_conversion_factor = total.divide(elec_only_plants).multiply(1000)  # MMBtu/Ton
        elec_only_billionbtu = elec_only_plants.multiply(assumed_conv_factor).multiply(1000) # Billion Btu
        chp_elec_billionbtu = chp_elec.multiply(assumed_conv_factor).multiply(0.001) # Billion Btu 
        chp_heat_billionbtu = chp_heat.multiply(assumed_conv_factor).multiply(0.001) # Billion Btu
        total_fuel = elec_only_billionbtu.add(chp_elec_billionbtu).add(chp_heat_billionbtu)

        # Cross Check
        total_short_tons = elec_only_plants + chp_elec + chp_heat # Short Tons
        implied_conversion_factor_cross = total.divide(total_short_tons).multiply(1000)  # MMBtu/Ton
        implied_conversion_factor_revised = elec_gen.divide(chp_elec.add(elec_only_plants)).multiply(1000) # MMBtu/Ton

        chp_plants_fuel = implied_conversion_factor_revised.multiply(chp_elec).multiply(0.000001)  # Trillion Btu
        elec_only_fuel =  elec_gen.multiply(.001).subtract(chp_plants_fuel)  # Trillion Btu
        resulting_total = chp_plants_fuel.add(elec_only_fuel)
        return chp_plants_fuel, elec_only_fuel

    def natgas_reconcile(self):
        energy_consumption_natgas = self.elec_power_eia.eia_api(id_='TOTAL.NNEIBUS.A', id_type='series')# Table21f11 column d
        consumption_for_electricity_generation_natgas = self.elec_power_eia.eia_api(id_='TOTAL.NNEIBUS.A', id_type='series')# Table84b11 column f
        consumption_combustible_fuels_electricity_generation_natgas= self.elec_power_eia.eia_api(id_='TOTAL.NGL1PUS.A', id_type='series')# Table85c11 column N
        consumption_combustible_fuels_useful_thermal_output_natgas = self.elec_power_eia.eia_api(id_='TOTAL.NGEIPUS.A', id_type='series')# Table86b11 column M

        
        # eia-923 pivot table
        """total: df, Billion Btu
        elec_gen: df, Billion Btu 
        elec_only_plants: df, Thou. Cu. Ft.
        chp_elec: df, Thou. CF
        assumed_conv_factor: float, MMBtu/Ton
        chp_heat: df, Thou. CF
        """
        total = energy_consumption_natgas
        elec_gen = consumption_for_electricity_generation_natgas
        elec_only_plants = consumption_combustible_fuels_electricity_generation_natgas
        chp_elec = consumption_combustible_fuels_electricity_generation_natgas # different part
        assumed_conv_factor = 1.028
        chp_heat = consumption_combustible_fuels_useful_thermal_output_natgas


        difference = total.subtract(elec_gen) # Billion Btu
        implied_conversion_factor = elec_gen.divide(elec_only_plants).multiply(1000)  # kBtu/CF  * different from coal_reconcile
        elec_only_trillionbtu = elec_only_plants.multiply(assumed_conv_factor).multiply(0.001) # Trillion Btu
        chp_elec_trillionbtu = chp_elec.multiply(assumed_conv_factor).multiply(0.001) # Trillion Btu 
        chp_heat_trillionbtu = chp_heat.multiply(assumed_conv_factor).multiply(0.001) # Trillion Btu
        total_fuel = elec_only_trillionbtu.add(chp_elec_trillionbtu).add(chp_heat_trillionbtu)

        # Cross Check
        total_thou_cf = elec_only_plants + chp_elec + chp_heat # Thou. CF
        implied_conversion_factor_cross = total.divide(total_thou_cf).multiply(1000)  # kBtu/CF
        implied_conversion_factor_revised = elec_gen.divide(chp_elec.add(elec_only_plants)).multiply(1000) # MMBtu/Ton

        chp_plants_fuel = implied_conversion_factor_revised.multiply(chp_elec).multiply(0.000001)  # Trillion Btu
        elec_only_fuel =  elec_gen.multiply(.001).subtract(chp_plants_fuel)  # Trillion Btu
        resulting_total = chp_plants_fuel.add(elec_only_fuel)
        return chp_plants_fuel, elec_only_fuel


    def petroleum_reconcile(self):
        energy_consumption_petroleum = self.elec_power_eia.eia_api(id_='TOTAL.PAEIBUS.A', id_type='series')# Table21f11 column F
        consumption_for_electricity_generation_petroleum = self.elec_power_eia.eia_api(id_='TOTAL.PAEIBUS.A', id_type='series')# Table84b11 column D
        consumption_combustible_fuels_electricity_generation_distillate_fuel_oil = self.elec_power_eia.eia_api(id_='TOTAL.DKL1PUS.A', id_type='series')# Table85c11 column D, F, H, J
        consumption_combustible_fuels_electricity_generation_residual_fuel_oil = self.elec_power_eia.eia_api(id_='TOTAL.RFL1PUS.A', id_type='series')
        consumption_combustible_fuels_electricity_generation_other_liquids = self.elec_power_eia.eia_api(id_='TOTAL.OLL1PUS.A', id_type='series')
        consumption_combustible_fuels_electricity_generation_petroleum_coke = self.elec_power_eia.eia_api(id_='TOTAL.PCL1MUS.A', id_type='series')
        consumption_combustible_fuels_electricity_generation_total_petroleum = self.elec_power_eia.eia_api(id_='TOTAL.PAL1PUS.A', id_type='series')

        fuels_list = [consumption_combustible_fuels_electricity_generation_distillate_fuel_oil, consumption_combustible_fuels_electricity_generation_residual_fuel_oil,
                    consumption_combustible_fuels_electricity_generation_other_liquids, consumption_combustible_fuels_electricity_generation_petroleum_coke]
        elec_only_plants_petroleum = reduce(lambda x, y: pd.merge(x, y, on ='Period'), fuels_list)
        
        consumption_combustible_fuels_useful_thermal_output_distillate_fuel_oil = self.elec_power_eia.eia_api(id_='TOTAL.DKEIPUS.A', id_type='series')# Table86b11 column D, F, G, I
        consumption_combustible_fuels_useful_thermal_output_residual_fuel_oil = self.elec_power_eia.eia_api(id_='TOTAL.RFEIPUS.A', id_type='series')
        consumption_combustible_fuels_useful_thermal_output_other_liquids = self.elec_power_eia.eia_api(id_='TOTAL.OLEIPUS.A', id_type='series')
        consumption_combustible_fuels_useful_thermal_output_petroleum_coke = self.elec_power_eia.eia_api(id_='TOTAL.PCEIMUS.A', id_type='series')
        consumption_combustible_fuels_useful_thermal_output_total_petroleum = self.elec_power_eia.eia_api(id_='TOTAL.PTEIPUS.A', id_type='series')
        chp_elec = consumption_combustible_fuels_electricity_generation_total_petroleum
        # eia-923 pivot table
        assumed_conversion_factors = [5.825, 6.287, 5.67, 30120]
        chp_plants_petroleum, elec_only_petroleum = self.reconcile(total=energy_consumption_petroleum, elec_gen=consumption_for_electricity_generation_petroleum, 
                                                            elec_only_plants=elec_only_plants_petroleum, chp_elec=chp_elec, assumed_conv_factor=assumed_conversion_factors,
                                                            chp_heat=consumption_combustible_fuels_useful_thermal_output_total_petroleum)
        return chp_plants_petroleum, elec_only_petroleum

    def othgas_reconcile(self):
        consumption_for_electricity_generation_fossil_fuels = self.elec_power_eia.eia_api(id_='TOTAL.FFEIBUS.A', id_type='series')# Table84b11 column H
        consumption_for_electricity_generation_oth_gas = consumption_for_electricity_generation_fossil_fuels - consumption_for_electricity_generation_petroleum - consumption_for_electricity_generation_natgas - consumption_for_electricity_generation_coal
        consumption_combustible_fuels_electricity_generation_oth_gas = self.elec_power_eia.eia_api(id_='TOTAL.OJL1BUS.A', id_type='series')# Table85c11 column P
        consumption_combustible_fuels_useful_thermal_output_othgas = self.elec_power_eia.eia_api(id_='TOTAL.OJEIBUS.A', id_type='series')# Table86b11 column O

        elec_gen = consumption_for_electricity_generation_oth_gas
        elec_only_plants = consumption_combustible_fuels_electricity_generation_oth_gas 
        chp_elec = consumption_combustible_fuels_electricity_generation_oth_gas # ** different part of the series?'
        chp_heat = consumption_combustible_fuels_useful_thermal_output_othgas
        total_other_gas = elec_only_plants.add(chp_elec).add(chp_heat)
        return chp_elec

    # consumption_combustible_fuels_electricity_generation = 
    # print(consumption_combustible_fuels_electricity_generation)
    # consumption_combustible_fuels_useful_thermal_output = 
    # print(consumption_combustible_fuels_useful_thermal_output)

    def industrial_sector_chp_renew(self):
        # excel = win32.gencache.EnsureDispatch('Excel.Application') #, encoding='cp1252') # # Table 8.4C column P # TBtu
        # excel.Visible = True
        # wb = excel.Workbooks.Open(self.Table84c_url)
        wb = pd.read_excel(self.Table84c_url)
        print(wb)
        exit()
        # wb.SaveAs('C:/Users/irabidea/Desktop/LMDI_Results/Table84c_2.xls'+"x", FileFormat = 51)    #FileFormat = 51 is for .xlsx extension
        # wb.Close()                               #FileFormat = 56 is for .xls extension
        # excel.Application.Quit()
        wood_energy = pd.read_excel('C:/Users/irabidea/Desktop/Table84c.xls'+"x", index_col=0, usecols='P', skiprows=32, skipfooter=55).multiply(0.001)
        print(wood_energy)
        exit()
        waste_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='R', skiprows=32, skipfooter=55).multiply(0.001) # Table 8.4C column R # TBtu

        wood_activity = self.elec_power_eia.eia_api(id_='TOTAL.WDI5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column R
        waste_activity = self.elec_power_eia.eia_api(id_='TOTAL.WSI5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column T

        data_dict = {'Wood': {'energy': {'primary': wood_energy}, 'activity': wood_activity}, 'Waste': {'energy': {'primary': waste_energy}, 'activity': waste_activity}}
        return data_dict

    def industrial_sector_chp_fossil(self):
        coal_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='B', skiprows=32, skipfooter=55).multiply(0.001) # Table 8.4c column B # TBtu
        petroleum_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='D', skiprows=32, skipfooter=55).multiply(0.001) # Table 8.4c column D # TBtu
        natgas_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='F', skiprows=32, skipfooter=55).multiply(0.001) # Table 8.4c column F # TBtu
        othgas_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='H', skiprows=32, skipfooter=55).multiply(0.001) # Table 8.4c column H # TBtu

        coal_activity = self.elec_power_eia.eia_api(id_='TOTAL.CLI5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column B
        petroleum_activity = self.elec_power_eia.eia_api(id_='TOTAL.PAI5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column D
        natgas_activity = self.elec_power_eia.eia_api(id_='TOTAL.NGI5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column F
        othgas_activity = self.elec_power_eia.eia_api(id_='TOTAL.OJI5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column H
        
        data_dict = {'Coal': {'energy': {'primary': coal_energy}, 'activity': coal_activity}, 
                     'Petroleum': {'energy': {'primary': petroleum_energy}, 'activity': petroleum_activity}, 
                     'Natural Gas': {'energy': {'primary': natgas_energy}, 'activity': natgas_activity}, 
                     'Other Gasses': {'energy': {'primary': othgas_energy}, 'activity': othgas_activity}}
        return data_dict

    def industrial_sector_total(self):
        """Note: Other includes batteries, chemicals, hydrogen, pitch, purchased steam, sulfur, and miscellaneous technologies

        """    
        hydroelectric_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='N', skiprows=32, skipfooter=55).multiply(0.001) # Table 8.4C11 column N
        other_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='AB', skiprows=32, skipfooter=55).multiply(0.001) # Table 8.4C11 column AB

        hydroelectric_activity = self.elec_power_eia.eia_api(id_='TOTAL.HVI5PUS.A', id_type='series').multiply(0.001) # Table 8.2d11 Column P
        other_activity = None #  self.elec_power_eia.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d11 Column AD
        data_dict = {'Hydroelectric': {'energy': {'primary': hydroelectric_energy}, 'activity': hydroelectric_activity},
                     'Other': {'energy': {'primary': other_energy}, 'activity': other_activity}}
        return data_dict

    def comm_sector_chp_renew(self):
        """As is, these are the same sources as ind sector, but should be different part of columns? 
        """    
        wood_energy =  pd.read_excel(self.Table84c_url, index_col=0, usecols='P', skiprows=8, skipfooter=31).multiply(0.001) # Table 8.4C column P # TBtu
        waste_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='R', skiprows=8, skipfooter=31).multiply(0.001) # Table 8.4C column R # TBtu

        wood_activity = None # self.elec_power_eia.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d column R
        waste_activity = self.elec_power_eia.eia_api(id_='TOTAL.WSC5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column T
        data_dict = {'Wood': {'energy': {'primary': wood_energy}, 'activity': wood_activity}, 
                     'Waste': {'energy': {'primary': waste_energy}, 'activity': waste_activity}}
        return data_dict

    def comm_sector_chp_fossil(self):
        coal_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='B', skiprows=8, skipfooter=31).multiply(0.001) # Table 8.4c column B # TBtu
        petroleum_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='D', skiprows=8, skipfooter=31).multiply(0.001) # Table 8.4c column D # TBtu
        natgas_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='F', skiprows=8, skipfooter=31).multiply(0.001) # Table 8.4c column F # TBtu
        othgas_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='H', skiprows=8, skipfooter=31).multiply(0.001) # Table 8.4c column H # TBtu

        coal_activity = self.elec_power_eia.eia_api(id_='TOTAL.CLC5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column B
        petroleum_activity = self.elec_power_eia.eia_api(id_='TOTAL.PAC5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column D
        natgas_activity = self.elec_power_eia.eia_api(id_='TOTAL.NGC5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column F
        othgas_activity = None #  self.elec_power_eia.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d column H

        # Industrial and Commercial have same spreadsheet sources but different parts of the columns
        data_dict = {'Coal': {'energy': {'primary': coal_energy}, 'activity': coal_activity}, 
                     'Petroleum': {'energy': {'primary': petroleum_energy}, 'activity': petroleum_activity}, 
                     'Natural Gas': {'energy': {'primary': natgas_energy}, 'activity': natgas_activity}, 
                     'Other Gasses': {'energy': {'primary': othgas_energy}, 'activity': othgas_activity}}
        return data_dict

    def comm_sector_total(self):
        """Note: Other includes batteries, chemicals, hydrogen, pitch, purchased steam, sulfur, and miscellaneous technologies

        """    
        hydroelectric_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='N', skiprows=8, skipfooter=31).multiply(0.001) # Table 8.4C11 column N
        other_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='AB', skiprows=8, skipfooter=31).multiply(0.001) # Table 8.4C11 column AB

        hydroelectric_activity = None # self.elec_power_eia.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d11 Column P
        other_activity = pd.read_excel(self.Table82d_url, sheet_name='Annual Data', skiprows=9, header=10, index_col=0, usecols='AD').multiply(0.001) # Table 8.2d11 Column AD

        data_dict = {'Hydroelectric': {'energy': {'primary': hydroelectric_energy}, 'activity': hydroelectric_activity}, 
                     'Other': {'energy': {'primary': other_energy}, 'activity': other_activity}}
        return data_dict

    def elec_power_sector_chp_renew(self):
        wood_energy =  pd.read_excel(self.Table85c_url, index_col=0, usecols='N', skiprows=8, skipfooter=31).multiply(0.001) # Table 8.5C column R # TBtu
        waste_energy = pd.read_excel(self.Table85c_url, index_col=0, usecols='N', skiprows=8, skipfooter=31).multiply(0.001) # Table 8.5C column T # TBtu

        wood_activity = self.elec_power_eia.eia_api(id_='TOTAL.WDEGPUS.A', id_type='series').multiply(0.001) # Table 8.2c column R
        waste_activity = self.elec_power_eia.eia_api(id_='TOTAL.WSEGPUS.A', id_type='series').multiply(0.001) # Table 8.2c column T

        data_dict = {'Wood': {'energy': {'primary': wood_energy}, 'activity': wood_activity}, 
                     'Waste': {'energy': {'primary': waste_energy}, 'activity': waste_activity}}
        return data_dict

    def elec_power_sector_chp_fossil(self):
        coal_energy = self.coal_reconcile()
        petroleum_energy = self.petroleum_reconcile()
        natgas_energy = self.natgas_reconcile()
        othgas_energy = self.othgas_reconcile()
        
        coal_activity = self.elec_power_eia.eia_api(id_='TOTAL.CLEGPUS.A', id_type='series').multiply(0.001) # Table 8.2c column B
        petroleum_activity = self.elec_power_eia.eia_api(id_='TOTAL.PAEGPUS.A', id_type='series').multiply(0.001) # Table 8.2c column D
        natgas_activity = self.elec_power_eia.eia_api(id_='TOTAL.NGEGPUS.A', id_type='series').multiply(0.001) # Table 8.2c column F
        othgas_activity = self.elec_power_eia.eia_api(id_='TOTAL.OJEGPUS.A', id_type='series').multiply(0.001) # Table 8.2c column H

        data_dict = {'Coal': {'energy': {'primary': coal_energy}, 'activity': coal_activity}, 
                     'Petroleum': {'energy': {'primary': petroleum_energy}, 'activity': petroleum_activity}, 
                     'Natural Gas': {'energy': {'primary': natgas_energy}, 'activity': natgas_activity}, 
                     'Other Gasses': {'energy': {'primary': othgas_energy}, 'activity': othgas_activity}}
        return data_dict

    def elec_power_sector_chp_total(self):

        other_energy = pd.read_excel(self.Table85c_url, index_col=0, usecols='N', skiprows=8, skipfooter=31).multiply(0.001) # Table 8.5c column V
        other_activty = None # self.elec_power_eia.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2c columnAD

        data_dict = {'Other': {'energy': {'primary': other_energy}, 'activity': other_activty}}
        return data_dict

    def electricity_only_renew(self):
        wood_activity = self.elec_power_eia.eia_api(id_='TOTAL.WDL1BUS.A', id_type='series').multiply(0.001)  # Table 8.3d I, 8.5C R
        waste_activity = self.elec_power_eia.eia_api(id_='TOTAL.WSL1BUS.A', id_type='series').multiply(0.001) # Table 8.3d J, 8.5C T
        geothermal_activity = self.elec_power_eia.eia_api(id_='TOTAL.GEEGBUS.A', id_type='series').multiply(0.001) # Table 8.4 T
        solar_activity = self.elec_power_eia.eia_api(id_='TOTAL.SOEGBUS.A', id_type='series').multiply(0.001)  # Table 8.4 V
        wind_activity = self.elec_power_eia.eia_api(id_='TOTAL.WYEGBUS.A', id_type='series').multiply(0.001) # Table 8.4 X

        wood_energy = self.elec_power_eia.eia_api(id_='TOTAL.WDEGPUS.A', id_type='series').multiply(0.001)  # Table 8.2b S, 8.2c R
        waste_energy = self.elec_power_eia.eia_api(id_='TOTAL.WSEGPUS.A', id_type='series').multiply(0.001)  # Table 8.2b T, 8.2c T
        geothermal_energy = self.elec_power_eia.eia_api(id_='TOTAL.GEEGPUS.A', id_type='series').multiply(0.001)  # Table 8.2b V, 8.2c V
        solar_energy = self.elec_power_eia.eia_api(id_='TOTAL.SOEGPUS.A', id_type='series').multiply(0.001) # Table 8.2b X, 8.2c X
        wind_energy = self.elec_power_eia.eia_api(id_='TOTAL.WYEGPUS.A', id_type='series').multiply(0.001)  # Table 8.2b Z, 8.2c Z

        data_dict = {'Wood': {'energy': {'primary': wood_energy}, 'activity': wood_activity}, 
                     'Waste': {'energy': {'primary': waste_energy}, 'activity': waste_activity}, 
                     'Geothermal': {'energy': {'primary': geothermal_energy}, 'activity': geothermal_activity}, 
                     'Solar': {'energy': {'primary': solar_energy}, 'activity': solar_activity}, 
                     'Wind': {'energy': {'primary': wind_energy}, 'activity': wind_activity}}
        return data_dict

    def electricity_only_fossil(self):
        coal_energy = self.coal_reconcile()
        petroleum_energy = self.petroleum_reconcile()
        natgas_energy = self.natgas_reconcile()
        othgas_energy = self.othgas_reconcile()
        
        coal_activity = self.elec_power_eia.eia_api(id_='TOTAL.CLEGPUS.A', id_type='series').multiply(0.001)  # 8.2b B, 8.2c B
        petroleum_activity = self.elec_power_eia.eia_api(id_='TOTAL.PAEGPUS.A', id_type='series').multiply(0.001)  # 8.2b D, 8.2c D
        natgas_activity = self.elec_power_eia.eia_api(id_='TOTAL.NGEGPUS.A', id_type='series').multiply(0.001)  # 8.2b B, 8.2c F
        othgas_activity = self.elec_power_eia.eia_api(id_='TOTAL.OJEGPUS.A', id_type='series').multiply(0.001) # 8.2c H

        data_dict = {'Coal': {'energy': {'primary': coal_energy}, 'activity': coal_activity}, 
                     'Petroleum': {'energy': {'primary': petroleum_energy}, 'activity': petroleum_activity}, 
                     'Natural Gas': {'energy': {'primary': natgas_energy}, 'activity': natgas_activity}, 
                     'Other Gasses': {'energy': {'primary': othgas_energy}, 'activity': othgas_activity}}
        return data_dict

    def electricity_only_total(self):
        nuclear_energy = self.elec_power_eia.eia_api(id_='TOTAL.NUEGBUS.A', id_type='series').multiply(0.001) # 8.4b L
        hydroelectric_energy = self.elec_power_eia.eia_api(id_='TOTAL.HVEGBUS.A', id_type='series').multiply(0.001)  # 8.4b N

        nuclear_activity = self.elec_power_eia.eia_api(id_='TOTAL.NUEGPUS.A', id_type='series').multiply(0.001) # 8.2b L
        hydroelectric_activity = self.elec_power_eia.eia_api(id_='TOTAL.HVEGPUS.A', id_type='series').multiply(0.001)  # 8.2b O

        data_dict = {'Hydro Electric': {'energy': {'primary': hydroelectric_energy}, 'activity': hydroelectric_activity}, 
                     'Nuclear': {'energy': {'primary': nuclear_energy}, 'activity': nuclear_activity}}
        return data_dict

    def collect_data(self):
        industrial_sector_chp_renew = self.industrial_sector_chp_renew()
        industrial_sector_chp_fossil = self.industrial_sector_chp_fossil()
        industrial_sector_total = self.industrial_sector_total()
        industrial_sector_total['Fossil Fuels'] = industrial_sector_chp_fossil
        industrial_sector_total['Renewable'] = industrial_sector_chp_renew

        comm_sector_chp_renew = self.comm_sector_chp_renew()
        comm_sector_chp_fossil = self.comm_sector_chp_fossil()
        comm_sector_total = self.comm_sector_total()
        comm_sector_total['Fossil Fuels'] = comm_sector_chp_fossil
        comm_sector_total['Renewable'] = comm_sector_chp_renew

        elec_power_sector_chp_renew = self.elec_power_sector_chp_renew()
        elec_power_sector_chp_fossil = self.elec_power_sector_chp_fossil()
        elec_power_sector_chp_total = self.elec_power_sector_chp_total()
        elec_power_sector_chp_total['Fossil Fuels'] = elec_power_sector_chp_fossil
        elec_power_sector_chp_total['Renewable'] = elec_power_sector_chp_renew

        all_chp = dict()
        all_chp['Industrial Sector'] = industrial_sector_total
        all_chp['Commercial Sector'] = comm_sector_total
        all_chp['Elec Power Sector'] = elec_power_sector_chp_total

        electricity_only_renew = self.electricity_only_renew()
        electricity_only_fossil = self.electricity_only_fossil()
        electricity_only_total = self.electricity_only_total()
        electricity_only_total['Fossil Fuels'] = electricity_only_fossil
        electricity_only_total['Renewable'] = electricity_only_total

        chp_elec_power_sector = dict()
        chp_renew = self.chp_renew()
        chp_fossil = self.chp_fossil()
        chp_elec_power_sector['Fossil Fuels'] = chp_fossil
        chp_elec_power_sector['Renewable'] = chp_renew

        elec_power_sector = dict()
        elec_power_sector['Electricity Only']  = electricity_only_total
        elec_power_sector['Combined Heat & Power'] = chp_elec_power_sector

        elec_generation_total = dict() # Need to build this from commerical and industrial_totals
        elec_generation_total['Elec Power Sector'] = elec_power_sector
        elec_generation_total['Commercial Sector'] = comm_sector_total
        elec_generation_total['Industrial Sector'] = industrial_sector_total

        data_dict = {'Elec Generation Total': elec_generation_total, 'All CHP': all_chp}
        return data_dict   

    def main(self, breakout, save_breakout, calculate_lmdi): 
        data_dict = self.collect_data()
        print(data_dict)
        results_dict, formatted_results = self.get_nested_lmdi(level_of_aggregation=self.level_of_aggregation, breakout=breakout, save_breakout=save_breakout, calculate_lmdi=calculate_lmdi, raw_data=data_dict)


if __name__ == '__main__':
    indicators = ElectricityIndicators(directory='C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020', output_directory='C:/Users/irabidea/Desktop/LMDI_Results', level_of_aggregation='All CHP.Elec Power Sector')
    indicators.main(breakout=False, save_breakout=False, calculate_lmdi=False)