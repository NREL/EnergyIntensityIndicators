import pandas as pd
from sklearn import linear_model
from outline import LMDI


class ElectricityIndicators(LMDI):

    def __init__(self, energy_data, activity_data, categories_list):
        super().__init__(energy_data, activity_data, categories_list)
        self.sub_categories_list = categories_list['electricity']
        self.Table21f = pd.read_excel('https://www.eia.gov/totalenergy/data/browser/xls.php?tbl=T02.06') #GetEIAData.eia_api(id_='711254') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711254'
        self.Table82a = GetEIAData.eia_api(id_='3') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=3'
        self.Table82b = GetEIAData.eia_api(id_='21')  # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=21'
        self.Table82c = GetEIAData.eia_api(id_='1736765') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=1736765' ?
        self.Table82d = GetEIAData.eia_api(id_='711282') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711282'
        self.Table82d_2012_and_later = GetEIAData.eia_api(id_='1017') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=1017'
        self.Table83d_03 = GetEIAData.eia_api(id_='711284') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711284' ?
        self.Table84b = GetEIAData.eia_api(id_='711284') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711284' ?
        self.Table85c = GetEIAData.eia_api(id_='379')  # ?
        self.Table86b = GetEIAData.eia_api(id_='463') # ?
        self.MER_T72b_1013_AnnualData = GetEIAData.eia_api(id_='21')  #'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=21'
        self.MER_T72c_1013_AnnualData = GetEIAData.eia_api(id_='2') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=2'
        
        self.Table84c_url = 'https://www.eia.gov/totalenergy/data/annual/xls/stb0804c.xls' # GetEIAData.eia_api(id_='456') 
        self.Table82d_url = 'https://www.eia.gov/totalenergy/data/browser/xls.php?tbl=T07.02C'
        self.Table85c_url = 'https://www.eia.gov/totalenergy/data/annual/xls/stb0805c.xls'
        self.Table82c_url = 'https://www.eia.gov/totalenergy/data/annual/xls/stb0802c.xls'
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
        net_generation['Grand_Total'] = net_generation[[]].sum(axis=1, skipna=True) # Should have 18 rows labeled by type of fuel and seven columns 
                                                                                    # plus one for 'Grand Total'. Note: rows is not an arg of pd.pivot_table
        elec_btu_consumption = pd.pivot_table(eia_923_schedules, colums='EIA Sector Number', rows='AERFuel Type Code', aggfunc='sum')  # page A-71,
                                                                                                    # 'Elec Fuel ConsumptionMMBTU' lower right-hand quadrant?
        elec_btu_consumption['Grand_Total'] = elec_btu_consumption[[]].sum(axis=1, skipna=True) # Should have 18 rows labeled by type of fuel and seven columns 
                                                                                    # plus one for 'Grand Total'
        previous_years_net_gen = pd.read_excel('./')
        previous_yeas_elec_btu_consumption = pd.read_excel('./')
        master_net_gen = previous_years_net_gen.concat(net_generation)
        maseter_elec_btu_consumption = previous_yeas_elec_btu_consumption.concat(elec_btu_consumption)
        # Aggregate data?? page A-72 fpr net generation and elec btu consumption

    def energy_consumption(self):
        """Trillion Btu
        """      
        sources = {'wood': 'Table84c', 'waste': 'Table84c'} 'Table84c'
        'Elec_Power_Sector_CHP>Renewable': 'Table85c'
        'Elec_Power_Sector_CHP>Total-other': 'Table85c'
        return self.sub_categories_list

    def activity(self):
        """Million kWh
        """          
        sources = {'wood': 'Table82d', 'waste': 'Table82d'} 'Table82d'
                'Elec_Power_Sector_CHP>Renewable': 'Table82c'
                'Elec_Power_Sector_CHP>Fossil': 'Table82c'
                        'Elec_Power_Sector_CHP>Total-other': 'Table82c'


def reconcile(total, elec_gen, elec_only_plants, chp_elec, assumed_conv_factor, chp_heat)
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

def coal_reconcile():
    energy_consumption_coal = self.eia_elec.eia_api(id_='TOTAL.CLEIBUS.A', id_type='series')# Table21f11 column b
    consumption_for_electricity_generation_coal = self.eia_elec.eia_api(id_='TOTAL.CLEIBUS.A', id_type='series')# Table84b11 column b

    consumption_combustible_fuels_electricity_generation_coal = self.eia_elec.eia_api(id_='TOTAL.CLL1PUS.A', id_type='series')# Table85c11 column B SHOULD BE separated Elec-only/CHP

    consumption_combustible_fuels_useful_thermal_output_coal = self.eia_elec.eia_api(id_='TOTAL.CLEIPUS.A', id_type='series')# Table86b11 column B

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

def natgas_reconcile():
    energy_consumption_natgas = self.eia_elec.eia_api(id_='TOTAL.NNEIBUS.A', id_type='series')# Table21f11 column d
    consumption_for_electricity_generation_natgas = self.eia_elec.eia_api(id_='TOTAL.NNEIBUS.A', id_type='series')# Table84b11 column f
    consumption_combustible_fuels_electricity_generation_natgas= self.eia_elec.eia_api(id_='TOTAL.NGL1PUS.A', id_type='series')# Table85c11 column N
    consumption_combustible_fuels_useful_thermal_output_natgas = self.eia_elec.eia_api(id_='TOTAL.NGEIPUS.A', id_type='series')# Table86b11 column M

    
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


def petroleum_reconcile():
    energy_consumption_petroleum = self.eia_elec.eia_api(id_='TOTAL.PAEIBUS.A', id_type='series')# Table21f11 column F
    consumption_for_electricity_generation_petroleum = self.eia_elec.eia_api(id_='TOTAL.PAEIBUS.A', id_type='series')# Table84b11 column D
    consumption_combustible_fuels_electricity_generation_distillate_fuel_oil = self.eia_elec.eia_api(id_='TOTAL.DKL1PUS.A', id_type='series')# Table85c11 column D, F, H, J
    consumption_combustible_fuels_electricity_generation_residual_fuel_oil = self.eia_elec.eia_api(id_='TOTAL.RFL1PUS.A', id_type='series')
    consumption_combustible_fuels_electricity_generation_other_liquids = self.eia_elec.eia_api(id_='TOTAL.OLL1PUS.A', id_type='series')
    consumption_combustible_fuels_electricity_generation_petroleum_coke = self.eia_elec.eia_api(id_='TOTAL.PCL1MUS.A', id_type='series')
    consumption_combustible_fuels_electricity_generation_total_petroleum = self.eia_elec.eia_api(id_='TOTAL.PAL1PUS.A', id_type='series')

    fuels_list = [consumption_combustible_fuels_electricity_generation_distillate_fuel_oil, consumption_combustible_fuels_electricity_generation_residual_fuel_oil
                  consumption_combustible_fuels_electricity_generation_other_liquids, consumption_combustible_fuels_electricity_generation_petroleum_coke]
    elec_only_plants_petroleum = reduce(lambda x, y: pd.merge(x, y, on ='Period'), fuels_list)
    
    consumption_combustible_fuels_useful_thermal_output_distillate_fuel_oil = self.eia_elec.eia_api(id_='TOTAL.DKEIPUS.A', id_type='series')# Table86b11 column D, F, G, I
    consumption_combustible_fuels_useful_thermal_output_residual_fuel_oil = self.eia_elec.eia_api(id_='TOTAL.RFEIPUS.A', id_type='series')
    consumption_combustible_fuels_useful_thermal_output_other_liquids = self.eia_elec.eia_api(id_='TOTAL.OLEIPUS.A', id_type='series')
    consumption_combustible_fuels_useful_thermal_output_petroleum_coke = self.eia_elec.eia_api(id_='TOTAL.PCEIMUS.A', id_type='series')
    consumption_combustible_fuels_useful_thermal_output_total_petroleum = self.eia_elec.eia_api(id_='TOTAL.PTEIPUS.A', id_type='series')
    chp_elec
    # eia-923 pivot table
    assumed_conversion_factors = [5.825, 6.287, 5.67, 30120]
    chp_plants_petroleum, elec_only_petroleum = reconcile(total=energy_consumption_petroleum, elec_gen=consumption_for_electricity_generation_petroleum, 
                                                          elec_only_plants=elec_only_plants_petroleum, chp_elec=, assumed_conv_factor=assumed_conversion_factors,
                                                          chp_heat)

def othgas_reconcile():
    consumption_for_electricity_generation_fossil_fuels = self.eia_elec.eia_api(id_='TOTAL.FFEIBUS.A', id_type='series')# Table84b11 column H
    consumption_for_electricity_generation_oth_gas = consumption_for_electricity_generation_fossil_fuels - consumption_for_electricity_generation_petroleum - consumption_for_electricity_generation_natgas - consumption_for_electricity_generation_coal
    consumption_combustible_fuels_electricity_generation_oth_gas = self.eia_elec.eia_api(id_='TOTAL.OJL1BUS.A', id_type='series')# Table85c11 column P
    consumption_combustible_fuels_useful_thermal_output_othgas = self.eia_elec.eia_api(id_='TOTAL.OJEIBUS.A', id_type='series')# Table86b11 column O

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

def industrial_sector_chp_renew():
    wood_energy =  pd.read_excel(self.Table84c_url, index_col=0, usecols='P', skiprows=32, skip_footer=55).multiply(0.001) # Table 8.4C column P # TBtu
    waste_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='R', skiprows=32, skip_footer=55).multiply(0.001) # Table 8.4C column R # TBtu

    wood_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d column R
    waste_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d column T

    # only primary 

def industrial_sector_chp_fossil():
    coal_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='B', skiprows=32, skip_footer=55).multiply(0.001) # Table 8.4c column B # TBtu
    petroleum_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='D', skiprows=32, skip_footer=55).multiply(0.001) # Table 8.4c column D # TBtu
    natgas_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='F', skiprows=32, skip_footer=55).multiply(0.001) # Table 8.4c column F # TBtu
    othgas_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='H', skiprows=32, skip_footer=55).multiply(0.001) # Table 8.4c column H # TBtu

    coal_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d column B
    petroleum_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d column D
    natgas_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d column F
    othgas_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d column H

    # only primary

def industrial_sector_total():
    """Note: Other includes batteries, chemicals, hydrogen, pitch, purchased steam, sulfur, and miscellaneous technologies

    """    
    fossil_fuels_total_energy = # from industrial_sector_chp_fossil total 
    hydroelectric_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='N', skiprows=32, skip_footer=55).multiply(0.001) # Table 8.4C11 column N
    renewable_energy =  # from industrial_sector_chp_renew total 
    other_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='AB', skiprows=32, skip_footer=55).multiply(0.001) # Table 8.4C11 column AB

    fossil_fuels_total_activity = # from industrial_sector_chp_fossil total 
    hydroelectric_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d11 Column P
    renewable_activity =  # from industrial_sector_chp_renew total 
    other_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d11 Column AD

def comm_sector_chp_renew():
    """As is, these are the same sources as ind sector, but should be different part of columns? 
    """    
    wood_energy =  pd.read_excel(self.Table84c_url, index_col=0, usecols='P', skiprows=8, skip_footer=31).multiply(0.001) # Table 8.4C column P # TBtu
    waste_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='R', skiprows=8, skip_footer=31).multiply(0.001) # Table 8.4C column R # TBtu

    wood_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d column R
    waste_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d column T

def comm_sector_chp_fossil():
    coal_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='B', skiprows=8, skip_footer=31).multiply(0.001) # Table 8.4c column B # TBtu
    petroleum_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='D', skiprows=8, skip_footer=31).multiply(0.001) # Table 8.4c column D # TBtu
    natgas_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='F', skiprows=8, skip_footer=31).multiply(0.001) # Table 8.4c column F # TBtu
    othgas_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='H', skiprows=8, skip_footer=31).multiply(0.001) # Table 8.4c column H # TBtu

    coal_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d column B
    petroleum_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d column D
    natgas_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d column F
    othgas_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d column H

    # Industrial and Commercial have same spreadsheet sources but different parts of the columns

def comm_sector_total():
    """Note: Other includes batteries, chemicals, hydrogen, pitch, purchased steam, sulfur, and miscellaneous technologies

    """    
    fossil_fuels_total_energy = # from comm_sector_chp_fossil total 
    hydroelectric_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='N', skiprows=8, skip_footer=31).multiply(0.001) # Table 8.4C11 column N
    renewable_energy =  # from comm_sector_chp_renew total 
    other_energy = pd.read_excel(self.Table84c_url, index_col=0, usecols='AB', skiprows=8, skip_footer=31).multiply(0.001) # Table 8.4C11 column AB

    fossil_fuels_total_activity = # from comm_sector_chp_fossil total 
    hydroelectric_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2d11 Column P
    renewable_activity =  # from comm_sector_chp_renew total 
    other_activity = pd.read_excel(self.Table82d_url, sheet_name='Annual Data', skiprows=9, header=10, index_col=0, usecols=).multiply(0.001) # Table 8.2d11 Column AD

def elec_power_sector_chp_renew():
    wood_energy =  pd.read_excel(self.Table85c_url, index_col=0, usecols='N', skiprows=8, skip_footer=31).multiply(0.001) # Table 8.5C column R # TBtu
    waste_energy = pd.read_excel(self.Table85c_url, index_col=0, usecols='N', skiprows=8, skip_footer=31).multiply(0.001) # Table 8.5C column T # TBtu

    wood_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2c column R
    waste_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2c column T

def elec_power_sector_chp_fossil():
    ~The reconciles~
    
    coal_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2c column B
    petroleum_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2c column D
    natgas_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2c column F
    othgas_activity = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2c column H

def elec_power_sector_chp_total():
    fossil_fuels_total_energy = 
    renewable_total_energy = 
    other_energy = pd.read_excel(self.Table85c_url, index_col=0, usecols='N', skiprows=8, skip_footer=31).multiply(0.001) # Table 8.5c column V
    
    fossil_fuels_total_activity = 
    renewable_total_activity = 
    other_activty = self.eia_elec.eia_api(id_='', id_type='series').multiply(0.001) # Table 8.2c columnAD

def all_chp():
    elec_power_sector_energy = 
    industrial_energy = 
    commercial_energy = 

    elec_power_sector_activity = 
    industrial_activity = 
    commercial_activity = 

def electricity_only_renew():

def electricity_only_fossil():

def electricity_only_total():

def elec_power_sector():

def elec_generation_total():

