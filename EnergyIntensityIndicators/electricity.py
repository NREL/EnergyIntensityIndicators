import pandas as pd
from sklearn import linear_model
# from outline import LMDI


class ElectricityIndicators(LMDI):

    def __init__(self, energy_data, activity_data, categories_list):
        super().__init__(energy_data, activity_data, categories_list)
        self.sub_categories_list = categories_list['electricity']
        self.Table21f10 = GetEIAData.eia_api(id_='711254') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711254'
        self.Table21f11 = GetEIAData.eia_api(id_='711254') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711254'
        self.Table82a10 = None  # blank ?
        self.Table82a11 = GetEIAData.eia_api(id_='3') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=3'
        self.Table82b10 = GetEIAData.eia_api(id_='21')  #  'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=21'
        self.Table82b11 = GetEIAData.eia_api(id_='21')  # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=21'
        self.Table82c11 = GetEIAData.eia_api(id_='1736765') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=1736765' ?
        self.Table82d10 = GetEIAData.eia_api(id_='711282') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711282'
        self.Table82d11 = GetEIAData.eia_api(id_='711282') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711282'
        self.Table82d11_2012_and_later = GetEIAData.eia_api(id_='1017') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=1017'
        self.Table83d_03 = GetEIAData.eia_api(id_='711284') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711284' ?
        self.Table84b10 = GetEIAData.eia_api(id_='711284') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711284' ?
        self.Table84b11 = GetEIAData.eia_api(id_='711284') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711284' ?
        self.Table84c11_industrial = GetEIAData.eia_api(id_='456') 
        self.Table84c11_commercial = GetEIAData.eia_api(id_='453') 
        self.Table85c10 = GetEIAData.eia_api(id_='379')  # ?
        self.Table84c10_industrial = GetEIAData.eia_api(id_='456') 
        self.Table84c10_commercial = GetEIAData.eia_api(id_='453') 
        self.Table85c11 = GetEIAData.eia_api(id_='379')  # ?
        self.Table86b09 = GetEIAData.eia_api(id_='463')  # ?
        self.Table86b10 = GetEIAData.eia_api(id_='463') # ?
        self.Table86b11 = GetEIAData.eia_api(id_='463') # ?
        self.MER_T72b_1013_AnnualData = GetEIAData.eia_api(id_='21')  #'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=21'
        self.MER_T72c_1013_AnnualData = GetEIAData.eia_api(id_='2') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=2'
        
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




