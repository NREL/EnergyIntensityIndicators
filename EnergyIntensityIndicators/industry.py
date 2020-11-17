import pandas as pd
from sklearn import linear_model
import os


from EnergyIntensityIndicators.pull_eia_api import GetEIAData
from EnergyIntensityIndicators.pull_bea_api import BEA_api
from EnergyIntensityIndicators.LMDI import CalculateLMDI
from EnergyIntensityIndicators.get_census_data import Asm
from EnergyIntensityIndicators.get_census_data import Econ_census
from EnergyIntensityIndicators.Industry.asm_price_fit import Mfg_prices
from EnergyIntensityIndicators.Industry.nonmanufacuturing import NonManufacturing
# from EnergyIntensityIndicators.Industry.manufacturing import Manufacturing


class IndustrialIndicators(CalculateLMDI):
    """Some of the specific steps to download and process the census data on construction energy costs are
        explained in the following paragraphs. The top-level census bureau website for the Economic Census is:
        https://www.census.gov/programs-surveys/economic-census.html. Scroll down the page until the
        words “2017 Data Tables” are found. After clicking on that link, the user will end up at
        https://www.census.gov/programs-surveys/economic-census/news-updates/updates/2017-datatables.html. The “2017 Data Table pages” now include direct links into data.census.gov and large ftp
        downloads. After clicking on pages, the webpage https://www.census.gov/programssurveys/economic-census/data/tables.html comes up. Scroll down this page until the entry
        “Construction (NAICS Sector 23)” is found. After selecting this entry, the user is then automatically
        transferred to: https://www.census.gov/data/tables/2017/econ/economic-census/naics-sector23.html. 
    """    

    def __init__(self, directory, output_directory, level_of_aggregation=None, lmdi_model='multiplicative', base_year=1985, end_year=2018):
        self.sub_categories_list = {'Manufacturing': {'Food, Beverages, & Tobacco': None, 'Textile Mills and Products': None, 
                                               'Apparel & Leather': None, 'Wood Products': None, 'Paper': None,
                                               'Printing & Allied Support': None, 'Petroleum & Coal Products': None, 'Chemicals': None,
                                               'Plastics & Rubber Products': None, 'Nonmetallic Mineral Products': None, 'Primary Metals': None,
                                               'Fabricated Metal Products': None, 'Machinery': None, 'Computer & Electronic Products': None,
                                               'Electical Equip. & Appliances': None, 'Transportation Equipment': None,
                                               'Furniture & Related Products': None, 'Miscellaneous': None},
                                    'Nonmanufacturing': {'Agriculture, Forestry & Fishing': None,
                                                         'Mining': {'Petroleum and Natural Gas': None, 
                                                                    'Other Mining': None, 
                                                                    'Petroleum drilling and Mining Services': None},
                                                         'Construction': None}}

        self.ind_eia = GetEIAData('industry')
        self.conversion_factors = self.ind_eia.conversion_factors()
        self.MER_Nov19_Table24 = self.ind_eia.eia_api(id_='711252') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711252'
        self.AER10_Table21d = self.ind_eia.eia_api(id_='711252') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711252'
        self.AER11_Table21d_MER0816 = self.ind_eia.eia_api(id_='711252') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711252'
        self.mer_dataT0204 = self.ind_eia.eia_api(id_='711252') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711252'
        self.BEA_Output_data = [0] # Chain-type Quantity Indexes for Value Added by Industry from Bureau of Economic Analysis
        self.energy_types = ['elec', 'fuels', 'deliv', 'source', 'source_adj']

        super().__init__(sector='industry', level_of_aggregation=level_of_aggregation, lmdi_models=lmdi_model, categories_dict=self.sub_categories_list, \
                    energy_types=self.energy_types, directory=directory, output_directory=output_directory, base_year=base_year)

    def reconcile_physical_units(self, ):
        """Convert physical units to Btu. (Prior to 2005, the data on energy consumption fuels to produce electricity were supplied in physical units (e.g. mcf of natural gas, tons of coal, etc))
        Data Source: EIA's Annual Energy Review (AER)"""
        pass

    def manufacturing(self):
        # categories = self.sub_categories_list['Manufacturing']
        # manufacturing_data = Manufacturing().manufacturing()
        # print('manufacturing_data: \n', manufacturing_data)
        manufacturing_data = None
        return manufacturing_data
    
    def non_manufacturing(self):
        """Primary Data Sources: Economic Census (previously the Census of Manufactures, Census of Agriculture, and Census of Mining)
                                Prior to 1985, primary data source is the National Energy Accounts (NEA)
        http://www.nass.usda.gov/Statistics_by_Subject/index.php
        """    
        # categories = self.sub_categories_list['Nonmanufacturing']
        non_manufacturing_data = NonManufacturing().nonmanufacturing_data()
        print('non_manufacturing_data: \n', non_manufacturing_data)

        return non_manufacturing_data
        
    def collect_data(self):
        man = self.manufacturing()
        non_man = self.non_manufacturing()
        data_dict = {'Manufacturing': man, 'Nonmanufacturing': non_man}
        return data_dict

    def total_industrial_util_adj_lmdi(self):
        util_adj_categories = ['Fuels', 'Delivered Electricity', 'Source Electricity', 'Total Source']  # This case is quite different from the others
        return util_adj_categories

    def main(self, breakout, save_breakout, calculate_lmdi):
        unit_conversion_factor = 1

        data_dict = self.collect_data()
        results_dict, formatted_results = self.get_nested_lmdi(level_of_aggregation=self.level_of_aggregation, breakout=breakout, save_breakout=save_breakout, calculate_lmdi=calculate_lmdi, raw_data=data_dict, account_for_weather=False)
        return formatted_results

if __name__ == '__main__': 
    print('os.getcwd()', os.getcwd())
    indicators = IndustrialIndicators(directory='C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020', output_directory='C:/Users/irabidea/Desktop/LMDI_Results', level_of_aggregation='Manufacturing')
    indicators.main(breakout=False, save_breakout=False, calculate_lmdi=False)  