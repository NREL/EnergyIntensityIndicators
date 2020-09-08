import pandas as pd
from sklearn import linear_model
from pull_eia_api import GetEIAData
# from outline import LMDI


class GetIndustryData():
    """Some of the specific steps to download and process the census data on construction energy costs are
        explained in the following paragraphs. The top-level census bureau website for the Economic Census is:
        https://www.census.gov/programs-surveys/economic-census.html. Scroll down the page until the
        words “2017 Data Tables” are found. After clicking on that link, the user will end up at
        https://www.census.gov/programs-surveys/economic-census/news-updates/updates/2017-datatables.html. The “2017 Data Table pages” now include direct links into data.census.gov and large ftp
        downloads. After clicking on pages, the webpage https://www.census.gov/programssurveys/economic-census/data/tables.html comes up. Scroll down this page until the entry
        “Construction (NAICS Sector 23)” is found. After selecting this entry, the user is then automatically
        transferred to: https://www.census.gov/data/tables/2017/econ/economic-census/naics-sector23.html. 
    """    
    BEA_data = pd.read_excel('./GrossOutput_1969-2018_PNNL:_123119.xlsx', sheet_name='ChainQTYIndexes', skiprows=298, usecols="DC, EA:EJ")

    def manufacturing():
        """Main datasource is the Manufacturing Energy Consumption Survey (MECS), conducted by the EIA since 1985 (supplemented for non-MECS years by 
        estimates derived from the Annual Survey of Manufactures (ASM) and the Economic Census (EC) conducted every five years)
        https://www.eia.gov/consumption/manufacturing/data/2014/
        https://www.eia.gov/consumption/manufacturing/data/2014/#r4 
        """    

    def non_manufacturing():
        """Primary Data Sources: Economic Census (previously the Census of Manufactures, Census of Agriculture, and Census of Mining)
                                Prior to 1985, primary data source is the National Energy Accounts (NEA)
        http://www.nass.usda.gov/Statistics_by_Subject/index.php
        """    

        def agriculture():
            miranowski_data =  pd.read_excel('./Agricultural_energy_010420.xlsx', sheet_name='Ag Cons by Use', skiprows=9, usecols='F:G', index_col=0, skip_footer=)# Annual Estimates of energy by fuel for the farm sector for the period 1965-2002
            nass_expenses_data =  # https://quickstats.nass.usda.gov/results/06763638-EB97-3879-AAF6-214CF147AED2

            nass_average_prices_data =  # 
            MER_fuel_price_data =  # 
            eia_table33 =  # Consumer Price estimates for Energy by Source, 1970-2009
            eia_table34 =  # Consumer price estimates for energy by end-use sector, 1970-2009
            eia_table523 = # All sellers sales prices for selected petroleum products, 1994-2010
            eia_table524 =  # Retail motor gasoline and on-highway diesel fuel prices, 1949-2010 
            
            adjustment_factor = 10500/3412 # Assume 10,500 Btu/Kwh
            gross_output =  # NonMan_output_data_010420.xlsx column S
            value_added =  # NonMan_output_data_010420.xlsx column G
            elec_prm = miranowski_data[0]
            elec_site = elec_prm.divide(adjustment_factor)
            fuels = miranowski_data[0].subtract(miranowski_data[0])
            electricity_intensity = elec_site.divide(0.001)
            fuels_intensity = fuels.divide(0.001)
            input_for_indicators = pd.DataFrame([electricity_intesity, fuels_intensity, gross_output,
                                                 value_added]).transpose().columns(['electricity_intesity', 
                                                                                    'fuels_intensity', 
                                                                                    'gross_output', 
                                                                                    'value_added'])
        def mining():
            """[summary]
            https://www.census.gov/data/tables/2017/econ/economic-census/naics-sector-21.html
            https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?src=bkmk
            http://factfinder2.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ECN_2007_US_21SG12&prodType=table
            https://www.census.gov/econ/census02/guide/INDRPT21.HTM
            http://www.census.gov/prod/www/abs/ec1997mining-ind.html
            http://www.census.gov/prod/1/manmin/92mmi/92minif.html
            """            
            BLS_data = pd.read_csv('./BLS_Data_011920.csv').transpose().rename(columns={'': 'year'})
            BEA_mining_data = BEA_data[['Oil and Gas Extraction', 'Mining, except oil and gas', 'Support Activities for Mining']]
            NEA_data = # NEA_Data

            crude_petroleum_natgas = BEA_mining_data['Oil and Gas Extraction'].multiply(0.001)
            crude_petroleum_natgas['Elec'] = 

            pass

        def propane():
            """http://www.eia.gov/totalenergy/data/annual/index.cfm
            """
            pass

        def bureau_labor_statistics_industry_output():
            """https://www.bls.gov/emp/data/industry-out-and-emp.htm
            """ 
            pass

        def construction():
            """https://www.census.gov/data/tables/2017/econ/economic-census/naics-sector-23.html
               https://www.census.gov/data/tables/2012/econ/census/construction.html
               http://factfinder2.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ECN_2007_US_23I1&prodType=table
               http://factfinder2.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ECN_2002_US_23I04A&prodType=table
               http://www.census.gov/epcd/www/97EC23.HTM
               http://www.census.gov/prod/www/abs/cciview.html
            """ 
            pass
        
        pass                                  


class IndustrialIndicators(LMDI):

    def __init__(self, energy_data, activity_data, categories_list):
        super().__init__(energy_data, activity_data, categories_list)
        self.sub_categories_list = categories_list['industry']
        self.ind_eia = GetEIAData('industry')
        self.conversion_factors = self.ind_eia.conversion_factors()
        self.MER_Nov19_Table24 = self.ind_eia.eia_api(id_='711252') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711252'
        self.AER10_Table21d = self.ind_eia.eia_api(id_='711252') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711252'
        self.AER11_Table21d_MER0816 = self.ind_eia.eia_api(id_='711252') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711252'
        self.mer_dataT0204 = self.ind_eia.eia_api(id_='711252') # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711252'
        self.BEA_Output_data =  # Chain-type Quantity Indexes for Value Added by Industry from Bureau of Economic Analysis

    def reconcile_physical_units(self, ):
        """Convert physical units to Btu. (Prior to 2005, the data on energy consumption fuels to produce electricity were supplied in physical units (e.g. mcf of natural gas, tons of coal, etc))
        Data Source: EIA's Annual Energy Review (AER)"""
        pass



    def fuel_electricity_consumption(categories, elec=True, fuels=True, deliv=True, source=True, source_adj=True):
        """Trillion Btu
        """        
        conversion_factors = self.ind_eia.conversion_factors('industry')
        conversion_factors = conversion_factors.set_index('year')
        energy_type_dfs = []
        
        if  elec:
            energy_type_dfs.append(delivered_electricity)

        if fuels: 

            energy_type_dfs.append(fuels)

        if deliv: 

            energy_type_dfs.append(delivered)

        if source:

            energy_type_dfs.append(total_source)
    
        if source_adj:

            energy_type_dfs.append(source_adj)

        fuel_electricity_consumption = pd.concat(energy_type_dfs)
        return fuel_electricity_consumption_df

    def activity(categories):
        """Million kWh, Activity is just repeated over and over
        """
        activity_df = 
        activity_df['total'] = activity_df.sum(axis=1)        
        return activity_df 

    def mining_lmdi():
        mining_categories =  self.sub_categories_list['Nonmanufcaturing']['Mining']:
        mining_energy_input_data = IndustrialIndicators.fuel_electricity_consumption(mining_categories, source_adj=False)
        mining_activity_input_data = IndustrialIndicators.activity(mining_categories)
        deliv = super().get_deliv()
        
        mining = super().lmdi_multiplicative(mining_activity_input_data, mining_energy_input_data, _base_year)

    def nonmanufacturing_lmdi():
        nonmanufcaturing_categories =  self.sub_categories_list['Nonmanufcaturing']
        nonmanufcaturing_energy_input_data = IndustrialIndicators.fuel_electricity_consumption(nonmanufcaturing_categories)

        nonmanufcaturing_activity_input_data = IndustrialIndicators.activity(nonmanufcaturing_categories)
        nonmanufacturing = super().lmdi_multiplicative(activity_input_data, energy_input_data, _base_year)

    def total_industrial_lmdi():
        industrial_categories = self.sub_categories_list.keys()
        industry_energy_input_data = IndustrialIndicators.fuel_electricity_consumption(industrial_categories)
        industry_energy_input_data = IndustrialIndicators.activity(industrial_categories)
        total_industrial = super().lmdi_multiplicative(activity_input_data, energy_input_data, _base_year)

    def total_industrial_util_adj_lmdi():
        util_adj_categories = ['Fuels', 'Delivered Electricity', 'Source Electricity', 'Total Source']
        industrial_total_lmdi_utiladj = super().lmdi_multiplicative(activity_input_data, energy_input_data, _base_year) # This case is quite different from the others

