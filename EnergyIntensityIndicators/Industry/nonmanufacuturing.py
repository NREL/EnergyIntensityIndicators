import pandas as pd 

from EnergyIntensityIndicators.pull_bea_api import BEA_api

class NonManufacturing:
    """ Prior to 2012, total nonmanufacturing
    energy consumption (electricity and fuels) was estimated as a residual between the supply-side
    estimates of industrial consumption published by EIA and the end-user estimates for manufacturing
    based upon the MECS (supplemented by census-based data, as described above). The residual-based
    method produced very unsatisfactory results; year-to-year changes in energy consumption were
    implausible in a large number of instances. A complicating factor for fuels is that industrial consumption
    estimates published by EIA include energy products used as chemical feedstocks and other nonfuel
    purposes. As a result, a preliminary effort was undertaken in mid-2012 to estimate energy consumption
    from the user side for these sectors.   


    """    
    def __init__(self):
        BEA_data = BEA_api()
        self.BEA_go_nominal = BEA_data.get_data(years=list(range(1949, 2018)), table_name='go_nominal')
        self.BEA_go_quant_index = BEA_data.get_data(years=list(range(1949, 2018)), table_name='go_quant_index')
        self.BEA_va_nominal = BEA_data.get_data(years=list(range(1949, 2018)), table_name='va_nominal')
        self.BEA_va_quant_index = BEA_data.get_data(years=list(range(1949, 2018)), table_name='va_quant_index')

        self.ALLFOS_historical = pd.read_csv('./Data/ALLFOS_historical.csv')
        self.ELECNEA_historical = pd.read_csv('./Data/ELECNEA_historical.csv')

        pass

    def agriculture(self):
        miranowski_data = [0] # Annual Estimates of energy by fuel for the farm sector for the period 1965-2002
        nass_expenses_data = [0] # https://quickstats.nass.usda.gov/results/06763638-EB97-3879-AAF6-214CF147AED2

        nass_average_prices_data = [0] # 
        MER_fuel_price_data = [0] # 
        eia_table33 = [0]  # Consumer Price estimates for Energy by Source, 1970-2009
        eia_table34 = [0] # Consumer price estimates for energy by end-use sector, 1970-2009
        eia_table523 = [0]  # All sellers sales prices for selected petroleum products, 1994-2010
        eia_table524 = [0]  # Retail motor gasoline and on-highway diesel fuel prices, 1949-2010 
        
        pass

    def indicators_nonman_2018_bea(self):
        """Reformat value added and gross output chain quantity indexes from 
        GrossOutput_1967-2018PNNL_213119.xlsx/ ChainQtyIndexes (EA301:EJ349) and 
        ValueAdded_1969-2018_PNNL_010120.xlsx/ ChainQtyIndexes (EA301:EJ349) respectively 
        """       
        va_quant_index, go_quant_index = BEA_api(years=list(range(1949, 2018))).chain_qty_indexes()
        # HERE: select columns 

        return value_added, gross_output

    def construction(self):
        """https://www.census.gov/data/tables/2017/econ/economic-census/naics-sector-23.html
        https://www.census.gov/data/tables/2012/econ/census/construction.html
        http://factfinder2.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ECN_2007_US_23I1&prodType=table
        http://factfinder2.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ECN_2002_US_23I04A&prodType=table
        http://www.census.gov/epcd/www/97EC23.HTM
        http://www.census.gov/prod/www/abs/cciview.html
        """ 
        value_added, gross_output = self.indicators_nonman_2018_bea() # NonMan_output_data / M, Y
        value_added = value_added['Construction']
        gross_output = gross_output['Construction']

        elec_intensity = 
        fuels_intensity = 
        electricity = elec_intensity.multiply(gross_output).multiply(0.0001)
        fuels = fuels_intensity.multiply(gross_output).multiply(0.0001)
        data_dict = {'energy': 
                        {'elec': electricity, 'fuels': fuels}, 
                     'activity': 
                        {'gross_output': gross_output, 'value_added': value_added}}
        pass

    @staticmethod
    def agriculture():
            miranowski_data =  pd.read_excel('./Agricultural_energy_010420.xlsx', sheet_name='Ag Cons by Use', skiprows=9, usecols='F:G', index_col=0)  # , skipfooter= Annual Estimates of energy by fuel for the farm sector for the period 1965-2002
            nass_expenses_data = [] # https://quickstats.nass.usda.gov/results/06763638-EB97-3879-AAF6-214CF147AED2

            nass_average_prices_data = [] # 
            MER_fuel_price_data = []  # 
            eia_table33 = [] # Consumer Price estimates for Energy by Source, 1970-2009
            eia_table34 = [] # Consumer price estimates for energy by end-use sector, 1970-2009
            eia_table523 = [] # All sellers sales prices for selected petroleum products, 1994-2010
            eia_table524 = [] # Retail motor gasoline and on-highway diesel fuel prices, 1949-2010 
            
            adjustment_factor = 10500/3412 # Assume 10,500 Btu/Kwh
            value_added, gross_output = self.indicators_nonman_2018_bea() # NonMan_output_data_010420.xlsx column G, S (value added and gross output chain qty indexes for farms)
            value_added = value_added['Farms']
            gross_output = gross_output['Farms']

            elec_prm = miranowski_data['Elec-tricity']
            elec_site = elec_prm.divide(adjustment_factor)
            fuels = miranowski_data[['Direct Ag Enery Use']].subtract(elec_prm)

            elec_intensity = 
            fuels_intensity = 

            electricity_final = elec_intensity.multiply(gross_output).multiply(0.001)
            fuels_final = fuels_intensity.multiply(gross_output).multiply(0.001)
            input_for_indicators = pd.DataFrame([electricity_final, fuels_final, gross_output,
                                                 value_added]).transpose().columns(['electricity', 
                                                                                    'fuels', 
                                                                                    'gross_output', 
                                                                                    'value_added'])
    @staticmethod
    def aggregate_mining_data(mining_df):
        mining_df = mining_df.transpose()
        mining_df['Crude Petroleum and Natural Gas'] = mining_df[['Crude Petroleum', 'Natural Gas', 'Natural Gas Liquids']].sum(axis=1)
        mining_df['Coal Mining'] = mining_df[['Anthracite Coal', 'Bituminous Coal']].sum(axis=1)
        mining_df['Metal Ore Mining'] = mining_df[['Iron and Ferroalloy mining', 'Uranium - vanadium ores', 'Nonferrous metals']].sum(axis=1)
        mining_df['Nonmetallic mineral mining'] = mining_df[['Stone and clay mining', 'Chemical and Fertilizer']].sum(axis=1)
        to_transfer = mining_df[['Crude Petroleum and Natural Gas', 'Coal Mining', 'Metal Ore Mining', 
                                 'Nonmetallic mineral mining', 'Oil and gas well drilling']].rename(columns={'Oil and gas well drilling': 
                                                                                                             'Support Activities'}) 
        return to_transfer

    def mining(self):
        """
        mining_2017 = 'https://www.census.gov/data/tables/2017/econ/economic-census/naics-sector-21.html'
        mining_2012 = 'https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?src=bkmk'
        mining_2007 = 'http://factfinder2.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ECN_2007_US_21SG12&prodType=table'
        mining_2002 = 'https://www.census.gov/econ/census02/guide/INDRPT21.HTM'  # extract Table 3 and Table 7
        mining_1997 = 'http://www.census.gov/prod/www/abs/ec1997mining-ind.html'  # extract Table 3 and Table 7
        mining_1992 = 'http://www.census.gov/prod/1/manmin/92mmi/92minif.html'   # extract Table 3 and Table 7
        """            
        # Mining energy_031020.xlsx/Compute_intensities (FF-FN, FQ-FS)

        BLS_data = pd.read_csv('./Data/BLS_Data_011920.csv').transpose().rename(columns={'': 'year'})
        BLS_output_data = pd.read_csv('./Data/BLS Output data 1972-2018.csv')
        BEA_mining_data = self.BEA_data[['Oil and Gas Extraction', 'Mining, except oil and gas', 'Support Activities for Mining']]
        NEA_data_elec = self.aggregate_mining_data(self.ELECNEA_historical) 
        NEA_data_fuels = self.aggregate_mining_data(self.ALLFOS_historical) 

        mining_types = [crude_petroleum_natgas, other_mining, drilling_and_mining_support]
        data_dict = dict()
        for m_type in mining_types = 
            output =  gross_output.multiply(10000)
            elec_intensity = 
            fuels_intensity = 
            electricity = elec_intensity.multiply(output)
            fuels = fuels_intensity.multiply(output)
            m_data = {'energy': {'elec': electricity, 'fuels': fuels}, 
                      'activity': {'gross_output': output}}
            data_dict[m_type] = m_data
        return data_dict

    def propane(self):
        """http://www.eia.gov/totalenergy/data/annual/index.cfm
        """
        pass

    def bureau_labor_statistics_industry_output(self):
        """https://www.bls.gov/emp/data/industry-out-and-emp.htm
        """ 
        pass
    
    def nonmanufacturing_data(self):
        """Collect all nonmanufacturing data
        """        
        # starting point: NonManufacturing_reconciliation_010420.xlsx
        agriculture = self.agriculture() # Agriculutral_energy_010420.xlsx/Intensity_estimates (Y-AB)
        mining = self.mining() # Mining energy_031020.xlsx/Compute_intensities (FQ-FS)
        construction = self.construction() # Construction_energy_011920.xlsx/Intensity_estimates (W-Z)
        data_dict = {'Agriculture': agriculture, 'Mining': mining, 'Construction': construction}
        return data_dict              

if __name__ == '__main__':
    data = NonManufacturing().nonmanufacturing_data()
    print(data)