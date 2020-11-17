import pandas as pd 
from functools import reduce
from datetime import datetime
import os

from EnergyIntensityIndicators.pull_bea_api import BEA_api
from EnergyIntensityIndicators.get_census_data import Econ_census
from EnergyIntensityIndicators.utilites.standard_interpolation import standard_interpolation


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
        self.currentYear = datetime.now().year
        BEA_data = BEA_api(years=list(range(1949, 2018)))
        self.BEA_go_nominal = BEA_data.get_data(table_name='go_nominal')
        self.BEA_go_quant_index = BEA_data.get_data(table_name='go_quant_index')
        self.BEA_va_nominal = BEA_data.get_data(table_name='va_nominal')
        self.BEA_va_quant_index = BEA_data.get_data(table_name='va_quant_index')

        self.ALLFOS_historical = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/ALLFOS_historical.csv')
        self.ELECNEA_historical = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/ELECNEA_historical.csv')

    # def agriculture(self):
    #     miranowski_data = [0] # Annual Estimates of energy by fuel for the farm sector for the period 1965-2002
    #     nass_expenses_data = [0] # https://quickstats.nass.usda.gov/results/06763638-EB97-3879-AAF6-214CF147AED2

    #     nass_average_prices_data = [0] # 
    #     MER_fuel_price_data = [0] # 
    #     eia_table33 = [0]  # Consumer Price estimates for Energy by Source, 1970-2009
    #     eia_table34 = [0] # Consumer price estimates for energy by end-use sector, 1970-2009
    #     eia_table523 = [0]  # All sellers sales prices for selected petroleum products, 1994-2010
    #     eia_table524 = [0]  # Retail motor gasoline and on-highway diesel fuel prices, 1949-2010 
        
    #     pass

    def indicators_nonman_2018_bea(self):
        """Reformat value added and gross output chain quantity indexes from 
        GrossOutput_1967-2018PNNL_213119.xlsx/ ChainQtyIndexes (EA301:EJ349) and 
        ValueAdded_1969-2018_PNNL_010120.xlsx/ ChainQtyIndexes (EA301:EJ349) respectively 
        """       
        va_quant_index, go_quant_index = BEA_api(years=list(range(1949, 2018))).chain_qty_indexes()
        # HERE: select columns 

        return value_added, gross_output
    
    def get_econ_census(self):
        economic_census = Econ_census()
        economic_census_years = list(range(1987, self.currentYear + 1, 5))       
        e_c_data = {str(y): economic_census.get_data(year=y) 
                    for y in economic_census_years}
        print(e_c_data)
        return e_c_data
    
    def construction_raw_data(self):
        """Equivalent to Construction_energy_011920.xlsx['Construction']
        """ 

        return construction_elec, construction_fuels

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
        electricity, fuels = self.construction_raw_data()

        elec_intensity = electricity.divide(gross_output * 0.0001)
        elec_intensity = elec_intensity(fuels_intensity).fillna(method='bfill')

        fuels_intensity = fuels.divide(gross_output * 0.0001)
        fuels_intensity.iloc[1982] = np.nan
        fuels_intensity.iloc[2002] = np.nan
        fuels_intensity = standard_interpolation(fuels_intensity).fillna(method='bfill')

        final_electricity = elec_intensity.multiply(gross_output * 0.0001)
        final_fuels = fuels_intensity.multiply(gross_output * 0.0001)
        data_dict = {'energy': 
                        {'elec': final_electricity, 'fuels': final_fuels}, 
                     'activity': 
                        {'gross_output': gross_output, 'value_added': value_added}}
        return data_dict

    def agriculture(self):
            miranowski_data =  pd.read_excel('./EnergyIntensityIndicators/Industry/Data/miranowski_data.xlsx', sheet_name='Ag Cons by Use', skiprows=9, usecols='F:G', index_col=0)  # , skipfooter= Annual Estimates of energy by fuel for the farm sector for the period 1965-2002
            
            adjustment_factor = 10500/3412 # Assume 10,500 Btu/Kwh
            value_added, gross_output = self.indicators_nonman_2018_bea() # NonMan_output_data_010420.xlsx column G, S (value added and gross output chain qty indexes for farms)
            value_added = value_added['Farms']
            gross_output = gross_output['Farms']

            elec_prm = miranowski_data['Elec-tricity']
            elec_site = elec_prm.divide(adjustment_factor)
            fuels = miranowski_data[['Direct Ag Enery Use']].subtract(elec_prm)

            elec_intensity = elec_site.divide(gross_output * 0.001)
            fuels_intensity = fuels.divide(gross_output * 0.001)

            electricity_final = elec_intensity.multiply(gross_output * 0.001).ffill()
            fuels_final = fuels_intensity.multiply(gross_output * 0.001)

            data_dict = {'energy': {'elec': electricity_final, 
                                    'fuels': fuels_final}, 
                         'activity': {'gross_output': gross_output}, 
                                      'value_added': value_added}
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
    
    @staticmethod
    def build_mining_output(factor, gross_output, elec, fuels):
        output_by_factor = gross_output.multiply(factor)
        elec_intensity = elec.divide(output_by_factor)
        elec_intensity = standard_interpolation(elec_intensity).ffill()

        fuels_intensity = fuels.divide(output_by_factor)
        fuels_intensity = standard_interpolation(fuels_intensity).ffill()

        electricity_final = elec_intensity.multiply(elec_intensity)
        fuels_final = fuels_intensity.multiply(fuels_intensity)
        data_dict = {'energy': {'elec': electricity_final, 'fuels': fuels_final}, 
                     'activity': {'gross_output': gross_output}}
        return data_dict

    def crude_petroleum_natgas(self, bea_bls_output, nea_elec, nea_fuels, sector_estimates):
        factor = 0.0001
        gross_output = bea_bls_output[['Oil & Gas']]
        elec = nea_elec[['Crude Pet']] 
        fuels = nea_fuels[['Crude Pet']]
        data_dict = self.build_mining_output(factor, gross_output, elec, fuels)
        return data_dict
    
    def coal_mining(self, bea_bls_output, nea_elec, nea_fuels, sector_estimates): 
        factor = 0.001

        col = ['Coal Mining']
        gross_output = bea_bls_output[col]
        elec = nea_elec[col] 
        fuels = nea_fuels[col] 
        data_dict = self.build_mining_output(factor, gross_output, elec, fuels)
        return data_dict
    
    def metal_mining(self, bea_bls_output, nea_elec, nea_fuels, sector_estimates):
        factor = 0.01
        gross_output = bea_bls_output[['Metal Ore Mining']]
        elec = nea_elec[['Metal Mining']] 
        fuels = nea_fuels[['Metal Mining']] 
        data_dict = self.build_mining_output(factor, gross_output, elec, fuels)
        return data_dict

    def nonmetallic_mineral_mining(self, bea_bls_output, nea_elec, nea_fuels, sector_estimates):
        factor = 0.01
        col = ['Nonmetallic Mineral Mining']
        gross_output = bea_bls_output[col]
        elec = nea_elec[col] 
        fuels = nea_fuels[col] 
        data_dict = self.build_mining_output(factor, gross_output, elec, fuels)
        return data_dict

    def other_mining(self, bea_bls_output, nea_elec, nea_fuels, sector_estimates): 
        """Collect data for "other mining" from the sum of nonmetallic mineral
        mining, metal mining and coal mining. 

        Args:
            bea_bls_output ([type]): [description]
            nea_elec ([type]): [description]
            nea_fuels ([type]): [description]
            sector_estimates ([type]): [description]

        Returns:
            [type]: [description]
        """        
        factor = 0.01
        gross_output = bea_bls_output['Other Mining']

        other_mining_types = {'nonmetallic_mineral_mining': self.nonmetallic_mineral_mining, 
                              'metal_mining': self.metal_mining, 
                              'coal_mining': self.coal_mining}
   
        other_mining_data = [m(bea_bls_output, nea_elec, nea_fuels, 
                               sector_estimates) for m in other_mining_types.values()]

        other_mining_elec = [m_df['energy']['elec'] for m_df in other_mining_data]
        elec = reduce(lambda x, y: x.add(y), other_mining_elec)

        other_mining_fuels = [m_df['energy']['fuels'] for m_df in other_mining_data]
        fuels = reduce(lambda x, y: x.add(y), other_mining_fuels)

        data_dict = self.build_mining_output(factor, gross_output, elec, fuels)
        return data_dict
    
    def drilling_and_mining_support(self, bea_bls_output, nea_elec, nea_fuels, sector_estimates):
        factor = 0.001
        col = ['Support Activities']
        gross_output = bea_bls_output[col]
        elec = nea_elec[col] 
        fuels = nea_fuels[col] 
        data_dict = self.build_mining_output(factor, gross_output, elec, fuels, sector_estimates)
        return data_dict
    
    @staticmethod
    def mining_fuels_adjust(ec_df):
        """[summary]

        Args:
            ec_df (dataframe): Economic Census data 
                               for NAICS code 21 (mining) 
                               at the 6-digit level

        Returns:
            ratio dataframe: ratio of total cost to the sum of reported 
        """        
        fuel_types = ['gasoline', 'gas', 'distillate', 'residual', 'coal']
        reported = ec_df[fuel_types].sum(axis=1)
        ratio = ec_df[['total_cost']].divide(ec_df['other_fuel'].add(reported)).subtract(1)
        return ratio

    @staticmethod
    def price_ratios():
        pass

    @staticmethod
    def calculate_physical_units():
        calc = current_cost.divide(previous_cost * 
                            current_price).multiply(previous_pyhsical_units)
        pass

    @staticmethod
    def mining_data_1987_2017():
        """ For updating estimates, cost of purchased fuels from the Economic
        Census and aggregate (annual) fuel prices from EIA (Monthly Energy Review). Output data (gross output
        and value added) derived from the Bureau of Economic Analysis (through spreadsheet
        NonMan_output_data_date, and gross output data from the Bureau of Labor Statistics (for detailed subsectors in mining).

        mining_2017 = 'https://www.census.gov/data/tables/2017/econ/economic-census/naics-sector-21.html'
        mining_2012 = 'https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?src=bkmk'
        mining_2007 = 'http://factfinder2.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ECN_2007_US_21SG12&prodType=table'
        mining_2002 = 'https://www.census.gov/econ/census02/guide/INDRPT21.HTM'  # extract Table 3 and Table 7
        mining_1997 = 'http://www.census.gov/prod/www/abs/ec1997mining-ind.html'  # extract Table 3 and Table 7
        mining_1992 = 'http://www.census.gov/prod/1/manmin/92mmi/92minif.html'   # extract Table 3 and Table 7
        """ 
        mining_2017 = pd.read_fwf('') # from economic census


        # return {'elec': , 'fuels': }
        pass
    
    def mining_sector_estimates(data_1987_2017):
        elec = data_1987_2017['elec']
        fuels = data_1987_2017['fuels']

        sector_estimates_elec = elec.transpose().multiply(0.000001 * 3412)
        sector_estimates_fuels = fuels.transpose()
        return sector_estimates_elec, sector_estimates_fuels

    def mining(self):
           
        # Mining energy_031020.xlsx/Compute_intensities (FF-FN, FQ-FS)

        BLS_data = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/BLS_Data_011920.csv').transpose().rename(columns={'': 'year'})
        BLS_output_data = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/BLS Output data 1972-2018.csv')
        BEA_mining_data = self.BEA_data[['Oil and Gas Extraction', 'Mining, except oil and gas', 'Support Activities for Mining']]
        NEA_data_elec = self.aggregate_mining_data(self.ELECNEA_historical) 
        NEA_data_fuels = self.aggregate_mining_data(self.ALLFOS_historical) 

        mining_types = [crude_petroleum_natgas, other_mining, drilling_and_mining_support]
        

        data_dict = dict()
        for m_type in mining_types: 
            m_type_data = self.m_type()
            data_dict[str(m_type)] = m_type_data
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
        # Agriculutral_energy_010420.xlsx/Intensity_estimates (Y-AB)
        # Mining energy_031020.xlsx/Compute_intensities (FQ-FS)
        # Construction_energy_011920.xlsx/Intensity_estimates (W-Z)
        data_dict = {'Agriculture': self.agriculture(), 'Mining': self.mining(), 
                     'Construction': self.construction()}
        return data_dict              

if __name__ == '__main__':
    print('main')
    data = NonManufacturing().get_econ_census()
    print(data)