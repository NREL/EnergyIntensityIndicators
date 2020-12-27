import pandas as pd 
from functools import reduce
from datetime import datetime
import os
import numpy as np

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
        self.BEA_data = BEA_api(years=list(range(1949, self.currentYear + 1)))
        self.BEA_go_nominal = self.BEA_data.get_data(table_name='go_nominal')
        self.BEA_go_quant_index = self.BEA_data.get_data(table_name='go_quant_index')
        self.BEA_va_nominal = self.BEA_data.get_data(table_name='va_nominal')
        self.BEA_va_quant_index = self.BEA_data.get_data(table_name='va_quant_index')

    def indicators_nonman_2018_bea(self):
        """Reformat value added and gross output chain quantity indexes from 
        GrossOutput_1967-2018PNNL_213119.xlsx/ ChainQtyIndexes (EA301:EJ349) and 
        ValueAdded_1969-2018_PNNL_010120.xlsx/ ChainQtyIndexes (EA301:EJ349) respectively 
        """       
        va_quant_index, go_quant_index = BEA_api(years=list(range(1949, 2018))).chain_qty_indexes()
        # HERE: select columns 
        print('va_quant_index:\n', va_quant_index.columns)
        print('go_quant_index:\n', go_quant_index.columns)
        return va_quant_index, go_quant_index
    
    def get_econ_census(self):
        """Collect economic census data"""
        economic_census = Econ_census()
        economic_census_years = list(range(1987, self.currentYear + 1, 5))       
        e_c_data = {str(y): economic_census.get_data(year=y) 
                    for y in economic_census_years}
        print(e_c_data)
        return e_c_data
    
    @staticmethod
    def petroleum_prices(retail_gasoline, retail_diesel, excl_tax_gasoline, excl_tax_diesel):
        """Get petroleum prices"""
        retail_gasoline.loc[2011] = 3.527
        retail_gasoline.loc[2012] = 3.644
        retail_gasoline.loc[2013] = 3.526
        retail_gasoline.loc[2014] = 3.367
        retail_gasoline.loc[2015] = 2.448
        retail_gasoline.loc[2016] = 2.142
        retail_gasoline.loc[2017] = 2.408

        retail_gasoline['Excl. Tax'] = retail_gasoline.divide(retail_gasoline.loc[1994, 'Retail']).multiply(excl_tax_gasoline.loc[1994])
        retail_gasoline['$/MMBtu'] = retail_gasoline.divide(retail_gasoline.loc[1994, 'Retail']).multiply(excl_tax_gasoline.loc[1994])
        
        retail_diesel['Excl. Tax'] = retail_diesel.divide(retail_diesel.loc[1994, 'Retail']).multiply(excl_tax_diesel.loc[1994])
        retail_diesel['$/MMBtu'] = retail_diesel.divide(retail_diesel.loc[1994, 'Retail']).multiply(excl_tax_diesel.loc[1994])

        gasoline_weight = 0.3
        diesel_weight = 0.7
        lubricant_weights = 2

        dollar_mmbtu = retail_diesel['$/MMBtu'] * diesel_weight + retail_gasoline['$/MMBtu'] * gasoline_weight
        lubricant = dollar_mmbtu.multiply(lubricant_weights)
        return dollar_mmbtu, lubricant


    def construction_raw_data(self):
        """Equivalent to Construction_energy_011920.xlsx['Construction']
        """ 
        stb0303 = pd.read_excel('./EnergyIntensityIndicators/Industry/Data/stb0303.xlsx', sheet_name='stb0303')
        stb0304 = pd.read_excel('./EnergyIntensityIndicators/Industry/Data/stb0304.xlsx', sheet_name='stb0304')

        stb0523 = pd.read_excel('./EnergyIntensityIndicators/Industry/Data/stb0523.xlsx', sheet_name='stb0523')
        stb0524 = pd.read_csv('https://www.eia.gov/totalenergy/data/browser/csv.php?tbl=T09.04')
        
        construction_elec_fuels = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/construction_elec_fuels.csv').set_index('Year')
        construction_elec_fuels = construction_elec_fuels.rename(columns={'  Electricity': 'Electricity'})
        print('construction_elec_fuels:\n', construction_elec_fuels)
        print('construction_elec_fuels columns:\n', construction_elec_fuels.columns)

        construction_elec = construction_elec_fuels[['Electricity']]
        construction_fuels = construction_elec_fuels[['Total Fuel']]

        return construction_elec, construction_fuels

    def construction(self):

        """Build data dictionary for the construction sector

        https://www.census.gov/data/tables/2017/econ/economic-census/naics-sector-23.html
        https://www.census.gov/data/tables/2012/econ/census/construction.html
        http://factfinder2.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ECN_2007_US_23I1&prodType=table
        http://factfinder2.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ECN_2002_US_23I04A&prodType=table
        http://www.census.gov/epcd/www/97EC23.HTM
        http://www.census.gov/prod/www/abs/cciview.html
        """ 
        value_added, gross_output = self.indicators_nonman_2018_bea() # NonMan_output_data / M, Y
        value_added = value_added[['Construction']].rename(columns={'Construction': 'Value Added'})
        print('value_added:\n', value_added)

        gross_output = gross_output[['Construction']].rename(columns={'Construction': 'Gross Output'})
        gross_output['Output*0.0001'] = gross_output['Gross Output'].multiply(0.0001)

        print('gross_output:\n', gross_output)
        electricity, fuels = self.construction_raw_data()
        print('fuels:\n', fuels)
        print('electricity:\n', electricity)

        elec_intensity = electricity.merge(gross_output, how='outer', left_index=True, right_index=True)
        elec_intensity['elec_intensity'] = elec_intensity['Electricity'].divide(elec_intensity['Output*0.0001'] .values)
        elec_intensity = standard_interpolation(elec_intensity, name_to_interp='elec_intensity', axis=1).fillna(method='bfill')
        print('elec_intensity after interp:\n', elec_intensity)
        fuels_intensity = fuels.merge(gross_output, how='outer', left_index=True, right_index=True)
        fuels_intensity['fuels_intensity'] = fuels_intensity['Total Fuel'].divide(fuels_intensity['Output*0.0001'] .values)
        print('fuels intensity:\n', fuels_intensity)
        print('fuels intensity index:\n', fuels_intensity.index)
        print('fuels intensity columns:\n', fuels_intensity.columns)

        fuels_intensity.loc[1982, 'fuels_intensity'] = np.nan
        fuels_intensity.loc[2002, 'fuels_intensity'] = np.nan
        fuels_intensity = standard_interpolation(fuels_intensity, name_to_interp='fuels_intensity', axis=1).fillna(method='bfill')
        print('fuels_intensity after interp:\n', fuels_intensity)

        final_electricity = elec_intensity['elec_intensity'].multiply(elec_intensity['Output*0.0001'].values).to_frame(name='electricity')
        print('final electricity:\n', final_electricity)
        final_fuels = fuels_intensity['fuels_intensity'].multiply(fuels_intensity['Output*0.0001'].values).to_frame(name='fuels')
        print('final fuels:\n', final_fuels)

        data_dict = {'energy': 
                        {'elec': final_electricity, 'fuels': final_fuels}, 
                     'activity': 
                        {'gross_output': gross_output, 'value_added': value_added}}
        return data_dict

    def agriculture(self):
        """Build data dictionary for the agricultural sector"""
        miranowski_data =  pd.read_excel('./EnergyIntensityIndicators/Industry/Data/miranowski_data.xlsx', 
                                         sheet_name='Ag Cons by Use', skiprows=4, skipfooter=9, usecols='A,F:G', 
                                         index_col=0, names=['Year', 'Electricity', 'Direct Ag. Energy Use'])  
                                         # Annual Estimates of energy by fuel for the farm sector for the period 1965-2002
        miranowski_data = miranowski_data.reset_index()
        miranowski_data['Year'] = pd.to_numeric(miranowski_data['Year'], errors='coerce')
        miranowski_data = miranowski_data.dropna(subset=['Year']).set_index('Year')
        print('miranowski_data:\n', miranowski_data)
        adjustment_factor = 10500/3412 # Assume 10,500 Btu/Kwh
        value_added, gross_output = self.indicators_nonman_2018_bea() # NonMan_output_data_010420.xlsx column G, S (value added and gross output chain qty indexes for farms)
        value_added = value_added[['Farms']]
        gross_output = gross_output[['Farms']]
        print('gross_output:\n', gross_output)
        print('value_added:\n', value_added)


        elec_prm = miranowski_data[['Electricity']].rename(columns={'Electricity': 'elec'})
        print('elec_prm:\n', elec_prm)
        elec_site = elec_prm.divide(adjustment_factor)
        fuels = miranowski_data[['Direct Ag. Energy Use']].subtract(elec_prm.values).rename(columns={'Direct Ag. Energy Use': 'fuels'})
        print('fuels:\n', fuels)

        elec_df = elec_site.merge(gross_output, how='outer', left_index=True, right_index=True)
        fuels_df = fuels.merge(gross_output, how='outer', left_index=True, right_index=True)

        elec_df['elec_intensity'] = elec_df['elec'].divide(elec_df['Farms'] * 0.001, axis='index')
        fuels_df['fuels_intensity'] = fuels_df['fuels'].divide(fuels_df['Farms'] * 0.001, axis='index')

        electricity_final = elec_df[['elec_intensity']].multiply(elec_df['Farms'] * 0.001, axis='index').ffill()
        fuels_final = fuels_df[['fuels_intensity']].multiply(fuels_df['Farms'] * 0.001, axis='index')

        data_dict = {'energy': {'elec': electricity_final, 
                                'fuels': fuels_final}, 
                        'activity': {'gross_output': gross_output, 
                                     'value_added': value_added}}
        print('data_dict:\n', data_dict)
        return data_dict

    def aggregate_mining_data(self, mining_df, allfos=False):
        print('mining df:\n', mining_df)
        mapping = {5: 'Iron and Ferroalloy mining', 6: 'Uranium - vanadium ores', 
                   7: 'Nonferrous metals', 8: 'Anthracite Coal', 9: 'Bituminous Coal', 
                   10: 'Crude Petroleum', 11: 'Natural Gas', 12: 'Natural Gas Liquids', 
                   13: 'Stone and clay mining', 14: 'Chemical and Fertilizer', 
                   15: 'Oil and gas well drilling'}

        mapping_df = pd.DataFrame.from_dict(mapping, orient='index', columns=['Industry'])
        mapping_df.index.name = 'Year'
        mapping_df = mapping_df.reset_index()
        if allfos:
            mapping_df['Year'] = mapping_df['Year'].subtract(1)
        
        mapping_df['Year']= mapping_df['Year'].astype(int)
        print('mapping_df:\n', mapping_df)

        mining_df = mining_df.merge(mapping_df, how='right', on='Year')
        mining_df = mining_df.drop(['Year', 'NAICS'], axis=1).set_index('Industry')

        print('mining_df:\n', mining_df)

        mining_df = mining_df.transpose()

        mining_df['Crude Petroleum and Natural Gas'] = mining_df[['Crude Petroleum', 'Natural Gas', 'Natural Gas Liquids']].sum(axis=1)
        mining_df['Coal Mining'] = mining_df[['Anthracite Coal', 'Bituminous Coal']].sum(axis=1)
        mining_df['Metal Ore Mining'] = mining_df[['Iron and Ferroalloy mining', 'Uranium - vanadium ores', 'Nonferrous metals']].sum(axis=1)
        mining_df['Nonmetallic mineral mining'] = mining_df[['Stone and clay mining', 'Chemical and Fertilizer']].sum(axis=1)
        print('mining_df:\n', mining_df)

        to_transfer = mining_df[['Crude Petroleum and Natural Gas', 'Coal Mining', 'Metal Ore Mining', 
                                 'Nonmetallic mineral mining', 'Oil and gas well drilling']].rename(columns={'Oil and gas well drilling': 
                                                                                                             'Support Activities', 
                                                                                                             'Crude Petroleum and Natural Gas':
                                                                                                             'Crude Pet'}) 
        print('to_transfer:\n', to_transfer)
        return to_transfer
    
    @staticmethod
    def build_mining_output(factor, gross_output, elec, fuels, sector_estimates_elec, 
                            sector_estimates_fuels, col_name):
        """Build data dictionary for the mining subsector"""
        print('elec:\n', elec)
        print('fuels:\n', fuels)
        elec = elec.rename(columns={col_name: 'elec'})
        fuels = fuels.rename(columns={col_name: 'fuels'})

        sector_estimates_elec = sector_estimates_elec.rename(columns={col_name: 'elec'})
        print('sector_estimates_elec:\n', sector_estimates_elec)
        sector_estimates_fuels = sector_estimates_fuels.rename(columns={col_name: 'fuels'})
        print('sector_estimates_fuels:\n', sector_estimates_fuels)
        elec = pd.concat([elec, sector_estimates_elec], axis=0)
        print('elec:\n', elec)
        fuels = pd.concat([fuels, sector_estimates_fuels], axis=0)
        print('fuels:\n', fuels)

        gross_output['output_by_factor'] = gross_output.multiply(factor)
        print('gross_output')
        elec_df = gross_output.merge(elec, how='outer', left_index=True, right_index=True)
        print('output_by_factor:\n', gross_output['output_by_factor'])
        print('elec_df:\n', elec_df)

        fuels_df = gross_output.merge(fuels, how='outer', left_index=True, right_index=True)
        print('fuels_df:\n', fuels_df)

        elec_df['elec_intensity'] = elec_df['elec'].divide(elec_df['output_by_factor'].values)
        print('elec_intensity:\n', elec_df['elec_intensity'])

        elec_df = standard_interpolation(elec_df, name_to_interp='elec_intensity', axis=1).ffill()

        fuels_df['fuels_intensity'] = fuels_df['fuels'].divide(fuels_df['output_by_factor'].values)
        print('fuels_intensity:\n', fuels_df['fuels_intensity'])

        fuels_df = standard_interpolation(fuels_df, name_to_interp='fuels_intensity', axis=1).ffill()

        electricity_final = elec_df['elec_intensity'].multiply(elec_df['output_by_factor'].values)
        fuels_final = fuels_df['fuels_intensity'].multiply(fuels_df['output_by_factor'].values)
        gross_output = gross_output.drop('output_by_factor', axis=1)
        data_dict = {'energy': {'elec': electricity_final, 'fuels': fuels_final}, 
                     'activity': {'gross_output': gross_output}}
        return data_dict

    def crude_petroleum_natgas(self, bea_bls_output, nea_elec, nea_fuels, sector_estimates):
        """Collect crude petroleum and natural gas data for the mining subsector"""


        factor = 0.0001
        gross_output = bea_bls_output[['Oil & Gas']]
        elec = nea_elec[['Crude Pet']].rename(columns={'Crude Pet': 'Oil & Gas'}) 
        fuels = nea_fuels[['Crude Pet']].rename(columns={'Crude Pet': 'Oil & Gas'}) 
        col_name = 'Oil & Gas'

        sector_estimates_elec = sector_estimates[0][['Oil and Gas']].rename(columns={'Oil and Gas': col_name})
        sector_estimates_fuels = sector_estimates[1][['Oil and Gas']].rename(columns={'Oil and Gas': col_name})

        data_dict = self.build_mining_output(factor, gross_output, elec, fuels, 
                                             sector_estimates_elec, sector_estimates_fuels, 
                                             col_name)
        return data_dict
    
    def coal_mining(self, bea_bls_output, nea_elec, nea_fuels, sector_estimates): 
        """Collect coal mining data for the mining subsector"""

        factor = 0.001

        col = 'Coal Mining'
        bea_bls_output = bea_bls_output.rename(columns={'Coal mining': col})
        gross_output = bea_bls_output[[col]]
        elec = nea_elec[[col]] 
        fuels = nea_fuels[[col]] 

        sector_estimates_elec = sector_estimates[0][[col]]
        sector_estimates_fuels = sector_estimates[1][[col]]

        data_dict = self.build_mining_output(factor, gross_output, elec, fuels,
                                             sector_estimates_elec, sector_estimates_fuels, 
                                             col_name=col)
        return data_dict
    
    def metal_mining(self, bea_bls_output, nea_elec, nea_fuels, sector_estimates):
        """Collect metal mining data for the mining subsector"""

        factor = 0.01
        gross_output = bea_bls_output[['Metal ore mining']].rename(columns={'Metal ore mining': 'Metal Mining'})
        print('nea_elec.columns:', nea_elec.columns)
        print('nea_fuels.columns:', nea_fuels.columns)
        nea_elec = nea_elec.rename(columns={'Metal Ore Mining': 'Metal Mining'})
        nea_fuels = nea_fuels.rename(columns={'Metal Ore Mining': 'Metal Mining'})

        elec = nea_elec[['Metal Mining']] 
        fuels = nea_fuels[['Metal Mining']] 

        sector_estimates_elec = sector_estimates[0][['Metal Mining']]
        sector_estimates_fuels = sector_estimates[1][['Metal Mining']]
        data_dict = self.build_mining_output(factor, gross_output, elec, fuels, 
                                             sector_estimates_elec, sector_estimates_fuels, col_name='Metal Mining')
        return data_dict

    def nonmetallic_mineral_mining(self, bea_bls_output, nea_elec, nea_fuels, sector_estimates):
        """Collect nonmetallic mineral mining data for the mining subsector"""
        factor = 0.01
        col = 'Nonmetallic Mineral Mining'
        print('bea_bls_output:\n', bea_bls_output)

        print('bea_bls_output cols', bea_bls_output.columns)
        bea_bls_output = bea_bls_output.rename(columns={'Nonmetallic mineral mining and quarrying': col})
        print('bea_bls_output:\n', bea_bls_output)
        print('type bea_bls_output:\n', type(bea_bls_output))

        gross_output = bea_bls_output[[col]]
        print('nea_elec:', nea_elec.columns)
        nea_elec = nea_elec.rename(columns={'Nonmetallic mineral mining': col})
        nea_fuels = nea_fuels.rename(columns={'Nonmetallic mineral mining': col})

        elec = nea_elec[[col]] 
        print('nea_fuels:', nea_fuels.columns)

        fuels = nea_fuels[[col]]
        sector_estimates_elec = sector_estimates[0][['Nonmetallic Mining, excl Other Chem']].rename(columns={'Nonmetallic Mining, excl Other Chem': col})
        sector_estimates_fuels = sector_estimates[1][['Nonmetallic Mining, excl Other Chem']].rename(columns={'Nonmetallic Mining, excl Other Chem': col})
        data_dict = self.build_mining_output(factor, gross_output, elec, fuels, sector_estimates_elec, sector_estimates_fuels, col_name=col)
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
   
        other_mining_data = [m(bea_bls_output, nea_elec, nea_fuels, sector_estimates) for m in other_mining_types.values()]

        other_mining_elec = [m_df['energy']['elec'] for m_df in other_mining_data]
        elec = reduce(lambda x, y: x.add(y), other_mining_elec)

        other_mining_fuels = [m_df['energy']['fuels'] for m_df in other_mining_data]
        fuels = reduce(lambda x, y: x.add(y), other_mining_fuels)
        print('fuels:\n', fuels)
        print('elec:\n', elec)
        data_dict = {'energy': {'elec': elec, 'fuels': fuels}, 
                     'activity': {'gross_output': gross_output}}
        # data_dict = self.build_mining_output(factor, gross_output, elec, fuels, sector_estimates[0], sector_estimates[1], )
        return data_dict
    
    def drilling_and_mining_support(self, bea_bls_output, nea_elec, nea_fuels, sector_estimates):
        """Collect drilling and mining support data for the mining subsector"""
        factor = 0.001
        col = 'Support Activities'
        print('bea_bls_output cols', bea_bls_output.columns)
        bea_bls_output = bea_bls_output.rename(columns={'Support activities for mining': col})
        gross_output = bea_bls_output[[col]]
        print('nea_elec cols', nea_elec.columns)
        print('nea_fuels cols', nea_fuels.columns)

        elec = nea_elec[[col]] 
        fuels = nea_fuels[[col]] 

        sector_cols = ['Drilling Oil and Gas Wells', 'Support Activities for Oil and Gas', 'Support Activities for Coal Mining', 
                       'Support Activities for Metal Mining', 'Support Activities for Nonmetallic Minerals']

        sector_estimates_elec = sector_estimates[0]
        sector_estimates_elec[col] = sector_estimates_elec[sector_cols].sum(axis=1)
        sector_estimates_elec = sector_estimates_elec[[col]]

        sector_estimates_fuels = sector_estimates[1]
        sector_estimates_fuels[col] = sector_estimates_fuels[sector_cols].sum(axis=1)
        sector_estimates_fuels = sector_estimates_fuels[[col]]

        data_dict = self.build_mining_output(factor, gross_output, elec, fuels, 
                                             sector_estimates_elec, sector_estimates_fuels, 
                                             col_name=col)
        return data_dict

    @staticmethod
    def mining_fuels_adjust(ec_df):
        """
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

    def price_ratios(self, asm_prices, agricultural_petroleum_prices, 
                     stb0709, stb0608, stb0523):
        mecs_years = list(range(1977, self.currentYear + 1, 5))
        prices = []
        return prices

    @staticmethod
    def calculate_physical_units(current_cost, previous_cost, current_price, previous_pyhsical_units):
        calc = current_cost.divide(previous_cost * 
                            current_price).multiply(previous_pyhsical_units)
        return calc

    @staticmethod
    def aggregate_sector_estimates(sector_estimates):
        sector_estimates = sector_estimates[sector_estimates['NAICS'].notnull()].fillna(np.nan)
        sector_estimates = sector_estimates.dropna(axis=0, how='all')
        sector_estimates = sector_estimates.set_index('NAICS').drop('Description', axis=1)
        print('sector_estimates:\n', sector_estimates)

        sector_estimates.index = sector_estimates.index.astype(int)
        print('sector_estimates:\n', sector_estimates)
        print('sector_estimates index:\n', sector_estimates.index)

        sector_estimates.loc['Oil and Gas', :] = sector_estimates.loc[[211111, 211112], :].sum(axis=0)
        sector_estimates.loc['Coal Mining', :] = sector_estimates.loc[[212111, 212112, 212113]].sum(axis=0)
        sector_estimates.loc['Metal Mining', :] = sector_estimates.loc[[212210, 212221, 212222, 212231, 212234, \
                                                                                                            212291, 212299], :].sum(axis=0)
        sector_estimates.loc['Nonmetallic Mining, excl Other Chem', :] = sector_estimates.loc[[212311, 212312, \
                                                        212313, 212319, 212321, 212322, 212324, 212325, 212391, 212392, 212399], :].sum(axis=0)
        sector_estimates.loc['Other Chemical and Fertilizer Minerals', :] = sector_estimates.loc[212393, :]
        sector_estimates.loc['Drilling Oil and Gas Wells', :] = sector_estimates.loc[213111, :]
        sector_estimates.loc['Support Activities for Oil and Gas', :] = sector_estimates.loc[213112, :]
        sector_estimates.loc['Support Activities for Coal Mining', :] = sector_estimates.loc[213113, :]
        sector_estimates.loc['Support Activities for Metal Mining', :] = sector_estimates.loc[213114, :]
        sector_estimates.loc['Support Activities for Nonmetallic Minerals', :] = sector_estimates.loc[213115, :]
        sector_estimates_T = sector_estimates.transpose()
        cols = ['Oil and Gas', 'Coal Mining', 'Metal Mining', 'Nonmetallic Mining, excl Other Chem', 'Other Chemical and Fertilizer Minerals', \
                'Drilling Oil and Gas Wells', 'Support Activities for Oil and Gas', 'Support Activities for Coal Mining', \
                'Support Activities for Metal Mining', 'Support Activities for Nonmetallic Minerals']
        sector_estimates_T = sector_estimates_T[cols]
        print('sector_estimates_T :\n', sector_estimates_T)
        return sector_estimates_T

    @staticmethod
    def preliminary_sector_estimates():
        pass

    def mining_data_1987_2017(self):
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
        mining_1987_2017 = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/mining_sector_estimates_historical.csv') # from economic census
        # mining_1987_2017 = mining_1987_2017.apply(lambda column: self.price_ratios(column))

        sector_estimates_elec = mining_1987_2017[mining_1987_2017['Energy Type'] == 'Electricity'].drop('Energy Type', axis=1)
        sector_estimates_elec = self.aggregate_sector_estimates(sector_estimates_elec)

        sector_estimates_fuels = mining_1987_2017[mining_1987_2017['Energy Type'] == 'Fuels'].drop('Energy Type', axis=1)
        sector_estimates_fuels = self.aggregate_sector_estimates(sector_estimates_fuels)
        # preliminary_sector_estimates should create this method to automatically update for future years 
        return {'elec': sector_estimates_elec, 'fuels': sector_estimates_fuels}
    
    def mining_sector_estimates(self):
        """Calculate electricity and fuels sector estimates for mining"""
        data_1987_2017 = self.mining_data_1987_2017()
        elec = data_1987_2017['elec'].multiply(0.000001 * 3412)
        fuels = data_1987_2017['fuels']
        print('elec columns:', elec.columns)
        print('fuels columns:', fuels.columns)
        return elec, fuels
    
    def mining(self):
        """Collect mining data"""
        # Mining energy_031020.xlsx/Compute_intensities (FF-FN, FQ-FS)
        ALLFOS_historical = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/ALLFOS_historical.csv')
        ELECNEA_historical = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/ELECNEA_historical.csv')

        NEA_data_elec = self.aggregate_mining_data(ELECNEA_historical) 

        NEA_data_fuels = self.aggregate_mining_data(ALLFOS_historical, allfos=True) 
        va_quant_index, go_quant_index = self.BEA_data.chain_qty_indexes() # Historical and current data

        sector_estimates = self.mining_sector_estimates()
        print('sector estimates:\n', sector_estimates)
        BLS_data = pd.read_excel('./EnergyIntensityIndicators/Industry/Data/BLS_BEA_Data.xlsx', sheet_name='BLS_Data_011920', index_col=0)
        BLS_data.index.name = 'Industry'
        print('BLS_data:\n', BLS_data)
        BLS_data = BLS_data.transpose().drop('Oil and gas extraction', axis=1)
        print('BLS_data transpose:\n', BLS_data)
        print('go_quant_index columns:\n', go_quant_index.columns)
        BEA_mining_data = go_quant_index[['Oil and gas extraction', 'Mining, except oil and gas', 'Support activities for mining']]
        BEA_mining_data = BEA_mining_data.rename(columns={'Support activities for mining': 'BEA- Support Activities', 
                                                          'Mining, except oil and gas': 'Other Mining', 'Oil and gas extraction': 'Oil & Gas'})
        bea_bls_output = BEA_mining_data.merge(BLS_data, how='outer', left_index=True, right_index=True)

        print('bea_bls_output:\n', bea_bls_output)
        print("BEA_mining_data:\n", BEA_mining_data)

        data_dict = {'crude_petroleum_natgas': self.crude_petroleum_natgas(bea_bls_output, NEA_data_elec, NEA_data_fuels, sector_estimates),
                     'other_mining': self.other_mining(bea_bls_output, NEA_data_elec, NEA_data_fuels, sector_estimates), 
                     'drilling_and_mining_support': self.drilling_and_mining_support(bea_bls_output, NEA_data_elec, NEA_data_fuels, sector_estimates)}
        
        return data_dict

    def nonmanufacturing_data(self):
        """Collect all nonmanufacturing data
        """        
        # starting point: NonManufacturing_reconciliation_010420.xlsx
        # Agriculutral_energy_010420.xlsx/Intensity_estimates (Y-AB)
        # Mining energy_031020.xlsx/Compute_intensities (FQ-FS)
        # Construction_energy_011920.xlsx/Intensity_estimates (W-Z)
        data_dict = {'Mining': self.mining(), 'Agriculture': self.agriculture(), 'Construction': self.construction()}
        print('data_dict:', data_dict)
        return data_dict              

if __name__ == '__main__':
    print('main')
    data = NonManufacturing().get_econ_census()
    print(data)



# 'Mining', 'Mining, except oil and gas', 'Support activities for mining', 'Nonmetallic mineral products', 'Oil and gas extraction','Petroleum and coal products', 'Primary metals'
