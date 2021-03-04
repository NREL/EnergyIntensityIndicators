
import pandas as pd
import os
import numpy as np

# from EnergyIntensityIndicators.industry import IndustrialIndicators
# from EnergyIntensityIndicators.residential import ResidentialIndicators
# from EnergyIntensityIndicators.commercial import CommercialIndicators
from EnergyIntensityIndicators.electricity import ElectricityIndicators
# from EnergyIntensityIndicators.transportation import TransportationIndicators
# from EnergyIntensityIndicators.LMDI import CalculateLMDI
from EnergyIntensityIndicators.economy_wide import EconomyWide
from EnergyIntensityIndicators.pull_eia_api import GetEIAData
from EnergyIntensityIndicators.utilites import dataframe_utilities as df_utils


class CO2EmissionsDecomposition: #(EconomyWide):
    """Class to decompose CO2 emissions by sector of the U.S. economy. 

    LMDI aspects: 
    - total activity (Q), 
    - activity share for subsector i (structure) (=Q_i/Q), 
    - energy intensity for subsector i (=E_i/Q_i), 
    - energy share of type j in subsector i (=E_ij/E_i), also called fuel mix
    - emissions rate of energy type j and subsector i (=C_ij/E_ij), also called emissions coefficient

    (for time series, all variables have t subscripts (i.e. no constants-- constant emissions 
    rates cancel out))
    """
    def __init__(self, directory, output_directory, level_of_aggregation=None, 
                 lmdi_model='multiplicative', base_year=1985, end_year=2018): 
        
        self.eia = GetEIAData('emissions')
        # super().__init__(directory=directory, output_directory=output_directory, 
        #                  level_of_aggregation=level_of_aggregation, 
        #                  lmdi_model=lmdi_model, base_year=base_year, end_year=end_year)

    @staticmethod
    def state_census_crosswalk():
        cw = pd.read_csv('./EnergyIntensityIndicators/Data/state_to_census_region.csv')
        state_abbrevs = pd.read_csv('./EnergyIntensityIndicators/Data/name-abbr.csv')
        cw =cw.merge(state_abbrevs, left_on='USPC', right_on='Abbrev', how='left')
        return cw

    def collect_elec_emissions_factors(self, sector=None, energy_type=None, region=None, 
                                       fuel_type=None, state_abbrev=None):
        """Collect electricity emissions factors from the EIA API (through GetEIAData). 
        If region is None, collect data for the U.S., if energy_type is None use total, 
        if sector is None use total

        Parameters:
            sector (str):
            energy_type (str): 
            region (str): 

        Returns: 
            emissions_factor (df, series or float):
        """
        eia_co2_emiss = f'EMISS.CO2-TOTV-{sector}-{fuel_type}-{state_abbrev}.A'
        data = self.eia.eia_api(id_=eia_co2_emiss, id_type='series') # , new_name='')

        return data

    @staticmethod
    def epa_eia_crosswalk(eia_data):
        """[summary]

        Args:
            eia_data ([type]): [description]

        Returns:
            [type]: [description]

        TODO: 
            - Handle fuel types with multiple factors
        """
        mapping_ = {'Coal': 'Mixed (Commercial Sector)', # what about residential?? 
                    # 'Distillate Fuel Oil': ['Distillate Fuel Oil No. 1', 
                    #                         'Distillate Fuel Oil No. 2', 
                    #                         'Distillate Fuel Oil No. 4'],
                    'Fuel Ethanol including Denaturant': 'Ethanol (100%)',  # is this the correct handling of the two ethanol categories?
                    # 'Fuel Ethanol excluding Denaturant':,
                    'Hydrocarbon gas liquids': 'Liquefied Petroleum Gases (LPG)', 
                    'Kerosene': 'Kerosene',
                    'Motor Gasoline': 'Motor Gasoline',
                    'Natural Gas including Supplemental Gaseous Fuels': 'Natural Gas',
                    'Petroleum Coke': 'Petroleum Coke',
                    'Propane': 'Propane',
                    # 'Residual Fuel Oil': ['Residual Fuel Oil No. 5', 
                    #                       'Residual Fuel Oil No. 6'],
                    'Waste': 'Municipal Solid Waste',
                    'Wood': 'Wood and Wood Residuals'} #,
                    # 'Wood and Waste': ['Wood and Wood Residuals', 
                    #                    'Municipal Solid Waste']}

        irrelevant_fuels = [f for f in eia_data.columns if f not in mapping_.keys() and f != 'Census Region']
        eia_data = eia_data.drop(irrelevant_fuels, axis=1)

        eia_data = eia_data.rename(columns=mapping_)

        return eia_data

    @staticmethod
    def seds_endpoints(sector, state, fuel):
        endpoints = {'All Petroleum Products': f'SEDS.PA{sector}P.{state}.A',
                    'Coal': f'SEDS.CL{sector}P.{state}.A', 
                    'Distillate Fuel Oil': f'SEDS.DF{sector}P.{state}.A',
                    'Electrical System Energy Losses': f'SEDS.LO{sector}B.{state}.A', 
                    'Electricity Sales': f'SEDS.ES{sector}P.{state}.A',
                    'Fuel Ethanol including Denaturant': f'SEDS.EN{sector}P.{state}.A',
                    'Fuel Ethanol excluding Denaturant': f'SEDS.EM{sector}B.{state}.A',
                    'Geothermal': f'SEDS.GE{sector}B.{state}.A',
                    'Hydrocarbon gas liquids': f'SEDS.HL{sector}P.{state}.A', # is the Lpg factor right for HGL?
                    'Hydroelectricity': f'SEDS.HY{sector}P.{state}.A',
                    'Kerosene': f'SEDS.KS{sector}P.{state}.A',
                    'Motor Gasoline': f'SEDS.MG{sector}P.{state}.A',
                    'Natural Gas including Supplemental Gaseous Fuels': f'SEDS.NG{sector}P.{state}.A',
                    'Petroleum Coke': f'SEDS.PC{sector}P.{state}.A',
                    'Propane': f'SEDS.PQ{sector}P.{state}.A',
                    'Residual Fuel Oil': f'SEDS.RF{sector}P.{state}.A',
                    'Solar Energy': f'SEDS.SOR7P.{state}.A',
                    'Total (per Capita)': f'SEDS.TE{sector[0]}PB.{state}.A',
                    'Total Energy excluding Electrical System Energy Losses': f'SEDS.TN{sector}B.{state}.A',
                    'Waste': f'SEDS.WS{sector}B.{state}.A',
                    'Wind Energy': f'SEDS.WY{sector}P.{state}.A',
                    'Wood': f'SEDS.WD{sector}B.{state}.A',
                    'Wood and Waste': f'SEDS.WW{sector}B.{state}.A'}
        return endpoints[fuel]
            
    def collect_seds(self, sector, states):
        """SEDS energy consumption data (in physical units unless unavailable, in which case in Btu-- indicated by P or B in endpoint)

        Args:
            sector ([type]): [description]
        """

        fuels = {'CC': ['All Petroleum Products',
                            'Coal', 
                            'Distillate Fuel Oil',
                            'Electrical System Energy Losses', 
                            'Electricity Sales',
                            'Fuel Ethanol including Denaturant',
                            'Fuel Ethanol excluding Denaturant',
                            'Geothermal',
                            'Hydrocarbon gas liquids', # is the Lpg factor right for HGL?
                            'Hydroelectricity',
                            'Kerosene',
                            'Motor Gasoline',
                            'Natural Gas including Supplemental Gaseous Fuels',
                            'Petroleum Coke',
                            'Propane',
                            'Residual Fuel Oil',
                            'Solar Energy',
                            'Total (per Capita)',
                            'Total Energy excluding Electrical System Energy Losses',
                            'Waste',
                            'Wind Energy',
                            'Wood',
                            'Wood and Waste'],
        'RC': ['All Petroleum Products',
                            'Coal', 
                            'Distillate Fuel Oil',
                            'Electrical System Energy Losses', # in BTU
                            'Electricity Sales',
                            'Geothermal',
                            'Hydrocarbon gas liquids',
                            'Kerosene',
                            'Natural Gas including Supplemental Gaseous Fuels',
                            'Propane',
                            'Solar Energy',
                            'Total (per Capita)',
                            'Total Energy excluding Electrical System Energy Losses',
                            'Wood']}
        fuels_data = []
        for f in fuels[sector]:
            state_data = []
            for s in states:
                try:
                    df = self.eia.eia_api(id_=self.seds_endpoints(sector, s, f), id_type='series', new_name=f, units_col=True)
                    state_data.append(df)
                except KeyError:
                    print(f'Endpoint failed for state {s}, sector {sector} and fuel type {f}')
                    continue

            region_data = pd.concat(state_data, axis=0)
            region_data = region_data.reset_index()
            region_data = region_data.groupby('Year').sum()
            fuels_data.append(region_data)

        fuels_data = df_utils.merge_df_list(fuels_data)
        return fuels_data
    
    def seds_energy_data(self, sector):
        states = self.state_census_crosswalk()
        sector_data = {'commercial': {'abbrev': 'CC', 'regions': [0]}, 
                       'residential': {'abbrev': 'RC', 'regions': [1, 2, 3, 4]}}
        grouped = states.groupby(states['Census Region'])
        all_data = []
        for g in sector_data[sector]['regions']:
            region_states = grouped.get_group(g)
            region_data = self.collect_seds(sector=sector_data[sector]['abbrev'], states=region_states['USPC'].unique())
            region_data['Census Region'] = g
            all_data.append(region_data)
        all_data = pd.concat(all_data, axis=0)
        return all_data

    @staticmethod
    def get_fuel_mix(region_data):
        region_data = region_data.drop('Census Region', axis=1, errors='ignore')
        region_data = df_utils.create_total_column(region_data, total_label='total')
        fuel_mix = df_utils.calculate_shares(region_data, total_label='total')
        return fuel_mix

    def calculate_emissions(self, energy_data, emissions_type='CO2 Factor', 
                            datasource='SEDS'):
        """Calculate emissions from the product of energy_data and 
        emissions_factor

        Parameters:
            energy_data (df): 
            emission_factor (df, series or float): 
        
        Returns: 
            emissions_data (df): 
        """
        emissions_factors = self.epa_emissions_data()

        if datasource == 'SEDS':
            energy_data = self.epa_eia_crosswalk(energy_data)
        elif datasource == 'MECS':
            energy_data = self.mecs_epa_mapping(energy_data)
            energy_data = energy_data.reset_index()
        elif datasource == 'eia_elec':
            pass
        
        print('energy_data:\n', energy_data)
        emissions_factors = self.get_factor(emissions_factors, 
                                            emissions_type)
        emissions_factors = emissions_factors.loc[energy_data.columns.tolist()]
        print('emissions_factors:\n', emissions_factors)

        emissions_data = energy_data.multiply(emissions_factors, axis='columns')
        emissions_data['Census Region'] = emissions_data['Census Region'].astype(int).astype(str)
        print('emissions_data:\n', emissions_data)

        energy_data['Census Region'] = energy_data['Census Region'].astype(int).astype(str)
        fuel_mix = indicators.get_fuel_mix(energy_data)
        print(fuel_mix)
        return emissions_data

    @staticmethod
    def epa_emissions_data():
        ef = pd.read_csv('./EnergyIntensityIndicators/Data/EPA_emissions_factors.csv')
        df_cols = ef.columns
        dfs = []
        grouped = ef.groupby(ef['Unit Type'])
        for g in ef['Unit Type'].unique():
            unit_data = grouped.get_group(g)
            unit_data.columns = unit_data.iloc[0]
            
            units_dict = dict(zip(unit_data.columns, df_cols))
            unit_data = unit_data.drop(g, axis=1)

            unit_data = unit_data.drop(unit_data.index[0])
            unit_data = unit_data.melt(id_vars=['Units', 'Fuel Type'], var_name='Unit')

            unit_data = unit_data.rename(columns={'Units': 'Category'})
            
            unit_data['Variable'] = unit_data['Unit'].map(units_dict)

            dfs.append(unit_data)
        emissions_factors = pd.concat(dfs, axis=0)

        return emissions_factors

    @staticmethod
    def get_factor(factors_df, emissions_type):
        """Lookup emissions factor for given fuel and emissions
        type

        Args:
            factors_df (df): EPA emisions hub data
            fuel_name (str): Fuel type to look up in factors_df
            emissions_type (str): Type of emissions to return (e.g CO2)

        Returns:
            emissions_factor (float): emissions factor for given params
        """
        factors_df = factors_df[factors_df['Variable'] == emissions_type]
        fuel_factor_df = factors_df[['Fuel Type', 'value']]
        fuel_factor_df['value'] = fuel_factor_df['value'].astype(float)
        new_row = {'Fuel Type': 'Census Region', 'value': 1}
        fuel_factor_df = fuel_factor_df.append(new_row, ignore_index=True)
        fuel_factor_df = fuel_factor_df.set_index('Fuel Type')
        fuel_factor_df = fuel_factor_df['value']        
        return fuel_factor_df

    def collect_emissions_data(self): 
        """[summary]

        Parameters:

        Returns: 
            emissions_data_dict (dict): Nested dictionary of all_data from EconomyWide 
                                        with energy data replaced with emissions data
                                         (with original dictionary keys remaining intact)
        TODO: 
            - Break sector_level_data into lowest levels of dict in order
              to manipulate energy data
            - replace energy_data with emissions_data in nested dictionary
        """
        all_data = self.collect_data() # This is currently dictionary of 
                                       # all data collected in EconomyWide.collect_data()
        
        state_abbrev = 'AK' # Alaska
        sectors = {'residential': 'RC', 
                   'commercial': 'CC', 
                   'industrial': 'IC', 
                   'transportation': 'TC',
                   'electric_power': 'EC'}
        fuels = {'coal': 'CO', 'natural_gas': 'NG', 'petroleum': 'PE'}
        children_fuels = {'coal': {'industrial_coking': '',
                                   'coal (electricity utility)': '',
                                   'industrial other': '', 
                                   'residential': ''}, 
                          'natural_gas': {'natural gas (pipeline)': ''}, 
                          'petroleum': {'distillate fuel': '', 
                                        'Lpg (fuel use)': '', 
                                        'kerosene': '', 
                                        'motor gasoline': '',
                                        'residual fuel': '',
                                        'petroleum coke': '', 
                                        'residual fuel': '',
                                        'Asphalt and road oil': '', 
                                        'lubricants': '',
                                        'weighted coeff for other pet': '',
                                        'aviation gasoline': '', 
                                        'jet fuel': ''}}

        # for sector in all_data.keys():
        #     sector_level_data = all_data[sector]

        #     # sector_level_data is a complex nested dictionary, 
        #     # infrastructure to handle this is contained in CalculateLMDI
        #     energy_data = 
        #     energy_type = 
        #     region = 
        #     emission_factor = self.collect_emissions_factors(sector=sector, 
        #                                                      energy_type=energy_type, 
        #                                                      region=region)

        #     emissions_data = self.calculate_emission(energy_data, emission_factor)

        #     # replace energy_data in nested dictionary with emissions_data
        
        emissions_data_dict = {}
        return emissions_data_dict
    
    @staticmethod
    def electric_power_sector():
        mapping_ = {'Coal': 'Mixed (Electric Power Sector)', # for industrial/commercial use 'Mixed (Commercial Sector)', 'Mixed (Industrial Coking)' or 'Mixed (Industrial Sector)'??
                    'Petroleum': '',
                    'Natural Gas': 'Natural Gas', 
                    'Other Gases': '',
                    'Waste': 'Municipal Solid Waste',
                    'Wood': 'Wood and Wood Residuals'}
        return mapping_
        
    def electric_power_co2(self):
        elec_data = ElectricityIndicators.collect_data()
        elec_gen_total = elec_data['Elec Generation Total']
        elec_power_sector = elec_gen_total['Elec Power Sector']
        elec_only = elec_power_sector['Electricity Only']

        comm_sector = elec_gen_total['Commercial Sector']
        ind_sector = elec_gen_total['Industrial Sector']

        all_chp = elec_data['All CHP']
        chp_elec_power = all_chp['Elec Power Sector']['Combined Heat & Power']
        chp_comm = all_chp['Commercial Sector']['Combined Heat & Power']
        chp_ind = all_chp['Industrial Sector']['Combined Heat & Power']

        data_cats = [elec_only, comm_sector, 
                     ind_sector, chp_elec_power, 
                     chp_comm, chp_ind]
        
        for d in data_cats:
            d = d.rename(columns=self.electric_power_sector())
            d_emissions = self.calculate_emissions(d, emissions_type='CO2 Factor', 
                                                   datasource='eia_elec')
        return elec_data
        

    @staticmethod
    def clean_industrial_data(raw_data, table_3_1=False):
        if table_3_1:
            raw_data.index = raw_data.index.str.strip()
        else:
            raw_data.index = raw_data.index.fillna(np.nan)
            raw_data.index = raw_data.index.astype('Int64')
            raw_data.index = raw_data.index.astype(str)
            raw_data.index.name = 'NAICS'   

        raw_data = raw_data.reset_index()
        regions = ['Total United States', 'Northeast Census Region', 'Midwest Census Region', 'South Census Region', 'West Census Region']
        conditions = [(raw_data['Total'] == r) for r in regions]
        raw_data['region'] = np.select(conditions, regions)
        raw_data['region'] = raw_data['region'].replace(to_replace='0', value=np.nan).fillna(method='ffill')
        raw_data = raw_data[~raw_data['Total'].isin(regions)]

        raw_data['NAICS'] = raw_data['NAICS'].fillna(raw_data['Subsector and Industry'])
        raw_data = raw_data.set_index(['region', 'NAICS', 'Subsector and Industry'])
        raw_data = raw_data.replace(to_replace=['*', 'Q', 'D'], value=np.nan)
        return raw_data

    def industrial_sector_data(self):
        mecs_3_1 = pd.read_excel('https://www.eia.gov/consumption/manufacturing/data/2018/xls/Table3_1.xlsx', skiprows=9, index_col=0).dropna(axis=0, how='all') # By Manufacturing Industry and Region (physical units)
        mecs_3_1 = mecs_3_1.drop('Code(a)', axis=0)
        mecs_3_1 = mecs_3_1.rename(columns={' ': 'Subsector and Industry'})
        mecs_3_1 = self.clean_industrial_data(mecs_3_1, table_3_1=True)


        mecs_3_2 = pd.read_excel('https://www.eia.gov/consumption/manufacturing/data/2018/xls/Table3_2.xlsx', skiprows=10, index_col=0).dropna(axis=0, how='all') # By Manufacturing Industry and Region (trillion Btu)
        mecs_3_2 = self.clean_industrial_data(mecs_3_2)
        rename_dict_3_1 = {'Electricity(b)': 'Net Electricity', 
                       'Fuel Oil': 'Residual Fuel Oil', 
                       'Fuel Oil(c)': 'Distillate Fuel Oil', 
                       'Gas(d)': 'Natural Gas',
                       'natural gasoline)(e)': 'HGL (excluding natural gasoline)', 
                       'and Breeze': 'Coke Coal and Breeze'}
        mecs_3_2 = mecs_3_2.rename(columns=rename_dict_3_1)

        mecs_3_5 = pd.read_excel('https://www.eia.gov/consumption/manufacturing/data/2018/xls/Table3_5.xlsx', skiprows=10, index_col=0).dropna(axis=0, how='all') # Byproducts in Fuel Consumption By Manufacturing Industry and Region
                    # (trillion Btu)
        rename_dict_3_5 = {'Oven Gases': 'Blast Furnace/Coke Oven Gases', 
                           'Gas': 'Waste Gas', 
                           'Coke': 'Petroleum Coke',
                           'Black Liquor': 'Pulping Liquor or Black Liquor', 
                           'Bark': 'Wood Chips, Bark', 
                           'Materials': 'Waste Oils/Tars and Waste Materials'}
        mecs_3_5 = mecs_3_5.rename(columns=rename_dict_3_5)
        mecs_3_5 = self.clean_industrial_data(mecs_3_5)

        mecs_3_2_other = mecs_3_2[['Other(f)']]

        mecs_3_5 = mecs_3_5.merge(mecs_3_2_other, left_index=True, right_index=True, how='inner')

        mecs_3_5['steam'] = mecs_3_5['Total'].subtract(mecs_3_5['Other(f)'])
        mecs_3_5 = mecs_3_5.drop(['Total', 'Other(f)'], axis=1)

        industrial_btu = mecs_3_5.merge(mecs_3_2, left_index=True, right_index=True, how='outer')
        print('industrial_btu:\n', industrial_btu)
        print('industrial_btu cols:\n', industrial_btu.columns)
        return industrial_btu

    def industrial_sector_energy(self):
        """TODO: do further processing to bridge Btu energy data with 
        physical units used for emissions factors
        """        
        industrial_data_btu = self.industrial_sector_data() # This is not in physical units!!
        industrial_renamed = self.mecs_epa_mapping(industrial_data_btu) 
        return industrial_renamed
    
    @staticmethod
    def mecs_epa_mapping(mecs_data):
        mapping_ = {
            # 'Blast Furnace/Coke Oven Gases': ['Blast Furnace Gas', 'Coke Oven Gas'], 
                    # 'Waste Gas', ??
                    'Petroleum Coke': 'Petroleum Coke',
                    # 'Pulping Liquor or Black Liquor', ??
                    'Wood Chips, Bark': 'Wood and Wood Residuals',
                    # 'Waste Oils/Tars and Waste Materials', ??
                    # 'steam', ?? 
                    # 'Net Electricity', ??
                    # 'Residual Fuel Oil': ['Residual Fuel Oil No. 5', 'Residual Fuel Oil No. 6'], 
                    # 'Distillate Fuel Oil': ['Distillate Fuel Oil No. 1', 
                    #                         'Distillate Fuel Oil No. 2', 
                    #                         'Distillate Fuel Oil No. 4'],
                    'Natural Gas': 'Natural Gas', 
                    'HGL (excluding natural gasoline)': 'Liquefied Petroleum Gases (LPG)', 
                    'Coal': 'Mixed (Industrial Sector)', # OR Mixed (Industrial Coking)?
                    'Coke Coal and Breeze': 'Coal Coke'}
        mecs_data = mecs_data.rename(columns=mapping_)
        return mecs_data

    def main(self, breakout, calculate_lmdi):
        """Calculate decomposition of CO2 emissions for the U.S. economy
        
        TODO: allow for different sectors to have different types of energy 
              and commercial and residential to have weather adjustment 
              (TODO carried over from EconomyWide)

        """        
        data_dict = self.collect_emissions_data()
        results_dict, formatted_results = self.get_nested_lmdi(level_of_aggregation=self.level_of_aggregation, 
                                                               breakout=breakout, calculate_lmdi=calculate_lmdi, 
                                                               raw_data=data_dict, lmdi_type='LMDI-I')
        return results_dict 



class EmissionsComparison(CO2EmissionsDecomposition):
    """Class to visualize the difference between emissions
    values calculated from energy data and emissions factors
    vs emissions values given by the EIA API
    """

    def __init__(self):
        pass

    def get_eia_emissions(self, sector=None, energy_type=None, region=None):
        """Collect emissions data from the EIA API (through GetEIAData). 
        If region is None, collect data for the U.S., if energy_type is None use total, 
        if sector is None use total

        Parameters:
            sector (str):
            energy_type (str): 
            region (str): 

        Returns: 
            emissions_factor (df, series or float):
        """
        pass

    def compare_values(self):
        # sector = 
        # energy_type = 
        # region = 
        # eia_data = self.get_eia_emissions(sector, energy_type, region)
        # calc_data = # unclear how to extract these values from the nested dictionary
        # pct_diff  = # extract perecent difference calculation currently in LMDITest 
        #             # (should be moved to utilities)
        pass


if __name__ == '__main__':
    indicators = CO2EmissionsDecomposition(directory='./EnergyIntensityIndicators/Data', 
                                           output_directory='./Results', level_of_aggregation=None, 
                                           end_year=2018, lmdi_model=['multiplicative', 'additive'])
    # indicators.main(breakout=True, calculate_lmdi=True)
    # ef = indicators.epa_emissions_data()

    # data = indicators.seds_energy_data(sector='residential')
    # print(data)
    # print(data.columns)
    # emissions_data = indicators.calculate_emissions(data, emissions_type='CO2 Factor', 
    #                                                 datasource='SEDS')

    # print('emissions_data:\n', emissions_data)
    mecs_data = indicators.industrial_sector_energy()
    industrial_emissions_data = indicators.calculate_emissions(mecs_data, emissions_type='CO2 Factor', 
                                                               datasource='MECS')
    print('industrial_emissions_data:\n', industrial_emissions_data)