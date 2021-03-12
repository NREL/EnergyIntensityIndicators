
import pandas as pd
import os
import numpy as np

# from EnergyIntensityIndicators.industry import IndustrialIndicators
# from EnergyIntensityIndicators.residential import ResidentialIndicators
# from EnergyIntensityIndicators.commercial import CommercialIndicators
from EnergyIntensityIndicators.electricity import ElectricityIndicators
# from EnergyIntensityIndicators.transportation import TransportationIndicators
# from EnergyIntensityIndicators.LMDI import CalculateLMDI
# from EnergyIntensityIndicators.economy_wide import EconomyWide
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
        # super().__init__(directory=directory, 
        #                  output_directory=output_directory, 
        #                  level_of_aggregation=level_of_aggregation, 
        #                  lmdi_model=lmdi_model, base_year=base_year, 
        #                  end_year=end_year)

    @staticmethod
    def state_census_crosswalk():
        cw = pd.read_csv('./EnergyIntensityIndicators/Data/state_to_census_region.csv')
        state_abbrevs = pd.read_csv('./EnergyIntensityIndicators/Data/name-abbr.csv')
        cw = cw.merge(state_abbrevs, left_on='USPC', right_on='Abbrev', 
                     how='left')
        return cw

    def collect_elec_emissions_factors(self, sector=None, energy_type=None,
                                       region=None, fuel_type=None, 
                                       state_abbrev=None):
        """Collect electricity emissions factors from the EIA API (through 
        GetEIAData). If region is None, collect data for the U.S., 
        if energy_type is None use total, if sector is None use total

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

        irrelevant_fuels = [f for f in eia_data.columns if f not in
                            mapping_.keys() and f != 'Census Region']
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
                    df = self.eia.eia_api(id_=self.seds_endpoints(sector, s, f),
                                          id_type='series', new_name=f, 
                                          units_col=True)
                    state_data.append(df)
                except KeyError:
                    print(f'Endpoint failed for state {s}, sector \
                            {sector} and fuel type {f}')
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
            region_data = self.collect_seds(
                                        sector=sector_data[sector]['abbrev'],
                                        states=region_states['USPC'].unique())
            region_data['Census Region'] = g
            all_data.append(region_data)
        all_data = pd.concat(all_data, axis=0)
        return all_data

    @staticmethod
    def get_fuel_mix(region_data):
        region_data = region_data.drop('Census Region',
                                       axis=1, errors='ignore')
        region_data = df_utils.create_total_column(region_data,
                                                   total_label='total')
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
        elif datasource == 'TEDB':
            energy_data = self.tedb_mecs_mapping(energy_data)
        
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
        print('emissions_factors:\n', emissions_factors)
        exit()
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
    def transportation_data():
        tedb_18 = pd.read_excel("https://tedb.ornl.gov/wp-content/uploads/2021/02/Table2_07_01312021.xlsx", skiprows=9, skipfooter=10, index_col=0, usecols='B:J')
        tedb_18 = tedb_18.rename(columns={'Electricityb': 'Electricity', 'Totalc': 'Total'})
        tedb_18.index = tedb_18.index.str.strip()
        tedb_18 = tedb_18.reset_index()
        print(tedb_18)
        categories = ['HIGHWAY', 'TOTAL HWY & NONHWYc', 'Air', 'Rail', 'Pipeline', 'Water']  # 'NONHIGHWAY', 
        conditions = [(tedb_18['index'] == r) for r in categories]
        tedb_18['Category'] = np.select(conditions, categories)
        tedb_18['Category'] = tedb_18['Category'].replace(to_replace='0', value=np.nan).fillna(method='ffill')
        tedb_18 = tedb_18[~tedb_18['index'].isin(categories)]
        tedb_18 = tedb_18.rename(columns={'index': 'Mode', ' Residual fuel oil': 'Residual fuel oil'})
        print(tedb_18.columns)
        tedb_fuel_types = ['Gasoline', 'Diesel fuel', 'Liquefied petroleum gas',
                           'Jet fuel',  'Residual fuel oil', 'Natural gas',
                           'Electricity', 'Total']

        tedb_fuel = pd.melt(tedb_18, id_vars=['Category', 'Mode'], value_vars=tedb_fuel_types, var_name='Fuel Type')
        tedb_fuel['Year'] = 2018
        tedb_fuel = tedb_fuel.replace({'HIGHWAY': 'Highway', 'Water': 'Waterborne'})
        print(tedb_fuel)
        historical_fuel_consump = pd.read_excel('./EnergyIntensityIndicators/Transportation/Data/FuelConsump.xlsx', 
                                                skipfooter=196, skiprows=2, usecols='A:BQ')
        historical_fuel_consump = historical_fuel_consump.fillna(np.nan)
        historical_fuel_consump.loc[0:2, :] = historical_fuel_consump.loc[0:2, :].ffill(axis=1)
        historical_fuel_consump.loc[0, 'Unnamed: 0'] = 'Category'
        historical_fuel_consump.loc[1, 'Unnamed: 0'] = 'Mode'

        historical_fuel_consump = historical_fuel_consump.set_index('Unnamed: 0')

        historical_fuel_consump = historical_fuel_consump.transpose()
        historical_fuel_consump = historical_fuel_consump.rename(columns={'Year': 'Fuel Type'})
        historical_fuel_consump = historical_fuel_consump.reset_index().drop('index', axis=1)

        year_cols = [c for c in historical_fuel_consump.columns if isinstance(c, int)]
        fuel = pd.melt(historical_fuel_consump, id_vars=['Category', 'Mode', 'Fuel Type'], value_vars=year_cols)
        fuel = fuel.rename(columns={'Unnamed: 0': 'Year'})
        fuel = fuel[(fuel['Fuel Type']!='Year') & (fuel['Mode']!='Not Used')]

        print('fuel:\n', fuel)

        print('fuel cols:\n', fuel.columns)
        transport_fuel = pd.concat([fuel, tedb_fuel], axis=0)
        print('transport_fuel:\n', transport_fuel)
        print('transport_fuel fuels', transport_fuel['Fuel Type'].unique())

        return transport_fuel



    @staticmethod
    def clean_industrial_data(raw_data, table_3_1=False, sic=False):

        if sic:
            code = 'SIC'
        else:
            code = 'NAICS'

        if table_3_1:
            raw_data.index = raw_data.index.str.strip()

        else:
            raw_data.index = raw_data.index.fillna(np.nan)
            raw_data.index = raw_data.index.str.strip()
            raw_data.index.name = code


        raw_data = raw_data.reset_index()
        regions = ['Total United States', 'Northeast Census Region', 'Midwest Census Region', 'South Census Region', 'West Census Region']
        conditions = [(raw_data['Total'] == r) for r in regions]
        raw_data['region'] = np.select(conditions, regions)
        raw_data['region'] = raw_data['region'].replace(to_replace='0', value=np.nan).fillna(method='ffill')
        raw_data = raw_data[~raw_data['Total'].isin(regions)]

        raw_data[code] = raw_data[code].fillna(raw_data['Subsector and Industry'])
        raw_data = raw_data.set_index(['region', code, 'Subsector and Industry'])
        raw_data = raw_data.replace({'*': 0.25, 'Q': np.nan, 'D': np.nan}, value=np.nan)
        return raw_data

    def mecs_data_by_year(self):
        # Energy Consumption as a Fuel
        # Table 3.1 : By Mfg. Industry & Region (physical units)
        # Table 3.2 : By Mfg. Industry & Region (trillion Btu)
        # Table 3.5 : Byproducts in Fuel Consumption by Mfg. Industry & Region (trillion Btu)
        mecs_data = {2018: 
                        {'table_3_1': {'endpoint': 'Table3_1.xlsx', 'skiprows': 9, 'skip_footer': 20}, 
                         'table_3_2': {'endpoint': 'Table3_2.xlsx', 'skiprows': 9, 'skip_footer': 14}, 
                         'table_3_5': {'endpoint': 'Table3_5.xlsx', 'skiprows': 9, 'skip_footer': 12},
                         'table_4_2': {'endpoint': 'Table4_2.xlsx'}},
                    2014: 
                        {'table_3_1': {'endpoint': 'table3_1.xlsx', 'skiprows': 9, 'skip_footer': 20}, 
                         'table_3_2': {'endpoint': 'table3_2.xlsx', 'skiprows': 9, 'skip_footer': 14}, 
                         'table_3_5': {'endpoint': 'table3_5.xlsx', 'skiprows': 9, 'skip_footer': 12},
                         'table_4_2': {'endpoint': 'table4_2.xlsx'}},
                    2010: 
                        {'table_3_1': {'endpoint': 'Table3_1.xls', 'skiprows': 9, 'skip_footer': 47}, 
                         'table_3_2': {'endpoint': 'Table3_2.xls', 'skiprows': 8, 'skip_footer': 47}, 
                         'table_3_5': {'endpoint': 'Table3_5.xls', 'skiprows': 9, 'skip_footer': 29},
                         'table_4_2': {'endpoint': 'Table4_2.xls'}},
                    2006: 
                        {'table_3_1': {'endpoint': 'Table3_1.xls', 'skiprows': 10, 'skip_footer': 49}, 
                         'table_3_2': {'endpoint': 'Table3_2.xls', 'skiprows': 9, 'skip_footer': 49}, 
                         'table_3_5': {'endpoint': 'Table3_5.xls', 'skiprows': 10, 'skip_footer': 31},
                         'table_4_2': {'endpoint': 'Table4_2.xls'}},
                    2002: 
                        {'table_3_1': {'endpoint': 'Table3.1_02.xls', 'skiprows': 7, 'skip_footer': 49}, 
                         'table_3_2': {'endpoint': 'Table3.2_02.xls', 'skiprows': 6, 'skip_footer': 49}, 
                         'table_3_5': {'endpoint': 'Table3.5_02.xls', 'skiprows': 7, 'skip_footer': 55},
                         'table_4_2': {'endpoint': 'Table4.2_02.xls'}},
                    1998: 
                        {'table_3_1': {'endpoint': 'd98n3_1.xls', 'skiprows': 7, 'skip_footer': 53},  # Fuel Consumption, 1998; Level: National and Regional Data; Row: NAICS Codes; Column: Energy Sources; Unit: Physical Units or Btu
                         'table_3_2': {'endpoint': 'd98n3_2.xls', 'skiprows': 6, 'skip_footer': 53},  # Fuel Consumption, 1998; Level: National and Regional Data; Row: NAICS Codes; Column: Energy Sources; Unit: Trillion Btu
                         'table_3_5': {'endpoint': 'd98n5_1.xls', 'skiprows': 7, 'skip_footer': 59},  # Selected Byproducts in Fuel Consumption, 1998; Level: National Data and Regional Totals; Row: NAICS Codes; Column: Energy Sources; Unit: Trillion Btu
                         'table_4_2': {'endpoint': 'd98n4_2.xls'}}, 
                    1994: 
                        {'table_3_1': {'endpoint': 'm94_04a.xls', 'skiprows': 6, 'skip_footer': 34}, 
                         'table_3_2': {'endpoint': 'm94_04b.xls', 'skiprows': 5, 'skip_footer': 34}, 
                         'table_3_5': {'endpoint': 'm94_06.xls', 'skiprows': 7, 'skip_footer': 21},
                         'table_4_2': {'endpoint': 'm94_05b.xls'}},
                    1991: 
                        {'table_3_1': {'endpoint': 'mecs04a.xls', 'skiprows': 6, 'skip_footer': 36}, 
                         'table_3_2': {'endpoint': 'mecs04b.xls', 'skiprows': 5, 'skip_footer': 38}, 
                         'table_3_5': None,
                         'table_4_2': {'endpoint': 'mecs05b.xls'}},
                    1985: 
                        {'table_3_1': None, 
                         'table_3_2': None, 
                         'table_3_5': None},
                         'table_4_2': None}

        all_3_1 = []
        all_3_2 = []
        all_3_5 = []
        all_4_2 = []

        sic_3_1 = []
        sic_3_2 = []
        sic_3_5 = []
        sic_4_2 = []

        for year, table_dict in mecs_data.items():
            if year < 1998: 
                sic = True
            else:
                sic = False

            for t, t_dict in table_dict.items():
                if t_dict:
                    endpoint = t_dict['endpoint'] 
                    general_url = f'https://www.eia.gov/consumption/manufacturing/data/{year}/xls/{endpoint}'
                    general_df = pd.read_excel(general_url, index_col=0)
                    col_labels = general_df.loc[:'Code(a)'].tail(4)

                    index_label = general_df.iloc[general_df.index.get_loc('Code(a)')-1].name

                    col_labels = col_labels.apply(lambda c: c.str.cat(sep=' '), axis=0)
                    col_labels = col_labels.apply(lambda s: s.strip())
                    col_labels = col_labels.to_frame(name=index_label).transpose()

                    df = pd.read_excel(general_url, skiprows=t_dict['skiprows'],
                                       skipfooter=t_dict['skip_footer'], 
                                       index_col=0)
                    df = df.iloc[df.index.get_loc('Code(a)')+1:]

                    df.columns = col_labels.loc[index_label, :]
                    df.columns.name = None

                    df = df.dropna(axis=1, how='all')

                    df = df.rename(columns={'Total (trillion Btu)': 'Total',
                                            'Industry Group and Industry': 'Subsector and Industry',
                                            'Industry Groups and Industry': 'Subsector and Industry',
                                            'LPG and NGL(e) (million bbl)': 'HGL (excluding natural gasoline)(e) (million bbl)',
                                            'LPG and NGL(e)': 'HGL (excluding natural gasoline)(e)'})
                    df = df.drop(['RSE Row Factors', ''], axis=1, errors='ignore')

                    if t == 'table_3_1':
                        mecs_3_1 = self.clean_industrial_data(df, table_3_1=True, sic=sic)
                        mecs_3_1['Year'] = year
                        
                        if sic:
                            sic_3_1.append(mecs_3_1)
                        else:
                            all_3_1.append(mecs_3_1)

                    elif t == 'table_3_5':
                        mecs_3_5 = self.clean_industrial_data(df, sic=sic)
                        mecs_3_5['Year'] = year
                        if sic:
                            sic_3_5.append(mecs_3_5)
                        else:
                            all_3_5.append(mecs_3_5)

                    elif t == 'table_3_2':
                        mecs_3_2 = self.clean_industrial_data(df, sic=sic)
                        mecs_3_2['Year'] = year
                        if sic:
                            sic_3_2.append(mecs_3_2)
                        else:
                            all_3_2.append(mecs_3_2)
                    
                    elif t == 'table_4_2':
                        mecs_4_2 = self.clean_industrial_data(df, sic=sic)
                        if sic:
                            sic_4_2.append(mecs_4_2)
                        else:
                            all_4_2.append(mecs_4_2)

        all_3_1 = pd.concat(all_3_1, axis=0).reset_index()
        all_3_1 = all_3_1.set_index(['Year', 'region', 'NAICS', 'Subsector and Industry'])
        sic_3_1 = pd.concat(sic_3_1, axis=0).reset_index()
        # sic_3_1 = sic_3_1.set_index(['Year', 'region', 'SIC', 'Subsector and Industry'])

        all_3_2 = pd.concat(all_3_2, axis=0).reset_index()
        # all_3_2 = all_3_2.set_index(['Year', 'region', 'NAICS', 'Subsector and Industry'])
        sic_3_2 = pd.concat(sic_3_2, axis=0).reset_index()
        # sic_3_2 = sic_3_2.set_index(['Year', 'region', 'SIC', 'Subsector and Industry'])

        all_3_5 = pd.concat(all_3_5, axis=0).reset_index()
        # all_3_5 = all_3_5.set_index(['Year', 'region', 'NAICS', 'Subsector and Industry'])
        sic_3_5 = pd.concat(sic_3_5, axis=0).reset_index()
        # sic_3_5 = sic_3_5.set_index(['Year', 'region', 'SIC', 'Subsector and Industry'])

        all_4_2 = pd.concat(all_4_2, axis=0).reset_index()
        # all_4_2 = all_4_2.set_index(['Year', 'region', 'NAICS', 'Subsector and Industry'])
        sic_4_2 = pd.concat(sic_4_2, axis=0).reset_index()
        # sic_4_2 = sic_4_2.set_index(['Year', 'region', 'SIC', 'Subsector and Industry'])

        print('all_3_2:\n', all_3_2)
        all_3_2 = all_3_2[['Other(f)']]

        all_3_5 = all_3_5.merge(all_3_2, left_index=True, right_index=True, how='inner')

        all_3_5['steam'] = all_3_5['Total'].subtract(all_3_5['Other(f)'])
        all_3_5 = all_3_5.drop(['Total', 'Other(f)'], axis=1)

        industrial_btu = all_3_5.merge(all_3_2, left_index=True, right_index=True, how='outer')
        print('industrial_btu:\n', industrial_btu)
        mecs = {'NAICS': {'3_1': all_3_1, '3_2': all_3_2, '3_5': all_3_5, '4_2': all_4_2},
                'SIC': {'3_1': sic_3_1, '3_2': sic_3_2, '3_5': sic_3_5, '4_2'}}
        return mecs



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
    
    @staticmethod
    def tedb_fuel_types(tedb_data):
        tedb_data = tedb_data.drop('Total Energy (Tbtu) - old series')
        tedb_data = tedb_data.replace({'Gasoline (million gallons)': 'Gasoline',
                                       'Diesel (million gallons)': 'Diesel',
                                       'Diesel fuel': 'Diesel',
                                       'LPG': 'Liquefied petroleum gas',
                                       })
        mapping = {['All Fuel' 'Gasoline' 'Gasohol' 'Diesel' 'CNG' 'LNG' 'Bio Diesel '
                    'Other' 'School' 'School (million bbl)' 'Intercity'
                    'Diesel Fuel & Distillate (1,000 bbl)' 'Residual Fuel Oil (1,000 bbl)'
                    'Domestic Operations ' 'International Operations '
                     'Jet fuel (million gallons)'
                    'Electricity (GWhrs)' 'Total Energy (Tbtu) '
                     'Total Energy (Tbtu)'
                    'Distillate Fuel Oil' 'Natural Gas (million cu. ft.)'
                    'Electricity (million kWh)' 'Diesel fuel' 'Liquefied petroleum gas'
                    'Jet fuel' 'Residual fuel oil' 'Natural gas' 'Electricity' 'Total']}

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

    # exit()
    # data = indicators.seds_energy_data(sector='residential')
    # print(data)
    # print(data.columns)
    # emissions_data = indicators.calculate_emissions(data, emissions_type='CO2 Factor', 
    #                                                 datasource='SEDS')

    # print('emissions_data:\n', emissions_data)
    # mecs_data = indicators.industrial_sector_energy()
    # industrial_emissions_data = indicators.calculate_emissions(mecs_data, emissions_type='CO2 Factor', 
    #                                                            datasource='MECS')
    # print('industrial_emissions_data:\n', industrial_emissions_data)
    # indicators.mecs_data_by_year()
    indicators.transportation_data()
    ef = indicators.epa_emissions_data()
    print('ef:\n', ef)
    print(ef['Fuel Type'].unique())
