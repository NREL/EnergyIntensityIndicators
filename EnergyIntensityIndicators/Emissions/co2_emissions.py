
import pandas as pd
import numpy as np
import os

from EnergyIntensityIndicators.electricity import ElectricityIndicators
from EnergyIntensityIndicators.transportation import TransportationIndicators
from EnergyIntensityIndicators.LMDI import CalculateLMDI
# from EnergyIntensityIndicators.economy_wide import EconomyWide
from EnergyIntensityIndicators.pull_eia_api import GetEIAData
from EnergyIntensityIndicators.utilities.dataframe_utilities \
    import DFUtilities as df_utils
from EnergyIntensityIndicators.utilities.standard_interpolation \
    import standard_interpolation
from EnergyIntensityIndicators.commercial \
    import CommercialIndicators
from EnergyIntensityIndicators.residential \
    import ResidentialIndicators
from EnergyIntensityIndicators.Emissions.noncombustion \
    import NonCombustion
from EnergyIntensityIndicators.industry \
    import IndustrialIndicators
from EnergyIntensityIndicators.lmdi_gen import GeneralLMDI


class CO2EmissionsDecomposition(CalculateLMDI):
    """Class to decompose CO2 emissions by
    sector of the U.S. economy.

    LMDI aspects:
    - total activity (Q),
    - activity share for subsector i (structure) (=Q_i/Q),
    - energy intensity for subsector i (=E_i/Q_i),
    - energy share of type j in subsector i (=E_ij/E_i),
        also called fuel mix
    - emissions rate of energy type j and subsector i (=C_ij/E_ij),
         also called emissions coefficient

    (for time series, all variables have t subscripts
    (i.e. no constants-- constant emissions rates cancel out))
    """
    def __init__(self, directory, output_directory,
                 sector, fname, categories_dict,
                 level_of_aggregation=None):

        self.sector = sector
        self.eia = GetEIAData('emissions')
        self.yaml_dir = 'C:/Users/irabidea/Desktop/yamls/'

        self.gen = GeneralLMDI(self.yaml_dir, class_=CO2EmissionsDecomposition)
        self.gen.read_yaml(fname)

        super().__init__(sector,
                         level_of_aggregation=level_of_aggregation,
                         lmdi_models=self.model,
                         categories_dict=categories_dict,
                         energy_types=self.energy_types,
                         directory=directory,
                         output_directory=output_directory,
                         primary_activity=None,
                         base_year=self.base_year,
                         end_year=self.end_year,
                         unit_conversion_factor=1,
                         weather_activity=None)

    def collect_emissions(self, sector=None,
                          fuel_type=None, state_abbrev=None):
        """Collect emissions from the EIA API (through
        GetEIAData).

        Args:
            sector (str, optional): Economic sector. Defaults to None.
            fuel_type (str, optional): Fuel Type. Defaults to None.
            state_abbrev (str, optional): State (e.g. 'AK'). Defaults to None.

        Returns:
            data (DataFrame): Emissions data for sector, fuel type and state
        """
        eia_co2_emiss = f'EMISS.CO2-TOTV-{sector}-{fuel_type}-{state_abbrev}.A'
        data = self.eia.eia_api(id_=eia_co2_emiss, id_type='series')
        # , new_name='')

        return data

    @staticmethod
    def get_fuel_mix(region_data):
        """Calculate shares of total fuel by fuel type

        Args:
            region_data (DataFrame): Fuel use data by fuel for a region

        Returns:
            fuel_mix (DataFrame): Fuel mix (i.e. share of total by fuel)
        """
        region_data = region_data.drop('Census Region',
                                       axis=1, errors='ignore')
        region_data = df_utils().create_total_column(region_data,
                                                     total_label='total')
        fuel_mix = \
            df_utils().calculate_shares(region_data, total_label='total')
        return fuel_mix

    @staticmethod
    def get_mean_factor(emissions_factors, input_cols, new_name):
        """[summary]

        Args:
            emissions_factors (DataFrame): emissions factors
            input_cols (list): List of emissions
                               factors to average
            new_name (str): Name of resulting factor

        """
        emissions_factors['value'] = \
            emissions_factors['value'].apply(lambda x: str(x).replace(',', ''))
        emissions_factors['value'] = emissions_factors['value'].astype(float)

        subset = \
            emissions_factors[emissions_factors['Fuel Type'].isin(input_cols)]

        grouped = \
            subset.groupby(by=['Category', 'Unit', 'Variable'])

        mean_df = grouped.mean()
        mean_df.loc[:, 'Fuel Type'] = new_name

        ef = pd.concat([emissions_factors, mean_df], axis=0)
        return ef

    def epa_emissions_data(self):
        """Read and process EPA emissions factors data

        Returns:
            emissions_factors (DataFrame): [description]
        """
        ef = pd.read_csv(
                './EnergyIntensityIndicators/Data/EPA_emissions_factors.csv')
        df_cols = ef.columns
        dfs = []
        grouped = ef.groupby(ef['Unit Type'])
        for g in ef['Unit Type'].unique():
            unit_data = grouped.get_group(g)
            unit_data.columns = unit_data.iloc[0]

            units_dict = dict(zip(unit_data.columns, df_cols))
            unit_data = unit_data.drop(g, axis=1)

            unit_data = unit_data.drop(unit_data.index[0])
            unit_data = unit_data.melt(id_vars=['Units', 'Fuel Type'],
                                       var_name='Unit')

            unit_data = unit_data.rename(columns={'Units': 'Category'})
            unit_data.loc[:, 'Variable'] = unit_data['Unit'].map(units_dict)

            dfs.append(unit_data)
        emissions_factors = pd.concat(dfs, axis=0)
        ef = self.get_mean_factor(emissions_factors,
                                  input_cols=['Distillate Fuel Oil No. 1',
                                              'Distillate Fuel Oil No. 2',
                                              'Distillate Fuel Oil No. 4'],
                                  new_name='Distillate Fuel Oil')
        ef = self.get_mean_factor(ef,
                                  input_cols=['Residual Fuel Oil No. 5',
                                              'Residual Fuel Oil No. 6'],
                                  new_name='Residual Fuel Oil')

        ef = self.get_mean_factor(ef,
                                  input_cols=['Blast Furnace Gas',
                                              'Coke Oven Gas'],
                                  new_name='Blast Furnace/Coke Oven Gases')

        ef = self.get_mean_factor(ef,
                                  input_cols=['North American Softwood',
                                              'North American Hardwood'],
                                  new_name='Pulping Liquor or Black Liquor')

        ef = self.get_mean_factor(ef,
                                  input_cols=['Motor Gasoline',
                                              'Ethanol (100%)'],
                                  new_name='Gasohol')
        ef = self.get_mean_factor(ef,
                                  input_cols=['Diesel Fuel',
                                              'Distillate Fuel Oil No. 1',
                                              'Distillate Fuel Oil No. 2',
                                              'Distillate Fuel Oil No. 4'],
                                  new_name='Diesel Fuel & Distillate')

        ef = self.get_mean_factor(ef,
                                  input_cols=['Distillate Fuel Oil No. 1',
                                              'Distillate Fuel Oil No. 2',
                                              'Distillate Fuel Oil No. 4',
                                              'Residual Fuel Oil No. 5',
                                              'Residual Fuel Oil No. 6'],
                                  new_name='Petroleum')

        return emissions_factors

    @staticmethod
    def mecs_epa_mapping(mecs_data):
        """Rename mecs_data columns so that labels match
        EPA emissions factors labels

        Args:
            mecs_data (DataFrame): [description]

        Returns:
            mecs_data (DataFrame): MECS data with column names
                                   that match EPA emissions data
                                   labels
        """
        mapping_ = {
                    # 'Blast Furnace/Coke Oven Gases':
                    #    ['Blast Furnace Gas',
                    #     'Coke Oven Gas'],  # take average
                    'Waste Gas': 'Fuel Gas',
                    'Petroleum Coke': 'Petroleum Coke',
                    # 'Pulping Liquor or Black Liquor':
                    #     ['North American Softwood',
                    #      'North American Hardwood'],  # take average
                    'Wood Chips, Bark': 'Wood and Wood Residuals',
                    'Waste Oils/Tars and Waste Materials': 'Used Oil',
                    'steam': 'Steam and Heat',  # From Table 7
                    'Net Electricity': 'Us Average',  # From Table 6,  Total Output Emissions Factors CO2 Factor
                    # 'Residual Fuel Oil':
                    #    ['Residual Fuel Oil No. 5',  # take average
                    #     'Residual Fuel Oil No. 6'],
                    # 'Distillate Fuel Oil':
                    #   ['Distillate Fuel Oil No. 1',  # take average
                    #    'Distillate Fuel Oil No. 2',
                    #    'Distillate Fuel Oil No. 4'],
                    'Natural Gas': 'Natural Gas',
                    'HGL (excluding natural gasoline)':
                        'Liquefied Petroleum Gases (LPG)',
                    'Coal':
                        'Mixed (Industrial Sector)',  # OR Mixed (Industrial Coking)?
                    'Coke Coal and Breeze':
                        'Coal Coke'}
        mecs_data = mecs_data.rename(columns=mapping_)
        return mecs_data

    @staticmethod
    def electric_epa_mapping():
        """[summary]

        Returns:
            [type]: [description]
        """
        mapping_ = {'Coal': 'Mixed (Electric Power Sector)',
                    # 'Petroleum':
                    #     ['Distillate Fuel Oil No. 1',
                    #      'Distillate Fuel Oil No. 2',
                    #      'Distillate Fuel Oil No. 4',
                    #      'Residual Fuel Oil No. 5',
                    #      'Residual Fuel Oil No. 6'],  # take average
                    'Natural Gas': 'Natural Gas',
                    'Other Gases': 'Fuel Gas',
                    'Waste': 'Municipal Solid Waste',
                    'Wood': 'Wood and Wood Residuals'}
        return mapping_

    @staticmethod
    def tedb_epa_mapping(tedb_data):
        """[summary]

        Args:
            tedb_data ([type]): [description]
        """
        # tedb_data = tedb_data.drop('Total Energy (Tbtu) - old series')

        

        mapping = {'Gasoline': 'Motor Gasoline',  # ef is in gallon
                    'Gasohol': 'Gasohol',  # ef is in gallon
                    'Diesel': 'Diesel Fuel', # ef is in gallon
                    'CNG': 'Compressed Natural Gas (CNG)', # ef is in scf 
                    'LNG': 'Liquefied Natural Gas (LNG)', # ef is in gallons
                    'Bio Diesel ': 'Biodiesel (100%)', # ef is in gallons
                    'Diesel Fuel & Distillate (1,000 bbl)': # ef is in gallon (42 gallons in a barrel)
                        'Diesel Fuel & Distillate',
                    'Residual Fuel Oil (1,000 bbl)': 'Residual Fuel Oil',  # ef is in gallon (42 gallons in a barrel)
                    'Jet fuel (million gallons)': 'Aviation Gasoline',  # ef is per gallon
                    'Electricity (GWhrs)': 'US Average', # ef is /MWh
                    'Distillate Fuel Oil': 'Distillate Fuel Oil',  # ef is per gallon
                    'Natural Gas (million cu. ft.)': 'Natural Gas',  # ef is per scf
                    'Electricity (million kWh)': 'US Average', # ef is /MWh
                    'Diesel fuel': 'Diesel Fuel',  # ef is per gallon
                    'Liquefied petroleum gas':
                        'Liquefied Petroleum Gases (LPG)',  # ef is per gallon
                    'Jet fuel': 'Aviation Gasoline',  # ef is per gallon
                    'Residual fuel oil': 'Residual Fuel Oil',  # ef is per gallon
                    'Natural gas': 'Natural Gas',  # ef is per scf
                    'Electricity': 'US Average'}  # ef is /MWh
        irrelevant_fuels = [f for f in tedb_data['Fuel Type'].unique()
                            if f not in mapping.keys()]
        tedb_data = tedb_data[~tedb_data['Fuel Type'].isin(irrelevant_fuels)]
        tedb_data['Fuel Type'] = tedb_data['Fuel Type'].replace(mapping)
        return tedb_data

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
        fuel_factor_df.loc[:, 'value'] = fuel_factor_df['value'].astype(float)
        new_row = {'Fuel Type': 'Census Region', 'value': 1}
        fuel_factor_df = fuel_factor_df.append(new_row, ignore_index=True)
        fuel_factor_df = fuel_factor_df.set_index('Fuel Type')
        fuel_factor_df = fuel_factor_df['value']
        return fuel_factor_df

    def collect_emissions_data(self):
        """Calculate emissions data for all sectors and fuel types
        (from energy and fuel mix data)

        Parameters:

        Returns:
            emissions_data_dict (dict): Nested dictionary of all_data
                                        from EconomyWide with energy
                                        data replaced with emissions data
                                        (with original dictionary keys
                                        remaining intact)
        TODO:
            - Break sector_level_data into lowest levels of dict in order
              to manipulate energy data
            - replace energy_data with emissions_data in nested dictionary
        """
        all_data = self.collect_data()  # This is currently dictionary of
                                        # all data collected in
                                        # EconomyWide.collect_data()

        state_abbrev = 'AK'  # Alaska
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

        #     emissions_data = self.calculate_emission(energy_data,
        #                                              emission_factor)

        #     # replace energy_data in nested dictionary with emissions_data

        emissions_data_dict = {}
        return emissions_data_dict

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
            energy_data = self.electric_epa_mapping(energy_data)
        elif datasource == 'TEDB':
            energy_data = self.tedb_epa_mapping(energy_data)

        # print('energy_data:\n', energy_data)
        emissions_factors = self.get_factor(emissions_factors,
                                            emissions_type)

        emissions_factors = \
            emissions_factors.to_frame(name='Emissions Factors')
        emissions_factors = emissions_factors.transpose()
        print('emissions_factors:\n', emissions_factors)
        print('emissions_factors cols', emissions_factors.columns)
        print('energy_data.columns.tolist():', energy_data.columns.tolist())
        try:
            emissions_factors = emissions_factors[energy_data.columns.tolist()]
        except KeyError:
            print('energy_data.columns.tolist() not in dataframe:',
                  energy_data.columns.tolist())
            return None
        print('emissions_factors:\n', emissions_factors)

        emissions_data = \
            energy_data.multiply(emissions_factors.to_numpy())
        print('emissions_data:\n', emissions_data)

        emissions_data.loc[:, 'Census Region'] = \
            emissions_data['Census Region'].astype(int).astype(str)
        print('emissions_data:\n', emissions_data)

        energy_data.loc[:, 'Census Region'] = \
            energy_data.loc[:, 'Census Region'].astype(int).astype(str)
        fuel_mix = self.get_fuel_mix(energy_data)
        print(fuel_mix)
        return emissions_data, fuel_mix

    def calc_lmdi(self, breakout, calculate_lmdi, data_dict):
        """Calculate decomposition of CO2 emissions for the U.S. economy

        TODO: allow for different sectors to have different types of energy
              and commercial and residential to have weather adjustment
              (TODO carried over from EconomyWide)

        """

        results_dict, formatted_results = \
            self.get_nested_lmdi(
                level_of_aggregation=self.level_of_aggregation,
                breakout=breakout, calculate_lmdi=calculate_lmdi,
                raw_data=data_dict, lmdi_type='LMDI-I', new_style=True)
        return results_dict


class SEDSEmissionsData(CO2EmissionsDecomposition):
    """Class to [Summary]

    """
    def __init__(self, directory,
                 output_directory, sector,
                 fname, categories_dict):

        super().__init__(directory=directory,
                         output_directory=output_directory,
                         sector=sector,
                         fname=fname,
                         categories_dict=categories_dict,
                         level_of_aggregation=None)

    @staticmethod
    def state_census_crosswalk():
        """Match states with Census Regions

        Returns:
            [type]: [description]
        """
        print('os.getcwd():', os.getcwd())
        try:
            cw = pd.read_csv(
                    './Data/state_to_census_region.csv')
            state_abbrevs = pd.read_csv(
                    './Data/name-abbr.csv')
        except FileNotFoundError:
            cw = pd.read_csv(
                    './EnergyIntensityIndicators/Data/state_to_census_region.csv')
            state_abbrevs = pd.read_csv(
                    './EnergyIntensityIndicators/Data/name-abbr.csv')
        cw = cw.merge(state_abbrevs, left_on='USPC',
                      right_on='Abbrev', how='left')
        return cw

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
        ethanol_cols = ['Fuel Ethanol excluding Denaturant',
                        'Fuel Ethanol including Denaturant']
        eia_data['Ethanol (100%)'] = eia_data[ethanol_cols].sum(axis=1)
        eia_data = eia_data.drop(ethanol_cols, axis=1)
        mapping_ = {'Coal': 'Mixed (Commercial Sector)',
                    'Distillate Fuel Oil': 'Distillate Fuel Oil',
                    # 'Distillate Fuel Oil': ['Distillate Fuel Oil No. 1',
                    #                         'Distillate Fuel Oil No. 2',
                    #                         'Distillate Fuel Oil No. 4'],  # take average
                    # 'Fuel Ethanol including Denaturant':
                    #     'Ethanol (100%)',
                    # 'Fuel Ethanol excluding Denaturant': 'Ethanol (100%)',
                    'Ethanol (100%)': 'Ethanol (100%)',
                    'Hydrocarbon gas liquids':
                        'Liquefied Petroleum Gases (LPG)',
                    'Kerosene':
                        'Kerosene',
                    'Motor Gasoline':
                        'Motor Gasoline',
                    'Natural Gas including Supplemental Gaseous Fuels':
                        'Natural Gas',
                    'Petroleum Coke':
                        'Petroleum Coke',
                    'Propane': 'Propane',
                    # 'Residual Fuel Oil': ['Residual Fuel Oil No. 5',
                    #                       'Residual Fuel Oil No. 6'],  # take average
                    'Residual Fuel Oil': 'Residual Fuel Oil',
                    'Waste': 'Municipal Solid Waste',
                    'Wood': 'Wood and Wood Residuals'}

        # irrelevant_fuels = [f for f in eia_data.columns if f not in
        #                     mapping_.keys() and f != 'Census Region']
        # eia_data = eia_data.drop(irrelevant_fuels, axis=1)

        eia_data = eia_data.rename(columns=mapping_)

        return eia_data

    @staticmethod
    def seds_endpoints(sector, state, fuel):
        """[summary]

        Args:
            sector ([type]): [description]
            state ([type]): [description]
            fuel ([type]): [description]

        Returns:
            [type]: [description]
        """
        endpoints = {'All Petroleum Products':
                        f'SEDS.PA{sector}P.{state}.A',
                     'Coal':
                        f'SEDS.CL{sector}P.{state}.A',
                     'Distillate Fuel Oil':
                        f'SEDS.DF{sector}P.{state}.A',
                     'Electrical System Energy Losses':
                        f'SEDS.LO{sector}B.{state}.A',
                     'Electricity Sales':
                        f'SEDS.ES{sector}P.{state}.A',
                     'Fuel Ethanol including Denaturant':
                        f'SEDS.EN{sector}P.{state}.A',
                     'Fuel Ethanol excluding Denaturant':
                        f'SEDS.EM{sector}B.{state}.A',
                     'Geothermal':
                        f'SEDS.GE{sector}B.{state}.A',
                     'Hydrocarbon gas liquids':
                        f'SEDS.HL{sector}P.{state}.A',
                     'Hydroelectricity':
                        f'SEDS.HY{sector}P.{state}.A',
                     'Kerosene':
                        f'SEDS.KS{sector}P.{state}.A',
                     'Motor Gasoline':
                        f'SEDS.MG{sector}P.{state}.A',
                     'Natural Gas including Supplemental Gaseous Fuels':
                        f'SEDS.NG{sector}P.{state}.A',
                     'Petroleum Coke':
                        f'SEDS.PC{sector}P.{state}.A',
                     'Propane':
                        f'SEDS.PQ{sector}P.{state}.A',
                     'Residual Fuel Oil':
                        f'SEDS.RF{sector}P.{state}.A',
                     'Solar Energy':
                        f'SEDS.SOR7P.{state}.A',
                     'Total (per Capita)':
                        f'SEDS.TE{sector[0]}PB.{state}.A',
                     'Total Energy excluding Electrical System Energy Losses':
                        f'SEDS.TN{sector}B.{state}.A',
                     'Waste':
                        f'SEDS.WS{sector}B.{state}.A',
                     'Wind Energy':
                        f'SEDS.WY{sector}P.{state}.A',
                     'Wood':
                        f'SEDS.WD{sector}B.{state}.A',
                     'Wood and Waste':
                        f'SEDS.WW{sector}B.{state}.A'}
        return endpoints[fuel]

    def collect_seds(self, sector, states):
        """SEDS energy consumption data (in physical units unless
        unavailable, in which case in Btu-- indicated by P or
        B in endpoint)

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
                        'Hydrocarbon gas liquids',
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
                        'Electrical System Energy Losses',  # in BTU
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
                    df = self.eia.eia_api(id_=self.seds_endpoints(sector,
                                                                  s, f),
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

        fuels_data = df_utils().merge_df_list(fuels_data)
        return fuels_data

    def seds_energy_data(self, sector):
        """[summary]

        Args:
            sector ([type]): [description]

        Returns:
            [type]: [description]
        """
        states = self.state_census_crosswalk()
        sector_data = {'commercial': {'abbrev': 'CC', 'regions': [0]},
                       'residential': {'abbrev': 'RC',
                                       'regions': [1, 2, 3, 4]}}
        grouped = states.groupby(states['Census Region'])
        all_data = []
        for g in sector_data[sector]['regions']:
            region_states = grouped.get_group(g)
            region_data = self.collect_seds(
                                        sector=sector_data[sector]['abbrev'],
                                        states=region_states['USPC'].unique())
            region_data.loc[:, 'Census Region'] = g
            all_data.append(region_data)
        all_data = pd.concat(all_data, axis=0)
        return all_data


class ResidentialEmissions(SEDSEmissionsData):
    def __init__(self, directory, output_directory,
                 level_of_aggregation='National'):
        if level_of_aggregation == 'National':
            fname = 'residential_all_emissions'
        else:
            fname = 'residential_regional'

        self.sub_categories_list = \
            {'National':
                {'Northeast':
                    {'Single-Family': None,
                     'Multi-Family': None,
                     'Manufactured-Homes': None},
                 'Midwest':
                    {'Single-Family': None,
                     'Multi-Family': None,
                     'Manufactured-Homes': None},
                 'South':
                    {'Single-Family': None,
                     'Multi-Family': None,
                     'Manufactured-Homes': None},
                 'West':
                    {'Single-Family': None,
                     'Multi-Family': None,
                     'Manufactured-Homes': None}}}

        super().__init__(directory, output_directory,
                         sector='Residential',
                         fname=fname,
                         categories_dict=self.sub_categories_list)

        self.res = \
            ResidentialIndicators(directory='./EnergyIntensityIndicators/Data',
                                  output_directory='./Results',
                                  level_of_aggregation=level_of_aggregation,
                                  lmdi_model=self.model,
                                  end_year=self.end_year,
                                  base_year=self.base_year)

    def main(self):

        res_data = self.res.collect_data()['National']

        energy_data = self.seds_energy_data(sector='residential')
        activity = res_data['activity']
        weather_factors = res_data['weather_factors']

        emissions = \
            self.calculate_emissions(energy_data,
                                     emissions_type='CO2 Factor',
                                     datasource='SEDS')
        return {'E_i_j': energy_data,
                'A_i': activity,
                'C_i_j': emissions,
                'WF_i': weather_factors}


class CommercialEmissions(SEDSEmissionsData):
    def __init__(self, directory, output_directory):
        fname = 'commercial_total'
        self.sub_categories_list = {'Commercial_Total': None}
        super().__init__(directory, output_directory,
                         sector='Commercial',
                         fname=fname,
                         categories_dict=self.sub_categories_list)
        self.level_of_aggregation = 'Commercial_Total'
        self.comm = \
            CommercialIndicators(
                directory='./EnergyIntensityIndicators/Data',
                output_directory='./Results',
                level_of_aggregation=self.level_of_aggregation,
                lmdi_model=self.model,
                end_year=self.end_year,
                base_year=self.base_year)

    def main(self):

        energy_data = self.seds_energy_data(sector='commercial')

        comm_data = self.comm.collect_data()['Commercial_Total']
        weather_factors = comm_data['weather_factors']
        activity = comm_data['activity']
        emissions = \
            self.calculate_emissions(energy_data,
                                     emissions_type='CO2 Factor',
                                     datasource='SEDS')

        return {'Commercial_Total':
                {'E_j': energy_data,
                 'C_j': emissions,
                 'WF': weather_factors,
                 'A': activity}}


class IndustrialEmissions(CO2EmissionsDecomposition):
    def __init__(self, directory, output_directory, level_of_aggregation):
        if level_of_aggregation == 'Manufacturing':
            fname = 'combustion_noncombustion_test'
        elif level_of_aggregation == 'NonManufacturing':
            fname = 'combustion_noncombustion_test'

        self.sub_categories_list = \
            {'Industry':
                {'Manufacturing':
                    {'Food and beverage and tobacco products': None,
                     'Textile mills and textile product mills': None,
                     'Apparel and leather and allied products': None,
                     'Wood products': None,
                     'Paper products': None,
                     'Printing and related support activities': None,
                     'Petroleum and coal products': None,
                     'Chemical products':
                        {'noncombustion':
                            {'Petrochemical Production': None,
                             'Titanium Dioxide Production': None,
                             'Nitric Acid Production': None,
                             'Phosphoric Acid Production': None,
                             'Adipic Acid Production': None,
                             'Ammonia Production': None,
                             'Carbide Production and Consumption': None,
                             'Soda Ash Production': None,
                             'N2O from Product Uses': None,
                             'Urea Consumption for NonAgricultural Purposes':
                                None,
                             'Caprolactam, Glyoxal, and Glyoxylic Acid Production':
                                None},
                         'combustion': None},
                     'Plastics and rubber products': None,
                     'Nonmetallic mineral products':
                        {'noncombustion':
                            {'Cement Production': None,
                             'Glass Production': None,
                             'Lime Production': None,
                             'Other Process Uses of Carbonates': None,
                             'Carbon Dioxide Consumption': None},
                         'combustion': None},
                     'Primary metals':
                        {'noncombustion':
                            {'Lead Production': None,
                             'Zinc Production': None,
                             'Aluminum Production': None},
                         'combustion': None},
                     'Fabricated metal products':
                        {'noncombustion':
                            {'Ferroalloy Production': None,
                             'Metallurgical coke': None,
                             'Iron and Steel': None},
                         'combustion': None},
                     'Machinery': None,
                     'Computer and electronic products': None,
                     'Electrical equipment, appliances, and components': None,
                     'Motor vehicles, bodies and trailers, and parts': None,
                     'Furniture and related products': None,
                     'Miscellaneous manufacturing': None},
                 'Nonmanufacturing':
                    {'Agriculture, Forestry & Fishing':
                        {'noncombustion':
                            {'Urea Fertilization': None,
                             'Agricultural Soil Management': None,
                             'Manure Management': None,
                             'Enteric Fermentation': None,
                             'Liming': None},
                         'combustion': None},
                     'Mining':
                        {'Petroleum and Natural Gas':
                            {'combustion': None},
                         'Other Mining':
                            {'noncombustion':
                                {'Coal Mining': None},
                             'combustion': None},
                         'Support Activities':
                            {'combustion': None}},
                     'Construction':
                        {'combustion': None},
                     'Waste':
                        {'noncombustion':
                            {'Landfills': None,
                             'Composting': None}},
                     'Energy':
                        {'noncombustion':
                            {'Stationary Combustion': None,
                             'Non-Energy Use of Fuels': None}}}}}

        super().__init__(directory, output_directory,
                         sector='Industry',
                         level_of_aggregation=level_of_aggregation,
                         fname=fname,
                         categories_dict=self.sub_categories_list)

    @staticmethod
    def energy_data():
        data_dir = './EnergyIntensityIndicators/Industry/Data/'
        construction_elec_fuels = \
            pd.read_csv(
                f'{data_dir}construction_elec_fuels.csv').set_index('Year')
        agriculture = \
            pd.read_excel(
                f'{data_dir}miranowski_data.xlsx',
                sheet_name='Ag Cons by Use', skiprows=4, skipfooter=9,
                usecols='A:F', index_col=0,
                names=['Year', 'Gasoline', 'Diesel', 'LP Gas',
                       'Natural Gas', 'Electricity'])
        # Mining
        mining = \
            pd.read_csv(
                f'{data_dir}mining_energy.csv')
        print('mining:\n', mining)
        mining = mining.fillna(np.nan)
        mining = mining.dropna(how='all', axis=1)
        mining = mining[mining['NAICS'].notnull()]
        mining = mining.astype({'Year': int,
                                'NAICS': int})
        all_mining = []
        for n in mining['NAICS'].unique():
            mining_naics = mining[mining['NAICS'] == n]
            mining_naics = mining_naics.drop('NAICS', axis=1)
            mining_naics = mining_naics.set_index(['Year'])
            mining_naics = \
                mining_naics.apply(
                    lambda col: pd.to_numeric(col, errors='coerce'), axis=1)
            print('mining:\n', mining)

            for c in mining_naics.columns:
                mining_naics = \
                    standard_interpolation(mining_naics,
                                           name_to_interp=c,
                                           axis=1)
            mining_naics['NAICS'] = n
            all_mining.append(mining_naics)
        all_mining = pd.concat(all_mining, axis=0)

        manufacturing = pd.read_csv(
            f'{data_dir}mecs_table42.csv')
        print('manufacturing:\n', manufacturing)
        manufacturing = manufacturing.dropna(how='all', axis=1)
        manufacturing = manufacturing.fillna(np.nan)
        manufacturing = manufacturing[manufacturing['NAICS'].notnull()]
        manufacturing = manufacturing.astype({'Year': int,
                                              'NAICS': int})
        all_manufacturing = []
        for n in manufacturing['NAICS'].unique():
            manufacturing_naics = manufacturing[manufacturing['NAICS'] == n]
            manufacturing_naics = manufacturing_naics.drop('NAICS', axis=1)
            manufacturing_naics = manufacturing_naics.set_index(['Year'])
            manufacturing_naics = \
                manufacturing_naics.apply(
                    lambda col: pd.to_numeric(col, errors='coerce'), axis=1)

            print('manufacturing_naics:\n', manufacturing_naics)
            for c in manufacturing_naics.columns:
                manufacturing_naics = \
                    standard_interpolation(manufacturing_naics,
                                           name_to_interp=c,
                                           axis=1)
            manufacturing_naics['NAICS'] = n
            all_manufacturing.append(manufacturing_naics)

        all_manufacturing = pd.concat(all_manufacturing, axis=0)
        return {'Manufacturing': all_manufacturing,
                'NonManufacturing':
                    {'Mining': all_mining,
                     'Construction': construction_elec_fuels,
                     'Agriculture, Forestry & Fishing': agriculture}}

    def collect_manufacturing_data(self, noncombustion_data, manufacturing,
                                   combustion_activity):
        cats = self.sub_categories_list['Industry']
        man = cats['Manufacturing']
        combustion_activity_m = combustion_activity['Manufacturing']
        manufacturing_dict = dict()
        print('manufacturing:\n', manufacturing)
        print('manufacturing:\n', manufacturing.columns)
        print('combustion_activity_m:\n', combustion_activity_m)
        print('combustion_activity_m:\n', combustion_activity_m.columns)

        for naics in man.keys():
            combustion_energy_data = \
                manufacturing = manufacturing[naics]['energy']
            # manufacturing[manufacturing['NAICS'] == naics]
            combustion_activity_naics = \
                combustion_activity_m[naics]
            # combustion_activity_m[combustion_activity_m['NAICS'] == naics]
            naics_dict = dict()
            noncombustion_activity = []
            noncombustion_emissions = []
            naics_emissions = \
                self.calculate_emissions(combustion_energy_data,
                                         emissions_type='CO2 Factor',
                                         datasource='MECS')
            naics_dict['combustion'] = {'E_i_j': combustion_energy_data,
                                        'A_i_k': combustion_activity_naics,
                                        'C_i_j_k': naics_emissions}

            if not man[naics]:
                continue
            else:
                for sub_category in man[naics]['noncombustion'].keys():
                    noncombustion_cat_data = noncombustion_data[sub_category]
                    e_ = noncombustion_cat_data['emissions']
                    a_ = noncombustion_cat_data['activity']

                    e_ = \
                        df_utils.create_total_column(e_, sub_category)
                    e_ = e_[[sub_category]]
                    noncombustion_emissions.append(e_)
                    a_ = \
                        df_utils.create_total_column(a_, sub_category)
                    a_ = a_[[sub_category]]
                    noncombustion_activity.append(a_)

                noncombustion_activity = \
                    df_utils.merge_df_list(noncombustion_activity)
                noncombustion_emissions = \
                    df_utils.merge_df_list(noncombustion_emissions)

                naics_dict['noncombustion'] = \
                    {'A_i_k': noncombustion_activity,
                     'C_i_j_k': noncombustion_emissions}
                manufacturing_dict[naics] = naics_dict

    def collect_nonmanufacturing_data(self, combustion_activity,
                                      nonman_data, noncombustion_data):
        cats = self.sub_categories_list['Industry']['Manufacturing']

        nonmanufacturing_dict = dict()
        for subcategory in cats.keys():
            subcategory_dict = dict()
            noncombustion_activity = []
            noncombustion_emissions = []

            if subcategory.isin(
                    ['Agriculture, Forestry & Fishing', 'Construction']):
                sub_data_combustion = nonman_data[subcategory]['combustion']
                sub_energy_data_combustion = sub_data_combustion['energy']
                sub_activity_data_combustion = sub_data_combustion['activity']
                sub_emissions_data_combustion = \
                    self.calculate_emissions(sub_energy_data_combustion,
                                             emissions_type='CO2 Factor',
                                             datasource='MECS')
                subcategory_dict['combustion'] = \
                    {'A_i_k': sub_activity_data_combustion,
                     'E_i_k_j': sub_energy_data_combustion,
                     'C_i_j_k': sub_emissions_data_combustion}

                sub_data_noncombustion = \
                    nonman_data[subcategory]['noncombustion']

                # s_data = cats[subcategory]
                noncombustion_activity, noncombustion_emissions = \
                    self.handle_noncombustion(sub_data_noncombustion,
                                              noncombustion_cat_data,
                                              subcategory)

                subcategory_dict['noncombustion'] = \
                    {'A_i_k': noncombustion_activity,
                     'C_i_j_k': noncombustion_emissions}

            elif subcategory == 'Mining':
                mining_dict = dict()
                s_data = cats[subcategory]
                for lower in s_data.keys():
                    if lower == 'Other Mining':
                        other_mining_dict = dict()

                        noncombustion = s_data[lower]['noncombustion']
                        noncombustion_activity = noncombustion['activity']
                        noncombustion_emissions = noncombustion['emissions']
                        other_mining_dict['noncombustion'] = \
                            {'A_i_k': noncombustion_activity,
                             'C_i_j_k': noncombustion_emissions}

                        combustion_activity = \
                            combustion_activity[subcategory][lower]['activity']
                        combustion_energy = \
                            combustion_activity[subcategory][lower]['energy']
                        combustion_emissions = \
                            self.calculate_emissions(
                                combustion_energy,
                                emissions_type='CO2 Factor',
                                datasource='MECS')
                        other_mining_dict['combustion'] = \
                            {'A_i_k': combustion_activity,
                             'C_i_j_k': combustion_emissions,
                             'E_i_j_k': combustion_energy}

                        mining_dict[lower] = other_mining_dict
                    else:
                        mining_combustion_activity = \
                            combustion_activity[subcategory][lower]['activity']
                        mining_combustion_energy = \
                            combustion_activity[subcategory][lower]['energy']
                        combustion_emissions = \
                            self.calculate_emissions(
                                                mining_combustion_energy,
                                                emissions_type='CO2 Factor',
                                                datasource='MECS')
                        mining_dict[lower] = \
                            {'combustion':
                                {'A_i_k': combustion_activity,
                                 'C_i_j_k': combustion_emissions,
                                 'E_i_j_k': combustion_energy}}
            else:
                s_data = cats[subcategory]
                noncombustion_activity, noncombustion_emissions = \
                    self.handle_noncombustion(s_data,
                                              noncombustion_cat_data,
                                              subcategory)

                subcategory_dict['noncombustion'] = \
                    {'A_i_k': noncombustion_activity,
                     'C_i_j_k': noncombustion_emissions}

            nonmanufacturing_dict[subcategory] = subcategory_dict

    @staticmethod
    def handle_noncombustion(s_data, noncombustion_cat_data,
                             sub_category):
        noncombustion_activity = []
        noncombustion_emissions = []
        for s in s_data['noncombustion'].keys():
            noncombustion_cat_data = noncombustion_data[s]
            e_ = noncombustion_cat_data['emissions']
            a_ = noncombustion_cat_data['activity']

            e_ = \
                df_utils.create_total_column(e_, sub_category)
            e_ = e_[[sub_category]]
            noncombustion_emissions.append(e_)
            a_ = \
                df_utils.create_total_column(a_, sub_category)
            a_ = a_[[sub_category]]
            noncombustion_activity.append(a_)

        noncombustion_activity = \
            df_utils.merge_df_list(noncombustion_activity)
        noncombustion_emissions = \
            df_utils.merge_df_list(noncombustion_emissions)

        return noncombustion_activity, noncombustion_emissions

    # @staticmethod
    # def handle_combustion():

    def main(self):
        noncombustion_data = NonCombustion().main()

        combustion = \
            IndustrialIndicators(directory='./EnergyIntensityIndicators/Data',
                                 output_directory='./Results',
                                 level_of_aggregation='Industry',
                                 lmdi_model=self.model,
                                 end_year=self.end_year,
                                 base_year=self.base_year)

        combustion_activity = combustion.collect_data()['Industry']
        manufacturing_combustion = combustion_activity['Manufacturing']
        nonmanufacturing_combustion = combustion_activity['Nonmanufacturing']

        energy_data = self.energy_data()

        manufacturing_data = \
            self.collect_manufacturing_data(noncombustion_data,
                                            manufacturing_combustion,
                                            combustion_activity)
        nonmanufacturing_data = \
            self.collect_nonmanufacturing_data(combustion_activity,
                                               nonmanufacturing_combustion,
                                               noncombustion_data)

        data = {'NonManufacturing': nonmanufacturing_data,
                'Manufacturing': manufacturing_data}
        return data


class TransportationEmssions(CO2EmissionsDecomposition):
    def __init__(self, directory, output_directory, level_of_aggregation):
        fname = 'transportation_emissions'
        self.sub_categories_list = \
            {'All_Transportation':
                {'All_Passenger':
                    {'Highway':
                        {'Passenger Cars and Trucks':
                            {'Passenger Car – SWB Vehicles':
                                {'Passenger Car': None,
                                 'SWB Vehicles': None},
                             'Light Trucks – LWB Vehicles':
                                {'Light Trucks': None,
                                 'LWB Vehicles': None},
                             'Motorcycles': None},
                         'Buses':
                            {'Urban Bus': None,
                             'Intercity Bus': None,
                             'School Bus': None},
                         'Paratransit':
                            None},
                     'Rail':
                        {'Urban Rail':
                            {'Commuter Rail': None,
                             'Heavy Rail': None,
                             'Light Rail': None},
                         'Intercity Rail': None},
                     'Air':
                        {'Commercial Carriers': None,
                         'General Aviation': None}},
                 'All_Freight':
                    {'Highway':
                        {'Single-Unit Truck': None,
                         'Combination Truck': None},
                     'Rail': None,
                     'Air': None,
                     'Waterborne': None,
                     'Pipeline':
                        {'Oil Pipeline': None,
                         'Natural Gas Pipeline': None}}}}

        super().__init__(directory, output_directory,
                         sector='Transportation',
                         level_of_aggregation=level_of_aggregation,
                         fname=fname,
                         categories_dict=self.sub_categories_list)
        self.transport = \
            TransportationIndicators(directory=directory,
                                     output_directory=output_directory,
                                     level_of_aggregation=level_of_aggregation,
                                     lmdi_models=self.model,
                                     base_year=self.base_year,
                                     end_year=self.end_year)

    @staticmethod
    def transportation_data():
        """[summary]

        Returns:
            [type]: [description]
        """
        tedb_18 = \
            pd.read_excel(
                "https://tedb.ornl.gov/wp-content/uploads/2021/02/Table2_07_01312021.xlsx",
                skiprows=9, skipfooter=10, index_col=0, usecols='B:J')
        tedb_18 = tedb_18.rename(columns={'Electricityb': 'Electricity',
                                          'Totalc': 'Total'})
        tedb_18.index = tedb_18.index.str.strip()
        tedb_18 = tedb_18.reset_index()
        print(tedb_18)
        categories = ['HIGHWAY', 'TOTAL HWY & NONHWYc',
                      'Air', 'Rail', 'Pipeline', 'Water']  # 'NONHIGHWAY',
        conditions = [(tedb_18['index'] == r) for r in categories]
        tedb_18.loc[:, 'Category'] = np.select(conditions, categories)
        tedb_18.loc[:, 'Category'] = \
            tedb_18['Category'].replace(to_replace='0',
                                        value=np.nan).fillna(method='ffill')
        tedb_18 = tedb_18[~tedb_18['index'].isin(categories)]
        tedb_18 = tedb_18.rename(columns={'index':
                                          'Mode',
                                          ' Residual fuel oil':
                                          'Residual fuel oil'})
        print(tedb_18.columns)
        tedb_fuel_types = ['Gasoline', 'Diesel fuel', 'Liquefied petroleum gas',
                           'Jet fuel',  'Residual fuel oil', 'Natural gas',
                           'Electricity', 'Total']

        tedb_fuel = pd.melt(tedb_18, id_vars=['Category', 'Mode'],
                            value_vars=tedb_fuel_types, var_name='Fuel Type')
        tedb_fuel.loc[:, 'Year'] = 2018
        tedb_fuel = tedb_fuel.replace({'HIGHWAY': 'Highway',
                                       'Water': 'Waterborne'})
        print(tedb_fuel)
        historical_fuel_consump = \
            pd.read_excel(
                './EnergyIntensityIndicators/Transportation/Data/FuelConsump.xlsx',
                skipfooter=196, skiprows=2, usecols='A:BQ')
        historical_fuel_consump = historical_fuel_consump.fillna(np.nan)
        historical_fuel_consump.loc[0:2, :] = \
            historical_fuel_consump.loc[0:2, :].ffill(axis=1)
        historical_fuel_consump.loc[0, 'Unnamed: 0'] = 'Category'
        historical_fuel_consump.loc[1, 'Unnamed: 0'] = 'Mode'

        historical_fuel_consump = \
            historical_fuel_consump.set_index('Unnamed: 0')

        historical_fuel_consump = historical_fuel_consump.transpose()
        historical_fuel_consump = \
            historical_fuel_consump.rename(columns={'Year': 'Fuel Type'})
        historical_fuel_consump = \
            historical_fuel_consump.reset_index().drop('index', axis=1)

        year_cols = \
            [c for c in historical_fuel_consump.columns if isinstance(c, int)]
        fuel = pd.melt(historical_fuel_consump, id_vars=['Category', 'Mode',
                                                         'Fuel Type'],
                       value_vars=year_cols)

        fuel = fuel.rename(columns={'Unnamed: 0': 'Year'})
        fuel = fuel[(fuel['Fuel Type'] != 'Year') &
                    (fuel['Mode'] != 'Not Used')]

        transport_fuel = pd.concat([fuel, tedb_fuel], axis=0)

        return transport_fuel

    def main(self):
        energy_data = self.transportation_data()
        print('transport energy_data:\n', energy_data)
        print('transport energy_data fuels:\n',
              energy_data['Fuel Type'].unique())

        energy_decomp_data = self.transport.collect_data()

        emissions = \
            self.calculate_emissions(energy_data,
                                     emissions_type='CO2 Factor',
                                     datasource='TEDB')
        return {'energy': energy_data,
                'emissions': emissions}


class ElectricPowerEmissions(CO2EmissionsDecomposition):
    """Class to decompose changes in Emissions from the electric
    power sector
    """
    def __init__(self, directory, output_directory, level_of_aggregation):
        self.directory = directory
        self.output_directory = output_directory
        fname = 'electric_power_sector_emissions'
        self.sub_categories_list = \
            {'Elec Generation Total':
                {'Elec Power Sector':
                    {'Electricity Only':
                        {'Fossil Fuels':
                            {'Coal': None,
                             'Petroleum': None,
                             'Natural Gas': None,
                             'Other Gasses': None},
                         'Nuclear': None,
                         'Hydro Electric': None,
                         'Renewable':
                            {'Wood': None,
                             'Waste': None,
                             'Geothermal': None,
                             'Solar': None,
                             'Wind': None}},
                     'Combined Heat & Power':
                        {'Fossil Fuels':
                            {'Coal': None,
                             'Petroleum': None,
                             'Natural Gas': None,
                             'Other Gasses': None},
                         'Renewable':
                            {'Wood': None,
                             'Waste': None}}},
                 'Commercial Sector': None,
                 'Industrial Sector': None},
             'All CHP':
                {'Elec Power Sector':
                    {'Combined Heat & Power':
                        {'Fossil Fuels':
                            {'Coal': None,
                             'Petroleum': None,
                             'Natural Gas': None,
                             'Other Gasses': None},
                         'Renewable':
                            {'Wood': None,
                             'Waste': None},
                         'Other': None}},
                 'Commercial Sector':
                    {'Combined Heat & Power':
                        {'Fossil Fuels':
                            {'Coal': None,
                             'Petroleum': None,
                             'Natural Gas': None,
                             'Other Gasses': None},
                         'Hydroelectric': None,
                         'Renewable':
                            {'Wood': None,
                             'Waste': None},
                         'Other': None}},
                 'Industrial Sector':
                    {'Combined Heat & Power':
                        {'Fossil Fuels':
                            {'Coal': None,
                             'Petroleum': None,
                             'Natural Gas': None,
                             'Other Gasses': None},
                         'Hydroelectric': None,
                         'Renewable':
                            {'Wood': None,
                             'Waste': None},
                         'Other': None}}}}
        super().__init__(self.directory,
                         self.output_directory,
                         sector='Electric Power',
                         level_of_aggregation=level_of_aggregation,
                         fname=fname,
                         categories_dict=self.sub_categories_list)
        self.elec_data = \
            ElectricityIndicators(directory=self.directory,
                                  output_directory=self.output_directory,
                                  level_of_aggregation='Electric Power',
                                  end_year=2018).collect_data()

    def electric_power_co2(self):
        """[summary]

        Returns:
            [type]: [description]
        """

        elec_gen_total = \
            self.elec_data['Elec Generation Total']
        elec_power_sector = elec_gen_total['Elec Power Sector']
        elec_only = elec_power_sector['Electricity Only']

        comm_sector = elec_gen_total['Commercial Sector']
        ind_sector = elec_gen_total['Industrial Sector']

        all_chp = self.elec_data['All CHP']
        chp_elec_power = all_chp['Elec Power Sector']['Combined Heat & Power']
        chp_comm = all_chp['Commercial Sector']['Combined Heat & Power']
        chp_ind = all_chp['Industrial Sector']['Combined Heat & Power']

        data_cats = [elec_only, comm_sector,
                     ind_sector, chp_elec_power,
                     chp_comm, chp_ind]

        emissions_data = dict()
        for d in data_cats:
            print('d:\n', d)
            for c in d.keys():
                print('c:', c)
                try:
                    activity = d[c]['activity']
                except KeyError:
                    print('d[c] keys', d[c].keys())
                print('activity:\n', activity)
                try:
                    energy = d[c]['energy']
                except KeyError:
                    print('d[c] keys', d[c].keys())
                print('energy:\n', energy)
                for e, e_df in energy.items():
                    print('e:', e)
                    # d = d.rename(columns=self.electric_power_sector())
                    no_emissions = ['solar', 'wind', 'nuclear', 'geothermal']
                    d_emissions = \
                        self.calculate_emissions(e_df,
                                                 emissions_type='CO2 Factor',
                                                 datasource='eia_elec')
                    e_df[no_emissions] = e_df[no_emissions].multiply(0)

                    emissions_data[c] = d_emissions
        print('emissions_data:\n', emissions_data)
        return emissions_data

    def main(self):

        return self.electric_power_co2()


if __name__ == '__main__':
    directory = './EnergyIntensityIndicators/Data'
    output_directory = './Results'

    module_dict = {
        'elec': ElectricPowerEmissions}
                #    'transport': TransportationEmssions}
                #    'industry': IndustrialEmissions}
                #    ,
                #    'residential': ResidentialEmissions,
                #    'commercial': CommercialEmissions}
    levels = {'elec': 'Electric Power',
              'transport': 'Transportation',
              'industry': 'Industry',
              'residential': 'National',
              'commercial': 'Commercial_Total'}
    results = dict()
    for sector, module_ in module_dict.items():
        print('sector:', sector)
        s = module_(directory, output_directory,
                    level_of_aggregation=levels[sector])
        s_data = s.main()
        results = s.calc_lmdi(breakout=True,
                              calculate_lmdi=True,
                              data_dict=s_data)
        print('s_data:\n', s_data)
        print('results:\n', results)

        results[sector] = s_data
