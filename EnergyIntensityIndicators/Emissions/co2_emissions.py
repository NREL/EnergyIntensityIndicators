
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
                 sector, config_path, categories_dict,
                 level_of_aggregation):
        self.sector = sector
        self.eia = GetEIAData('emissions')
        self.config_path = config_path
        self.level_of_aggregation = level_of_aggregation

        super().__init__(sector,
                         level_of_aggregation=self.level_of_aggregation,
                         categories_dict=categories_dict,
                         directory=directory,
                         output_directory=output_directory,
                         primary_activity=None,
                         unit_conversion_factor=1,
                         weather_activity=None,
                         use_yaml_config=True,
                         config_path=self.config_path)

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
        subset = \
            emissions_factors[emissions_factors['Fuel Type'].isin(input_cols)]

        grouped = \
            subset.groupby(by=['Category', 'Unit', 'Variable'])
        mean_df = grouped.mean()
        mean_df.loc[:, 'Fuel Type'] = new_name
        mean_df = mean_df.reset_index()
        mean_df = mean_df[['Category', 'Fuel Type',
                           'Unit', 'value', 'Variable']]

        ef = pd.concat([emissions_factors, mean_df], axis=0)

        return ef

    def epa_emissions_data(self):
        """Read and process EPA emissions factors data

        Returns:
            emissions_factors (DataFrame): [description]
        """
        print('os.getcwd(),', os.getcwd())
        try:
            ef = pd.read_csv(
                    './EnergyIntensityIndicators/Data/EPA_emissions_factors.csv')
        except FileNotFoundError:
            os.chdir('..')
            print('changed dir:', os.getcwd())
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

        emissions_factors['value'] = \
            emissions_factors['value'].apply(lambda x: str(x).replace(',', ''))

        emissions_factors['value'] = emissions_factors['value'].astype(float)
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

        return ef

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
        rename_dict = {col: col.strip() for col in mecs_data.columns}
        mecs_data = mecs_data.rename(columns=rename_dict)
        mapping_ = {'Waste Gas': 'Fuel Gas',
                    'Petroleum Coke': 'Petroleum Coke',
                    'Wood Chips, Bark': 'Wood and Wood Residuals',
                    'Waste Oils/Tars and Waste Materials': 'Used Oil',
                    'steam': 'Steam and Heat',  # From Table 7
                    'Net Electricity': 'Us Average',  # From Table 6,  Total Output Emissions Factors CO2 Factor
                    'Electricity': 'US Average',
                    'Residual': 'Residual Fuel Oil',
                    'Distillate': 'Distillate Fuel Oil',
                    'Nat. Gas': 'Natural Gas',
                    'Natural Gas': 'Natural Gas',
                    'HGL (excluding natural gasoline)':
                        'Liquefied Petroleum Gases (LPG)',
                    'Coal':
                        'Mixed (Industrial Sector)',
                    'Coke Coal and Breeze':
                        'Coal Coke',
                    'Coke': 'Coal Coke',
                    'LPG': 'Liquefied Petroleum Gases (LPG)',
                    'Diesel': 'Diesel Fuel',
                    'LP Gas': 'Liquefied Petroleum Gases (LPG)',
                    'Gasoline': 'Motor Gasoline',
                    'Gas': 'Natural Gas'}

        mecs_data = mecs_data.rename(columns=mapping_)
        mecs_data = mecs_data.drop('Total Fuel', axis=1, errors='ignore')

        return mecs_data

    @staticmethod
    def electric_epa_mapping(elec_data):
        """[summary]

        Returns:
            [type]: [description]
        """

        rename_dict = {col: col[:col.find('Consumption')].strip()
                       for col in elec_data.columns if 'Consumption' in col}
        rename_dict2 = {col: col[:col.find('Consumed')].strip()
                        for col in elec_data.columns if 'Consumed' in col}
        rename_dict.update(rename_dict2)
        others = {'Electricity Net Generation From Wood, Electric Power Sector, Annual, Million Kilowatthours': 'Wood'}
        rename_dict.update(others)
        elec_data = elec_data.rename(columns=rename_dict)

        mapping_ = {'Coal': 'Mixed (Electric Power Sector)',
                    'Natural Gas': 'Natural Gas',
                    'Other Gases': 'Fuel Gas',
                    'Waste': 'Municipal Solid Waste',
                    'Wood': 'Wood and Wood Residuals',
                    'hydroelectric': 'Hydroelectric'}
        elec_data = elec_data.rename(columns=mapping_)
        return elec_data

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
        # irrelevant_fuels = [f for f in tedb_data.columns
        #                     if f not in mapping.keys()]
        # tedb_data = tedb_data[~tedb_data['Fuel Type'].isin(irrelevant_fuels)]
        tedb_data = tedb_data.rename(columns=mapping)
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
        no_emissions = ['Solar', 'Wind',
                        'Nuclear', 'Geothermal',
                        'Hydroelectric']
        e_data = [0]*len(no_emissions)
        no_emissions_df = pd.DataFrame(data=e_data,
                                       index=no_emissions,
                                       columns=['value'])

        no_emissions_df.index.name = 'Fuel Type'
        fuel_factor_df = pd.concat([fuel_factor_df, no_emissions_df], axis=0)
        fuel_factor_df = fuel_factor_df.transpose()
        return fuel_factor_df

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
        print('energy_data:\n', energy_data)
        emissions_factors = self.epa_emissions_data()

        if datasource == 'SEDS':
            energy_data = self.epa_eia_crosswalk(energy_data)
        elif datasource == 'MECS':
            energy_data = self.mecs_epa_mapping(energy_data)
            # energy_data = energy_data.reset_index()
        elif datasource == 'eia_elec':
            energy_data = self.electric_epa_mapping(energy_data)
        elif datasource == 'TEDB':
            energy_data = self.tedb_epa_mapping(energy_data)

        energy_data = energy_data.drop('Total', axis=1,
                                       errors='ignore')

        ## TEMPORARY!! (need to add these factors to csv)
        energy_data = energy_data.drop(['Other'], axis=1,
                                       errors='ignore')

        emissions_factors = self.get_factor(emissions_factors,
                                            emissions_type)

        try:
            emissions_factors = \
                emissions_factors[energy_data.columns.tolist()]
        except KeyError:
            print('energy_data.columns.tolist() not in dataframe:',
                  energy_data.columns.tolist())
            for t in energy_data.columns.tolist():
                if t not in emissions_factors.columns.tolist():
                    print('t not in list:', t)
            print('emissions_factors columns:', emissions_factors.index)
            raise KeyError('Emissions data does not contain' +
                           'all energy sources')

        print('emissions_factors:\n', emissions_factors)
        print('energy_data:\n', energy_data)
        print('emissions_factors cols:\n', emissions_factors.columns)
        print('energy_data cols:\n', energy_data.columns)
        emissions_data = \
            energy_data.multiply(emissions_factors.to_numpy())
        print('emissions_data:\n', emissions_data)

        try:
            energy_data.loc[:, 'Census Region'] = \
                energy_data.loc[:, 'Census Region'].astype(int).astype(str)
            census_region = True

        except KeyError:
            census_region = False
        try:
            # ensure string Census Regions are ints (at least some
            # start as floats-- need to match)
            emissions_data.loc[:, 'Census Region'] = \
                emissions_data['Census Region'].astype(int).astype(str)
            print('emissions_data:\n', emissions_data)
        except KeyError:
            if census_region:
                energy_data = energy_data[energy_data['Census Region'] == '0']
            else:
                pass

        return emissions_data, energy_data

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
                raw_data=data_dict, lmdi_type=self.gen.lmdi_type)
        return results_dict


class SEDSEmissionsData(CO2EmissionsDecomposition):
    """Class to [Summary]

    """
    def __init__(self, directory,
                 output_directory, sector,
                 fname, categories_dict,
                 level_of_aggregation):

        super().__init__(directory=directory,
                         output_directory=output_directory,
                         sector=sector,
                         categories_dict=categories_dict,
                         config_path=fname,
                         level_of_aggregation=level_of_aggregation)

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
        if ethanol_cols in eia_data.columns.to_list():
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

        irrelevant_fuels = [f for f in eia_data.columns if f not in
                            mapping_.keys() and f != 'Census Region']
        eia_data = eia_data.drop(irrelevant_fuels, axis=1)

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

    def collect_weather_data(self,
                             energy_data,
                             activity_input_data,
                             weather_data, total_label):

        energy_type = 'deliv'
        energy_input_data = \
            self.calculate_energy_data(energy_type, energy_data)
        energy_input_data = energy_input_data.drop('Energy_Type', axis=1)
        # energy_input_data = \
        #     df_utils().create_total_column(
        #         energy_input_data, total_label)

        for a, a_df in activity_input_data.items():
            if isinstance(a_df, pd.Series):
                a_df = a_df.to_frame()
            a_df = \
                df_utils().create_total_column(
                    a_df, total_label)
            activity_input_data[a] = a_df

        lower_level_intensity_df = pd.DataFrame()
        data = self.prepare_lmdi_inputs(energy_type,
                                        energy_input_data,
                                        activity_input_data,
                                        lower_level_intensity_df,
                                        total_label, weather_data)
        weather_data = data['structure']['weather']
        return weather_data

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
        census_regions = {'4': 'West', '3': 'South',
                          '2': 'Midwest', '1': 'Northeast',
                          '0': 'US'}

        grouped = states.groupby(states['Census Region'])
        all_data = dict()
        for g in sector_data[sector]['regions']:
            region_states = grouped.get_group(g)
            region_data = self.collect_seds(
                                        sector=sector_data[sector]['abbrev'],
                                        states=region_states['USPC'].unique())

            region_data.loc[:, 'Census Region'] = census_regions[str(g)]
            region_data = region_data.fillna(np.nan)
            all_data[census_regions[str(g)]] = region_data

        return all_data


if __name__ == '__main__':
    pass
    # directory = './EnergyIntensityIndicators/Data'
    # output_directory = './Results'

    # module_dict = {
    #     # 'elec': ElectricPowerEmissions}
    #             #    'transport': TransportationEmssions}  #,
    #             #    'industry': IndustrialEmissions} #,
    #             #    'residential': ResidentialEmissions}  #,
    #                'commercial': CommercialEmissions}
    # levels = {'elec': 'Elec Generation Total',
    #           'transport': 'All_Transportation',
    #           'industry': 'Industry',
    #           'residential': 'National',
    #           'commercial': 'Commercial_Total'}
    # results = dict()
    # for sector, module_ in module_dict.items():
    #     print('sector:', sector)
    #     s = module_(directory, output_directory,
    #                 level_of_aggregation=levels[sector])
    #     s_data = s.main()
    #     results = s.calc_lmdi(breakout=True,
    #                           calculate_lmdi=True,
    #                           data_dict=s_data)
    #     print('s_data:\n', s_data)
    #     print('results:\n', results)

    #     results[sector] = s_data
