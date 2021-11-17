
from numpy.core.fromnumeric import mean
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
    def weighted(x, cols, w="weights"):
        """Calculate weighted average

        Args:
            x ([type]): [description]
            cols ([type]): [description]
            w (str, optional): [description]. Defaults to "weights".

        Returns:
            (pd.Series): Weighted average
        """
        return pd.Series(np.average(x[cols], weights=x[w], axis=0), cols)

    def get_mean_factor(self, emissions_factors,
                        input_cols, new_name, portions=None):
        """Calculate average of given emissions factors

        Args:
            emissions_factors (pd.DataFrame): emissions factors
            input_cols (list): List of emissions
                               factors to average
            new_name (str): Name of resulting factor
            portions (dict): Optional. Weights for weighted average
                             (keys are cols)

        Returns:
            ef (pd.DataFrame): emissions factors with new type
        """
        subset = \
            emissions_factors[emissions_factors['Fuel Type'].isin(input_cols)]
        subset['weights'] = np.nan
        if portions:
            subset_ = []
            for k, v in portions.items():
                fuel_ = subset[subset['Fuel Type'] == k]
                fuel_['weights'] = v
                subset_.append(fuel_)

            subset = pd.concat(subset_, axis=0)
            subset.loc[:, 'weighted_value'] = \
                subset['value'].multiply(subset['weights'].values)

            grouped = \
                subset.groupby(by=['Unit', 'Variable'])
            mean_df = grouped.sum()
            mean_df = mean_df.drop('value', axis=1)

            mean_df['value'] = \
                mean_df['weighted_value'].divide(mean_df['weights'].values)
            mean_df = mean_df.drop('weighted_value', axis=1)

        else:
            grouped = \
                subset.groupby(by=['Unit', 'Variable'])
            mean_df = grouped.mean()

        mean_df.loc[:, 'Fuel Type'] = new_name
        mean_df.loc[:, 'Category'] = 'Merged Category'

        mean_df = mean_df.reset_index()
        mean_df = mean_df[['Category', 'Fuel Type',
                           'Unit', 'value', 'Variable']]

        ef = pd.concat([emissions_factors, mean_df], axis=0)
        return ef

    def epa_emissions_data(self):
        """Read and process EPA emissions factors data

        Returns:
            emissions_factors (DataFrame): EPA emissions factors data
        """
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
                                  new_name='Gasohol',
                                  portions={'Ethanol (100%)': 0.16,
                                            'Motor Gasoline': 0.84})
        ef = self.get_mean_factor(ef,
                                  input_cols=['Diesel Fuel',
                                              'Motor Gasoline'],
                                  new_name='School',
                                  portions={'Diesel Fuel': 0.9,
                                            'Motor Gasoline': 0.1})

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
            mecs_data (DataFrame): Industrial sector energy use
                                   by fuel type

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
                    'Net Electricity(b)': 'US Average',
                    'Residual': 'Residual Fuel Oil',
                    'Distillate': 'Distillate Fuel Oil',
                    'Distillate Fuel Oil(c)': 'Distillate Fuel Oil',

                    'Nat. Gas': 'Natural Gas',
                    'Natural Gas(d)': 'Natural Gas',
                    'Natural Gas': 'Natural Gas',
                    'HGL (excluding natural gasoline)':
                        'Liquefied Petroleum Gases (LPG)',
                    'Coal':
                        'Mixed (Industrial Sector)',
                    'Coke Coal and Breeze':
                        'Coal Coke',
                    'Coke and Breeze': 'Coal Coke',
                    'Coke': 'Coal Coke',
                    'LPG': 'Liquefied Petroleum Gases (LPG)',
                    'Diesel': 'Diesel Fuel',
                    'LP Gas': 'Liquefied Petroleum Gases (LPG)',
                    'HGL (excluding natural gasoline)(e)': 'Liquefied Petroleum Gases (LPG)',
                    'Gasoline': 'Motor Gasoline',
                    'Gas': 'Natural Gas'}

        mecs_data = mecs_data.rename(columns=mapping_)
        mecs_data = mecs_data.drop('Total Fuel', axis=1, errors='ignore')

        return mecs_data

    @staticmethod
    def electric_epa_mapping(elec_data):
        """Rename elec power sector data (from EIA) columns so
        that labels match EPA emissions factors labels

        Args:
            elec_data (pd.DataFrame): Energy consumption for the
                                      elec power sector by fuel type

        Returns:
            elec_data (pd.DataFrame): Elec data with column names
                                      that match EPA emissions data
                                      labels
        """
        rename_dict = {col: col[:col.find('Consumption')].strip()
                       for col in elec_data.columns if 'Consumption' in col}
        rename_dict2 = {col: col[:col.find('Consumed')].strip()
                        for col in elec_data.columns if 'Consumed' in col}

        intersect = []
        for key, value in rename_dict2.items():
            if value in rename_dict.values():
                intersect.append(key)

        elec_data = elec_data.drop(intersect, axis=1)
        rename_dict.update(rename_dict2)
        others = {'Electricity Net Generation From Wood, Electric Power Sector, Annual, Million Kilowatthours': 'Wood',
                  'Electricity Net Generation From Waste, Electric Power Sector, Annual, Million Kilowatthours': 'Waste'}
        rename_dict.update(others)
        elec_data = elec_data.rename(columns=rename_dict)
        mapping_ = {'Coal': 'Mixed (Electric Power Sector)',
                    'Natural Gas': 'Natural Gas',
                    'Other Gases': 'Fuel Gas',
                    'Other Gas': 'Fuel Gas',
                    'Waste': 'Municipal Solid Waste',
                    'Wood': 'Wood and Wood Residuals',
                    'hydroelectric': 'Hydroelectric',
                    'Other': 'US Average',
                    'Other Petroleum Liquids': 'Liquefied Petroleum Gases (LPG)'}  # might be wrong
        elec_data = elec_data.rename(columns=mapping_)
        elec_data = elec_data.drop('Total Petroleum', axis=1, errors='ignore')
        return elec_data

    @staticmethod
    def tedb_epa_mapping(tedb_data):
        """Rename transportation sector data (from TEDB) columns so
        that labels match EPA emissions factors labels

        Args:
            tedb_data (pd.DataFrame): Energy consumption for the
                                      transportation sector by fuel type

        Returns:
            tedb_data_ (pd.DataFrame): transportation data with
                                       column names that match
                                       EPA emissions data labels
        """

        unit_coversion = {'Diesel Fuel & Distillate (1,000 bbl)': 1000/42,
                          'Residual Fuel Oil (1,000 bbl)': 1000/42}
        unit_coversion = {k: unit_coversion[k] for k in
                          unit_coversion.keys() if k in tedb_data.columns}
        if len(unit_coversion) > 0:
            tedb_data = tedb_data.assign(**unit_coversion).mul(tedb_data)

        if 'Domestic Operations ' and 'International Operations ' in tedb_data.columns:
            tedb_data['Air'] = \
                tedb_data[['Domestic Operations ',
                           'International Operations ']].sum(axis=1)
            tedb_data = tedb_data[['Air']]

        mapping = {
            'Gasoline': 'Motor Gasoline',  # ef is in gallon
            'Gasoline (million gallons)': 'Motor Gasoline',  # ef is in gallon
            'Gasohol': 'Gasohol',  # ef is in gallon
            'Diesel (million gallons)': 'Diesel Fuel', # ef is in gallon
            'Diesel': 'Diesel Fuel', # ef is in gallon
            'CNG': 'Compressed Natural Gas (CNG)', # ef is in scf
            'LNG': 'Liquefied Natural Gas (LNG)', # ef is in gallons
            'Bio Diesel ': 'Biodiesel (100%)', # ef is in gallons
            'Diesel Fuel & Distillate (1,000 bbl)': # ef is in gallon (42 gallons in a barrel)
                'Diesel Fuel',
            'Residual Fuel Oil (1,000 bbl)': 'Residual Fuel Oil',  # ef is in gallon (42 gallons in a barrel)
            'Jet fuel (million gallons)': 'Aviation Gasoline',  # ef is per gallon
            'Electricity (GWhrs)': 'US Average', # ef is kg/MWh
            'Distillate Fuel Oil': 'Diesel Fuel',  # ef is per gallon
            'Natural Gas (million cu. ft.)': 'Natural Gas',  # ef is per scf
            'Electricity (million kWh)': 'US Average', # ef is kg/MWh
            'Diesel fuel': 'Diesel Fuel',  # ef is per gallon
            'Liquefied petroleum gas':
                'Liquefied Petroleum Gases (LPG)',  # ef is per gallon
            'LPG': 'Liquefied Petroleum Gases (LPG)',  # ef is per gallon
            'Air': 'Aviation Gasoline',  # ef is per gallon
            'Jet fuel': 'Aviation Gasoline',  # ef is per gallon
            'Residual fuel oil': 'Residual Fuel Oil',  # ef is per gallon
            'Natural gas': 'Natural Gas',  # ef is per scf
            'Electricity': 'US Average',  # ef is kg/MWh
            'Intercity': 'Diesel Fuel'  # ef is in gallon
            }  

        tedb_data_ = tedb_data.rename(columns=mapping)
        tedb_data_ = tedb_data_.drop('School (million bbl)',
                                     axis=1, errors='ignore')
        tedb_data_ = \
            tedb_data_.drop(['Total Energy (Tbtu)',
                             'Total Energy (Tbtu) ',
                             'Total Energy (Tbtu) - old series'],
                            axis=1, errors='ignore')

        return tedb_data_

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
        factors_df = \
            factors_df[factors_df['Variable'].isin(
                ['Heat Content (HHV)', emissions_type])]

        factors_df['value'] = factors_df['value'].fillna(1)

        factors_df = \
            factors_df.groupby(['Category', 'Fuel Type']).prod()
        factors_df = factors_df.reset_index()
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
            energy_data (pd.DataFrame): energy consumption by fuel data
            emissions_type (str): Type of emissions factor.
                                  Defaults to 'CO2 Factor'.
            datasource (str): 'SEDS', 'MECS', 'eia_elec', or 'TEDB'.
                              Data source for energy consumption by
                              fuel data. Defaults to 'SEDS'.
        Returns:
            emissions_data (pd.DataFrame): Emissions by fuel type
            energy_data (pd.DataFrame): Energy consumption by fuel data
                                        converted to MMBtu, if necessary,
                                        with columns matching emissions data
                                        

        TODO: Handle Other category (should not be dropped)
        """

        # Also need to convert TEDB fuel (in gal or ft3) to MMBtu
        mapping_heat = {
            'Motor Gasoline': 125000,  # ef is in gallon
            'Gasohol': 120900,  # ef is in gallon
            'Diesel Fuel': 138700,  # ef is in gallon
            'Compressed Natural Gas (CNG)': 129400,  # ef is in scf
            'Liquefied Natural Gas (LNG)': 84800,  # ef is in gallons
            'Liquefied Petroleum Gases (LPG)': 91300,
            'Biodiesel (100%)': 128520,  # ef is in gallons
            'Residual Fuel Oil': 149700,  # ef is in gallon (42 gallons in a barrel)
            'Aviation Gasoline': 120900,  # ef is per gallon
            'US Average': 10339,  # ef is kg/MWh; Btu/kWh
            'Natural Gas': 1031,  # ef is per scf
            'Diesel Fuel': 138700,  # ef is per gallon
            }  

        energy_data = energy_data.drop('region', axis=1, errors='ignore')
        emissions_factors = self.epa_emissions_data()

        if datasource == 'SEDS':
            energy_data = self.epa_eia_crosswalk(energy_data)
        elif datasource == 'MECS':
            energy_data = self.mecs_epa_mapping(energy_data)
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
            raise KeyError('Emissions data does not contain ' +
                           'all energy sources')

        emissions_data = \
            energy_data.multiply(emissions_factors.to_numpy())

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
        except KeyError:
            if census_region:
                energy_data = energy_data[energy_data['Census Region'] == '0']
            else:
                pass

        # Convert TEDB fuels to MMBtu
        if datasource == 'TEDB':
            try:
                energy_data = \
                    energy_data.apply(lambda x: mapping_heat[x.name]*x, axis=0)
            except KeyError:
                print('energy_data.columns.tolist() not in dataframe:',
                      energy_data.columns.tolist())
                for t in energy_data.columns.tolist():
                    if t not in mapping_heat.keys():
                        print('t not in list:', t)
                raise KeyError(
                    'Heat content conversion data does not contain ' +
                    'all energy sources'
                    )

        return emissions_data, energy_data

    def calc_lmdi(self, breakout, calculate_lmdi, data_dict):
        """Calculate decomposition of CO2 emissions for the U.S. economy

        TODO: Could simply call lmdi_gen main with a few slight adjustments
        """
        results_dict, formatted_results = \
            self.get_nested_lmdi(
                level_of_aggregation=self.level_of_aggregation,
                breakout=breakout, calculate_lmdi=calculate_lmdi,
                raw_data=data_dict, lmdi_type=self.gen.lmdi_type)
        return results_dict


class SEDSEmissionsData(CO2EmissionsDecomposition):
    """Class to collect energy consumption by fuel type
    from SEDS (for the Residential and Commercial Sectors)
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
            cw (pd.DataFrame): Crosswalk between states and census
                               regions
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
        """Rename EIA data (from SEDS) columns so
        that labels match EPA emissions factors labels

        Args:
            eia_data (pd.DataFrame): Energy consumption for the
                                     Commercial or Residential
                                     sector by fuel type

        Returns:
            eia_data (pd.DataFrame): EIA data with
                                     column names that match
                                     EPA emissions data labels
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
        """Gather SEDS API endpoint for sector, state, fuel
        combination

        Args:
            sector (str): abbreviation for the commercial
                          or residential sector
            state (str): abbreviation for state
            fuel (str): Key in endpoints dict to select

        Returns:
            (str): SEDS API endpoint
        """
        endpoints = {
            'All Petroleum Products': f'SEDS.PA{sector}P.{state}.A',
            'Coal': f'SEDS.CL{sector}P.{state}.A',
            'Distillate Fuel Oil': f'SEDS.DF{sector}P.{state}.A',
            'Electrical System Energy Losses': f'SEDS.LO{sector}B.{state}.A',
            'Electricity Sales': f'SEDS.ES{sector}P.{state}.A',
            'Fuel Ethanol including Denaturant': f'SEDS.EN{sector}P.{state}.A',
            'Fuel Ethanol excluding Denaturant': f'SEDS.EM{sector}B.{state}.A',
            'Geothermal': f'SEDS.GE{sector}B.{state}.A',
            'Hydrocarbon gas liquids': f'SEDS.HL{sector}P.{state}.A',
            'Hydroelectricity': f'SEDS.HY{sector}P.{state}.A',
            'Kerosene': f'SEDS.KS{sector}P.{state}.A',
            'Motor Gasoline': f'SEDS.MG{sector}P.{state}.A',
            'Natural Gas including Supplemental Gaseous Fuels':
                f'SEDS.NG{sector}P.{state}.A',
            'Petroleum Coke': f'SEDS.PC{sector}P.{state}.A',
            'Propane': f'SEDS.PQ{sector}P.{state}.A',
            'Residual Fuel Oil': f'SEDS.RF{sector}P.{state}.A',
            'Solar Energy': f'SEDS.SOR7P.{state}.A',
            'Total (per Capita)': f'SEDS.TE{sector[0]}PB.{state}.A',
            'Total Energy excluding Electrical System Energy Losses':
                f'SEDS.TN{sector}B.{state}.A',
            'Waste': f'SEDS.WS{sector}B.{state}.A',
            'Wind Energy': f'SEDS.WY{sector}P.{state}.A',
            'Wood': f'SEDS.WD{sector}B.{state}.A',
            'Wood and Waste': f'SEDS.WW{sector}B.{state}.A'
            }
        return endpoints[fuel]

    def collect_seds(self, sector, states):
        """SEDS energy consumption data (in physical units unless
        unavailable, in which case in Btu-- indicated by P or
        B in endpoint)

        Args:
            sector (str): abbreviation for the commercial
                          or residential sector ('CC' or 'RC'
                          respectively)
            states (list): States in region-- used to collect
                           SEDS API data

        Returns:
            fuels_data (pd.DataFrame): Energy consumption by fuel
                                       for region (by state) and sector
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
                             weather_data, total_label,
                             weather_activity, sector='Residential'):
        """Collect weather factors for 'deliv' energy type (from 'elec' and
        'fuels' weather factors) for sector

        Args:
            energy_data (dict): Dictionary of dataframes of energy data
                                from the energy decomposition (keys are
                                'elec' and 'fuels')
            activity_input_data (dict): activity data for the sector
            weather_data (dict): weather factors for 'elec' and 'fuels'
            total_label (str): level total name
            weather_activity (str): Activity data to use in weather data
                                    inference (?)
            sector (str, optional): 'Residential' or 'Commercial'.
                                    Defaults to 'Residential'.

        Returns:
            weather_data (pd.DataFrame): Weather factors for 'deliv'.
        """
        energy_type = 'deliv'
        energy_input_data = \
            self.calculate_energy_data(energy_type, energy_data)
        energy_input_data = energy_input_data.drop('Energy_Type', axis=1)
        energy_data['deliv'] = energy_input_data
        if total_label not in energy_input_data.columns:
            energy_input_data = \
                df_utils().create_total_column(
                    energy_input_data, total_label)

        for a, a_df in activity_input_data.items():
            if isinstance(a_df, pd.Series):
                a_df = a_df.to_frame()
            a_df = \
                df_utils().create_total_column(
                    a_df, total_label)
            activity_input_data[a] = a_df
        setattr(self, 'energy_types', ['elec', 'fuels', 'deliv'])
        base_weather = weather_data

        if self.sector == 'Commercial':
            input_data = energy_data
            weather_data = \
                self.weather_adjustment(
                    input_data,
                    base_weather,
                    energy_type)

        elif self.sector == 'Residential':
            input_data = dict()
            for e in self.energy_types:
                type_df = energy_data[e]
                activity_df = activity_input_data[weather_activity]
                nominal_intensity = \
                    self.nominal_energy_intensity(type_df, activity_df)
                input_data[e] = nominal_intensity

            weather_data = \
                self.weather_adjustment(
                    input_data,
                    base_weather,
                    energy_type)

        setattr(self, 'energy_types', ['all'])
        return weather_data

    def seds_energy_data(self, sector):
        """Collect SEDS energy consumption data
        by fuel type and region for sector

        Args:
            sector (str): 'Commercial' or 'Residential'

        Returns:
            all_data (pd.DataFrame): Energy Use by fuel type
                                     and region.
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
