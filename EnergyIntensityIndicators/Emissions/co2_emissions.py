
import pandas as pd
import numpy as np

# from EnergyIntensityIndicators.industry import IndustrialIndicators
# from EnergyIntensityIndicators.residential import ResidentialIndicators
# from EnergyIntensityIndicators.commercial import CommercialIndicators
from EnergyIntensityIndicators.electricity import ElectricityIndicators
# from EnergyIntensityIndicators.transportation import TransportationIndicators
from EnergyIntensityIndicators.LMDI import CalculateLMDI
# from EnergyIntensityIndicators.economy_wide import EconomyWide
from EnergyIntensityIndicators.pull_eia_api import GetEIAData
from EnergyIntensityIndicators.utilities.dataframe_utilities \
    import DFUtilities as df_utils


class CO2EmissionsDecomposition:  # CalculateLMDI
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
    def __init__(self, directory, output_directory, sector,
                 level_of_aggregation=None, lmdi_model='multiplicative',
                 base_year=1985, end_year=2018):

        self.sector = sector
        self.eia = GetEIAData('emissions')

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
    def epa_emissions_data():
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
        print('emissions_factors:\n', emissions_factors)
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
                    'Blast Furnace/Coke Oven Gases':
                       ['Blast Furnace Gas',
                        'Coke Oven Gas'],  # take average
                    'Waste Gas': 'Fuel Gas',
                    'Petroleum Coke': 'Petroleum Coke',
                    'Pulping Liquor or Black Liquor':
                        ['North American Softwood',
                         'North American Hardwood'],  # take average
                    'Wood Chips, Bark': 'Wood and Wood Residuals',
                    'Waste Oils/Tars and Waste Materials': 'Used Oil',
                    'steam': 'Steam and Heat',  # From Table 7
                    'Net Electricity': 'Us Average',  # From Table 6,  Total Output Emissions Factors CO2 Factor
                    'Residual Fuel Oil':
                       ['Residual Fuel Oil No. 5',  # take average
                        'Residual Fuel Oil No. 6'],
                    'Distillate Fuel Oil':
                      ['Distillate Fuel Oil No. 1',  # take average
                       'Distillate Fuel Oil No. 2',
                       'Distillate Fuel Oil No. 4'],
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
    def tedb_epa_mapping(tedb_data):

        # tedb_data = tedb_data.drop('Total Energy (Tbtu) - old series')
        print('tedb_data:\n', tedb_data)
        print('tedb_data cols:\n', tedb_data.columns)

        print('tedb_data fuel types:\n', tedb_data['Fuel Type'].unique())

        # tedb_data = tedb_data.replace({'Gasoline (million gallons)': 'Gasoline',
        #                                'Diesel (million gallons)': 'Diesel',
        #                                'Diesel fuel': 'Diesel',
        #                                'LPG': 'Liquefied petroleum gas',
        #                                })
        # mapping = {'All Fuel':
        #            'Gasoline': 'Motor Gasoline'
        #            'Gasohol': ['Motor Gasoline', 'Ethanol (100%)']
        #            'Diesel': 'Diesel Fuel'
        #            'CNG': 'Compressed Natural Gas (CNG)',
        #            'LNG': 'Liquefied Natural Gas (LNG)',
        #            'Bio Diesel ': 'Biodiesel (100%)'
        #            'Other':
        #            'School':
        #            'School (million bbl)':
        #            'Intercity':
        #             'Diesel Fuel & Distillate (1,000 bbl)': 
        #                                     ['Diesel Fuel',
        #                                      'Distillate Fuel Oil No. 1',
        #                                      'Distillate Fuel Oil No. 2',
        #                                      'Distillate Fuel Oil No. 4']  # Take Average
        #              'Residual Fuel Oil (1,000 bbl)'
        #             'Domestic Operations ':
        #             'International Operations ':
        #              'Jet fuel (million gallons)':
        #             'Electricity (GWhrs)':
        #              'Total Energy (Tbtu) ':
        #              'Total Energy (Tbtu)':
        #             'Distillate Fuel Oil': ['Distillate Fuel Oil No. 1',
        #                                     'Distillate Fuel Oil No. 2',
        #                                     'Distillate Fuel Oil No. 4']  # Take Average
        #              'Natural Gas (million cu. ft.)': 'Natural Gas',
        #             'Electricity (million kWh)': 'US Average'  # From table 6
        #             'Diesel fuel':
        #             'Liquefied petroleum gas': 'Liquefied Petroleum Gases (LPG)'
        #             'Jet fuel':
        #              'Residual fuel oil': 'Residual Fuel Oil'
        #              'Natural gas': 'Natural Gas',
        #               'Electricity': 'US Average'  # From table 6
        #             }

        # mapping_ = {'Jet fuel': 'Aviation Gasoline',
        #             'Bio Diesel ': 'Biodiesel (100%)',
        #             'Natural gas': 'Compressed Natural Gas (CNG)'
        #             'Diesel fuel': 'Diesel Fuel',
        #             'Gasohol': 'Ethanol (100%)',
        #             'Kerosene-Type Jet Fuel',
        #             'Liquefied Natural Gas (LNG)',
        #             'Liquefied Petroleum Gases (LPG)',
        #             'Motor Gasoline',
        #             'Residual Fuel Oil'}

        tedb_data = tedb_data.rename(columns=mapping_)
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
            pass
        elif datasource == 'TEDB':
            energy_data = self.tedb_epa_mapping(energy_data)

        print('energy_data:\n', energy_data)
        emissions_factors = self.get_factor(emissions_factors,
                                            emissions_type)

        emissions_factors = emissions_factors.to_frame(name='Emissions Factors')
        emissions_factors = emissions_factors.transpose()
        print('emissions_factors:\n', emissions_factors)
        print('emissions_factors cols', emissions_factors.columns)
        print('energy_data.columns.tolist():', energy_data.columns.tolist())
        try:
            emissions_factors = emissions_factors[energy_data.columns.tolist()]
        except KeyError:
            print('energy_data.columns.tolist() not in dataframe:', energy_data.columns.tolist())
            return None
        print('emissions_factors:\n', emissions_factors)

        emissions_data = \
            energy_data.multiply(emissions_factors, axis='columns')
        emissions_data.loc[:, 'Census Region'] = \
            emissions_data['Census Region'].astype(int).astype(str)
        print('emissions_data:\n', emissions_data)

        energy_data.loc[:, 'Census Region'] = \
            energy_data['Census Region'].astype(int).astype(str)
        fuel_mix = self.get_fuel_mix(energy_data)
        print(fuel_mix)
        return emissions_data, fuel_mix

    def main(self, breakout, calculate_lmdi):
        """Calculate decomposition of CO2 emissions for the U.S. economy

        TODO: allow for different sectors to have different types of energy
              and commercial and residential to have weather adjustment
              (TODO carried over from EconomyWide)

        """
        data_dict = self.collect_emissions_data()
        results_dict, formatted_results = \
            self.get_nested_lmdi(
                level_of_aggregation=self.level_of_aggregation,
                breakout=breakout, calculate_lmdi=calculate_lmdi,
                raw_data=data_dict, lmdi_type='LMDI-I')
        return results_dict


class SEDSEmissionsData(CO2EmissionsDecomposition):
    """Class to [Summary]

    """
    def __init__(self, directory, output_directory, sector):

        super().__init__(directory, output_directory, sector,
                         level_of_aggregation=None,
                         lmdi_model='multiplicative',
                         base_year=1985, end_year=2018)

    @staticmethod
    def state_census_crosswalk():
        """Match states with Census Regions

        Returns:
            [type]: [description]
        """
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
        mapping_ = {'Coal':
                      'Mixed (Commercial Sector)',
                    'Distillate Fuel Oil': ['Distillate Fuel Oil No. 1',
                                            'Distillate Fuel Oil No. 2',
                                            'Distillate Fuel Oil No. 4'],  # take average
                    'Fuel Ethanol including Denaturant':
                        'Ethanol (100%)',
                    'Fuel Ethanol excluding Denaturant': 'Ethanol (100%)',
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
                    'Residual Fuel Oil': ['Residual Fuel Oil No. 5',
                                          'Residual Fuel Oil No. 6'],  # take average
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
                        f'SEDS.HL{sector}P.{state}.A',  # is the Lpg factor right for HGL?
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
    def __init__(self, directory, output_directory):
        super().__init__(directory, output_directory,
                         sector='Residential')

    def main(self):
        energy_data = self.seds_energy_data(sector='residential')

        emissions = \
            self.calculate_emissions(energy_data,
                                     emissions_type='CO2 Factor',
                                     datasource='SEDS')
        return {'energy': energy_data,
                'emissions': emissions}


class CommercialEmissions(SEDSEmissionsData):
    def __init__(self, directory, output_directory):
        super().__init__(directory, output_directory,
                         sector='Commercial')

    def main(self):
        energy_data = self.seds_energy_data(sector='commercial')

        emissions = \
            self.calculate_emissions(energy_data,
                                     emissions_type='CO2 Factor',
                                     datasource='SEDS')

        return {'energy': energy_data,
                'emissions': emissions}


class IndustrialEmissions(CO2EmissionsDecomposition):
    def __init__(self, directory, output_directory):
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
                             'Urea Consumption for NonAgricultural Purposes': None,
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
                        {'Petroleum and Natural Gas': None,
                         'Other Mining':
                            {'noncombustion':
                                {'Coal Mining': None},
                             'combustion': None},
                         'Support Activities': None},
                     'Construction': None,
                     'Waste':
                        {'Landfills':
                            {'noncombustion': None},
                         'Composting':
                            {'noncombustion': None}},
                     'Energy':
                        {'noncombustion':
                            {'Stationary Combustion': None,
                             'Non-Energy Use of Fuels': None},
                         'combustion': None}}}}

        super().__init__(directory, output_directory,
                         sector='Industry',
                         level_of_aggregation=None,
                         lmdi_model='multiplicative',
                         base_year=1985, end_year=2018)

    def main(self):
        energy_data = []
        emissions = \
            self.calculate_emissions(energy_data,
                                     emissions_type='CO2 Factor',
                                     datasource='MECS')
        return {'energy': energy_data,
                'emissions': emissions}


class TransportationEmssions(CO2EmissionsDecomposition):
    def __init__(self, directory, output_directory):
        super().__init__(directory, output_directory,
                         sector='Transportation',
                         level_of_aggregation=None,
                         lmdi_model='multiplicative',
                         base_year=1985, end_year=2018)
    @staticmethod
    def tedb_fuel_types(tedb_data):
        """[summary]

        Args:
            tedb_data ([type]): [description]
        """
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
        fuel = fuel[(fuel['Fuel Type'] != 'Year') & (fuel['Mode'] != 'Not Used')]

        print('fuel:\n', fuel)

        print('fuel cols:\n', fuel.columns)
        transport_fuel = pd.concat([fuel, tedb_fuel], axis=0)
        print('transport_fuel:\n', transport_fuel)
        print('transport_fuel fuels', transport_fuel['Fuel Type'].unique())

        return transport_fuel

    def main(self):
        energy_data = self.transportation_data()
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
    def __init__(self, directory, output_directory):
        self.directory = directory
        self.output_directory = output_directory
        super().__init__(directory, output_directory,
                         sector='Electric Power',
                         level_of_aggregation=None,
                         lmdi_model='multiplicative',
                         base_year=1985, end_year=2018)
    @staticmethod
    def electric_power_sector():
        """[summary]

        Returns:
            [type]: [description]
        """
        mapping_ = {'Coal': 'Mixed (Electric Power Sector)',
                    'Petroleum':
                        ['Distillate Fuel Oil No. 1',
                         'Distillate Fuel Oil No. 2',
                         'Distillate Fuel Oil No. 4',
                         'Residual Fuel Oil No. 5',
                         'Residual Fuel Oil No. 6'],  # take average
                    'Natural Gas': 'Natural Gas',
                    'Other Gases': 'Fuel Gas',
                    'Waste': 'Municipal Solid Waste',
                    'Wood': 'Wood and Wood Residuals'}
        return mapping_

    def electric_power_co2(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        elec_data = \
            ElectricityIndicators(directory=self.directory,
                                  output_directory=self.output_directory,
                                  level_of_aggregation='Electric Power',
                                  end_year=2018).collect_data()
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
                    d_emissions = \
                        self.calculate_emissions(e_df,
                                                 emissions_type='CO2 Factor',
                                                 datasource='eia_elec')
                    emissions_data[c] = d_emissions
        return emissions_data

    def main(self):

        return self.electric_power_co2()


if __name__ == '__main__':
    # indicators = \
    #     CO2EmissionsDecomposition(directory='./EnergyIntensityIndicators/Data',
    #                               output_directory='./Results',
    #                               level_of_aggregation=None,
    #                               end_year=2018,
    #                               lmdi_model=['multiplicative', 'additive'])

    # indicators.mecs_sic_crosswalk()
    # indicators.mecs_data_by_year()
    directory = './EnergyIntensityIndicators/Data'
    output_directory = './Results'

    module_dict = {'elec': ElectricPowerEmissions,
                #    'transport': TransportationEmssions,
                #    'industry': IndustrialEmissions,
                   'residential': ResidentialEmissions,
                   'commercial': CommercialEmissions}
    results = dict()
    for sector, module_ in module_dict.items():
        print('sector:', sector)
        s = module_(directory, output_directory)
        s_data = s.main()
        print('s_data:\n', s_data)
        results[sector] = s_data
