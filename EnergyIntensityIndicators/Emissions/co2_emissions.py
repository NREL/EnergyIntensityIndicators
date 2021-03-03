
import pandas as pd
import os

# from EnergyIntensityIndicators.industry import IndustrialIndicators
# from EnergyIntensityIndicators.residential import ResidentialIndicators
# from EnergyIntensityIndicators.commercial import CommercialIndicators
# from EnergyIntensityIndicators.electricity import ElectricityIndicators
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
                            datasource='EIA'):
        """Calculate emissions from the product of energy_data and 
        emissions_factor

        Parameters:
            energy_data (df): 
            emission_factor (df, series or float): 
        
        Returns: 
            emissions_data (df): 
        """
        emissions_factors = self.epa_emissions_data()

        if datasource == 'EIA':
            energy_data = self.epa_eia_crosswalk(energy_data)
        
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
    ef = indicators.epa_emissions_data()

    data = indicators.seds_energy_data(sector='residential')
    print(data)
    print(data.columns)
    emissions_data = indicators.calculate_emissions(data, emissions_type='CO2 Factor', 
                                                    datasource='EIA')

    print('emissions_data:\n', emissions_data)
