import pandas as pd
import numpy as np

from EnergyIntensityIndicators.LMDI import CalculateLMDI
from EnergyIntensityIndicators.pull_eia_api import GetEIAData
from EnergyIntensityIndicators.utilities.dataframe_utilities \
    import DFUtilities as df_utils


class ElectricityIndicators(CalculateLMDI):

    def __init__(self, directory, output_directory,
                 level_of_aggregation, end_year,
                 lmdi_model=['multiplicative'],
                 base_year=1985):
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
                            'Hydroelectric': None,
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

        self.chp_cats = \
            {'All CHP':
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
    
        self.elec_power_eia = GetEIAData(sector='electricity')
        self.energy_types = ['primary']
        # Consumption of combustible fuels for electricity generation:
        # Commercial and industrial sectors (selected fuels)
        # # elec_power_eia.eia_api(id_='456')
        self.Table84c = \
            pd.read_csv(
                'https://www.eia.gov/totalenergy/data/browser/csv.php?tbl=T07.03C')
        # Consumption of combustible fuels for electricity generation:
        # Electric power sector
        self.Table85c = \
            pd.read_csv(
                'https://www.eia.gov/totalenergy/data/browser/csv.php?tbl=T07.03B')
        self.Table82d = \
            pd.read_csv(
                'https://www.eia.gov/totalenergy/data/browser/csv.php?tbl=T07.02C')

        super().__init__(sector='electric',
                         level_of_aggregation=level_of_aggregation,
                         lmdi_models=lmdi_model,
                         categories_dict=self.sub_categories_list,
                         energy_types=self.energy_types,
                         directory=directory,
                         output_directory=output_directory,
                         base_year=base_year,
                         end_year=end_year)

    # @staticmethod
    # def data():
    #     data_dict = \
    #         {'Elec Generation Total':
    #                 {'Elec Power Sector':
    #                     {'Electricity Only':
    #                         {'Fossil Fuels':
    #                             {'Coal': {'activity': '8.2b10 B * 0.001', 'energy': },
    #                             'Petroleum': {'activity': '8.2b10 D * 0.001', 'energy': },
    #                             'Natural Gas': {'activity': '8.2b10 F * 0.001', 'energy': },
    #                             'Other Gasses': {'activity': '8.2c11 H * 0.001', 'energy': }},
    #                         'Nuclear': {'energy': 'Table8.4b11 L * 0.001', 'activity': 'Table8.2b11 L * 0.001'},
    #                         'Hydroelectric': {'energy': 'Table8.4b11 N * 0.001', 'activity': 'Table8.2b11 O * 0.001'},
    #                         'Renewable':
    #                             {'Wood': {'energy': 'Table 8.5c11 R * 0.001', 'activity': 'Table 8.2c11 R * 0.001'},
    #                             'Waste': {'energy': 'Table 8.5c11 T * 0.001', 'activity': 'Table 8.2c11 T * 0.001'},
    #                             'Geothermal': {'energy': 'Table 8.4b11 T * 0.001', 'activity': 'Table 8.2c11 V * 0.001'},
    #                             'Solar': {'energy': 'Table 8.4b11 V * 0.001', 'activity': 'Table 8.2c11 X * 0.001'},
    #                             'Wind': {'energy': 'Table 8.4b11 X * 0.001', 'activity': 'Table 8.2c11 Z * 0.001'}}},
    #                     'Combined Heat & Power':
    #                         {'Fossil Fuels':
    #                             {'Coal': None,
    #                              'Petroleum': None,
    #                              'Natural Gas': None,
    #                              'Other Gasses': None},
    #                         'Renewable':
    #                             {'Wood': None,
    #                             'Waste': None},
    #                         'Other': None}},
    #                 'Commercial Sector':
    #                     {'Combined Heat & Power':
    #                         {'Fossil Fuels':
    #                             {'Coal': None,
    #                              'Petroleum': None,
    #                              'Natural Gas': None,
    #                              'Other Gasses': None},
    #                          'Hydroelectric': None,
    #                          'Renewable':
    #                             {'Wood': None,
    #                              'Waste': None},
    #                          'Other': None},
    #                 'Industrial Sector':
    #                     {'Combined Heat & Power':
    #                         {'Fossil Fuels':
    #                             {'Coal': None,
    #                              'Petroleum': None,
    #                              'Natural Gas': None,
    #                              'Other Gasses': None},
    #                          'Hydroelectric': None,
    #                          'Renewable':
    #                                 {'Wood': None,
    #                                  'Waste': None},
    #                          'Other': None}}}}

    #     sectors  = ['Elec Power Sector', 'Commercial Sector',
    #                 'Industrial Sector']
    #     chp_data_dict = dict()
    #     for s in sector:
    #         chp_sector_data = data_dict['Elec Generation Total'][s]['Combined Heat & Power']
    #         chp_sector_data = {'Combined Heat & Power': chp_sector_data}
    #         chp_data_dict[s] = chp_sector_data


    @staticmethod
    def get_eia_aer():
        """Prior to 2012, the data for the indicators were taken
        directly from tables published (and downloaded in
        Excel format) from EIA's Annual Energy Review.
        """
        pass

    @staticmethod
    def get_reconciles():
        """ The EIA Annual Energy Review data (pre 2012) for energy
        consumption to produce electricity were generally supplied
        in physical units only (e.g., mcf of natural gas, tons of
        coal, etc.) The values needed to be converted to Btu, and
        still be consistent with aggregate energy consumption for
        this sector as published by EIA. For each major fossil fuel,
        a separate worksheet was developed; these worksheets are
        identified with the suffix “reconcile.” Thus, the worksheet
        “NatGas Reconcile” seeks to produce an estimate of the Btu
        consumption of natural gas used to generate electricity.
        Similar worksheets were developed for coal, petroleum,
        and other fuels.

        ///   Does this need to be done or can download Btu directly instead
        of converting physical units to Btu --> no
        """
        pass

    def process_utility_level_data(self):
        """The indicators for electricity are derived entirely from data
        collected by EIA. Since 2012, the indicators
        are based entirely upon te EIA-923 survey
        """
        eia_923_schedules = pd.read_excel('./')
        # page A-71, 'Net Generation' lower right-hand quadrant?
        net_generation = pd.pivot_table(eia_923_schedules,
                                        columns='EIA Sector Number',
                                        index='AERFuel Type Code',
                                        aggfunc='sum')
        # Should have 18 rows labeled by type of fuel and seven columns
        # plus one for 'Grand Total'. Note: rows is not an arg
        # of pd.pivot_table
        net_generation.loc[:, 'Grand_Total'] = \
            net_generation.sum(axis=1, skipna=True)
        # page A-71, 'Elec Fuel ConsumptionMMBTU' lower
        # right-hand quadrant?
        elec_btu_consumption = pd.pivot_table(eia_923_schedules,
                                              columns='EIA Sector Number',
                                              index='AERFuel Type Code',
                                              aggfunc='sum')
        # Should have 18 rows labeled by type of fuel
        # and seven columns plus one for 'Grand Total'
        elec_btu_consumption.loc[:, 'Grand_Total'] = \
            elec_btu_consumption.sum(axis=1, skipna=True)
        previous_years_net_gen = pd.read_excel('./')
        previous_yeas_elec_btu_consumption = pd.read_excel('./')
        master_net_gen = previous_years_net_gen.concat(net_generation)
        maseter_elec_btu_consumption = \
            previous_yeas_elec_btu_consumption.concat(elec_btu_consumption)
        # Aggregate data?? page A-72 fpr net generation and elec btu
        # consumption
        return None

    @staticmethod
    def reconcile(total, elec_gen, elec_only_plants, chp_elec,
                  assumed_conv_factor, chp_heat):
        """total: df, Btu
        elec_gen: df, Btu
        elec_only_plants: df, Short tons
        chp_elec: df, Short tons
        assumed_conv_factor: float, MMBtu/Ton
        chp_heat
        """
        difference = \
            total.subtract(elec_gen)  # Btu
        implied_conversion_factor = \
            total.divide(elec_only_plants).multiply(1000)  # MMBtu/Ton
        elec_only_billionbtu = \
            elec_only_plants.multiply(assumed_conv_factor[0]).multiply(1000)  # Billion Btu
        chp_elec_billionbtu = \
            chp_elec.multiply(assumed_conv_factor[1]).multiply(0.001)  # Billion Btu
        chp_heat_billionbtu = \
            chp_heat.multiply(assumed_conv_factor[2]).multiply(0.001)  # Billion Btu
        total_fuel = \
            elec_only_billionbtu.add(
                chp_elec_billionbtu).add(chp_heat_billionbtu)

        # Cross Check
        total_short_tons = \
            elec_only_plants + chp_elec + chp_heat  # Short Tons
        implied_conversion_factor_cross = \
            total.divide(total_short_tons).multiply(1000)  # MMBtu/Ton
        implied_conversion_factor_revised = \
            elec_gen.divide(chp_elec.add(elec_only_plants)).multiply(1000)  # MMBtu/Ton

        chp_plants_fuel = \
            implied_conversion_factor_revised.multiply(
                chp_elec).multiply(0.000001)  # Trillion Btu
        elec_only_fuel = \
            elec_gen.multiply(.001).subtract(chp_plants_fuel)  # Trillion Btu
        resulting_total = chp_plants_fuel.add(elec_only_fuel)
        return chp_plants_fuel, elec_only_fuel

    def coal_reconcile(self):
        """Reconcile coal data from physical units into Btu
        """

        energy_consumption_coal = \
            self.elec_power_eia.eia_api(
                id_='TOTAL.CLEIBUS.A', id_type='series')  # Table21f11 column b
        consumption_for_electricity_generation_coal = \
            self.elec_power_eia.eia_api(
                id_='TOTAL.CLEIBUS.A', id_type='series')  # Table84b11 column b

        consumption_combustible_fuels_electricity_generation_coal = \
            self.elec_power_eia.eia_api(
                id_='TOTAL.CLL1PUS.A', id_type='series')  # Table85c11 column
                                                          # B SHOULD BE
                                                          # separated Elec-only/CHP

        consumption_combustible_fuels_useful_thermal_output_coal = \
            self.elec_power_eia.eia_api(
                id_='TOTAL.CLEIPUS.A', id_type='series')  # Table86b11 column B

        assumed_conversion_factor = 20.9
        total = energy_consumption_coal
        elec_gen = consumption_for_electricity_generation_coal
        # should be separate part?
        elec_only_plants = \
            consumption_combustible_fuels_electricity_generation_coal
        # should be separate part?
        chp_elec = \
            consumption_combustible_fuels_electricity_generation_coal
        assumed_conv_factor = assumed_conversion_factor
        chp_heat = consumption_combustible_fuels_useful_thermal_output_coal
        # eia-923 pivot table

        difference = total.subtract(elec_gen)  # Btu
        implied_conversion_factor = \
            total.divide(elec_only_plants).multiply(1000)  # MMBtu/Ton
        elec_only_billionbtu = \
            elec_only_plants.multiply(assumed_conv_factor).multiply(1000)  # Billion Btu
        chp_elec_billionbtu = \
            chp_elec.multiply(assumed_conv_factor).multiply(0.001)  # Billion Btu
        chp_heat_billionbtu = \
            chp_heat.multiply(assumed_conv_factor).multiply(0.001)  # Billion Btu
        total_fuel = \
            elec_only_billionbtu.add(
                chp_elec_billionbtu).add(
                    chp_heat_billionbtu)

        # Cross Check
        total_short_tons = \
            elec_only_plants + chp_elec + chp_heat # Short Tons
        implied_conversion_factor_cross = \
            total.divide(total_short_tons).multiply(1000)  # MMBtu/Ton
        implied_conversion_factor_revised = \
            elec_gen.divide(
                chp_elec.add(elec_only_plants)).multiply(1000) # MMBtu/Ton

        chp_plants_fuel = implied_conversion_factor_revised.multiply(
            chp_elec).multiply(0.000001)  # Trillion Btu
        elec_only_fuel =  elec_gen.multiply(.001).subtract(
            chp_plants_fuel)  # Trillion Btu
        resulting_total = chp_plants_fuel.add(elec_only_fuel)
        return chp_plants_fuel, elec_only_fuel

    def natgas_reconcile(self):
        """Reconcile natural gas data from physical units into Btu
        """

        energy_consumption_natgas = \
            self.elec_power_eia.eia_api(
                id_='TOTAL.NNEIBUS.A', id_type='series')  # Table21f11 column d
        consumption_for_electricity_generation_natgas = \
            self.elec_power_eia.eia_api(
                id_='TOTAL.NNEIBUS.A', id_type='series')  # Table84b11 column f
        consumption_combustible_fuels_electricity_generation_natgas = \
            self.elec_power_eia.eia_api(
                id_='TOTAL.NGL1PUS.A', id_type='series')  # Table85c11 column N
        consumption_combustible_fuels_useful_thermal_output_natgas = \
            self.elec_power_eia.eia_api(
                id_='TOTAL.NGEIPUS.A', id_type='series')  # Table86b11 column M

        # eia-923 pivot table
        """total: df, Billion Btu
        elec_gen: df, Billion Btu
        elec_only_plants: df, Thou. Cu. Ft.
        chp_elec: df, Thou. CF
        assumed_conv_factor: float, MMBtu/Ton
        chp_heat: df, Thou. CF
        """
        total = energy_consumption_natgas
        elec_gen = consumption_for_electricity_generation_natgas
        elec_only_plants = consumption_combustible_fuels_electricity_generation_natgas
        chp_elec = consumption_combustible_fuels_electricity_generation_natgas # different part
        assumed_conv_factor = 1.028
        chp_heat = consumption_combustible_fuels_useful_thermal_output_natgas


        difference = total.subtract(elec_gen) # Billion Btu
        implied_conversion_factor = elec_gen.divide(elec_only_plants).multiply(1000)  # kBtu/CF  * different from coal_reconcile
        elec_only_trillionbtu = elec_only_plants.multiply(assumed_conv_factor).multiply(0.001) # Trillion Btu
        chp_elec_trillionbtu = chp_elec.multiply(assumed_conv_factor).multiply(0.001) # Trillion Btu
        chp_heat_trillionbtu = chp_heat.multiply(assumed_conv_factor).multiply(0.001) # Trillion Btu
        total_fuel = elec_only_trillionbtu.add(chp_elec_trillionbtu).add(chp_heat_trillionbtu)

        # Cross Check
        total_thou_cf = elec_only_plants + chp_elec + chp_heat # Thou. CF
        implied_conversion_factor_cross = total.divide(total_thou_cf).multiply(1000)  # kBtu/CF
        implied_conversion_factor_revised = elec_gen.divide(chp_elec.add(elec_only_plants)).multiply(1000) # MMBtu/Ton

        chp_plants_fuel = implied_conversion_factor_revised.multiply(chp_elec).multiply(0.000001)  # Trillion Btu
        elec_only_fuel =  elec_gen.multiply(.001).subtract(chp_plants_fuel)  # Trillion Btu
        resulting_total = chp_plants_fuel.add(elec_only_fuel)
        return chp_plants_fuel, elec_only_fuel


    def petroleum_reconcile(self):
        """Reconcile petroleum data from physical units into Btu
        """

        energy_consumption_petroleum = self.elec_power_eia.eia_api(id_='TOTAL.PAEIBUS.A', id_type='series')# Table21f11 column F
        consumption_for_electricity_generation_petroleum = self.elec_power_eia.eia_api(id_='TOTAL.PAEIBUS.A', id_type='series')# Table84b11 column D
        consumption_combustible_fuels_electricity_generation_distillate_fuel_oil = self.elec_power_eia.eia_api(id_='TOTAL.DKL1PUS.A', id_type='series')# Table85c11 column D, F, H, J
        consumption_combustible_fuels_electricity_generation_residual_fuel_oil = self.elec_power_eia.eia_api(id_='TOTAL.RFL1PUS.A', id_type='series')
        consumption_combustible_fuels_electricity_generation_other_liquids = self.elec_power_eia.eia_api(id_='TOTAL.OLL1PUS.A', id_type='series')
        consumption_combustible_fuels_electricity_generation_petroleum_coke = self.elec_power_eia.eia_api(id_='TOTAL.PCL1MUS.A', id_type='series')
        consumption_combustible_fuels_electricity_generation_total_petroleum = self.elec_power_eia.eia_api(id_='TOTAL.PAL1PUS.A', id_type='series')

        fuels_list = [consumption_combustible_fuels_electricity_generation_distillate_fuel_oil, consumption_combustible_fuels_electricity_generation_residual_fuel_oil,
                    consumption_combustible_fuels_electricity_generation_other_liquids, consumption_combustible_fuels_electricity_generation_petroleum_coke]
        elec_only_plants_petroleum = df_utils().merge_df_list(fuels_list)
        
        consumption_combustible_fuels_useful_thermal_output_distillate_fuel_oil = self.elec_power_eia.eia_api(id_='TOTAL.DKEIPUS.A', id_type='series')# Table86b11 column D, F, G, I
        consumption_combustible_fuels_useful_thermal_output_residual_fuel_oil = self.elec_power_eia.eia_api(id_='TOTAL.RFEIPUS.A', id_type='series')
        consumption_combustible_fuels_useful_thermal_output_other_liquids = self.elec_power_eia.eia_api(id_='TOTAL.OLEIPUS.A', id_type='series')
        consumption_combustible_fuels_useful_thermal_output_petroleum_coke = self.elec_power_eia.eia_api(id_='TOTAL.PCEIMUS.A', id_type='series')
        consumption_combustible_fuels_useful_thermal_output_total_petroleum = self.elec_power_eia.eia_api(id_='TOTAL.PTEIPUS.A', id_type='series')
        chp_elec = consumption_combustible_fuels_electricity_generation_total_petroleum
        # eia-923 pivot table
        assumed_conversion_factors = [5.825, 6.287, 5.67, 30120]
        chp_plants_petroleum, elec_only_petroleum = self.reconcile(total=energy_consumption_petroleum, elec_gen=consumption_for_electricity_generation_petroleum, 
                                                            elec_only_plants=elec_only_plants_petroleum, chp_elec=chp_elec, assumed_conv_factor=assumed_conversion_factors,
                                                            chp_heat=consumption_combustible_fuels_useful_thermal_output_total_petroleum)
        return chp_plants_petroleum, elec_only_petroleum

    def othgas_reconcile(self):
        """Reconcile other gas data from physical units into Btu
        """

        consumption_for_electricity_generation_fossil_fuels = self.elec_power_eia.eia_api(id_='TOTAL.FFEIBUS.A', id_type='series')# Table84b11 column H
        consumption_combustible_fuels_electricity_generation_oth_gas = self.elec_power_eia.eia_api(id_='TOTAL.OJL1BUS.A', id_type='series')# Table85c11 column P
        consumption_combustible_fuels_useful_thermal_output_othgas = self.elec_power_eia.eia_api(id_='TOTAL.OJEIBUS.A', id_type='series')# Table86b11 column O
        consumption_for_electricity_generation_oth_gas = consumption_for_electricity_generation_fossil_fuels #- consumption_for_electricity_generation_petroleum - consumption_for_electricity_generation_natgas - consumption_for_electricity_generation_coal

        elec_gen = consumption_for_electricity_generation_oth_gas
        elec_only_plants = consumption_combustible_fuels_electricity_generation_oth_gas 
        chp_elec = consumption_combustible_fuels_electricity_generation_oth_gas # ** different part of the series?'
        chp_heat = consumption_combustible_fuels_useful_thermal_output_othgas
        total_other_gas = elec_only_plants.add(chp_elec).add(chp_heat)
        return chp_elec

    @staticmethod
    def format_eii_table(table, factor, name):
        """
        Format EII input data (that isn't from the API). The Month 13 represents annual data.
        """
        table.loc[:, 'YYYYMM'] = table['YYYYMM'].astype(str)
        table.loc[:, 'Month'] = table['YYYYMM'].apply(lambda x: x[-2:])
        table = table[table['Month'] == '13']
        table.loc[:, 'Year'] = table['YYYYMM'].apply(lambda x: x[:-2]).astype(int)

        table = table[['Year', 'Value']].set_index('Year')
        table = table.replace('Not Available', np.nan)
        try:
            table.loc[:, 'Value'] = table['Value'].astype(float)
            table = table.multiply(factor)
        except Exception as e:
            print('error:', e)
            print('Table failed with datatypes:\n', table, table.dtypes)
            print("table['Value'].unique():\n", table['Value'].unique())
            exit()

        table = table.rename(columns={'Value': name})
        return table

    def industrial_sector_chp_renew(self):
        """Collect Industrial_Sector_CHP>Renew data
        """

        wood_energy = self.Table84c[self.Table84c['Description'] == 'Wood Consumption for Electricity Generation, Industrial Sector']
        wood_energy = self.format_eii_table(wood_energy, factor=0.001, name='Wood')
        waste_energy = self.Table84c[self.Table84c['Description'] == 'Waste Consumption for Electricity Generation, Industrial Sector'] # Table 8.4C column R # TBtu
        waste_energy = self.format_eii_table(waste_energy, factor=0.001, name='Waste')


        wood_activity = self.elec_power_eia.eia_api(id_='TOTAL.WDI5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column R
        if wood_activity.empty:
            print('industrial_sector_chp_renew')
            exit()
        waste_activity = self.elec_power_eia.eia_api(id_='TOTAL.WSI5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column T

        data_dict = {'Wood': {'energy': {'primary': wood_energy}, 'activity': wood_activity}, 'Waste': {'energy': {'primary': waste_energy}, 'activity': waste_activity}}
        return data_dict

    def industrial_sector_chp_fossil(self):
        """Collect Industrial_Sector_CHP>Fossil data
        """

        coal_energy = self.Table84c[self.Table84c['Description'] == 'Coal Consumption for Electricity Generation, Industrial Sector'] # Table 8.4c column B # TBtu
        coal_energy = self.format_eii_table(coal_energy, factor=0.001, name='Coal')

        petroleum_energy = self.Table84c[self.Table84c['Description'] == 'Petroleum Consumption for Electricity Generation, Industrial Sector'] # Table 8.4c column D # TBtu
        petroleum_energy = self.format_eii_table(petroleum_energy, factor=0.001, name='Petroleum')

        natgas_energy = self.Table84c[self.Table84c['Description'] == 'Natural Gas Consumption for Electricity Generation, Industrial Sector'] # Table 8.4c column F # TBtu
        natgas_energy = self.format_eii_table(natgas_energy, factor=0.001, name='Natural Gas')

        othgas_energy = self.Table84c[self.Table84c['Description'] == 'Other Gases Consumption for Electricity Generation, Industrial Sector'] # Table 8.4c column H # TBtu
        othgas_energy = self.format_eii_table(othgas_energy, factor=0.001, name='Other Gas')

        coal_activity = self.elec_power_eia.eia_api(id_='TOTAL.CLI5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column B
        petroleum_activity = self.elec_power_eia.eia_api(id_='TOTAL.PAI5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column D
        natgas_activity = self.elec_power_eia.eia_api(id_='TOTAL.NGI5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column F
        othgas_activity = self.elec_power_eia.eia_api(id_='TOTAL.OJI5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column H
        
        data_dict = {'Coal': {'energy': {'primary': coal_energy}, 'activity': coal_activity}, 
                     'Petroleum': {'energy': {'primary': petroleum_energy}, 'activity': petroleum_activity}, 
                     'Natural Gas': {'energy': {'primary': natgas_energy}, 'activity': natgas_activity}, 
                     'Other Gasses': {'energy': {'primary': othgas_energy}, 'activity': othgas_activity}}
        return data_dict

    def industrial_sector_total(self):
        """Collect Industrial_Sector>Total data
        Note: Other includes batteries, chemicals, hydrogen, pitch, purchased steam, sulfur, and miscellaneous technologies
        """
        other_energy = self.Table84c[self.Table84c['Description'] == 'Other Consumption for Electricity Generation, Industrial Sector'] # Table 8.4C11 column AB
        other_energy = self.format_eii_table(other_energy, factor=0.001, name='Other')
        new_index = other_energy.index

        hydroelectric_energy = pd.read_excel('./EnergyIntensityIndicators/Electricity/Data/8.4c11.xlsx', index_col=0, usecols='A,N,AG', 
                                             skipfooter=16, skiprows=9, header=None, names=['Year', 'hydroelectric', 'sector'])
        hydroelectric_energy = hydroelectric_energy[hydroelectric_energy['sector'] == 'industrial'].drop('sector', axis=1).multiply(0.001) # Table 8.4C11 column N               
        hydroelectric_energy.index = hydroelectric_energy.index.astype(int)
        hydroelectric_energy = hydroelectric_energy.reindex(new_index)
        hydroelectric_energy = hydroelectric_energy.interpolate(method='linear', axis=0)

        hydroelectric_activity = self.elec_power_eia.eia_api(id_='TOTAL.HVI5PUS.A', id_type='series').multiply(0.001) # Table 8.2d11 Column P
        other_activity = pd.read_excel('./EnergyIntensityIndicators/Electricity/Data/8.2d11.xlsx', index_col=0, usecols='A,N,AH', skipfooter=16, 
                                       skiprows=9, header=None, names=['Year', 'Other', 'sector']) # Table 8.2d11 Column AD
        other_activity = other_activity[other_activity['sector'] == 'industrial'].drop('sector', axis=1).multiply(0.001)                          
        other_activity.index = other_activity.index.astype(int)
        other_activity = other_activity.reindex(new_index)
        other_activity = other_activity.interpolate(method='linear', axis=0)

        data_dict = {'Hydroelectric': {'energy': {'primary': hydroelectric_energy}, 'activity': hydroelectric_activity},
                     'Other': {'energy': {'primary': other_energy}, 'activity': other_activity}}
        return data_dict

    def comm_sector_chp_renew(self):
        """Collect Comm_Sector_CHP>Renewable data
        As is, these are the same sources as ind sector, but should be different part of columns? 
        """  
        waste_activity = self.elec_power_eia.eia_api(id_='TOTAL.WSC5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column T
        new_index = waste_activity.index
    
        wood_energy =  pd.read_excel('./EnergyIntensityIndicators/Electricity/Data/8.4c11.xlsx', index_col=0, usecols='A,P,AG', 
                                      skipfooter=16, skiprows=9, header=None, names=['Year', 'Wood', 'sector']) # Table 8.4C column P # TBtu
        wood_energy = wood_energy[wood_energy['sector'] == 'commercial'].drop('sector', axis=1).multiply(0.001)                          
        wood_energy.index = wood_energy.index.astype(int)
        wood_energy = wood_energy.reindex(new_index)
        wood_energy = wood_energy.interpolate(method='linear', axis=0)

        waste_energy = pd.read_excel('./EnergyIntensityIndicators/Electricity/Data/8.4c11.xlsx', index_col=0, usecols='A,R,AG', 
                                      skipfooter=16, skiprows=9, header=None, names=['Year', 'Waste', 'sector']) # Table 8.4C column R # TBtu
        waste_energy = waste_energy[waste_energy['sector'] == 'commercial'].drop('sector', axis=1).multiply(0.001)                          
        waste_energy.index = waste_energy.index.astype(int)
        waste_energy = waste_energy.reindex(new_index)
        waste_energy = waste_energy.interpolate(method='linear', axis=0)
    
        wood_activity = pd.read_excel('./EnergyIntensityIndicators/Electricity/Data/8.2d11.xlsx', index_col=0, usecols='A,R,AH', skipfooter=16, 
                                       skiprows=9, header=None, names=['Year', 'Wood', 'sector'])
        print('wood_activity:\n', wood_activity)
        wood_activity = wood_activity[wood_activity['sector'] == 'commercial'].drop('sector', axis=1).multiply(0.001)                          
        wood_activity.index = wood_activity.index.astype(int)
        wood_activity = wood_activity.reindex(new_index)
        wood_activity = wood_activity.interpolate(method='linear', axis=0)
    
        data_dict = {'Wood': {'energy': {'primary': wood_energy}, 'activity': wood_activity}, 
                     'Waste': {'energy': {'primary': waste_energy}, 'activity': waste_activity}}
        return data_dict

    def comm_sector_chp_fossil(self):
        """Collect Comm_Sector_CHP>Fossil data
        """

        coal_energy = self.Table84c[self.Table84c['Description'] == 'Coal Consumption for Electricity Generation, Commercial Sector'] # Table 8.4c column B # TBtu
        coal_energy = self.format_eii_table(coal_energy, factor=0.001, name='Coal')
        new_index = coal_energy.index

        petroleum_energy = self.Table84c[self.Table84c['Description'] == 'Petroleum Consumption for Electricity Generation, Commercial Sector'] # Table 8.4c column D # TBtu
        petroleum_energy = self.format_eii_table(petroleum_energy, factor=0.001, name='Petroleum')

        natgas_energy = self.Table84c[self.Table84c['Description'] == 'Natural Gas Consumption for Electricity Generation, Commercial Sector'] # Table 8.4c column F # TBtu
        natgas_energy = self.format_eii_table(natgas_energy, factor=0.001, name='Natural Gas')

        othgas_energy = pd.read_excel('./EnergyIntensityIndicators/Electricity/Data/8.4c11.xlsx', index_col=0, usecols='A,H,AG', skipfooter=16, skiprows=9, header=None, names=['Year', 'Other Gas', 'sector'])# Table 8.4c column H # TBtu
        othgas_energy = othgas_energy[othgas_energy['sector'] == 'commercial'].drop('sector', axis=1).replace(' ', np.nan).fillna(np.nan).astype(float).multiply(0.001) 
        othgas_energy.index = othgas_energy.index.astype(int)
        othgas_energy = othgas_energy.reindex(new_index)
        othgas_energy = othgas_energy.interpolate(method='linear', axis=0)
    
        coal_activity = self.elec_power_eia.eia_api(id_='TOTAL.CLC5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column B
        petroleum_activity = self.elec_power_eia.eia_api(id_='TOTAL.PAC5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column D
        natgas_activity = self.elec_power_eia.eia_api(id_='TOTAL.NGC5PUS.A', id_type='series').multiply(0.001) # Table 8.2d column F
        
        othgas_activity = pd.read_excel('./EnergyIntensityIndicators/Electricity/Data/8.2d11.xlsx', index_col=0, usecols='A,H,AH', skipfooter=16, skiprows=9, header=None, names=['Year', 'Other Gas', 'sector']) # Table 8.2d column H
        othgas_activity = othgas_activity[othgas_activity['sector'] == 'commercial'].drop('sector', axis=1).replace(' ', np.nan).replace('\x96', np.nan).fillna(np.nan).astype(float).multiply(0.001)             
        othgas_activity.index = othgas_activity.index.astype(int)
        othgas_activity = othgas_activity.reindex(new_index)
        othgas_activity = othgas_activity.interpolate(method='linear', axis=0)

        # Industrial and Commercial have same spreadsheet sources but different parts of the columns
        data_dict = {'Coal': {'energy': {'primary': coal_energy}, 'activity': coal_activity}, 
                     'Petroleum': {'energy': {'primary': petroleum_energy}, 'activity': petroleum_activity}, 
                     'Natural Gas': {'energy': {'primary': natgas_energy}, 'activity': natgas_activity}, 
                     'Other Gasses': {'energy': {'primary': othgas_energy}, 'activity': othgas_activity}}
        return data_dict

    def comm_sector_total(self, new_index):
        """Collect Comm_Sector>Total data
        Note: Other includes batteries, chemicals, hydrogen, pitch, purchased steam, sulfur, and miscellaneous technologies

        """    
        hydroelectric_energy = pd.read_excel('./EnergyIntensityIndicators/Electricity/Data/8.4c11.xlsx', index_col=0, usecols='A,N,AG', skipfooter=16, skiprows=9, header=None, names=['Year', 'Hydroelectric', 'sector']) # Table 8.4C11 column N
        hydroelectric_energy = hydroelectric_energy[hydroelectric_energy['sector'] == 'commercial'].drop('sector', axis=1).multiply(0.001)                          
        hydroelectric_energy.index = hydroelectric_energy.index.astype(int)
        hydroelectric_energy = hydroelectric_energy.reindex(new_index)
        hydroelectric_energy = hydroelectric_energy.interpolate(method='linear', axis=0)

        other_energy = pd.read_excel('./EnergyIntensityIndicators/Electricity/Data/8.4c11.xlsx', index_col=0, usecols='A,AB,AG', skipfooter=16, skiprows=9, header=None, names=['Year', 'Other', 'sector']) # Table 8.4C11 column AB
        other_energy = other_energy[other_energy['sector'] == 'commercial'].drop('sector', axis=1).replace(' ', np.nan).fillna(np.nan).astype(float).multiply(0.001)                          
        other_energy.index = other_energy.index.astype(int)
        other_energy = other_energy.reindex(new_index)
        other_energy = other_energy.interpolate(method='linear', axis=0)

        hydroelectric_activity = pd.read_excel('./EnergyIntensityIndicators/Electricity/Data/8.2d11.xlsx', index_col=0, usecols='A,P,AH', skipfooter=16, skiprows=9, header=None, names=['Year', 'Hydroelectric', 'sector']) # Table 8.2d11 Column P
        hydroelectric_activity = hydroelectric_activity[hydroelectric_activity['sector'] == 'commercial'].drop('sector', axis=1).replace(' ', np.nan).fillna(np.nan).astype(float).multiply(0.001)                          
        hydroelectric_activity.index = hydroelectric_activity.index.astype(int)
        hydroelectric_activity = hydroelectric_activity.reindex(new_index)
        hydroelectric_activity = hydroelectric_activity.interpolate(method='linear', axis=0)
    
        other_activity = pd.read_excel('./EnergyIntensityIndicators/Electricity/Data/8.2d11.xlsx', index_col=0, usecols='A,AD,AH', skipfooter=16, skiprows=9, header=None, names=['Year', 'Other', 'sector']) # Table 8.2d11 Column AD
        other_activity = other_activity[other_activity['sector'] == 'commercial'].drop('sector', axis=1).replace(' ', np.nan).replace('\x96', np.nan).fillna(np.nan).astype(float).multiply(0.001)                          
        other_activity.index = other_activity.index.astype(int)
        other_activity = other_activity.reindex(new_index)
        other_activity = other_activity.interpolate(method='linear', axis=0)

        data_dict = {'Hydroelectric': {'energy': {'primary': hydroelectric_energy}, 'activity': hydroelectric_activity}, 
                     'Other': {'energy': {'primary': other_energy}, 'activity': other_activity}}
        return data_dict

    def elec_power_sector_chp_renew(self):
        """Collect Elec_Power_Sector_CHP>Renewable data
        """

        wood_energy = self.Table85c[self.Table85c['Description'] == 'Wood Consumption for Electricity Generation, Electric Power Sector'] # Table 8.5C column R # TBtu
        wood_energy = self.format_eii_table(wood_energy, factor=0.001, name='Wood')

        waste_energy = self.Table85c[self.Table85c['Description'] == 'Waste Consumption for Electricity Generation, Electric Power Sector'] # Table 8.5C column T # TBtu
        waste_energy = self.format_eii_table(waste_energy, factor=0.001, name='Waste')

        wood_activity = self.elec_power_eia.eia_api(id_='TOTAL.WDEGPUS.A', id_type='series').multiply(0.001) # Table 8.2c column R
        if wood_activity.empty:
            print('elec_power_sector_chp_renew')
            exit()
        waste_activity = self.elec_power_eia.eia_api(id_='TOTAL.WSEGPUS.A', id_type='series').multiply(0.001) # Table 8.2c column T

        data_dict = {'Wood': {'energy': {'primary': wood_energy}, 'activity': wood_activity}, 
                     'Waste': {'energy': {'primary': waste_energy}, 'activity': waste_activity}}
        return data_dict

    def elec_power_sector_chp_fossil(self):
        """Collect Elec_Power_Sector_CHP>Fossil data
        """

        coal_energy, elec_only_fuel = self.coal_reconcile() 
        petroleum_energy, elec_only_petroleum = self.petroleum_reconcile()
        natgas_energy, elec_only_fuel_nat_gas = self.natgas_reconcile() 
        othgas_energy = self.othgas_reconcile()
        
        coal_activity = self.elec_power_eia.eia_api(id_='TOTAL.CLEGPUS.A', id_type='series').multiply(0.001) # Table 8.2c column B
        petroleum_activity = self.elec_power_eia.eia_api(id_='TOTAL.PAEGPUS.A', id_type='series').multiply(0.001) # Table 8.2c column D
        natgas_activity = self.elec_power_eia.eia_api(id_='TOTAL.NGEGPUS.A', id_type='series').multiply(0.001) # Table 8.2c column F
        othgas_activity = self.elec_power_eia.eia_api(id_='TOTAL.OJEGPUS.A', id_type='series').multiply(0.001) # Table 8.2c column H

        data_dict = {'Coal': {'energy': {'primary': coal_energy}, 'activity': coal_activity}, 
                     'Petroleum': {'energy': {'primary': petroleum_energy}, 'activity': petroleum_activity}, 
                     'Natural Gas': {'energy': {'primary': natgas_energy}, 'activity': natgas_activity}, 
                     'Other Gasses': {'energy': {'primary': othgas_energy}, 'activity': othgas_activity}}
        return data_dict

    def elec_power_sector_chp_total(self):
        """Collect Elec_Power_Sector_CHP>Total data
        """

        other_energy = self.Table85c[self.Table85c['Description'] == 'Other Consumption for Electricity Generation, Electric Power Sector'] # Table 8.5c column V
        other_energy = self.format_eii_table(other_energy, factor=0.001, name='Other')
         
        other_activity = pd.read_csv('./EnergyIntensityIndicators/Electricity/Data/other_activity82c.csv') # Table 8.2c columnAD
        other_activity = other_activity[other_activity['sector'] == 'chp'].drop('sector', axis=1).replace(' ', np.nan).fillna(np.nan).astype(float).multiply(0.001)                          
        other_activity.index = other_activity.index.astype(int)

        data_dict = {'Other': {'energy': {'primary': other_energy}, 'activity': other_activity}}
        return data_dict

    def electricity_only_renew(self):
        """Collect Eletricity-Only>Renewable data
        """

        wood_activity = self.elec_power_eia.eia_api(id_='TOTAL.WDL1BUS.A', id_type='series').multiply(0.001)  # Table 8.3d I, 8.5C R
        if wood_activity.empty:
            print('electricity_only_renew')
            exit()
        waste_activity = self.elec_power_eia.eia_api(id_='TOTAL.WSL1BUS.A', id_type='series').multiply(0.001) # Table 8.3d J, 8.5C T
        geothermal_activity = self.elec_power_eia.eia_api(id_='TOTAL.GEEGBUS.A', id_type='series').multiply(0.001) # Table 8.4 T
        solar_activity = self.elec_power_eia.eia_api(id_='TOTAL.SOEGBUS.A', id_type='series').multiply(0.001)  # Table 8.4 V
        wind_activity = self.elec_power_eia.eia_api(id_='TOTAL.WYEGBUS.A', id_type='series').multiply(0.001) # Table 8.4 X

        wood_energy = self.elec_power_eia.eia_api(id_='TOTAL.WDEGPUS.A', id_type='series').multiply(0.001)  # Table 8.2b S, 8.2c R
        waste_energy = self.elec_power_eia.eia_api(id_='TOTAL.WSEGPUS.A', id_type='series').multiply(0.001)  # Table 8.2b T, 8.2c T
        geothermal_energy = self.elec_power_eia.eia_api(id_='TOTAL.GEEGPUS.A', id_type='series').multiply(0.001)  # Table 8.2b V, 8.2c V
        solar_energy = self.elec_power_eia.eia_api(id_='TOTAL.SOEGPUS.A', id_type='series').multiply(0.001) # Table 8.2b X, 8.2c X
        wind_energy = self.elec_power_eia.eia_api(id_='TOTAL.WYEGPUS.A', id_type='series').multiply(0.001)  # Table 8.2b Z, 8.2c Z

        data_dict = {'Wood': {'energy': {'primary': wood_energy}, 'activity': wood_activity}, 
                     'Waste': {'energy': {'primary': waste_energy}, 'activity': waste_activity}, 
                     'Geothermal': {'energy': {'primary': geothermal_energy}, 'activity': geothermal_activity}, 
                     'Solar': {'energy': {'primary': solar_energy}, 'activity': solar_activity}, 
                     'Wind': {'energy': {'primary': wind_energy}, 'activity': wind_activity}}
        return data_dict

    def electricity_only_fossil(self):
        """Collect Eletricity-Only>Fossil data
        """

        chp_plants_fuel, coal_energy = self.coal_reconcile() 
        chp_plants_petroleum, petroleum_energy = self.petroleum_reconcile()
        natgas_energy, elec_only_fuel_nat_gas = self.natgas_reconcile()
        othgas_energy = self.othgas_reconcile()
        
        coal_activity = self.elec_power_eia.eia_api(id_='TOTAL.CLEGPUS.A', id_type='series').multiply(0.001)  # 8.2b B, 8.2c B
        petroleum_activity = self.elec_power_eia.eia_api(id_='TOTAL.PAEGPUS.A', id_type='series').multiply(0.001)  # 8.2b D, 8.2c D
        natgas_activity = self.elec_power_eia.eia_api(id_='TOTAL.NGEGPUS.A', id_type='series').multiply(0.001)  # 8.2b B, 8.2c F
        othgas_activity = self.elec_power_eia.eia_api(id_='TOTAL.OJEGPUS.A', id_type='series').multiply(0.001) # 8.2c H

        data_dict = {'Coal': {'energy': {'primary': coal_energy}, 'activity': coal_activity}, 
                     'Petroleum': {'energy': {'primary': petroleum_energy}, 'activity': petroleum_activity}, 
                     'Natural Gas': {'energy': {'primary': natgas_energy}, 'activity': natgas_activity}, 
                     'Other Gasses': {'energy': {'primary': othgas_energy}, 'activity': othgas_activity}}
        return data_dict

    def electricity_only_total(self):
        """Collect Eletricity-Only>Total data
        """

        nuclear_energy = self.elec_power_eia.eia_api(id_='TOTAL.NUEGBUS.A', id_type='series').multiply(0.001) # 8.4b L
        hydroelectric_energy = self.elec_power_eia.eia_api(id_='TOTAL.HVEGBUS.A', id_type='series').multiply(0.001)  # 8.4b N

        nuclear_activity = self.elec_power_eia.eia_api(id_='TOTAL.NUEGPUS.A', id_type='series').multiply(0.001) # 8.2b L
        hydroelectric_activity = self.elec_power_eia.eia_api(id_='TOTAL.HVEGPUS.A', id_type='series').multiply(0.001)  # 8.2b O

        data_dict = {'Hydroelectric': {'energy': {'primary': hydroelectric_energy}, 'activity': hydroelectric_activity}, 
                     'Nuclear': {'energy': {'primary': nuclear_energy}, 'activity': nuclear_activity}}
        return data_dict

    def collect_data(self): 
        """Collect all data for use in the decomposition of energy use in the
        Electric Power sector
        """

        industrial_sector_chp_renew = self.industrial_sector_chp_renew()
        industrial_sector_chp_fossil = self.industrial_sector_chp_fossil()
        industrial_sector_total = self.industrial_sector_total()
        industrial_sector_total['Fossil Fuels'] = industrial_sector_chp_fossil
        industrial_sector_total['Renewable'] = industrial_sector_chp_renew

        comm_sector_chp_renew = self.comm_sector_chp_renew()
        comm_sector_chp_fossil = self.comm_sector_chp_fossil()
        new_index = comm_sector_chp_fossil['Coal']['activity'].index
        comm_sector_total = self.comm_sector_total(new_index=new_index)
        comm_sector_total['Fossil Fuels'] = comm_sector_chp_fossil
        comm_sector_total['Renewable'] = comm_sector_chp_renew

        elec_power_sector_chp_renew = self.elec_power_sector_chp_renew()
        elec_power_sector_chp_fossil = self.elec_power_sector_chp_fossil()
        elec_power_sector_chp_total = self.elec_power_sector_chp_total()
        elec_power_sector_chp_total['Fossil Fuels'] = elec_power_sector_chp_fossil  # Coal, Petroleum, Natural Gas, Other Gasses
        elec_power_sector_chp_total['Renewable'] = elec_power_sector_chp_renew  # Wood, Waste

        electricity_only_renew = self.electricity_only_renew()
        electricity_only_fossil = self.electricity_only_fossil()
        electricity_only_total = self.electricity_only_total() # Nuclear, Hydro
        electricity_only_total['Fossil Fuels'] = electricity_only_fossil # Coal, Petroleum, Natural Gas, Other Gasses
        electricity_only_total['Renewable'] = electricity_only_renew # Wood, Waste, Geothermal, Solar, Wind

        data_dict = {'Elec Generation Total': 
                        {'Elec Power Sector': 
                            {'Electricity Only': electricity_only_total, # Fossil Fuels, Nuclear, Hydro, Renewable
                             'Combined Heat & Power': elec_power_sector_chp_total},  # Fossil Fuels, Renewable
                        'Commercial Sector':
                            {'Combined Heat & Power':
                                comm_sector_total}, 
                        'Industrial Sector':
                            {'Combined Heat & Power':
                                industrial_sector_total}}}
        return data_dict   

    def main(self, breakout, calculate_lmdi): 
        """Calculate decomposition for the Electric Power sector
        """
        data_dict = self.collect_data()
        results_dict, formatted_results = self.get_nested_lmdi(level_of_aggregation=self.level_of_aggregation, 
                                                               breakout=breakout, calculate_lmdi=calculate_lmdi, 
                                                               raw_data=data_dict, lmdi_type='LMDI-I')
        return results_dict, formatted_results

if __name__ == '__main__':
    indicators = ElectricityIndicators(directory='./EnergyIntensityIndicators/Data', 
                                       output_directory='./Results', 
                                       level_of_aggregation='Elec Generation Total', end_year=2018,
                                       lmdi_model=['multiplicative', 'additive'])
    indicators.main(breakout=True, calculate_lmdi=True)