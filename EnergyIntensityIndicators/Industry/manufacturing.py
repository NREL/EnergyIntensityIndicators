import pandas as pd
from datetime import datetime
import os
from functools import reduce

from EnergyIntensityIndicators.pull_bea_api import BEA_api
from EnergyIntensityIndicators.get_census_data import (Econ_census, Asm)
from EnergyIntensityIndicators.Industry.asm_price_fit import Mfg_prices

from EnergyIntensityIndicators.utilites.standard_interpolation import standard_interpolation

class Manufacturing:
    """Class to collect and process manufacturing data for the industrial sector
    """
    def __init__(self):
        self.end_year = datetime.now().year

        # 2014_MECS = 'https://www.eia.gov/consumption/manufacturing/data/2014/'  # Table 4.2

        # Table 3.1 and 3.2 (MECS total fuel consumption)  Table 3.1 shows energy
        # consumption by fuel in physical units, including the total across all fuels expressed in trillion Btu and
        # electricity in kWh. From Table 3.1, total fuel consumption in Btu can be calculated as difference between
        # total energy and electricity consumption after conversion to Btu. Table 3.2 only differs from Table 3.1 by
        # showing all fuel types in Btu.

        # For 2014, the values for total energy consumption and electricity consumption, both defined in terms of
        # trillion Btu, from Table 3.2 are transferred to spreadsheet ind_hap3. Worksheet MECS_Fuel in this
        # spreadsheet has been used to collect the fuel consumption estimates for all the MECS dating back to the
        # first MECS in 1985. The 2014 data are located in the cell range F218:F238.
        # The first six NAICS sectors are aggregated into three sectors (311-312, 313-314, and 315-316) as a part
        # of the set of manufacturing indicators. The energy consumption data under this revised sectoring
        # classification are shown in the columns to the right, columns O and P.


    @staticmethod
    def mecs_fuel():
        historical_mecs_31_32 = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/historical_mecs_31_32.csv')
        mecs_fuel = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/MECS_Fuel_Historical.csv').set_index('NAICS')
        return mecs_fuel

    def get_historical_mecs(self):
        """Read in historical MECS csv, format (as in e.g. Coal (MECS) Prices)
        """
        historical_mecs = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/MECS_Fuel_Historical.csv') # NAICS ARE ALREADY AGGREGATED
        return historical_mecs

    def manufacturing_prices(self):
        """Call ASM API method from Asm class in get_census_data.py
        Specify three-digit NAICS Codes
        """
        fuel_types = ['Gas', 'Coal', 'Distillate', 'Residual', 'LPG', 'Coke', 'Other']
        asm_cols = {'Gas': "Gas $/MBTU",
                    'Coal': 'Pre-2013 Price Estimate $/MMBtu',
                    'Distillate': "Distilate $/MBTU",
                    'Residual': "Residual $/MBTU", 
                    'LPG': 'LPG (Use Propane Price) cents/gal', 
                    'Coke': "Anthracite $/MBTU", 
                    'Other': "Bituminous $/MBTU"}
        naics = [311, 312, 313, 314, 315, 316, 321, 322, 323, 324, 
                 325, 326, 327, 331, 332, 333, 334, 335, 336, 337, 339]

        # asm_price_data = []
        # for f in fuel_types: 
        #     predicted_fuel_price = Mfg_prices().main(latest_year=self.end_year, fuel_type=f, naics=naics, asm_col_map=asm_cols)
        #     asm_price_data.append(predicted_fuel_price)

        # asm_price_data = pd.concat(asm_price_data, axis=1)
        asm_price_data = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/asm_price_fit.csv')
        asm_price_data['Year'] = asm_price_data['Year'].astype(int)
        asm_price_data['NAICS'] = asm_price_data['NAICS'].astype(int)

        return asm_price_data

    # Corresponds to data in MECS_Fuel tab in Indhap3 spreadsheet, which
    # is connected to the MECS_Annual_Fuel1 and MECS_Annual_Fuel2
    # tabs in the same spreadsheet.
    def import_mecs_fuel(self):
        """
        Imports MECS data on fuel use by 3-digit NAICS code. In the future,
        these values will need to be manually downloaded from Table 3.2
        and added to csv.
        """
        fallhap3b = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/fallhap3b.csv')
        return fallhap3b
    
    # Data used in ASMdata_010220.xlsx[3DNAICS]
    def call_census_data(self):
        """
        Use Census_api class to call fuel and electricity expenditures from
        Annual Survey of Manufacturers and Economic Census (for years ending
        with 2 and 7).
        """
        economic_census = Econ_census()
        economic_census_years = list(range(1987, self.end_year + 1, 5))       
        e_c_data = {str(y): economic_census.get_data(year=y) 
                    for y in economic_census_years}
        print(e_c_data)

        asm_ = Asm()
        census_years = list(range(1987, self.end_year + 1))     
        asm_data = {str(y): asm_.get_data(year=y) 
                    for y in census_years}

        return e_c_data, asm_data

    # Data used in industrial_indicators[Manufacturing]
    def call_activity_data(self):
        """
        Call BEA API for gross ouput and value add by 3-digit NAICS.
        """
        va_quant_index, go_quant_index = BEA_api(years=list(range(1949, 2018))).chain_qty_indexes()
        return va_quant_index, go_quant_index

    @staticmethod
    def import_mecs_electricity(asm): 
        """
        ### NOT SURE IF ASM or MECS ELECTRICITY DATA ARE USED ###
        Imports MECS data on electricityuse by 3-digit NAICS code.
        In the future,these values will need to be manually downloaded from
        Table 3.2 and added to csv.
        """
        # import a CSV file of historical MECS electricity use from Table 3.2
        
        # elechap3b, all historical
        mecs_elec = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/elechap3b.csv').set_index('NAICS')

        # Ind_hap3_122219.xlsx[ASM_Annual_Elec_1970on]
        link_ratio = asm[[1987]].divide(mecs_elec[[1987]])
        print('link ratio:\n', link_ratio)
        nea_based_data_linked = mecs_elec.multiply(link_ratio, axis=1)

        electricity_consumption = pd.concat([nea_based_data_linked, asm], axis=1)
        print('electricity_consumption pre transpose:\n', electricity_consumption)
        electricity_consumption = electricity_consumption.transpose()
        print('electricity_consumption:\n', electricity_consumption)
        electricity_consumption.index = electricity_consumption.index.astype('int64')
        return electricity_consumption
    
    @staticmethod
    def mecs_annual_fuel(mecs_fuel, electricity_data):  # Ind_hap3_122219.xlsx[MECS_Fuel]
        """TODO: Do NAICS codes 324, 325 have different data?

        Between-MECS-year interpolations are made in MECS_Annual_Fuel1
        and MECS_Annual_Fuel2 tabs in Ind_hap3 spreadsheet.
        Interpolations are also based on estimates developed in
        ASMdata_010220.xlsx[3DNAICS], which ultimately tie back to MECS fuel data
        from Table 4.2 and Table 3.2

        Args:
            mecs_fuel (dataframe): from Ind_hap3/MECS_Fuel
            electricity_data (dataframe): from Ind_hap3/ASM_Annual_Elec_1970on
        """        
        trend_1994 = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/ASM2_Elec.csv').set_index('NAICS')
        mecs_years = mecs_fuel.columns.tolist()
        interpolation_ratios = [] # 1 for mecs years, ratios for other years

        mecs_annual_fuel = [] # perform non-standard interpolation on mecs_fuel
        return mecs_annual_fuel

    @staticmethod
    def calc_quantity_shares(mecs42_df): # From ASMdata_010220.xlsx[Quantity_shares_revised]
        """
        For a given MECS year, take 3-digit NAICS by fuel (TBtu),
        calculate sum, then calcuate quantity shares
        """

        # MECS_data[Quantity Shares_1998 forward] : 
        cols = ['Residual', 'Distillate', 'Nat. Gas', 'LPG', 'Coal', 'Coke', 'Other']
        mecs42_df = mecs42_df.set_index(['Year', 'NAICS'])

        quantity = mecs42_df[cols]

        quantity_shares = quantity.divide(mecs42_df['  Calc. Total'], axis="index")

        return quantity_shares

    @staticmethod
    def quantity_shares_1985_1998():
        """
        Take the sum of the product of quantity shares and the interpolated
        prices (calcualted using asm_price_fit.py), by 3-digit NAICS for a
        given MECS year.
        """
        composite_price = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/historical_composite_prices.csv').rename(columns={'Composite Price ': 'Composite Price'})

        return composite_price

    def expenditure_ratios_revised(self, asm_data):
        mecs = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/mecs_calc_purchased_fuels_historical.csv') # E from MECS_prices_122419.xlsx[MECS_data]/AN and NAICS3D/J (also called EXPFUEL)
        print('mecs:\n', mecs)
        print('mecs:\n', mecs.columns)

        mecs['Year'] = mecs['Year'].astype(int)

        dataset = mecs.merge(asm_data, how='outer', on=['Year', 'NAICS']).set_index('Year')        
        dataset.index = dataset.index.astype(int)
        dataset['NAICS'] = dataset['NAICS'].astype(int)
        print('dataset:\n', dataset)
        print('dataset:\n', dataset.columns)

        dataset['mecs_asm_ratio'] = dataset['Calc. Cost of Fuels'].divide(dataset['EXPFUEL'], axis='index').multiply(1000) # G

        dataset_ = self.interpolate_mecs(dataset, col_name='mecs_asm_ratio').rename(columns={'mecs_asm_ratio': 'mecs_asm_ratio_interp'})
        dataset =  dataset_.merge(dataset, how='outer', on=['Year', 'NAICS']).set_index('Year')       

        dataset['mecs_based_expenditure'] = dataset['Calc. Cost of Fuels'].multiply(1000) # I depends on MECS year/not
        dataset['fill_values'] = dataset['EXPFUEL'].multiply(dataset['mecs_asm_ratio_interp'], axis='index')
        dataset['mecs_based_expenditure'] = dataset['mecs_based_expenditure'].fillna(dataset['fill_values'])

        mecs_based_expenditure = dataset.reset_index()[['Year', 'NAICS', 'mecs_based_expenditure']]
        return mecs_based_expenditure

    def expend_ratios_revised_85_97(self):

        mecs_based_expenditure_hist = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/Expend_ratios_revised_1985-97.csv')  
        return mecs_based_expenditure_hist

    @staticmethod
    def aggregate_naics(df, values):
        
        df = df.pivot(index='NAICS', columns='Year', values=values)
        print('df:\n', df)
        df.index = df.index.astype(int)

        df.loc['311+312', :] = df.loc[[311, 312], :].sum(axis=0)
        df.loc['313+314', :] = df.loc[[313, 314], :].sum(axis=0)
        df.loc['315+316', :] = df.loc[[315, 316], :].sum(axis=0)
        df = df.drop(index=[313, 314, 315, 316, 311, 312]).reset_index()
        print('df:\n', df)

        df = pd.melt(df, id_vars='NAICS', value_name=values, var_name='Year')
        print('df:\n', df)
        return df

    def quantity_shares_1998_forward(self):
        mecs_years_prices_and_interpolations = self.manufacturing_prices()
        print('mecs_years_prices_and_interpolations:\n', mecs_years_prices_and_interpolations)

        mecs42_df = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/mecs_table42.csv')
        mecs42_df['Year'] = mecs42_df['Year'].astype(int)
        mecs42_df['NAICS'] = mecs42_df['NAICS'].astype(int)

        # mecs42_df = Mfg_prices().
        mecs_data_qty_shares = self.calc_quantity_shares(mecs42_df)
        print('mecs_data_qty_shares:\n', mecs_data_qty_shares)
        mecs_cols = mecs_data_qty_shares.columns
        mecs_data_qty_shares = mecs_data_qty_shares.reset_index()
        fuel_quanity_shares = []
        # interpolate mecs_data_qty_shares data (has 3 dimensions: fuel type, year, naics)
        print('dataset:\n', mecs_data_qty_shares)
        print('dataset:\n', mecs_data_qty_shares.columns)

        for fuel_type in mecs_cols:
            fuel_df = mecs_data_qty_shares[['Year', 'NAICS', fuel_type]]
            fuel_df = self.interpolate_mecs(fuel_df, fuel_type, reindex=mecs_years_prices_and_interpolations['Year'].unique())
            fuel_quanity_shares.append(fuel_df)

        fuel_quanity_shares = reduce(lambda df1,df2: df1.merge(df2, how='outer', 
                                     on=['Year', 'NAICS']), fuel_quanity_shares)
        print('fuel_quanity_shares:\n', fuel_quanity_shares)
        fuel_quanity_shares = fuel_quanity_shares.set_index('Year')
        print('fuel_quanity_shares:\n', fuel_quanity_shares)

        mecs_years_prices_and_interpolations = mecs_years_prices_and_interpolations.set_index('Year')

        # composite_price = mecs_years_prices_and_interpolations.multiply(fuel_quanity_shares, axis='index').sum(axis=1)
        composite_price = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/current_composite_price.csv')
        composite_price['NAICS'] = composite_price['NAICS'].astype(int)
        composite_price['Year'] = composite_price['Year'].astype(int)

        return composite_price

    @staticmethod
    def mecs_data_sic():
        mecs_data_sic = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/MECS_data_SIC.csv') # from [MECS_prices_101116b.xlsx]MECS_data_SIC BA
        mecs_data_sic = mecs_data_sic[mecs_data_sic['Year'].notnull()]
        mecs_data_sic['Year'] = mecs_data_sic['Year'].astype(int)
        mecs_data_sic['NAICS'] = mecs_data_sic['NAICS'].astype(int)

        return mecs_data_sic

    @staticmethod
    def interpolate_mecs(mecs_data, col_name, reindex=None):
        if not 'Year' in mecs_data.columns:
            mecs_data = mecs_data.reset_index()
        mecs_data = mecs_data.pivot(index='Year', columns='NAICS', values=col_name)
        if reindex is not None:
            mecs_data = mecs_data.reindex(reindex)
        for c in mecs_data.columns:
            mecs_data = standard_interpolation(mecs_data, name_to_interp=c, axis=1) # from mixed sources
        
        mecs_data = pd.melt(mecs_data.reset_index(), id_vars='Year', var_name='NAICS', value_name=col_name)
        return mecs_data

    def pre_1998_quantities(self):

        dollar_per_mmbtu = self.quantity_shares_1985_1998() # from quantity_shares_revised CW --> '[MECS_prices_122419.xlsx]Quantity Shares_1985-1998'!
        dollar_per_mmbtu['Year'] = dollar_per_mmbtu['Year'].astype(int)

        mecs_based_expenditure = self.expend_ratios_revised_85_97() # from Expend_ratios_revised_1985-97 and Expend_ratios_revised

        mecs_data_sic = self.mecs_data_sic()

        mecs_data_sic = mecs_data_sic[mecs_data_sic['Variable'] == 'Scale Factor'].drop('Variable', axis=1)
        mecs_data_sic = mecs_data_sic.rename(columns={'Value': 'ratio_fuel_to_offsite'})

        dataset = dollar_per_mmbtu.merge(mecs_based_expenditure, how='outer', on=['NAICS', 'Year'])
        
        dataset = dataset.merge(mecs_data_sic, how='outer', on=['NAICS', 'Year'])
        dataset = self.interpolate_mecs(dataset, col_name='ratio_fuel_to_offsite')

        dataset = dataset[dataset['Year'] <= 1997]
        return dataset

    def quantities_1998_forward(self, NAICS3D):
        quantity_shares_1998_forward = self.quantity_shares_1998_forward() # MECSPrices122419[Quantity shares 1998 forward]
        asm_data = NAICS3D.reset_index()[['Year', 'NAICS', 'EXPFUEL']]

        mecs_based_expenditure = self.expenditure_ratios_revised(asm_data)
        mecs_based_expenditure['Year'] = mecs_based_expenditure['Year'].astype(int)
        dataset = mecs_based_expenditure.merge(quantity_shares_1998_forward, how='outer', on=['NAICS', 'Year'])

        dataset = dataset[dataset['Year'] >= 1998]
        return dataset

    def final_quantities_asm_85(self):
        """
        Between-MECS-year interpolations are made in MECS_Annual_Fuel1
        and MECS_Annual_Fuel2 tabs in Ind_hap3 spreadsheet.
        Interpolations are also based on estimates developed in
        ASMdata_010220.xlsx[3DNAICS], which ultimately tie back to MECS fuel data
        from Table 4.2 and Table 3.2
        """
        NAICS3D = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/3DNAICS.csv').set_index('Year')
        
        pre_1998_quantities = self.pre_1998_quantities().rename(columns={'Calc. Cost of Fuels': 'Composite Price'})
        print('pre_1998_quantities:\n', pre_1998_quantities)
        quantities_1998_forward = self.quantities_1998_forward(NAICS3D)

        quantities = pd.concat([pre_1998_quantities, quantities_1998_forward], axis=0)

        quantities['jan_2020_estimate'] = quantities['mecs_based_expenditure'].divide(quantities['Composite Price'], axis='index').multiply(0.001)

        quantities['final_quantities_asm_85'] = quantities['jan_2020_estimate'].multiply(quantities['ratio_fuel_to_offsite'], axis='index') # ASMdata_010330.xlsx , Final_quant_elec_w_ASM_87'

        final_quantities_asm_85 = quantities[['NAICS', 'Year', 'final_quantities_asm_85']].pivot(index='NAICS', columns='Year', values='final_quantities_asm_85')
        final_quantities_asm_85_agg = self.aggregate_naics(final_quantities_asm_85, values='final_quantities_asm_85')
        # final_quantities_asm_85 = pd.read_csv('./EnergyIntensityIndicators/Indutry/Data/final_quantities_asm_85.csv').set_index('NAICS')
        return final_quantities_asm_85_agg
    
    def get_manufacturing_fuels(self, electricity_data):
        # Ind_hap3_122219.xlsx[ASM_Annual_Fuel3_1970on]
        print('electricity_data:\n', electricity_data)

        fuels_nea = self.import_mecs_fuel() # fallhap3
        print('fuels_nea:\n', fuels_nea)

        mecs_fuel = self.get_historical_mecs()
        print('mecs_fuel:\n', mecs_fuel)
        exit()
        # electricity_data = self.aggregate_naics(electricity_data, values=)
        # fuels_nea = self.aggregate_naics(fuels_nea, values=)
        
        mecs_interpolated_data = self.mecs_annual_fuel(mecs_fuel, electricity_data)

        link_ratio = mecs_interpolated_data[[1985]].divide(fuels_nea[1985].values)
        nea_adjusted = fuels_nea.multiply(link_ratio)
        fuels_consumption = pd.concat([nea_adjusted, mecs_interpolated_data], axis=1)
        fuels_consumption = fuels_consumption.transpose()
        print('fuels_consumption:\n', fuels_consumption)

        fuels_consumption.index = fuels_consumption.index.astype(int)
        return fuels_consumption

    def manufacturing_energy(self):
        """Collect electricity and fuels for the manufacturing sector
        """        
        asm = self.final_quantities_asm_85()
        electricity_consumption = self.import_mecs_electricity(asm)
        
        fuels_consumption = self.get_manufacturing_fuels(asm)
      
        return electricity_consumption, fuels_consumption # Transfered to industrial_indicators_060220.xlsx[Manufacturing_Energy_Data]

    def manufacturing(self):
        """Main datasource is the Manufacturing Energy Consumption Survey (MECS), conducted by the EIA since 1985 (supplemented for non-MECS years by
        estimates derived from the Annual Survey of Manufactures (ASM) and the Economic Census (EC) conducted every five years)
        https://www.eia.gov/consumption/manufacturing/data/2014/
        https://www.eia.gov/consumption/manufacturing/data/2014/#r4
        """
        va_quant_index, go_quant_index = self.call_activity_data()
        electricity_consumption, fuels_consumption = self.manufacturing_energy()

        data_dict = {'energy': {'elec': electricity_consumption, 
                                'fuels': fuels_consumption}, 
                     'activity': {'gross_output': go_quant_index,
                                  'value_added': va_quant_index}}
        return data_dict

if __name__ == '__main__':
    data = Manufacturing().manufacturing()
    print(data)
    pass