import pandas as pd
from datetime import datetime
import os

from EnergyIntensityIndicators.pull_bea_api import BEA_api
from EnergyIntensityIndicators.get_census_data import (Econ_census, Asm)
from EnergyIntensityIndicators.Industry.asm_price_fit import Mfg_prices

from EnergyIntensityIndicators.utilites.standard_interpolation import standard_interpolation

class Manufacturing:
    """Class to collect and process manufacturing data for the industrial sector
    """
    def __init__(self):
        self.end_year = datetime.now().year

    # ASMdata_date.xlsx

    # ind_hap3_date.xlsx

    # 2014_MECS = 'https://www.eia.gov/consumption/manufacturing/data/2014/'  # Table 4.2


    # Table 3.1 and 3.2 (MECS total fuel consumption)  Table 3.1 shows energy
    # consumption by fuel in physical units, including the total across all fuels expressed in trillion Btu and
    # electricity in kWh. From Table 3.1, total fuel consumption in Btu can be calculated as difference between
    # total energy and electricity consumption after conversion to Btu. Table 3.2 only differs from Table 3.1 by
    # showing all fuel types in Btu.



    # MER_Table24_Industrial_Energy_Consumption = [0]

    # For 2014, the values for total energy consumption and electricity consumption, both defined in terms of
    # trillion Btu, from Table 3.2 are transferred to spreadsheet ind_hap3. Worksheet MECS_Fuel in this
    # spreadsheet has been used to collect the fuel consumption estimates for all the MECS dating back to the
    # first MECS in 1985. The 2014 data are located in the cell range F218:F238.
    # The first six NAICS sectors are aggregated into three sectors (311-312, 313-314, and 315-316) as a part
    # of the set of manufacturing indicators. The energy consumption data under this revised sectoring
    # classification are shown in the columns to the right, columns O and P.



    # Energy prices
    # MECS_Table72 = [0]

    @staticmethod
    def mecs_fuel():
        # historical_mecs_31_32 = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/historical_mecs_31_32.csv')
        mecs_fuel = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/MECS_Fuel_Historical.csv').set_index('NAICS')
        return mecs_fuel

    def get_historical_mecs(self):
        
        """Read in historical MECS csv, format (as in e.g. Coal (MECS) Prices)
        """
        pass

    def manufacturing_prices(self):
        """Call ASM API method from Asm class in get_census_data.py
        Specify three-digit NAICS Codes
        """
        fuel_types = ['Gas', 'Coal', 'Distillate', 'Residual', 'LPG', 'Coke', 'Other']
        naics = [311, 312, 313, 314, 315, 316, 321, 322, 323, 324, 
                 325, 326, 327, 331, 332, 333, 334, 335, 336, 337, 339]

        asm_price_data = []
        for f in fuel_types: 
            predicted_fuel_price = Mfg_prices().calc_calibrated_predicted_price(latest_year=self.end_year, fuel_type=f, naics=naics)
            asm_price_data.append(predicted_fuel_price)

        asm_price_data = pd.concat(asm_price_data)
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

    # @staticmethod
    # def interpolate_mecs(mecs_fuel, ASMdata_010220_xlsx_data):
    #     """
    #     Between-MECS-year interpolations are made in MECS_Annual_Fuel1
    #     and MECS_Annual_Fuel2 tabs in Ind_hap3 spreadsheet.
    #     Interpolations are also based on estimates developed in
    #     ASMdata_010220.xlsx[3DNAICS], which ultimately tie back to MECS fuel data
    #     from Table 4.2 and Table 3.2
    #     """
    #     # standard_interpolation(dataframe=, name_to_interp= , axis=)
    #     quantity_shares_revised =  # Quantity_shares_revised!CW14

    #     # in ASMdata_010220.xlsx[Final_quantities_w_ASM_85] 
    #     mecs_data_sic = self.mecs_data_sic() # 

    #     ratio_fuel_offsite_pre98 = standard_interpolation(dataframe=mecs_data_sic, name_to_interp= , axis=)

    #     data_98 = []
    #     mecs_tables_31_32 = []
    #     mecs_table42 = []

    #     ratio_fuel_offsite = []
 
    #     interpolated_mecs =

    #     return interpolated_mecs

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
        link_ratio = asm[['1987']].divide(mecs_elec[['1987']])
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


        # for index, y_ in enumerate(increment_years):
        #     if index > 0:
        #         year_before = increment_years[index - 1]
        #         num_years = y_ - year_before
        #         resid_year_before = dataframe.xs(year_before)[name_to_interp]
        #         resid_y_ = dataframe.xs(y_)[name_to_interp]
        #         increment = 1 / num_years
        #         for delta in range(num_years):
        #             value = resid_year_before * (1 - increment * delta) + \
        #                 resid_y_ * (increment * delta)
        #             year = year_before + delta
        #             dataframe.loc[year, name_to_interp] = value



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
        quantity_shares = mecs42_df[cols].divide(mecs42_df['  Calc. Total'])

        return quantity_shares

    @staticmethod
    def quantity_shares_1985_1998():
        """
        Take the sum of the product of quantity shares and the interpolated
        prices (calcualted using asm_price_fit.py), by 3-digit NAICS for a
        given MECS year.
        """

        # shares_total_fuel_use_2013 = self.calc_quantity_shares() # from Ind_ _Prices / Predicted__ Prices
        # mecs_sic = self.mecs_data_sic()
        # mecs_sic['Other'] = 0
        # mecs_sic['Total'] = mecs_sic.sum(axis=1)
        # mecs_years_prices_and_interpolations = standard_interpolation(mecs_sic) #, name_to_interp=, axis=)

        # composite_price = shares_total_fuel_use_2013.multiply(mecs_years_prices_and_interpolations.values).sum(axis=1)
        print('os.getcwd()', os.getcwd())
        composite_price = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/historical_composite_prices.csv')
        print('composite_price:\n', composite_price)
        print('composite_price cols:\n', composite_price.columns)

        composite_price = composite_price.pivot(index='NAICS', columns='Year', values='Composite Price ')
        print('composite_price:\n', composite_price)
        composite_price.index = composite_price.index.astype(int)

        # composite_price = composite_price.loc[321:339, :] 

        composite_price.loc['311+312', :] = composite_price.loc[[311, 312], :].sum(axis=0)
        composite_price.loc['313+314', :] = composite_price.loc[[313, 314], :].sum(axis=0)
        composite_price.loc['315+316', :] = composite_price.loc[[315, 316], :].sum(axis=0)
        composite_price = composite_price.drop(index=[313, 314, 315, 316, 311, 312]).reset_index()
        print('composite_price:\n', composite_price)

        composite_price = pd.melt(composite_price, id_vars='NAICS', value_name='Composite Price', var_name='Year')
        print('composite_price:\n', composite_price)

        return composite_price

    @staticmethod
    def expenditure_ratios_revised(NAICS3D):
        mecs = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/mecs_calc_purchased_fuels_historical.csv') # E from MECS_prices_122419.xlsx[MECS_data]/AN and NAICS3D/AW (also called EXPFUEL)
        # \\mecs is mixed with NAICS
        NAICS3D = NAICS3D.loc[1988:, ['NAICS', 'EXPFUEL']].reset_index()

        print('NAICS3D:\n', NAICS3D)

        NAICS3D = NAICS3D.pivot(index='NAICS', columns='Year', values='EXPFUEL')
        print('NAICS3D:\n', NAICS3D)

        NAICS3D.loc['311+312', :] = NAICS3D.loc[[311, 312], :].sum(axis=0)
        NAICS3D.loc['313+314', :] = NAICS3D.loc[[313, 314], :].sum(axis=0)
        NAICS3D.loc['315+316', :] = NAICS3D.loc[[315, 316], :].sum(axis=0)
        NAICS3D = NAICS3D.drop(index=[313, 314, 315, 316, 311, 312]).reset_index()
        NAICS3D = pd.melt(NAICS3D, id_vars='NAICS', var_name='Year', value_name='EXPFUEL')
        NAICS3D['Year'] = NAICS3D['Year'].astype(int)

        mecs = mecs.pivot(index='NAICS', columns='Year', values='Calc. Cost of Fuels')
        mecs.loc['311+312', :] = mecs.loc[[311, 312], :].sum(axis=0)
        mecs.loc['313+314', :] = mecs.loc[[313, 314], :].sum(axis=0)
        mecs.loc['315+316', :] = mecs.loc[[315, 316], :].sum(axis=0)
        mecs = mecs.drop(index=[313, 314, 315, 316, 311, 312]).reset_index()
        mecs = pd.melt(mecs, id_vars='NAICS', var_name='Year', value_name='Calc. Cost of Fuels')
        mecs['Year'] = mecs['Year'].astype(int)

        print('NAICS3D:\n', NAICS3D)
        print('NAICS3D.index:\n', NAICS3D.dtypes)

        print('mecs_data:\n', mecs)
        print('mecs_data index:\n', mecs.dtypes)

        dataset = mecs.merge(NAICS3D, how='outer', on=['Year', 'NAICS']).set_index('Year')        
        print('dataset after merge index:', dataset.index)
        print('dataset:\n', dataset)
        dataset['mecs_asm_ratio'] = dataset['Calc. Cost of Fuels'].divide(dataset['EXPFUEL'].values).multiply(1000) # G
        dataset = dataset.reset_index()
        print('dataset:\n', dataset)
        print('dataset years:', dataset['Year'].unique())

        dataset_ = dataset.pivot(index='Year', columns='NAICS', values='mecs_asm_ratio')
        print('dataset:\n', dataset_)

        for c in dataset_.columns:
            dataset_ = standard_interpolation(dataset_, name_to_interp=c, axis=1) # H
        
        dataset_ = pd.melt(dataset_.reset_index(), id_vars='Year', var_name='NAICS', value_name='mecs_asm_ratio_interp')

        print('dataset after interp:\n', dataset_)
        print('dataset years:', dataset['Year'].unique())

        dataset = dataset.merge(dataset_, how='outer', on=['Year', 'NAICS'])
        print('dataset after interp:\n', dataset)
        print('dataset years:', dataset['Year'].unique())
        dataset['mecs_based_expenditure'] = dataset['Calc. Cost of Fuels'].multiply(1000) # I depends on MECS year/not
        dataset['fill_values'] = dataset['EXPFUEL'].multiply(dataset['mecs_asm_ratio_interp'].values)
        dataset['mecs_based_expenditure'] = dataset['mecs_based_expenditure'].fillna(dataset['fill_values'])
        print('dataset now:\n', dataset)
        mecs_based_expenditure = dataset[['Year', 'NAICS', 'mecs_based_expenditure']]
        return mecs_based_expenditure

    def expend_ratios_revised_85_97(self, NAICS3D):

        mecs_data_sic = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/MECS_data_SIC.csv').drop('Variable', axis=1) # from [MECS_prices_101116b.xlsx]MECS_data_SIC BA
        mecs_data_sic = mecs_data_sic[mecs_data_sic['Year'].notnull()]
        mecs_data_sic['Year'] = mecs_data_sic['Year'].astype(int)
        mecs_data_sic['NAICS'] = mecs_data_sic['NAICS'].astype(int)

        mecs_data_sic = mecs_data_sic.pivot(index='Year', columns='NAICS', values='Value')

        print('mecs_data_sic:\n', mecs_data_sic)
        mecs_data_sic = mecs_data_sic.reset_index()
        mecs_data_sic = pd.melt(mecs_data_sic, id_vars='Year', value_name='mecs_data_sic') # , value_vars='value', var_name='Year')
        mecs_based_expenditure_hist = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/Expend_ratios_revised_1985-97.csv')  
        return mecs_based_expenditure_hist, mecs_data_sic

        # print('mecs_data_sic:\n', mecs_data_sic)
        # # \\ mecs data mixed with NAICS too 
        # asm = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/ASM_Fuel_Cost_1985-88.csv') # [Ind_hap3_101316.xlsx]ASM_Fuel_Cost_1985-88
        # asm = asm.rename(columns={'Unnamed: 0': 'NAICS'})
        # print('asm:\n', asm)

        # # asm = asm.transpose()
        # asm = pd.melt(asm, id_vars='NAICS', var_name='Year')
        # print('asm:\n', asm)
        # asm.index = asm.index.astype(int)
        # print('asm:\n', asm)
        # print('asm index:\n', asm.index)
        # print('NAICS3D:\n', NAICS3D)
        # NAICS3D = NAICS3D.loc[1988:, ['NAICS', 'EXPFUEL']].reset_index()
        # NAICS3D = NAICS3D.pivot(index='NAICS', columns='Year', values='EXPFUEL')

        # print('NAICS3D:\n', NAICS3D)
        # print('NAICS3D index:\n', NAICS3D.index)
        # NAICS3D.loc['311+312', :] = NAICS3D.loc[[311, 312], :].sum(axis=0)
        # NAICS3D.loc['313+314', :] = NAICS3D.loc[[313, 314], :].sum(axis=0)
        # NAICS3D.loc['315+316', :] = NAICS3D.loc[[315, 316], :].sum(axis=0)
        # NAICS3D = NAICS3D.drop(index=[313, 314, 315, 316, 311, 312])
        # print('NAICS3D:\n', NAICS3D)
        # NAICS3D = NAICS3D.reset_index()
        # print('NAICS3D:\n', NAICS3D)

        # NAICS3D = pd.melt(NAICS3D, id_vars='NAICS', var_name='Year', value_name='EXPFUEL').set_index('Year')
        # NAICS3D.index = NAICS3D.index.astype(int)
        # NAICS3D = NAICS3D.reset_index()
        # print('NAICS3D:\n', NAICS3D)

        # # e_c_data, asm_data = self.call_census_data()
        # # print('asm_data:\n', asm_data)
        # # mecs =[] # from MECS_prices_101116b.xlsx[MECS_data_SIC]/BL
        # asm = asm.set_index('Year')
        # asm.index = asm.index.astype(int)
        # print('asm:\n', asm)
        # asm_data = asm.loc[[1985, 1986, 1987], :].reset_index('Year')
        # asm_data = asm_data.rename(columns={'value': 'EXPFUEL'})
        # print('asm_data.dtypes:\n', asm_data.dtypes)
        # print('NAICS3D.dtypes:\n', NAICS3D.dtypes)

        # asm_ = pd.concat([asm_data, NAICS3D], axis=0)

        # print('asm_:\n', asm_)


        # # ratio = mecs.divide(asm)

        # mecs_asm = asm_.merge(mecs_data_sic, on=['Year', 'NAICS'], how='outer')
        # print('mecs_asm:\n', mecs_asm)
        # # \\ asm filled with naics starting 1988
        # mecs_asm['mecs_asm_ratio'] = mecs_asm['mecs_data_sic'].divide(mecs_asm['EXPFUEL'].values)

        # print('mecs_asm:\n', mecs_asm)

        # mecs_asm['interpolated_ratio'] = standard_interpolation(mecs_asm, name_to_interp='mecs_asm_ratio', axis=1)

        # mecs_asm['expenditure'] = mecs_asm['EXPFUEL'].multiply(mecs_asm['interpolated_ratio']).multiply(1000)
        # mecs_based_expenditure = mecs_asm[['expenditure']]
        # # mecs_based_expenditure = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/mecs_based_expenditure.csv').set_index('Year')



    
    def final_quantities_asm_85(self):
        NAICS3D = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/3DNAICS.csv').set_index('Year')
        
        mecs_based_expenditure_historical, mecs_data_sic = self.expend_ratios_revised_85_97(NAICS3D) # from Expend_ratios_revised_1985-97 and Expend_ratios_revised
        mecs_based_expenditure = self.expenditure_ratios_revised(NAICS3D)
        mecs_based_expenditure['Year'] = mecs_based_expenditure['Year'].astype(int)

        """DONE"""
        dollar_per_mmbtu = self.quantity_shares_1985_1998() # from quantity_shares_revised CW --> '[MECS_prices_122419.xlsx]Quantity Shares_1985-1998'!
        dollar_per_mmbtu['Year'] = dollar_per_mmbtu['Year'].astype(int)
        print('dollar_per_mmbtu:\n', dollar_per_mmbtu)
        jan_2020_estimate = mecs_based_expenditure.merge(dollar_per_mmbtu, on=['Year', 'NAICS'], how='outer')
        jan_2020_estimate['jan_2020_estimate'] = jan_2020_estimate['mecs_based_expenditure'].divide(jan_2020_estimate['Composite Price'].values).multiply(0.001)
        print('jan_2020_estimate:\n', jan_2020_estimate)
        print('mecs_data_sic:\n', mecs_data_sic)
        mecs_data_sic['mecs_data_sic'] = mecs_data_sic['mecs_data_sic'].astype(float)
        mecs_data_sic = mecs_data_sic.pivot(index='Year', columns='NAICS', values='mecs_data_sic')
        mecs_data_sic = mecs_data_sic.reindex(jan_2020_estimate['Year'].unique().tolist())

        print('mecs_data_sic:\n', mecs_data_sic)
        for c in mecs_data_sic.columns:
            mecs_data_sic = standard_interpolation(mecs_data_sic, name_to_interp=c, axis=1) # from mixed sources
        
        mecs_data_sic = pd.melt(mecs_data_sic.reset_index(), id_vars='Year', var_name='NAICS', value_name='mecs_data_sic')
        print('mecs_data_sic:\n', mecs_data_sic)
        ratio_fuel_to_offsite = pd.concat([mecs_data_sic, NAICS3D.reset_index()], axis=0)
        print('NAICS3D:\n', NAICS3D)
        print('ratio_fuel_to_offsite:\n', ratio_fuel_to_offsite)

        jan_2020_estimate = jan_2020_estimate.merge(ratio_fuel_to_offsite, on=['Year', 'NAICS'], how='outer')
        print('jan_2020_estimate:\n', jan_2020_estimate)
        final_quantities_asm_85 = jan_2020_estimate['jan_2020_estimate'].multiply(jan_2020_estimate['ratio_fuel_to_offsite'].values) # ASMdata_010330.xlsx , Final_quant_elec_w_ASM_87'

        
        # final_quantities_asm_85 = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/final_quantities_asm_85.csv').set_index('NAICS')
        # asm_data = final_quantities_asm_85
        return final_quantities_asm_85
    
    def get_manufacturing_fuels(self, electricity_data):
        # Ind_hap3_122219.xlsx[ASM_Annual_Fuel3_1970on]
        fuels_nea = self.import_mecs_fuel() # fallhap3
        mecs_fuel = self.get_historical_mecs()
        mecs_annual_fuel = self.mecs_annual_fuel(mecs_fuel, electricity_data)

    
        historical_ASMdata_010220_xlsx_data = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/historical_asm_data.csv')
        

        mecs_interpolated_data = self.interpolate_mecs(mecs_fuel, historical_ASMdata_010220_xlsx_data) # MECS_Annual_Fuel2!

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