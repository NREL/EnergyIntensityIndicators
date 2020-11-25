import pandas as pd

from EnergyIntensityIndicators.pull_bea_api import BEA_api
from EnergyIntensityIndicators.get_census_data import Econ_census
from EnergyIntensityIndicators.get_census_data import Asm
from EnergyIntensityIndicators.utilites.standard_interpolation import standard_interpolation

class Manufacturing:
    """Class to collect and process manufacturing data for the industrial sector
    """
    def __init__(self):
        pass
    # ASMdata_date.xlsx

    # ind_hap3_date.xlsx

    #2014_MECS = 'https://www.eia.gov/consumption/manufacturing/data/2014/'  # Table 4.2


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


    # def get_historical_mecs(self):
        
    #     """Read in historical MECS csv, format (as in e.g. Coal (MECS) Prices)
    #     """
    #     historical_mecs = pd.read_csv('./')
    #     historical_mecs_31_32 = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/historical_mecs_31_32.csv')
    #     return historical_mecs

    # def manufacturing_prices(self):
    #     """Call ASM API method from Asm class in get_census_data.py
    #     Specify three-digit NAICS Codes
    #     """
    #     fuel_types = ['Gas', 'Coal', 'Distillate', 'Residual', 'LPG', 'Coke', 'Other']
    #     naics = []

    #     asm_price_data = []
    #     for f in fuel_types: 
    #         predicted_fuel_price = Mfg_prices().calc_calibrated_predicted_price(latest_year=self.end_year, f, naics)
    #         asm_price_data.append(predicted_fuel_price)

    #     asm_price_data = pd.concat(asm_price_data)
    #     return asm_price_data

    # @staticmethod
    # def calc_quantity_shares(mecs42_df): # From ASMdata_010220.xlsx[Quantity_shares_revised]
    #     """
    #     For a given MECS year, take 3-digit NAICS by fuel (TBtu),
    #     calculate sum, then calcuate quantity shares
    #     """
    #     # MECS_data[Quantity Shares_1998 forward] : 
    #     cols = ['Residual', 'Distillate', 'Nat. Gas', 'LPG', 'Coal', 'Coke', 'Other']
    #     quantity_shares = mecs42_df[cols].divide(mecs42_df['  Calc. Total'])

    #     return quantity_shares

    # @staticmethod
    # def calc_composite_price(quantity_shares, interp_prices):  # Where is this used???
    #     """
    #     Take the sum of the product of quantity shares and the interpolated
    #     prices (calcualted using asm_price_fit.py), by 3-digit NAICS for a
    #     given MECS year.


    #     """
    #     composite_prices = quantity_shares.multiply(interp_prices, axis=1).sum(axis=1)

    #     return composite_prices

    # # Corresponds to data in MECS_Fuel tab in Indhap3 spreadsheet, which
    # # is connected to the MECS_Annual_Fuel1 and MECS_Annual_Fuel2
    # # tabs in the same spreadsheet.
    # def import_mecs_fuel(self):
    #     """
    #     Imports MECS data on fuel use by 3-digit NAICS code. In the future,
    #     these values will need to be manually downloaded from Table 3.2
    #     and added to csv.
    #     """

    #     fallhap3b = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/fallhap3b.csv')
    #     mecs_fuel = []

    #     return fallhap3b, mecs_fuel
    
    # def mecs_annual_fuel(mecs_fuel):
    #     """TODO: Do NAICS codes 324, 325 have different data?

    #     Args:
    #         mecs_fuel ([type]): [description]
    #     """        
    #     mecs_annual_fuel = standard_interpolation(mecs_fuel, axis=1)


    # def import_mecs_electricity(self):
    #     """
    #     ### NOT SURE IF ASM or MECS ELECTRICITY DATA ARE USED ###
    #     Imports MECS data on electricityuse by 3-digit NAICS code.
    #     In the future,these values will need to be manually downloaded from
    #     Table 3.2 and added to csv.
    #     """
        
    #     # ELEC
    #     elechap3b = pd.read_csv('./EnergyIntensityIndicators/Industry/Data/elechap3b.csv')

    #     # import a CSV file of historical MECS electricity use from Table 3.2
    #     # Will need to aggregate NAICS 311+312, 313+314, and 315+316
    #     mecs_published = []
    #     table31_net_elec = [] # different in years < 2010 and after
    #     mecs_fuel = mecs_published.subtract(table31_net_elec)
    #     mecs_fuel_to_transfer = mecs_fuel.loc[321:339, ['Net Elec', 'Total Fuel']]
    #     mecs_fuel_to_transfer.loc['311+312', :] = mecs_fuel.loc[311:312, :].sum(axis=0)
    #     mecs_fuel_to_transfer.loc['313+314', :] = mecs_fuel.loc[313:314, :].sum(axis=0)
    #     mecs_fuel_to_transfer.loc['315+316', :] = mecs_fuel.loc[315:316, :].sum(axis=0)
    #     mecs_interpolated_data = [] # USE STANDARD INTERPOLATION METHOD # from mecs_annual_fuel2
    #     mecs_interpolated_data.loc[324:325, :] = elec.loc[324:325, :] 
    #     return mecs_elect
    
    # # Data used in ASMdata_010220.xlsx[3DNAICS]
    # def call_census_data(self):
    #     """
    #     Use Census_api class to call fuel and electricity expenditures from
    #     Annual Survey of Manufacturers and Economic Census (for years ending
    #     with 2 and 7).
    #     """
    #     paa

    # # Data used in industrial_indicators[Manufacturing]
    # def call_activity_data(self):
    #     """
    #     Call BEA API for gross ouput and value add by 3-digit NAICS.
    #     """
    #     va_quant_index, go_quant_index = BEA_api(years=list(range(1949, 2018))).chain_qty_indexes()
    #     # HERE: select columns 
    #     return va_quant_index, go_quant_index

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

    #     # in ASMdata_010220.xlsx[Final_quantities_w_ASM_85] 
    #     mecs_data_sic = []
    #     # ratio_fuel_offsite_pre98 = standard_interpolation(dataframe=mecs_data_sic, name_to_interp= , axis=)

    #     data_98 = []
    #     mecs_tables_31_32 = []
    #     mecs_table42 = []

    #     ratio_fuel_offsite = []
    
    # def expenditure_ratios_revised(NAICS3D):
    #     mecs = [] # E from MECS_prices_122419.xlsx[MECS_data]/AN and NAICS3D/AW (also called EXPFUEL)

    #     asm = NAICS3D[['EXPFUEL']] # F
        
    #     mecs_asm_ratio = mecs.divide(asm).multiply(1000) # G

    #     interpolated_ratio = standard_interpolation(mecs_asm_ratio) # H
    #     mecs_based_expenditure = [] # I depends on MECS year/not

        
    #     return mecs_based_expenditure

    # def expend_ratios_revised_85_97(self, NAICS3D):
    #     mecs =[] # from MECS_prices_101116b.xlsx[MECS_data_SIC]/BL
    #     asm_data = [] # from Ind_hap3_102316.xlsx[ASM_Fuel_Cost_1985-88]/ R
    #     asm_data = asm_data.iloc[[1985, 1986, 1987]]
    #     NAICS3D = NAICS3D.loc[1988:, ['EXPFUEL']]
    #     asm = pd.concat([asm_data, NAICS3D])
    #     ratio = mecs.divide(asm)

    #     interpolated_ratio = standard_interpolation(ratio)

    #     mecs_based_expenditure = asm.multiply(interpolated_ratio * 1000)
    #     return mecs_based_expenditure
    
    # def final_quantities_asm_85(self):
    #     mecs = [] # from [MECS_prices_101116b.xlsx]MECS_data_SIC
    #     asm = [] # [Ind_hap3_101316.xlsx]ASM_Fuel_Cost_1985-88
    #     mecs_asm_ratio = mecs.divide(asm)
        
    #     mecs_based_expenditure =  [] # from Expend_ratios_revised_1985-97 and Expend_ratios_revised
    #     dollar_per_mmbtu = [] # from quantity_shares_revised
    #     jan_2020_estimate = mecs_based_expenditure.divide(dollar_per_mmbtu)
    #     ratio_fuel_to_offsite = [] # from mixed sources


    #     final_quantities_asm_85 = jan_2020_estimate.multiply(ratio_fuel_to_offsite)
    #     asm_data = final_quantities_asm_85.loc[321:339, :] # ASMdata_010330.xlsx , Final_quant_elec_w_ASM_87'
    #     asm.loc['311+312', :] = final_quantities_asm_85.loc[311:312, :].sum(axis=0)
    #     asm.loc['313+314', :] = final_quantities_asm_85.loc[313:314, :].sum(axis=0)
    #     asm.loc['315+316', :] = final_quantities_asm_85.loc[315:316, :].sum(axis=0)
        
    #     return final_quanities

    # def manufacturing_energy():
    #     """Collect electricity and fuels for the manufacturing sector
    #     """        
    #     asm = self.final_quantities_asm_85()
    #     nea_based_data = self.import_mecs_electricity()
    #     # Ind_hap3_122219.xlsx[ASM_Annual_Elec_1970on]
    #     link_ratio = asm[[1987]].divide(nea_based_data[1987])

    #     nea_based_data_linked = nea_based_data.multiply(link_ratio, axis=1)

    #     electricity_consumption = pd.concat([nea_based_data_linked, asm_data], axis=1)
    #     electricity_consumption = electricity_consumption.transpose()

        

    #     fuels_nea, mecs_fuel = self.import_mecs_fuel() # fallhap3


    #     # Ind_hap3_122219.xlsx[ASM_Annual_Fuel3_1970on]
    #     link_ratio = mecs_interpolated_data[[1985]].divide(fuels_nea[1985])
    #     nea_adjusted = fuels_nea.multiply(link_ratio)
    #     fuels_consumption = pd.concat([nea_adjusted, mecs_interpolated_data], axis=1)
    #     fuels_consumption = fuels_consumption.transpose()

    #     return electricity_consumption, fuels_consumption # Transfered to industrial_indicators_060220.xlsx[Manufacturing_Energy_Data]

    # def manufacturing(self):
    #     """Main datasource is the Manufacturing Energy Consumption Survey (MECS), conducted by the EIA since 1985 (supplemented for non-MECS years by
    #     estimates derived from the Annual Survey of Manufactures (ASM) and the Economic Census (EC) conducted every five years)
    #     https://www.eia.gov/consumption/manufacturing/data/2014/
    #     https://www.eia.gov/consumption/manufacturing/data/2014/#r4
    #     """
    #     electricity_consumption, fuels_consumption = self.manufacturing_energy()
    #     va_quant_index, go_quant_index = self.call_activity_data()
    #     data_dict = {'energy': {'elec': electricity_consumption, 
    #                             'fuels': fuels_consumption}, 
    #                  'activity': {'gross_output': go_quant_index,
    #                               'value_added': va_quant_index}}
    #     return data_dict

if __name__ == '__main__':
    # data = manufacturing()
    # print(data)
    pass