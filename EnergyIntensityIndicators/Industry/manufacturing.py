import pandas as pd

from EnergyIntensityIndicators.Industry import BEA_api
from EnergyIntensityIndicators.get_census_data import Econ_census
from EnergyIntensityIndicators.get_census_data import Asm
from EnergyIntensityIndicators.utilites.standard_interpolation import standard_interpolation

class Manufacturing:
    """Class to collect and process manufacturing data for the industrial sector
    """
    def __init__(self):
    # ASMdata_date.xlsx

    # ind_hap3_date.xlsx

    #2014_MECS = 'https://www.eia.gov/consumption/manufacturing/data/2014/'  # Table 4.2


    # Table 3.1 and 3.2 (MECS total fuel consumption)  Table 3.1 shows energy
    # consumption by fuel in physical units, including the total across all fuels expressed in trillion Btu and
    # electricity in kWh. From Table 3.1, total fuel consumption in Btu can be calculated as difference between
    # total energy and electricity consumption after conversion to Btu. Table 3.2 only differs from Table 3.1 by
    # showing all fuel types in Btu.



    MER_Table24_Industrial_Energy_Consumption = [0]

    # For 2014, the values for total energy consumption and electricity consumption, both defined in terms of
    # trillion Btu, from Table 3.2 are transferred to spreadsheet ind_hap3. Worksheet MECS_Fuel in this
    # spreadsheet has been used to collect the fuel consumption estimates for all the MECS dating back to the
    # first MECS in 1985. The 2014 data are located in the cell range F218:F238.
    # The first six NAICS sectors are aggregated into three sectors (311-312, 313-314, and 315-316) as a part
    # of the set of manufacturing indicators. The energy consumption data under this revised sectoring
    # classification are shown in the columns to the right, columns O and P.



    # Energy prices
    MECS_Table72 = [0]


    def get_historical_mecs(self):
        
        """Read in historical MECS csv, format (as in e.g. Coal (MECS) Prices)
        """
        historical_mecs = pd.read_csv('./')
        return historical_mecs

    def manufacturing_prices(self):
        """Call ASM API method from Asm class in get_census_data.py
        Specify three-digit NAICS Codes
        """
        fuel_types = ['Gas', 'Coal', 'Distillate', 'Residual', 'LPG', 'Coke', 'Other']
        naics = 

        asm_price_data = []
        for f in fuel_types: 
            predicted_fuel_price = Mfg_prices().calc_calibrated_predicted_price(latest_year=self.end_year, f, naics)
            asm_price_data.append(predicted_fuel_price)

        asm_price_data = pd.concat(asm_price_data)
        return asm_price_data

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
    def calc_composite_price(quantity_shares, interp_prices):  # Where is this used???
        """
        Take the sum of the product of quantity shares and the interpolated
        prices (calcualted using asm_price_fit.py), by 3-digit NAICS for a
        given MECS year.


        """
        composite_prices = quantity_shares.multiply(interp_prices, axis=1).sum(axis=1)

        return composite_prices

    # Corresponds to data in MECS_Fuel tab in Indhap3 spreadsheet, which
    # is connected to the MECS_Annual_Fuel1 and MECS_Annual_Fuel2
    # tabs in the same spreadsheet.
    def import_mecs_fuel(self):
        """
        Imports MECS data on fuel use by 3-digit NAICS code. In the future,
        these values will need to be manually downloaded from Table 3.2
        and added to csv.
        """

        # import a CSV file of historical MECS fuel use from Table 3.2
        # Will need to aggregate NAICS 311+312, 313+314, and 315+316
        allfos = pd.read_csv('./Data/ALLFOS_historical.csv')
        allfos = allfos.groupby('NAICS').sum()
        allfos = allfos.loc[:, 1970:1988]
        allfos_to_transfer = allfos.loc[321:339, :]
        allfos_to_transfer.loc['311+312', :] = allfos.loc[311:312, :].sum(axis=0)
        allfos_to_transfer.loc['313+314', :] = allfos.loc[313:314, :].sum(axis=0)
        allfos_to_transfer.loc['315+316', :] = allfos.loc[315:316, :].sum(axis=0)
        return mecs_fuel

    @staticmethod
    def adjust_323(printing):
        """Adjustments where there are 4-digit series consistent over 1977 to 1987
        """        

        data_2711 = ['2711', 2484.2, 2412.5, np.nan, np.nan, np.nan, np.nan, np.nan, 
                     2591.7, np.nan, np.nan, 2485.1, 2556.6, 2598.6, 2435.7, 
                     2837.7, 2895.5, 3076.9, 3357.1]
        data_2721 = ['2721', 405.8, 381.5, np.nan, np.nan, np.nan, np.nan, np.nan, 
                     298.8, np.nan, np.nan, 283.2, 346, 467.4, 565.6, 394, 
                     441.7, 425, 471.2]
        data_2731 = ['2731', 676.1, 968.2, np.nan, np.nan, np.nan, np.nan, np.nan,
                     267.6, np.nan, np.nan, 257.3, 246.4, 236.9, 238.2, 
                     314.7, 299.1, 310.9, 321.4]
        printing_data = [data_2711, 
                         data_2721,
                         data_2731]
        col_names = ['printing_type'] + list(range(1970, 1987 + 1))
        missing_years = [1972, 1973, 1974, 1975, 1976]
        printing_df = pd.DataFrame(printing_data, columns=col_names).set_index('printing_type')
        printing_df.loc['total', :] = printing_df.sum(axis=0)
        printing_df.loc['tbtu', :] = printing_df.loc['total', :].multiply(3.412 * 0.001)
        printing_df.loc['share', :] = printing_df.loc['tbtu', :].divide(printing)
        printing_df.loc['share', missing_years] = [0.4, 0.38, 0.36, 0.34, 0.32]
        printing_df.loc['share', 1978] = (0.33 * printing_df.loc['share', 1977] 
                                          + 0.67 * printing_df.loc['share', 1980]) * printing[1978]
        printing_df.loc['share', 1979] = (0.67 * printing_df.loc['share', 1977] 
                                          + 0.33 * printing_df.loc['share', 1980]) * printing[1979]

        printing_df.loc['adj_323', :] = printing_df.loc['share', :]
        printing_df.loc['adj_323', missing_years] = printing_df.loc['share', missing_years].multiply(printing[missing_years])
        return printing_df.loc['adj_323', :]
    
    @staticmethod
    def adjust_elecnea():
        """TODO: Finish
        """        
        adj_321_337 = [416.4, np.nan, np.nan, 391.6, 335.7, 341.3, 
                       390.6, 483.6, 458.9, 474.8, 641.7] # 146
        adj_321_337 = pd.DataFrame 

         # 142

        
    def import_mecs_electricity(self):
        """
        ### NOT SURE IF ASM or MECS ELECTRICITY DATA ARE USED ###
        Imports MECS data on electricityuse by 3-digit NAICS code.
        In the future,these values will need to be manually downloaded from
        Table 3.2 and added to csv.
        """
        
        # ELEC
        elec_nea = pd.read_csv('./Data/ELECNEA_historical.csv')
        elechap3b = elec_nea.groupby('NAICS').sum()
        elechap3b = elechap3b[list(range(1985, 1987 + 1))]
        elechap3b.loc['323', :] = self.adjust_323(elechap3b.loc['323', :])

        man_intensity_study = pd.read_csv('./Industry/Data/data_from_manufacturing_intensity_study.csv')
        man_intensity_study = man_intensity_study.groupby('Indicators').sum().reset_index()
        NAICS_codes = ['311+312', '313+314', '315+316', '321', '322', '323', '324', '325', '326', '327', 
                       '331', '332', '333', '334', '335', '336', '337', '339']
        naics_to_indicators = dict((i, n) for i, n in enumerate(NAICS_codes))
        man_intensity_study['NAICS'] = man_intensity_study['Indicators'].apply(lambda x: naics_to_indicators.get(x))
        man_intensity_study = man_intensity_study.set_index('NAICS') # from the ASM (Ind_hap3_122219/ASM2)
        asm2 = man_intensity_study.multiply(3.412 * 0.001)
        asm2 = asm2[[1985, 1986, 1987]]
        sic37 = [34537, 35831, 38291]
        asm2.loc[336, [1985, 1986, 1987]] = [s * 3.412 * 0.001 for s in sic37] # For 336 use use total for SIC 37


        merged_data = elechap3b.merge(asm2, left_index=True, right_index=True, how='outer') # from elechap3b and ASM2

        nea_based_data.loc['323', :] = 
        nea_based_data.loc[['321', '332', '333', '334', '335', '336', '337', '339'], list(range(1970, 1976 + 1))] = 
        nea_based_data.loc[['321', '333'], 1977 :] = 
        nea_based_data.loc[['331', '337'], 1977 :] = 
        factors = nea_based_data[[1977]].divide(merged_data[1977])
        nea_based_data[1970:1976] = merged_data[1970:1976].multiply(factors)


        # import a CSV file of historical MECS electricity use from Table 3.2
        # Will need to aggregate NAICS 311+312, 313+314, and 315+316
        mecs_published = 
        table31_net_elec =  # different in years < 2010 and after
        mecs_fuel = mecs_published.subtract(table31_net_elec)
        mecs_fuel_to_transfer = mecs_fuel.loc[321:339, ['Net Elec', 'Total Fuel']]
        mecs_fuel_to_transfer.loc['311+312', :] = mecs_fuel.loc[311:312, :].sum(axis=0)
        mecs_fuel_to_transfer.loc['313+314', :] = mecs_fuel.loc[313:314, :].sum(axis=0)
        mecs_fuel_to_transfer.loc['315+316', :] = mecs_fuel.loc[315:316, :].sum(axis=0)
        mecs_interpolated_data = # USE STANDARD INTERPOLATION METHOD # from mecs_annual_fuel2
        mecs_interpolated_data.loc[324:325, :] = elec.loc[324:325, :] 
        return mecs_elect
    
    # Data used in ASMdata_010220.xlsx[3DNAICS]
    def call_census_data(self):
        """
        Use Census_api class to call fuel and electricity expenditures from
        Annual Survey of Manufacturers and Economic Census (for years ending
        with 2 and 7).
        """

    # Data used in industrial_indicators[Manufacturing]
    def call_activity_data(self):
        """
        Call BEA API for gross ouput and value add by 3-digit NAICS.
        """
        va_quant_index, go_quant_index = BEA_api(years=list(range(1949, 2018))).chain_qty_indexes()
        # HERE: select columns 
        return va_quant_index, go_quant_index

    @staticmethod
    def interpolate_mecs(mecs_fuel, ASMdata_010220_xlsx_data):
        """
        Between-MECS-year interpolations are made in MECS_Annual_Fuel1
        and MECS_Annual_Fuel2 tabs in Ind_hap3 spreadsheet.
        Interpolations are also based on estimates developed in
        ASMdata_010220.xlsx[3DNAICS], which ultimately tie back to MECS fuel data
        from Table 4.2 and Table 3.2
        """
        standard_interpolation(dataframe=, name_to_interp= , axis=)

        # in ASMdata_010220.xlsx[Final_quantities_w_ASM_85] 
        mecs_data_sic = 
        ratio_fuel_offsite_pre98 = standard_interpolation(dataframe=mecs_data_sic, name_to_interp= , axis=)

        data_98 = 
        mecs_tables_31_32 = 
        mecs_table42 = 

        ratio_fuel_offsite = 

    def manufacturing(self):
        """Main datasource is the Manufacturing Energy Consumption Survey (MECS), conducted by the EIA since 1985 (supplemented for non-MECS years by
        estimates derived from the Annual Survey of Manufactures (ASM) and the Economic Census (EC) conducted every five years)
        https://www.eia.gov/consumption/manufacturing/data/2014/
        https://www.eia.gov/consumption/manufacturing/data/2014/#r4
        """

        mecs = # from [MECS_prices_101116b.xlsx]MECS_data_SIC
        asm = # [Ind_hap3_101316.xlsx]ASM_Fuel_Cost_1985-88
        mecs_asm_ratio = mecs.divide(asm)
        mecs_based_expenditure = # from Expend_ratios_revised_1985-97 and Expend_ratios_revised
        dollar_per_mmbtu =  # from quantity_shares_revised
        jan_2020_estimate = mecs_based_expenditure.divide(dollar_per_mmbtu)
        ratio_fuel_to_offsite = # from mixed sources


        final_quantities_asm_85 = jan_2020_estimate.multiply(ratio_fuel_to_offsite)
        asm_data = final_quantities_asm_85.loc[321:339, :] # ASMdata_010330.xlsx , Final_quant_elec_w_ASM_87'
        asm.loc['311+312', :] = final_quantities_asm_85.loc[311:312, :].sum(axis=0)
        asm.loc['313+314', :] = final_quantities_asm_85.loc[313:314, :].sum(axis=0)
        asm.loc['315+316', :] = final_quantities_asm_85.loc[315:316, :].sum(axis=0)

        # Ind_hap3_122219.xlsx[ASM_Annual_Elec_1970on]
        link_ratio = asm[[1987]].divide(nea_based_data[1987])

        nea_based_data_linked = nea_based_data.multiply(link_ratio, axis=1)

        electricity_consumption = pd.concat([nea_based_data_linked, asm_data], axis=1)
        electricity_consumption = electricity_consumption.transpose()

        

        fuels_nea =  # fallhap3


        # Ind_hap3_122219.xlsx[ASM_Annual_Fuel3_1970on]
        link_ratio = mecs_interpolated_data[[1985]].divide(fuels_nea[1985])
        nea_adjusted = fuels_nea.multiply(link_ratio)
        fuels_consumption = pd.concat([nea_adjusted, mecs_interpolated_data], axis=1)
        fuels_consumption = fuels_consumption.transpose()

        

        return electricity_consumption, fuels_consumption # Transfered to industrial_indicators_060220.xlsx[Manufacturing_Energy_Data]
    
    def main(self):
        return None

if __name__ == '__main__':
    main()