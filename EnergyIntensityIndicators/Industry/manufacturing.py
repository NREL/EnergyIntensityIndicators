import pandas as pd

from EnergyIntensityIndicators.Industry import BEA_api

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

        asm_price_data = Mfg_prices().calc_calibrated_predicted_price(latest_year=self.end_year, fuel_type, naics)
        return asm_price_data

    @staticmethod
    def calc_quantity_shares(mecs42_df): # From ASMdata_010220.xlsx[Quantity_shares_revised]
        """
        For a given MECS year, take 3-digit NAICS by fuel (TBtu),
        calculate sum, then calcuate quantity shares
        """

        return quantity_shares

    @staticmethod
    def calc_composite_price(quantity_shares, interp_prices):  # Where is this used???
        """
        Take the sum of the product of quantity shares and the interpolated
        prices (calcualted using asm_price_fit.py), by 3-digit NAICS for a
        given MECS year.


        """

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
        return mecs_fuel

    def import_mecs_electricity(self):
        """
        ### NOT SURE IF ASM or MECS ELECTRICITY DATA ARE USED ###
        Imports MECS data on electricityuse by 3-digit NAICS code.
        In the future,these values will need to be manually downloaded from
        Table 3.2 and added to csv.
        """

        # import a CSV file of historical MECS electricity use from Table 3.2
        # Will need to aggregate NAICS 311+312, 313+314, and 315+316
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
        go_nominal = BEA_api.get_data(years=self.years, table_name='go_nominal')
        va_nomial = BEA_api.get_data(years=self.years, table_name='va_nomial')
        va_nominal_12 = va_nomial[2012]
        
        go_quant_index = BEA_api.get_data(years=self.years, table_name='go_quant_index')
        va_quant_index = BEA_api.get_data(years=self.years, table_name='va_quant_index')
        historical_data = BEA_api.import_historical()
        historical_va = historical_data['historical_va'] 
        
        # nonmanufacturing_index
        value_added = historical_va.loc[['Agriculture, forestry, fishing, and hunting', 'Mining', 'Construction'], list(range(1969, 1997 + 1))]
        value_added.loc[['Agriculture, forestry, fishing, and hunting', 'Mining', 'Construction'], 1998 : ] = va_nomial.loc[['  Agriculture, forestry, fishing, and hunting', '  Mining', '  Construction'], 1998 : ]
        value_added.loc['Sum'] = value_added.sum(axis=0)
        
        quantity_index = historical_va.loc[['Agriculture, forestry, fishing, and hunting', 'Mining', 'Construction'], list(range(1969, 1997 + 1))]
        quantity_index.loc[['Agriculture, forestry, fishing, and hunting', 'Mining', 'Construction'], 1998 : ] = va_nomial.loc[['  Agriculture, forestry, fishing, and hunting', '  Mining', '  Construction'], 1998 : ]

        laspeyres_quantity
        paasche_quantity = 
        fisher_nonmanufacturing_relatives = 
        industrial_quantity_index = 
        quantity_index.loc[['Nonmanufacturing', 'Manufacturing'], :] = 


        
        transformed_va_quant_index = va_quant_index.multiply(va_nominal_12, index=1).multiply(.01)

    @staticmethod
    def interolate_mecs(mecs_fuel, ASMdata_010220_xlsx_data):
        """
        Between-MECS-year interpolations are made in MECS_Annual_Fuel1
        and MECS_Annual_Fuel2 tabs in Ind_hap3 spreadsheet.
        Interpolations are also based on estimates developed in
        ASMdata_010220.xlsx[3DNAICS], which ultimately tie back to MECS fuel data
        from Table 4.2 and Table 3.2
        """



    def manufacturing(self):
        """Main datasource is the Manufacturing Energy Consumption Survey (MECS), conducted by the EIA since 1985 (supplemented for non-MECS years by
        estimates derived from the Annual Survey of Manufactures (ASM) and the Economic Census (EC) conducted every five years)
        https://www.eia.gov/consumption/manufacturing/data/2014/
        https://www.eia.gov/consumption/manufacturing/data/2014/#r4
        """

        return None
    
    def main(self):
        return None

if __name__ == '__main__':
    main()