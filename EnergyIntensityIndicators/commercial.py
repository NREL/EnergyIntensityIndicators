import pandas as pd
import datetime as dt
from sklearn import linear_model
import numpy as np
import math
import statsmodels.api as sm
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from EnergyIntensityIndicators.pull_eia_api import GetEIAData
from EnergyIntensityIndicators.weather_factors import WeatherFactors
from EnergyIntensityIndicators.residential import ResidentialIndicators
from EnergyIntensityIndicators.LMDI import CalculateLMDI


"""Overview and Assumptions: 
A. A national time series of floorspace for the commercial buildings in the US
B. Weather adjustment for the four census regions
C. Adjustment for the major reclassifications of customers by individual utilities. The reclassifications generally involve customers 
   whose electricity purchases were classified under an industrial rate structure being moved to commercial rate struture, and to a
   lesser degree, customers moving from a commercial rate to an industrial rate classification. These reclassfications can distort the 
   short term aggregate changes in commercial and industrial electricity consumption, as reported by EIA's Estimation of National Commercial 
   Floorspace

Data Sources: New construction is based on data from Dodge Data and Analytics, available from the published versions of the Statistical
              Abstract of the United States (SAUS)

Methodology: Perpetual inventory model, where estimates of new additions and removals are added to the previous year's estimate of stock
             to update the current year"""

class CommercialIndicators(CalculateLMDI):
    """
    Data Sources: 
    - New construction is based on data from Dodge Data and Analytics. Dodge data on new floor space additions is available 
    from the published versions of the Statistical Abstract of the United States (SAUS). The Most recent data is from the 2020 
    SAUS, Table 995 "Construction Contracts Started- Value of the Construction and Floor Space of Buildings by Class of Construction:
    2014 to 2018". 
    """    

    def __init__(self, directory, output_directory, level_of_aggregation, lmdi_model=['multiplicative'], end_year=2018, base_year=1985):
        self.end_year = end_year
        self.sub_categories_list = {'Commercial': {'Commercial_Total': None}} #, 'Total_Commercial_LMDI_UtilAdj': None}
        self.eia_comm = GetEIAData('commercial')
        self.energy_types = ['elec', 'fuels', 'deliv', 'source', 'source_adj']
        super().__init__(sector='commercial', level_of_aggregation=level_of_aggregation,lmdi_models=lmdi_model, 
                         directory=directory, output_directory=output_directory, categories_dict=self.sub_categories_list, energy_types=self.energy_types, 
                         base_year=base_year, end_year=end_year)
        # self.cbecs = 
        # self.residential_housing_units = [0] # Use regional estimates of residential housing units as interpolator, extrapolator via regression model

        # self.mer_data23_May_2016 = GetEIAData.eia_api(id_='711251')  # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711251'
        # self.mer_data23_Jan_2017 = GetEIAData.eia_api(id_='711251')   # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711251'
        # self.mer_data23_Dec_2019 =  GetEIAData.eia_api(id_='711251')  # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711251'
        # self.AER11_Table21C_Update = GetEIAData.eia_api(id_='711251')  # Estimates?

    def collect_input_data(self, dataset_name):
        datasets = {'national_calibration': self.eia_comm.national_calibration(), 'conversion_factors': self.eia_comm.conversion_factors(include_utility_sector_efficiency=True), 
                    'SEDS_CensusRgn': self.eia_comm.get_seds(), 'mer_data_23': self.eia_comm.eia_api(id_='711251', id_type='category')}
        return datasets[dataset_name]

    def adjusted_supplier_data(self):
        """
        This worksheet adjusts some of commercial energy consumption data
        as reported in the Annual Energy Review.  These adjustments are 
        based upon state-by-state analysis of energy consumption in the 
        industrial and commercial sectors.  For electricity, there have been 
        a number of reclassifications by utilities since 1990 that has moved 
        sales from the industrial sector to the commercial sector. 

        The adjustment for electricity consumption is based upon a
        state-by-state examination of commercial and electricity 
        sales from 1990 through 2011.  This data is collected
        by EIA via Survey EIA-861.  Significant discontinuities
        in the sales data from one year to the next were removed.  
        In most cases, these adjustments caused industrial consumption
        to increase and commercial consumption to decrease.  The
        spreadsheet with these adjustments is Sectoral_reclassification5.xls  (10/25/2012).

        In 2009, there was a significant decline in commercial
        electricity sales in MA and a corresponding increase in industrial sales
        Assuming that industrial consumption would have
        fallen by 2% between 2008 and 2009, the adjustment
        to both the commercial (+) and industrial sectors (-) was
        estimated to be 7.61 TWh.  .
        The 7.61 TWh converted to Tbtu is 26.0.  This value is then added 
        to the negative 164.0 Tbtu in 2009 and subsequent years.  

        State Energy Data System (Jan. 2017) via National Calibration worksheet 

        """

        # 1949-1969 
        published_consumption_trillion_btu = self.eia_comm.eia_api(id_='TOTAL.ESCCBUS.A', id_type='series')  # Column W (electricity retail sales to the commercial sector) # for years 1949-69
        published_consumption_trillion_btu = published_consumption_trillion_btu.rename(columns={'Electricity Retail Sales to the Commercial Sector, Annual, Trillion Btu': 'published_consumption_trillion_btu'})
        # 1970-2018
        national_calibration = self.collect_input_data('national_calibration')
        published_consumption_trillion_btu.loc['1970':, ['published_consumption_trillion_btu']] = national_calibration.loc['1970':, ['Final Est. (Trillion Btu)_elec']].values  # Column G (electricity final est) # for years 1970-2018
        # 1977-1989

        years = list(range(1977, max(published_consumption_trillion_btu.index.astype(int)) + 1))
        years = [str(y) for y in years]
        # adjustment_to_commercial_trillion_btu_early = number_for_1990
        adjustment_to_commercial_trillion_btu = [9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340312799975,
                                                 9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340312799975,
                                                 9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340312799975, 9.21340654000005, 
                                                 29.77918535999970, 10.21012680399960, 1.70263235599987, -40.63866012000020, -40.63865670799990, 
                                                 -117.72073870000000, -117.72073528800000, -117.72073187600000, -117.72072846400000, -162.61452790400100, 
                                                 -136.25241618800100, -108.91594645600000, -125.97594304400000, -125.97593963200100, -163.95020989600000,
                                                 -163.95020648400000, -163.95020307200000, -137.98708428968000, -137.98487966000100, -137.98487966000100, 
                                                 -137.98487966000100, -137.98487966000100, -137.98487966000100, -137.98487966000100, -137.98487966000100, 
                                                 -137.98487966000100, -137.98487966000100] # First value is for 1977 - 2018
        adjustment_df = pd.DataFrame([years, adjustment_to_commercial_trillion_btu]).transpose()
        adjustment_df.columns = ['Year', 'adjustment_to_commercial_trillion_btu']

        adjusted_supplier_data = adjustment_df.merge(published_consumption_trillion_btu, how='outer', on='Year')
        adjusted_supplier_data['adjustment_to_commercial_trillion_btu'] = adjusted_supplier_data['adjustment_to_commercial_trillion_btu'].fillna(0)

        adjusted_supplier_data = adjusted_supplier_data.set_index('Year')
        adjusted_supplier_data['adjusted_consumption_trillion_btu'] = adjusted_supplier_data['adjustment_to_commercial_trillion_btu'].add(adjusted_supplier_data['published_consumption_trillion_btu'])
        adjusted_supplier_data = adjusted_supplier_data.sort_index(ascending=True)

        return adjusted_supplier_data[['adjusted_consumption_trillion_btu']]


    @staticmethod
    def get_saus():
        """Get Data from the Statistical Abstract of the United States (SAUS)
        """        
        saus_2002 = pd.read_csv('./EnergyIntensityIndicators/Data/SAUS2002_table995.csv').set_index('Year')
        saus_1994 = {1980: 738, 1981: 787, 1982: 631, 1983: 716, 1984: 901, 1985: 1039, 1986: 960, 1987: 933, 
                    1988: 883, 1989: 867, 1990: 694, 1991: 477, 1992: 462, 1993: 479}
        saus_2001 = {1980: 738, 1981: None, 1982: None, 1983: None, 1984: None, 1985: 1039, 1986: None, 1987: None, 
                    1988: None, 1989: 867, 1990: 694, 1991: 476, 1992: 462, 1993: 481, 1994: 600, 1995: 700, 
                    1996: 723, 1997: 855, 1998: 1106, 1999: 1117, 2000: 1176}
        saus_merged = dict() 
        for (year, value) in saus_2001.items():
            if value == None: 
                set_value = saus_1994[year]
            else: 
                set_value = value
            saus_merged[year] = set_value

        saus_merged_df = pd.DataFrame.from_dict(saus_merged, orient='index', columns=['Value'])
        return saus_2002, saus_merged_df

    @staticmethod
    def dod_compare_old():
        """[summary]

        DODCompareOld Note from PNNL (David B. Belzer): "These series are of unknown origin--need to check Jackson and Johnson 197 (sic)?

        Returns:
            [type]: [description]
        """        
        dod_old = pd.read_csv('./EnergyIntensityIndicators/Data/DODCompareOld.csv').set_index('Year')

        # dod_old['Misc'] = dod_old['Soc/Misc'].subtract(dod_old['Soc/Amuse'])
        # dod_old = dod_old.drop(columns='Soc/Misc')
        # dod_old['Total'] = dod_old.sum(axis=1)
        cols_list = ['Retail', 'Auto R', 'Office', 'Warehouse']
        dod_old['Commercial'] = dod_old[cols_list].sum(axis=1)
        dod_old_subset = dod_old.loc[list(range(1960, 1982)), cols_list]
        dod_old_hotel = dod_old.loc[list(range(1980, 1990)), ['Commercial']]
        return dod_old, dod_old_subset, dod_old_hotel 

    def dodge_adjustment_ratios(self, dodge_dataframe, start_year, stop_year, adjust_years, late):
        """(1985, 1990) or (1960, 1970)

        Args:
            dodge_dataframe ([type]): [description]
            start_year ([type]): [description]
            stop_year ([type]): [description]

        Returns:
            [type]: [description]
        """        
        year_indices = self.years_to_str(start_year, stop_year)
        revision_factor_commercial = dodge_dataframe.loc[year_indices, ['Commercial']].sum(axis=0).values
        categories = ['Retail', 'Auto R', 'Office', 'Warehouse', 'Hotel']
        if late:
            col = 'Commercial'
        else: 
            col = 'Commercial, Excl Hotel'
        for category in categories: 

            revision_factor_cat = dodge_dataframe.loc[year_indices, [category]].sum(axis=0).values / revision_factor_commercial
            dodge_dataframe.loc[adjust_years, [category]] = dodge_dataframe.loc[adjust_years, [col]].values * revision_factor_cat[0]

        return dodge_dataframe

    def west_inflation(self):
        """Jackson and Johnson Estimate of West Census Region Inflation Factor, West Region Shares Based on CBECS

        Note from PNNL: "Staff: Based upon CBECS, the percentage of construction in west census region was slightly greater
         in the 1900-1919 period than in 1920-1945.  Thus, factor is set to 1.12, approximately same as 1925 value published
         by Jackson and Johnson"
"
        """        
        # hist_stat = self.hist_stat()['Commercial  (million SF)'] # hist_stat column E
        # # west inflation column Q
        ornl_78 = {1925: 1.127, 1930: 1.144, 1935: 1.12, 1940: 1.182, 1945: 1.393, 1950: 1.216, 1951: 1.237, 
                   1952: 1.224, 1953: 1.209, 1954: 1.213, 1955: 1.229}
        all_years = list(range(min(ornl_78.keys()), max(ornl_78.keys()) + 1))
        increment_years = list(ornl_78.keys())
        
        final_factors = {year: 1.12 for year in list(range(1919, 1925))}
        for index, y_ in enumerate(increment_years):
            if index > 0:
                year_before = increment_years[index - 1]
                num_years = y_ - year_before
                infl_factor_year_before = ornl_78[year_before] 
                infl_factor_y_ = ornl_78[y_]  
                increment = 1 / num_years
                for delta in range(num_years):
                    value = infl_factor_year_before * (1 - increment * delta) + \
                        infl_factor_y_ * (increment * delta)
                    year = year_before + delta
                    final_factors[year] = value
        final_factors_df = pd.DataFrame.from_dict(final_factors, columns=['Final Factors'], orient='index')
        return final_factors_df

    @staticmethod
    def years_to_str(start_year, end_year):
        list_ = list(range(start_year, end_year + 1))
        return [str(l) for l in list_]
    
    def hist_stat(self):
        """Historical Dodge Data through 1970

        Data Source: Series N 90-100 Historical Statistics of the U.S., Colonial Times to 1970
        """
        historical_dodge = pd.read_csv('./EnergyIntensityIndicators/Data/historical_dodge_data.csv').set_index('Year')        
        pub_inst_values = historical_dodge.loc[list(range(1919, 1925)), ['Pub&Institutional']].values
        total_1925_6 = pd.DataFrame.sum(historical_dodge.loc[list(range(1925, 1927)),], axis=0).drop(index='Commercial  (million SF)')
        inst_pub_total_1925 = pd.DataFrame.sum(historical_dodge.loc[1925,].drop('Commercial  (million SF)'), axis=0)
        inst_pub_total_1926 = pd.DataFrame.sum(historical_dodge.loc[1926,].drop('Commercial  (million SF)'), axis=0)
        inst_pub_total_1925_6 = inst_pub_total_1925 + inst_pub_total_1926

        shares = total_1925_6.divide(inst_pub_total_1925_6)
        for col in list(total_1925_6.index): 
            values = historical_dodge.loc[list(range(1919, 1925)), ['Pub&Institutional']].multiply(shares[col]).values
            historical_dodge.at[list(range(1919, 1925)), col] = values

        historical_dodge.at[list(range(1919, 1925)), ['Pub&Institutional']] = pub_inst_values
        return historical_dodge

    def hist_stat_adj(self):
        """Adjust historical Dodge data to account for omission of data for the West Census Region prior to 1956

        Returns:
            [type]: [description]
        """        
        hist_data = self.hist_stat()
        west_inflation = self.west_inflation()
        hist_data = hist_data.merge(west_inflation, how='outer', left_index=True, right_index=True)
        hist_data['Final Factors'] = hist_data['Final Factors'].fillna(1)
        adjusted_for_west = hist_data.drop(columns=['Final Factors', 'Pub&Institutional']).multiply(hist_data['Final Factors'].values, axis=0)
        return adjusted_for_west.loc[list(range(1919, 1960)), :]

    def dodge_revised(self):
        """Dodge Additions, adjusted for omission of West Census Region prior to 1956
        """       
        saus_2002, saus_merged = self.get_saus()
        dod_old, dod_old_subset, dod_old_hotel = self.dod_compare_old()
        west_inflation = self.hist_stat_adj()
        
        dodge_revised = pd.read_csv('./EnergyIntensityIndicators/Data/Dodge_Data.csv').set_index('Year')
        dodge_revised.index = dodge_revised.index.astype(str)

        dodge_revised = dodge_revised.reindex(dodge_revised.columns.tolist() + ['Commercial, Excl Hotel', 'Hotel'], axis=1).fillna(np.nan)

        years_1919_1989 = self.years_to_str(1919, 1990)
        years_1990_1997 = self.years_to_str(1990, 1997)

        dodge_revised.loc[self.years_to_str(1960, 1981), ['Retail', 'Auto R', 'Office', 'Warehouse']] = dod_old_subset.values 
        
        dodge_revised.loc[self.years_to_str(1919, 1959), ['Commercial, Excl Hotel']] =  west_inflation['Commercial  (million SF)'].values.reshape(41, 1) # hist_stat_adj column Q
        hist_adj_cols = ['Education', 'Hospital', 'Public', 'Religious', 'Soc/Amuse', 'Misc']
        dodge_revised.loc[self.years_to_str(1919, 1959), hist_adj_cols] = west_inflation.drop('Commercial  (million SF)', axis=1).values
        dodge_revised.loc[self.years_to_str(1990, 1998), hist_adj_cols] = saus_2002.loc[self.years_to_str(1990, 1998), ['Educational', 'Health', 'Pub. Bldg', 'Religious', 'Soc/Rec', 'Misc.']].values
        dodge_revised.loc[self.years_to_str(1990, 2003), ['Soc/Misc']] = saus_2002.loc[self.years_to_str(1990, 2001), ['Soc/Rec']].add(saus_2002.loc[self.years_to_str(1990, 2001), ['Misc.']].values)
        dodge_revised.loc[self.years_to_str(1999, 2001), 'Misc']  = saus_2002.loc[self.years_to_str(1999, 2001), ['Misc.']].values.reshape(3,)
        dodge_revised.loc[self.years_to_str(1961, 1989), 'Misc'] = dodge_revised.loc[self.years_to_str(1961, 1989), 'Soc/Misc'].subtract(dodge_revised.loc[self.years_to_str(1961, 1989), 'Soc/Amuse'].values)
        dodge_revised.loc[str(2000), 'Hospital'] = saus_2002.loc[str(2000), 'Health']

        dodge_revised.loc[self.years_to_str(1960, 1989), ['Commercial, Excl Hotel']] =  dodge_revised.loc[self.years_to_str(1960, 1989), ['Retail', 'Auto R', 'Office', 'Warehouse']].sum(axis=1)


        hotel_80_89 = saus_merged.loc[list(range(1980, 1989 + 1)), ['Value']].subtract(dod_old_hotel.values) 

        dodge_revised.loc[self.years_to_str(1980, 1989), ['Hotel']] = hotel_80_89

        hotel_80_89_ratio = hotel_80_89.sum(axis=0).values / dodge_revised.loc[self.years_to_str(1980, 1989), ['Commercial, Excl Hotel']].sum(axis=0).values

        dodge_revised.loc[self.years_to_str(1919, 1979), ['Hotel']] = dodge_revised.loc[self.years_to_str(1919, 1979), ['Commercial, Excl Hotel']].values * hotel_80_89_ratio


        dodge_revised.loc[years_1990_1997, ['Commercial, Incl Hotel']] =  saus_2002.loc[years_1990_1997, ['Commercial']].values

  
        dodge_revised.loc[self.years_to_str(1985, 1989), ['Commercial']] = saus_merged.loc[list(range(1985, 1989 + 1)), ['Value']].values
        dodge_revised.loc[self.years_to_str(1990, 2018), ['Commercial']] = dodge_revised.loc[self.years_to_str(1990, 2018), ['Commercial, Incl Hotel']].values

        dodge_revised = self.dodge_adjustment_ratios(dodge_revised, 1960, 1969, adjust_years=self.years_to_str(1919, 1959), late=False)
        dodge_revised = self.dodge_adjustment_ratios(dodge_revised, 1985, 1989, adjust_years=self.years_to_str(1990, 2018), late=True)
        

        dodge_revised.loc[years_1919_1989, ['Commercial, Incl Hotel']] = dodge_revised.loc[years_1919_1989, ['Commercial, Excl Hotel']].add(dodge_revised.loc[years_1919_1989, ['Hotel']].values)
        dodge_revised['Total'] = dodge_revised.drop(['Commercial, Incl Hotel', 'Commercial, Excl Hotel'], axis=1).sum(axis=1).values
        return dodge_revised

    def dodge_to_cbecs(self):
        """Redefine the Dodge building categories more along the lines of CBECS categories. Constant fractions of floor space are moved among categories. 

        Returns:
            dodge_to_cbecs (dataframe): redefined data
        """    
        # Key Assumptions: 
        education_floor_space_office = .10
        auto_repair_retail = .80
        retail_merc_service = .80  # remainder to food service and sales
        retail_merc_service_food_sales = .11
        retail_merc_service_food_service = .90
        education_assembly = .05
        education_misc = .05 # (laboratories)
        health_transfered_to_cbecs_health = .75 # 25% to lodging (nursing homes)
        misc_public_assembly = .10 # (passenger terminals)

        dodge_revised = self.dodge_revised() # dataframe
        
        dodge_to_cbecs = pd.DataFrame(dodge_revised[['Total', 'Religious', 'Warehouse']]).rename(columns={'Total': 'Dodge_Totals'})
        dodge_to_cbecs['Office'] = dodge_revised['Office'] + education_floor_space_office * dodge_revised['Education']
        dodge_to_cbecs['Merc/Serv'] = retail_merc_service * (dodge_revised['Retail'] + auto_repair_retail * dodge_revised['Auto R'])
        dodge_to_cbecs['Food_Sales'] = retail_merc_service_food_sales * (dodge_revised['Retail'] + auto_repair_retail * dodge_revised['Auto R'])
        dodge_to_cbecs['Food_Serv'] = retail_merc_service_food_service * (dodge_revised['Retail'] + auto_repair_retail * dodge_revised['Auto R'])
        dodge_to_cbecs['Education'] = (1 - education_floor_space_office - education_assembly - education_misc) * dodge_revised['Education']
        dodge_to_cbecs['Health'] = health_transfered_to_cbecs_health * dodge_revised['Hospital']
        dodge_to_cbecs['Lodging'] = dodge_revised['Hotel']+ (1 - health_transfered_to_cbecs_health) * dodge_revised['Hospital']
        dodge_to_cbecs['Assembly'] = dodge_revised['Soc/Amuse'] +  misc_public_assembly * dodge_revised['Misc'] + education_assembly * dodge_revised['Education']
        dodge_to_cbecs['Other'] = dodge_revised['Public'] + (1 - misc_public_assembly) * dodge_revised['Misc'] + (1 - auto_repair_retail) * dodge_revised['Auto R'] + education_misc * dodge_revised['Education']
        dodge_to_cbecs['Redefined_Totals'] = dodge_to_cbecs.drop('Dodge_Totals', axis=1).sum(axis=1).values
        # dodge_to_cbecs = dodge_to_cbecs.drop()  # don't need totals?
        return dodge_to_cbecs

    def nems_logistic(self, dataframe, params):
        """[summary]

        Args:
            dataframe ([type]): [description]
            params (list): gamma, lifetime, 
        
        PNNL errors found: 
            - Column S in spreadsheet has CO-StatePop2.xls are incorrectly aligned with years
            - Column AL does not actually scale by 1.28 as suggested in the column header

        """    
        current_year = dt.datetime.now().year

        link_factors = pd.read_excel(f'{self.directory}/CO-EST_statepop2.xls', sheet_name='Stock', usecols='D:E', header=1, skiprows=158).rename(columns={1789: 'Year'})
        state_pop = link_factors.set_index('Year').rename(columns={' New': 'state_pop'})
        state_pop = state_pop[state_pop.index.notnull()]
        state_pop.index = state_pop.index.astype(int)
        state_pop.index = state_pop.index.astype(str)

        dataframe = dataframe.merge(state_pop, how='outer', left_index=True, right_index=True)
        dataframe = dataframe[dataframe.index.notnull()]
        dataframe = dataframe.reindex(columns=dataframe.columns.tolist() + ['adjusted_state_pop', 'adjusted_state_pop_scaled_b', 'adjusted_state_pop_scaled_c', 
                                                                            'scaled_additions_estimate_a', 'scaled_additions_estimate_b', 'scaled_additions_estimate_c', 
                                                                            'removal', 'adjusted_removals', 'old_stk_retain', 'floorspace_bsf'])
        dataframe['Year_Int'] = dataframe.index.astype(int)
        dataframe['age'] = dataframe['Year_Int'].subtract(current_year).multiply(-1)
        dataframe['remaining'] = ((dataframe['age'].divide(params[1])).pow(params[0]).add(1)).pow(-1)
        dataframe['inflate_fac'] = dataframe['remaining'].pow(-1)

        link_factor = 0.1
        adjusted_state_pop_1 = 40

        timing_wgts_current_yr = 0.4
        timing_wgts_lag_yr = 0.6
        benchmark_factor = 1

        dataframe.loc[str(1838), ['state_pop']] = 400
        dataframe.loc[self.years_to_str(1838, 1919), ['adjusted_state_pop']] = dataframe.loc[self.years_to_str(1838, 1919), ['state_pop']].values * link_factor
        dataframe.loc[self.years_to_str(1920, current_year), ['adjusted_state_pop']] = dataframe.loc[self.years_to_str(1920, current_year), ['Redefined_Totals']].values

        for year in self.years_to_str(1838, current_year):
            adjusted_state_pop_value = dataframe.loc[year, ['adjusted_state_pop']].values
            if year == '1838': 
                vpip_estimate = adjusted_state_pop_value
            elif year == '1920': 
                vpip_estimate = adjusted_state_pop_value
            else:
                adjusted_state_pop_year_before = dataframe.loc[str(int(year) - 1), ['adjusted_state_pop']].values
                vpip_estimate = (timing_wgts_current_yr * adjusted_state_pop_value + timing_wgts_lag_yr * adjusted_state_pop_year_before) * benchmark_factor
            dataframe.loc[year, 'VPIP-Estimate'] = vpip_estimate
        _variable = 1.2569  # This should be solved for
        x_column_value = _variable 
        db_estimates = 1.2
        db_estimates2 = [1.25 - 0.01*d for d in list(range(1990, 2021))]

        post_1989_scaling_factor = db_estimates # Should choose this
        variable_2 = 1.533  # This should be solved for

        without_lags = dataframe.loc[self.years_to_str(1990, current_year), ['adjusted_state_pop']].multiply(post_1989_scaling_factor)

        dataframe.loc[: str(1989), ['scaled_additions_estimate_a']] = dataframe.loc[: str(1989), ['VPIP-Estimate']].values * variable_2
        dataframe.loc[self.years_to_str(1990, current_year), ['scaled_additions_estimate_a']] = dataframe.loc[self.years_to_str(1990, current_year), ['VPIP-Estimate']].values * post_1989_scaling_factor
        dataframe.loc[:str(1989), ['adjusted_state_pop_scaled_b']] = dataframe.loc[self.years_to_str(1790, 1989), ['scaled_additions_estimate_a']].values
        dataframe.loc[self.years_to_str(1990, 2001), ['adjusted_state_pop_scaled_b']] = dataframe.loc[self.years_to_str(1990, 2001), ['scaled_additions_estimate_a']].values * 1.15
        dataframe.loc[self.years_to_str(2002, current_year), ['adjusted_state_pop_scaled_b']] = dataframe.loc[self.years_to_str(2002, current_year), ['scaled_additions_estimate_a']].values 
        
        dataframe.loc[str(1790), ['adjusted_state_pop_scaled_c']] = 1
        dataframe.loc[self.years_to_str(1791, 1989), ['adjusted_state_pop_scaled_c']] = dataframe.loc[self.years_to_str(1791, 1989), ['scaled_additions_estimate_a']].values
        dataframe.loc[self.years_to_str(1990, 2001), ['adjusted_state_pop_scaled_c']] = dataframe.loc[self.years_to_str(1990, 2001), ['scaled_additions_estimate_a']].values * 1.28
        dataframe.loc[self.years_to_str(2002, current_year), ['adjusted_state_pop_scaled_c']] = dataframe.loc[self.years_to_str(2002, current_year), ['scaled_additions_estimate_a']].values
        
        for y in self.years_to_str(1839, current_year):  
            years_diff = current_year - 1870
            start_year = int(y) - years_diff
            # first_index_year = int(dataframe.index[0])
            # if start_year < first_index_year:
            #     start_year = first_index_year
            year_index = self.years_to_str(start_year, int(y))
            remaining = dataframe.loc[self.years_to_str(1870, current_year), ['remaining']].values.flatten()
            adjusted_state_pop_scaled_b = dataframe.loc[year_index, ['adjusted_state_pop_scaled_b']].fillna(0).values.flatten()
            adjusted_state_pop_scaled_c = dataframe.loc[year_index, ['adjusted_state_pop_scaled_c']].fillna(0).values.flatten()

            b_value = np.dot(adjusted_state_pop_scaled_b, remaining)
            c_value = np.dot(adjusted_state_pop_scaled_c, remaining)
 
            dataframe.loc[y, ['scaled_additions_estimate_b']] = b_value
            dataframe.loc[y, ['scaled_additions_estimate_c']] = c_value

        removal_chg = 1 # Not sure what this is about
        fractions = [0.3, 0.4, 0.4, 0.35, 0.35, 0.35, 0.35, 0.3, 0.3, 0.3]
        fraction_retained = [f * removal_chg for f in fractions]
        
        for i in dataframe.index:
            if i >= '1870': 
                removal = dataframe.loc[i, ['scaled_additions_estimate_c']].values - dataframe.loc[str(int(i) - 1), ['scaled_additions_estimate_c']].values - dataframe.loc[i, ['scaled_additions_estimate_a']].values
                dataframe.loc[i, ['removal']] = removal

        dataframe.loc[self.years_to_str(2009, 2009 + len(fractions) - 1), ['adjusted_removals']] = dataframe.loc[self.years_to_str(2009, 2009 + len(fractions) -1 ), ['removal']].values.flatten() * fraction_retained

        for y_ in list(range(2009, 2009 + len(fractions))):
            if y_ == 2009: 
                dataframe.loc[str(y_), ['old_stk_retain']] = dataframe.loc[str(y_), ['adjusted_removals']]
            else: 
                dataframe.loc[str(y_), ['old_stk_retain']] = dataframe.loc[str(y_ - 1), ['old_stk_retain']].values + dataframe.loc[str(y_), ['adjusted_removals']].values

        dataframe['adjusted_removals'] = dataframe['adjusted_removals'].fillna(0)
        dataframe.loc[self.years_to_str(1960, current_year), ['floorspace_bsf']] = dataframe.loc[self.years_to_str(1960, current_year), ['scaled_additions_estimate_c']].values - dataframe.loc[self.years_to_str(1960, current_year), ['adjusted_removals']].values

        return dataframe[['floorspace_bsf']].dropna()
        
    def solve_logistic(self, dataframe):
        """Solve NES logistic parameters
        """    
        pnnl_coefficients = [3.92276415015621, 73.2238120168849]  # [gamma, lifetime]
        # popt, pcov = curve_fit(self.nems_logistic, xdata=dataframe[], ydata=dataframe[] , p0=pnnl_coefficients)
        # return popt 
        return pnnl_coefficients

    def activity(self):
        """Use logistic parameters to find predicted historical floorspace
        """ 
        dodge_to_cbecs = self.dodge_to_cbecs() # columns c-m starting with year 1920 (row 17)
        coeffs = self.solve_logistic(dodge_to_cbecs)
        historical_floorspace_late = self.nems_logistic(dodge_to_cbecs, coeffs)  # properly formatted?

        historical_floorspace_early = {1949: 27235.1487296062, 1950: 27788.6370796569, 1951: 28246.642791733, 1952: 28701.4989706012, 
                                       1953: 29253.2282427217, 1954: 29913.8330998026, 1955: 30679.7157232176, 1956: 31512.6191323126,
                                       1957: 32345.382764321, 1958: 33206.8483392728, 1959: 34088.6640247816}
        historical_floorspace_early = pd.DataFrame.from_dict(historical_floorspace_early, columns=['floorspace_bsf'], orient='index')
        historical_floorspace_early.index = historical_floorspace_early.index.astype(str)
    
        historical_floorspace = pd.concat([historical_floorspace_early, historical_floorspace_late])
        historical_floorspace_billion_sq_feet = historical_floorspace.multiply(0.001)
        return historical_floorspace_billion_sq_feet

    def fuel_electricity_consumption(self):
        """Trillion Btu
        """
        year_range = list(range(1949, 1970))
        year_range = [str(y) for y in year_range]
        national_calibration = self.collect_input_data('national_calibration')
        total_primary_energy_consumption = self.eia_comm.eia_api(id_='TOTAL.TXCCBUS.A', id_type='series') # pre 1969: AER table 2.1c update column U 
        total_primary_energy_consumption = total_primary_energy_consumption.rename(columns={'Total Primary Energy Consumed by the Commercial Sector, Annual, Trillion Btu': 'total_primary'})
        # total_primary_energy_consumption = total_primary_energy_consumption[total_primary_energy_consumption.index.isin(year_range)]
        # total_primary_energy_consumption = total_primary_energy_consumption.multiply(0.001)

        fuels_dataframe = total_primary_energy_consumption.copy()
        replacement_data = national_calibration.loc['1970':, ['Final Est. (Trillion Btu)_fuels']]  # >= 1970: National Calibration Column 0
        fuels_dataframe.loc['1970':, ['total_primary']] = replacement_data.values
        fuels_dataframe = fuels_dataframe.rename(columns={'total_primary': 'adjusted_consumption_trillion_btu'})
        elec_dataframe =  self.adjusted_supplier_data() 

        energy_data = {'elec': elec_dataframe, 'fuels': fuels_dataframe}
        return energy_data

    def get_seds(self):
        seds = self.collect_input_data('SEDS_CensusRgn')
        census_regions = {4: 'West', 3: 'South', 2: 'Midwest', 1: 'Northeast'}
        total_fuels = seds[0].rename(columns=census_regions)
        elec = seds[1].rename(columns=census_regions)
        return {'elec': elec, 'fuels': total_fuels}

    def collect_weather(self, comm_activity):
        seds = self.get_seds()
        res = ResidentialIndicators(directory=self.directory, output_directory=self.output_directory, base_year=self.base_year)
        residential_activity_data = res.get_floorspace()
        residential_floorspace = residential_activity_data['floorspace_square_feet']
        weather = WeatherFactors(sector='commercial', directory=self.directory, activity_data=comm_activity, residential_floorspace=residential_floorspace)
        weather_factors = weather.get_weather(seds_data=seds)
        # weather_factors = weather.adjust_for_weather() # What should this return?? (e.g. weather factors or weather adjusted data, both?)
        return weather_factors

    def collect_data(self):
        # Activity: Floorspace_Estimates column U, B
        # Energy: Elec --> Adjusted Supplier Data Column D
        #         Fuels --> AER11 Table 2.1C_Update column U, National Calibration Column O
        activity_data = self.activity()
        print('Activity data collected without issue')
        energy_data = self.fuel_electricity_consumption()
        print('Energy data collected without issue')

        weather_factors = self.collect_weather(comm_activity=activity_data) # need to integrate this into the data passed to LMDI
        print('weather_factors', weather_factors)

        data_dict = {'Commercial_Total': {'energy': energy_data, 'activity': activity_data, 'weather_factors': weather_factors}}
        return data_dict

    def main(self, breakout, save_breakout, calculate_lmdi):
        data_dict = self.collect_data()
        results_dict, formatted_results = self.get_nested_lmdi(level_of_aggregation=self.level_of_aggregation, 
                                                               breakout=breakout, save_breakout=save_breakout, 
                                                               calculate_lmdi=calculate_lmdi, raw_data=data_dict,
                                                               lmdi_type='LMDI-I')
        print(formatted_results)
        formatted_results.to_csv(f"{self.output_directory}/commercial_results2.csv")

        return formatted_results

if __name__ == '__main__':
    indicators = CommercialIndicators(directory='C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020', output_directory='C:/Users/irabidea/Desktop/LMDI_Results', level_of_aggregation='Commercial_Total', lmdi_model=['multiplicative'])
    indicators.main(breakout=False, save_breakout=False, calculate_lmdi=False)





   




#     @staticmethod
#     def floorspace_estimates():
#         previous_year_stock = pd.read_excel('./')
#         new_construction = pd.read_excel('./')  # Floor space reported in million square feet
#         additions_completed_same_year = .4  # Fraction of new construction completed the same year construction began
#         additions_with_lagged_completion = .6  # Fraction of new construction completed the following year'
#         dodge_adjustment = 1.2  # Account for underreporting by Dodge (column AD in spreadsheet)

#         def survival_function():
#             """Non-linear regression model applied to the vintage data from the 1989 and 1999 CBECS
#             """
#             recessional_reductions_level_of_removals = [0] # 30 to 40% 
#             demolitions_before_2008 = 
#             demolitions_after_2008 =             
#             pass

#         pass

#     @staticmethod
#     def reclassification_electricity_sales():
#         """[summary]
#         """


#     def estimate_intensity_indexes_regional(self, parameter_list):
#         """Data Sources: Fuel Consumption and electricity consumption from SEDS, Shares of regional floorspace from CBECs
#         Purpose: used to produce weather adjustment facotrs"""
#                 """[summary]
#         -Regional etsimates of floorspace are derived by applying regional shares to the total national floor space that were 
#         estimatedwithin the spreadsheet historical_floorspace.xlsx
#         -The time series of regional shares are estimated in the worksheet "Regional Shares" (key data for this process are the 
#         shares of regional floor space reported in the various Commercial Building Energy Consumption Surveys (CBECS) or NBECS for 
#         years prior to 1986)
#         -To provide annual estimates of the shares, the assumption was made that commercial floor space in each region would 
#         generally follow the same trends as population or housing units. Here residential estimates of residential housing units
#         were used to reflect these overall trends. 
#         -For each region, a simple regression model was estimated between the regional housing unit share and the regional commmercial
#         building floor space share from the NBECS/CBECS. The regression employed both shares in log form. Based on the estimated 
#         coefficients from this regression, the annual housing unit values are used to predict the share of commercial floor space 
#         in each region. The normalized shares of floorspace by census region (normalized to sum to 1) are contained in Regional_Floorspace
#         columns E through H. 
#         -Regional floor space levels are calculated by multiplying the regional shares times the national estimate of floorspace  (taken 
#         from sheet called Commercial_Total)

#         """
        
#         total_national_floorspace = 

#         X = regional_housing_unit_share
#         Y = regional_commercial_building_floorspace_shares  # from historical CBECS

#         X = sm.add_constant(x)  # Add constant

#         model = sm.OLS(X, Y).fit
#         predictions = model.predict(x)

#         print(model.summary())

#         # reg = linear_model.LinearRegression()
#         # reg.fit(X, Y)
#         # coefficients = reg.coef_
#         # regional_shares_floorspace = dict()  # assumed commercial floorspace in each region follows same trends as population or housing units
#         # regional_floorspace = dict()
#         # regional_intensity_index = dict()
#         # for region in regions: 
#         #     regional_energy_consumption = [0] # consumption of electricity and fuels
#         #     predicted_share = reg.predict(annual_housing_unit_values)
#         #     regional_shares_floorspace[region] = predicted_share
#         #     region_floorspace =  predicted_share * total_national_floorspace
#         #     regional_floorspace[region] = region_floorspace
#         #     regional_intensity_index[region] = regional_energy_consumption / region_floorspace  

#         return None
          
#     def estimate_reclassification_electricity_sector_sales(self,):
#         """Data Source: Commercial electricity sales from EIA SEDS
#         Assumption: The significant changes (same magnitude but opposite directions in the same year) in state-level electricity slaes, typically
#                     showing up in the commercial and industrial sectors reflect reclassification of some customers form the industrial rate class
#                     to the commercial rate class or vice versa. 
#         Strategy: adjust the more recent data by adding or subtracting a constant vlaue, determined from the year in which the reclassification was 
#                   judged to have occured."""
#         state_level_adjustment = 
#         pass


