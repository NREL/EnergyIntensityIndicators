from sklearn import linear_model
import pandas as pd

class WeatherFactors(LMDI):
    def __init__(self, region, energy_type, type, sector):
        self.hdd_by_division = GetEIAData.eia_api(id_='1566347')
        self.cdd_by_division = GetEIAData.eia_api(id_='1566348')
        self.region = region
        self.energy_type = energy_type  # 'electricity' or 'fuels'
        self.type = type  # 'delivered' etc
        self.setor = sector

        """need tables:
        Table 5.2, RECS C&E 1993 (Household Energy Consumption and Expenditures 1993)
        Table 5.14, RECS C&E 1993, calculated from kWh converted to Btu
        Table 5.20, RECS C&E 1993
        Table 5.2, RECS C&E 1993; Major energy sources, column 1.
        Table 5.2, RECS C&E 1993; Major energy sources, column 3.
        Table 5.11, RECS C&E 1993
        Table 5.14, RECS C&E 1993 
        Table 5.20, RECS C&E 1993
        
        EnergyPrices_by_Sector_010820_DBB.xlsx / LMDI-Prices'!EY123"""

    def lmdi_prices():
        pass
    
    def adjust_data(self):
        adjustment_factor_electricity =  # Weights derived from 1995 CBECS
        adjustment_factor_fuels = 
        self.adjusted_hdd = weights * self.hdd_by_division
        self.adjusted_cdd = weights * self.cdd_by_division


    def weather_factors(self):
        """Estimate a simple regression model to fit the regional intensity to a linear function of time (included squared and cubed values of time) and degree days. 
        -electricity model: constant term, heating degree day (HDD), cooling degree day (CDD), time, time-squared, and time-cubed
        -fuels model: contant term?, HDD, HDD*Time, Time, Time-squared and composite fuel price index (the composite fuel price index was developed as a weighted average of the national distillate
            fuel oil price index and a national average price for natural gas)
        Weather factors are applied at the regional level to generate the weather-normalized intensity indexes for each of the four Census regions
        
        -The weather factors for delivered energy and source nergy are computed implicityl. For delivered energy, they are calculated
        as the sum of reported electricity and fuels divided by the sum of the weather-adjusted electricity and weather-adjusted fuels. 
        A similar procedure is followed for source energt. As such, the implied weather factors are a result of the process, not an independent
        variable that influences the values of intensity indexes for delivered energy and source energy. All of these computation occur within Commercial_Total worksheet.

        TODO: Input data and 
        """

        sub_regions_dict = {'northeast': ['New England', 'Middle Atlantic'], 'midwest': ['East North Central', 'West North Central'], 
                    'south': ['South Atlantic', 'East South Central', 'West South Central'], 'west': ['Mountain', 'Pacific']}
        subregions = sub_regions_dict[self.region]
        columns_ = ['Year'] + subregions

        if self.sector == 'residential':
            number_of_households_using_electricity_for_heating_cooling = 
            number_of_households_using_fuels_for_main_space_heating = 
            if self.energy_type == 'electricity':
                factor1 =
                factor2 =   
                cool_factor1 = 
                cool_factor2 =  
            elif self.energy_type == 'fuels':
                factor1 = 
                factor2 = 

        elif self.sector == 'commercial':
            total_floorsplace_using_electricity_for_heating_cooling = 
            number_of_households_using_fuels_for_main_space_heating = 
            if self.energy_type == 'electricity':
                factor1 =
                factor2 =   
                cool_factor1 = 
                cool_factor2 =  
            elif self.energy_type == 'fuels':
                factor1 = 
                factor2 = 


        heating_degree_days = self.hdd_by_division[columns_]
        heating_degree_days[self.region] = heating_degree_days[subregions[0]].multiply(factor1).add(heating_degree_days[subregions[1]].multiply(factor2)) # use pd.dot to account for difference in number subregions
        cooling_degree_days = self.cdd_by_division[columns_]
        cooling_degree_days[self.region] = cooling_degree_days[subregions[0]].multiply(cool_factor1).add(cooling_degree_days[subregions[1]].multiply(cool_factor2) # use pd.dot to account for difference in number subregion

        weather_factors_df = heating_degree_days[['Year', self.region]].rename(columns={self.region: 'HDD'})

        weather_factors_df['Time'] = weather_factors_df['Year'].subract(1969)
        weather_factors_df['Time^2'] = weather_factors_df['Time'].pow(2)

        if self.energy_type == 'electricity': 
            weather_factors_df = weather_factors_df.merge(cooling_degree_days[['Year', self.region]], how=outer, on='Year').rename(columns={self.region: 'CDD'})
            weather_factors_df['Time^3'] = weather_factors_df['Time'].pow(3)
            X = 
        elif self.energy_type == fuels: 
            weather_factors_df['HDD*Time'] = heating_degree_days[self.region].multiply(weather_factors_df['Time'])
            weather_factors_df['Price'] = selected_variable
            X = 

        elif self.type == 'delivered':
            weather_factor = (reported_electricity + fuels) / (weather_adjusted_electrity + weather_adjusted_fuels)
            return weather_factor
        else:
            return None

        actual_intensity =  # from region_intensity (aggregate)
        reg = linear_model.LinearRegression()
        reg.fit(X, Y)
        coefficients = reg.coef_
        predicted_value_intensity_actualdd = reg.predict(X_actualdd)  # Predicted value of the intensity based on actual degree days
        predicted_value_intensity_ltaveragesdd = reg.predict(X_ltaveragesdd)  # Predicted value of the intensity based on the long-term averages of the degree days
        weather_factor = predicted_value_intensity_actualdd / predicted_value_intensity_ltaveragesdd 
        weather_normalized_intensity = actual_intensity / weather_factor
        return weather_factor, weather_normalized_intensity

    def national_method1_fixed_end_use_share_weights():
    
    def national_method2_regression_models(self, moving_average_weights=True, implicit_national_factors=False):
        if self.sector == 'commercial':
            seds_census_region_electricity = 
            seds_census_region_electricity['National'] = seds_census_region_electricity.sum(axis=1)
            weather_adjusted_consumption_electricity = seds_census_region_electricity.multiply(weather_factors_electricity)
            weather_adjusted_consumption_electricity['National'] = weather_adjusted_consumption_electricity.sum(axis=1)
            implicit_national_weather_factor_elec = seds_census_region_electricity['National'].divide(weather_adjusted_consumption_electricity['National'])

            seds_census_region_fuels = 
            seds_census_region_fuels['National'] = seds_census_region_fuels.sum(axis=1)
            weather_adjusted_consumption_fuels = seds_census_region_fuels.multiply(weather_factors_fuels)
            weather_adjusted_consumption_fuels['National'] = weather_adjusted_consumption_fuels.sum(axis=1)
            implicit_national_weather_factor_fuels = seds_census_region_fuels['National'].divide(weather_adjusted_consumption_fuels['National'])

        if moving_average_weights:

        if implicit_national_factors:




"""RESIDENTIAL NOTE: The methodology for the est_imation of “weather factors” is the same as the one set out at the beginning
of this summary write-up with one addition/modification. The weather factors are applied at the
regional level to generate the weather-normalized intensity indexes for each of the four census regions."""