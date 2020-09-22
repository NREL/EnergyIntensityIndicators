from sklearn import linear_model
import pandas as pd

class WeatherFactors(LMDI):
    def __init__(self, region, energy_type, sector):
        self.hdd_by_division = GetEIAData.eia_api(id_='1566347')
        self.cdd_by_division = GetEIAData.eia_api(id_='1566348')
        self.region = region
        self.energy_type = energy_type  # 'electricity' or 'fuels' or 'delivered'
        self.sector = sector
        self.lmdi_prices = pd.read_excel('./EnergyPrices_by_Sector_010820_DBB', sheet_name='LMDI-Prices', header=14, usecols='A:B, EY')
        self.regions_subregions = ['northeast', 'new_england', 'middle_atlantic', 'midwest', 'east_north_central', 'west_north_central', 
                        'south', 'south_atlantic', 'east_south_central', 'west_south_central', 'west', 'mountain', 'pacific']
        self.sub_regions_dict = {'northeast': ['New England', 'Middle Atlantic'], 'midwest': ['East North Central', 'West North Central'], 
                                 'south': ['South Atlantic', 'East South Central', 'West South Central'], 'west': ['Mountain', 'Pacific']}
        
        """
        Table 5.2, RECS C&E 1993 (Household Energy Consumption and Expenditures 1993)
        Table 5.14, RECS C&E 1993, calculated from kWh converted to Btu
        Table 5.20, RECS C&E 1993
        Table 5.2, RECS C&E 1993; Major energy sources, column 1.
        Table 5.2, RECS C&E 1993; Major energy sources, column 3.
        Table 5.11, RECS C&E 1993
        Table 5.14, RECS C&E 1993 
        Table 5.20, RECS C&E 1993
        
        EnergyPrices_by_Sector_010820_DBB.xlsx / LMDI-Prices'!EY123"""
    
    def adjust_data(self):
        adjustment_factor_electricity =  # Weights derived from 1995 CBECS
        adjustment_factor_fuels = 
        self.adjusted_hdd = weights * self.hdd_by_division
        self.adjusted_cdd = weights * self.cdd_by_division

    def collect_data(self):
        """Create dataframe of electricity and fuel data
        """
        if self.sector == 'residential':
            residential_electricity = pd.DataFrame(self.residential_data_electricity) 
            residential_fuels = pd.DataFrame(self.residential_data_fuels)  
            return residential_electricity, residential_fuels
        elif self.sector == 'commercial':
            commercial_electricity = pd.DataFrame(self.commercial_data_electricity)
            commercial_fuels = pd.DataFrame(self.commercial_data_fuels)     
            return commercial_electricity, commercial_fuels

    def process_prices(self):
        lmdi_prices = self.lmdi_prices
        distributed_lag = 
        time_cubed = 
        selected_variable = 

    def regional_shares(self, dataframe, cols):
        """Calulate shares of regional totals by subregion

        Args:
            dataframe ([type]): [description]

        Returns:
            [type]: [description]
        """        
        dataframe = dataframe.set_index('regions_subregions')
        weights_data = dict()
        for col in cols: 
            shares_dict = dict()
            for region, subregions in self.sub_regions_dict.items():
                regions = subregions.append(region)
                region_total = dataframe.loc[region, col]
                for r in regions:
                    share_value = dataframe.loc[r, col].divide(region_total)
                    shares_dict[r] = share_value
            weights_data[col] = shares_dict
        return weights_data

    def weights_data(self):
        """Calculate weights to aggregate subregions into four regions
        """        
        if self.sector == 'residential':
            electricity_data = {'total_elec_tbtu': {'northeast': 470, 'midwest': 740, 'south': 1510,
                                    'west': 560}, 'heating_tbtu': {'northeast': 12 * 3.412, 'midwest': 22 * 3.412, 'south': 61 * 3.412,
                                    'west': 25 * 3.412}, 'cooling_tbtu': {'northeast': 40, 'midwest': 80, 'south': 310,
                                    'west': 30}}
            fuels_data = {'all_energy_tbtu': {'northeast': 2380, 'midwest': 3130, 'south': 2950,
                                    'west': 1550}, 'electricity_tbtu': {'northeast': 470, 'midwest': 740, 'south': 1510,
                                    'west': 560}, 'heating_all_energy_tbtu': {'northeast': 1490, 'midwest': 1920, 'south': 1210,
                                    'west': 700}}
            # Residential Heating Households Millions
            heating_activity = [4.1, 1, 3.1, 5.8, 3.5, 2.4, 18.8, 10.7, 3.4, 4.8, 8.3, 2, 6.3]
            # Residential Cooling Households Millions
            cooling_activity = [10.9, 2.1, 8.8, 16.4, 10.8, 5.6, 29.4, 15, 5.3, 9.2, 7.1, 2.1, 5.1]             
            all_energy = [19.1, 4.9, 14.2, 23.2, 16.3, 6.9, 32.8, 16.8, 5.9, 10.1, 19.4, 5.3, 14.1]
            electricity = [1.9, 0.5, 1.4, 2.9, 1.6, 1.3, 14.6, 8.7, 2.5, 3.4, 5.6, 1.4, 4.2]

        elif self.sector == 'commmercial':
            electricity_data = {'total_elec_tbtu': {'northeast': 436, 'midwest': 558, 'south': 1027,
                                    'west': 587}, 'heating_tbtu': {'northeast': 18, 'midwest': 23, 'south': 43,
                                    'west': 28}, 'cooling_tbtu': {'northeast': 44, 'midwest': 60, 'south': 172,
                                    'west': 64}}
            fuels_data = {'all_energy_tbtu': {'northeast': 1035, 'midwest': 1497, 'south': 1684,
                                    'west': 1106}, 'electricity_tbtu': {'northeast': 436, 'midwest': 558, 'south': 1027,
                                    'west': 587}, 'heating_all_energy_tbtu': {'northeast': 385, 'midwest': 668, 'south': 376,
                                    'west': 275}}
            # Commercial Heating Floorspace Million SF
            heating_activity = [657, 137, 520, 779, 345, 434, 3189, 1648, 1140, 401, 1219, 469, 750]
            # Commercial Cooling Floorspace Million SF
            cooling_activity = [5919, 1472, 4447, 10860, 7301, 3559, 13666, 6512, 3265, 3889, 7058, 2812, 4246]
            all_energy = [7661, 2031, 5630, 10860, 7301, 3559, 13666, 6512, 3265, 3889, 7065, 2819, 4246]
            electricity = [657, 137, 520, 779, 345, 434, 3189, 1648, 1140, 401, 1219, 469, 750]
        
        weights_col_names = ['regions_subregions', 'heating_activity', 'cooling_activity', 'all_energy', 'electricity']

        weights_data = pd.DataFrame([self.regions_subregions, heating_activity, cooling_activity, all_energy, 
                                    electricity]).transpose().columns(weights_col_names)
        weights_data['fuels'] = weights_data['all_energy'].subtract(weights_data['electricity'])
        regional_weights = self.regional_shares(dataframe=weights_data, cols=['heating_activity', 'cooling_activity', 'fuels'])
        return regional_weights


    def weather_factors(self):
        """Estimate a simple regression model to fit the regional intensity to a linear function of time (included squared and cubed values of time) and degree days. 
        -electricity model: constant term, heating degree day (HDD), cooling degree day (CDD), time, time-squared, and time-cubed
        -fuels model: contant term?, HDD, HDD*Time, Time, Time-squared and composite fuel price index (the composite fuel price index was developed as a weighted average of the national distillate
            fuel oil price index and a national average price for natural gas)
        Weather factors are applied at the regional level to generate the weather-normalized intensity indexes for each of the four Census regions
        
        -The weather factors for delivered energy and source energy are computed implicitly. For delivered energy, they are calculated
        as the sum of reported electricity and fuels divided by the sum of the weather-adjusted electricity and weather-adjusted fuels. 
        A similar procedure is followed for source energt. As such, the implied weather factors are a result of the process, not an independent
        variable that influences the values of intensity indexes for delivered energy and source energy. All of these computation occur within Commercial_Total worksheet.

        TODO: Input data 
        """
        regional_weights = self.weights_data()

        subregions = sub_regions_dict[self.region]
        hdd_activity_weights = [regional_weights['heating_activity'][r_] for r_ in subregions]
        cdd_activity_weights = [regional_weights['cooling_activity'][r_] for r_ in subregions]
        fuels_weights = [regional_weights['fuels'][r_] for r_ in subregions]
        columns_ = ['Year'] + subregions

        heating_degree_days = self.hdd_by_division[columns_]
        heating_degree_days[self.region] = heating_degree_days[subregions].dot(hdd_activity_weights)

        cooling_degree_days = self.cdd_by_division[columns_]
        cooling_degree_days[self.region] = cooling_degree_days[subregions].dot(cdd_activity_weights)

        fuels_heating_degree_days = heating_degree_days
        fuels_heating_degree_days[self.region] = fuels_heating_degree_days[subregions].dot(fuels_weights)

        weather_factors_df = heating_degree_days[['Year', self.region]].rename(columns={self.region: 'HDD'})

        weather_factors_df['Time'] = weather_factors_df['Year'].subract(1969)
        weather_factors_df['Time^2'] = weather_factors_df['Time'].pow(2)

        if self.energy_type == 'electricity': 
            weather_factors_df = weather_factors_df.merge(cooling_degree_days[['Year', self.region]], how=outer, on='Year').rename(columns={self.region: 'CDD'})
            weather_factors_df['Time^3'] = weather_factors_df['Time'].pow(3)
            X = weather_factors_df[['HDD', 'CDD', 'Time', 'Time^2', 'Time^3']]
        elif self.energy_type == fuels: 
            weather_factors_df['HDD*Time'] = heating_degree_days[self.region].multiply(weather_factors_df['Time'])
            weather_factors_df['Price'] = selected_variable
            X = weather_factors_df[['HDD', 'HDD*Time', 'Time', 'Time^2', 'Price']]

        elif self.energy_type == 'delivered':
            weather_factor = (reported_electricity + fuels) / (weather_adjusted_electrity + weather_adjusted_fuels)
            return weather_factor
        else:
            return None

        actual_intensity =  # from region_intensity (aggregate): just seds_census_rgn / regional_floorspace
        reg = linear_model.LinearRegression()
        reg.fit(X, Y)
        coefficients = reg.coef_
        predicted_value_intensity_actualdd = reg.predict(X_actualdd)  # Predicted value of the intensity based on actual degree days
        predicted_value_intensity_ltaveragesdd = reg.predict(X_ltaveragesdd)  # Predicted value of the intensity based on the long-term averages of the degree days
        weather_factor = predicted_value_intensity_actualdd / predicted_value_intensity_ltaveragesdd 
        weather_normalized_intensity = actual_intensity / weather_factor
        return weather_factor, weather_normalized_intensity
    
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