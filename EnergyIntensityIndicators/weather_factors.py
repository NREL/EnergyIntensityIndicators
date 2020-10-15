from sklearn import linear_model
import pandas as pd
from pull_eia_api import GetEIAData
from Residential.residential_floorspace import ResidentialFloorspace
import numpy as np
from functools import reduce
from sklearn.linear_model import LinearRegression
import math
import os


class WeatherFactors: 
    def __init__(self, sector, directory, activity_data=None, residential_floorspace=None, nominal_energy_intensity=None, end_year=2018):
        self.end_year = end_year
        self.directory = directory
        self.sector = sector
        self.activity_data = activity_data
        self.nominal_energy_intensity = nominal_energy_intensity
        self.residential_floorspace = residential_floorspace
        self.eia_data = GetEIAData(self.sector)
        self.lmdi_prices = pd.read_excel(f'{self.directory}/EnergyPrices_by_Sector_010820_DBB.xlsx', sheet_name='LMDI-Prices', header=14, usecols='A:B, EY')
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

    @staticmethod
    def adjust_data(subregions, hdd_by_division, hdd_activity_weights, cooling=True, cdd_by_division=None, cdd_activity_weights=None, use_weights_1961_90=True):
        """Calculate weights for adjusted weather factors prediction

        Args:
            subregions ([type]): [description]
            hdd_by_division ([type]): [description]
            cdd_by_division ([type]): [description]

        Returns:
            [type]: [description]
        """        

        years_1961_90 = list(range(1961, 1990 + 1))
        years_1981_2010 = list(range(1981, 1990 + 1))

        if cooling:
            cdd_by_division = cdd_by_division.set_index('Year')
            cdd_by_division.index = cdd_by_division.index.astype(int)

            averages_1961_90_cooling = cdd_by_division.loc[years_1961_90, :].mean(axis=0)
            averages_1981_2010_cooling = cdd_by_division.loc[years_1981_2010, :].mean(axis=0)


        hdd_by_division = hdd_by_division.set_index('Year')
        hdd_by_division.index = hdd_by_division.index.astype(int)

        averages_1961_90_heating = hdd_by_division.loc[years_1961_90, :].mean(axis=0)
        averages_1981_2010_heating = hdd_by_division.loc[years_1981_2010, :].mean(axis=0)
        
        all_s_weights_heating = []
        all_s_weights_cooling = []

        for s in subregions:
            if use_weights_1961_90:
                subregion_weights_heating = averages_1961_90_heating.loc[s] * hdd_activity_weights[s]

                if cooling:
                    subregion_weights_cooling = averages_1961_90_cooling.loc[s] * cdd_activity_weights[s]
                    all_s_weights_cooling.append(subregion_weights_cooling)

            else:
                subregion_weights_heating = averages_1981_2010_heating.loc[s] * hdd_activity_weights[s]

                if cooling:
                    subregion_weights_cooling = averages_1981_2010_cooling.loc[s] * cdd_activity_weights[s]
                    all_s_weights_cooling.append(subregion_weights_cooling)

            
            all_s_weights_heating.append(subregion_weights_heating)

        weights_dict = dict()
        if cooling:
            weights_cooling = sum(all_s_weights_cooling)
            weights_dict['cooling'] = weights_cooling

        weights_heating = sum(all_s_weights_heating)
        weights_dict['heating'] = weights_heating
        return weights_dict

    def process_prices(self, weather_factors_df):
        lmdi_prices = self.lmdi_prices
        # distributed_lag = 
        # time_cubed = 
        selected_variable = [1] * len(weather_factors_df)
        return selected_variable
    
    @staticmethod
    def cbecs_1995_shares():
        electricty_consumption_tbtu = {'Northeast': 436, 'Midwest': 558, 'South': 1027, 'West': 587}
        energy_tbtu = [1035, 1497, 1684, 1106] 
        energy_tbtu.append(sum(energy_tbtu))
        electricty_consumption_tbtu['Total'] = sum(electricty_consumption_tbtu.values())
        shares_df = pd.DataFrame.from_dict(electricty_consumption_tbtu, orient='index', columns=['electricity_consumption_tbtu'])
        shares_df['elec_share'] = shares_df.electricity_consumption_tbtu.divide(shares_df.loc['Total', 'electricity_consumption_tbtu'])
        shares_df['energy'] = energy_tbtu
        shares_df['fuel_consumption'] = shares_df.energy.subtract(shares_df.electricity_consumption_tbtu)
        shares_df['fuel_share'] = shares_df.fuel_consumption.divide(shares_df.loc['Total', 'fuel_consumption'])
        return shares_df

    @staticmethod 
    def recs_1993_shares():
        """Need to fill this in

        Returns:
            [type]: [description]
        """        
        shares_df = None
        return shares_df

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
            for r_, subregions in self.sub_regions_dict.items():
                subregions = [s.lower().replace(' ', '_') for s in subregions]
                regions_ = subregions + [r_]
                region_total = dataframe.loc[r_, col]
                for r in regions_:
                    share_value = dataframe.loc[r, col] / region_total
                    shares_dict[r] = share_value
            weights_data[col] = shares_dict
        return weights_data

    def gather_weights_data(self):
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

        elif self.sector == 'commercial':
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
        else:
            return None
        
        weights_data_ = {'regions_subregions': self.regions_subregions, 'heating_activity': heating_activity, 
                        'cooling_activity': cooling_activity, 'all_energy': all_energy, 'electricity': electricity}
        
        weights_df = pd.DataFrame(data=weights_data_)
        
        weights_df['fuels'] = weights_df['all_energy'].subtract(weights_df['electricity'])
        return weights_df
    
    def heating_cooling_data(self):
        hdd_by_division_historical = pd.read_csv('./Data/historical_hdd_census_division.csv').set_index('Year')
        cdd_by_division_historical = pd.read_csv('./Data/historical_cdd_census_division.csv').set_index('Year')

        hdd_by_division = self.eia_data.eia_api(id_='1566347', id_type='category')
        hdd_to_drop = [c for c in list(hdd_by_division.columns) if 'Monthly' in c]
        hdd_by_division = hdd_by_division.drop(hdd_to_drop, axis=1)
        hdd_rename_dict = {c: c.replace(', Annual, Number', '') for c in list(hdd_by_division.columns)}
        hdd_by_division = hdd_by_division.rename(columns=hdd_rename_dict)


        hdd_by_division = pd.concat([hdd_by_division_historical, hdd_by_division], sort=True)
        
        cdd_by_division = self.eia_data.eia_api(id_='1566348', id_type='category')
        cdd_to_drop = [c for c in list(cdd_by_division.columns) if 'Monthly' in c]
        cdd_by_division = cdd_by_division.drop(cdd_to_drop, axis=1)
        cdd_rename_dict = {c: c.replace(', Annual, Number', '') for c in list(cdd_by_division.columns)}

        cdd_by_division = cdd_by_division.rename(columns=cdd_rename_dict)

        cdd_by_division = pd.concat([cdd_by_division_historical, cdd_by_division], sort=True)


        title_case_regions = [s.replace('_', ' ').title() for s in self.regions_subregions]
        hdd_names = [f'Heating Degree-Days, {r}' for r in title_case_regions]
        cdd_names = [f'Cooling Degree-Days, {r}' for r in title_case_regions]

        hdd_new_names_dict = {name: name_title for name, name_title in zip(hdd_names, title_case_regions)}
        cdd_new_names_dict = {name: name_title for name, name_title in zip(cdd_names, title_case_regions)}

        hdd_by_division = hdd_by_division.rename(columns=hdd_new_names_dict)
        cdd_by_division = cdd_by_division.rename(columns=cdd_new_names_dict)

        return hdd_by_division, cdd_by_division

    def estimate_regional_shares(self):
        """Spreadsheet equivalent: Commercial --> 'Regional Shares' 
        assumed commercial floorspace in each region follows same trends as population or housing units"""
        regions = ['Northeast', 'Midwest', 'South', 'West']

        cbecs_data = pd.read_csv('./Data/cbecs_data_millionsf.csv').set_index('Year')
        cbecs_data.index = cbecs_data.index.astype(str)
        cbecs_years = list(cbecs_data.index)
        cbecs_data = cbecs_data.rename(columns={'Midwest ': 'Midwest', ' South': 'South', ' West': 'West'})

        cbecs_data.loc['1979', regions] = cbecs_data.loc['1983', regions].subtract([826, 972, 2665, 1212])
        cbecs_data.loc['1979', ['U.S.']] = sum(cbecs_data.loc['1979', regions].values)

        cbecs_data['U.S. (calc)'] = cbecs_data.sum(axis=1)

        comm_regional_shares = cbecs_data.drop(['U.S.', 'U.S. (calc)'], axis=1).divide(cbecs_data['U.S. (calc)'].values.reshape(len(cbecs_data), 1))
        comm_regional_shares_ln = np.log(comm_regional_shares)

        residential_data = ResidentialFloorspace(end_year=self.end_year)  # change to pull from residential().activity()
        final_results_total_floorspace_regions, regional_estimates_all, avg_size_all_regions = residential_data.final_floorspace_estimates()
        
        regional_dfs = [regional_estimates_all[r][['Total']].rename(columns={'Total': r}) for r in regions]
        residential_housing_units = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'), regional_dfs)
        residential_housing_units['U.S.'] = residential_housing_units.sum(axis=1)
        residential_housing_units.index = residential_housing_units.index.astype(str)
        regional_shares_residential_housing_units = residential_housing_units.drop('U.S.', axis=1).divide(residential_housing_units['U.S.'].values.reshape(len(residential_housing_units), 1))
        regional_shares_residential_housing_units_ln = np.log(regional_shares_residential_housing_units)

        regional_shares_residential_housing_units_cbecs_years = regional_shares_residential_housing_units.loc[cbecs_years, :]
        regional_shares_residential_housing_units_cbecs_years_ln = np.log(regional_shares_residential_housing_units_cbecs_years)
        
        predictions_df = pd.DataFrame(columns=comm_regional_shares.columns, index=residential_housing_units.index)
        for region in comm_regional_shares.columns:
            x_values = comm_regional_shares_ln[region].values
            X = x_values.transpose()
            y = regional_shares_residential_housing_units_cbecs_years_ln[region].values

            # reg = LinearRegression().fit(X, y)
            # prediction = reg.predict(regional_shares_residential_housing_units_ln[region])
            # predictions_df[region] = prediction

            p = np.polyfit(X, y, 1)
            predictions_df[region] = np.exp(regional_shares_residential_housing_units_ln[region].multiply(p[0]).add(p[1]))

        predictions_df['Predicted Sum'] = predictions_df.sum(axis=1)
        normalized_shares = predictions_df.drop('Predicted Sum', axis=1).divide(predictions_df['Predicted Sum'].values.reshape(len(predictions_df), 1))
        return normalized_shares
    
    def commercial_estimate_regional_floorspace(self):
        regional_shares = self.estimate_regional_shares()
        commercial_floorspace = self.activity_data 

        regional_shares_index = regional_shares.index.astype(str)
        commercial_floorspace_reshape = commercial_floorspace.loc[regional_shares_index, :]

        regional_floorspace = regional_shares.multiply(commercial_floorspace_reshape.values)
        return regional_floorspace

    def commercial_regional_intensity_aggregate(self):
        """Calculate Energy Intensities (kBtu/sq. ft.) by region and fuel type (i.e. Fuels and Electricity) for use
        in calculating weather factors
        Returns:
            dictionary with keys: 'electricity' and 'fuels', values: dataframes of intensity data for the commercial sector
            with Year index and Region columns
        """        
        regional_floorspace = self.commercial_estimate_regional_floorspace()
        total_fuels_to_indicators, elec_to_indicators = self.eia_data.get_seds()
        
        regional_floorspace_index = regional_floorspace.index
        elec_to_indicators =  elec_to_indicators.loc[regional_floorspace_index, :]
        total_fuels_to_indicators =  total_fuels_to_indicators.loc[regional_floorspace_index, :]

        print('total_fuels_to_indicators, elec_to_indicators:', total_fuels_to_indicators, elec_to_indicators)

        fuels_regional = regional_floorspace.multiply(total_fuels_to_indicators.drop('National', axis=1).values)
        elec_regional = regional_floorspace.multiply(elec_to_indicators.drop('National', axis=1).values)

        return {'fuels': fuels_regional, 'electricity': elec_regional}
    
    def residential_regional_intensity_aggregate(self):
        """This function does not need to exist if nominal_energy_intensity is properly formated, change formatting here if not
        Returns:
            dictionary with keys: 'electricity' and 'fuels', values: dataframes of intensity data for the residential sector
            with Year index and Region columns
            i.e. {'fuels': fuels_regional, 'electricity': elec_regional}
        """        

        nominal_energy_intensity = self.nominal_energy_intensity # nominal_energy_intensity should already be formated in this way 

        return nominal_energy_intensity

    def weather_factors(self, region, energy_type, actual_intensity):
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
        weights_df = self.gather_weights_data()
        print('WEIGHTS DF: \n', weights_df)
        regional_weights = self.regional_shares(dataframe=weights_df, cols=['heating_activity', 'cooling_activity', 'fuels'])
        print('REGIONAL WEIGHTS: \n', regional_weights)

        subregions = self.sub_regions_dict[region]
        subregions_lower = [s.lower().replace(' ', '_') for s in subregions]
        hdd_activity_weights = [regional_weights['heating_activity'][r_] for r_ in subregions_lower]
        hdd_activity_weights_dict = {r : regional_weights['heating_activity'][r_] for r, r_ in zip(subregions, subregions_lower)}
        cdd_activity_weights = [regional_weights['cooling_activity'][r_] for r_ in subregions_lower]
        cdd_activity_weights_dict = {r : regional_weights['cooling_activity'][r_] for r, r_ in zip(subregions, subregions_lower)}
        fuels_weights = [regional_weights['fuels'][r_] for r_ in subregions_lower]
        
        hdd_by_division, cdd_by_division = self.heating_cooling_data()

        heating_degree_days = hdd_by_division[subregions]

        heating_degree_days = heating_degree_days.reset_index('Year')

        heating_degree_days[region] = heating_degree_days[subregions].dot(hdd_activity_weights)
        print('heating_degree_days: \n', heating_degree_days)

        fuels_heating_degree_days = heating_degree_days
        fuels_heating_degree_days[region] = fuels_heating_degree_days[subregions].dot(fuels_weights)
        print('fuels_heating_degree_days: \n', fuels_heating_degree_days)

        weather_factors_df = heating_degree_days[['Year', region]].rename(columns={region: 'HDD'})
        weather_factors_df['Year'] = weather_factors_df['Year'].astype(int)

        weather_factors_df['Time'] = weather_factors_df['Year'].values - 1969
        weather_factors_df['Time^2'] = weather_factors_df[['Time']].pow(2).values

        if energy_type == 'electricity': 
            cooling_degree_days = cdd_by_division[subregions]
            cooling_degree_days[region] = cooling_degree_days[subregions].dot(cdd_activity_weights)
            cooling_degree_days = cooling_degree_days.reset_index('Year')
            cooling_degree_days['Year'] = cooling_degree_days['Year'].astype(int)

            weather_factors_df_cooling = cooling_degree_days[['Year', region]].rename(columns={region: 'CDD'})
            weather_factors_df = weather_factors_df.merge(weather_factors_df_cooling, on='Year', how='outer')

            weather_factors_df['Time^3'] = weather_factors_df[['Time']].pow(3).values
            weather_factors_df = weather_factors_df.set_index('Year')
            weather_factors_df.index = weather_factors_df.index.astype(int)
            
            X_data = weather_factors_df[['HDD', 'CDD', 'Time', 'Time^2', 'Time^3']]

        elif energy_type == 'fuels': 
            weather_factors_df['HDD*Time'] = heating_degree_days[region].multiply(weather_factors_df['Time'])
            weather_factors_df['Price'] = self.process_prices(weather_factors_df)
            weather_factors_df = weather_factors_df.set_index('Year')
            weather_factors_df.index = weather_factors_df.index.astype(int)
            X_data = weather_factors_df[['HDD', 'HDD*Time', 'Time', 'Time^2', 'Price']]

        # elif self.energy_type == 'delivered':
        #     weather_factor = (reported_electricity + fuels) / (weather_adjusted_electrity + weather_adjusted_fuels)
        #     return weather_factor
        else:
            return None

        actual_intensity.index = actual_intensity.index.astype(int)  
        data = X_data.merge(actual_intensity, left_index=True, right_index=True, how='inner').dropna()
        X = data.drop(region.capitalize(), axis=1)
        Y = data[[region.capitalize()]]

        reg = linear_model.LinearRegression()
        reg.fit(X, Y)
        coefficients = reg.coef_
        coefficients = coefficients[0]
        print(f'{energy_type} coefficient for region {region}:', coefficients)
        intercept = reg.intercept_
        print(f'{energy_type} intercept for region {region}:', intercept)
        predicted_value_intensity_actualdd = reg.predict(X)  # Predicted value of the intensity based on actual degree days

        if  energy_type == 'electricity': 
            prediction2_weights = self.adjust_data(subregions=subregions, hdd_by_division=heating_degree_days, cdd_by_division=cooling_degree_days, 
                                                   cdd_activity_weights=cdd_activity_weights_dict, hdd_activity_weights=hdd_activity_weights_dict,
                                                   use_weights_1961_90=True)
            predicted_value_intensity_ltaveragesdd = intercept + coefficients[0] * prediction2_weights['heating'] + coefficients[1] * prediction2_weights['cooling'] + \
                                                    coefficients[2] * data['Time'] + coefficients[3] * data['Time^2'] + coefficients[4] * data['Time^3']  # Predicted value of the intensity based on the long-term averages of the degree days
        
        elif energy_type == 'fuels': 
            prediction2_weights = self.adjust_data(subregions=subregions, hdd_by_division=heating_degree_days, 
                                                   hdd_activity_weights=hdd_activity_weights_dict, cooling=False,
                                                   use_weights_1961_90=True)
            predicted_value_intensity_ltaveragesdd = intercept + coefficients[0] * prediction2_weights['heating'] + coefficients[1] * data['Time'] + \
                                                     coefficients[2] * data['Time'] + coefficients[3] * data['Time^2'] + coefficients[4] * data['Price'] # Predicted value of the intensity based on the long-term averages of the degree days

        print('predicted_value_intensity_actualdd: \n', predicted_value_intensity_actualdd)
        print('predicted_value_intensity_ltaveragesdd: \n', predicted_value_intensity_ltaveragesdd)

        weather_factor = predicted_value_intensity_actualdd.flatten() / predicted_value_intensity_ltaveragesdd.values.flatten()
        print('weather factor here: \n', weather_factor)
        print('actual_intensity here: \n', actual_intensity)

        weather_normalized_intensity = actual_intensity.loc[data.index] / weather_factor
        weather_factor_df = pd.DataFrame(data={'Year': data.index, f'{region}_weather_factor': weather_factor}).set_index('Year')
        print('weather_factor_df', weather_factor_df)
        return weather_factor_df, weather_normalized_intensity
    
    def national_method1_fixed_end_use_share_weights(self):
        """Used fixed weights to develop from regional factors, weighted by regional energy share from 1995 CBECS
        """
        if self.sector == 'commercial':
            shares = self.cbecs_1995_shares()
            regional_intensity_dict = self.commercial_regional_intensity_aggregate()

        elif self.sector == 'residential':
            regional_intensity_dict = self.residential_regional_intensity_aggregate()
            shares = self.recs_1993_shares()
        
        fuel_type_weather_factors = dict()
        
        for energy_type in ['electricity', 'fuels']:
            intensity_df = regional_intensity_dict[energy_type]
            intensity_df = intensity_df.reindex(columns=list(intensity_df.columns) + ['final_electricity_factor', 'final_fuels_factor'])
            print('intensity_df: \n', intensity_df)

            regional_weather_factors = []

            for region in self.sub_regions_dict.keys():
                region_cap = region.capitalize()
                regional_intensity = intensity_df[region_cap]
                weather_factors, weather_normalized_intensity = self.weather_factors(region, energy_type, actual_intensity=regional_intensity)
                regional_weather_factors.append(weather_factors)

            weather_factors_all = pd.concat(regional_weather_factors, axis=1)
            for y in weather_factors_all.index:
                if energy_type == 'electricity':
                    share_name = 'elec_share'
                else:
                    share_name = 'fuel_share'
                year_weather = weather_factors_all.loc[y, :]
                print('year_weather: \n', year_weather)
                weights = shares[share_name].drop('Total')
                print('weights: \n', weights)
                print('weights numpy: \n', weights.to_numpy())

                year_factor = year_weather.dot(weights.to_numpy())
                intensity_df.loc[y, f'final_{energy_type}_factor'] = year_factor

        return intensity_df[['final_electricity_factor', 'final_fuels_factor']]
        
    def national_method2_regression_models(self, moving_average_weights=True, implicit_national_factors=False):
        if self.sector == 'commercial':
            seds_census_region_electricity = []
            seds_census_region_electricity['National'] = seds_census_region_electricity.sum(axis=1)
            weather_adjusted_consumption_electricity = seds_census_region_electricity.multiply(weather_factors_electricity)
            weather_adjusted_consumption_electricity['National'] = weather_adjusted_consumption_electricity.sum(axis=1)
            implicit_national_weather_factor_elec = seds_census_region_electricity['National'].divide(weather_adjusted_consumption_electricity['National'])

            seds_census_region_fuels = [] 
            seds_census_region_fuels['National'] = seds_census_region_fuels.sum(axis=1)
            weather_adjusted_consumption_fuels = seds_census_region_fuels.multiply(weather_factors_fuels)
            weather_adjusted_consumption_fuels['National'] = weather_adjusted_consumption_fuels.sum(axis=1)
            implicit_national_weather_factor_fuels = seds_census_region_fuels['National'].divide(weather_adjusted_consumption_fuels['National'])

        # if moving_average_weights:

        # if implicit_national_factors:

        return None
    
    def adjust_for_weather(self, data, energy_type):
        """purpose
            Parameters
            ----------
            data: dataframe
                dataset to adjust by weather
            weather_factors: array?
                description
            Returns
            -------
            weather_adjusted_data: dataframe ? 
        """
        weather = WeatherFactors(energy_type, sector=self.sector, directory=self.directory)
        weather_factors = weather.national_method1_fixed_end_use_share_weights()
        weather_adjusted_data = data / weather_factors[energy_type]
        return weather_adjusted_data

    def main():
        if weather_adjust: 
            for type, energy_dataframe in energy_data_by_type.items():
                weather_adj_energy = self.adjust_for_weather(energy_dataframe, type) 
                energy_data_by_type[f'{type}_weather_adj'] = weather_adj_energy
        return

# if __name__ == '__main__':
#     weather = WeatherFactors(sector='commercial', directory='C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020', activity_data=)
#     weather.national_method1_fixed_end_use_share_weights()



"""RESIDENTIAL NOTE: The methodology for the est_imation of “weather factors” is the same as the one set out at the beginning
of this summary write-up with one addition/modification. The weather factors are applied at the
regional level to generate the weather-normalized intensity indexes for each of the four census regions."""


# comm = WeatherFactors('commercial', directory='C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020')
# x = comm.heating_cooling_data()
# # x = comm.cbecs_1995_shares() \\ works
# # x = comm.national_method1_fixed_end_use_share_weights()
# # x = comm.national_method2_regression_models
# x = comm.weather_factors('northeast')
# # x = comm.gather_weights_data() \\works
# # x = comm.hdd_by_division.columns
# print('here:', x)