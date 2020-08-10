import pandas as pd
from sklearn import linear_model
from .weather_factors import weather_factors
import math
import statsmodels.api as sm
from commercial.commercial import GetCommercialData


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



class CommercialIndicators(LMDI):
    
    def __init__(self, energy_data, activity_data, categories_list):
        super().__init__(energy_data, activity_data, categories_list)
        self.sub_categories_list = categories_list['commercial']

    def load_data(self,):
        GetCommercialData.__()
        cbecs = 
        residential_housing_units = # Use regional estimates of residential housing units as interpolator, extrapolator via regression model
        SEDS_CensusRgn =  # 'https://www.eia.gov/state/seds/sep_use/total/csv/use_all_btu.csv
        mer_data23_May_2016 = GetEIAData.eia_api(id_='711251')  # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711251'
        mer_data23_Jan_2017 = GetEIAData.eia_api(id_='711251')   # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711251'
        mer_data23_Dec_2019 =  GetEIAData.eia_api(id_='711251')  # 'http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE&category_id=711251'
        AER11_Table21C_Update = GetEIAData.eia_api(id_='711251')  # Estimates?
        # CDD_by_Division18 = 
        # HDD_by_Division18 =
        pass


    def estimate_regional_floorspace_share(self,):
    """assumed commercial floorspace in each region follows same trends as population or housing units"""

    def estimate_intensity_indexes_regional(self, parameter_list):
        """Data Sources: Fuel Consumption and electricity consumption from SEDS, Shares of regional floorspace from CBECs
        Purpose: used to produce weather adjustment facotrs"""
                """[summary]
        -Regional etsimates of floorspace are derived by applying regional shares to the total national floor space that were 
        estimatedwithin the spreadsheet historical_floorspace.xlsx
        -The time series of regional shares are estimated in the worksheet "Regional Shares" (key data for this process are the 
        shares of regional floor space reported in the various Commercial Building Energy Consumption Surveys (CBECS) or NBECS for 
        years prior to 1986)
        -To provide annual estimates of the shares, the assumption was made that commercial floor space in each region would 
        generally follow the same trends as population or housing units. Here residential estimates of residential housing units
        were used to reflect these overall trends. 
        -For each region, a simple regression model was estimated between the regional housing unit share and the regional commmercial
        building floor space share from the NBECS/CBECS. The regression employed both shares in log form. Based on the estimated 
        coefficients from this regression, the annual housing unit values are used to predict the share of commercial floor space 
        in each region. The normalized shares of floorspace by census region (normalized to sum to 1) are contained in Regional_Floorspace
        columns E through H. 
        -Regional floor space levels are calculated by multiplying the regional shares times the national estimate of floorspace  (taken 
        from sheet called Commercial_Total)

        """
        
        total_national_floorspace = 

        X = regional_housing_unit_share
        Y = regional_commercial_building_floorspace_shares  # from historical CBECS

        X = sm.add_constant(x)  # Add constant

        model = sm.OLS(X, Y).fit
        predictions = model.predict(x)

        print(model.summary())

        # reg = linear_model.LinearRegression()
        # reg.fit(X, Y)
        # coefficients = reg.coef_
        # regional_shares_floorspace = dict()  # assumed commercial floorspace in each region follows same trends as population or housing units
        # regional_floorspace = dict()
        # regional_intensity_index = dict()
        # for region in regions: 
        #     regional_energy_consumption =  # consumption of electricity and fuels
        #     predicted_share = reg.predict(annual_housing_unit_values)
        #     regional_shares_floorspace[region] = predicted_share
        #     region_floorspace =  predicted_share * total_national_floorspace
        #     regional_floorspace[region] = region_floorspace
        #     regional_intensity_index[region] = regional_energy_consumption / region_floorspace  

        return None

    def estimate_reclassification_electricity_sector_sales(self,):
        """Data Source: Commercial electricity sales from EIA SEDS
        Assumption: The significant changes (same magnitude but opposite directions in the same year) in state-level electricity slaes, typically
                    showing up in the commercial and industrial sectors reflect reclassification of some customers form the industrial rate class
                    to the commercial rate class or vice versa. 
        Strategy: adjust the more recent data by adding or subtracting a constant vlaue, determined from the year in which the reclassification was 
                  judged to have occured."""
        state_level_adjustment = 
        pass

    def adjusted_supplier_data(self,):
        """
        """"
        pass

    def national_calibration(self,):
        """"""
        pass

    def regional_intensity_aggregate(self,):
        """""""
        pass

