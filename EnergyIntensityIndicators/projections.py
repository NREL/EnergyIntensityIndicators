import pandas as pd 

from EnergyIntensityIndicators.pull_eia_api import GetEIAData
from EnergyIntensityIndicators.LMDI import CalculateLMDI

class MakeProjections:

    def __init__(self, ):
        pass

    def heating_cooling_degree_days():
        regions = ['ENC', 'ESC', 'MATL', 'MTN', 'NENGL', 'PCF', 'SATL', 'WNC', 'WSC', 'USA']
        regions_abbrev_dict = {'ENC': 'east_north_central', 'ESC': 'east_south_central', 'MATL': 'middle_atlantic',
                               'MTN': 'mountain', 'NENGL': 'new_england', 'PCF': 'pacific', 'SATL': 'south_atlantic',
                               'WNC': 'west_north_central', 'WSC': 'west_south_central', 'USA': 'National'}
        type_days = ['HDD', 'CDD']
        for region in regions: 
            for t in type_days:
                if sector_ is 'residential':
                    standard_id = f'AEO.2020.AEO2019REF.KEI_{t}_RESD_NA_NA_NA_{region}_{t}.A'
                elif sector_ is 'commercial':
                    standard_id = f'AEO.2020.AEO2019REF.KEI_NA_COMM_NA_NA_NA_{region}_{t}.A'
        return None

    def commercial_projections(self):
        """activity: floorspace
        energy: consumption trillion Btu
        """        
        # Commercial : Total Floorspace : New Additions, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.KEI_NA_COMM_NA_TFP_NADN_USA_BLNSQFT.A'
        # Commercial : Total Floorspace : Surviving, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.KEI_NA_COMM_NA_TFP_SURV_USA_BLNSQFT.A'
        # Commercial : Total Floorspace : Total, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.KEI_NA_COMM_NA_TFP_TOT_USA_BLNSQFT.A'


        commercial_categories = {'Commercial_Total': None, 'Total_Commercial_LMDI_UtilAdj': None}

        commercial_eia = GetEIAData('commercial')
        energy_use_commercial_electricity_us = commercial_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_COMM_NA_ELC_NA_NA_QBTU.A', id_type='series')
        energy_use_commercial_total_us = commercial_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_COMM_NA_TOT_NA_NA_QBTU.A', id_type='series')
        energy_use_commercial_delivered_energy_us = commercial_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_COMM_NA_DELE_NA_NA_QBTU.A', id_type='series')
        
        commercial_total_floorspace = commercial_eia.eia_api(id_='AEO.2020.AEO2019REF.KEI_NA_COMM_NA_TFP_TOT_USA_BLNSQFT.A', id_type='series')
        activity_data =  
        energy_data = 
        
        pass
    
    def industrial_projections(self):
        """activity: 
                - Value added --> Gross Domestic Product (Total Industrial only)
                - Gross Output
                - Value Added
            energy: Energy Consumption Trillion Btu
        """        

        {'energy': {'elec': , 'deliv': }, 'activity': {'value_added': , 'gross_output': }} 
        # Industrial : Value of Shipments : Agriculture, Mining, and Construction, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.ECI_NA_IDAL_NMFG_VOS_NA_USA_BLNY09DLR.A'
        # Industrial : Value of Shipments : Manufacturing, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.ECI_NA_IDAL_MANF_VOS_NA_USA_BLNY09DLR.A'
        # Industrial : Value of Shipments : Total, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.ECI_NA_IDAL_NA_VOS_NA_USA_BLNY09DLR.A'

        industrial_categories = {'Manufacturing': {'Food, Beverages, & Tobacco': None, 'Textile Mills and Products': None, 
                                               'Apparel & Leather': None, 'Wood Products': None, 'Paper': {'energy': {'elec': , 'deliv': }, 'activity': {'value_added': 'AEO.2020.AEO2019REF.ECI_VOS_IDAL_PPM_NA_NA_NA_BLNY09DLR.A', 'gross_output': }} , # vl
                                               'Printing & Allied Support': None, 'Petroleum & Coal Products': None, 'Chemicals': None,
                                               'Plastics & Rubber Products': None, 'Nonmetallic Mineral Products': None, 'Primary Metals': None,
                                               'Fabricated Metal Products': None, 'Machinery': None, 'Computer & Electronic Products': None,
                                               'Electical Equip. & Appliances': None, 'Transportation Equipment': None,
                                               'Furniture & Related Products': None, 'Miscellaneous': None},
                             'Nonmanufacturing': {'Agriculture, Forestry & Fishing': {'Agriculture': {'energy': {'elec': 'AEO.2020.REF2020.CNSM_NA_IDAL_AGG_PRC_NA_NA_TRLBTU.A', 'deliv': }, 'activity': {'value_added': , 'gross_output': }} ,'Forestry': , 'Fishing': }, # here elec is purchased electricity, Note: try to find total elec
                                                  'Mining': {'Petroleum and Natural Gas': None, 
                                                             'Other Mining': None, 
                                                             'Petroleum drilling and Mining Services': None},
                                                  'Construction': None}}

        industrial_eia = GetEIAData('industrial')
        energy_use_industrial_electricity_us = industrial_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_IDAL_NA_ELC_NA_NA_QBTU.A', id_type='series')
        energy_use_industrial_total_us = industrial_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_IDAL_NA_TOT_NA_NA_QBTU.A', id_type='series')
        energy_use_industrial_delivered_energy_us = industrial_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_IDAL_NA_DELE_NA_NA_QBTU.A', id_type='series')

        industrial_value_of_shipments_total = 

        activity_data = 
        energy_data = 
        
        pass

    def transportation_projections(self):
        """activity: 
                - Passenger-miles [P-M] (Passenger)
                - Ton-miles [T-M] (Freight)
            energy: Energy Consumption Trillion Btu
        """        
        {energy_use: '', 'activity': ''}

        # Commercial Carriers? --> domestic air carriers: AEO.2020.AEO2019REF.CNSM_NA_TRN_AIR_DAC_NA_NA_TRLBTU.A
        #                          international air carriers: AEO.2020.AEO2019REF.CNSM_NA_TRN_AIR_IAC_NA_NA_TRLBTU.A

        # Transportation Energy Use : Highway : Commercial Light Trucks, Reference, AEO2020: AEO.2020.REF2020.CNSM_NA_TRN_HWY_CML_NA_NA_TRLBTU.A
        # Transportation Energy Use : Highway : Freight Trucks : Large, Reference, AEO2020: AEO.2020.REF2020.CNSM_NA_TRN_HWY_FGHT_LGT26KLBS_NA_TRLBTU.A
        # Transportation Energy Use : Highway : Freight Trucks : Light Medium, Reference, AEO2020: AEO.2020.REF2020.CNSM_NA_TRN_HWY_FGHT_LITEMED_NA_TRLBTU.A
        # Transportation Energy Use : Highway : Freight Trucks : Medium, Reference, AEO2020: AEO.2020.REF2020.CNSM_NA_TRN_HWY_FGHT_MD10T26KLB_NA_TRLBTU.A
        # Transportation Energy Use : Highway : Freight Trucks, Reference, AEO2020: AEO.2020.REF2020.CNSM_NA_TRN_HWY_FGHT_NA_NA_TRLBTU.A

        # Transportation Energy Use : Non-Highway : Rail : Freight, Reference, AEO2020: AEO.2020.REF2020.CNSM_NA_TRN_RAIL_FGT_NA_NA_TRLBTU.A
        transportation_categories =  {'All_Passenger':
                                    {'Highway': 
                                        {'Passenger Cars and Trucks': 
                                            {'Passenger Car – SWB Vehicles': 
                                                {'Passenger Car': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_HWY_LDV_AUTO_NA_TRLBTU.A', 'activity': ''}, 'SWB Vehicles': {'energy_use': '', 'activity': ''}}, # Transportation Energy Use : Highway : Light-Duty Vehicles : Automobiles, Reference, AEO2020
                                             'Light Trucks – LWB Vehicles': 
                                                {'Light Trucks': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_HWY_LDV_LTRT_NA_TRLBTU.A', 'activity': ''}, 'LWB Vehicles': {'energy_use': '', 'activity': ''}}, # AEO.2020.REF2020.CNSM_NA_TRN_HWY_LDV_LTRT_NA_TRLBTU.A
                                             'Motorcycles': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_HWY_LDV_MCYCL_NA_TRLBTU.A', 'activity': ''}}, 
                                        'Buses': 
                                            {'Urban Bus': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_HWY_BUS_TNST_NA_TRLBTU.A', 'activity': ''}, 
                                            'Intercity Bus': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_HWY_BUS_ICYT_NA_TRLBTU.A', 'activity': ''}, 
                                            'School Bus': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_HWY_BUS_SCBU_NA_TRLBTU.A', 'activity': ''}}, 
                                        'Paratransit':
                                            {'energy_use': '', 'activity': ''}}, 
                                    'Rail': 
                                        {'Urban Rail': 
                                            {'Commuter Rail': {'energy_use': 'AEO.2020.AEO2019REF.CNSM_NA_TRN_RAIL_PSG_CMTR_NA_TRLBTU.A', 'activity': ''},
                                             'Heavy Rail': {'energy_use': '', 'activity': ''}, 
                                             'Light Rail': {'energy_use': '', 'activity': ''}}, 
                                        'Intercity Rail': {'energy_use': 'AEO.2020.AEO2019REF.CNSM_NA_TRN_RAIL_PSG_ICYT_NA_TRLBTU.A', 'activity': ''}}, 
                                    'Air': {'Commercial Carriers': {'Domestic Air Carriers': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_AIR_DAC_NA_NA_TRLBTU.A', 'activity': ''}, 'International Air Carriers': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_AIR_IAC_NA_NA_TRLBTU.A', 'activity': ''}}, 'General Aviation': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_AIR_GAV_NA_NA_TRLBTU.A', 'activity': ''}}}, 
                                'All_Freight': 
                                    {'Highway': 
                                        {'Freight-Trucks': 
                                            {'Single-Unit Truck': {'energy_use': '', 'activity': ''}, 'Combination Truck': {'energy_use': '', 'activity': ''}}}, 
                                    'Rail': {'energy_use': 'AEO.2020.AEO2019REF.CNSM_NA_TRN_RAIL_FGT_NA_NA_TRLBTU.A', 'activity': ''}, 
                                    'Air': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_AIR_FTC_NA_NA_TRLBTU.A', 'activity': ''}, 
                                    'Waterborne': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_WTR_DMT_NA_NA_TRLBTU.A', 'activity': ''}, # This is only domestic-- is that correct?
                                    'Pipeline': 
                                        {'Oil Pipeline': {'energy_use': '', 'activity': ''}, 'Natural Gas Pipeline': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_PIPL_NG_NA_NA_TRLBTU.A', 'activity': ''}}}}, 

        transportation_eia = GetEIAData('transportation')
        energy_use_transportation_electricity_us = transportation_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_TRN_NA_ELC_NA_NA_QBTU.A', id_type='series')
        energy_use_transportation_total_us = transportation_eia.eia_api(id_='YOUR_API_KEY_HERE&series_id=AEO.2020.AEO2019REF.CNSM_ENU_TRN_NA_TOT_NA_NA_QBTU.A', id_type='series')
        energy_use_transportation_delivered_energy_us = transportation_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_TRN_NA_DELE_NA_NA_QBTU.A', id_type='series')

        activity_data = 
        energy_data = 
        
        pass

    def residential_projections(self):
        """activity: 
                - Occupied Housing Units
                - Floorspace, Square Feet

            energy: Energy Consumption Trillion Btu
        """        
        {energy_use: '', occupied_housing_units: '', floorspace: ''}
        {'energy': {'elec': 'AEO.2020.REF2020.CNSM_ENU_RESD_NA_ELC_NA_NEENGL_QBTU.A', 'deliv': 'AEO.2020.REF2020.CNSM_ENU_RESD_NA_DELE_NA_NEENGL_QBTU.A'}, 'activity': {'occupied_housing_units': , 'floorspace': }}  
        # Residential : Key Indicators : Average House Square Footage, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.KEI_HHD_RESD_NA_NA_NA_USA_SQFT.A'
        # Residential : Key Indicators : Households : Mobile Homes, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.KEI_HHD_RESD_MBH_NA_NA_USA_MILL.A'
        # Residential : Key Indicators : Households : Multifamily, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.KEI_HHD_RESD_MFR_NA_NA_USA_MILL.A'
        # Residential : Key Indicators : Households : Single-Family, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.KEI_HHD_RESD_SFR_NA_NA_USA_MILL.A'
        # Residential : Key Indicators : Households : Total, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.KEI_HHD_RESD_TEN_NA_NA_USA_MILL.A'

        residential_categories = {'Northeast': {'Single-Family': None, 'Multi-Family': None, 'Manufactured Homes': None}, 
                               'Midwest': {'Single-Family': None, 'Multi-Family': None, 'Manufactured Homes': None},
                               'South': {'Single-Family': None, 'Multi-Family': None, 'Manufactured Homes': None},
                               'West': {'Single-Family': None, 'Multi-Family': None, 'Manufactured Homes': None}}

        self.sub_regions_dict = {'Northeast': {'New England': 'NENGL', 'Middle Atlantic': 'MATL'}, 'Midwest': {'East North Central': 'ENC', 'West North Central': 'WNC'}, 
                                 'South': {'South Atlantic': 'SATL', 'East South Central': 'ESC', 'West South Central': 'WSC'}, 'West': {'Mountain': 'MTN', 'Pacific': 'PCF'}}


        residential_eia = GetEIAData('residential')
        energy_use_residential_electricity_us = residential_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_RESD_NA_ELC_NA_NA_QBTU.A', id_type='series')
        energy_use_residential_total_us = residential_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_RESD_NA_TOT_NA_NA_QBTU.A', id_type='series')
        energy_use_residential_delivered_energy_us = residential_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_RESD_NA_DELE_NA_NA_QBTU.A', id_type='series')

        activity_data = 
        energy_data = 
        
        pass

    def electricity_projections():
        """activity: Million kWh
            energy: Energy Consumption Trillion Btu
        """        

        {energy_use: '', million_kwh: ''}

        electricity_categories = {'Elec Generation Total': 
                                    {'Elec Power Sector': 
                                        {'Electricity Only':
                                            {'Fossil Fuels': 
                                                {'Coal':  {energy_use: '', million_kwh: 'AEO.2020.AEO2019REF.GEN_NA_ELEP_POW_CL_NA_USA_BLNKWH.A'}, 
                                                'Petroleum':  {energy_use: '', million_kwh: 'AEO.2020.AEO2019REF.GEN_NA_ELEP_POW_PET_NA_USA_BLNKWH.A'}, 
                                                'Natural Gas':  {energy_use: '', million_kwh: 'AEO.2020.AEO2019REF.GEN_NA_ELEP_POW_NG_NA_USA_BLNKWH.A'}, 
                                                'Other Gasses':  {energy_use: '', million_kwh: ''}},
                                            'Nuclear':  {energy_use: '', million_kwh: 'AEO.2020.AEO2019REF.GEN_NA_ELEP_POW_NUP_NA_USA_BLNKWH.A'}, 
                                            'Hydro Electric':  {energy_use: '', million_kwh: ''}, 
                                            'Renewable':
                                                {'Wood':  {energy_use: '', million_kwh: ''}, 'Waste':  {energy_use: '', million_kwh: ''}, 
                                                'Geothermal':  {energy_use: '', million_kwh: ''}, 'Solar':  {energy_use: '', million_kwh: ''}, 
                                                'Wind':  {energy_use: '', million_kwh: ''}}},
                                        'Combined Heat & Power': 
                                            {'Fossil Fuels'
                                                {'Coal':  {energy_use: '', million_kwh: ''}, 
                                                'Petroleum':  {energy_use: '', million_kwh: ''}, 
                                                'Natural Gas':  {energy_use: '', million_kwh: ''}, 
                                                'Other Gasses':  {energy_use: '', million_kwh: ''}},
                                            'Renewable':
                                                {'Wood':  {energy_use: '', million_kwh: ''}, 
                                                'Waste':  {energy_use: '', million_kwh: ''}}}}, 
                                    'Commercial Sector':  {energy_use: '', million_kwh: ''}, 
                                    'Industrial Sector':  {energy_use: '', million_kwh: ''}},
                                  'All CHP':
                                    {'Elec Power Sector': 
                                        {'Combined Heat & Power':
                                            {'Fossil Fuels':
                                                {'Coal':  {energy_use: '', million_kwh: 'AEO.2020.AEO2019REF.GEN_NA_ELEP_CHP_CL_NA_USA_BLNKWH.A'}, 
                                                 'Petroleum':  {energy_use: 'AEO.2020.AEO2019REF.GEN_NA_ELEP_CHP_PET_NA_USA_BLNKWH.A', million_kwh: ''}, 
                                                 'Natural Gas':  {energy_use: '', million_kwh: 'AEO.2020.AEO2019REF.GEN_NA_ELEP_CHP_NG_NA_USA_BLNKWH.A'}, 
                                                 'Other Gasses':  {energy_use: '', million_kwh: ''}},
                                             'Renewable':
                                                {'Wood':  {energy_use: '', million_kwh: ''}, 
                                                 'Waste':  {energy_use: '', million_kwh: ''}},
                                             'Other':  {energy_use: '', million_kwh: ''}}}

        electricity_eia = GetEIAData('electricity')
        energy_use_electricity_electricity_us = electricity_eia.eia_api(id_='', id_type='series')
        energy_use_electricity_total_us = electricity_eia.eia_api(id_='', id_type='series')
        energy_use_electricity_delivered_energy_us = electricity_eia.eia_api(id_='', id_type='series')

        activity_data = 
        energy_data = 
        
        pass

    def economy_wide_projections(self):
        """activity: 
            energy: 
        """        
        activity_data = 
        energy_data = 
        
        pass
