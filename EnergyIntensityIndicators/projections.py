import pandas as pd 

from EnergyIntensityIndicators.pull_eia_api import GetEIAData
from EnergyIntensityIndicators.transportation import TransportationIndicators
from EnergyIntensityIndicators.industry import IndustrialIndicators
from EnergyIntensityIndicators.commercial import CommercialIndicators
from EnergyIntensityIndicators.residential import ResidentialIndicators
from EnergyIntensityIndicators.electricity import ElectricityIndicators



from EnergyIntensityIndicators.LMDI import CalculateLMDI

class MakeProjections():

    def __init__(self, directory, output_directory, level_of_aggregation, 
                 lmdi_model='multiplicative', base_year=1985, end_year=2018):
        self.directory = directory
        self.output_directory = output_directory
        self.level_of_aggregation = level_of_aggregation
        self.lmdi_model = lmdi_model 
        self.base_year = base_year
        self.end_year = end_year

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

        data_dict = {'Commercial_Total': {'energy': {'elec': energy_use_commercial_electricity_us, 'deliv': energy_use_commercial_delivered_energy_us}, 
                                          'activity': commercial_total_floorspace}, 
                     'Total_Commercial_LMDI_UtilAdj': None}
        
        return data_dict
    
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

        industrial_categories = {'Manufacturing': {'Food, Beverages, & Tobacco': {'Food Products': {'energy': {'total_energy': 'AEO.2020.REF2020.CNSM_NA_IDAL_FDP_NA_NA_NA_TRLBTU.A', 
                                                                                                               'purchased_electricity': 'AEO.2020.REF2020.CNSM_NA_IDAL_FDP_PRC_NA_NA_TRLBTU.A'}, 
                                                                                                    'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_IDAL_FDP_NA_NA_NA_BLNY09DLR.A'}},
                                                                                #   'Beverages and Tobacco Products': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                                                #                                      'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_MANF_BVTP_NA_NA_NA_BLNY09DLR.A'}}
                                                                                  }, 
                                                    # 'Textile Mills and Products': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                    #                                 'activity': {'shipments': 'AEO.2020.REF2020.ECI_VOS_MANF_TMP_NA_NA_NA_BLNY09DLR.A'}}, 
                                                    'Wood Products': {'energy': {'total_energy': 'AEO.2020.REF2020.CNSM_NA_IDAL_WDP_NA_NA_NA_TRLBTU.A', 
                                                                                 'purchased_electricity': 'AEO.2020.REF2020.CNSM_NA_IDAL_WDP_ELC_NA_NA_TRLBTU.A'}, 
                                                                      'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_SHP_IDAL_WDP_NA_NA_NA_BLNY09DLR.A'}},  
                                                    'Paper': {'energy': {'total_energy': 'AEO.2020.REF2020.CNSM_NA_IDAL_PPM_NA_NA_NA_TRLBTU.A', 
                                                                         'purchased_electricity': 'AEO.2020.REF2020.CNSM_NA_IDAL_PPM_PRC_NA_NA_TRLBTU.A'}, 
                                                              'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_IDAL_PPM_NA_NA_NA_BLNY09DLR.A'}} ,
                                                    # 'Printing': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                    #              'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_MANF_PRN_NA_NA_NA_BLNY09DLR.A'}}, 
                                                    'Petroleum & Coal Products': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                                                  'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_MANF_PCL_NA_NA_NA_BLNY09DLR.A'}}, 
                                                    'Chemicals': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                                  'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_MANF_CHE_NA_NA_NA_BLNY09DLR.A'}},
                                                    'Plastics & Rubber Products': {'energy': {'total_energy': 'AEO.2020.REF2020.CNSM_NA_IDAL_PLI_NA_NA_NA_TRLBTU.A', 
                                                                                              'purchased_electricity': 'AEO.2020.REF2020.CNSM_NA_IDAL_PLI_ELC_NA_NA_TRLBTU.A'}, 
                                                                                   'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_MANF_PRP_NA_NA_NA_BLNY09DLR.A'}},, 
                                                    'Nonmetallic Mineral Products': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                                                     'activity': {'shipments': 'AEO.2020.REF2020.ECI_VOS_MANF_SCG_NA_NA_NA_BLNY09DLR.A'}}, , 
                                                    'Primary Metals': {'energy': {'total_energy': {'iron_and_steel': 'AEO.2020.REF2020.CNSM_NA_IDAL_ISM_TOT_NA_NA_TRLBTU.A', 
                                                                                                   'aluminum': 'AEO.2020.REF2020.CNSM_NA_IDAL_AAP_TOT_NA_NA_TRLBTU.A'}, 
                                                                                  'purchased_electricity': {'iron_and_steel': 'AEO.2020.REF2020.CNSM_NA_IDAL_ISM_PRC_NA_NA_TRLBTU.A', 
                                                                                                            'aluminum': 'AEO.2020.REF2020.CNSM_NA_IDAL_AAP_PRC_NA_NA_TRLBTU.A'}}, 
                                                                       'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_MANF_PMM_NA_NA_NA_BLNY09DLR.A'}}},
                                                    'Fabricated Metal Products': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                                                  'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_IDAL_FABM_NA_NA_NA_BLNY09DLR.A'}}, , 
                                                    'Machinery': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                                  'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_IDAL_MCHI_NA_NA_NA_BLNY09DLR.A'}}, , 
                                                    'Computer & Electronic Products': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                                                       'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_MANF_CMPEL_NA_NA_NA_BLNY09DLR.A'}}, 
                                                    'Electrical Equipment': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                                             'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_IDAL_EEI_NA_NA_NA_BLNY09DLR.A'}}},
                                                    'Transportation Equipment': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                                                 'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_IDAL_TEQ_NA_NA_NA_BLNY09DLR.A'}}, 
                                                    'Furniture & Related Products': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                                                     'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_MANF_FRP_NA_NA_NA_BLNY09DLR.A'}}, 
                                                    'Miscellaneous': {'energy': {'total_energy': 'AEO.2020.REF2020.CNSM_NA_IDAL_BMF_NA_NA_NA_TRLBTU.A', 
                                                                                 'purchased_electricity': 'AEO.2020.REF2020.CNSM_NA_IDAL_BMF_ELC_NA_NA_TRLBTU.A'}, # Energy data from "Balance of Manufacturing"
                                                                      'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_MANF_MISC_NA_NA_NA_BLNY09DLR.A'}}},
                             'Nonmanufacturing': {'Agriculture/Forestry/Fishing/Hunting': {'energy': {'total_energy': 'AEO.2020.REF2020.CNSM_NA_IDAL_AGG_NA_NA_NA_TRLBTU.A',
                                                                                                      'purchased_electricity': 'AEO.2020.REF2020.CNSM_NA_IDAL_AGG_PRC_NA_NA_TRLBTU.A'}, 
                                                                                           'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_NMFG_AFF_NA_NA_NA_BLNY09DLR.A'}}, # here elec is purchased electricity, Note: try to find total elec
                                                  'Mining': {{'energy': {'total_energy': 'AEO.2020.REF2020.CNSM_NA_IDAL_MING_NA_NA_NA_TRLBTU.A', 
                                                                         'purchased_electricity': {'excluding_oil_shale': 'AEO.2020.REF2020.CNSM_NA_IDAL_MING_PEO_NA_NA_TRLBTU.A', 
                                                                                                   'including_oil_shale': 'AEO.2020.REF2020.CNSM_NA_IDAL_MING_PES_NA_NA_TRLBTU.A'}}, 
                                                              'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_NMFG_MING_NA_NA_NA_BLNY09DLR.A'}}},
                                                  'Construction': {'energy': {'total_energy': 'AEO.2020.REF2020.CNSM_NA_IDAL_CNS_NA_NA_NA_TRLBTU.A',
                                                                              'purchased_electricity': 'AEO.2020.REF2020.CNSM_NA_IDAL_CNS_PRC_NA_NA_TRLBTU.A'}, 
                                                                   'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_NMFG_CNS_NA_NA_NA_BLNY09DLR.A'}}}}

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
                                                {'Light Trucks': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_HWY_LDV_LTRT_NA_TRLBTU.A', 'activity': 'AEO.2020.REF2020.KEI_TRV_TRN_NA_CML_NA_NA_BLNVEHMLS.A'}, 'Light-Duty Vehicles': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_HWY_LDV_LTRT_NA_TRLBTU.A', 'activity': 'AEO.2020.REF2020.KEI_TRV_TRN_NA_LDV_NA_NA_BLNVEHMLS.A'}},
                                             'Motorcycles': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_HWY_LDV_MCYCL_NA_TRLBTU.A', 'activity': ''}}, 
                                        'Buses': {'activity': 'AEO.2020.REF2020._TRV_TRN_NA_BST_NA_NA_BPM.A'
                                                 'energy_use': {'Urban Bus': 'AEO.2020.REF2020.CNSM_NA_TRN_HWY_BUS_TNST_NA_TRLBTU.A', 
                                                                'Intercity Bus': 'AEO.2020.REF2020.CNSM_NA_TRN_HWY_BUS_ICYT_NA_TRLBTU.A', 
                                                                'School Bus': 'AEO.2020.REF2020.CNSM_NA_TRN_HWY_BUS_SCBU_NA_TRLBTU.A'}}, 
                                        'Paratransit':
                                            {'energy_use': '', 'activity': ''}}, 
                                    'Rail': {'energy_use': , 'activity': 'AEO.2020.REF2020._TRV_TRN_NA_RLP_NA_NA_BPM.A'}
                                        {'Urban Rail': 
                                            {'Commuter Rail': {'energy_use': 'AEO.2020.AEO2019REF.CNSM_NA_TRN_RAIL_PSG_CMTR_NA_TRLBTU.A', 'activity': ''},
                                             'Heavy Rail': {'energy_use': '', 'activity': ''}, 
                                             'Light Rail': {'energy_use': '', 'activity': ''}}, 
                                        'Intercity Rail': {'energy_use': 'AEO.2020.AEO2019REF.CNSM_NA_TRN_RAIL_PSG_ICYT_NA_TRLBTU.A', 'activity': ''}}, 
                                    'Air': {'Commercial Carriers': {'Domestic Air Carriers': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_AIR_DAC_NA_NA_TRLBTU.A', 'activity': ''}, 'International Air Carriers': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_AIR_IAC_NA_NA_TRLBTU.A', 'activity': ''}}, 'General Aviation': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_AIR_GAV_NA_NA_TRLBTU.A', 'activity': ''}}}, 
                                'All_Freight': 
                                    {'Highway': 
                                        {'Freight-Trucks': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_FGHT_USE_NA_NA_QBTU.A', 'activity': 'AEO.2020.REF2020.KEI_TRV_TRN_NA_FGHT_NA_NA_BLNVEHMLS.A'}}, 
                                    'Rail': {'energy_use': 'AEO.2020.AEO2019REF.CNSM_NA_TRN_RAIL_FGT_NA_NA_TRLBTU.A', 'activity': 'AEO.2020.REF2020.KEI_TRV_TRN_NA_RAIL_NA_NA_BLNTNMLS.A'}, 
                                    'Air': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_AIR_FTC_NA_NA_TRLBTU.A', 'activity': ''}, 
                                    'Waterborne': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_WTR_DMT_NA_NA_TRLBTU.A', 'activity': ''}, # This is only domestic-- is that correct?
                                    'Pipeline': {'energy_use': 'AEO.2020.REF2020.CNSM_NA_TRN_PFT_USE_NA_NA_QBTU.A'}
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

        activity_data = {'households': 'AEO.2020.AEO2019REF.KEI_HHD_RESD_TEN_NA_NA_USA_MILL.A', 'average_size': 'AEO.2020.AEO2019REF.KEI_HHD_RESD_NA_NA_NA_USA_SQFT.A'}
        energy_data =  {'elec': 'AEO.2020.REF2020.CNSM_ENU_RESD_NA_ELC_NA_NEENGL_QBTU.A', 'deliv': 'AEO.2020.REF2020.CNSM_ENU_RESD_NA_DELE_NA_NEENGL_QBTU.A'}
        data_dict = {'energy': energy_data, 'activity': activity_data}
        
        return data_dict

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
                                                'Natural Gas':  {energy_use: 'AEO.2020.REF2020.CNSM_ENU_ELEP_NA_NG_NA_NA_QBTU.A', million_kwh: 'AEO.2020.AEO2019REF.GEN_NA_ELEP_POW_NG_NA_USA_BLNKWH.A'}, 
                                                'Other Gasses':  {energy_use: '', million_kwh: ''}},
                                            'Nuclear':  {energy_use: 'AEO.2020.REF2020.CNSM_ENU_ELEP_NA_NUC_NA_NA_QBTU.A', million_kwh: 'AEO.2020.AEO2019REF.GEN_NA_ELEP_POW_NUP_NA_USA_BLNKWH.A'}, 
                                            'Hydro Electric':  {energy_use: '', million_kwh: 'AEO.2020.REF2020.GEN_NA_ELEP_NA_HYD_CNV_NA_BLNKWH.A'}, 
                                            'Renewable': # 'energy_use': 'AEO.2020.REF2020.CNSM_ENU_ELEP_NA_REN_NA_NA_QBTU.A', 
                                                {'Wood':  {energy_use: 'AEO.2020.REF2020.CNSM_NA_ELEP_NA_WBM_NA_NA_QBTU.A', million_kwh: 'AEO.2020.REF2020.GEN_NA_ELEP_NA_WBM_NA_NA_BLNKWH.A'}, # This is wood and other biomass
                                                'Waste':  {energy_use: '', million_kwh: 'AEO.2020.REF2020.GEN_NA_ELEP_NA_BGM_NA_NA_BLNKWH.A'}, # This is Biogenic Municipal Waste
                                                'Geothermal':  {energy_use: '', billion_kwh: 'AEO.2020.REF2020.GEN_NA_ELEP_NA_GEOTHM_NA_NA_BLNKWH.A'}, 
                                                'Solar':  {'Solar Photovoltaic': {energy_use: '', million_kwh: 'AEO.2020.REF2020.GEN_NA_ELEP_NA_SLR_PHTVL_NA_BLNKWH.A'}, 
                                                           'Solar Thermal': {energy_use: '', million_kwh: 'AEO.2020.REF2020.GEN_NA_ELEP_NA_SLR_THERM_NA_BLNKWH.A'}}, 
                                                'Wind':  {energy_use: 'AEO.2020.REF2020.CNSM_NA_ELEP_NA_OFW_NA_NA_QBTU.A', million_kwh: 'AEO.2020.REF2020.GEN_NA_ELEP_NA_WND_NA_NA_BLNKWH.A'}}},
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
        activity_data = {}
        energy_data = 
        
        pass
    
    def collect_data(self, economy_wide):
        if economy_wide:
            data_dict = self.economy_wide_projections()
        else:
            data_dict = {'commercial': self.commercial_projections(),
                        'residential': self.residential_projections(), 
                        'industrial': self.industrial_projections(), 
                        'transportation': self.transportation_projections(),
                        'electricity': self.electricity_projections()}
        

    def main(breakout, save_breakout, calculate_lmdi): 
        data_dict = self.collect_data()
        results = self.get_nested_lmdi(level_of_aggregation=self.level_of_aggregation, 
                                       breakout=breakout, save_breakout=save_breakout, 
                                       calculate_lmdi=calculate_lmdi, raw_data=data_dict,
                                       lmdi_type='LMDI-I')
if __name__ == '__main__': 
    indicators = MakeProjections(directory='C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020', 
                                 output_directory='C:/Users/irabidea/Desktop/LMDI_Results', 
                                 level_of_aggregation='', lmdi_model=['multiplicative', 'additive']) # 
    indicators.main(breakout=False, save_breakout=True, calculate_lmdi=True)
