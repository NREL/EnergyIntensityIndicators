import pandas as pd 

from EnergyIntensityIndicators.pull_eia_api import GetEIAData
from EnergyIntensityIndicators.transportation import TransportationIndicators
from EnergyIntensityIndicators.industry import IndustrialIndicators
from EnergyIntensityIndicators.commercial import CommercialIndicators
from EnergyIntensityIndicators.residential import ResidentialIndicators
from EnergyIntensityIndicators.electricity import ElectricityIndicators
from EnergyIntensityIndicators.weather_factors import WeatherFactors


from EnergyIntensityIndicators.LMDI import CalculateLMDI

class MakeProjections(CalculateLMDI):

    def __init__(self, directory, output_directory, level_of_aggregation, 
                 lmdi_model='multiplicative', base_year=1985, end_year=2018):
        self.directory = directory
        self.output_directory = output_directory
        self.level_of_aggregation = level_of_aggregation
        self.lmdi_model = lmdi_model 
        self.base_year = base_year
        self.end_year = end_year
        self.sub_categories_list = {'Economy_Wide': {'Transportation': {'All_Passenger':
                                                                            {'Highway':  
                                                                                {'Passenger Cars and Trucks': 
                                                                                    {'Light Trucks – LWB Vehicles': 
                                                                                        {'Light Trucks': None,
                                                                                        'Light-Duty Vehicles': None}}, 
                                                                                'Buses': None, 
                                                                            'Rail': None,

                                                                            'Air': None}}, 
                                                                        'All_Freight': 
                                                                            {'Highway': 
                                                                                {'Freight-Trucks': None}, 
                                                                            'Rail': None, 
                                                                            'Air': None}},
                                                    'Residential': None, 
                                                    'Commercial': None, 
                                                    'Industrial': {'Manufacturing': {'Food, Beverages, & Tobacco': 
                                                                                        {'Food Products': None}, 
                                                                
                                                                                    'Wood Products': None,  
                                                                                    'Paper': None,
                                                                                    'Petroleum & Coal Products': None, 
                                                                                    'Chemicals': None,
                                                                                    'Plastics & Rubber Products': None, 
                                                                                    'Primary Metals': None,
                                                                                    'Fabricated Metal Products': None, 
                                                                                    'Machinery': None, 
                                                                                    'Computer & Electronic Products': None, 
                                                                                    'Electrical Equipment': None,
                                                                                    'Transportation Equipment': None, 
                                                                                    'Miscellaneous': None},
                                                                    'Nonmanufacturing': {'Agriculture/Forestry/Fishing/Hunting': None,
                                                                                        'Mining': None,
                                                                                        'Construction': None}}, 
                                                    'Electricity': 
                                                        {'Electricity Only':
                                                            {'Fossil Fuels': 
                                                                {'Coal':  None, 
                                                                'Petroleum':  None, 
                                                                'Natural Gas':  None,
                                                            'Nuclear':  None, 
                                                            'Hydro Electric': None, 
                                                            'Renewable': 
                                                                {'Wood':  None, # This is wood and other biomass
                                                                'Waste': None, # This is Biogenic Municipal Waste
                                                                'Wind':  None}}}}}}

        self.energy_types = ['elec', 'deliv', 'total_energy', 'purchased_energy', 'primary', ]
        super().__init__(sector='Projections', level_of_aggregation=level_of_aggregation, lmdi_models=lmdi_model, categories_dict=self.sub_categories_list, 
                         energy_types=self.energy_types, directory=directory, output_directory=output_directory, base_year=base_year, end_year=end_year)

    def commercial_projections(self):
        """Gather Commercial projections data

        activity: floorspace
        energy: consumption trillion Btu
        """        
        # Commercial : Total Floorspace : New Additions, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.KEI_NA_COMM_NA_TFP_NADN_USA_BLNSQFT.A'
        # Commercial : Total Floorspace : Surviving, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.KEI_NA_COMM_NA_TFP_SURV_USA_BLNSQFT.A'
        # Commercial : Total Floorspace : Total, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.KEI_NA_COMM_NA_TFP_TOT_USA_BLNSQFT.A'


        commercial_categories = {'Commercial_Total': None, 'Total_Commercial_LMDI_UtilAdj': None}

        commercial_eia = GetEIAData('commercial')

        energy_use_commercial_electricity_us = commercial_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_COMM_NA_ELC_NA_NA_QBTU.A', id_type='series', new_name='Commercial')
        energy_use_commercial_total_us = commercial_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_COMM_NA_TOT_NA_NA_QBTU.A', id_type='series', new_name='Commercial')
        energy_use_commercial_delivered_energy_us = commercial_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_COMM_NA_DELE_NA_NA_QBTU.A', id_type='series', new_name='Commercial')
        
        commercial_total_floorspace = commercial_eia.eia_api(id_='AEO.2020.AEO2019REF.KEI_NA_COMM_NA_TFP_TOT_USA_BLNSQFT.A', id_type='series', new_name='Commercial')
     
        data_dict = {'Commercial_Total': {'energy': {'elec': energy_use_commercial_electricity_us, 'deliv': energy_use_commercial_delivered_energy_us}, 
                                          'activity': commercial_total_floorspace}}
        
        return data_dict
    
    def industrial_projections(self):
        """Gather Industrial projections data
        activity: 
                - Value added --> Gross Domestic Product (Total Industrial only)
                - Gross Output
                - Value Added
            energy: Energy Consumption Trillion Btu
        """        

        # Industrial : Value of Shipments : Agriculture, Mining, and Construction, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.ECI_NA_IDAL_NMFG_VOS_NA_USA_BLNY09DLR.A'
        # Industrial : Value of Shipments : Manufacturing, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.ECI_NA_IDAL_MANF_VOS_NA_USA_BLNY09DLR.A'
        # Industrial : Value of Shipments : Total, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.ECI_NA_IDAL_NA_VOS_NA_USA_BLNY09DLR.A'
        industrial_eia = GetEIAData('industrial')

        industrial_categories = {'Manufacturing': {'Food, Beverages, & Tobacco': {'Food Products': None}, 
                    
                                                    'Wood Products': None,  
                                                    'Paper': None,
                                                    'Petroleum & Coal Products': None, 
                                                    'Chemicals': None,
                                                    'Plastics & Rubber Products': None, 
                                                    'Primary Metals': None,
                                                    'Fabricated Metal Products': None, 
                                                    'Machinery': None, 
                                                    'Computer & Electronic Products': None, 
                                                    'Electrical Equipment': None,
                                                    'Transportation Equipment': None, 
                                                    'Miscellaneous': None},
                                 'Nonmanufacturing': {'Agriculture/Forestry/Fishing/Hunting': None,
                                                  'Mining': None,
                                                  'Construction': None}}

        data_dict = {'Manufacturing': {'Food, Beverages, & Tobacco': {'Food Products': {'energy': {'total_energy': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_FDP_NA_NA_NA_TRLBTU.A', id_type='series', new_name='Food Products'), 
                                                                                                               'purchased_electricity': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_FDP_PRC_NA_NA_TRLBTU.A', id_type='series', new_name='Food Products')}, 
                                                                                                    'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_IDAL_FDP_NA_NA_NA_BLNY09DLR.A', id_type='series', new_name='Food Products')}},
                                                                                #   'Beverages and Tobacco Products': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                                                #                                      'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_MANF_BVTP_NA_NA_NA_BLNY09DLR.A'}}
                                                                                  }, 
                                                    # 'Textile Mills and Products': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                    #                                 'activity': {'shipments': 'AEO.2020.REF2020.ECI_VOS_MANF_TMP_NA_NA_NA_BLNY09DLR.A'}}, 
                                                    'Wood Products': {'energy': {'total_energy': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_WDP_NA_NA_NA_TRLBTU.A', id_type='series', new_name='Wood Products'), 
                                                                                 'purchased_electricity': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_WDP_ELC_NA_NA_TRLBTU.A', id_type='series', new_name='Wood Products')}, 
                                                                      'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_SHP_IDAL_WDP_NA_NA_NA_BLNY09DLR.A', id_type='series', new_name='Wood Products')}},  
                                                    'Paper': {'energy': {'total_energy': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_PPM_NA_NA_NA_TRLBTU.A', id_type='series', new_name='Paper'), 
                                                                         'purchased_electricity': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_PPM_PRC_NA_NA_TRLBTU.A', id_type='series', new_name='Paper')}, 
                                                              'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_IDAL_PPM_NA_NA_NA_BLNY09DLR.A', id_type='series', new_name='Paper')}} ,
                                                    # 'Printing': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                    #              'activity': {'value_of_shipments': 'AEO.2020.REF2020.ECI_VOS_MANF_PRN_NA_NA_NA_BLNY09DLR.A'}}, 
                                                    # 'Petroleum & Coal Products': {'energy': {'total_energy': industrial_eia.eia_api(id_='', id_type='series'), 
                                                    #                                          'purchased_electricity': industrial_eia.eia_api(id_='', id_type='series')}, 
                                                    #                               'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_MANF_PCL_NA_NA_NA_BLNY09DLR.A', id_type='series')}}, 
                                                    'Chemicals': {'energy': {'total_energy': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_BCH_NA_NA_NA_TRLBTU.A', id_type='series', new_name='Chemicals'), 
                                                                             'purchased_electricity': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_BCH_HAP_PRC_NA_TRLBTU.A', id_type='series', new_name='Chemicals')}, 
                                                                  'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_MANF_CHE_NA_NA_NA_BLNY09DLR.A', id_type='series', new_name='Chemicals')}},
                                                    'Plastics & Rubber Products': {'energy': {'total_energy': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_PLI_NA_NA_NA_TRLBTU.A', id_type='series', new_name='Plastics & Rubber Products'), 
                                                                                              'purchased_electricity': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_PLI_ELC_NA_NA_TRLBTU.A', id_type='series', new_name='Plastics & Rubber Products')}, 
                                                                                   'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_MANF_PRP_NA_NA_NA_BLNY09DLR.A', id_type='series', new_name='Plastics & Rubber Products')}},
                                                    # 'Nonmetallic Mineral Products': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                                                    #  'activity': {'shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_MANF_SCG_NA_NA_NA_BLNY09DLR.A', id_type='series')}}, , 
                                                    'Primary Metals': {'energy': {'total_energy': {'iron_and_steel': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_ISM_TOT_NA_NA_TRLBTU.A', id_type='series', new_name='Primary Metals'), 
                                                                                                   'aluminum': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_AAP_TOT_NA_NA_TRLBTU.A', id_type='series', new_name='Primary Metals')}, 
                                                                                  'purchased_electricity': {'iron_and_steel': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_ISM_PRC_NA_NA_TRLBTU.A', id_type='series', new_name='Primary Metals'), 
                                                                                                            'aluminum': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_AAP_PRC_NA_NA_TRLBTU.A', id_type='series', new_name='Primary Metals')}}, 
                                                                       'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_MANF_PMM_NA_NA_NA_BLNY09DLR.A', id_type='series', new_name='Primary Metals')}},
                                                    # 'Fabricated Metal Products': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                    #                               'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_IDAL_FABM_NA_NA_NA_BLNY09DLR.A', id_type='series')}}, , 
                                                    # 'Machinery': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                    #               'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_IDAL_MCHI_NA_NA_NA_BLNY09DLR.A', id_type='series')}}, , 
                                                    # 'Computer & Electronic Products': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                    #                                    'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_MANF_CMPEL_NA_NA_NA_BLNY09DLR.A', id_type='series')}}, 
                                                    # 'Electrical Equipment': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                    #                          'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_IDAL_EEI_NA_NA_NA_BLNY09DLR.A', id_type='series')}}},
                                                    # 'Transportation Equipment': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                    #                              'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_IDAL_TEQ_NA_NA_NA_BLNY09DLR.A', id_type='series')}}, 
                                                    # 'Furniture & Related Products': {'energy': {'total_energy': , 'purchased_electricity': }, 
                                                                                    #  'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_MANF_FRP_NA_NA_NA_BLNY09DLR.A', id_type='series')}}, 
                                                    'Miscellaneous': {'energy': {'total_energy': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_BMF_NA_NA_NA_TRLBTU.A', id_type='series', new_name='Miscellaneous'), 
                                                                                 'purchased_electricity': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_BMF_ELC_NA_NA_TRLBTU.A', id_type='series', new_name='Miscellaneous')}, # Energy data from "Balance of Manufacturing"
                                                                      'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_MANF_MISC_NA_NA_NA_BLNY09DLR.A', id_type='series', new_name='Miscellaneous')}}},
                     'Nonmanufacturing': {'Agriculture/Forestry/Fishing/Hunting': {'energy': {'total_energy': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_AGG_NA_NA_NA_TRLBTU.A', id_type='series', new_name='Agriculture/Forestry/Fishing/Hunting'),
                                                                                                'purchased_electricity': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_AGG_PRC_NA_NA_TRLBTU.A', id_type='series', new_name='Agriculture/Forestry/Fishing/Hunting')}, 
                                                                                    'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_NMFG_AFF_NA_NA_NA_BLNY09DLR.A', id_type='series', new_name='Agriculture/Forestry/Fishing/Hunting')}}, # here elec is purchased electricity, Note: try to find total elec
                                            'Mining': {'energy': {'total_energy': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_MING_NA_NA_NA_TRLBTU.A', id_type='series', new_name='Mining'), 
                                                                    'purchased_electricity': {'excluding_oil_shale': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_MING_PEO_NA_NA_TRLBTU.A', id_type='series', new_name='Mining'), 
                                                                                            'including_oil_shale': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_MING_PES_NA_NA_TRLBTU.A', id_type='series', new_name='Mining')}}, 
                                                        'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_NMFG_MING_NA_NA_NA_BLNY09DLR.A', id_type='series', new_name='Mining')}},
                                            'Construction': {'energy': {'total_energy': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_CNS_NA_NA_NA_TRLBTU.A', id_type='series', new_name='Construction'),
                                                                        'purchased_electricity': industrial_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_IDAL_CNS_PRC_NA_NA_TRLBTU.A', id_type='series', new_name='Construction')}, 
                                                            'activity': {'value_of_shipments': industrial_eia.eia_api(id_='AEO.2020.REF2020.ECI_VOS_NMFG_CNS_NA_NA_NA_BLNY09DLR.A', id_type='series', new_name='Construction')}}}}



        energy_use_industrial_electricity_us = industrial_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_IDAL_NA_ELC_NA_NA_QBTU.A', id_type='series', new_name='Industry')
        energy_use_industrial_total_us = industrial_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_IDAL_NA_TOT_NA_NA_QBTU.A', id_type='series', new_name='Industry')
        energy_use_industrial_delivered_energy_us = industrial_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_IDAL_NA_DELE_NA_NA_QBTU.A', id_type='series', new_name='Industry')
        
        return data_dict

    def transportation_projections(self):
        """Gather Transportation projections data
        activity: 
            - Passenger-miles [P-M] (Passenger)
            - Ton-miles [T-M] (Freight)
        energy: Energy Consumption Trillion Btu
        """        

        # Commercial Carriers? --> domestic air carriers: AEO.2020.AEO2019REF.CNSM_NA_TRN_AIR_DAC_NA_NA_TRLBTU.A
        #                          international air carriers: AEO.2020.AEO2019REF.CNSM_NA_TRN_AIR_IAC_NA_NA_TRLBTU.A

        # Transportation Energy Use : Highway : Commercial Light Trucks, Reference, AEO2020: AEO.2020.REF2020.CNSM_NA_TRN_HWY_CML_NA_NA_TRLBTU.A
        # Transportation Energy Use : Highway : Freight Trucks : Large, Reference, AEO2020: AEO.2020.REF2020.CNSM_NA_TRN_HWY_FGHT_LGT26KLBS_NA_TRLBTU.A
        # Transportation Energy Use : Highway : Freight Trucks : Light Medium, Reference, AEO2020: AEO.2020.REF2020.CNSM_NA_TRN_HWY_FGHT_LITEMED_NA_TRLBTU.A
        # Transportation Energy Use : Highway : Freight Trucks : Medium, Reference, AEO2020: AEO.2020.REF2020.CNSM_NA_TRN_HWY_FGHT_MD10T26KLB_NA_TRLBTU.A
        # Transportation Energy Use : Highway : Freight Trucks, Reference, AEO2020: AEO.2020.REF2020.CNSM_NA_TRN_HWY_FGHT_NA_NA_TRLBTU.A

        # Transportation Energy Use : Non-Highway : Rail : Freight, Reference, AEO2020: AEO.2020.REF2020.CNSM_NA_TRN_RAIL_FGT_NA_NA_TRLBTU.A
        transportation_eia = GetEIAData('transportation')

        transportation_categories =  {'All_Passenger':
                                    {'Highway':  
                                        {'Passenger Cars and Trucks': 
                                            {'Light Trucks – LWB Vehicles': 
                                                {'Light Trucks': {'energy': {'primary': transportation_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_TRN_HWY_LDV_LTRT_NA_TRLBTU.A', id_type='series', new_name='Light Trucks')}, 
                                                                  'activity': transportation_eia.eia_api(id_='AEO.2020.REF2020.KEI_TRV_TRN_NA_CML_NA_NA_BLNVEHMLS.A', id_type='series', new_name='Light Trucks')},
                                                'Light-Duty Vehicles': {'energy': {'primary': transportation_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_TRN_HWY_LDV_LTRT_NA_TRLBTU.A', id_type='series', new_name='Light-Duty Vehicles')}, 
                                                                        'activity': transportation_eia.eia_api(id_='AEO.2020.REF2020.KEI_TRV_TRN_NA_LDV_NA_NA_BLNVEHMLS.A', id_type='series', new_name='Light-Duty Vehicles')}}}, 
                                        'Buses': {'activity': transportation_eia.eia_api(id_='AEO.2020.REF2020._TRV_TRN_NA_BST_NA_NA_BPM.A', id_type='series', new_name='Buses'),
                                                 'energy': {'primary': transportation_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_TRN_HWY_BUS_NA_NA_TRLBTU.A', id_type='series', new_name='Buses')}}}, 
                                    'Rail': {'energy': {'primary': transportation_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_TRN_RAIL_PSG_PSG_NA_TRLBTU.A', id_type='series', new_name='Rail')}, 
                                             'activity': transportation_eia.eia_api(id_='AEO.2020.REF2020._TRV_TRN_NA_RLP_NA_NA_BPM.A', id_type='series', new_name='Rail')}},

                                    'Air': {'energy': {'primary': transportation_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_TRN_AIR_IAC_NA_NA_TRLBTU.A', id_type='series', new_name='Air')}, 
                                             'activity': transportation_eia.eia_api(id_='AEO.2020.REF2020.KEI_TRV_TRN_NA_AIR_NA_NA_BLNSEATMLS.A', id_type='series', new_name='Air')}, 
                                'All_Freight': 
                                    {'Highway': 
                                        {'Freight-Trucks': {'energy': {'primary': transportation_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_TRN_FGHT_USE_NA_NA_QBTU.A', id_type='series', new_name='Freight-Trucks')},
                                                            'activity': transportation_eia.eia_api(id_='AEO.2020.REF2020.KEI_TRV_TRN_NA_FGHT_NA_NA_BLNVEHMLS.A', id_type='series', new_name='Freight-Trucks')}}, 
                                    'Rail': {'energy': {'primary': transportation_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_NA_TRN_RAIL_FGT_NA_NA_TRLBTU.A', id_type='series', new_name='Rail')}, 
                                             'activity': transportation_eia.eia_api(id_='AEO.2020.REF2020.KEI_TRV_TRN_NA_RAIL_NA_NA_BLNTNMLS.A', id_type='series', new_name='Rail')}, 
                                    'Air': {'energy': {'primary': transportation_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_TRN_AIR_FTC_NA_NA_TRLBTU.A', id_type='series', new_name='Air')}, 
                                            'activity': transportation_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_TRN_AIR_FTC_NA_NA_TRLBTU.A', id_type='series', new_name='Air')}}}

        energy_use_transportation_electricity_us = transportation_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_TRN_NA_ELC_NA_NA_QBTU.A', id_type='series', new_name='Transportation')
        energy_use_transportation_total_us = transportation_eia.eia_api(id_='YOUR_API_KEY_HERE&series_id=AEO.2020.AEO2019REF.CNSM_ENU_TRN_NA_TOT_NA_NA_QBTU.A', id_type='series', new_name='Transportation')
        energy_use_transportation_delivered_energy_us = transportation_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_TRN_NA_DELE_NA_NA_QBTU.A', id_type='series', new_name='Transportation')

        data_dict = transportation_categories
        return data_dict

    def residential_projections(self):
        """Gather Residential projections data
        activity: 
            - Occupied Housing Units
            - Floorspace, Square Feet

        energy: Energy Consumption Trillion Btu
        """        
        # Residential : Key Indicators : Average House Square Footage, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.KEI_HHD_RESD_NA_NA_NA_USA_SQFT.A'
        # Residential : Key Indicators : Households : Mobile Homes, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.KEI_HHD_RESD_MBH_NA_NA_USA_MILL.A'
        # Residential : Key Indicators : Households : Multifamily, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.KEI_HHD_RESD_MFR_NA_NA_USA_MILL.A'
        # Residential : Key Indicators : Households : Single-Family, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.KEI_HHD_RESD_SFR_NA_NA_USA_MILL.A'
        # Residential : Key Indicators : Households : Total, AEO2019, AEO2020 --> 'AEO.2020.AEO2019REF.KEI_HHD_RESD_TEN_NA_NA_USA_MILL.A'

        self.sub_regions_dict = {'Northeast': {'New England': 'NENGL', 'Middle Atlantic': 'MATL'}, 'Midwest': {'East North Central': 'ENC', 'West North Central': 'WNC'}, 
                                 'South': {'South Atlantic': 'SATL', 'East South Central': 'ESC', 'West South Central': 'WSC'}, 'West': {'Mountain': 'MTN', 'Pacific': 'PCF'}}

        residential_eia = GetEIAData('residential')
        energy_use_residential_electricity_us = residential_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_RESD_NA_ELC_NA_NA_QBTU.A', id_type='series')
        energy_use_residential_total_us = residential_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_RESD_NA_TOT_NA_NA_QBTU.A', id_type='series')
        energy_use_residential_delivered_energy_us = residential_eia.eia_api(id_='AEO.2020.AEO2019REF.CNSM_ENU_RESD_NA_DELE_NA_NA_QBTU.A', id_type='series')


        res_housholds= residential_eia.eia_api(id_='AEO.2020.AEO2019REF.KEI_HHD_RESD_TEN_NA_NA_USA_MILL.A', id_type='series', new_name='Residential')
        res_avg_size = residential_eia.eia_api(id_='AEO.2020.AEO2019REF.KEI_HHD_RESD_NA_NA_NA_USA_SQFT.A', id_type='series', new_name='Residential')
        residential_floorspace = res_avg_size.multiply(res_housholds.values)

        activity_data = {'households': res_housholds, 
                         'average_size': res_avg_size, 
                         'total_floorspace': residential_floorspace}

        energy_data =  {'elec': residential_eia.eia_api(id_='AEO.2020.REF2020.CNSM_ENU_RESD_NA_ELC_NA_NEENGL_QBTU.A', id_type='series', new_name='Residential'),
                        'deliv': residential_eia.eia_api(id_='AEO.2020.REF2020.CNSM_ENU_RESD_NA_DELE_NA_NEENGL_QBTU.A', id_type='series', new_name='Residential')}

        data_dict = {'energy': energy_data, 'activity': activity_data} 
        
        return data_dict

    @staticmethod
    def electricity_projections():
        """Gather Commercial projections data
        activity: Million kWh
        energy: Energy Consumption Trillion Btu 
        """        
        electricity_eia = GetEIAData('electricity')

        data_dict = {'Elec Generation Total': 
                        {'Elec Power Sector': 
                            {'Electricity Only':
                                {'Fossil Fuels': 
                                    {'Coal':  {'energy': {'primary': electricity_eia.eia_api(id_='AEO.2020.REF2020.CNSM_ENU_ELEP_NA_STC_NA_NA_QBTU.A', id_type='series', new_name='Coal')}, 
                                                'activity': electricity_eia.eia_api(id_='AEO.2020.AEO2019REF.GEN_NA_ELEP_POW_CL_NA_USA_BLNKWH.A', id_type='series', new_name='Coal')}, 
                                    'Petroleum':  {'energy': {'primary': electricity_eia.eia_api(id_='AEO.2020.REF2020.CNSM_ENU_ELEP_NA_LFL_NA_NA_QBTU.A', id_type='series', new_name='Petroleum')}, 
                                                    'activity': electricity_eia.eia_api(id_='AEO.2020.AEO2019REF.GEN_NA_ELEP_POW_PET_NA_USA_BLNKWH.A', id_type='series', new_name='Petroleum')}, 
                                    'Natural Gas':  {'energy': {'primary': electricity_eia.eia_api(id_='AEO.2020.REF2020.CNSM_ENU_ELEP_NA_NG_NA_NA_QBTU.A', id_type='series', new_name='Natural Gas')}, 
                                                        'activity': electricity_eia.eia_api(id_='AEO.2020.AEO2019REF.GEN_NA_ELEP_POW_NG_NA_USA_BLNKWH.A', id_type='series', new_name='Natural Gas')}},
                                'Nuclear':  {'energy': {'primary': electricity_eia.eia_api(id_='AEO.2020.REF2020.CNSM_ENU_ELEP_NA_NUC_NA_NA_QBTU.A', id_type='series', new_name='Nuclear')}, 
                                                'activity': electricity_eia.eia_api(id_='AEO.2020.AEO2019REF.GEN_NA_ELEP_POW_NUP_NA_USA_BLNKWH.A', id_type='series', new_name='Nuclear')}, 
                                # 'Hydro Electric':  {'energy': '', 
                                #                     'activity': electricity_eia.eia_api(id_='AEO.2020.REF2020.GEN_NA_ELEP_NA_HYD_CNV_NA_BLNKWH.A', id_type='series')}, 
                                'Renewable': 
                                    {'Wood':  {'energy': {'primary': electricity_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_ELEP_NA_WBM_NA_NA_QBTU.A', id_type='series', new_name='Wood')}, 
                                                'activity': electricity_eia.eia_api(id_='AEO.2020.REF2020.GEN_NA_ELEP_NA_WBM_NA_NA_BLNKWH.A', id_type='series', new_name='Wood')}, # This is wood and other biomass
                                    'Waste':  {'energy': {'primary': electricity_eia.eia_api(id_='AEO.2020.REF2020.CNSM_ENU_ELEP_NA_NBMSW_NA_NA_QBTU.A', id_type='series', new_name='Waste'), 
                                                'activity': electricity_eia.eia_api(id_='AEO.2020.REF2020.GEN_NA_ELEP_NA_BGM_NA_NA_BLNKWH.A', id_type='series', new_name='Waste')}, # This is Biogenic Municipal Waste
                                    # 'Solar':  {'Solar Photovoltaic': {'energy': electricity_eia.eia_api(id_='', id_type='series'), 
                                    #                                     'activity': electricity_eia.eia_api(id_='AEO.2020.REF2020.GEN_NA_ELEP_NA_SLR_PHTVL_NA_BLNKWH.A', id_type='series')}, 
                                    #             'Solar Thermal': {'energy': electricity_eia.eia_api(id_='', id_type='series'), 
                                    #                                 'activity': electricity_eia.eia_api(id_='AEO.2020.REF2020.GEN_NA_ELEP_NA_SLR_THERM_NA_BLNKWH.A', id_type='series')}}, 
                                    'Wind':  {'energy': {'primary': electricity_eia.eia_api(id_='AEO.2020.REF2020.CNSM_NA_ELEP_NA_OFW_NA_NA_QBTU.A', id_type='series', new_name='Wind')}, 
                                                'activity': electricity_eia.eia_api(id_='AEO.2020.REF2020.GEN_NA_ELEP_NA_WND_NA_NA_BLNKWH.A', id_type='series', new_name='Wind')}}}}}}}
                                                    
        
        return data_dict 
    
    def collect_data(self):
        """Calculate decomposition of projected energy use for selected sectors
        """
        
        data_dict = {'Economy_Wide': {'Residential': self.residential_projections(), 
                                      'Industrial': self.industrial_projections(),
                                      'Transportation': self.transportation_projections(),
                                      'Electricity': self.electricity_projections(),
                                      'Commerical': self.commercial_projections()}}
        return data_dict
        

    def main(self, breakout, calculate_lmdi): 
        data_dict = self.collect_data()
        results_dict, formatted_results = self.get_nested_lmdi(level_of_aggregation=self.level_of_aggregation, 
                                                               breakout=breakout, calculate_lmdi=calculate_lmdi, 
                                                               raw_data=data_dict, lmdi_type='LMDI-I')
if __name__ == '__main__': 
    indicators = MakeProjections(directory='C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020', 
                                 output_directory='./Results', 
                                 level_of_aggregation='Economy_Wide', lmdi_model=['multiplicative', 'additive']) 
    indicators.main(breakout=False, calculate_lmdi=True)


 

    
                