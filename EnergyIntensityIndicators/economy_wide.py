from EnergyIntensityIndicators.industry import IndustrialIndicators
from EnergyIntensityIndicators.residential import ResidentialIndicators
from EnergyIntensityIndicators.commercial import CommercialIndicators
from EnergyIntensityIndicators.electricity import ElectricityIndicators
from EnergyIntensityIndicators.transportation import TransportationIndicators
from EnergyIntensityIndicators.LMDI import CalculateLMDI

class EconomyWide(CalculateLMDI):

    def __init__(self, directory, output_directory, level_of_aggregation=None, lmdi_model='multiplicative', base_year=1985, end_year=2018): 
        self.energy_types = ['elec', 'fuels', 'deliv', 'source', 'source_adj', 'source_adj2']
        self.res = ResidentialIndicators(directory=directory, level_of_aggregation=level_of_aggregation, lmdi_model=lmdi_model, base_year=base_year, end_year=end_year)
        self.comm = CommercialIndicators(directory=directory, level_of_aggregation=level_of_aggregation, lmdi_model=lmdi_model, base_year=base_year, end_year=end_year)
        self.ind = IndustrialIndicators(directory=directory, level_of_aggregation=level_of_aggregation, lmdi_model=lmdi_model, base_year=base_year, end_year=end_year)
        self.elec = ElectricityIndicators(directory=directory, level_of_aggregation=level_of_aggregation, lmdi_model=lmdi_model, base_year=base_year, end_year=end_year)
        self.trans = TransportationIndicators(directory=directory, level_of_aggregation=level_of_aggregation, lmdi_model=lmdi_model, base_year=base_year, end_year=end_year)

        self.res_cat = self.res.sub_categories_list
        self.comm_cat = self.comm.sub_categories_list
        self.ind_cat = self.ind.sub_categories_list
        self.elec_cat = self.elec.sub_categories_list
        self.trans_cat = self.trans.sub_categories_list

        self.sub_categories_list = {'Residential': self.res_cat, 'Commercial': self.comm_cat, 'Industrial': self.ind_cat, 'Transporation': self.trans_cat, 'Elec Power': self.elec_cat}
        super().__init__(sector='economy_wide', level_of_aggregation=level_of_aggregation, lmdi_models=lmdi_model, categories_dict=self.sub_categories_list, \
                         energy_types=self.energy_types, directory=directory, output_directory=output_directory, base_year=base_year)

    def collect_data(self):
        all_data = dict()
        abbrevs = {'Residential': self.res, 'Commercial': self.comm, 'Industrial': self.ind, 'Transporation': self.trans, 'Elec Power': self.elec}
        for sector in self.sub_categories_list.keys():
            abbrev = abbrevs[sector]
            formatted_data = abbrev.collect_data()
            all_data[sector] = formatted_data
        return all_data

    def main(self, breakout, save_breakout, calculate_lmdi):
        """TODO: allow for different sectors to have different types of energy and commercial and residential to have weather adjustment

        """        
        data_dict = self.collect_data()
        results_dict, formatted_results = self.get_nested_lmdi(level_of_aggregation=self.level_of_aggregation, breakout=breakout, save_breakout=save_breakout, calculate_lmdi=calculate_lmdi, raw_data=data_dict)
        return results 

if __name__ == '__main__':
    indicators = EconomyWide(directory='C:/Users/irabidea/Desktop/Indicators_Spreadsheets_2020', output_directory='C:/Users/irabidea/Desktop/LMDI_Results', level_of_aggregation='All_Residential')
    indicators.main(breakout=False, save_breakout=False, calculate_lmdi=False)  