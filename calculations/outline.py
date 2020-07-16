import pandas as pd
from sklearn import linear_model

class LMDIMultiplicative:
    index_base_year_primary = 1985
	index_base_year_secondary = 1996  # not used
	charts_starting_year = 1985
	charts_ending_year = 2003 

	def __init__(self, df, categories_list):
		self.dataset = df 
        self.categories_list = categories_list

    def load_energy_data(self):
        pass

    def caculate_energy_shares(self):
        """"sum row, calculate each as percentage of total"""
        self.dataset['Energy_Consumption_Total'] = self.dataset[[self.categories_list]].sum(axis=0, skipna=True)
        for category in self.categories_list:
            self.dataset[f'{category}_energy_shares'] = self.dataset.apply(lamba x: x[category] / x['Total'])
        return self.dataset

    def calculate_log_mean_weights(self):
        for i in self.categories_list:
            for i in self.dataset.index: 
                if i >= 1:
                    self.dataset.loc[i, f'{category}_log_mean_divisia_weights'] = self.dataset.loc[i, f'{category}_energy_shares'] - self.dataset.loc[i-1, f'{category}_energy_shares'] / (self.dataset.loc[i, f'{category}_energy_shares'] / (math.log(self.dataset.loc[i-1, f'{category}_energy_shares'])))
                else:
                    self.dataset.loc[i, f'{category}_log_mean_divisia_weights'] = 0
        self.dataset['Log-Mean Divisia Weights Total'] = self.dataset[[self.categories_list]].sum(axis=0, skipna=True)
        return self.dataset

    def calculate_log_mean_weights_normalized(self):
        for category in self.categories_list:
            for i in self.dataset.index:
                if i >= 1:
                    self.dataset.loc[i, f'{category}_log_mean_divisia_weights_normalized'] = self.dataset.loc[i, f'{category}_log_mean_divisia_weights'] / 

    ///////////////////////////////////////////////////
    def load_activity_data(self):
        pass

    def calculate_activity_shares(self):
        """"sum row, calculate each as percentage of total"""
        self.dataset['Activity_Total'] = self.dataset[[self.categories_list]].sum(axis=0, skipna=True)
        for category in self.categories_list:
            self.dataset[f'{category}_energy_shares'] = self.dataset.apply(lamba x: x[category] / x['Total'])
        return self.dataset

    def calculate_log_changes_activity_shares(self, df_name, df):                                                                        
        for i in df.index: 
            for category in self.categories_list:
                df.at[i, f'{categories_list}_df_name_activities_share'] = df.loc

    def compute_structure_index(self):
        self.index_chg = 
        self.index = 
        self.index_normalized =  # 1985=1
    ///////////////////////////////////////////////////

    def calculate_energy_intensity_nominal(self, ):
        """input: base_year"""
        pass

    def calculate_log_changes_intensity(self, ):
        pass

    def compute_intensity_index(self, ):
        pass

    ///////////////////////////////////////////////////
    def activity_index(self, ):
        pass

    def index_of_aggregate_intensity(self, ):
        pass

    def structure_index(self, ):
        pass

    def component_intensity_index(self, ):
        pass


class LMDIAdditive:
    index_base_year_primary = 1985
	index_base_year_secondary = 1996  # not used
	charts_starting_year = 1985
	charts_ending_year = 2003 

	def __init__(self, df, ):
		self.dataset = df 

    def load_energy_data(self, ):
        pass

    def caculate_energy_shares(self, ):
        """"sum row, calculate each as percentage of total"""
        pass

    def calculate_log_mean_weights(self, ):
        pass

    def calculate_log_mean_weights_normalized(self, ):
        pass

    ///////////////////////////////////////////////////
    def load_activity_data(self, ):
        pass

    def calculate_activity_shares((self, ):
        pass

    def calculate_log_changes_activity_shares((self, ):
        pass

    def compute_structure_index((self, ):
        pass

    ///////////////////////////////////////////////////

    def calculate_energy_intensity_nominal((self, ):
        """Input: base_year""""
        pass

    def calculate_log_changes_intensity((self, ):
        pass

    def compute_intensity_index((self, ):
        pass

    ///////////////////////////////////////////////////

    def activity_index((self, ):
        pass

    def index_of_aggregate_intensity((self, ):
        pass

    def structure_index((self, ):
        pass

    def component_intensity_index((self, ):
        pass


def sum_row(self):
