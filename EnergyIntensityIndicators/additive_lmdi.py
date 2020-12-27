import pandas as pd
import numpy as np
from sklearn import linear_model
from functools import reduce
import os
from datetime import date
import matplotlib.pyplot as plt
import seaborn
import plotly.graph_objects as go
import plotly.express as px


class AdditiveLMDI():

    def __init__(self, output_directory, energy_data, energy_shares,
                 base_year, end_year, total_label, lmdi_type='LMDI-I'):

        self.energy_data = energy_data
        self.energy_shares = energy_shares
        self.total_label = total_label
        self.lmdi_type = lmdi_type
        self.end_year = end_year
        self.base_year = base_year
        self.output_directory = output_directory

    def log_mean_divisia_weights(self):
        """Calculate log mean weights for the additive model where T=t, 0 = t - 1

        Args:
            energy_data (dataframe): energy consumption data
            energy_shares (dataframe): Shares of total energy for each category in level of aggregation
            total_label (str): Name of aggregation of categories in level of aggregation
            lmdi_type (str, optional): 'LMDI-I' or 'LMDI-II'. Defaults to 'LMDI-I' because it is 'consistent in aggregation and perfect 
                                        in decomposition at the subcategory level' (Ang, B.W., 2015. LMDI decomposition approach: A guide for 
                                        implementation. Energy Policy 86, 233-238.).
        """        
        print(f'ADDITIVE LMDI TYPE: {self.lmdi_type}')
        if not self.lmdi_type:
            self.lmdi_type = 'LMDI-I'
        
        print(f'ADDITIVE LMDI TYPE: {self.lmdi_type}')

        log_mean_shares_labels = [f"log_mean_shares_{col}" for col in self.energy_shares.columns]
        log_mean_weights = pd.DataFrame(index=self.energy_data.index)
        log_mean_values_df = pd.DataFrame(index=self.energy_data.index)

        for col in self.energy_shares.columns: 
            self.energy_data[f"{col}_shift"] = self.energy_data[col].shift(periods=1, axis='index', fill_value=0)

            # apply generally not preferred for row-wise operations but?
            log_mean_values = self.energy_data[[col, f"{col}_shift"]].apply(lambda row: 
                                                                self.logarithmic_average(row[col],
                                                                row[f"{col}_shift"]), axis=1) 

            log_mean_values_df[col] = log_mean_values.values 

            self.energy_shares[f"{col}_shift"] = self.energy_shares[col].shift(periods=1, axis='index', fill_value=0)
            # apply generally not preferred for row-wise operations but?
            log_mean_shares = self.energy_shares[[col, f"{col}_shift"]].apply(lambda row: 
                                                                self.logarithmic_average(row[col], \
                                                                        row[f"{col}_shift"]), axis=1)
            self.energy_shares[f"log_mean_shares_{col}"] = log_mean_shares

            log_mean_weights[f'log_mean_weights_{col}'] = log_mean_shares * log_mean_values
        

        cols_to_drop1 = [col for col in self.energy_shares.columns if col.startswith('log_mean_shares_')]
        self.energy_shares = self.energy_shares.drop(cols_to_drop1, axis=1)

        cols_to_drop = [col for col in self.energy_shares.columns if col.endswith('_shift')]
        self.energy_shares = self.energy_shares.drop(cols_to_drop, axis=1)

        cols_to_drop_ = [col for col in self.energy_data.columns if col.endswith('_shift')]
        self.energy_data = self.energy_data.drop(cols_to_drop_, axis=1)

        if self.lmdi_type == 'LMDI-I':
            return log_mean_values_df

        elif self.lmdi_type == 'LMDI-II':
            sum_log_mean_shares = self.energy_shares[log_mean_shares_labels].sum(axis=1)
            log_mean_weights_normalized = log_mean_weights.divide(sum_log_mean_shares.values.reshape(len(sum_log_mean_shares), 1))

            log_mean_weights_normalized = log_mean_weights_normalized.drop([c for c in log_mean_weights_normalized.columns \
                                                                            if not c.startswith('log_mean_weights_')], axis=1)
            return log_mean_weights_normalized
            
        else:
            return log_mean_values_df
    
    @staticmethod
    def logarithmic_average(x, y):
        """The logarithmic average of two positive numbers x and y
        """        
        try:
            x = float(x)
            y = float(y)
        except TypeError:
            L = np.nan
            return L       

        if x > 0 and y > 0:
            if x != y:
                difference = x - y
                log_difference = np.log(x) - np.log(y)
                L = difference / log_difference
            else:
                L = x
        else: 
            L = np.nan

        return L

    def calculate_effect(self, ASI):
        """Calculate effect from changes to activity, structure, 
        and intensity in the additive model
        """

        ASI['effect'] = ASI.sum(axis=1)

        return ASI

    @staticmethod
    def aggregate_additive(additive, base_year):
        """Aggregate additive data (allows for loop through every year as a base year, if desired)"""
        print('additive:\n', additive)
        cols = [c for c in list(additive.columns) if c != 'Year']
        additive.loc[additive['Year'] <= base_year, cols] = 0
        additive = additive.set_index('Year')
        df = additive.cumsum(axis=0)
        return df

    def decomposition(self, ASI):
        """Format component data, collect overall effect, return aggregated 
        dataframe of the results for the additive LMDI model.
        """
        ASI.pop('lower_level_structure', None)

        ASI_df = reduce(lambda df1,df2: df1.merge(df2, how='outer', left_index=True, right_index=True), list(ASI.values()))


        df = self.calculate_effect(ASI_df)
        df = df.reset_index()
        if 'Year' not in df.columns:
            df = df.rename(columns={'index': 'Year'})

        aggregated_df = self.aggregate_additive(df, self.base_year)
        aggregated_df["@filter|Measure|BaseYear"] = self.base_year

        return aggregated_df
    
    def visualizations(self, data, base_year, end_year, loa, model, energy_type, rename_dict):
        """Visualize additive LMDI results in a waterfall chart, opens in internet browsers and
        user must save manually (from plotly save button)
        """
        data = data[data['@filter|Model'] == model.capitalize()]

        x_data = ["@filter|Measure|Activity"]
        print('os.getcwd()', os.getcwd())

        if '@filter|Measure|Structure' in data.columns:
            x_data.append('@filter|Measure|Structure')
        else: 
            for c in data.columns: 
                if c.endswith('Structure'):
                    x_data.append(c)

        if "@filter|Measure|Intensity" in data.columns:
            x_data.append("@filter|Measure|Intensity")
        else: 
            for c in data.columns: 
                if c.endswith('Intensity'):
                    x_data.append(c)

        loa = [l.replace("_", " ") for l in loa]
        loa = [loa[0], loa[-1]]
        final_year = max(data['@timeseries|Year'])

        data_base = data[data['@timeseries|Year'] == base_year][x_data]
        data_base['intial_energy'] = self.energy_data.loc[base_year, self.total_label]

        data = data[data['@timeseries|Year'] == end_year][x_data]
        if self.end_year in self.energy_data.index:
            data['final_energy'] = self.energy_data.loc[end_year, self.total_label]
        else:
            data['final_energy'] = self.energy_data.loc[max(self.energy_data.index), self.total_label]

        x_data = ['intial_energy'] + x_data + ['final_energy']
        y_data = pd.concat([data_base, data], ignore_index=True, axis=0).fillna(0)
        y_data = y_data[x_data]

        y_data.loc[:, 'final_energy'] = 0
        print('additive data to plot:\n', y_data)
        y_data = y_data.sum(axis=0).values.tolist()
        y_labels = [self.format_y_vals(y) for y in y_data]

        print('additive data to plot:\n', y_data)

        x_labels = ['intial_energy', 'activity', 'intensity', 'structure', 'final_energy']
        x_labels = [x.replace("_", " ").capitalize() for x in x_labels]
        
        measure = ['relative'] * 4 + ['total']

        fig = go.Figure(go.Waterfall(name="Change", orientation="v", measure=measure, x=x_labels, 
                                     textposition="outside", y=y_data, text=y_labels,
                                     connector={"line":{"color":"rgb(63, 63, 63)"}}))
        
        title = f"Change in {energy_type.capitalize()} Energy Use (Trillion British thermal units [TBtu]) {base_year}-{final_year} {' '.join(loa)}"
        # title = loa + f" {model.capitalize()}" + f" {' '.join(loa)} {energy_type.capitalize()}" 
        fig.update_layout(title=title, showlegend = True)
        fig.show()
        # fig.write_image(f"{self.output_directory}/{title}.png")

    @staticmethod
    def format_y_vals(y):
        """Format y values into appropriate string 
        labels for use in waterfall chart"""

        if y > 0:
            sign = "+"
        elif y < 0:
            sign = "-"
        else:
            return "Total"
        
        label = sign + str(round(y, 2))
        return label