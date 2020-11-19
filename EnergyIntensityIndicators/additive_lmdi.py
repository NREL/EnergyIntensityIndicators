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

    def __init__(self, energy_data, energy_shares, base_year, end_year, total_label, lmdi_type_='LMDI-I'):
        self.energy_data = energy_data
        self.energy_shares = energy_shares
        self.total_label = total_label
        self.lmdi_type_ = lmdi_type_
        self.end_year = end_year
        self.base_year = base_year

    def log_mean_divisia_weights(self):
        """Calculate log mean weights for the additive model where T=t, 0 = t - 1

        Args:
            energy_data (dataframe): energy consumption data
            energy_shares (dataframe): Shares of total energy for each category in level of aggregation
            total_label (str): Name of aggregation of categories in level of aggregation
            lmdi_type_ (str, optional): 'LMDI-I' or 'LMDI-II'. Defaults to 'LMDI-I' because it is 'consistent in aggregation and perfect 
                                        in decomposition at the subcategory level' (Ang, B.W., 2015. LMDI decomposition approach: A guide for 
                                        implementation. Energy Policy 86, 233-238.).
        """        
        print(f'ADDITIVE LMDI TYPE: {self.lmdi_type_}')
        log_mean_shares_labels = [f"log_mean_shares_{col}" for col in self.energy_shares.columns]
        log_mean_weights = pd.DataFrame(index=self.energy_data.index)
        log_mean_values_df = pd.DataFrame(index=self.energy_data.index)

        print('self.energy_shares.columns:', self.energy_shares.columns)
        for col in self.energy_shares.columns: 
            self.energy_data[f"{col}_shift"] = self.energy_data[col].shift(periods=1, axis='index', fill_value=0)

            # apply generally not preferred for row-wise operations but?
            log_mean_values = self.energy_data[[col, f"{col}_shift"]].apply(lambda row: 
                                                                self.logarithmic_average(row[col],
                                                                row[f"{col}_shift"]), axis=1) 

            log_mean_values_df[col] = log_mean_values.values 
            print(log_mean_values)                             

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

        if self.lmdi_type_ == 'LMDI-I':
            print('log_mean_values:', log_mean_values_df)
            return log_mean_values_df

        elif self.lmdi_type_ == 'LMDI-II':
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

        ASI['effect'] = ASI.sum(axis=1)

        return ASI

    @staticmethod
    def aggregate_additive(additive, base_year):
        additive.loc[additive['Year'] <= base_year, ['activity', 'intensity', 'structure', 'effect']] = 0
        additive = additive.set_index('Year')
        df = additive.cumsum(axis=0)
        return df

    def decomposition(self, ASI):
        """Loop through 
        """
        additive_results = []

        df = self.calculate_effect(ASI)
        df = df.reset_index()

        for year in df['Year']:
            aggregated_df = self.aggregate_additive(df, year)
            aggregated_df["@filter|Measure|BaseYear"] = year
            additive_results.append(aggregated_df)

        additive_results_df = pd.concat(additive_results, axis=0)
        return additive_results_df
    
    def visualizations(self, data, base_year, end_year, loa, model, energy_type, *x_data):

        figure_labels = []
        loa = [l.replace("_", " ") for l in loa]
        final_year = max(data['@timeseries|Year'])
        title = f"Change {base_year}-{final_year} {' '.join(loa)} {model.capitalize()}"
        # title = loa + f" {model.capitalize()}" + f" {' '.join(loa)} {energy_type.capitalize()}" 
        x_data = ['initial_energy'] + list(x_data) + ['final_energy']
        data = data[data['@timeseries|Year'] == self.end_year][x_data]
        y_data = data.ravel()
        x_labels = [x.replace("_", " ").capitalize() for x in x_data]
        
        # for example: ["relative", "relative", "total", "relative", "relative", "total"]
        measure =  ['relative'] * len(list(x_labels)) 
        fig = go.Figure(go.Waterfall(name="Change", orientation="v", measure=measure, x=x_labels, 
                                     textposition="outside", text=figure_labels, y=y_data, 
                                     connector={"line":{"color":"rgb(63, 63, 63)"}}))
                                      #  color_discrete_sequence=px.colors.qualitative.Vivid,

        fig.update_layout(title=title, showlegend = True)

        fig.show()
        # fig.save(f"{path}/{title}.png")