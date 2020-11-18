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


class MultiplicativeLMDI():

    def __init__(self, energy_data, energy_shares, base_year, total_label, lmdi_type=None):
        self.energy_data = energy_data
        self.energy_shares = energy_shares

    def log_mean_divisia_weights(self):
        """Calculate log mean weights where T = t, 0 = t-1

        Multiplicative model uses the LMDI-II model because 'the weights...sum[] to unity, a 
        desirable property in index construction.' (Ang, B.W., 2015. LMDI decomposition approach: A guide for 
                                        implementation. Energy Policy 86, 233-238.).
        """

        log_mean_weights = pd.DataFrame(index=self.energy_data.index)

        for col in self.energy_shares.columns: 
            self.energy_shares[f"{col}_shift"] = self.energy_shares[col].shift(periods=1, axis='index', fill_value=0)
            
            # apply generally not preferred for row-wise operations but?
            log_mean_weights[f'log_mean_weights_{col}'] = self.energy_shares.apply(lambda row: \
                                                          self.logarithmic_average(row[col], row[f"{col}_shift"]), axis=1) 
        
        sum_log_mean_shares = log_mean_weights.sum(axis=1)
        log_mean_weights_normalized = log_mean_weights.divide(sum_log_mean_shares.values.reshape(len(sum_log_mean_shares), 1))
        return log_mean_weights_normalized

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

    def decomposition(self, ASI):

        results = ASI.apply(lambda col: np.exp(col), axis=1)

        results['effect'] = results.product(axis=1)
        
        return results

    @staticmethod
    def lineplot(data, loa, model, energy_type, *lines_to_plot): # path
        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set2')

        for i, l in enumerate(lines_to_plot):
            label_ = l.replace("_", " ").capitalize()
            plt.plot(data.index, data[l], marker='', color=palette(i), linewidth=1, alpha=0.9, label=label_)
        
        loa = [l_.replace("_", " ") for l_ in loa]
        loa = " /".join(loa)
        title = loa + f" {model.capitalize()}" + f" {energy_type.capitalize()}" 
        plt.title(title, fontsize=12, fontweight=0)
        plt.xlabel('Year')
        # plt.ylabel('')
        plt.legend(loc=2, ncol=2)
        plt.show()
        # plt.save(f"{path}/{title}.png")