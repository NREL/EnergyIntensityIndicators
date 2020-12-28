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

    def __init__(self, output_directory, energy_data=None, energy_shares=None, 
                 base_year=None, end_year=None, total_label=None, lmdi_type=None):
        self.energy_data = energy_data
        self.energy_shares = energy_shares
        self.base_year = base_year
        self.end_year = end_year
        self.total_label = total_label
        self.lmdi_type = lmdi_type
        self.output_directory = output_directory


    def log_mean_divisia_weights(self):
        """Calculate log mean weights where T = t, 0 = t-1

        Multiplicative model uses the LMDI-II model because 'the weights...sum[] to unity, a 
        desirable property in index construction.' (Ang, B.W., 2015. LMDI decomposition approach: A guide for 
                                        implementation. Energy Policy 86, 233-238.).
        """
        if self.energy_shares.shape[1] == 1:
            return self.energy_shares
        else:
            log_mean_weights = pd.DataFrame(index=self.energy_data.index)
            for col in self.energy_shares.columns: 
                self.energy_shares[f"{col}_shift"] = self.energy_shares[col].shift(periods=1, axis='index', fill_value=0)
                print('self.energy_shares[f"{col}_shift"]:\n', self.energy_shares[f"{col}_shift"])
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
        try:
            x = float(x)
            y = float(y)
        except TypeError:
            L = np.nan
            return L  
                 
        if x > 0 and y > 0:
            if x != y:
                difference = x - y
                log_difference = np.log(x / y)
                L = difference / log_difference
            else:
                L = x
        else: 
            L = np.nan

        return L
    
    def compute_index(self, component, base_year_):
        """Compute index of components (indexing to chosen base_year_), 
        replicating methodology in PNNL spreadsheets for the multiplicative model
        """         
        index = pd.DataFrame(index=component.index, columns=['index'])
        component = component.fillna(1)

        for y in component.index:
            if y == min(component.index):
                index.loc[y, 'index'] = 1
                print('final_index', y, index.loc[y, 'index'])
            else:
                if component.loc[y] == np.nan:
                    index.loc[y, 'index'] = index.loc[y - 1, 'index']
                    print('final_index', y, index.loc[y, 'index'])

                else:
                    index.loc[y, 'index'] = index.loc[y - 1, 'index'] * component.loc[y]
                    print('index:', y, index.loc[y - 1, 'index'])
                    print('component:', y, component.loc[y])
                    print('final_index', y,  index.loc[y, 'index'])

        print('index:\n', index)
        index_normalized = index.divide(index.loc[base_year_]) # 1985=1
        print('index_normalized:\n', index_normalized)
        return index_normalized 

    def decomposition(self, ASI):
        """Format component data, collect overall effect, return indexed 
        dataframe of the results for the multiplicative LMDI model.
        """

        print('ASI:\n', ASI)
        # ASI_df = pd.DataFrame.from_dict(data=ASI, orient='columns')
        print('columns multiplicative decomposition df', [l.columns for l in list(ASI.values())])

        ASI_df = reduce(lambda df1,df2: df1.merge(df2, how='outer', left_index=True, right_index=True), list(ASI.values()))
        print('ASI_df:\n', ASI_df)
        results = ASI_df.apply(lambda col: np.exp(col), axis=1)
        print(' log ASI_df:\n', results)

        for col in results.columns:
            results[col] = self.compute_index(results[col], self.base_year)

        print('indexed log ASI df:\n', results)
        results['effect'] = results.product(axis=1)
        print("results['effect']:\n", results['effect'])
        print('all results df:\n', results)
        results["@filter|Measure|BaseYear"] = self.base_year
        return results

    def visualizations(self, data, base_year, end_year, loa, model, energy_type, rename_dict): 
        """Visualize multiplicative LMDI results in a line plot 
        """
        data = data[(data['@timeseries|Year'] >=  base_year) & (data['@filter|Model'] == model.capitalize())]
        print('DATA TO PLOT:\n', data)

        structure_cols = []
        for column in data.columns: 
            if 'Structure' in column or 'structure' in column:
                structure_cols.append(column)

        print('structure_cols:', structure_cols)
        if len(structure_cols) == 1:
            data = data.rename(columns={structure_cols[0]: '@filter|Measure|Structure'})
        elif len(structure_cols) > 1:
            data['@filter|Measure|Structure'] = data[structure_cols].product(axis=1)  # Current level total structure
            to_drop = [s for s in structure_cols if s != '@filter|Measure|Structure']
            data = data.drop(to_drop, axis=1)
        print('Multiplicative DATA TO PRINT:\n', data)
        print('data columns', data.columns)
        
        lines_to_plot = ["@filter|Measure|Activity", "@filter|Measure|Effect"]  
        print('os.getcwd()', os.getcwd())

        if '@filter|Measure|Structure' in data.columns:
            lines_to_plot.append('@filter|Measure|Structure')
        # else: 
        #     for c in data.columns: 
        #         if c.endswith('Structure'):
        #             lines_to_plot.append(c)

        if "@filter|Measure|Intensity" in data.columns:
            lines_to_plot.append("@filter|Measure|Intensity")
        # else: 
        #     for c in data.columns: 
        #         if c.endswith('Intensity'):
        #             lines_to_plot.append(c)
                          
        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set2')

        for i, l in enumerate(lines_to_plot):
            label_ = l.replace('@filter|Measure|', '').replace("_", " ").title()
            plt.plot(data['@timeseries|Year'], data[l], marker='', color=palette(i), linewidth=1, alpha=0.9, label=label_)
        
        loa_ = [l_.replace("_", " ") for l_ in loa]
        loa = [loa[0], loa[-1]]

        title = f"Change in {energy_type.capitalize()} Energy Use {' '.join(loa)}"

        fig_name = "_".join(loa) + f"{model}_{energy_type}_{base_year}" 

        plt.title(title, fontsize=12, fontweight=0)
        plt.xlabel('Year')
        plt.ylabel('Trillion British thermal units [TBtu]')
        plt.legend(loc=2, ncol=2)
        try:
            plt.savefig(f"{self.output_directory}/{fig_name}.png")
        except FileNotFoundError:
            plt.savefig(f".{self.output_directory}/{fig_name}.png")
        plt.show()