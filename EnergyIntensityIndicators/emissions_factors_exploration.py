import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn

from EnergyIntensityIndicators.pull_eia_api import GetEIAData
from EnergyIntensityIndicators.utilites import dataframe_utilities as df_utils

class EmissionsDataExploration:

    def __init__(self):
        self.eia = GetEIAData('emissions')

    def eia_data(self, id_, label):
        data = self.eia.eia_api(id_=id_, id_type='series')
        col = list(data.columns)[0]
        data = data.rename(columns={col: label}) #, inplace=True)

        return data

    def all_fuels_data(self):
        commercial_co2 = 'EMISS.CO2-TOTV-CC-TO-US.A'
        electric_power_co2 = 'EMISS.CO2-TOTV-EC-TO-US.A'
        industrial_co2 = 'EMISS.CO2-TOTV-IC-TO-US.A'
        residential_co2 = 'EMISS.CO2-TOTV-RC-TO-US.A'
        transportation_co2 = 'EMISS.CO2-TOTV-TC-TO-US.A'
        sectors = {'commercial_co2': commercial_co2, 
                   'electric_power_co2': electric_power_co2, 
                   'industrial_co2': industrial_co2, 
                   'residential_co2': residential_co2, 
                   'transportation_co2': transportation_co2}

        all_data = {s: self.eia_data(id_=sectors[s], label='CO2 Emissions') \
                    for s in sectors}
        return all_data
    
    def all_fuels_all_sector_data(self):
        all_fuels_all_sector = 'EMISS.CO2-TOTV-TT-TO-US.A'
        all_sector = self.eia_data(id_=all_fuels_all_sector, 
                                   label='CO2 Emissions')
        return all_sector

    @staticmethod
    def lineplot(datasets, y_label):


        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set2')

        for i, label in enumerate(datasets.keys()):
            df = datasets[label]

            plt.plot(df.index, df, 
                     marker='', color=palette(i), linewidth=1, 
                     alpha=0.9, label=label)

        title = 'CO2 Emissions from All Fuels by Sector'
        plt.title(title, fontsize=12, fontweight=0)
        plt.xlabel('Year')
        plt.ylabel(y_label)
        plt.legend(loc=2, ncol=2)
        plt.show()

    def get_emissions_plots(self):
        sectors = self.all_fuels_data()
        total = self.all_fuels_all_sector_data()
        # sectors['total'] = total
        self.lineplot(sectors, y_label='CO2 Emissions (Million Metric Tons)')
    
    def get_emissions_factors_plots(self):
        emissions = self.all_fuels_data()
        energy = self.economy_wide()
        sectors = ['commercial', 'industrial', 'residential',
                   'transportation', 'electric_power']
        emissions_factors = dict()
        for s in sectors:
            em = emissions[f'{s}_co2']
            en = energy[f'{s}_energy']
            em, en = df_utils.ensure_same_indices(em, en)
            factor = em.divide(en.values, axis=1)
            factor = factor.rename(columns={'CO2 Emissions': 'Million Metric Tons per Trillion Btu'})
            emissions_factors[s] = factor
        self.lineplot(emissions_factors, y_label='Million Metric Tons CO2 per Trillion Btu')

    def economy_wide(self):
        commercial_energy = 'TOTAL.TECCBUS.A'
        electric_power_energy = 'TOTAL.TXEIBUS.A'
        industrial_energy = 'TOTAL.TEICBUS.A'
        residential_energy = 'TOTAL.TERCBUS.A'
        transportation_energy = 'TOTAL.TEACBUS.A'
        sectors = {'commercial_energy': commercial_energy, 
                   'electric_power_energy': electric_power_energy, 
                   'industrial_energy': industrial_energy, 
                   'residential_energy': residential_energy, 
                   'transportation_energy': transportation_energy}

        all_data = {s: self.eia_data(id_=sectors[s], label='Energy') \
                    for s in sectors}
        return all_data

if __name__ == '__main__':
    em = EmissionsDataExploration()
    d = em.all_fuels_data()
    print(d)
    em.get_emissions_plots()
    economy_wide_data = em.economy_wide()
    print('economy_wide_data:\n', economy_wide_data)
    em.get_emissions_factors_plots()