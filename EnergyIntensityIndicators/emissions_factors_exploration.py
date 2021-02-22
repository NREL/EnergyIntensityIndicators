import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn

from EnergyIntensityIndicators.pull_eia_api import GetEIAData


class EmissionsDataExploration:

    def __init__(self):
        self.eia = GetEIAData('emissions')

    def eia_data(self, id_, label):
        print('id_:\n', id_)
        data = self.eia.eia_api(id_=id_, id_type='series')
        print('data:\n', data)
        col = list(data.columns)[0]
        print('col:', col)
        print('data[col]', data[[col]])
        data = data.rename(columns={col: label}) #, inplace=True)
        print('data:\n', data)

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
        for s in sectors:
            print('s:', s)
            print('sectors[s]:\n', sectors[s])

        all_data = {s: self.eia_data(id_=sectors[s], label='CO2 Emissions') \
                    for s in sectors}
        return all_data
    
    def all_fuels_all_sector_data(self):
        all_fuels_all_sector = 'EMISS.CO2-TOTV-TT-TO-US.A'
        all_sector = self.eia_data(id_=all_fuels_all_sector, 
                                   label='CO2 Emissions')
        return all_sector

    @staticmethod
    def lineplot(datasets):


        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set2')

        for i, label in enumerate(datasets.keys()):
            print('i', i)
            df = datasets[label]
            print('df:\n', df)

            plt.plot(df.index, df, 
                     marker='', color=palette(i), linewidth=1, 
                     alpha=0.9, label=label)

        title = 'CO2 Emissions from All Fuels by Sector'
        plt.title(title, fontsize=12, fontweight=0)
        plt.xlabel('Year')
        plt.ylabel('CO2 Emissions (Million Metric Tons')
        plt.legend(loc=2, ncol=2)
        plt.show()

    def get_emissions_plots(self):
        sectors = self.all_fuels_data()
        total = self.all_fuels_all_sector_data()
        # sectors['total'] = total
        self.lineplot(sectors)

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
        for s in sectors:
            print('s:', s)
            print('sectors[s]:\n', sectors[s])

        all_data = {s: self.eia_data(id_=sectors[s], label='Energy') \
                    for s in sectors}
        return all_data

if __name__ == '__main__':
    em = EmissionsDataExploration()
    d = em.all_fuels_data()
    print(d)
    em.get_plots()
    economy_wide_data = em.economy_wide()
    print('economy_wide_data:\n', economy_wide_data)
