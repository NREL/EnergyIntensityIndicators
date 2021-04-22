import pandas as pd
import os
import numpy as np
import zipfile
import requests
import io
import glob


class NonCombustion:
    """Class to handle and explore
    zipped Emissions data from the EPA
    """
    def __init__(self):
        self.annex = 'https://www.epa.gov/sites/production/files/2020-07/annex_1.zip'
        self.chapter_0 = 'https://www.epa.gov/sites/production/files/2020-08/chapter_0.zip'
        self.archive = "https://www.eia.gov/electricity/data/eia923/archive/xls/f906nonutil1989.zip"
        self.years = list(range(1990, 2018 + 1))
        self.categories_level1 = {
                            'Liming': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 5-22'}, # Table 5-22
                                 'emissions': 
                                    {'source': 'EPA', 'table': 'Table 5-21'}},  # Emissions from Liming (MMT CO2 Eq.)
                            'Adipic Acid Production': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-31'},  # Adipic Acid Production (kt)
                                 'emissions': 
                                    {'source': 'EPA', 'table': 'Table 4-30'}},  # N2O Emissions from Adipic Acid Production (MMT CO2 Eq. and kt N2O)
                            'Aluminum Production': 
                                {'activity':
                                    {'source': 'EPA', 'table': 'Table 4-82'},  # Production of Primary Aluminum (kt)
                                 'emissions': 
                                    {'source': 'EPA', 'table': 'Table 4-79'}}  # CO2 Emissions from Aluminum Production (MMT CO2 Eq. and kt)
                                    # PFC Emissions from Aluminum Production (MMT CO2 Eq.) TAble 4-80 ?
                            'Ammonia Production': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-21'},  # Ammonia Production, Recovered CO2 Consumed for Urea Production, and Urea Production (kt)
                                 'emissions': {'source': 'EPA', 'table': 'Table 4-19'}},  # CO2 Emissions from Ammonia Production (MMT CO2 Eq.)
                            'Caprolactam, Glyoxal, and Glyoxylic Acid Production': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-34'},  # Caprolactam Production (kt)
                                 'emissions': 
                                    {'source': 'EPA', 'table': 'Table 4-33'}},  # N2O Emissions from Caprolactam Production (MMT CO2 Eq. and kt N2O)
                                    # Table 4-35 Approach 2 Quantitative Uncertainty Estimates for N2O Emissions from Caprolactam, Glyoxal and Glyoxylic Acid Production (MMT CO2 Eq. and Percent) ?
                            'Carbide Production and Consumption': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-38'}, # Production and Consumption of Silicon Carbide (Metric Tons)
                                 'emissions': 
                                    {'source': 'EPA', 'table': 'Table 4-36'}},  # CO2 and CH4 Emissions from Silicon Carbide Production and Consumption (MMT CO2 Eq.)
                            'Carbon Dioxide Consumption': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-54'}, # CO2 Production (kt CO2) and the Percent Used for Non-EOR Applications
                                 'emissions': 
                                    {'source': 'EPA', 'table': 'Table 4-53'}},  # CO2 Emissions from CO2 Consumption (MMT CO2 Eq. and kt)
                            'Cement Production': 
                                {'activity':
                                    {'source': 'USGS', 'table': np.nan},
                                 'emissions': 
                                    {'source': 'EPA', 'table': 'Table 4-3'}},  # CO2 Emissions from Cement Production (MMT CO2 Eq. and kt)
                            'Coal Mining': 
                                {'activity':
                                    {'source': 'EPA', 'table': 'Table 3-29'},  # Coal Production (kt)
                                 'emissions': 
                                    {'source': 'EPA', 'table': 'Table 3-30'}},  # CH4 Emissions from Coal Mining (MMT CO2 Eq.)
                            'Composting': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 7-20'},  # U.S. Waste Composted (kt)
                                 'emissions':
                                    {'source': 'EPA', 'table': 'Table 7-18'}},  # CH4 and N2O Emissions from Composting (MMT CO2 Eq.)
                            'Ferroalloy Production': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-77'},  # Production of Ferroalloys (Metric Tons)
                                 'emissions':
                                    {'source': 'EPA', 'table': 'Table 4-75'}}, # CO2 and CH4 Emissions from Ferroalloy Production (MMT CO2 Eq.)
                            'Glass Production': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-12'},  # Limestone, Dolomite, and Soda Ash Consumption Used in Glass Production (kt)
                                 'emissions':
                                    {'source': 'EPA', 'table': 'Table 4-11'}},  # CO2 Emissions from Glass Production (MMT CO2 Eq. and kt)
                            'Lead Production': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-89'},  # Lead Production (Metric Tons)
                                 'emissions':
                                    {'source': 'EPA', 'table': 'Table 4-88'}},  # CO2 Emissions from Lead Production (MMT CO2 Eq. and kt)
                            'Lime Production': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-9'},  # Adjusted Lime Production (kt)
                                 'emissions':
                                    {'source': 'EPA', 'table': 'Table 4-6'}},  # CO2 Emissions from Lime Production (MMT CO2 Eq. and kt)
                            'N2O from Product Uses': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-109'},  # N2O Production (kt)
                                 'emissions':
                                    {'source': 'EPA', 'table': 'Table 4-110'}},  # N2O Emissions from N2O Product Usage (MMT CO2 Eq. and kt)
                            'Nitric Acid Production': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-28'},  # Nitric Acid Production (kt)
                                 'emissions':
                                    {'source': 'EPA', 'table': 'Table 4-27'}},  # N2O Emissions from Nitric Acid Production (MMT CO2 Eq. and kt N2O)
                            'Other Process Uses of Carbonates': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-16'},  # Limestone and Dolomite Consumption (kt)
                                 'emissions':
                                    {'source': 'EPA', 'table': 'Table 4-14'}}, # CO2 Emissions from Other Process Uses of Carbonates (MMT CO2 Eq.)
                            'Petrochemical Production': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-48'},  # Production of Selected Petrochemicals (kt)
                                 'emissions':
                                    {'source': 'EPA', 'table': 'Table 4-46'}},  # CO2 and CH4 Emissions from Petrochemical Production (MMT CO2 Eq.)
                            'Phosphoric Acid Production': 
                                {'activity': {'source': 'EPA', 'table': 
                                                                ['Table 4-57',  # Phosphate Rock Domestic Consumption, Exports, and Imports (kt)
                                                                 'Table 4-58']},  # Chemical Composition of Phosphate Rock (Percent by Weight)
                                 'emissions':
                                    {'source': 'EPA', 'table': 'Table 4-56'}},  # CO2 Emissions from Phosphoric Acid Production (MMT CO2 Eq. and kt)
                            'Soda Ash Production': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-44'},  # Soda Ash Production (kt)
                                 'emissions':
                                    {'source': 'EPA', 'table': 'Table 4-43'}},  # CO2 Emissions from Soda Ash Production (MMT CO2 Eq. and kt CO2)
                            'Stationary Combustion': 
                                {'source': 'EPA', 'table': ['Table A-89',
                                                            'Table A-90']},
                            'Titanium Dioxide Production': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-41'},  # Titanium Dioxide Production (kt)
                                 'emissions':
                                    {'source': 'EPA', 'table': 'Table 4-40'}},  # CO2 Emissions from Titanium Dioxide (MMT CO2 Eq. and kt)
                            'Urea Consumption for NonAgricultural Purposes': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-25'},  # Urea Production, Urea Applied as Fertilizer, Urea Imports, and Urea Exports (kt) ** subtract urea applied as fertilizer
                                 'emissions':
                                    {'source': 'EPA', 'table': 'Table 4-23'}},  # CO2 Emissions from Urea Consumption for Non-Agricultural Purposes (MMT CO2 Eq.)
                            'Urea Fertilization': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-25'},  # Urea Production, Urea Applied as Fertilizer, Urea Imports, and Urea Exports (kt) 
                                 'emissions':
                                    {'source': 'EPA', 'table': 'Table 5-25'}},  # CO2 Emissions from Urea Fertilization (MMT CO2 Eq.)
                            'Zinc Production': 
                                {'activity': 
                                    {'source': 'EPA', 'table': 'Table 4-92'},  # Zinc Production (Metric Tons)
                                 'emissions':
                                    {'source': 'EPA', 'table': 'Table 4-91'}},  # CO2 Emissions from Zinc Production (MMT CO2 Eq. and kt)
        self.categories_level2 = {'Iron and Steel Production & Metallurgical Coke Production':
                                   {'activity': {'source': 'EPA', 'table': 
                                        {'Metallurgical coke': ['Table 4-67',   # Production and Consumption Data for the Calculation of CO2 Emissions from Metallurgical Coke Production (Thousand Metric Tons)
                                                                'Table 4-69'],  # Material Carbon Contents for Iron and Steel Production
                                         'Iron and Steel': ['Table 4-72',  # Production and Consumption Data for the Calculation of CO2 and CH4 Emissions from Iron and Steel Production (Thousand Metric Tons)
                                                            'Table 4-73']}},  # Production and Consumption Data for the Calculation of CO2 Emissions from Iron and Steel Production (Million ft3 unless otherwise specified)
                                    'emissions': {'source': 'EPA', 'table': 
                                        {'Metallurgical coke': 'Table 4-60',  # CO2 Emissions from Metallurgical Coke Production (MMT CO2 Eq.)
                                         'Iron and Steel': ['Table 4-62',  # CO2 Emissions from Iron and Steel Production (MMT CO2 Eq.)
                                                            'Table 4-64']}}},  # CH4 Emissions from Iron and Steel Production (MMT CO2 Eq.)
                                  'Non-Energy Use of Fuels': 
                                    {'activity': 
                                        {'source': 'EPA', 'table': ['Table 3-21',  # Adjusted Consumption of Fossil Fuels for Non-Energy Uses (TBtu)
                                                                    'Table 3-22']}, # 2018 Adjusted Non-Energy Use Fossil Fuel Consumption, Storage, and Emissions
                                        # Table 3-21 Adjusted Consumption of Fossil Fuels for Non-Energy Uses (TBtu) ?
                                    'emissions': 
                                        {'source': 'EPA', 'table': 'Table 3-20'}}}  # CO2 Emissions from Non-Energy Use Fossil Fuel Consumption (MMT CO2 Eq. and Percent)
    @staticmethod
    def unpack_noncombustion_data(zip_file):
        """Unpack zipped file into folder stored locally

        Args:
            zip_file (str): URL / path to zipfile
        """        
        print('collecting noncombustion_fuels')
        r = requests.get(zip_file)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        print('zipfile')
        z.extractall('C:/Users/irabidea/Desktop/emissions_data/')
        print('zipfile collected')
    
    @staticmethod
    def noncombustion_emissions(base_dir='C:/Users/irabidea/Desktop/emissions_data'):
        """Create a dataframe (saved to base_dir) matching the filename and title 
        of each csv in base_dir

        Args:
            base_dir (str, optional): Local folder containing unzipped emissions data
                                      Defaults to 'C:/Users/irabidea/Desktop/emissions_data'.
        """        
        files_list = glob.glob(f"{base_dir}/*.csv")

        data = dict()
        for f in files_list:
            f = f.replace('\\', '/')
            df = pd.read_csv(f, engine='python')
            data_key = df.columns[0]
            data[data_key] = df

        table_dict = dict()
        for k in data.keys():
            split_ = k.split(':')
            try:
                table_num = split_[0].strip()
                table_name = split_[1].strip()
                table_dict[table_num] = table_name
            except IndexError as e:
                print('e:', e)
                print('split_', split_)
                pass

        print('tables_dict:\n', table_dict)
        tables_df = pd.DataFrame.from_dict(table_dict, orient='index', columns=['Table Name'])
        tables_df.to_csv(f'{base_dir}/table_names.csv')
        print('tables_df:\n', tables_df)

    def walk_folders(self, directory):
        """Append file information from sub-directories
         to dataframe in directory, capturing filenames and 
         titles of each csv in the subdirectories


        Args:
            directory (str): Directory containing subfolders
            of Emissions data
        """        
        walk = [x[0] for x in os.walk(directory)]
        print('walk:\n', walk)
        names = []
        for w in walk:
            # self.noncombustion_emissions(base_dir=w)
            table_names = pd.read_csv(f'{w}/table_names.csv')
            table_names['folder'] = w
            # table_names['columns'] = '/'.join(table_names.columns.tolist())
            names.append(table_names)
            print('table_names:\n', table_names)
        
        all_names = pd.concat(names)
        all_names.to_csv('C:/Users/irabidea/Desktop/emissions_data/all_names.csv', index=False)

    def noncombustion_activity_epa(self, table_name):
        directory = 'C:/Users/irabidea/Desktop/emissions_data/'

        try:
            if table_name.startswith('Table A-'):
                f_path = directory + 'Annex/'
            elif table_name.startswith('Table 1-'):
                f_path = directory + 'Chapter Text/Ch 1 - Intro/'
            elif table_name.startswith('Table 2-'):
                f_path = directory + 'Chapter Text/Ch 2 - Trends/'
            elif table_name.startswith('Table 3-'):
                f_path = directory + 'Chapter Text/Ch 3 - Energy/'
            elif table_name.startswith('Table 4-'):
                f_path = directory + 'Chapter Text/Ch 4 - Industrial Processes/'
            elif table_name.startswith('Table 5-'):
                f_path = directory + 'Chapter Text/Ch 5 - Agriculture/'
            elif table_name.startswith('Table 6-'):
                f_path = directory + 'Chapter Text/Ch 6 - LULUCF/'
            elif table_name.startswith('Table 7-'):
                f_path = directory + 'Chapter Text/Ch 7 - Waste/'
            elif table_name.startswith('Table ES-'):
                f_path = directory + 'Chapter Text/Executive Summary/'

            table = pd.read_csv(f'{f_path}{table_name}.csv',
                                encoding='latin1',
                                ).dropna(axis=1, how='all'
                                ).dropna(axis=0, how='all')
            if 'Year' not in table.columns:
                table.columns = table.iloc[0]
                table = table.drop(table.index[0])
                
                if 'Year' not in table.columns:
                    years = [str(y) for y in self.years]
                    year_cols = [c for c in table.columns
                                 if c in years]
                    if len(year_cols) == 0:
                        years = self.years
                    not_year_cols = [c for c in table.columns
                                     if c not in years]

                    if len(not_year_cols) == 1:
                        table = table.set_index(not_year_cols[0])
                    elif len(not_year_cols) > 1:
                        table = table.set_index(not_year_cols)
                    try:
                        table = table.transpose()
                        table.index = table.index.astype('int')
                        table.index.name = 'Year'
                        table = table.dropna(how='all', axis=1)
                    except Exception as e:
                        print(f'year col {table_name} failed with error {e}')

                else:
                    table = table.set_index('Year')
                    table.columns.name = None
                    table.index.name = 'Year'
            if table.empty:
                t = pd.read_csv(f'{f_path}{table_name}.csv',
                                encoding='latin1',
                                ).dropna(axis=1, how='all'
                                ).dropna(axis=0, how='all')
                print('t columns:', t.columns)
                print('not_year_cols:', not_year_cols)
                print('t:\n', t)
            # else:
            #     print('table:\n', table)
            #     print('table cols:\n', table.columns)

            return table

        except Exception as e:
            print(f'Table {table_name} failed with error', e)
            return None

    def noncombustion_activity_level1(self):
        categories = self.categories_level1

        data_dict = dict()
        for c, info in categories.items():
            if info['source'] == 'EPA':
                table_name = info['table']
                if isinstance(table_name, str):
                    data = self.noncombustion_activity_epa(table_name)
                elif isinstance(table_name, list):
                    data = [self.noncombustion_activity_epa(t)
                            for t in table_name]
            data_dict[c] = data

        return data_dict

    def noncombustion_activity_level2(self):
        categories = self.categories_level2
        data_dict = dict()
        for c, info in categories.items():
            if info['source'] == 'EPA':
                tables = info['table']
                if isinstance(tables, dict):
                    data = dict()
                    for s, table_names in tables.items():
                        if isinstance(table_names, list):
                            tables_list = [self.noncombustion_activity_epa(t)
                                           for t in table_names]
                            data[s] = tables_list
                elif isinstance(tables, list):
                    data = [self.noncombustion_activity_epa(t)
                            for t in tables]

            data_dict[c] = data

        return data_dict

    def agricultural_soil_management(self):
        """
        - Separate tables ( in A 3.12) into Organic vs
          Mineral Soil (these are top level categories)
        - Test that sum of data from each sub-category match
          the total hectares given in A-182
        - Aggregate emissions from each category into
          Organic vs Mineral
        - Check overall emissions sum match values published
          under that emissions source category
        """
        print('start agricultural_soil_management')
        # Total Cropland and Grassland Area Estimated with Tier 1/2
        # and 3 Inventory Approaches (Million Hectares)
        activity = self.noncombustion_activity_epa('Table A-199')
        for t in range(200, )
        'Table A-185: Total Rice Harvested Area Estimated with Tier 1 and 3 Inventory Approaches (Million Hectares)'
        'Table A-186: Sources of Soil Nitrogen (kt N)'
        'Table A-187: U.S. Soil Groupings Based on the IPCC Categories and Dominant Taxonomic Soil, and Reference 2 Carbon Stocks (Metric Tons C/ha)'
        '1 Table A-188: Soil Organic Carbon Stock Change Factors for the United States and the IPCC Default Values 2 Associated with Management Impacts on Mineral Soils'
        '190-195', '197-200', 
        mineral_tables = []
        organic_tables = []
        emissions = self.noncombustion_activity_epa('Table A-178')
        return {'activity': activity, 'emissions': emissions}

    def enteric_fermentation(self):
        """
        - Table A-167: Animals by Type (for Cattle only)
        - Table A-158: Calculated Annual National Emission
          Factors for Cattle by Animal Type
        - Table A-163: Emissions by animal type (cattle only)
        """
        print('start enteric_fermentation')
        # Livestock Population (1,000 Head)
        activity = self.noncombustion_activity_epa('Table A-184')
        print('activity:\n', activity)

        # Calculated Annual National Emission Factors for Cattle by 
        # Animal Type, for 2017 (kg CH4/head/year), 86F[1]
        activity2 = self.noncombustion_activity_epa('Table A-175')
        print('activity2:\n', activity2)

        # CH4 Emissions from Enteric Fermentation (MMT CO2 Eq.)
        emissions = self.noncombustion_activity_epa('Table A-180')
        print('emissions:\n', emissions)

        # exit()
        return {'activity': activity,
                'activity2': activity2,
                'emissions': emissions}

    def landfills(self):
        """
        - Table A-221: Total MSW Landfilled + Total Industrial
          Waste Landfilled
          - Table A-228 CH4 Emissions from Landfills (kt)
        - (if time-- decompose)
        """
        print('start landfills')
        # Solid Waste in MSW and Industrial Waste Landfills Contributing
        # to CH4 Emissions (MMT unless otherwise noted)
        activity = self.noncombustion_activity_epa('Table A-236')
        print('activity:\n', activity)

        # CH4 Emissions from Landfills (MMT CO2 Eq.)
        emissions = self.noncombustion_activity_epa('Table 7-3')
        print('emissions:\n', emissions)

        return {'activity': activity, 'emissions': emissions}

    def manure_management(self):
        """
        - Table A-167: Animals by Type (all)
        - Table A-178, A-179: Emissions by animal type (all) 
          for Methane and Nitrous Oxide
        """
        print('start manure_management')
        # Livestock Population (1,000 Head)
        activity = self.noncombustion_activity_epa('Table A-184')
        print('activity:\n', activity)
        # Total Methane Emissions from Livestock Manure Management (kt)a
        methane = self.noncombustion_activity_epa('Table A-195')
        print('methane:\n', methane)
        # Total (Direct and Indirect) Nitrous Oxide Emissions from
        #  Livestock Manure Management (kt)
        nitrous_oxide = self.noncombustion_activity_epa('Table A-196')
        print('nitrous_oxide:\n', nitrous_oxide)

        # return {'activity': activity, 'emissions': emissions}

    def petroleum_systems(self):
        """Activity is number of wells (Oil and HF Oil)
        """
        link = 'https://www.epa.gov/sites/production/files/2020-02/2020_ghgi_petroleum_systems_annex35_tables.xlsx'
        sheet = '3.5-5'
        petroleum = pd.read_excel(link, sheet_name=sheet,
                                  skiprows=6)
        petroleum = petroleum.dropna(thresh=3)
        petroleum = petroleum.set_index('Segment/Source')
        petroleum = petroleum.loc[['Total Oil Wells', 'Total HF Oil Wells'], :]
        petroleum = petroleum.drop('Activity Units', axis=1)
        petroleum.loc['Petroleum Systems', :] = petroleum.sum(axis=0)

        petroleum_activity = petroleum.transpose()[['Petroleum Systems']]
        petroleum_activity.columns.name = None

        emissions = self.noncombustion_activity_epa('Table ES-4')
        emissions = emissions[['Petroleum Systems']]
        emissions.columns.name = None

        return {'activity': petroleum_activity, 'emissions': emissions}

    def natural_gas_systems(self):
        """Activity is Total Active Gas Wells
        """
        link = 'https://www.epa.gov/sites/production/files/2020-02/2020_ghgi_natural_gas_systems_annex36_tables.xlsx'
        sheet = '3.6-7'
        natgas = pd.read_excel(link, sheet_name=sheet,
                               skiprows=5)
        natgas = natgas.dropna(thresh=3)
        natgas = natgas.set_index('Segment/Source')

        natgas = natgas.xs('Total Active Gas Wells').to_frame()
        natgas.index.name = 'Year'
        natgas = natgas.rename(columns={'Total Active Gas Wells':
                                        'Natural Gas Systems'})

        emissions = self.noncombustion_activity_epa('Table ES-4')
        emissions = emissions[['Natural Gas Systems']]
        emissions.columns.name = None

        return {'activity': natgas, 'emissions': emissions}

    def noncombustion_level_3(self):
        """[summary]
        """
        agricultural_soil_management = self.agricultural_soil_management()
        enteric_fermentation = self.enteric_fermentation()
        landfills = self.landfills()
        manure_management = self.manure_management()
        return {'manure_management': manure_management,
                'enteric_fermentation': enteric_fermentation,
                'landfills': landfills,
                'agricultural_soil_management': agricultural_soil_management}

    def main(self):
        # activity_level1 = self.noncombustion_activity_level1()
        # activity_level2 = self.noncombustion_activity_level2()
        # level3 = self.noncombustion_level_3()
        results = self.landfills()
        print('results:\n', results)

        results = self.enteric_fermentation()
        print('results:\n', results)
        results = self.manure_management()
        print('results:\n', results)
if __name__ == '__main__':
    com = NonCombustion()
    # chapter_0 = com.chapter_0
    # com.unpack_noncombustion_data(chapter_0)
    # com.walk_folders('C:/Users/irabidea/Desktop/emissions_data/Chapter Text/')
    com.main()
