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
        self.categories_level1 = {
                            'Liming': 
                                {'source': 'EPA', 'table': 'Table 5-22'},
                            'Adipic Acid Production': 
                                {'source': 'EPA', 'table': 'Table 4-31'},
                            'Aluminum Production': 
                                {'source': 'EPA', 'table': 'Table 4-82'},
                            'Ammonia Production': 
                                {'source': 'EPA', 'table': 'Table 4-21'},
                            'Caprolactam, Glyoxal, and Glyoxylic Acid Production': 
                                {'source': 'EPA', 'table': 'Table 4-34'},
                            'Carbide Production and Consumption': 
                                {'source': 'EPA', 'table': 'Table 4-38'},
                            'Carbon Dioxide Consumption': 
                                {'source': 'EPA', 'table': 'Table 4-54'},
                            'Cement Production': 
                                {'source': 'USGS', 'table': np.nan},
                            'Coal Mining': 
                                {'source': 'EPA', 'table': 'Table 3-29'},
                            'Composting': 
                                {'source': 'EPA', 'table': 'Table 7-20'},
                            'Ferroalloy Production': 
                                {'source': 'EPA', 'table': 'Table 4-77'},
                            'Glass Production': 
                                {'source': 'EPA', 'table': 'Table 4-12'},
                            'Lead Production': 
                                {'source': 'EPA', 'table': 'Table 4-89'},
                            'Lime Production': 
                                {'source': 'EPA', 'table': ['Table 4-8', 'Table 4-9']},
                            'N2O from Product Uses': 
                                {'source': 'EPA', 'table': 'Table 4-109'},
                            'Nitric Acid Production': 
                                {'source': 'EPA', 'table': 'Table 4-28'},
                            'Other Process Uses of Carbonates': 
                                {'source': 'EPA', 'table': 'Table 4-16'},
                            'Petrochemical Production': 
                                {'source': 'EPA', 'table': 'Table 4-48'},
                            'Phosphoric Acid Production': 
                                {'source': 'EPA', 'table': ['Table 4-57', 'Table 4-58']},
                            'Soda Ash Production': 
                                {'source': 'EPA', 'table': 'Table 4-44'},
                            'Stationary Combustion': 
                                {'source': 'EPA', 'table': ['A-89', 'A-90']},
                            'Titanium Dioxide Production': 
                                {'source': 'EPA', 'table': 'Table 4-41'},
                            'Urea Consumption for NonAgricultural Purposes': 
                                {'source': 'EPA', 'table': np.nan},
                            'Urea Fertilization': 
                                {'source': 'EPA', 'table': 'Table 4-25'},
                            'Zinc Production': 
                                {'source': 'EPA', 'table': 'Table 4-92'}}
        self.categories_level2 = {'Iron and Steel Production & Metallurgical Coke Production':
                                    {'source': 'EPA', 'table': 
                                        {'Metallurgical coke': ['Table 4-67', 'Table 4-69'],
                                         'Iron and Steel': ['Table 4-72', 'Table 4-73']}},
                                  'Non-Energy Use of Fuels': 
                                    {'source': 'EPA', 'table': ['Table 3-21', 'Table 3-22']}}
        self.categories_level3 = {'Agricultural Soil Management': 
                                    {'source': 'EPA', 'table': ''},
                                  'Enteric Fermentation': 
                                    {'source': 'EPA', 'table': ''},
                                  'Landfills': 
                                    {'source': 'EPA', 'table': ''},
                                  'Manure Management': 
                                    {'source': 'EPA', 'table': ''},
                                  'Natural Gas Systems': 
                                    {'source': 'EPA', 'table': ''},
                                  'Petroleum Systems': 
                                    {'source': 'EPA', 'table': ''}}

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

    def noncombustion_activity_level1(self):
        directory = 'C:/Users/irabidea/Desktop/emissions_data/'
        data_dict = dict()
        for c, info in self.categories_level1.items():
            if info['source'] == 'EPA':
                try:
                    table_name = info['table']
                    if table_name.startswith('A-'):
                        f_path = directory + 'Annex/'
                    elif table_name.startswith('Table 4-'):
                        f_path = directory + 'Chapter Text/Ch 4 - Industrial Processes/'
                    elif table_name.startswith('Table 5-'):
                        f_path = directory + 'Chapter Text/Ch 5 - Agriculture/'

                    table = pd.read_csv(f'{f_path}{table_name}.csv'
                                        ).dropna(axis=1, how='all'
                                        ).dropna(axis=0, how='all')
                    if 'Year' not in table.columns:
                        table.columns = table.iloc[0]
                        table = table.drop(table.index[0])
                        if 'Year' not in table.columns:
                            if 'Source' in table.columns:
                                try:
                                    table = table.set_index('Source')
                                    table = table.transpose()
                                    table.index = table.index.astype('int')
                                    table.index.name = 'Year'
                                    table = table.dropna(how='all', axis=1)
                                except Exception as e:
                                    print(f'year col {table_name} failed with error', e)
                                # if 'Year' not in table.columns:
                                #     print('table missing year:\n', table)
                                #     exit()
                                #     # raise KeyError('Table missing year columns')
                        else:
                            table = table.set_index('Year')
                            table.columns.name = None
                            table.index.name = 'Year'

                except Exception as e:
                    print(f'Table {table_name} failed with error', e)
                    continue

            data_dict[c] = table

        print('data_dict:\n', data_dict)

    def noncombustion_activity_level1(self):
        self.categories_level2


if __name__ == '__main__':
    com = NonCombustion()
    # chapter_0 = com.chapter_0
    # com.unpack_noncombustion_data(chapter_0)
    # com.walk_folders('C:/Users/irabidea/Desktop/emissions_data/Chapter Text/')
    com.noncombustion_activity()
