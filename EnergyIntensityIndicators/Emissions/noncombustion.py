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
                                {'source': 'EPA', 'table': ['Table 4-57',
                                                            'Table 4-58']},
                            'Soda Ash Production': 
                                {'source': 'EPA', 'table': 'Table 4-44'},
                            'Stationary Combustion': 
                                {'source': 'EPA', 'table': ['Table A-89',
                                                            'Table A-90']},
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

            table = pd.read_csv(f'{f_path}{table_name}.csv',
                                encoding='latin1',
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
                            print(f'year col {table_name} failed with error {e}')

                else:
                    table = table.set_index('Year')
                    table.columns.name = None
                    table.index.name = 'Year'

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
        print('categories:', categories)
        data_dict = dict()
        for c, info in categories.items():
            print('category:', c)
            if info['source'] == 'EPA':
                tables = info['table']
                if isinstance(tables, dict):
                    data = dict()
                    for s, table_names in tables.items():
                        print('subcategory:', s)
                        print('table_names:', table_names)
                        if isinstance(table_names, list):
                            tables_list = [self.noncombustion_activity_epa(t)
                                           for t in table_names]
                            print('tables_list:\n', tables_list)
                            data[s] = tables_list
                elif isinstance(tables, list):
                    print('tables:', tables)
                    data = [self.noncombustion_activity_epa(t)
                            for t in tables]
                    print('data:\n', data)

            data_dict[c] = data
            print(data_dict)

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
        # activity = self.noncombustion_activity_epa('Table A-167')
        # print('activity:\n', activity)
        # emissions = self.noncombustion_activity_epa('Table A-178')
        # print('emissions:\n', emissions)
        # return {'activity': activity, 'emissions': emissions}
        return None

    def enteric_fermentation(self):
        """
        - Table A-167: Animals by Type (for Cattle only)
        - Table A-158: Calculated Annual National Emission
          Factors for Cattle by Animal Type
        - Table A-163: Emissions by animal type (cattle only)
        """
        activity = self.noncombustion_activity_epa('Table A-167')
        print('activity:\n', activity)
        activity2 = self.noncombustion_activity_epa('Table A-158')
        print('activity2:\n', activity2)

        emissions = self.noncombustion_activity_epa('Table A-163')
        print('emissions:\n', emissions)
        return {'activity': activity, 'emissions': emissions}

    def landfills(self):
        """
        - Table A-221: Total MSW Landfilled + Total Industrial
          Waste Landfilled
        - (if time-- decompose)
        """
        activity = self.noncombustion_activity_epa('Table A-221')
        print('activity:\n', activity)
        # emissions = self.noncombustion_activity_epa('Table A-178')
        # print('emissions:\n', emissions)
        # return {'activity': activity, 'emissions': emissions}
        return activity

    def manure_management(self):
        """
        - Table A-167: Animals by Type (all)
        - Table A-178, A-179: Emissions by animal type (all) 
          for Methane and Nitrous Oxide
        """
        activity = self.noncombustion_activity_epa('Table A-167')
        print('activity:\n', activity)
        emissions = self.noncombustion_activity_epa('Table A-178')
        emissions = emissions.set_index('Cattle Type ').transpose()
        emissions.index.name = 'Year'
        print('emissions:\n', emissions)
        return {'activity': activity, 'emissions': emissions}

    def noncombustion_activity_level_3(self):
        """[summary]
        """        
        # agricultural_soil_management = self.agricultural_soil_management()
        enteric_fermentation = self.enteric_fermentation()
        # landfills = self.landfills()
        manure_management = self.manure_management()
        pass

    def main(self):
        # activity_level1 = self.noncombustion_activity_level1()
        # activity_level2 = self.noncombustion_activity_level2()
        activity_level3 = self.noncombustion_activity_level_3()


if __name__ == '__main__':
    com = NonCombustion()
    # chapter_0 = com.chapter_0
    # com.unpack_noncombustion_data(chapter_0)
    # com.walk_folders('C:/Users/irabidea/Desktop/emissions_data/Chapter Text/')
    com.main()
