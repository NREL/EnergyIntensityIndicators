import pandas as pd
import os
import numpy as np
import zipfile
import requests
import io
import glob


class NonCombustion:

    def __init__(self):
        self.annex = 'https://www.epa.gov/sites/production/files/2020-07/annex_1.zip'
        self.chapter_0 = 'https://www.epa.gov/sites/production/files/2020-08/chapter_0.zip'
        self.archive = "https://www.eia.gov/electricity/data/eia923/archive/xls/f906nonutil1989.zip"

    @staticmethod
    def unpack_noncombustion_data(zip_file):
        """[summary]

        Args:
            zip_file ([type]): [description]
        """        
        print('collecting noncombustion_fuels')
        r = requests.get(zip_file)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        print('zipfile')
        z.extractall('C:/Users/irabidea/Desktop/emissions_data/')
        print('zipfile collected')
    
    @staticmethod
    def noncombustion_emissions(base_dir='C:/Users/irabidea/Desktop/emissions_data'):
        """[summary]

        Args:
            base_dir (str, optional): [description]. Defaults to 'C:/Users/irabidea/Desktop/emissions_data'.
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
        """[summary]

        Args:
            directory ([type]): [description]
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


class IndustrialEmissions:

    def __init__(self):
        self.sub_categories_list = {'Industry': {',': {'Food and beverage and tobacco products': None, 'Textile mills and textile product mills': None, 
                                                                'Apparel and leather and allied products': None, 'Wood products': None, 'Paper products': None,
                                                                'Printing and related support activities': None, 'Petroleum and coal products': None, 'Chemical products': None,
                                                                'Plastics and rubber products': None, 'Nonmetallic mineral products': None, 'Primary metals': None,
                                                                'Fabricated metal products': None, 'Machinery': None, 'Computer and electronic products': None,
                                                                'Electrical equipment, appliances, and components': None, 'Motor vehicles, bodies and trailers, and parts': None,
                                                                'Furniture and related products': None, 'Miscellaneous manufacturing': None},
                                            'Nonmanufacturing': {'Agriculture, Forestry & Fishing': None,
                                                                'Mining': {'Petroleum and Natural Gas': None, 
                                                                            'Other Mining': None, 
                                                                            'Support Activities': None},
                                                                'Construction': None}}}

                                                                {'aluminum': {'noncombustion': None, 'combustion': None}}
                                                                {'noncombustion': {'aluminum': None, 'iron': None, 'magnesium': None}, 'combustion': None} # This one


if __name__ == '__main__':
    com = NonCombustion()
    chapter_0 = com.chapter_0
    com.unpack_noncombustion_data(chapter_0)
    com.walk_folders('C:/Users/irabidea/Desktop/emissions_data/Chapter Text/')
