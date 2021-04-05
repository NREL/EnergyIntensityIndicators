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


if __name__ == '__main__':
    com = NonCombustion()
    chapter_0 = com.chapter_0
    com.unpack_noncombustion_data(chapter_0)
    com.walk_folders('C:/Users/irabidea/Desktop/emissions_data/Chapter Text/')
