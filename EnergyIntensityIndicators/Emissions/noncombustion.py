import pandas as pd
import os
import zipfile
import requests
import io
import glob
import numpy as np

from EnergyIntensityIndicators.utilities.dataframe_utilities \
    import DFUtilities as df_utils


class NonCombustion:
    """Class to handle and explore
    zipped Emissions data from the EPA

    TODO automate to use latest data available. Is file naming consistent?

    """
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.annex = \
            'https://www.epa.gov/sites/production/files/2020-07/annex_1.zip'
        self.chapter_0 = \
            'https://www.epa.gov/sites/production/files/2020-08/chapter_0.zip'
        self.archive = \
            "https://www.eia.gov/electricity/data/eia923/archive/xls/f906nonutil1989.zip"
        self.years = list(range(1990, 2018 + 1))
        self.categories_level1 = \
            {'Liming':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 5-22'},  # Emissions from Liming (MMT C)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 5-21'}},  # Emissions from Liming (MMT CO2 Eq.)
                'Adipic Acid Production':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-31'},  # Adipic Acid Production (kt)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-30'}},  # N2O Emissions from Adipic Acid Production (MMT CO2 Eq. and kt N2O)
                'Aluminum Production':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-82'},  # Production of Primary Aluminum (kt)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-79'}},  # CO2 Emissions from Aluminum Production (MMT CO2 Eq. and kt)
                    # PFC Emissions from Aluminum Production (MMT CO2 Eq.) TAble 4-80 ?
                'Ammonia Production':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-21'},  # Ammonia Production, Recovered CO2 Consumed for Urea Production, and Urea Production (kt)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-19'}},  # CO2 Emissions from Ammonia Production (MMT CO2 Eq.)
                'Caprolactam, Glyoxal, and Glyoxylic Acid Production':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-34'},  # Caprolactam Production (kt)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-33'}},  # N2O Emissions from Caprolactam Production (MMT CO2 Eq. and kt N2O)
                    # Table 4-35 Approach 2 Quantitative Uncertainty Estimates for N2O Emissions from Caprolactam, Glyoxal and Glyoxylic Acid Production (MMT CO2 Eq. and Percent) ?
                'Carbide Production and Consumption':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-38'},  # Production and Consumption of Silicon Carbide (Metric Tons)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-36'}},  # CO2 and CH4 Emissions from Silicon Carbide Production and Consumption (MMT CO2 Eq.)
                'Carbon Dioxide Consumption':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-54'}, # CO2 Production (kt CO2) and the Percent Used for Non-EOR Applications
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-53'}},  # CO2 Emissions from CO2 Consumption (MMT CO2 Eq. and kt)
                'Cement Production':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-4'},  # Production Thousand Tons
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-3'}},  # CO2 Emissions from Cement Production (MMT CO2 Eq. and kt)
                'Coal Mining':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 3-29'},  # Coal Production (kt)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 3-30'}},  # CH4 Emissions from Coal Mining (MMT CO2 Eq.)
                'Composting':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 7-20'},  # U.S. Waste Composted (kt)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 7-18'}},  # CH4 and N2O Emissions from Composting (MMT CO2 Eq.)
                'Ferroalloy Production':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-77'},  # Production of Ferroalloys (Metric Tons)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-75'}}, # CO2 and CH4 Emissions from Ferroalloy Production (MMT CO2 Eq.)
                'Glass Production':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-12'},  # Limestone, Dolomite, and Soda Ash Consumption Used in Glass Production (kt)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-11'}},  # CO2 Emissions from Glass Production (MMT CO2 Eq. and kt)
                'Lead Production':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-89'},  # Lead Production (Metric Tons)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-88'}},  # CO2 Emissions from Lead Production (MMT CO2 Eq. and kt)
                'Lime Production':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-9'},  # Adjusted Lime Production (kt)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-6'}},  # CO2 Emissions from Lime Production (MMT CO2 Eq. and kt)
                'N2O from Product Uses':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-109'},  # N2O Production (kt)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-110'}},  # N2O Emissions from N2O Product Usage (MMT CO2 Eq. and kt)
                'Nitric Acid Production':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-28'},  # Nitric Acid Production (kt)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-27'}},  # N2O Emissions from Nitric Acid Production (MMT CO2 Eq. and kt N2O)
                'Other Process Uses of Carbonates':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-16'},  # Limestone and Dolomite Consumption (kt)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-14'}},  # CO2 Emissions from Other Process Uses of Carbonates (MMT CO2 Eq.)
                'Petrochemical Production':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-48'},  # Production of Selected Petrochemicals (kt)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-46'}},  # CO2 and CH4 Emissions from Petrochemical Production (MMT CO2 Eq.)
                'Phosphoric Acid Production':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-57'},  # Phosphate Rock Domestic Consumption, Exports, and Imports (kt) ** Use domestic consumption
                                                #  'Table 4-58']},  # Chemical Composition of Phosphate Rock (Percent by Weight)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-56'}},  # CO2 Emissions from Phosphoric Acid Production (MMT CO2 Eq. and kt)
                'Soda Ash Production':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-44'},  # Soda Ash Production (kt)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-43'}},  # CO2 Emissions from Soda Ash Production (MMT CO2 Eq. and kt CO2)
                'Stationary Combustion':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table A-90'},  # Fuel Consumption by Stationary Combustion for Calculating CH4 and N2O Emissions (TBtu)
                            #    'Table A-91']},  # CH4 and N2O Emission Factors by Fuel Type and Sector (g/GJ)a
                    'emissions':
                    {'source': 'EPA',
                        'table': ['Table 3-10',  # CH4 Emissions from Stationary Combustion (MMT CO2 Eq.)
                                'Table 3-11']}},  # N2O Emissions from Stationary Combustion (MMT CO2 Eq.)
                'Titanium Dioxide Production':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-41'},  # Titanium Dioxide Production (kt)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-40'}},  # CO2 Emissions from Titanium Dioxide (MMT CO2 Eq. and kt)
                'Urea Consumption for NonAgricultural Purposes':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-25'},  # Urea Production, Urea Applied as Fertilizer, Urea Imports, and Urea Exports (kt) ** subtract urea applied as fertilizer
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-23'}},  # CO2 Emissions from Urea Consumption for Non-Agricultural Purposes (MMT CO2 Eq.)
                'Urea Fertilization':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-25'},  # Urea Production, Urea Applied as Fertilizer, Urea Imports, and Urea Exports (kt) 
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 5-25'}},  # CO2 Emissions from Urea Fertilization (MMT CO2 Eq.)
                'Zinc Production':
                {'activity':
                    {'source': 'EPA',
                        'table': 'Table 4-92'},  # Zinc Production (Metric Tons)
                    'emissions':
                    {'source': 'EPA',
                        'table': 'Table 4-91'}}}  # CO2 Emissions from Zinc Production (MMT CO2 Eq. and kt)
        self.categories_level2 = {
            'Iron and Steel Production & Metallurgical Coke Production': {
                'activity': {
                    'source': 'EPA',
                    'table': {
                        'Metallurgical coke': ['Table 4-67',   # Production and Consumption Data for the Calculation of CO2 Emissions from Metallurgical Coke Production (Thousand Metric Tons)
                                                'Table 4-68'],  # Material Carbon Contents for Iron and Steel Production
                        'Iron and Steel': ['Table 4-72',  # Production and Consumption Data for the Calculation of CO2 and CH4 Emissions from Iron and Steel Production (Thousand Metric Tons)
                                        'Table 4-73']}  # Production and Consumption Data for the Calculation of CO2 Emissions from Iron and Steel Production (Million ft3 unless otherwise specified)
                    }, 
                'emissions': {
                    'source': 'EPA',
                    'table': {
                        'Metallurgical coke': 'Table 4-60',  # CO2 Emissions from Metallurgical Coke Production (MMT CO2 Eq.)
                        'Iron and Steel': ['Table 4-62',  # CO2 Emissions from Iron and Steel Production (MMT CO2 Eq.)
                                           'Table 4-64']}}},  # CH4 Emissions from Iron and Steel Production (MMT CO2 Eq.)
                'Non-Energy Use of Fuels': {
                    'activity': {
                        'source': 'EPA',
                        'table': 'Table 3-21'},  # Adjusted Consumption of Fossil Fuels for Non-Energy Uses (TBtu)
                                                # 'Table 3-22']}, # 2018 Adjusted Non-Energy Use Fossil Fuel Consumption, Storage, and Emissions
                    'emissions': {
                        'source': 'EPA',
                        'table': 'Table 3-20'
                        }
                    }
            }  # CO2 Emissions from Non-Energy Use Fossil Fuel Consumption (MMT CO2 Eq. and Percent)

    def unpack_noncombustion_data(self, zip_file):
        """Unpack zipped file into folder stored locally

        Args:
            zip_file (str): URL / path to zipfile
        """
        r = requests.get(zip_file)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(self.base_dir)

        print('zipfile collected')

    @staticmethod
    def noncombustion_emissions(dir):
        """Create a dataframe (saved to base_dir) matching the filename and title
        of each csv in base_dir

        Args:
            base_dir (str): Local folder containing unzipped
                                      emissions data
        """
        files_list = glob.glob(f"{dir}*.csv")

        if files_list:
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

            tables_df = pd.DataFrame.from_dict(table_dict,
                                            orient='index',
                                            columns=['Table Name'])
            tables_df.to_csv(f'{dir}table_names.csv')

        else:
            pass

    def walk_folders(self):
        """Append file information from sub-directories
         to dataframe in directory, capturing filenames
         and titles of each csv in the subdirectories

        Args:
            directory (str): Directory containing subfolders
            of Emissions data
        """
        walk = [os.path.join(x[0], "") for x in os.walk(self.base_dir)]
        # walk = [x[0] for x in os.walk(directory)]
        # print('walk:\n', walk)
        names = []
        for w in walk:
            self.noncombustion_emissions(w)

            try:
                table_names = pd.read_csv(f'{w}/table_names.csv')

            except FileNotFoundError:
                continue

            else:
                table_names['folder'] = w
                # table_names['columns'] = '/'.join(table_names.columns.tolist())
                names.append(table_names)
            # print('table_names:\n', table_names)

        all_names = pd.concat(names)
        all_names.to_csv(
            self.base_dir + '/all_names.csv',
            index=False)

    def noncombustion_epa_data(self, table_name):
        """Collect and clean noncombustion data from
        the EPA

        Args:
            table_name (str): Table to collect noncom

        Returns:
            table (pd.DataFrame): Data for table_name
        """
        directory = self.base_dir
        print('table_name:', table_name)
        # try:
        if table_name.startswith('Table A-'):
            f_path = directory + 'Annex/'
        elif table_name.startswith('Table 1-'):
            f_path = directory + 'Chapter Text/Ch 1 - Intro/'
        elif table_name.startswith('Table 2-'):
            f_path = directory + 'Chapter Text/Ch 2 - Trends/'
        elif table_name.startswith('Table 3-'):
            f_path = directory + 'Chapter Text/Ch 3 - Energy/'
        elif table_name.startswith('Table 4-'):
            f_path = directory + \
                'Chapter Text/Ch 4 - Industrial Processes/'
        elif table_name.startswith('Table 5-'):
            f_path = directory + 'Chapter Text/Ch 5 - Agriculture/'
        elif table_name.startswith('Table 6-'):
            f_path = directory + 'Chapter Text/Ch 6 - LULUCF/'
        elif table_name.startswith('Table 7-'):
            f_path = directory + 'Chapter Text/Ch 7 - Waste/'
        elif table_name.startswith('Table ES-'):
            f_path = directory + 'Chapter Text/Executive Summary/'

        if table_name == 'Table A-199':
            table = pd.read_csv(f'{f_path}{table_name}.csv',
                                encoding='latin1', header=3,
                                index_col=0).dropna(
                                    axis=1, how='all').dropna(
                                        axis=0, how='all')
            table = table.rename(columns={'Total': 'Mineral',
                                          'Tier 1/2.1': 'Organic',
                                          'Total[1]': 'Total'})
            table = table[['Mineral', 'Organic', 'Total']]
        elif table_name == 'Table 3-29':
            table = pd.read_csv(f'{f_path}{table_name}.csv',
                                encoding='latin1', header=3,
                                index_col=1).dropna(
                                    axis=1, how='all').dropna(
                                        axis=0, how='all')
            table = table.rename(
                columns={'Number of Mines': 'Number of Underground Mines',
                         'Production': 'Underground Production',
                         'Number of Mines.1': 'Number of Surface Mines',
                         'Production.1': 'Surface Production',
                         'Number of Mines.2': 'Total Number of Mines',
                         'Production.2': 'Total Production'})
            table.index.name = 'Year'
        elif table_name == 'Table 4-54':
            total_name = \
                'Total CO2 Production from Extraction and Capture Facilities (kt)'
            new_name = 'Carbon Dioxide Consumption'
            rename_ = \
                {total_name: new_name}
            table = pd.read_csv(f'{f_path}{table_name}.csv',
                                encoding='latin1', header=2,
                                index_col=1, skipfooter=4,
                                engine='python').dropna(
                                    axis=1, how='all')
            table = table[[total_name]].rename(columns=rename_)
            table[new_name] = \
                table[new_name].apply(lambda s:
                    str(s).replace(' b', '').replace(',', ''))
            table[new_name] = table[new_name].astype(float)
            # print('table:\n', table)
            # print('table cols:\n', table.columns)
        elif table_name == 'Table A-236':
            table = pd.read_csv(f'{f_path}{table_name}.csv',
                                encoding='latin1', skiprows=1,
                                skipfooter=4, engine='python')
            table = table.set_index('Unnamed: 0')
            table = table.transpose()
            table.index.name = 'Year'
            table.index = table.index.astype(int)
            table.columns.name = None
            table = table[['Total Industrial Waste Landfilled']]

        else:
            table = pd.read_csv(f'{f_path}{table_name}.csv',
                                encoding='latin1').dropna(  # , header=2
                                    axis=1, how='all').dropna(
                                        axis=0, how='any')

            if table[table.columns[0]].str.contains('Year').any():
                year_row = 0
                # print('year_row:', year_row)
                new_header = table.iloc[year_row]
                table = table[year_row + 1:]
                table.columns = new_header

            if 'Year' in table.columns:
                table = table.set_index('Year')
                try:
                    table.index = table.index.astype(int)
                except TypeError:
                    try:
                        table.index = table.index.astype(int)
                    except TypeError:
                        table = table.transpose()
                        table.index = table.index.astype(int)
            else:
                table = table.set_index(table.columns[0])
                table = table.transpose()
                try:
                    table.index = table.index.astype(int)
                except TypeError:
                    table = table.set_index(table.columns[0])
                    try:
                        table.index = table.index.astype(int)
                    except TypeError:
                        table = table.transpose()
                        table.index = table.index.astype(int)

                    # print('table failed to set index to int:\n', table)

        table.columns.name = None
        table.index.name = 'Year'

        table = table.applymap(lambda x: self.replace_value(x))

        table = \
            table.applymap(
                lambda x: float(
                    str(x).replace("(", "-").replace(
                        ",", "").replace('%', '').rstrip(")")))

        table = table.ffill().bfill()

        return table

    @staticmethod
    def replace_value(value):
        """Replace non-numeric value

        Args:
            value (str or numeric): Value to clean

        Returns:
            new_val (numeric): cleaned value
        """
        replace_dict = {'+': 0.025, '+ ': 0.025,
                        'C': np.nan, ' NA ': np.nan,
                        'NE': np.nan, 'NO': np.nan,
                        '-': np.nan, ' -   ': np.nan,
                        '          -   ': np.nan,
                        'NA': np.nan}
        if isinstance(value, str):
            value = value.strip()

        if value in replace_dict.keys():
            new_val = replace_dict[value]
        else:
            new_val = value
        return new_val

    def noncombustion_level1(self):
        """Collect non combustion data for
        categories of 'level1' complexity

        Returns:
            data_dict (dict): Nested dictionary
                              keys are categories
                              inner dictionary are
                              variables and values
                              are respective dataframes
        """
        categories = self.categories_level1

        data_dict = dict()
        for c, var_info in categories.items():
            c_data = dict()
            for v, info in var_info.items():
                if info['source'] == 'EPA':
                    table_name = info['table']
                    if isinstance(table_name, str):
                        data = self.noncombustion_epa_data(table_name)
                    elif isinstance(table_name, list):
                        data = [self.noncombustion_epa_data(t)
                                for t in table_name]
                c_data[v] = data
            data_dict[c] = c_data

        return data_dict

    def noncombustion_level2(self):
        """Collect non combustion data for
        categories of 'level2' complexity

        - Use heat content from Emissions Hub
        - Table 4-67
            - mmBtu per short ton
            - Mixed (Industrial Coking)
            - Coal Coke
            - Coke oven gas
            - Natural gas
        - Table 4-68
            - Total energy value as single activity
              data
            - (scf is standard cubic feet)
        - Table 4-74
            - Iron and steel
                - sum of BOF steel production and
                  EAF steel production
        - NonEnergy Use of fuels
            - Total from 3-21 as activity

        Returns:
            data_dict (dict): Nested dictionary
                              keys are categories
                              inner dictionary are
                              variables and values
                              are respective dataframes
        """
        categories = self.categories_level2
        data_dict = dict()
        for c, var_info in categories.items():

            if c == \
                  'Iron and Steel Production & Metallurgical Coke Production':
                e_tables = var_info['emissions']['table']
                for s, tables in e_tables.items():
                    if s == 'Iron and Steel':
                        iron_and_steel = []
                        for t in tables:
                            table = self.noncombustion_epa_data(t)
                            table = table[['Total']]

                            if t == 'Table 4-62':
                                table = table.rename(
                                    columns={'Total':
                                             'CO2 Emissions (MMT CO2 Eq.)'})
                                # print(f'{s} emissions table {t}:\n', table)

                            # + Does not exceed 0.05 MMT CO₂ Eq.
                            elif t == 'Table 4-64':
                                table = table.rename(
                                    columns={'Total':
                                             'CH4 Emissions (MMT CO2 Eq.)'})
                                # print(f'{s} {t} table:\n', table)
                            iron_and_steel.append(table)
                        iron_and_steel_emissions = \
                            df_utils().merge_df_list(iron_and_steel)
                    elif s == 'Metallurgical coke':
                        table = self.noncombustion_epa_data(tables)
                        met_coke_emissions = table[['Total']].rename(
                            columns={'Total': 'CO2 Emissions (MMT CO2 Eq.)'})
                        # print(f'{s} emissions table {tables}:\n', table)

                a_tables = var_info['activity']['table']
                for s, tables in a_tables.items():
                    if s == 'Iron and Steel':
                        iron_and_steel_activity = []
                        for t in tables:
                            table = self.noncombustion_epa_data(t)
                            if t == 'Table 4-72':
                                table = table[['EAF Steel Production',
                                               'BOF Steel Production']]
                                table['Iron and Steel Production'] = \
                                    table.sum(axis=1)
                                table = table[['Iron and Steel Production']]
                                # print(f'{s} emissions table {t}:\n', table)
                            elif t == 'Table 4-73':
                                # heat_content =
                                print(f'{s} activity table {t}:\n', table)
                                continue
                            iron_and_steel_activity.append(table)
                            iron_and_steel_activity = df_utils().merge_df_list(
                                iron_and_steel_activity)
                    elif s == 'Metallurgical coke':
                        metallurgical_coke_activity = []
                        for t in tables:
                            table = self.noncombustion_epa_data(t)
                            metallurgical_coke_activity.append(table)

                        metallurgical_coke_activity = df_utils().merge_df_list(
                            metallurgical_coke_activity)

            elif c == 'Non-Energy Use of Fuels':
                e_table = var_info['emissions']['table']
                e_table = self.noncombustion_epa_data(e_table)

                e_table = e_table[['Emissions']].rename(
                    columns={'Emissions': 'CO2 Emissions (MMT CO2 Eq)'})
                a_table = var_info['activity']['table']
                a_table = self.noncombustion_epa_data(a_table)

                a_table = a_table[['Total']].rename(
                    columns={'Total': 'Non-Energy Use of Fuels (TBtu)'})

                data_dict['Non-Energy Use of Fuels'] = \
                    {'emissions': e_table,
                     'activity': a_table}

            metallurgical_coke = {'Metallurgical coke':
                                    {'emissions': met_coke_emissions,
                                     'activity': metallurgical_coke_activity}}
            iron_and_steel = {'Iron and Steel':
                                {'emissions': iron_and_steel_emissions,
                                 'activity': iron_and_steel_activity}}

        data_dict.update(metallurgical_coke)
        data_dict.update(iron_and_steel)

        return data_dict

    def agricultural_soil_management(self):
        """Collect noncombustion Emissions and Activity data for
        Agricultural Soil Management

        - Separate tables ( in A 3.12) into Organic vs
          Mineral Soil (these are top level categories)
        - Test that sum of data from each sub-category match
          the total hectares given in A-182
        - Aggregate emissions from each category into
          Organic vs Mineral
        - Check overall emissions sum match values published
          under that emissions source category

        Returns:
            data (dict): keys are variables and values
                         are respective dataframes
        """
        # Total Cropland and Grassland Area Estimated with Tier 1/2
        # and 3 Inventory Approaches (Million Hectares)

        # Note: The report shows this table as A-198
        organic_mineral_managed = \
            self.noncombustion_epa_data('Table A-199')
        organic_mineral_managed = organic_mineral_managed.rename(
            columns={'Total': 'Organic and Mineral Soil Managed'})

        rice_cultivation = self.noncombustion_epa_data('Table A-202')
        # Land Areas (Million Hectares)
        rice_cultivation = rice_cultivation[['Total']].rename(
            columns={'Total': 'Rice Cultivation'})

        n2o_cropland_mineral = self.noncombustion_epa_data('Table A-207')
        n2o_cropland_mineral = \
            n2o_cropland_mineral[['Total Cropland Mineral Soil Emission']]
        n2o_grassland_mineral = self.noncombustion_epa_data('Table A-208')
        n2o_grassland_mineral = \
            n2o_grassland_mineral[['Total Grassland Mineral Soil Emission']]
        # sum A-206, A-207 in report are the emissions data for organic
        # activity from table A-198
        n2o_mineral = \
            n2o_cropland_mineral.merge(n2o_grassland_mineral,
                                       how='outer', left_index=True,
                                       right_index=True)
        n2o_mineral['N2O Mineral'] = n2o_mineral.sum(axis=1)
        n2o_mineral = n2o_mineral[['N2O Mineral']]

        # Do the same thing with tables A-208 and A-209 to get
        # change in carbon stocks from organic hectares
        carbon_st_organic_crop = self.noncombustion_epa_data('Table A-209')
        carbon_st_organic_crop = \
            carbon_st_organic_crop[['Total Cropland SOC Stock Change']]
        carbon_st_organic_grass = self.noncombustion_epa_data('Table A-210')
        carbon_st_organic_grass = \
            carbon_st_organic_grass[['Total Grassland SOC Stock Change']]
        carbon_stock_organic =\
            carbon_st_organic_crop.merge(carbon_st_organic_grass,
                                         how='outer', left_index=True,
                                         right_index=True)
        carbon_stock_organic['Carbon Stock Organic'] = \
            carbon_stock_organic.sum(axis=1)   # MMT CO2 Eq.
        carbon_stock_organic = carbon_stock_organic[['Carbon Stock Organic']]

        n2o_organic = self.noncombustion_epa_data('Table A-212')
        n2o_organic = n2o_organic[['Total Organic Soil Emissions']]

        carbon_st_organic_crop_drain = \
            self.noncombustion_epa_data('Table A-214')
        carbon_st_organic_crop_drain = \
            carbon_st_organic_crop_drain[['Total Cropland SOC Stock Change']]
        carbon_st_organic_grass_drain = \
            self.noncombustion_epa_data('Table A-215')
        carbon_st_organic_grass_drain = \
            carbon_st_organic_grass_drain[['Total Grassland SOC Stock Change']]
        carbon_stock_organic_drain = carbon_st_organic_crop_drain.merge(
                                        carbon_st_organic_grass_drain,
                                        how='outer', left_index=True,
                                        right_index=True)
        carbon_stock_organic_drain['Carbon Stock Organic Drain'] = \
            carbon_stock_organic_drain.sum(axis=1)   # MMT CO2 Eq.
        carbon_stock_organic_drain = \
            carbon_stock_organic_drain[['Carbon Stock Organic Drain']]

        # (MMT CO2 Eq.)
        rice_methane = self.noncombustion_epa_data('Table A-211')
        rice_methane = rice_methane[['Total Rice Methane Emission']]

        # mineral_leaching_n2o_emissions for cropland and grassland
        # (Table A-215, Table A-216 ) do the same as above
        cropland_indirect = self.noncombustion_epa_data('Table A-216')
        cropland_indirect = \
            cropland_indirect[['Total Cropland Indirect Emissions']]
        grassland_indirect = self.noncombustion_epa_data('Table A-217')
        grassland_indirect = \
            grassland_indirect[['Total Grassland Indirect Emissions']]
        indirect_emissions = cropland_indirect.merge(
                                        grassland_indirect,
                                        how='outer', left_index=True,
                                        right_index=True)
        indirect_emissions['Indirect Emissions Mineral'] = \
            indirect_emissions.sum(axis=1)   # MMT CO2 Eq.
        indirect_emissions = indirect_emissions[['Indirect Emissions Mineral']]

        emissions_tables = [n2o_mineral, carbon_stock_organic,
                            n2o_organic, carbon_stock_organic_drain,
                            rice_methane, indirect_emissions]
        emissions = df_utils().merge_df_list(emissions_tables)
        emissions['Agricultural Soil Management'] = emissions.sum(axis=1)
        emissions = emissions[['Agricultural Soil Management']]

        organic_mineral_managed.index = \
            organic_mineral_managed.index.astype(int)
        organic_mineral_managed = \
            organic_mineral_managed.drop(
                'Organic and Mineral Soil Managed', axis=1)
        rice_cultivation.index = rice_cultivation.index.astype(int)

        activity = df_utils().merge_df_list([organic_mineral_managed,
                                             rice_cultivation])
        activity['Agricultural Soil Management'] = activity.sum(axis=1)
        activity = activity[['Agricultural Soil Management']]
        data = {'activity': activity, 'emissions': emissions}
        return data

    def enteric_fermentation(self):
        """Collect noncombustion Emissions and Activity data for
        Enteric Fermentation

        - Table A-167: Animals by Type (for Cattle only)
        - Table A-158: Calculated Annual National Emission
          Factors for Cattle by Animal Type
        - Table A-163: Emissions by animal type (cattle only)

        Returns:
            data (dict): keys are variables and values
                         are respective dataframes
        """
        # Livestock Population (1,000 Head)
        activity = self.noncombustion_epa_data('Table A-184')

        # Calculated Annual National Emission Factors for
        # Cattle by Animal Type, for 2017
        # (kg CH4/head/year), 86F[1]
        activity2 = self.noncombustion_epa_data('Table A-175')

        # CH4 Emissions from Enteric Fermentation (MMT CO2 Eq.)
        emissions = self.noncombustion_epa_data('Table A-180')
        print('emissions:\n', emissions)
        data = {'activity': activity,
                'activity2': activity2,
                'emissions': emissions}
        return data

    def landfills(self):
        """Collect noncombustion Emissions and Activity data for
        Landfills

        - Table A-221: Total MSW Landfilled + Total Industrial
          Waste Landfilled
          - Table A-228 CH4 Emissions from Landfills (kt)
        - (if time-- decompose)

        Returns:
            data (dict): keys are variables and values
                         are respective dataframes

        """
        # Solid Waste in MSW and Industrial Waste Landfills Contributing
        # to CH4 Emissions (MMT unless otherwise noted)
        activity = self.noncombustion_epa_data('Table A-236')

        # CH4 Emissions from Landfills (MMT CO2 Eq.)
        emissions = self.noncombustion_epa_data('Table 7-3')

        data = {'activity': activity, 'emissions': emissions}
        return data

    def manure_management(self):
        """Collect noncombustion Emissions and Activity data for
        Manure Management

        - Table A-167: Animals by Type (all)
        - Table A-178, A-179: Emissions by animal type (all)
          for Methane and Nitrous Oxide

        Returns:
            data (dict): keys are variables and values
                         are respective dataframes

        """
        # Livestock Population (1,000 Head)
        activity = self.noncombustion_epa_data('Table A-184')
        # Total Methane Emissions from Livestock Manure Management (kt)a
        methane = self.noncombustion_epa_data('Table A-195')
        methane = methane.multiply(25/1000)  # Convert to MMTCO2eq
        # Total (Direct and Indirect) Nitrous Oxide Emissions from
        #  Livestock Manure Management (kt)
        nitrous_oxide = self.noncombustion_epa_data('Table A-196')
        nitrous_oxide = nitrous_oxide.multiply(298/1000)  # Convert to MMTCO2eq
        utils = df_utils()
        methane, nitrous_oxide = \
            utils.ensure_same_indices(methane, nitrous_oxide)
        # Some columns are subtotals. Remove total columns to 
        # avoid double-counting.
        total_cols = ['Dairy Cattle', 'Swine', 'Beef Cattle', 'Poultry']
        for df in [methane, nitrous_oxide]:
            df.drop(total_cols, axis=1, inplace=True)

        emissions = utils.merge_df_list([methane, nitrous_oxide])
        name = 'Manure Management'
        emissions = utils.create_total_column(emissions, name)
        emissions = emissions[[name]]
        # Table 5-7 CH4 and N2O Emissions from Manure Management
        # (MMT CO2 Eq.) ?
        data = {'activity': activity, 'emissions': emissions}
        return data

    def petroleum_systems(self):
        """Collect noncombustion Emissions and Activity data for
        Petroleum Systems

        Activity is number of wells (Oil and HF Oil)

        Returns:
            data (dict): keys are variables and values
                         are respective dataframes
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

        emissions = self.noncombustion_epa_data('Table ES-4')
        emissions = emissions[['Petroleum Systems']]
        emissions.columns.name = None
        data = {'activity': petroleum_activity, 'emissions': emissions}
        return data

    def natural_gas_systems(self):
        """Collect noncombustion Emissions and Activity data for
        Natural Gas Systems

        Activity is Total Active Gas Wells

        Returns:
            data (dict): keys are variables and values
                         are respective dataframes
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

        emissions = self.noncombustion_epa_data('Table ES-4')
        emissions = emissions[['Natural Gas Systems']]
        emissions.columns.name = None
        data = {'activity': natgas, 'emissions': emissions}
        return data

    def noncombustion_level_3(self):
        """Collect non combustion data for
        categories of 'level3' complexity

        Returns:
            data_dict (dict): Nested dictionary
                              keys are categories
                              inner dictionary are
                              variables and values
                              are respective dataframes
        """
        agricultural_soil_management = self.agricultural_soil_management()
        enteric_fermentation = self.enteric_fermentation()
        landfills = self.landfills()
        manure_management = self.manure_management()
        data = {'Manure Management': manure_management,
                'Enteric Fermentation': enteric_fermentation,
                'Landfills': landfills,
                'Agricultural Soil Management': agricultural_soil_management}
        return data

    def main(self):
        """Collect all noncombustion activity and
        emissions data for the industrial sector

        data structure is as follows:
            {'noncombustion':
                {'aluminum': None,
                 'iron': None,
                 'magnesium': None},
             'combustion': None}

        Returns:
            noncombustion_data (dict): Non combustion
                                       activity and emissions
                                       data by category (nested)
        """

        for zf in [self.annex, self.chapter_0, self.archive]:
            self.unpack_noncombustion_data(zf)

        self.noncombustion_emissions(dir=self.base_dir)

        self.walk_folders()

        activity_level1 = self.noncombustion_level1()
        activity_level2 = self.noncombustion_level2()
        level3 = self.noncombustion_level_3()

        noncombustion_data = activity_level1.copy()
        noncombustion_data.update(activity_level2)
        noncombustion_data.update(level3)

        return noncombustion_data


if __name__ == '__main__':

    base_dir = 'c:/users/cmcmilla/Desktop/emissions_data/'

    com = NonCombustion(base_dir)
    com.main()
