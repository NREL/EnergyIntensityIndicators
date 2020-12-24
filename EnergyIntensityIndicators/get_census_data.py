
import pandas as pd
import requests
import io
import numpy as np
import json
import re

class Census_api:
    """
    Class for querying Census API for a specified year.
    """

    def __init__(self):
        """
        Parameters
        ----------
        dataset : str
            Name of Census data to return.
        """

        self._params = {}

        with open('U:/API_auth.json') as jfile:
            apis = json.load(jfile)
            self.key = apis['census_API']

        self.naics1712 = pd.read_excel(
            'https://www.census.gov/eos/www/naics/concordances/' +
            '2017_to_2012_NAICS.xlsx', skiprows=[0, 1], usecols='A,C',
            names=['NAICS2017', 'NAICS2012']
            )

        self.naics0712 = pd.read_excel(
            'https://www.census.gov/eos/www/naics/concordances/' +
            '2007_to_2012_NAICS.xls', skiprows=[0, 1], usecols='A,C',
            names=['NAICS2007', 'NAICS2012']
            )

    @property
    def params(self):
        """
        API parameters.

        Returns
        -------
        dict
        """
        return self._params

    def __setitem__(self, key, value):
        """
        Set API parameters.
        """
        self._params[key] = value

    def naics_xwalk(self, year, census_df):
        """
        Method to crosswalk 2007 and 2017 NAICS with 2012 NAICS.

        Parameters
        ----------
        year : int
            Year of NAICS to be crosswalked. Either 2007 or 2017
        census_df : pandas.DataFrame
            DataFrame of data called from Census API.

        Returns
        -------
        census_df : pandas.DataFrame
            Original dataframe with NAICS codes
        """

        xwalks = {2007: self.naics0712, 2017: self.naics1712}

        try:
            xwalk_df = xwalks[year]
        except KeyError:
            print('{} is not a valid NAICS year'.format(year))

        census_df = pd.merge(census_df, xwalk_df, on='NAICS'+str(year),
                             how='outer')

        census_df.NAICS2012.update(census_df['NAICS'+str(year)])
        census_df = census_df[census_df['NAICS'+str(year)].notnull()]

        def test_mfg(naics):
            if str(naics)[0] == '3':
                return True
            else:
                return np.nan

        census_df['mfg'] = census_df.NAICS2012.apply(lambda x: test_mfg(x))
        census_df.dropna(subset=['mfg'], inplace=True)
        census_df.drop('mfg', axis=1, inplace=True)

        return census_df

    def dl_naics_titles(year, n_digits):
        """
        Download titles of manufacturing NAICS codes. Specify naics hierarchy
        with n_digits
        """

        n_digits = n_digits-1

        all_naics = \
            requests.get('http://api.naics.us/v0/q?year=%s' % str(year))

        mfg_naics = []

        for n in all_naics.json():
            try:
                if (3*10**n_digits) < int(n['code']) < (4*10**n_digits):
                    mfg_naics.append((n['code'], n['title']))
            except ValueError:
                continue

        mfg_naics = pd.DataFrame(mfg_naics, columns=['NAICS', 'NAICS_Title'])

        return mfg_naics

    @staticmethod
    def find_naics_col(census_data):
        """Locate name of column containing NAICS codes in ASM and EC"""

        naics_col_mask = [re.match(r'(NAICS)', x) for x in census_data.columns]
        naics_col_mask = [1 if x is None else 0 for x in naics_col_mask]
        naics_col = np.ma.array(census_data.columns, mask=naics_col_mask)
        naics_col = naics_col.compressed()[0]

        return naics_col

    @staticmethod
    def find_naics_year(year):
        """
        Method for setting naics year for retrieving census data based on
        dataset and year.

        Parameters
        ----------
        year : int
            Year of data to download

        Returns
        -------
        naics_year : int
            NAICS year that corresponds to year of data.
        naics_col : str
            Name of NAICS column used by census.
        """
        naics_years = {range(2017, 2019): {'year': 2017,
                                           'col_name': 'NAICS2017'},
                       range(2012, 2017): {'year': 2012,
                                           'col_name': 'NAICS2012'},
                       range(2010, 2012): {'year': 2010,
                                           'col_name': 'NAICS2010'}}

        for k in naics_years.keys():
            if year in k:
                naics_year = naics_years[k]['year']
                naics_col = naics_years[k]['col_name']

        return naics_year, naics_col

    def format_api_data(self, year, naics_year, naics_col, census_data):
        """
        Simple, final formatting of downloaded Census data. Crosswalks
        to NAICS2012, if applicable.

        Parameters
        ----------
        year : int
            Year of data being downloaded
        naics_year : int
            NAICS year corresponding to year of data
        naics_col : str
            Name of dataframe column corresponding to NAICS codes of convention
        year.
        census_data : pandas.DataFrame
            Data downloaded from Census

        Returns
        -------
        census_data : pandas.DataFrame
            Formatted data. NAICS codes mapped to 2014 vintage.
        """

        census_data = census_data[(census_data[naics_col] != '31-33') &
                                  (census_data[naics_col] != '44-45') &
                                  (census_data[naics_col] != '48-49')]
        census_data.loc[:, naics_col] = census_data[naics_col].astype(int)

        if year not in range(2012, 2017):
            census_data = self.naics_xwalk(naics_year, census_data)
            census_data.drop(naics_col, axis=1, inplace=True)

        else:
            census_data.loc[:, 'mfg'] = census_data[naics_col].apply(
                lambda x: int(str(x)[0]) == 3
                )
            census_data = census_data[census_data.mfg == True]
            census_data.drop('mfg', axis=1, inplace=True)

        census_data.reset_index(inplace=True, drop=True)

        return census_data


class Asm(Census_api):
    """Class for Annual Survey of Manufacturers"""
    def __init__(self):

        Census_api.__init__(self)

    def get_data(self, year):
        """
        Class for retrieving data from Annual Survey of Manufacturers.

        Parameters
        ----------
        year : int
            Year of data to download.

        Returns
        -------
        census_data : pandas.DataFrame

        """
        url = None

        naics_year, naics_col = self.find_naics_year(year)

        if year == 2018:
            url = 'https://api.census.gov/data/timeseries/asm/area2017'
            self._params = \
                {'get': 'RCPTOT,VALADD,CEXBLD,CEXMCH,CEXMCHA,CEXMCHC,' +
                 'CEXMCHO,CEXTOT,CSTELEC,CSTFU,CSTMTOT,EMP,PAYANN,NAICS2017',
                 'for': 'us:*', 'key': self.key}

            r = requests.get(url, params=self._params)
            census_data = pd.read_csv(
                io.StringIO(r.content[1:].decode('utf-8')), header=0
                )
        # Years ending in 2 and 7 are covered by Economic Census
        elif int(str(year)[-1]) in (2, 7):
            print('This is a Economic Census Year. Use class Econ_census')
            return

        else:
            url = 'https://www2.census.gov/programs-surveys/asm/data/' + \
                '{}/ASM_{}_31GS101_with_ann.xlsx'.format(str(year),
                                                         str(year))
            census_data = pd.read_excel(url)
            census_data.rename(
                columns={str(naics_year)+' NAICS code': naics_col},
                inplace=True
                )

            census_data = census_data[census_data.Year == year]

        census_data = self.format_api_data(year, naics_year, naics_col,
                                           census_data)

        census_data.rename(columns={
            'Cost of purchased electricity ($1,000)': 'CSTELEC',
            'Cost of purchased fuels consumed ($1,000)': 'CSTFU',
            'Total cost of materials ($1,000)': 'CSTMOT',
            'Annual payroll ($1,000)': 'PAYANN',
            'Total capital expenditures ($1,000)': 'CEXTOT',
            'Value added ($1,000)': 'VALADD'
            }, inplace=True)

        return census_data


class Econ_census(Census_api):
    """Class for Economic Census"""
    def __init__(self):
        Census_api.__init__(self)
        self.__setitem__('get',
                         'ESTAB,RCPTOT,VALADD,CEXBLD,CEXMCH,CEXMCHA,CEXMCHC,' +
                         'CEXTOT,CEXMCHO,INDLEVEL')
        self.__setitem__('for', 'us:*')
        self.__setitem__('key', self.key)

    def get_data(self, year):
        """
        Class for retrieving data from Economic Census.

        Parameters
        ----------
        year : int
            Year of data to download.

        Returns
        -------
        census_data : pandas.DataFrame
        """

        naics_year, naics_col = self.find_naics_year(year)

        if int(str(year)[-1]) not in (2, 7):
            print('This is not an Economic Census year')
            return

        else:
            url = 'https://api.census.gov/data/{}/{}'.format(str(year),
                                                             'ecnbasic')

        r_params_1 = self._params.copy()
        r_params_2 = self._params.copy()
        r_params_1[naics_col] = '*'
        r_params_2['get'] = 'CSTELEC,CSTFU,CSTMPRT,CSTMTOT,EMP,ESTAB,PAYANN,' + \
            'PCHTT,INDLEVEL'
        r_params_2[naics_col] = '*'

        def make_request(params, naics_column):
            """Makes Economic Census API request"""
            try:
                r = requests.get(url, params=params)

            except requests.exceptions.RequestException as e:
                print(e)

            else:
                if r.status_code == 200:
                    census_data = pd.DataFrame.from_records(r.json())
                    census_data.columns = census_data.iloc[0, :]
                    census_data = census_data.iloc[1:, :]
                    census_data.loc[:, 'us'] = 1
                    census_data.set_index(['us', naics_column, 'INDLEVEL'],
                                          inplace=True)
                    # census_data = pd.read_csv(
                    #     io.StringIO(r.content[1:].decode('utf-8')), header=0,
                    #     index_col=naics_col)
                else:
                    bad_request = pd.Series([r.status_code])
                    return bad_request

            return census_data

        census_data = pd.concat(
            [make_request(par, naics_col) for par in [r_params_1, r_params_2]],
            axis=1, ignore_index=False
            )

        census_data.reset_index(inplace=True)

        if len(census_data) == 2:
            print('requests exception')
            return

        else:
            census_data = self.format_api_data(year, naics_year, naics_col,
                                               census_data)

        return census_data
