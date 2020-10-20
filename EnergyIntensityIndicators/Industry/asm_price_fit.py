import pandas as pd
import numpy as np
import requests
from scipy.optimize import leastsq
from bs4 import BeautifulSoup
from get_census_data import Asm
from get_census_data import Econ_census


class Mfg_prices:
    # def __ini__(self):
    #     """
    #     Class for importing and interpolating historical energy prices for
    #     the manufacturing sector.
    #
    #     Historical Manufacturing Energy Consumption Survey (MECS) data are
    #     based on prior work from the Pacific Northwest National Laboratory
    #     (PNNL).
    #     """
    #     # Historical MECS data with missing observations estimated
    #     # ad-hoc by PNNL.
    #     self.mecs_historical_prices = pd.read_csv('mecs_historical_prices.csv')

    @staticmethod
    def import_mecs_historical(file_path):
        """
        Import and format csv of historical fuel prices from Manufacturing
        Energy Consumption Survey (MECS). MECS was conducted every three
        years from 1985 - 1994 and every four years since.

        Parameters
        ----------
        file_path : str
            File path of csv.

        Returns
        -------
        mecs_prices : dataframe

        """

        mecs_prices = pd.read_csv(file_path)

        mecs_prices = pd.melt(mecs_prices, id_vars=['NAICS', 'fuel'],
                              var_name='year', value_name='mecs_price')

        return mecs_prices

    @staticmethod
    def check_recent_mecs(latest_year, last_historical_year):
        """
        MECS has been conducted at four-year intervals since 1994. New data
        will need to be manually downloaded, formatted, and added to the
        historical price file, 'mfg_mecs_energy_prices.csv'.

        Parameters
        ----------
        latest_year : int
            Latest year of historical decomposition.

        last_historical_year : int
            Lastest year of price data available in
            'mfg_mecs_energy_prices.csv'.

        Returns
        -------

        """

        if latest_year <= last_historical_year:

            raise Exception("Historical MECS prices are latest available")

            return

        else:

            check_url = 'https://www.eia.gov/consumption/manufacturing/' + \
                        'data/{}/xls/table7_2.xlsx'.format(str(latest_year))

            r = requests.get(check_url)

            soup = BeautifulSoup(r.text, 'html.parser')

            # Check if updated MECS are available by requesting Excel file.
            # If no Excel file exists, requests will return the
            # Consumption and Efficiency page. If the Excel file does exist,
            # there will be an exception, which prompts the user to download
            # and format the data.
            try:
                title = soup.title.name

            except AttributeError:
                print("Updated mecs data are now available.\n" +
                      "Please download Table 7.2 and Table 7.6\nand update" +
                      "'mfg_mecs_energy_prices.csv'")

            else:
                print("Updated MECS data are not yet available")

            finally:
                return

    @staticmethod
    def import_asm_historical(file_path):
        """"
        Prices in $/MMBtu
        """


    @staticmethod
    def get_census_prices(latest_year, start_year=1983):
        """
        Get fuel prices from Census Bureau's Annual Survey of
        Manufacturers and Economic Census (years ending in 2 and 7)

        Parameters
        ----------
        latest_year : int
            Most recent year of historical LMDI analysis.

        start_year : int, default 1983
            Beginning year of price data.

        Returns
        -------
        asm_prices : pandas.Series
            Pandas series of ASM price data from YYYY - latest_year
        """
        year_range = range(start_year, latest_year + 1)

        asm_prices = pd.DataFrame()



        return asm_prices

    @staticmethod
    def get_latest_mecs(latest_year):
        """
        Deterime status of most recent MECS price data. MECS data are released
        on a quadrennial schedule, with 2014 as the last available year
        (as of October 2020).

        """
        if (latest_year - 2014)/4 < 1:
            latest_mecs_year = 2014
        # Test if latest mecs is available.
        r = requests.get('https://www.eia.gov/consumption/manufacturing/' +
                         'data/{}/pdf/table7_2.pdf'.format(latest_mecs_year))
        print(r.raise_for_status())

        # alse need to get MECS price data from Table 7.6"""

        return

    @staticmethod
    def build_price_df(asm_prices, mecs_prices):
        """
        Build a dataframe for ASM prices and MECS prices, including
        a column for year
        """

        return price_df

    @staticmethod
    def price_func(asm_prices, *params):
        """
        Calculates predicted fuel price in terms of current and lagged fuel
        prices.
        """
        a, b = params

        for index_, price in enumerate(asm_prices):
            if index_ == 0:
                predicted_price_series = np.array([0])
            else:
                predicted_price = a*price + b*asm_prices[index_-1]
                predicted_price_series = np.vstack([predicted_price_series,
                                                    predicted_price])

        early_mecs = predicted_price_series.flatten()[2:12:3]
        later_mecs = predicted_price_series.flatten()[15:len(asm_prices):4]
        final_mecs = np.append(early_mecs, later_mecs)
        # Align with available MECS
        final_mecs = final_mecs[[x[0] for x in enumerate(mecs_prices)]]
        return final_mecs

    @staticmethod
    def residuals(params, asm_prices, mecs_prices, price_func):

        return mecs_prices - price_func(asm_prices, *params)

    @staticmethod
    def calc_predicted_coeffs(asm_prices, mecs_prices, start_params):
        """
        Parameters
        ----------
        asm_prices : numpy.array
            Array of ASM prices

        mecs_prices : numpy.array
            Array of MECS prices, including nan values

        start_params : list
            starting parameters for optimization

        Returns
        -------
        coeff : np.array
            parameters of price prediction equation

        Examples
        --------
        @calmc successully ran this using data from the Gas-311 tab (
        copying cells C6:E42 to clipboard) in Ind_Gas_Prices_123019.xlsx.
        >>>price_data = pd.read_clipboard()
        >>># Using read_clipboard introduces a column of nans
        >>>price_data.dropna(axis=1, how='all', inplace=True)
        >>>price_data.columns = ['year', 'asm_prices', 'mecs_prices']
        >>>price_data.head()
           year  asm_prices  mecs_prices
        0  1983        4.05          NaN
        1  1984        4.10          NaN
        2  1985        3.83         4.21
        3  1986        3.14          NaN
        4  1987        2.85          NaN

        >>>asm_prices = price_data.asm_prices.values
        >>>mecs_prices = price_data.mecs_prices.dropna().values
        >>>coeff = predict_prices(asm_prices, mecs_prices, [1, 1])
        >>>coeff
        array([0.91430597, 0.1074011 ])
        """

        coeff, flag = leastsq(Mfg_prices.residuals, start_params,
                              args=(asm_prices, mecs_prices,
                                    Mfg_prices.price_func))

        return coeff

    @staticmethod
    def calc_predicted_prices(coeff, asm_prices):
        """Calculate predicted price with leastsq coeffs"""
        return Mfg_prices.price_func(asm_prices, *coeff)

    @staticmethod
    def resid_filler(price_df):
        """
        Apply to a dataframe that includes year and residual.

        price_df.columns = ['year', 'asm_price', 'MECS_price', 'predicted',
        'residual']

        Parameters
        ----------
        residual : np.array
            Array of residuals, including nan values.
        """

        increment_years = price_df.dropna()['year'].values

        # Lines 165 - 173 should be placed in separate method.
        predicted_prices = Mfg_prices.calc_predicted_prices(coeff, asm_prices)
        predicted_prices = pd.DataFrame(predicted_prices,
                                        columns=['predicted'],
                                        index=increment_years)

        price_df.set_index('year', inplace=True)
        price_df = pd.concat([price_df, predicted_prices], axis=1)
        price_df['residual'] = price_df.MECS_price - price_df.predicted

        price_df['interp_resid'] = np.nan

        for index, y_ in enumerate(increment_years):
            if index > 0:
                year_before = increment_years[index - 1]
                num_years = y_ - year_before
                resid_year_before = price_df.xs(year_before)['residual']
                resid_y_ = price_df.xs(y_)['residual']
                increment = 1 / num_years
                for delta in range(num_years):
                    value = resid_year_before * (1 - increment * delta) + \
                        resid_y_ * (increment * delta)
                    year = year_before + delta
                    price_df.loc[year, 'interp_resid'] = value

        # Fill in remaining missing values
        price_df.loc[increment_years[-1], 'interp_resid'] = \
            price_df.xs(increment_years[-1])['residual']
        price_df.interp_resid.fillna(method='ffill', inplace=True)

        price_df['calibrated_prediction'] = \
            price_df.predicted + price_df.interp_resid

        price_df.reset_index(inplace=True)

        return price_df


    def calc_calibrated_predicted_price(self, latest_year, fuel_type, naics):
        """
        Return the calibrated prediced prices, which is calculated as the

        """
        # Get asm prices from separate method?

        mecs_prices = self.mecs_historical_prices(fuel_type, naics)

        fit_coeffs = Mfg_prices.calc_predicted_coeffs()
        predicted = Mfg_prices.calc_predicted_prices()



        price_df['calibrated_prediction'] = \
            price_df.interp_resid + price_df.predicted

        return price_df

    # @staticmethod
    # def interpolate_residuals(predicted_prices, price_df):
    #     """Interpolate residuals"""
    #
    #     price_df_updated = price_df.copy(deep=True)
    #     price_df_updated['predicted_price'] = predicted_prices
    #     price_df_updated['residual'] = price_df_updated.mecs.subtract(
    #         price_df_updated.predicted_price
    #         )
    #
    #     fill = price_df_updated.dropna(subset=['residual'], axis=0).year.diff()
    #     # fill.fillna(0, inplace=True)
    #
    #     price_df_updated['fill'] = fill
    #
    #
    #     return interpolated_resid
    #

    # Predicted prices are use where? Check documentation to figure out [Search NAICS]
