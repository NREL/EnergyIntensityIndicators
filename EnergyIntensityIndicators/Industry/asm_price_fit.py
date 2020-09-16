import pandas as pd
import numpy as np
from scipy.optimize import leastsq


# Industry to do:
# * Energy Prices
#     * difference between ASM_EnergyPrices and ASMdata_ spreadsheets

# Should this be its own class?
class mfg_prices:

    @staticmethod
    def get_asm_prices():
        """Get fuel prices"""

        return

    @staticmethod
    def get_mecs_prices_1985_1994():
        """
        Import CSV of MECS fuel prices for 1985, 1988, 1991, and 1994 by
        3-digit NAICS code level
        """
        return


    def get_mecs_1998():
        url = 'https://www.eia.gov/consumption/manufacturing/data/1998/xls/d98e8_2.xls'
        return mecs_prices_1998


    def get_mecs_prices_1998_onwards(year):
        """download post-1998 MECS price data from Table 7.2"

        """
        if year > 2010:
            f_ex = '.xlsx'
        else:
            f_ex = '.xls'

        price_url = 'https://www.eia.gov/consumption/manufacturing/data/{}/xls/table7_2{}'.format(year, f_ex)

        price_df = pd.read_excel(price_url, sheet_name=[0], skiprows=19,
                                 usecols='A:AN')

        # There are manual assumptions made for filling in withheld data.
        # Should these simply be hard coded and the method written to raise
        # exceptions for any new MECS data that have missing values?

        return


    def get_mecs_prices_other(year):
        """Get MECS price data from Table 7.6"""
        return mecs_other_prices


    def build_price_df():
        """Build a dataframe for ASM prices and MECS prices, including
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
    def predict_prices(asm_prices, mecs_prices, start_params):
        """
        Parameters
        ----------

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

        coeff, flag = leastsq(mfg_prices.residuals, start_params,
                              args=(asm_prices, mecs_prices,
                                    mfg_prices.price_func))

        return coeff

    @staticmethod
    def calc_predicted_prices(coeff, asm_prices, price_func):
        """Calculate predicted price with leastsq coeffs"""
        return mfg_prices.price_func(asm_prices, *coeff)

    @staticmethod
    def resid_filler(residual):
        """
        Apply to a dataframe that includes year and residual.

        Parameters
        ----------
        residual : np.array
            Array of residuals, including nan values.
        """
        fill = 1
        for index, r in enumerate(residual):
            if index == 0:
                r_fill = np.array([0])
            elif index < 3:
                r_fill = np.vstack([r_fill, 0])
            else:
                if np.isnan(r):
                    r_fill = np.vstack([r_fill, fill])
                    fill += 1
                else:
                    r_fill = np.vstack([r_fill, r])
                    fill = 1

        # Code from @iisabeller
        # Define increment_years by dropping nan values in Residual series.
        # increment_years = [1970, 1974, 1980, 1984, 1987, 1990, 1993, 1997,
        #                     2001, 2005, 2009, 2015]
        #
        # for index, y_ in enumerate(increment_years):
        #     if index > 0:
        #         year_before = increment_years[index - 1]
        #         num_years = y_ - year_before
        #         difference = recs_total[y_] - recs_total[year_before]
        #         increment = difference / num_years
        #         for delta in range(num_years):
        #             value = recs_total[year_before]  + delta * increment
        #             year = year_before + delta
        #             manh_size[year] = value

        return r_fill

    @staticmethod
    def interpolate_residuals(predicted_prices, price_df):
        """Interpolate residuals"""

        price_df_updated = price_df.copy(deep=True)
        price_df_updated['predicted_price'] = predicted_prices
        price_df_updated['residual'] = price_df_updated.mecs.subtract(
            price_df_updated.predicted_price
            )

        fill = price_df_updated.dropna(subset=['residual'], axis=0).year.diff()
        # fill.fillna(0, inplace=True)

        price_df_updated['fill'] = fill


        return interpolated_resid

    @staticmethod
    def calc_calibrated_predicted_price(interpolated_resid, asm_prices):
        """
        Return the calibrated prediced prices, which is calculated as the

        """
        calibrated = interpolated_resid + asm_prices

        return calibrated
