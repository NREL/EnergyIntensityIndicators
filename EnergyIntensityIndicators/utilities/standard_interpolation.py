
def standard_interpolation(dataframe, name_to_interp=None, axis=1):
    """Interpolate data by splitting the difference between 
    increment years over intermediate years

    Args:
        dataframe (df): dataset to interpolate
        name_to_interp (str): name of columm if dataframe has
                              year index
        axis (int or str): if 1 or "columns" interpolate
                            over column, if 0 or "index",
                            interpolate over row


    Returns:
        dataframe [df]: dataframe with
                        interpolated data
    """
    print('dataframe:\n', dataframe)
    print('name_to_interp:', name_to_interp)
    if axis == 1 or axis == 'columns':
        if name_to_interp:
            increment_years = list(dataframe[[name_to_interp]].dropna().index)
        else:
            raise AttributeError('standard_interpolation method missing name_to_interp')

    elif axis == 0 or axis == 'index':
        increment_years = list(dataframe.dropna(axis=1, how='all').columns)

    else:
        raise AttributeError(f'standard_interpolation method missing valid axis, given {axis}')

    for index, y_ in enumerate(increment_years):

        if index > 0:
            year_before = increment_years[index - 1]
            num_years = y_ - year_before
            print('num_years:', num_years)
            print('num_years:', type(num_years))

            resid_year_before = dataframe.xs(year_before)[name_to_interp]
            print('resid_year_before:\n', resid_year_before)
            print('resid_year_before:\n', type(resid_year_before))

            resid_y_ = dataframe.xs(y_)[name_to_interp]
            print('resid_y_:\n', resid_y_)
            print('resid_y_:\n', type(resid_y_))

            increment = 1 / num_years
            for delta in range(num_years):
                print('delta:\n', delta)
                print('delta:\n', type(delta))

                value = resid_year_before * (1 - increment * delta) + \
                    resid_y_ * (increment * delta)
                year = year_before + delta
                dataframe.loc[year, name_to_interp] = value
    return dataframe