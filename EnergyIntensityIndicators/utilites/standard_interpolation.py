
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
            resid_year_before = dataframe.xs(year_before)[name_to_interp]
            resid_y_ = dataframe.xs(y_)[name_to_interp]
            increment = 1 / num_years
            for delta in range(num_years):

                value = resid_year_before * (1 - increment * delta) + \
                    resid_y_ * (increment * delta)
                year = year_before + delta
                dataframe.loc[year, name_to_interp] = value
    return dataframe