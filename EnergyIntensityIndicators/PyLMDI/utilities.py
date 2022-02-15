import pandas as pd


def get_nested_paths(dictionary, t=tuple(), reform={}):
    """Builds paths to endpoint dataframes
    from nested dictionary

    Parameters
    ----------
    dictionary : dict
        nested dictionary to reform
    t : tuple
        tuple to use for key
    reform : dict, optional
        reformed placeholder dictionary

    Returns
    -------
    paths : list[list]
        List of lists of paths from nested
        dictionary

    """
    for key, val in dictionary.items():
        t = t + (key,)
        if isinstance(val, dict):
            get_nested_paths(val, t, reform)
        else:
            reform.update({t: val})
        t = t[:-1]
    paths = [list(p) for p in reform.keys()]
    return paths


def reform_dict(dictionary, t=tuple(), reform={}):
    """Builds dictionary with tuples as keys
    from nested dictionary

    Parameters
    ----------
    dictionary : dict
        nested dictionary to reform
    t : tuple
        tuple to use for key
    reform : dict, optional
        reformed dictionary

    Returns
    -------
    reform : dict
        final reformed dictionary
    """

    for key, val in dictionary.items():
        t = t + (key,)
        v = val
        if isinstance(v, dict):
            reform_dict(v, t, reform)
        elif isinstance(v, pd.DataFrame):
            v = {k: v[k].values for k in v.columns}
            reform_dict(v, t, reform)
        else:
            reform.update({t: v})
        t = t[:-1]
    return reform


def get_years(dictionary, years=set()):
    """Gets years in the dataframes within the nested
    dictionary

    Parameters
    ----------
    dictionary : dict
        nested dictionary to extract years from
    years : set, optional
        set of years to return, by default set()

    Returns
    -------
    years : set
        set of years to return
    """
    for _, val in dictionary.items():
        if isinstance(val, dict):
            get_years(val, years)
        elif isinstance(val, pd.DataFrame):
            val = val.to_dict()
            val = val[list(val.keys())[0]]
            val = val.keys()
            for v in val:
                years.add(v)
    return years


def convert_nested_dict_to_multi(dictionary):
    """Convert nasty nested dictionary with
    dataframe endpoint values to a multiIndex

    Parameters
    ----------
    dictionary : dict
        Nested dictionary with dataFrame endpoint values

    Returns
    -------
    df : pd.MultiIndex
        MultiIndex DataFrame converted from nested dictionary
    """
    reformed = reform_dict(dictionary)
    df = pd.DataFrame.from_dict(reformed, orient='index').transpose()
    columns = pd.MultiIndex.from_tuples(list(reformed.keys()))
    years = get_years(dictionary)
    df.columns = columns
    df.index = years
    return df
