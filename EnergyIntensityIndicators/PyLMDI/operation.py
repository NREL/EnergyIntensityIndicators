import numpy  as np
def Lfun(yt,y0):
    """Logarithmic mean

    Parameters
    ----------
    yt : float
        first number to use in
        log mean
    y0 : float
        second number to use in
        log mean

    Returns
    -------
    mean : float
        final logarithmic mean
    """
    if yt == y0:
        return 0
    else:
        return (yt-y0)/(np.log(yt) - np.log(y0))
