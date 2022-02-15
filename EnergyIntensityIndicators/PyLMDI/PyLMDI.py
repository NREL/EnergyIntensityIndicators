import numpy as np
from EnergyIntensityIndicators.PyLMDI.operation import Lfun


class PyLMDI():
    """Class to compute additive and
    multiplicative LMDI
    """

    def __init__(self, Vt, V0, Xt, X0):
        self.V0 = V0
        self.Vt = Vt
        self.X0 = X0
        self.Xt = Xt

    def Add(self):
        """Additive LMDI

        Returns
        -------
        Delta_V : list
            list of changes computed from the LMDI
        """
        Delta_V = [sum(self.Vt) - np.sum(self.V0)]
        for start, end in zip(self.X0, self.Xt):
            temp = sum([Lfun(self.Vt[i], self.V0[i]) *
                        np.log(end[i] / start[i])
                        for i in range(len(start))])
            Delta_V.append(temp)
        return Delta_V

    def Mul(self):
        """Multiplicative LMDI

        Returns
        -------
        D_V : list
            list of changes computed from the LMDI
        """
        D_V = [sum(self.Vt) / np.sum(self.V0)]
        for start, end in zip(self.X0, self.Xt):
            temp = sum([Lfun(self.Vt[i], self.V0[i]) /
                        Lfun(sum(self.Vt), sum(self.V0)) *
                        np.log(end[i] / start[i])
                        for i in range(len(start))])
            D_V.append(np.exp(temp))
        return D_V
