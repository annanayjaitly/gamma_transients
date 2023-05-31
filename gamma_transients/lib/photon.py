class Photon:
    """
    class used for the multiplet scanner to store ra,dec,time and id of single photons.
    """

    def __init__(self, ra, dec, t, id_):
        self.ra = ra
        self.dec = dec
        self.t = t
        self.id_ = id_

    def __eq__(self, other):
        return self.ra == other.ra and self.dec == other.dec and self.t == other.t

    def __repr__(self):
        return 'id: ' + str(self.id_) + 'ra: ' + str(self.ra) + ', dec: ' + str(self.dec) + ', t: ' + str(self.t)

    def _cmp(self, other):
        return self.t - other.t

    # comparison methods
    def __lt__(self, other):
        return self._cmp(other) < 0

    def __le__(self, other):
        return self._cmp(other) <= 0

    def eq_t(self, other):
        return self._cmp(other) == 0

    def __ne__(self, other):
        return self._cmp(other) != 0

    def __ge__(self, other):
        return self._cmp(other) >= 0

    def __gt__(self, other):
        return self._cmp(other) > 0

    def get_values(self):
        return print('(ra: ', self.ra, ', dec: ', self.dec, ', t: ', self.t, ', id_:', self.id_, ')')

class PhotonSim:
    """
    class used for the multiplet scanner to store ra,dec,time and id of single photons. This class is used for the storing the simulated photons.
    """

    def __init__(self, ra, dec, t):
        self.ra = ra
        self.dec = dec
        self.t = t

    def __eq__(self, other):
        return self.ra == other.ra and self.dec == other.dec and self.t == other.t

    def __repr__(self):
        return 'ra: ' + str(self.ra) + ', dec: ' + str(self.dec) + ', t: ' + str(self.t)

    def _cmp(self, other):
        return self.t - other.t

    # comparison methods
    def __lt__(self, other):
        return self._cmp(other) < 0

    def __le__(self, other):
        return self._cmp(other) <= 0

    def eq_t(self, other):
        return self._cmp(other) == 0

    def __ne__(self, other):
        return self._cmp(other) != 0

    def __ge__(self, other):
        return self._cmp(other) >= 0

    def __gt__(self, other):
        return self._cmp(other) > 0

    def get_values(self):
        return print('(ra: ', self.ra, ', dec: ', self.dec, ', t: ', self.t, ', id_:', self.id_, ')')
