import numpy as np
from typing import Tuple,Union
from numpy.random import choice
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as pl


class Timeseries:
    def __init__(self,time : np.ndarray, flux : np.ndarray, t_w : float = None, label : str = None,child=False,tic_id : int = None,idx : int = None):
        """
        The Timeseries object represents the basic light curve object. This should be the primary accessor
        for all light curves. It automatically deals with subsequencing, normalization, etc.
        :param time: Time axis of the light curve
        :param flux: Flux axis of the light curve. Flux should be given in magnitude.
        :param label: Label of the object, if one is available
        :param subsequence: A flag, that defines if this object is a subsequence of a lightcurve
        :param idx: If this value is set, it needs to be a unique value within the sample.
        """
        self._time,self._flux = self._normalize(time,flux,child)
        self._orig_time,self._orig_flux = time,flux
        self._label = label
        self._subsequence : Timeseries = None
        self.parent = None
        self.end = np.amax(self._time)
        self.tic_id = tic_id
        self.child = child
        self.index = idx
        self.t_w = t_w

        if not child:
            self.create_subseq(t_w)

    def _normalize(self,time : np.ndarray, flux : np.ndarray,child : bool) -> Tuple[np.ndarray,np.ndarray]:
        """
        Normalizes the light curve according to the procedure described in Valenzuela (2018). Additionally,
        we remove all nans and infs from the flux column.
        :param time: Time column
        :param flux: Flux column
        :return: Reduced time and flux column
        """
        inv_mask = np.logical_or(np.isnan(flux),np.isinf(flux))

        _time = time[~inv_mask]
        _flux = flux[~inv_mask]

        sort_mask = np.argsort(time)

        _time = _time[sort_mask]
        _flux = _flux[sort_mask]

        if not child:
            _flux -= np.mean(_flux)
            _flux /= np.std(_flux)

        try:
            _time -= np.amin(_time)
        except:
            pass

        return _time,_flux

    def plot(self,ax : Axes = None,show = True) -> Union[Figure,None]:
        """
        Plots the light curve
        """
        if ax is None:
            fig,ax = pl.subplots(1,1,figsize=(16,10))

        ax.plot(self._time,self._flux,'ko',markersize=2,label='Light curve')
        if self._subsequence is not None:
            ax.plot(self._subsequence._orig_time
                    ,self._subsequence._orig_flux,'ro',markersize=2,label='Subsequence')

        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Flux")
        if self._label is not None:
            ax.set_title(self._label)

        if show:
            pl.tight_layout()
            pl.show()

        if ax is None:
            return fig
        else:
            return None

    @property
    def time(self) -> np.ndarray:
        return self._time

    @property
    def flux(self) -> np.ndarray:
        return self._flux

    @property
    def subseq(self):
        if self.child:
            raise AttributeError("Subsequence is disabled for subsequences")

        if self._subsequence is None:
            raise ValueError("Subsequence not computed! Use create_subseq before accessing it!")

        return self._subsequence

    @property
    def label(self):
        return self._label

    def window(self,t_s : float):
        """
        Window function. Returns a sliding window over the whole light curve.
        :param t_w: width of the window
        :param t_s: step size of the window
        :return: Timeseries object of the subsequence
        """
        if self.child:
            raise AttributeError("Window is disabled for subsequences")

        t_f = self.time[-1]
        last_index = np.where(self.time > t_f - self.t_w)[0][0]
        last_pos_time = self.time[last_index]
        start_time = self.time[0]
        while start_time < last_pos_time:
            yield self._get_subsequence(start_time,self.t_w)
            start_time += t_s
            start_index = np.where(self.time >= start_time)[0][0]
            start_time = self.time[start_index]


    def create_subseq(self,t_w : float):
        """
        Creates a subsequence object, by randomly assigning a starting point and creating a window of size t_w
        over it.
        :param t_w: Size of window
        :return: Timeseries object of the subsequence
        """
        if self.child:
            raise AttributeError("Subsequence is disabled for subsequences")
        start_time = choice(self.time)
        self._subsequence = self._get_subsequence(start_time,t_w)
        self.t_w = t_w

    def _get_subsequence(self,start_time : float, t_w : float):
        """
        Returns a subsequence from starting point 'start_time' with a width of 't_w'
        :param start_time: start time of the subsequence
        :param t_w: width of the window
        :return: Timeseries object of the subsequence.
        """
        end_point = start_time + t_w
        mask = np.logical_and(self.time > start_time,self.time < end_point)

        subsequence = Timeseries(self._time[mask],self._flux[mask],label=self._label,child=True,tic_id=self.tic_id,idx=self.index)
        subsequence.disable_sugseq = True
        subsequence.parent = self
        return subsequence
