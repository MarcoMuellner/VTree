import numpy as np
from typing import Union,List,Dict,Optional
from pandas import DataFrame as df,Series
from sklearn import manifold
from collections import OrderedDict

from VTree.timeseries.Timeseries import Timeseries
from VTree.timeseries.ISample import ISample
from VTree.distance.distance import distance_matrix

class Sample(ISample):
    def __init__(self,time : List[np.ndarray],flux : [np.ndarray],t_w : float,label : List[str] = None, tic_id : List[int] = None):
        """
        The sample represents the basic training sample for the VTree. You need to put in a list of time and flux for
        each star at minimum. Optionally, you can provide labels and TIC ids for the stars you want to add
        :param time: List of time arrays from your stars
        :param flux: List of flux arrays from your stars
        :param t_w: Size of window for random subsequence (see 'Timeseries')
        :param label: Labels of the stars
        :param tic_id: TIC ids of the stars
        """

        super().__init__()
        if len(time) != len(flux):
            raise ValueError("Time and flux must have the same length!")

        if label is None:
            label = [None for i in range(len(time))]
        elif len(label) != len(time):
            raise ValueError("Length of label list not equal to length of LCs")


        if tic_id is None:
            tic_id = [None for i in range(len(time))]
        elif len(tic_id) != len(tic_id):
            raise ValueError("Length of TIC ID list not equal to length of LCs")

        self._objs = OrderedDict()

        for i,(t,f,l,tic) in enumerate(zip(time,flux,label,tic_id)):
            self._objs[i] = Timeseries(t, f, t_w, label=l, tic_id=tic,idx=i)

        self._subs_obj = [o.subseq for o in self._objs.values()]

        self.distance_matrix = distance_matrix(self.subs_obj)

    @property
    def subs_obj(self):
        return self._subs_obj




