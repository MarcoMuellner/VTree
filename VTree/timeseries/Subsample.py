from typing import List

from VTree.timeseries.Timeseries import Timeseries
from VTree.timeseries.ISample import ISample
from VTree.distance.distance import distance_matrix

class Subsample(ISample):
    def __init__(self,data : List[Timeseries]):
        super().__init__()
        self._objs = data
        self.node = None
        self.distance_matrix = distance_matrix(self.objs)