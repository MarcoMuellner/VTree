from typing import Union, List, Optional
import numpy as np
from pandas import DataFrame as df
import matplotlib.pyplot as pl

from VTree.timeseries.Timeseries import Timeseries
from VTree.vtree.Node import Node
from VTree.timeseries.Sample import Sample


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


class VTree:
    def __init__(self, t_w: float, k: int, m: int
                 , time: List[np.ndarray] = None
                 , flux: List[np.ndarray] = None
                 , label: List['str'] = None
                 , tic_id: List[int] = None):
        self.t_w = t_w
        self.k = k
        self.m = m
        self.master_node : Node = None

        if time is not None and flux is not None:
            self._sample = Sample(time=time,flux=flux,label=label,tic_id=tic_id,t_w=t_w)
        else:
            self._sample: Sample = None

    def create_training_sample(self, time: List[np.ndarray]
                 , flux: List[np.ndarray]
                 , label: List['str']
                 , tic_id: List[int]):

        self._sample = Sample(time=time, flux=flux, label=label, tic_id=tic_id, t_w=self.t_w)


    def train(self,t_s:float):

        if self._sample is None:
            raise ValueError("Please provide a training set before training.")

        self.master_node = Node(0,self.k,self.m,self._sample)
        self.master_node.visualize()
        self.master_node.populate_tree(t_s)
        self.master_node.create_d_matrix()
        pass

    def query(self,data : Timeseries,t_s):
        self.master_node.reset_query_ts_count()
        self.master_node.add_query_ts(t_s,data)
        q_vector = self.master_node.q_vector()
        similarity = self.master_node.w_d_matrix.dot(q_vector)
        data = {
            'obj':[],
            'similarity':[]
        }
        for i in np.argsort(similarity)[::-1]:
            data['obj'].append(self.master_node.sample[i])
            data['similarity'].append(similarity[i])
        return df.from_dict(data)