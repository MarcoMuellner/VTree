from typing import Union,List
from operator import itemgetter
import numpy as np
from collections import OrderedDict
from pandas import DataFrame as df
from graphviz import Digraph

from VTree.timeseries.Sample import Sample
from VTree.distance.kmedoid import kMedoids
from VTree.distance.distance import distances,distance_matrix
from VTree.timeseries.Subsample import Subsample
from VTree.timeseries.Timeseries import Timeseries

class Node:
    def __init__(self,level : int, k : int, m: int,sample : Union[Sample,Subsample], parent = None,center : int=None,sample_len = None,sub_id = None):
        self.level = level
        if sub_id is None:
            self.sub_id = 0
        else:
            self.sub_id = sub_id
        self.k = k
        self.m = m
        self.sample = sample
        self.distance_matrix = sample.distance_matrix

        self.subclusters = None
        if parent is None:
            self.parent = self
        else:
            self.parent = parent
        self.node_id = None
        self._index = None
        self.center = center
        self.ordered_nodes = None

        if sample_len is None:
            self.sample_len = len(sample)
        else:
            self.sample_len = sample_len

        if self.leaf:
            self._associated_ts = np.zeros(self.sample_len)

        self._query_ts_count = 0

        self.cluster()

    def cluster(self):
        if len(self.sample) < self.k or self.level >= self.m:
            return

        m, c = kMedoids(self.distance_matrix,self.k)

        subclusters = []

        for i,(key,value) in enumerate(c.items()):
            try:
                objs = itemgetter(*value.tolist())(self.sample.subs_obj)
            except AttributeError:
                objs = itemgetter(*value.tolist())(self.sample.objs)

            if not isinstance(objs,tuple):
                sample = Subsample([objs])
            else:
                sample = Subsample(list(objs))
            n = Node(self.level+1,self.k,self.m,sample,self,np.where(value==m[key])[0][0],self.sample_len,sub_id=i)

            subclusters.append(n)

        self.subclusters = subclusters

    def _order_nodes(self):
        node_list = self.get_sub_clusters([])
        ordered_nodes = df.from_dict({
            'obj' : node_list,
            'level' : [i.level for i in node_list],
        }).sort_values(by=['level'])

        level_list = []
        for i in ordered_nodes.groupby('level'):
            level_list.append((i[0], i[1].obj.tolist()))

        ordered_nodes = []
        for level,objs in level_list:
            if len(objs) == 1:
                ordered_nodes.append(objs[0])
                continue

            sub_ids = [i.sub_id for i in objs]

            if len(np.unique([sub_ids])) == len(sub_ids):
                ordered_nodes = ordered_nodes + list(itemgetter(*np.argsort(sub_ids).tolist())(objs))
            else:
                ordered_nodes = ordered_nodes + df.from_dict({
                                                'parent_sub':[i.parent.sub_id for i in objs],
                                                'sub_id':sub_ids,
                                                'objs':objs
                                                }).sort_values(by=['parent_sub','sub_id']).objs.tolist()

        for i,(prev,next) in enumerate(zip(ordered_nodes[:-1],ordered_nodes[1:])):
            prev._next = next
            next._prev = prev
            prev.index = i

        ordered_nodes[-1].index = i+1
        return ordered_nodes

    def create_d_matrix(self):
        if self.level != 0:
            raise ValueError("You can only link from root!")

        self.ordered_nodes = self._order_nodes()
        self.d_matrix = np.array([i.associated_ts for i in self.ordered_nodes]).T

        self.weigh_vector = self.sample_len/np.sum(self.d_matrix,axis=0)
        self.weigh_vector[np.isinf(self.weigh_vector)] = 0

        self.w_d_matrix = self.weigh_vector*self.d_matrix
        d_norm = np.linalg.norm(self.w_d_matrix, axis=1)
        self.w_d_matrix =(self.w_d_matrix.T/d_norm).T
        pass

    def q_vector(self):
        if self.level != 0:
            raise ValueError("Q vector only available from root")

        _q_vector = np.array([i.query_ts_count for i in self.ordered_nodes])
        _q_vector = _q_vector*self.weigh_vector
        d_norm = np.linalg.norm(_q_vector)
        _q_vector /= d_norm
        return _q_vector

    def visualize(self):
        if self.ordered_nodes is None:
            self.ordered_nodes = self._order_nodes()

        dot = Digraph("Variability tree")
        for i in self.ordered_nodes:
            dot.node(f'{i.index}',f'{i.node_name}')

        conn_list = self.form_edges()
        for i,j in conn_list:
            dot.edge(i,j)
        dot.view()

    def form_edges(self):
        connection_list = []
        for i in self.subclusters:
            connection_list.append((f'{self.index}',f'{i.index}'))
            if not i.leaf:
                connection_list = connection_list + i.form_edges()

        return connection_list

    def get_sub_clusters(self,node_list):
        if self.subclusters is None:
            return node_list + [self]

        for i in self.subclusters:
            node_list =  i.get_sub_clusters(node_list)

        node_list = [self] + node_list

        return node_list

    def next_node(self, data: Timeseries,query : bool = False):
        if not self.leaf:
            objs = [data] + [i.center_obj for i in self.subclusters]
            next_cluster = np.argmin(distance_matrix(objs)[0][1:])
            self.subclusters[next_cluster].next_node(data,query)
        else:
            self.add_ts_count(data.index,query)

    def populate_tree(self, t_s : float):
        if self.level != 0:
            raise ValueError("Tree can only be populated from the master node.")

        for i in self.sample:
            for j in i.window(t_s):
                self.next_node(j)

    def add_query_ts(self, t_s : float, data : Timeseries):
        if self.level != 0:
            raise ValueError("Tree can only be queried from root!")

        for i in data.window(t_s):
            self.next_node(i,True)

    def add_ts_count(self, index, query : bool):
        if not self.leaf:
            raise ValueError("You can only add associated timeseries to leaf objects")

        if query:
            self._query_ts_count +=1
        else:
            self._associated_ts[index] += 1

    def reset_query_ts_count(self):
        self._query_ts_count = 0
        if not self.leaf:
            for i in self.subclusters:
                i.reset_query_ts_count()

    @property
    def center_obj(self):
        if self.center is None:
            raise ValueError("Cluster has no center!")
        return self.sample[self.center]

    @property
    def associated_ts(self):
        if self.leaf:
            return self._associated_ts
        else:
            return np.sum([i.associated_ts for i in self.subclusters],axis=0)

    @property
    def query_ts_count(self):
        if self.leaf:
            return self._query_ts_count
        else:
            return np.sum([i.query_ts_count for i in self.subclusters],axis=0)

    @property
    def center_obj(self):
        return self.sample[self.center]

    @property
    def leaf(self):
        return self.subclusters is None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self,val):
        self._index = val
        self.node_name = fr'Node {self._index}'


