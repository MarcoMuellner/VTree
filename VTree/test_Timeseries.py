import numpy as np
import os
from VTree import VTree
from VTree.timeseries import Timeseries

time = []
flux = []

np.random.seed(52)

for subdir,dirs,files in os.walk("VTree/tests/testdata/tess/"):
    for i in files:
        if i.endswith('.txt'):
            data = np.loadtxt(os.path.join(subdir, i)).T
            time.append(data[0])
            flux.append(data[1])

        if len(time) > 60:
            break
v = VTree(t_w=5,k=5,m=5,time=time,flux=flux)
v.train(t_s=1)

t = v.master_node.sample[0]

result = v.query(t,1)

pass
