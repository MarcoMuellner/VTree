from collections import OrderedDict

import numpy as np

class ISample:
    def __init__(self):
        self._idx = 0
        self._objs = []

    def __iter__(self):
        return self

    def __next__(self):
        self._idx +=1

        try:
            return self._objs[self._idx - 1]
        except (IndexError,KeyError):
            self._idx = 0
            raise StopIteration

    def __getitem__(self, item):
        if isinstance(item,np.ndarray):
            if isinstance(self._objs,OrderedDict):
                return list(self._objs.values())[item.tolist()]
            else:
                return self._objs[item.tolist()]

        else:
            try:
                if isinstance(self._objs, OrderedDict):
                    return list(self._objs.values())[item]
                else:
                    return self._objs[item]
            except:
                raise IndexError(f"Can't slice objs with type {type(item)}")

    def __len__(self):
        return len(self._objs)

    @property
    def objs(self):
        if isinstance(self._objs,OrderedDict):
            return list(self._objs.values())
        else:
            return self._objs

