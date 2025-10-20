import pickle
import zarr
import numpy as np


def savez(store, *args, allow_pickle=True, **kwargs):
    store = store.replace('.npz', '.zarr')
    store = zarr.storage.LocalStore(store) #type:ignore
    def save_array(name, obj):
        if isinstance(obj, (list, tuple)):
            obj = np.array(obj)
        if allow_pickle and not isinstance(obj, np.ndarray):
            store_group.attrs[name] = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            zarr.create_array(store=store_group.store, name=name, data=obj, overwrite=True, chunks='auto')
    store_group = zarr.open_group(store, mode='w')
    for i, obj in enumerate(args):
        save_array(f"arr_{i}", obj)
    for name, obj in kwargs.items():
        save_array(name, obj)
  
class load:
    def __init__(self, store, allow_pickle=True, mmap_mode=False):
        self.store = store.replace('.npz', '.zarr')
        self.allow_pickle = allow_pickle
        self.mmap_mode = mmap_mode
        self.group = None
        self.data = {}

    def __enter__(self):
        self.group = zarr.open_group(self.store, mode='r')
        out = {}
        for key in self.group.array_keys():
            if self.mmap_mode:
                out[key] = self.group[key]
            else:
                out[key] = self.group[key][:] #type:ignore
        for key in self.group.attrs:
            if self.allow_pickle:
                out[key] = pickle.loads(self.group.attrs[key]) #type:ignore
            else:
                out[key] = self.group.attrs[key]

        self.data = out
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.group = None

    def list_arrays(self, include_attrs=False):
        if self.group is None:
            group = zarr.open_group(self.store, mode='r')
        else:
            group = self.group

        keys = list(group.array_keys())
        if include_attrs:
            keys += list(group.attrs.keys())
        return keys

    def __getitem__(self, key):
        return self.data[key]

    def files(self):
        return self.list_arrays(include_attrs=True)


