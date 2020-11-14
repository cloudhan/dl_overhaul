import logging
import h5py

logger = logging.Logger(__name__)

class Node:
    def __iter__(self):
        raise NotImplementedError

class DataSource(Node): ...
class DataSink(Node): ...

class HDF5Source(DataSource):
    def __init__(self, hdf5_file_path: str):
        self._hdf5 = h5py.File(hdf5_file_path, "r")
