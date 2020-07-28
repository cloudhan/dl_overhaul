from ..utils import ensure_file, logger
import pathlib
import tarfile
import numpy as np

this_dir = pathlib.Path(__file__).resolve().absolute().parent

cifar_10 = ensure_file("https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz",
                       this_dir.joinpath("downloads/cifar-10-matlab.tar.gz"),
                       "70270af85842c9e89bb428ec9976c926")


class CIFAR10Dataset:

    def __init__(self):
        from scipy.io import loadmat

        t = tarfile.open(cifar_10)
        def load_mat(path, data, labels):
            logger.debug("loading {0} from {1}".format(path, cifar_10))
            mat = loadmat(t.extractfile(path))
            X = mat["data"].reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1]) # in RGB HWC format
            y = mat["labels"].flatten()
            data.append(X)
            labels.append(y)

        data, labels = [], []
        load_mat("cifar-10-batches-mat/data_batch_1.mat", data, labels)
        load_mat("cifar-10-batches-mat/data_batch_2.mat", data, labels)
        load_mat("cifar-10-batches-mat/data_batch_3.mat", data, labels)
        load_mat("cifar-10-batches-mat/data_batch_4.mat", data, labels)
        load_mat("cifar-10-batches-mat/data_batch_5.mat", data, labels)
        load_mat("cifar-10-batches-mat/test_batch.mat", data, labels)

        self.train_X = np.concatenate(data[:-2], axis=0)
        self.train_y = np.concatenate(labels[:-2], axis=0)

        self.val_X = data[-2]
        self.val_y = labels[-2]

        self.test_X = data[-1]
        self.test_y = labels[-1]
