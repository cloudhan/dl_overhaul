from ..utils import ensure_file
import pathlib
import numpy as np


this_dir = pathlib.Path(__file__).resolve().absolute().parent


train_32x32 = ensure_file("http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                          this_dir.joinpath("downloads/train_32x32.mat"),
                          "e26dedcc434d2e4c54c9b2d4a06d8373")

test_32x32 = ensure_file("http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                         this_dir.joinpath("downloads/test_32x32.mat"),
                         "eb5a983be6a315427106f1b164d9cef3")

extra_32x32 = ensure_file("http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                          this_dir.joinpath("downloads/extra_32x32.mat"),
                          "a93ce644f1a588dc4d68dda5feec44a7")


class SVHNDataset:
    def __init__(self):
        Xs = []
        ys = []
        X, y = self._load_mat(train_32x32)
        Xs.append(X)
        ys.append(y)

        X, y = self._load_mat(extra_32x32)
        Xs.append(X)
        ys.append(y)

        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)

        num_samples = X.shape[0]
        split = int(num_samples * 0.8)

        self.train_X = X[:split]
        self.train_y = y[:split]

        self.val_X = X[split:]
        self.val_y = y[split:]

        self.test_X, self.test_y = self._load_mat(test_32x32)

    def _load_mat(self, mat_path):
        from scipy.io import loadmat

        mat = loadmat(mat_path)
        X = mat["X"]
        y = mat["y"]

        X = np.transpose(X, [3, 0, 1, 2])
        y = y.flatten()
        return X, y
