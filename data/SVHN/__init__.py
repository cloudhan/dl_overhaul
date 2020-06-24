from data import utils
from ..utils import ensure_file
import pathlib

this_dir = pathlib.Path(__file__).resolve().absolute().parent


train_32x32 = ensure_file("http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                          this_dir.joinpath("train_32x32.mat"),
                          "e26dedcc434d2e4c54c9b2d4a06d8373")

test_32x32 = ensure_file("http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                         this_dir.joinpath("test_32x32.mat"),
                         "eb5a983be6a315427106f1b164d9cef3")

extra_32x32 = ensure_file("http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                          this_dir.joinpath("extra_32x32.mat"),
                          "a93ce644f1a588dc4d68dda5feec44a7")
