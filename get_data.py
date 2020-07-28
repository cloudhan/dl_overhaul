import os
import logging

handler = logging.StreamHandler()
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
root.addHandler(handler)

import data.CIFAR10
import data.SVHN
import data.COCO17
