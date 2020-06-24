import os
import logging

handler = logging.StreamHandler()
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
root.addHandler(handler)

import data.SVHN
import data.COCO
