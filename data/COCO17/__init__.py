from ..utils import ensure_file

import logging
import pathlib

import boto3
from botocore import UNSIGNED
from botocore.client import Config

this_dir = pathlib.Path(__file__).resolve().absolute().parent


def s3_handler(filename: str, filepath: str):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_name = 'fast-ai-coco'
    s3.download_file(bucket_name, str(filename), str(filepath))


annotations = ensure_file("annotations_trainval2017.zip",
                          this_dir.joinpath("downloads/annotations_trainval2017.zip"),
                          "f4bbac642086de4f52a3fdda2de5fa2c",
                          s3_handler)

val2017 = ensure_file("val2017.zip",
                      this_dir.joinpath("downloads/val2017.zip"),
                      "442b8da7639aecaf257c1dceb8ba8c80",
                      s3_handler)

train2017 = ensure_file("train2017.zip",
                        this_dir.joinpath("downloads/train2017.zip"),
                        "cced6f7f71b7629ddf16f17bbcfab6b2",
                        s3_handler)
