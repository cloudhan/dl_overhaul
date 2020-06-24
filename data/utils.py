from io import BytesIO
from typing import *

import os
import hashlib
import logging
import requests
import logging

logger = logging.getLogger(__name__)


def http_download(url: str, filepath: str):
    resp = requests.get(url)
    assert resp.status_code == 200
    with open(filepath, "wb") as f:
        f.write(resp.content)


def __block_hash(bio, h) -> str:
    for chunk in iter(lambda: bio.read(65536), b""):
        h.update(chunk)
    return h.hexdigest()


def md5sum(bio) -> str:
    md5 = hashlib.md5()
    return __block_hash(bio, md5)


def sha1sum(bio) -> str:
    sha1 = hashlib.sha1()
    return __block_hash(bio, sha1)


def sha256sum(bio) -> str:
    sha256 = hashlib.sha256()
    return __block_hash(bio, sha256)


def sha512sum(bio) -> str:
    sha512 = hashlib.sha512()
    return __block_hash(bio, sha512)


def ensure_file(url: str,
                filepath: str,
                checksum: str,
                url_handler: Callable[[Any, str], None] = http_download,
                checksum_handler: Callable[[BytesIO], str] = md5sum,
                force_check=False) -> str:
    should_download = not os.path.exists(filepath)
    if should_download:
        checksum = checksum.lower()
        logger.debug(f"downloading from {url} to {filepath}")
        url_handler(url, filepath)

    if should_download or force_check:
        with open(filepath, "rb") as f:
            if checksum_handler(f).lower() != checksum:
                logger.error(f"download or check failed, please retry")
                raise IOError(f"file {filepath} check failed with {url_handler}")

    return filepath
