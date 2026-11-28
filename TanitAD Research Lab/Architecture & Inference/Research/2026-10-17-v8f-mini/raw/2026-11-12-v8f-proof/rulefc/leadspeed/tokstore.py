"""tokstore.py -- card ADD149 amendment 3: the 4x10 tokens of a clip .pt file as a READ-ONLY np.memmap of their bytes inside the file's zip archive.
A read-only file mapping is backed by the file (page cache), not by the commit charge; torch.load(mmap=True) maps the whole file copy-on-write, which is.

  member(path, shape)      -> (offset, nbytes) of the one stored zip member whose size is prod(shape) bytes (the fp8 4x10 tokens); raises if not exactly one
  open_tokens(path, shape) -> np.memmap uint8 [shape], mode "r"
"""
import struct
import zipfile

import numpy as np


def member(path, shape):
    nbytes = int(np.prod(shape))
    with zipfile.ZipFile(path) as zf:
        hits = [i for i in zf.infolist() if i.file_size == nbytes and "/data/" in i.filename]
        if len(hits) != 1:
            raise ValueError("%s: %d zip members of %d bytes (need exactly 1)" % (path, len(hits), nbytes))
        zi = hits[0]
        if zi.compress_type != zipfile.ZIP_STORED:
            raise ValueError("%s: token member is compressed" % path)
    with open(path, "rb") as f:
        f.seek(zi.header_offset)
        h = f.read(30)
        if h[:4] != b"PK\x03\x04":
            raise ValueError("%s: bad local header" % path)
        n_name, n_extra = struct.unpack("<HH", h[26:30])
    return zi.header_offset + 30 + n_name + n_extra, nbytes


def open_tokens(path, shape):
    off, _ = member(path, shape)
    return np.memmap(path, dtype=np.uint8, mode="r", offset=off, shape=tuple(shape))
