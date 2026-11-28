"""CPU-check shim (scratch only): the three cv2 calls train.py:898-912 makes, on PIL.  imdecode -> BGR uint8 like cv2; resize = bilinear (cv2.INTER_LINEAR default).
Pixel values differ from real cv2 by sub-LSB-scale resampling details: declared as a deviation in the check's record."""
import io
import numpy as np
from PIL import Image

IMREAD_COLOR = 1


def imdecode(buf, flag):
    im = Image.open(io.BytesIO(bytes(buf))).convert("RGB")
    return np.asarray(im)[:, :, ::-1].copy()          # BGR


def resize(im, size):
    w, h = size
    rgb = Image.fromarray(np.ascontiguousarray(im[:, :, ::-1]))
    out = rgb.resize((int(w), int(h)), Image.BILINEAR)
    return np.asarray(out)[:, :, ::-1].copy()
