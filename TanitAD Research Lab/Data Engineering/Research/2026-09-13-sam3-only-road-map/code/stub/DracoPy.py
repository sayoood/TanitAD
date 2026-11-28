"""Stub: on Thor the map scorer reads LiDAR sweeps pre-decoded on the dev box (DracoPy 2.0.0). Decoding here is a bug."""


def decode(*args, **kwargs):
    raise RuntimeError("DracoPy stub called: sweeps must come from the pre-decoded cache (decode_sweeps.py)")
