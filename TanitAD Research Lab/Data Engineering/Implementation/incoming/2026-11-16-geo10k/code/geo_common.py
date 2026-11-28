"""Shared setup: the cdbff971 tree on sys.path, and the egomotion lookup pointed at the flat dir of the keep10k parquets by
CONFIGURATION (the module constants of tanitad.data.egomotion_source), not by editing library code."""
import os, sys
TREE = "/root/geo10k/tree/stack"
for p in (TREE, TREE + "/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
FLAT = "/root/geo10k/ego_flat"
OUT = "/root/geo10k/out"
IN = "/root/geo10k/inputs"
from tanitad.data import egomotion_source as ES


def point_ego_lookup(flat=FLAT):
    ES.EGO_GLOB = "/nonexistent/egomotion/*.zip"
    ES.EGO_FLAT = f"{flat}/*.parquet"
    ES._index.cache_clear()
