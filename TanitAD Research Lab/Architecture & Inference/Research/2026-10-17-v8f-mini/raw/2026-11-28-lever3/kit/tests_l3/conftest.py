"""Loads kit/dscale10k_check_l3.py as the module `dscale10k_check` (the pod's study tests import that name) and the unchanged pod checker as `s3_ref` (for the identity test)."""
import importlib.util
import os
import sys

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
if KIT not in sys.path:
    sys.path.insert(0, KIT)
POD_S3 = KIT + "/../rec/x/rollhist/kit_s3/dscale10k_check_s3.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


L3 = _load("dscale10k_check", KIT + "/dscale10k_check_l3.py")
S3 = _load("s3_ref", POD_S3) if os.path.isfile(POD_S3) else None
