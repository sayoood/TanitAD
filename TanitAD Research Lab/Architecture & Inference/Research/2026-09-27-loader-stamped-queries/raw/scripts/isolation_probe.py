"""SCRATCH (never landed): runs AFTER test_loader_stamped_queries.py in the same process and asserts its
red-arm fixture left nothing behind -- no hooked importer, no stripped trainer, no attribute planted on a
lazy module -- and that a pre-A9 record still loads at 100 through both entry points.

Run (a SCRATCH tree, never the repo): copy to <tree>/stack/tests/test_zz_i3_isolation.py, then
    pytest -q tests/test_loader_stamped_queries.py tests/test_zz_i3_isolation.py
    pytest -q tests/test_loader_stamped_queries.py tests/test_zz_i3_isolation.py -k "RED_ARM_without or red_arm_left"
Named without the test_ prefix here so a repo-root pytest never collects it from the package."""
import importlib.util
import sys

import torch

import test_loader_stamped_queries as M

ORIG = importlib.util.spec_from_file_location          # captured at COLLECTION, before any red arm


def _arm_trainers():
    out = {}
    for name, mod in list(sys.modules.items()):
        d = getattr(mod, "__dict__", None) or {}
        if str(d.get("__file__") or "").endswith("refcv3_arm.py") and d.get("_TRAINER") is not None:
            out[name] = d["_TRAINER"]
    t = sys.modules.get("refc_v3_train_for_arm")
    if t is not None:
        out["sys.modules[refc_v3_train_for_arm]"] = t
    return out


def test_the_red_arm_left_no_hook_and_no_stripped_trainer(tmp_path):
    assert importlib.util.spec_from_file_location is ORIG
    assert ORIG.__module__ == "_frozen_importlib_external"
    assert "_TRAINER" not in vars(torch.ops) and "rebuild_config" not in vars(torch.ops)
    for name, tr in _arm_trainers().items():                   # whatever SURVIVED the file: never stripped
        assert "agent_queries_as_trained" in vars(tr), name
    ck = M.write_record(tmp_path / "pre", queries=100, pre_a9=True)
    for entry in M.ENTRIES.values():
        model, prov = entry(ck)
        assert M._strict(prov) == M.STRICT and M._counts(model) == (100, 100)
    after = _arm_trainers()                                     # control: the check above is not vacuous
    assert "refcv3_arm" in after and "sys.modules[refc_v3_train_for_arm]" in after
    assert all("agent_queries_as_trained" in vars(tr) for tr in after.values())
