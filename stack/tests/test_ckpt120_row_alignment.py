"""PREREG_V8F_CKPT120: the two label tables that train the arms are ROW-ALIGNED to the trainer's join (window (clip c, step t) reads row off[c] + t) and to navtok_table.npz.

Runs on the REAL release-of-record tables when they are on this machine (skipped otherwise). The loaders under test are the trainer's own (`rc120_arrays`, `nav_strat_arrays`), so a pass is a statement about the path the trainer takes.
Mutations: a table with one clip's offset shifted, and one with a clip renamed, must each be REFUSED.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tanitad.data import v8f_mini_data as D  # noqa: E402

B = Path(os.environ.get("CKPT120_BUILD", "F:/TanitAD-scratch/strat1/ckpt120build"))
RC = B / "labels" / "rc120_table_rel-83e30bd3.npz"
NS = B / "nav_s" / "navtext_nav_s.npz"
NT = B / "in" / "navtok_table.npz"
JT, JE = B / "in" / "join_train.npz", B / "in" / "join_eval141.npz"
RC_SHA = "0f122570be5c0b9a60641ea9938102deb212752f5a67c411049e9dd116e01bf2"
NS_SHA = "0f7431a18622bf11cc678a39db5fc230b21eeefb5d254928617b56525db57c81"

pytestmark = pytest.mark.skipif(not all(p.is_file() for p in (RC, NS, NT, JT, JE)), reason="the real tables / joins are not on this machine")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _join(p):
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in ("clip", "off", "n")}


def test_the_tables_are_the_recorded_files():
    assert _sha(RC) == RC_SHA and _sha(NS) == NS_SHA


@pytest.mark.parametrize("split,jp", [("train", JT), ("eval", JE)])
def test_rc120_and_nav_s_are_row_aligned_to_the_join_and_to_navtok(split, jp):
    J = _join(jp)
    N = int(J["off"][-1] + J["n"][-1])
    with np.load(NT, allow_pickle=False) as t:                                   # navtok: same clip / off / n as the join
        assert np.array_equal(t[f"{split}_clip"].astype(str), J["clip"].astype(str))
        assert np.array_equal(t[f"{split}_off"], J["off"]) and np.array_equal(t[f"{split}_n"], J["n"])
        assert t[f"{split}_cls_id"].shape[0] == N
    rc, ok, rec = D.rc120_arrays(J, RC, RC_SHA, split)                           # the trainer's loaders raise on any shifted clip / off / n
    nav, nrec = D.nav_strat_arrays(J, NS, NS_SHA, split)
    assert rc.shape == (N, 3) and ok.shape == (N,) and nav.shape == (N, 5)
    # a window (c, t) with t < n[c] reads row off[c] + t; the last row of every clip is the last row before the next clip's offset
    assert np.array_equal(J["off"][1:], (J["off"] + J["n"])[:-1])
    assert rec["n_ok_table"] == {"train": 427805, "eval": 13148}[split]          # the counts of BUILD_CKPT120_P2A section 1
    assert nrec["n_rows_differ_from_real"] == 0


@pytest.mark.parametrize("which", ["off", "clip"])
def test_mutation_a_shifted_offset_or_a_renamed_clip_is_refused(which, tmp_path):
    J = _join(JT)
    with np.load(RC, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if which == "off":
        d["train_off"] = d["train_off"].copy()
        d["train_off"][5] += 1
    else:
        d["train_clip"] = d["train_clip"].copy()
        d["train_clip"][7] = "x" + str(d["train_clip"][7])[1:]
    p = tmp_path / "mut.npz"
    np.savez(p, **d)
    with pytest.raises(ValueError, match="misaligned"):
        D.rc120_arrays(J, p, _sha(p), "train")
