"""L1 / BACKLOG R37 — the loss term the trainer has NEVER logged.

⛔ THE DEFECT, MEASURED 2026-09-03. The model computes
``out["loss_lat_label"]`` / ``out["loss_lon_label"]`` (`refa_v1.py:1645`) and
folds them into the objective with ``w_tac_label`` (`:1648`); it does the same
for ``out["loss_route_label"]`` with ``w_str_label`` (`:1662`, `:1663`). On the
two banked step-1,000 checkpoints the tactical label term is **13.59 %**
(incumbent) / **20.51 %** (ep2) of the label+feature sum
(`2026-09-03-tactical-decoder/raw/intent_probe_*.json`) — and **not one row of
the nine banked `train_log.jsonl` carries it**, because `refa_v1_train.py`'s log
row had no such key. A term nobody can see cannot be tuned, and
`Project Steering/PREREG_TACTICAL_DECODER.md` §3 makes its absence a PREFLIGHT
REFUSAL for every arm in that document.

⭐ WHAT IS PINNED HERE
  1. the labelled path ON  -> all four keys present, all floats;
  2. the labelled path OFF -> all four **exactly `None`**;
  3. ⛔ **`None`, NEVER 0.0** — a 0.0 in this column reads as *"supervised, and
     perfect"*, so an UNLABELLED step and a PERFECTLY CLASSIFIED step would
     print the same character. `test_c_*` rejects `0.0` explicitly, by
     identity, not by truthiness;
  4. the weighted share matches a HAND COMPUTATION from the row's own numbers
     and the config's own weights, folded exactly as the model folds them;
  5. **THE LOAD-BEARING REGRESSION** — every key the row carried BEFORE this
     change is still present, still the same type, still bit-identical across
     two identically-seeded runs, and the only new keys are the four added
     ones. Nothing was renamed, dropped, or re-valued.
  6. an ALL-IGNORED label family (-100 everywhere) reads `None`, not 0.0 —
     `refa_v1.py:1643` skips it rather than averaging a NaN, so this is the
     common case (8.13 % of batch-8 steps carry no lateral label at all,
     MEASURED `raw/window_band_census.json`), not an edge one.

TIER: n/a (a trainer-contract test on `--smoke`, no data, no GPU).
EVIDENCE CLASS: MEASURED.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_TRAIN = (Path(__file__).resolve().parents[1] / "scripts" / "refa_v1_train.py")

#: ⭐ THE PRE-CHANGE ROW, KEY FOR KEY. Copied from `refa_v1_train.py`'s log row
#: as it stood BEFORE the L1 edit; `cf_*` are conditional on `--w-cf` and are
#: not exercised here. If any of these ever disappears or is renamed, this test
#: fails and names it — which is the whole point of writing them out.
PRE_CHANGE_KEYS = {
    "step", "loss", "precision", "tf32",
    "loss_feat_op", "loss_feat_tac", "loss_feat_str",
    "grad_norm", "adapter_std", "participation",
    "loss_sigreg", "loss_varfloor",
    "tgt_std_op", "tgt_std_tac", "tgt_std_str",
    "ema_decay", "clip", "skipped_steps",
    "tac_target_s", "str_target_s", "elapsed_s",
}
NEW_KEYS = {"loss_lat_label", "loss_lon_label", "loss_route_label",
            "loss_label_weighted_share"}
#: wall-clock; excluded from the bit-identity check for obvious reasons
NONDETERMINISTIC = {"elapsed_s"}


# --------------------------------------------------------------------------- #
def _load_trainer():
    """Import the trainer as a module so `SmokeData` can be swapped. The file
    is a script, not a package member, so it is loaded by path."""
    spec = importlib.util.spec_from_file_location("_refa_v1_train_under_test",
                                                  _TRAIN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _labelled(base_cls, *, lat=0, lon=0, route=0):
    """A `SmokeData` that also emits v7.2-shaped labels. `-100` is the loader's
    own IGNORE id and is what an out-of-band window carries."""
    class _D(base_cls):
        def batch(self, bs=None):
            b = super().batch(bs)
            n = b["feats"].shape[0]
            for key, val in (("lat_label", lat), ("lon_label", lon),
                             ("route_label", route)):
                if val is not None:
                    b[key] = torch.full((n,), int(val), dtype=torch.long)
            return b
    return _D


def _run(tmp_path, mod, *, extra=(), seed=0) -> dict:
    """One smoke step; returns the single log row."""
    out = Path(tmp_path)
    argv = ["--smoke", "--steps", "1", "--log-every", "1", "--bs", "2",
            "--device", "cpu", "--seed", str(seed), "--out", str(out),
            *extra]
    assert mod.main(argv) == 0
    rows = [json.loads(x) for x in
            (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    assert len(rows) == 1, rows
    return rows[0]


def _weights(tmp_path) -> tuple[float, float]:
    cfg = json.loads((Path(tmp_path) / "config.json").read_text("utf-8"))
    c = cfg.get("cfg", cfg)
    return float(c["w_tac_label"]), float(c["w_str_label"])


# =========================================================================== #
#  (a) the labelled path ON — all four keys, all floats                        #
# =========================================================================== #
def test_a_the_labelled_path_emits_all_four_keys(tmp_path):
    mod = _load_trainer()
    mod.SmokeData = _labelled(mod.SmokeData)
    row = _run(tmp_path, mod)
    for k in NEW_KEYS:
        assert k in row, (k, sorted(row))
        assert isinstance(row[k], float), (k, type(row[k]))
    assert row["loss_lat_label"] > 0.0 and row["loss_lon_label"] > 0.0
    assert row["loss_route_label"] > 0.0
    assert 0.0 < row["loss_label_weighted_share"] < 1.0, row


# =========================================================================== #
#  (b) the HAND COMPUTATION — folded exactly as refa_v1.py folds it            #
# =========================================================================== #
def test_b_the_weighted_share_matches_a_hand_computation(tmp_path):
    mod = _load_trainer()
    mod.SmokeData = _labelled(mod.SmokeData)
    row = _run(tmp_path, mod)
    w_tac, w_str = _weights(tmp_path)
    # refa_v1.py:1646-1649 — the MEAN over the present families, * w_tac_label
    # refa_v1.py:1663-1664 — + w_str_label * the route term
    hand = (w_tac * (row["loss_lat_label"] + row["loss_lon_label"]) / 2.0
            + w_str * row["loss_route_label"]) / row["loss"]
    assert row["loss_label_weighted_share"] == pytest.approx(hand, rel=1e-9)
    # and the weights really are the shipped ones, not a test fiction
    assert (w_tac, w_str) == (0.1, 0.1)


# =========================================================================== #
#  (c) ⛔ None, NEVER 0.0 — the whole reason the key exists                     #
# =========================================================================== #
def test_c_an_unlabelled_step_reads_None_and_never_zero(tmp_path):
    mod = _load_trainer()                       # stock SmokeData: no labels
    row = _run(tmp_path, mod)
    for k in NEW_KEYS:
        assert k in row, (k, sorted(row))
        assert row[k] is None, (k, row[k])
        # by identity, not truthiness: `0.0 == None` is False but
        # `not 0.0` is True, and a laxer assertion would pass on 0.0
        assert row[k] is not False and row[k] != 0.0, (k, row[k])
    # the row must still be a complete, finite training row
    assert row["loss"] > 0.0 and row["step"] == 1


def test_c2_a_partly_labelled_step_reads_None_only_where_it_should(tmp_path):
    """lat + route present, lon ABSENT: the share must fold only what exists."""
    mod = _load_trainer()
    mod.SmokeData = _labelled(mod.SmokeData, lat=0, lon=None, route=0)
    row = _run(tmp_path, mod)
    assert isinstance(row["loss_lat_label"], float)
    assert row["loss_lon_label"] is None
    assert isinstance(row["loss_route_label"], float)
    w_tac, w_str = _weights(tmp_path)
    hand = (w_tac * row["loss_lat_label"]      # mean over ONE present family
            + w_str * row["loss_route_label"]) / row["loss"]
    assert row["loss_label_weighted_share"] == pytest.approx(hand, rel=1e-9)


# =========================================================================== #
#  (d) ALL-IGNORED labels — the common case, not an edge one                   #
# =========================================================================== #
def test_d_an_all_ignored_label_family_reads_None_not_zero(tmp_path):
    mod = _load_trainer()
    mod.SmokeData = _labelled(mod.SmokeData, lat=-100, lon=-100, route=-100)
    row = _run(tmp_path, mod)
    for k in NEW_KEYS:
        assert row[k] is None, (k, row[k])
    assert row["loss"] > 0.0


# =========================================================================== #
#  (e) THE LOAD-BEARING REGRESSION                                             #
# =========================================================================== #
@pytest.mark.parametrize("labelled", [False, True])
def test_e_every_pre_existing_key_survives_unchanged(tmp_path, labelled):
    mod = _load_trainer()
    if labelled:
        mod.SmokeData = _labelled(mod.SmokeData)
    a = _run(tmp_path / "a", mod, seed=0)
    b = _run(tmp_path / "b", _reload(mod, labelled), seed=0)
    missing = PRE_CHANGE_KEYS - set(a)
    assert not missing, f"the L1 edit DROPPED pre-existing log keys: {missing}"
    added = set(a) - PRE_CHANGE_KEYS
    assert added == NEW_KEYS, f"unexpected new/renamed keys: {added ^ NEW_KEYS}"
    for k in PRE_CHANGE_KEYS - NONDETERMINISTIC:
        assert type(a[k]) is type(b[k]), (k, type(a[k]), type(b[k]))
        assert a[k] == b[k], (k, a[k], b[k])       # bit-identical, same seed


def _reload(mod, labelled):
    m2 = _load_trainer()
    if labelled:
        m2.SmokeData = _labelled(m2.SmokeData)
    return m2


# =========================================================================== #
#  (f) the accounting identity — the row's `loss` really is the model's loss   #
# =========================================================================== #
def test_f_the_row_reconciles_with_the_models_own_objective(tmp_path):
    """`loss` = the three weighted feature terms + the weighted label terms.
    This is what makes the share's DENOMINATOR meaningful, and it is the check
    that would catch a share computed against the wrong total."""
    mod = _load_trainer()
    mod.SmokeData = _labelled(mod.SmokeData)
    row = _run(tmp_path, mod)
    cfg = json.loads((Path(tmp_path) / "config.json").read_text("utf-8"))
    c = cfg.get("cfg", cfg)
    feat = (float(c["w_feat_op"]) * row["loss_feat_op"]
            + float(c["w_feat_tac"]) * row["loss_feat_tac"]
            + float(c["w_feat_str"]) * row["loss_feat_str"])
    lab_w = row["loss_label_weighted_share"] * row["loss"]
    # the smoke path carries no str_ext targets and no sigreg/varfloor/cf, so
    # these are the only live terms; if that ever changes this test says so
    # rather than the share silently drifting.
    assert row["loss_sigreg"] is None and row["loss_varfloor"] is None
    assert feat + lab_w == pytest.approx(row["loss"], rel=1e-6), (
        feat, lab_w, row["loss"])
