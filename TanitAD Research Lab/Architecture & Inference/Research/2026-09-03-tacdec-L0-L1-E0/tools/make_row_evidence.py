"""Bank a REAL `train_log.jsonl` carrying the four L1 keys — labelled and not.

Runs the shipped trainer twice on `--smoke` (tiny CPU model, no data): once
with a `SmokeData` that also emits v7.2-shaped labels, once stock. The two rows
are the artifact behind the L1 claim, so the RESULT quotes a file, not a test.
"""
import importlib.util
import json
import sys
from pathlib import Path

import torch

# argv: <out dir> [<path to refa_v1_train.py>]. The default is the RUN
# MIRROR, because the G: mount cannot import `tanitad` (Errno 22).
OUT = Path(sys.argv[1])
TRAIN = Path(sys.argv[2] if len(sys.argv) > 2
             else "C:/Users/Admin/tanitad-wt/stack/scripts/refa_v1_train.py")


def load():
    spec = importlib.util.spec_from_file_location("_t", TRAIN)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def labelled(base):
    class D(base):
        def batch(self, bs=None):
            b = super().batch(bs)
            n = b["feats"].shape[0]
            for k in ("lat_label", "lon_label", "route_label"):
                b[k] = torch.zeros(n, dtype=torch.long)
            return b
    return D


rows = {}
for tag, lab in (("labelled", True), ("unlabelled", False)):
    m = load()
    if lab:
        m.SmokeData = labelled(m.SmokeData)
    d = OUT / tag
    assert m.main(["--smoke", "--steps", "2", "--log-every", "1", "--bs", "2",
                   "--device", "cpu", "--seed", "0", "--out", str(d)]) == 0
    rows[tag] = [json.loads(x) for x in
                 (d / "train_log.jsonl").read_text("utf-8").splitlines()
                 if x.strip()]
    cfgj = json.loads((d / "config.json").read_text("utf-8"))
    rows[tag + "_weights"] = {
        "w_tac_label": cfgj["cfg"]["w_tac_label"],
        "w_str_label": cfgj["cfg"]["w_str_label"]}

# the hand computation, printed beside the row so the reader can check it
for r in rows["labelled"]:
    w = rows["labelled_weights"]
    hand = (w["w_tac_label"] * (r["loss_lat_label"] + r["loss_lon_label"]) / 2
            + w["w_str_label"] * r["loss_route_label"]) / r["loss"]
    r["_hand_computed_share"] = hand
    r["_hand_matches"] = abs(hand - r["loss_label_weighted_share"]) <= 1e-12

(OUT / "l1_train_log_rows.json").write_text(
    json.dumps(rows, indent=1), encoding="utf-8")
print(json.dumps(rows["labelled"][0], indent=1))
print("--- unlabelled ---")
print(json.dumps({k: rows["unlabelled"][0][k] for k in
                  ("step", "loss", "loss_lat_label", "loss_lon_label",
                   "loss_route_label", "loss_label_weighted_share")}, indent=1))
print("wrote", OUT / "l1_train_log_rows.json")
