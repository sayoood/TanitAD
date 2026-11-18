"""navrb2_ctx_nofog.py -- the detected-information read of the S1R arms under the PI ruling "Drop fog for now" (2026-11-17; whead/nofog/AMEND_WH_NOFOG.md): weather is scored on clear / rain / snow,
fog-labelled windows are UNSCORED for every arm and control (n printed), and the fog logit is masked before the argmax.  A study wrapper over the frozen strat1_ctx_read.py (UNEDITED): only
label_rows (fog -> -1), split_logits (the fog column of the weather head -> -inf) and table_baselines (the table with fog steps marked invalid in BOTH splits, so the constants are the no-fog constants) are patched, in memory.
The frozen reader's LABELS_REPORT weather check (0.846) is a fog-inclusive number: it is replaced for weather by the no-fog constant recomputed here from the table (a re-derivation of the same quantity, named as such in the output).
Usage: python navrb2_ctx_nofog.py --t0 VIEW/t0 --out OUT.json [--seeds 0 1] [--n-boot 2000]   (the frozen reader's arguments)"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navrb2_lib as NL  # noqa: E402

sys.path.insert(0, NL.SR)
import strat1_ctx_read as C  # noqa: E402  (frozen)
import strat1_lib as L  # noqa: E402

FOG_ID, W = 1, C.CTX_VARS.index("weather")


def mask_table(tab):
    """a copy of the table with fog steps invalid in both splits"""
    t = dict(tab)
    for sp in ("train", "eval"):
        t[f"{sp}_weather_valid"] = np.asarray(tab[f"{sp}_weather_valid"], bool) & (np.asarray(tab[f"{sp}_weather_id"], np.int64) != FOG_ID)
    return t


def patch():
    orig_label, orig_split, orig_base = C.label_rows, C.split_logits, C.table_baselines

    def label_rows(tab, r, mask_flag=True):
        y = orig_label(mask_table(tab), r, mask_flag)
        return y

    def split_logits(lg):
        parts = [np.array(p, copy=True) for p in orig_split(lg)]
        parts[W] = parts[W].astype(np.float64)
        parts[W][:, FOG_ID] = -np.inf
        return parts

    def table_baselines(tab):
        return orig_base(mask_table(tab))
    C.label_rows, C.split_logits, C.table_baselines = label_rows, split_logits, table_baselines
    return orig_label, orig_split, orig_base


def main():
    tab = dict(np.load(L.pinned("strat_labels_table.npz", NL.SR + "/inputs")))
    base_nf = C.table_baselines(mask_table(tab))
    C.LABELS_REPORT = dict(C.LABELS_REPORT, weather=base_nf["weather"]["report_basis_acc"])      # recomputed no-fog constant (a re-derivation of the same quantity)
    patch()
    argv = sys.argv[1:]
    C.L.INP = NL.SR + "/inputs"
    C.sys.argv = ["strat1_ctx_read.py"] + argv
    C.main()
    out = argv[argv.index("--out") + 1]
    rep = json.load(open(out, encoding="utf-8"))
    ev_fog = (np.asarray(tab["eval_weather_id"], np.int64) == FOG_ID) & np.asarray(tab["eval_weather_valid"], bool) & (np.asarray(tab["eval_flag"]) == 0)
    clip_of = np.repeat(np.arange(len(tab["eval_off"])), np.asarray(tab["eval_n"], np.int64))
    rep["fog_unscored"] = {"ruling": "PI 2026-11-17 'Drop fog for now' (fog label 2/10 in the blind PI audit)", "eval_steps_fog": int(ev_fog.sum()), "eval_clips_with_fog_steps": int(len(np.unique(clip_of[ev_fog]))),
                           "scored_classes": ["clear", "rain", "snow"], "fog_logit": "masked to -inf before the argmax", "weather_constant_check": "no-fog constant recomputed from the table (no LABELS_REPORT value exists without fog)",
                           "weather_report_basis_acc_nofog": C.LABELS_REPORT["weather"]}
    json.dump(rep, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
