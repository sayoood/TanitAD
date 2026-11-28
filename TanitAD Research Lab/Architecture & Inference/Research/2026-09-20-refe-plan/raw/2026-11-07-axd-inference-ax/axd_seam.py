#!/usr/bin/env python3
"""YAXD multi-arm seam (PREREG_AXD.md section 3). Runs the PRODUCTION seam `eval/refe_navtest_seam.py::main` unchanged; the only change is a class-level wrap of
`REFePlanner.infer` (planner.py:459) that, per token, (1) runs the untouched call first (A0: unpatched `_ego_vec`, fresh trunk) and returns ITS result to the seam, so the seam's own
output file IS the A0 seam; (2) then re-runs `infer` with `_ego_vec` (planner.py:440, the only builder of the ego input) patched to feed a transformed slot 2, for every arm.
Trunk memo on `model.backbone` (as refe_ego_probe.py): IDT runs with a FRESH trunk (determinism + identity control), IDTm and the rest through the memo.

    python axd_seam.py --arms-out <dir> [--regress none|transform_not_applied|idt_perturbed|memo_stale|ax0_leak] -- <refe_navtest_seam.py args incl. --tokens --export --out>
Writes <arms-out>/<ARM>/seam.npz + seam.report.json (seam format, A0's token order), <arms-out>/records.npz (fed ego per arm, CRUISE gain arms), <arms-out>/seam_meta.json.
Last line: ZZAXDSEAM OK|FAILED.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import axd_common as A  # noqa: E402

C = A.C
sys.path.insert(0, str(C.EVAL))
import ax_analyze as AA  # noqa: E402  (YAX, read-only: cruise_inputs = PREREG_AXOFF section 4's CRUISE definition)

UNPERT = ("IDT", "IDTm") + A.CANDIDATES


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    k = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-out", required=True)
    ap.add_argument("--regress", default="none", choices=("none", "transform_not_applied", "idt_perturbed", "memo_stale", "ax0_leak"))
    a = ap.parse_args(argv[:k])
    seam_args = argv[k + 1:]
    sa = dict(zip(seam_args[::2], seam_args[1::2]))
    out = Path(a.arms_out)
    out.mkdir(parents=True, exist_ok=True)
    exp_path = sa["--export"]
    doc = json.load(gzip.open(exp_path, "rt", encoding="utf-8"))["tokens"]
    want = json.load(open(sa["--tokens"], encoding="utf-8"))["tokens"]
    toks = [t for t in doc if t in set(want)]
    assert Path(exp_path).resolve() == Path(C.export_path()).resolve(), (exp_path, C.export_path())
    _, _, _, _, cr = AA.cruise_inputs(toks)
    cruise = {t for t, c in zip(toks, cr) if c}
    hist = {t: A.history_ax(doc, t) for t in toks}
    print(f"YAXD seam: {len(toks)} tokens, {len(cruise)} CRUISE, regress={a.regress}", flush=True)

    import torch
    import refe_navtest_seam as S
    import planner as PLM
    P = PLM.REFePlanner
    orig_infer, orig_ego, orig_init = P.infer, P._ego_vec, P.__init__
    state = {"spec": None}
    memo_stats = {"hits": 0, "misses": 0, "bypass": 0}
    REC: dict = {}

    class Memo(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner, self.bypass, self.last_in, self.last_out = inner, True, None, None

        def forward(self, x):
            if not self.bypass and self.last_in is not None and self.last_in.shape == x.shape and torch.equal(self.last_in, x):
                memo_stats["hits"] += 1
                o = self.last_out.clone()
                return o + 1e-3 if a.regress == "memo_stale" else o
            memo_stats["bypass" if self.bypass else "misses"] += 1
            o = self.inner(x)
            self.last_in, self.last_out = x.clone(), o.clone()
            return o

    def init_patched(self, *args, **kw):
        orig_init(self, *args, **kw)
        self._axd_memo = Memo(self.model.backbone)
        self.model.backbone = self._axd_memo

    def ego_patched(self, ego):
        v = orig_ego(self, ego)
        sp = state["spec"]
        if sp is None or a.regress == "transform_not_applied":
            return v
        t, d, mode = sp
        tok = self._scenario._initial_lidar_token
        new = A.transform("IDT" if (t == "AX0" and a.regress == "ax0_leak") else t, float(v[0, A.AX]), hist[tok][0], d, mode)
        if a.regress == "idt_perturbed" and t == "IDT" and d == 0.0:
            new = np.float32(new + np.float32(1e-3))
        v = v.clone()
        v[0, A.AX] = float(new)
        return v

    def infer_multi(self, ego, img):
        tok = self._scenario._initial_lidar_token
        memo = self._axd_memo
        state["spec"], memo.bypass = None, True
        res = orig_infer(self, ego, img)                                  # A0: the seam's own call, unpatched ego, fresh trunk
        a0_inputs = self.last_inputs
        rec = {"raw": a0_inputs[0].detach().cpu().numpy()[0].copy(),
               "A0": S.to_navsim(self.executed(res[0], res[2]).float().cpu().numpy()).astype(np.float32), "arms": {}}
        arms = [("IDT", "IDT", 0.0, "inst", True), ("IDTm", "IDT", 0.0, "inst", False)] + [(t, t, 0.0, "inst", False) for t in A.CANDIDATES]
        if tok in cruise:
            arms += [(n, t, d, m, False) for n, t, d, m in A.gain_arms()]
        for name, t, d, mode, fresh in arms:
            state["spec"], memo.bypass = (t, d, mode), fresh
            traj, _s, kk = orig_infer(self, ego, img)
            rec["arms"][name] = (self.last_inputs[0].detach().cpu().numpy()[0].copy(),
                                 S.to_navsim(self.executed(traj, kk).float().cpu().numpy()).astype(np.float32), int(kk))
        state["spec"], memo.bypass = None, True
        self.last_inputs = a0_inputs
        REC[tok] = rec
        return res

    P.__init__, P._ego_vec, P.infer = init_patched, ego_patched, infer_multi
    t0 = time.time()
    sys.argv = ["refe_navtest_seam.py"] + seam_args
    rc = S.main()
    z = np.load(sa["--out"])
    order = [str(t) for t in z["token"]]
    rep = json.load(open(Path(sa["--out"]).with_suffix(".report.json")))
    miss = [t for t in order if t not in REC]
    for arm in UNPERT:
        d = out / arm
        d.mkdir(exist_ok=True)
        np.savez(d / "seam.npz", token=z["token"], fingerprint=z["fingerprint"], poses=np.stack([REC[t]["arms"][arm][1] for t in order]), sampling=z["sampling"], arm=np.array(f"REFe_{arm}"))
        json.dump(dict(rep, transform=arm, misses=int(rep.get("misses", 0)) + len(miss), source="axd_seam.py multi-arm wrap"), open(d / "seam.report.json", "w"), indent=1)
    ctoks = [t for t in order if t in cruise]
    gnames = [n for n, *_ in A.gain_arms()]
    hx = np.asarray([hist[t][0] for t in order])
    np.savez(out / "records.npz", token=np.array(order), raw=np.stack([REC[t]["raw"] for t in order]), a0_poses=np.stack([REC[t]["A0"] for t in order]),
             a0_seam_poses=z["poses"], hist=hx, ax_export_t0=np.asarray([hist[t][1] for t in order]),
             **{f"fed__{n}": np.stack([REC[t]["arms"][n][0] for t in order]) for n in UNPERT},
             **{f"pick__{n}": np.asarray([REC[t]["arms"][n][2] for t in order]) for n in UNPERT},
             cruise_token=np.array(ctoks), cruise_raw=np.stack([REC[t]["raw"] for t in ctoks]) if ctoks else np.zeros((0, 7), np.float32),
             cruise_hist=np.asarray([hist[t][0] for t in ctoks]) if ctoks else np.zeros((0, 3)),
             **{f"gfed__{n}": (np.stack([REC[t]["arms"][n][0] for t in ctoks]) if ctoks else np.zeros((0, 7), np.float32)) for n in gnames},
             **{f"gpose__{n}": (np.stack([REC[t]["arms"][n][1] for t in ctoks]) if ctoks else np.zeros((0, 8, 3), np.float32)) for n in gnames})
    meta = {"rc": rc, "regress": a.regress, "n_tokens": len(order), "n_cruise": len(ctoks), "missing_records": miss[:10], "memo": memo_stats, "seconds": round(time.time() - t0, 1),
            "seam_args": seam_args, "seam_misses": rep.get("misses"), "frame_control": rep.get("frame_control")}
    json.dump(meta, open(out / "seam_meta.json", "w"), indent=1)
    shutil.copy(Path(sa["--out"]).with_suffix(".report.json"), out / "A0_seam.report.json")
    ok = rc == 0 and not miss
    print(json.dumps(meta)[:1500])
    print(f"ZZAXDSEAM {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
