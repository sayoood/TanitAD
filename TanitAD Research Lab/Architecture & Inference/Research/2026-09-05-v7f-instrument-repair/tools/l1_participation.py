"""L1 (non-collapse) participation on the CORPUS-MATCHED bar.  0-GPU by construction.

WHY THIS EXISTS
---------------
The programme's frozen-DINOv3 "participation floor" reads 5.756 / 8.56 / 20.23 / 40.77
in four different banked artifacts, and the reconcile artifacts concluded "do not fail
any arm on it".  The four values are NOT four readings of one quantity: each was taken
on a DIFFERENT (corpus sample x pooling) pair.  Participation of a FROZEN encoder is a
property of the evaluated sample's scene diversity, not of the encoder -- so a floor is
only meaningful against arms measured on the SAME clips.

Provenance of the four, established from source (see RESULT.md):
  5.756  floor_valclips.py  -> v7tiny_probe.dinov3_encode, frame-mean of the 640 patch
                               tokens, corpus physicalai-val-w120-256x640cyl sorted()[:12]
                               x 120 frames, n=1440 d=1024, top8 0.8387.
  8.56   v6F_v7tiny_rank_probe.txt:41, n=1440 d=1024, top8 0.783 -- on TWELVE CLIPS THAT
                               EXIST IN NO CACHE ON THIS BOX (ids 001d413e-2 .. 729c7c83-8;
                               control-verified absence).  Emitter script not in the repo.
  20.23  floor_reconcile.py, slotprobe-lead130 corpus (130 clips / 5617 frames), banked
                               DINOv3 fields, subsampled to n=1440.  20.516 at full n.
  40.77  E_TRUNK_3_LADDER.md:20, `dino_pooled` CELLS (a flattened spatial grid, not a
                               frame mean) -- top8 0.348, a different pooling entirely.

The arms in gateb_panel.json / v7tiny_val_rank_5way.json were measured by
full_panel.py:88,103 -- sorted(VAL)[:12] x 120 frames on physicalai-val-w120-256x640cyl,
n=1440.  That is EXACTLY floor_valclips.py's sample.  => 5.756 is the only one of the
four measured on the arms' own evaluation clips, and therefore the only admissible bar
for those arms.

WHAT THIS ADDS
--------------
Row 2 of the v7f status board records that NONE of the three arms carrying T1 numbers
(emao14_30k, emao14_30k_tauramp, o14fut30k) nor postrain30k_freeze has ever been measured
on L1.  This measures them, on the corpus-matched sample, with the same instrument.

CONTROLS THAT MUST READ KNOWN VALUES (the panel is inadmissible without them)
  * rdw8p30k MUST reproduce gateb_panel.json's participation_val 25.583.  It is the same
    corpus, same n, same instrument -- if it does not reproduce, the local cache is not
    the cache full_panel.py read and NO comparison below is admissible.
  * splitp30k is NOT available on this box; it is reported as missing, never imputed.
  * effective_rank is emitted BESIDE participation only to show the C132 divergence.
    It is NEVER the criterion (it passes a representation with 55% of its energy in one
    direction).

The clip ids are written into the artifact.  The entire four-value confusion above exists
because no artifact named its corpus SAMPLE, so this one does.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import p2_leak_rescore as P  # noqa: E402  (sets CUDA_VISIBLE_DEVICES=-1 before torch)
import numpy as np  # noqa: E402

ASSETS = P.ASSETS
EXTRA_CKPTS = {
    "postrain30k_freeze": ASSETS / "v7tiny_postrain30k_freeze/ckpt.pt",
    "emao14_30k": ASSETS / "v7tiny_emao14_30k/ckpt.pt",
    "emao14_30k_tauramp": ASSETS / "v7tiny_emao14_30k_tauramp/ckpt.pt",
    "o14fut30k": ASSETS / "v7tiny_o14fut30k/ckpt.pt",
    "splitp30k": ASSETS / "v7tiny_splitp30k/ckpt.pt",
}

# The four candidate bars, each with the sample that produced it.
BARS = {
    "corpus_matched_5.756": 5.756,
    "code_floor_8.56": 8.56,
    "lead130_20.23": 20.23,
    "e_trunk3_cells_40.77": 40.77,
}
CONTROL_EXPECT = {"rdw8p30k": 25.583}     # gateb_panel.json, same corpus/n/instrument
CONTROL_TOL = 0.02                        # 2 % -- a reproduction, not an approximation


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="L1 participation, corpus-matched (0-GPU)")
    ap.add_argument("--arms", default="rdw8p30k,postrain30k,postrain30k_freeze,"
                                      "emao14_30k,emao14_30k_tauramp,o14fut30k")
    ap.add_argument("--nclips", type=int, default=12)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--stack", default=r"C:\Users\Admin\tanitad-wt\stack")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    tf = P.guard_and_stack(a.stack)
    import torch
    from tanitad.models.v6 import spectrum_report
    from tanitad.eval.spectral import effective_rank

    clips = sorted(P.ACTDIV_CORPUS.glob("*.v2ep.pt"))[: a.nclips]
    if not clips:
        print(f"[REFUSED] no clips under {P.ACTDIV_CORPUS}", file=sys.stderr)
        return 2
    clip_ids = [c.name[:10] for c in clips]
    print(f"  corpus {P.ACTDIV_CORPUS.name}  sorted()[:{a.nclips}]  frames {a.frames}")
    print(f"  clip ids: {' '.join(clip_ids)}", flush=True)

    out = P.stamp({
        "subcommand": "L1 participation (corpus-matched)",
        "_spec": "TanitAD Research Lab/Architecture & Inference/Research/"
                 "2026-09-05-v7f-instrument-repair/SPEC.md",
        "corpus": str(P.ACTDIV_CORPUS),
        "corpus_selection": f"sorted(glob('*.v2ep.pt'))[:{a.nclips}]",
        "clip_ids": clip_ids,
        "frames_per_clip": a.frames,
        "instrument": "tanitad.models.v6.spectrum_report -> participation_ratio "
                      "(centred covariance, p prop sigma^2)",
        "representation": "z_op via world.encode_window (n_stack 3) -- the READOUT, "
                          "d 2048.  NB the DINOv3 bars are ENCODER readings at d 1024; "
                          "see RESULT.md for that residual scope caveat.",
        "bars": BARS,
        "admissible_bar": "corpus_matched_5.756",
        "admissible_bar_reason": "floor_valclips.py measured it on THIS corpus, THIS "
                                 "selection, THIS n, THIS instrument.  The other three "
                                 "were measured on different clip samples or a different "
                                 "pooling and are inadmissible as a bar for these arms.",
        "rank_necessary_not_sufficient": "C131 -- flagship v1-era holds the programme's "
                                         "highest participation and had no environment "
                                         "interpretation.  Never pass an arm on rank alone.",
        "arms": {}, "arms_missing": [], "controls": {},
    })

    for arm in a.arms.split(","):
        arm = arm.strip()
        if not arm:
            continue
        ck = EXTRA_CKPTS.get(arm) or P.CKPTS.get(arm)
        if ck is None or not pathlib.Path(ck).is_file():
            print(f"  [MISSING] {arm}: no checkpoint at {ck}", flush=True)
            out["arms_missing"].append({"arm": arm, "expected_at": str(ck)})
            continue
        t0 = time.time()
        world, step, prov = P.load_arm(arm, ckpt_override=str(ck))
        rows = []
        for c in clips:
            z = P.encode_clip(world, str(c), a.frames)[2]
            rows.append(z.float().numpy().astype(np.float64))
        del world
        A = np.concatenate(rows)
        # content assertion -- a zero/non-finite bank scores like a valid one
        if not np.isfinite(A).all() or float(np.abs(A).mean()) == 0.0:
            print(f"[REFUSED] {arm}: non-finite or all-zero latents", file=sys.stderr)
            return 2
        rep = spectrum_report(torch.from_numpy(A).float())
        zc = torch.from_numpy(A - A.mean(0, keepdims=True))
        er = effective_rank(torch.linalg.svdvals(zc))
        pr = float(rep["participation_ratio"])
        row = {**prov, "n": int(A.shape[0]), "d": int(A.shape[1]),
               "participation_ratio": round(pr, 3),
               "top8_share": round(float(rep["top_k_share"]), 4),
               "rank_ceiling": int(rep["rank_ceiling"]),
               "effective_rank_NOT_THE_CRITERION": round(float(er), 3),
               "feature_abs_mean": round(float(np.abs(A).mean()), 4),
               "verdict_vs_bars": {k: ("PASS" if pr >= v else "FAIL")
                                   for k, v in BARS.items()},
               "ratio_to_admissible_bar": round(pr / BARS["corpus_matched_5.756"], 3),
               "seconds": round(time.time() - t0, 1)}
        out["arms"][arm] = row
        print(f"  {arm:<22} step {step:>6}  n {row['n']:>5} d {row['d']:>5}  "
              f"PR {pr:8.3f}  top8 {row['top8_share']:.4f}  "
              f"effrank {row['effective_rank_NOT_THE_CRITERION']:7.3f}  "
              f"({row['seconds']:.0f}s)", flush=True)

    # ---- the control that makes the panel admissible ------------------------
    for arm, expect in CONTROL_EXPECT.items():
        got = out["arms"].get(arm, {}).get("participation_ratio")
        if got is None:
            out["controls"][arm] = {"expected": expect, "got": None,
                                    "status": "NOT_RUN -- panel INADMISSIBLE"}
            continue
        rel = abs(got - expect) / expect
        out["controls"][arm] = {
            "expected": expect, "expected_source": "gateb_panel.json participation_val",
            "got": got, "rel_err": round(rel, 4), "tol": CONTROL_TOL,
            "status": "REPRODUCED" if rel <= CONTROL_TOL else "DID_NOT_REPRODUCE",
            "meaning": ("the local cache IS the cache full_panel.py read; the panel is "
                        "corpus-matched and admissible")
            if rel <= CONTROL_TOL else
            ("the local cache is NOT the cache full_panel.py read -- every comparison "
             "in this artifact is INADMISSIBLE until the corpus is reconciled"),
        }
        print(f"\n  CONTROL {arm}: expected {expect}  got {got}  "
              f"rel {rel:.4f}  -> {out['controls'][arm]['status']}", flush=True)

    out["_tanitad_imported_from"] = tf
    outp = pathlib.Path(a.out) if a.out else HERE.parent / "raw" / "l1_participation.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
