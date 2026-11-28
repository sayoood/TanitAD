"""E-DEC-29 — IS AGENT COUNTING THE ONLY ENVIRONMENT WE CARRY? (PI challenge)

⛔ THE CRITICISM THAT PROMPTED THIS, AND IT IS CORRECT. Every "environment"
claim in this campaign rests on TWO targets: `n_agents` (works, +0.3881) and a
lead descriptor (fails at every arm). `n_agents` is a **permutation-invariant,
position-free scalar** — a latent could satisfy it with a scene-clutter statistic
and know nothing about WHERE anything is. That is a thin basis for "the world
model learned the environment", exactly as the PI said.

⭐ WHY A PROBE AND NOT A HEAD. Bounding-box heads were tried and failed, and PSG
(a supervised spatial LOSS from the same cuboids) destroyed the predictor at every
weight (E-DEC-18b/c). **A probe is a different question**: it asks what the latent
ALREADY CONTAINS, trains nothing, and cannot damage an arm. If the spatial
structure is already there, no new loss is needed; if it is absent, we know the
deficit precisely before spending GPU on a head.

⚠️ NOT CIRCULAR: `splitp30k` and `rdw8p30k` never trained with PSG, so the
per-column target is unseen by them.

THE TARGETS — the upgrade from "how many" to "where", using the SAME 8 azimuth
columns as the readout (valid because the corpus is CYLINDRICAL, so image column
is linear in azimuth; the +x fwd/+y LEFT convention is MEASURED, not assumed):

    occ_col0..7     agents per azimuth column, log1p   -> WHERE, not how many
    occ_left/ctr/rt coarse 3-bin version               -> robust aggregate
    n_free_cols     how many of the 8 columns are EMPTY -> the CORRIDOR signal;
                    the PI's "the corridor is limited by obstacle vehicles and
                    the ego will evade" is a statement about free space
    n_agents        the incumbent, carried for comparison

CONTROLS (standing rules): a CONSTANT reading exactly 0.0000, a RAW-PIXEL floor,
frozen DINOv3 as the external reference, and n printed per target.
⚠️ T0-DIAGNOSTIC, and IN-SAMPLE until the held-out corpus lands.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
# ⭐ E-DEC-31: corpus/labels/out are now ENV-OVERRIDABLE so the HELD-OUT read
# is a RE-POINT of the identical instrument, not a forked copy that could
# drift from it. Defaults reproduce the in-sample run byte-for-byte.
LEAD = pathlib.Path(os.environ.get("SPD_CORPUS",
    str(SP / "sp2/cache/slotprobe-lead130-w120-256x640cyl")))
LABELS = pathlib.Path(os.environ.get("SPD_LABELS",
    str(SP / "sp2/lead130_agents.jsonl")))
OUT = pathlib.Path(os.environ.get("SPD_OUT", str(SP / "spatialenv.json")))
SPLIT = os.environ.get("SPD_SPLIT", "IN-SAMPLE")
ARMS = ["splitp30k", "rdw8p30k", "scale1"]
N_CLIPS, F = 24, 100


def spatial_targets(ag, m):
    """per-frame spatial descriptors from the ego-frame cuboids.

    ⛔⛔ RETURNS A LABELLED-MASK, AND THE CALLER MUST APPLY IT. The original
    wrote `ag.get(i, [])` and let an UNLABELLED frame become `n_agents = 0` —
    a missing value silently wearing the costume of a real one, which is the
    C150 defect exactly. It was invisible on the in-sample corpus (coverage
    100.00 %, 0 fake zeros, because that corpus was BUILT from those labels)
    and severe on the held-out one: MEASURED 2026-08-24, **4.90 % of frames
    unlabelled, concentrated in 7 of 124 clips with <50 labelled frames and one
    with ZERO** — which in LOEO asks the probe to predict a constant 0 from real
    imagery and drives R2 sharply negative. It collapsed frozen DINOv3 (+0.2754
    -> +0.0114) just as hard as our arms, and a control that moves with the
    treatment is the tell that the MEASUREMENT changed, not the model.
    """
    from tanitad.data.psg_targets import PSG_N_COLS, azimuth_column
    cols = np.zeros((m, PSG_N_COLS))
    nag = np.zeros(m)
    lab = np.zeros(m, dtype=bool)
    for i in range(m):
        if i not in ag:
            continue                      # UNLABELLED: leave masked, never 0
        lab[i] = True
        a = ag.get(i, [])
        nag[i] = len(a)
        for q in a:
            c = azimuth_column(float(q["cx"]), float(q["cy"]))
            if c is not None:
                cols[i, c] += 1.0
    occ = np.log1p(cols)
    out = {f"occ_col{k}": occ[:, k:k + 1] for k in range(PSG_N_COLS)}
    out["occ_left"] = occ[:, :3].sum(1, keepdims=True)
    out["occ_center"] = occ[:, 3:5].sum(1, keepdims=True)
    out["occ_right"] = occ[:, 5:].sum(1, keepdims=True)
    # the CORRIDOR signal: how much of the forward field is free
    out["n_free_cols"] = (cols == 0).sum(1, keepdims=True).astype(float)
    out["n_agents"] = nag[:, None]
    return out, lab


def main() -> int:
    import v7tiny_g2 as G
    import v7tiny_probe as P

    dev = torch.device("cuda")
    LAB = {}
    with open(LABELS, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                LAB.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r.get("agents", [])
    # ⛔⛔ THE SELECTION CONFOUND, AND THE KNOB THAT REMOVES IT. The in-sample
    # corpus is named `slotprobe-LEAD130` because it was SELECTED FOR LEAD
    # PRESENCE: MEASURED 2026-08-24, 130/130 of its clips carry >=20 lead-present
    # frames (mean 96.0 of 100), against 70/122 (57 %, mean 42.2, median 29.5)
    # on the unselected held-out val set. It is also traffic-denser (n_agents
    # mean 52.96 vs 32.90). ⇒ AN ABSOLUTE R2 COMPARED ACROSS THE TWO SPLITS IS
    # CONFOUNDED BY SELECTION, not a clean generalisation read — and every
    # column including frozen DINOv3 and raw pixels drops, which is the tell.
    # SPD_MIN_LEAD restricts the held-out set to clips meeting the SAME
    # criterion, so the two corpora agree on the thing one of them was chosen
    # for. Default 0 reproduces the unmatched run exactly.
    MIN_LEAD = int(os.environ.get("SPD_MIN_LEAD", "0"))

    def _lead_frames(cid):
        ag = LAB.get(cid, {})
        return sum(1 for i in range(F)
                   if any(abs(x.get("cy", 9e9)) < 1.8 and x.get("cx", -1) > 0
                          for x in ag.get(i, [])))

    _all = [c for c in sorted(LEAD.glob("*.v2ep.pt"))
            if torch.load(c, map_location="cpu", weights_only=False)["clip_id"] in LAB]
    if MIN_LEAD > 0:
        _kept = [c for c in _all
                 if _lead_frames(torch.load(c, map_location="cpu",
                                            weights_only=False)["clip_id"]) >= MIN_LEAD]
        print(f"  LEAD-MATCHED SELECTION: {len(_kept)}/{len(_all)} clips carry "
              f">= {MIN_LEAD} lead-present frames (the in-sample corpus is "
              f"100 % by construction)", flush=True)
        _all = _kept
    clips = _all[:N_CLIPS]
    present = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    print(f"\n  E-DEC-29 · SPATIAL environment · arms {present} · {len(clips)} clips\n"
          f"  'where', not just 'how many' — 8 azimuth columns + a corridor signal\n",
          flush=True)

    COLS, TG, PIX, MASK = {}, {}, [], []
    for arm in present:
        w, _ = G.load_arm(arm, dev)
        col = []
        for c in clips:
            z, _, _ = G.encode_clip(w, c, dev, F)
            col.append(z.numpy().astype(np.float64))
            if arm == present[0]:
                d, raw, off, n_all, _ = G.frames_of(c)
                m = len(col[-1])
                imgs = [torch.from_numpy(np.asarray(
                    Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("RGB")).copy())
                    .permute(2, 0, 1).float() / 255.0 for i in range(m)]
                PIX.append(torch.nn.functional.adaptive_avg_pool2d(
                    torch.stack(imgs)[:, -3:], (8, 8)).reshape(m, -1).numpy().astype(np.float64))
                tg_, lab_ = spatial_targets(LAB[d["clip_id"]], m)
                MASK.append(lab_)
                for k, v in tg_.items():
                    TG.setdefault(k, []).append(v)
        COLS[f"z_op {arm}"] = col
        del w
        torch.cuda.empty_cache()
    # ⛔ EVERY COLUMN MUST EXIST BEFORE THE MASK IS APPLIED. The first version of
    # this fix masked `COLS` and then added `pixel (floor)` and `frozen DINOv3`
    # AFTERWARDS — so the two reference columns went in UNMASKED, at 100 rows
    # against the targets' ~92, and the panel died in the probe's `Xc.T @ Yc`
    # (899 vs 900). ⚠️ IT PASSED IN-SAMPLE, because at 100.00 % coverage the mask
    # is all-True and the no-op hid the length mismatch — **the same
    # "invisible on the only corpus it met" pattern as the C152 defect this
    # block exists to fix**. Building the full column set first makes the
    # masking loop total by construction rather than by discipline.
    COLS["pixel (floor)"] = PIX
    COLS["frozen DINOv3"] = P.dinov3_encode(clips, F, dev)

    # ⭐ APPLY THE LABELLED-MASK AND STATE THE COUNT. A clip whose coverage is
    # too thin is DROPPED rather than silently zero-filled, and how many were
    # dropped is printed — an aggregate that does not report what it compared
    # is the vacuous-freeze-check defect in another costume.
    MIN_COV = 0.80
    keep_ci = [i for i in range(len(MASK)) if MASK[i].mean() >= MIN_COV]
    dropped = len(MASK) - len(keep_ci)
    tot = sum(len(m_) for m_ in MASK)
    unl = sum(int((~m_).sum()) for m_ in MASK)
    print(f"  LABEL COVERAGE: {tot - unl}/{tot} frames labelled "
          f"({100 * (1 - unl / max(tot, 1)):.2f} %); "
          f"clips kept {len(keep_ci)}/{len(MASK)} at >= {MIN_COV:.0%} coverage, "
          f"DROPPED {dropped}", flush=True)
    if len(keep_ci) < 8:
        raise SystemExit(f"[FATAL] only {len(keep_ci)} clips clear the coverage "
                         f"floor - the probe needs >= 8 for its fit split")
    MASK = [MASK[i] for i in keep_ci]
    for k in list(COLS):
        COLS[k] = [COLS[k][i][MASK[j]] for j, i in enumerate(keep_ci)]
    for k in list(TG):
        TG[k] = [TG[k][i][MASK[j]] for j, i in enumerate(keep_ci)]
    # the constant control is built from the ALREADY-MASKED targets, so it is
    # length-correct by construction and reads exactly 0.0000 as it must.
    COLS["constant (control)"] = [np.ones((len(v), 1)) for v in TG["n_agents"]]

    # ⛔ ASSERT THE ROW COUNTS AGREE. This is the check whose ABSENCE turned a
    # column-alignment bug into a `matmul` error 200 lines away, in a panel that
    # had already spent 8 minutes encoding DINOv3. Every column and every target
    # must have identical per-clip lengths before a single probe is fit.
    nrows = [len(v) for v in TG["n_agents"]]
    for k, v in COLS.items():
        got = [len(x) for x in v]
        if got != nrows:
            raise SystemExit(f"[FATAL] column {k!r} has per-clip lengths {got} "
                             f"but the targets have {nrows} — a column was "
                             f"added after the mask was applied")
    print(f"  row-count check: {len(COLS)} columns x {len(nrows)} clips all "
          f"agree at {sum(nrows)} rows", flush=True)

    def loeo(X, Y):
        o = []
        for i in range(len(X)):
            Xf = [X[j] for j in range(len(X)) if j != i]
            Yf = [Y[j] for j in range(len(Y)) if j != i]
            o.append(P.probe(Xf + [X[i]], Yf + [Y[i]], 128)[0])
        return np.array(o, dtype=np.float64)

    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "split": SPLIT,
           "corpus": str(LEAD), "labels": str(LABELS), "arms": present,
           "label_coverage_floor": 0.80,
           "min_lead_frames": int(os.environ.get("SPD_MIN_LEAD", "0")), "method": "LOEO paired probe, PCA d_eff=128 fit on training clips only; "
                     "8 azimuth columns matching the readout, cylindrical projection, "
                     "+x fwd/+y LEFT (MEASURED convention)", "targets": {}}
    order = (["n_agents", "occ_left", "occ_center", "occ_right", "n_free_cols"]
             + [f"occ_col{k}" for k in range(8)])
    hdr = f"  {'target':<14}" + "".join(f"{a[:11]:>13}" for a in present) \
        + f"{'pixels':>10}{'DINOv3':>10}{'const':>8}{'n':>7}"
    print(hdr); print("  " + "-" * (len(hdr) - 2), flush=True)
    for tn in order:
        Y = TG[tn]
        R = {k: loeo(v, Y) for k, v in COLS.items()}
        nrow = sum(len(y) for y in Y)
        rep["targets"][tn] = {"n_rows": nrow, "n_clips": len(Y), "columns": {}}
        flo = R["pixel (floor)"]
        cells = ""
        for a in present:
            r = R[f"z_op {a}"]
            df = r - flo
            t = float(df.mean()) / max(float(df.std(ddof=1) / np.sqrt(len(df))), 1e-12)
            rep["targets"][tn]["columns"][a] = {
                "r2": round(float(r.mean()), 4),
                "delta_vs_pixel_floor": round(float(df.mean()), 4),
                "t_vs_pixel_floor": round(t, 2),
                "beats_raw_input": bool(t > 2.0)}
            cells += f"{float(r.mean()):>+13.4f}"
        for k in ("pixel (floor)", "frozen DINOv3", "constant (control)"):
            rep["targets"][tn]["columns"][k] = {"r2": round(float(R[k].mean()), 4)}
        print(f"  {tn:<14}{cells}{float(flo.mean()):>+10.4f}"
              f"{float(R['frozen DINOv3'].mean()):>+10.4f}"
              f"{float(R['constant (control)'].mean()):>+8.4f}{nrow:>7}", flush=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
