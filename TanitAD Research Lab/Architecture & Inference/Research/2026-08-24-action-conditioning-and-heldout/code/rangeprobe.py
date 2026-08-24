"""E-DEC-35 — IS METRIC RANGE MISSING FROM *OUR* ENCODER, OR FROM MONOCULAR VIDEO?

⛔ THE FINDING THAT FORCES THIS QUESTION (E-DEC-32c, MEASURED 2026-08-24 on the
lead-matched held-out set, 23 clips / ~2,000 rows — POWERED for the first time):
**nothing in the programme carries the lead vehicle's metric range or closing
rate.** A CONSTANT beats every arm — `lead_range_m` t 11.51 (23/23 clips),
`lead_closing` t 28.36 (23/23) — while on the *same rows* `n_agents` is carried
at **+0.2215 / +0.2704 / +0.2385**. So the deficit is specific: we carry **WHERE**
(occupancy +0.3351, corridor +0.2080, count +0.27) and not **HOW FAR**.

⭐⭐ WHY THIS MATTERS MORE THAN THE OTHER OPEN ITEMS. The PI's physics goal — *"if
the vehicle in front decelerates, the ego must react on it"* — needs BOTH (1) an
action-conditioned predictor (O11-CF is testing that) and (2) the latent to carry
the lead's range and closing rate. **(2) is missing, and no amount of (1) supplies
it**: a perfectly action-conditioned world model cannot reason about a gap it
cannot see. This probe decides where the fix has to go.

THE QUESTION, AND WHY IT IS DECIDABLE WITHOUT TRAINING ANYTHING:

    * If frozen DINOv3 CARRIES range and we do not  -> the deficit is OUR ENCODER.
      The fix is representational (better trunk, or the frozen-DINOv3-trunk
      architecture already under consideration in Phase 3 of the plan).
    * If frozen DINOv3 ALSO FAILS                   -> monocular metric range is
      not linearly available from a single frame at all, and the fix must be
      EXPLICIT: a supervised range target, or temporal/motion-parallax cues.
      A better trunk would not have helped and the GPU is better spent elsewhere.

COLUMNS (all on identical rows, identical LOEO folds, PCA d_eff = 128 fit on the
training clips only):

    z_op splitp30k     the programme's best CONTENT carrier
    z_op rdw8p30k      the programme's best PREDICTOR
    tokens 16x8        the UNPROJECTED encoder tokens — separates "the encoder
                       never had range" from "the readout's 128->64 projection
                       threw it away" (E-DEC-25/26 localised the loss there, and
                       E-DEC-26 refuted the "oriented away" mechanism)
    frozen DINOv3      THE DECIDING COLUMN
    pixels (floor)     raw input; a representation that does not beat it added
                       nothing
    constant (control) reads EXACTLY 0.0000 by construction

TARGETS, with a POSITIVE CONTROL that is the whole point of the panel:

    lead_range_m   the question
    lead_closing   d(range)/dt — the PI's deceleration signal
    n_agents       ⭐ THE POSITIVE CONTROL. It is CARRIED (+0.27 held-out). If
                   n_agents works and range fails on the SAME rows, the SAME
                   columns and the SAME folds, the failure is a property of the
                   TARGET and not of the setup. Without it a null here is
                   uninterpretable — it could just be a broken panel.

⚠️ SCOPE. T0-DIAGNOSTIC. Held-out, lead-matched (the C153 selection rule).
⚠️ A LINEAR NEGATIVE IS ONLY A NEGATIVE ABOUT LINEAR MAPS — the standing rule.
   A ridge probe failing proves range is not LINEARLY decodable, not that it is
   unlearnable. The verdict below states the function class.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
LEAD = Path(os.environ.get("SPD_CORPUS",
                           str(SP / "sp2/cache/physicalai-val130-heldout")))
LABELS = Path(os.environ.get("SPD_LABELS", str(SP / "sp2/val130_agents.jsonl")))
OUT = Path(os.environ.get("SPD_OUT", str(SP / "rangeprobe.json")))
MIN_LEAD = int(os.environ.get("SPD_MIN_LEAD", "20"))
ARMS = ["splitp30k", "rdw8p30k"]
N_CLIPS, F = 24, 100
TOK_ROWS, TOK_COLS = 16, 40


def targets(ag, m):
    """lead range (nan when no lead), closing rate, agent count, and the mask."""
    rng_, nag = [], []
    for i in range(m):
        if i not in ag:
            rng_.append(np.nan); nag.append(np.nan); continue
        a = ag.get(i, [])
        inl = [x["cx"] for x in a
               if abs(x.get("cy", 9e9)) < 1.8 and x.get("cx", -1) > 0]
        rng_.append(min(inl) if inl else np.nan)
        nag.append(float(len(a)))
    r = np.array(rng_)
    clo = np.full(m, np.nan)
    clo[1:] = r[1:] - r[:-1]
    return r, clo, np.array(nag)


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

    def lead_frames(cid):
        ag = LAB.get(cid, {})
        return sum(1 for i in range(F)
                   if any(abs(x.get("cy", 9e9)) < 1.8 and x.get("cx", -1) > 0
                          for x in ag.get(i, [])))

    allc = [c for c in sorted(LEAD.glob("*.v2ep.pt"))
            if torch.load(c, map_location="cpu", weights_only=False)["clip_id"] in LAB]
    if MIN_LEAD > 0:
        allc = [c for c in allc
                if lead_frames(torch.load(c, map_location="cpu",
                                          weights_only=False)["clip_id"]) >= MIN_LEAD]
    clips = allc[:N_CLIPS]
    present = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    print(f"\n  E-DEC-35 · IS METRIC RANGE MISSING FROM OUR ENCODER, OR FROM "
          f"MONOCULAR VIDEO?\n  arms {present} · {len(clips)} lead-matched "
          f"held-out clips (>= {MIN_LEAD} lead frames)\n", flush=True)

    COLS, RNG, CLO, NAG, MASK = {}, [], [], [], []
    PIX, TOK = [], []
    for arm in present:
        w, _ = G.load_arm(arm, dev)
        stk = w.stack
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
                if float(imgs[0].abs().mean()) == 0.0:
                    raise SystemExit(f"[FATAL] {c.name} all-zero frames")
                PIX.append(torch.nn.functional.adaptive_avg_pool2d(
                    torch.stack(imgs)[:, -3:], (8, 8))
                    .reshape(m, -1).numpy().astype(np.float64))
                tk = []
                with torch.no_grad():
                    for s in range(0, m, 16):
                        ch = []
                        for i in range(s, min(s + 16, m)):
                            idx = [max(i - j, 0) for j in range(G.N_STACK - 1, -1, -1)]
                            ch.append(torch.cat([imgs[k] for k in idx], 0))
                        x = torch.stack(ch).to(dev)
                        t = stk.encoder(x)
                        b = t.shape[0]
                        g = t.reshape(b, TOK_ROWS, TOK_COLS, -1).permute(0, 3, 1, 2)
                        q = torch.nn.functional.adaptive_avg_pool2d(g, (TOK_ROWS, 8))
                        tk.append(q.reshape(b, -1).float().cpu().numpy())
                TOK.append(np.concatenate(tk).astype(np.float64))
                r_, cl_, na_ = targets(LAB[d["clip_id"]], m)
                RNG.append(r_); CLO.append(cl_); NAG.append(na_)
                MASK.append(~np.isnan(r_) & ~np.isnan(na_))
        COLS[f"z_op {arm}"] = col
        del w
        torch.cuda.empty_cache()
    COLS["tokens 16x8 (unprojected)"] = TOK
    COLS["pixels (floor)"] = PIX
    COLS["frozen DINOv3"] = P.dinov3_encode(clips, F, dev)
    COLS["constant (control)"] = [np.ones((len(r), 1)) for r in RNG]

    nr = [len(r) for r in RNG]
    for k, v in COLS.items():
        got = [len(x) for x in v]
        if got != nr:
            raise SystemExit(f"[FATAL] column {k!r} lengths {got} vs targets {nr}")
    print(f"  row-count check: {len(COLS)} columns x {len(nr)} clips agree "
          f"at {sum(nr)} rows", flush=True)

    def loeo(X, Y):
        o = []
        for i in range(len(X)):
            Xf = [X[j] for j in range(len(X)) if j != i]
            Yf = [Y[j] for j in range(len(Y)) if j != i]
            o.append(P.probe(Xf + [X[i]], Yf + [Y[i]], 128)[0])
        return np.array(o, dtype=np.float64)

    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "min_lead_frames": MIN_LEAD,
           "split": "HELD-OUT, LEAD-MATCHED",
           "function_class": "ridge (LINEAR). A negative here is a negative about "
                             "LINEAR decodability only — the standing rule.",
           "targets": {}}
    order = [("lead_range_m", RNG), ("lead_closing", CLO),
             ("n_agents (POSITIVE CONTROL — known carried)", NAG)]
    hdr = (f"\n  {'target':<42}" + "".join(f"{a[:11]:>13}" for a in present)
           + f"{'tokens':>10}{'DINOv3':>10}{'pixels':>10}{'const':>8}{'n':>7}")
    print(hdr); print("  " + "-" * (len(hdr) - 3), flush=True)
    for tn, Y in order:
        mk = [MASK[i] & ~np.isnan(Y[i]) for i in range(len(Y))]
        keep = [i for i in range(len(Y)) if int(mk[i].sum()) >= 20]
        Yl = [np.nan_to_num(Y[i])[mk[i]][:, None] for i in keep]
        Cl = {k: [v[i][mk[i]] for i in keep] for k, v in COLS.items()}
        R = {k: loeo(v, Yl) for k, v in Cl.items()}
        nrow = sum(len(y) for y in Yl)
        cells = "".join(f"{float(R[f'z_op {a}'].mean()):>+13.4f}" for a in present)
        rep["targets"][tn] = {
            "n_clips": len(keep), "n_rows": nrow,
            "columns": {k: round(float(v.mean()), 4) for k, v in R.items()}}
        print(f"  {tn:<42}{cells}"
              f"{float(R['tokens 16x8 (unprojected)'].mean()):>+10.4f}"
              f"{float(R['frozen DINOv3'].mean()):>+10.4f}"
              f"{float(R['pixels (floor)'].mean()):>+10.4f}"
              f"{float(R['constant (control)'].mean()):>+8.4f}{nrow:>7}", flush=True)

    # ⭐ dump the MASKED columns and targets so the NONLINEAR probe (E-DEC-36)
    # scores the IDENTICAL rows, folds and masks. Re-deriving them there would
    # let the two panels drift apart silently, and their difference is the whole
    # point of running both.
    if os.environ.get("SPD_DUMP"):
        dump_c, dump_t = {}, {}
        for tn, Y in order:
            mk = [MASK[i] & ~np.isnan(Y[i]) for i in range(len(Y))]
            keep = [i for i in range(len(Y)) if int(mk[i].sum()) >= 20]
            dump_t[tn] = np.array(
                [np.nan_to_num(Y[i])[mk[i]][:, None] for i in keep], dtype=object)
            if not dump_c:
                for k, v in COLS.items():
                    if k == "constant (control)":
                        continue
                    dump_c[k] = np.array([v[i][mk[i]] for i in keep], dtype=object)
        np.savez(SP / "rangeprobe_cache.npz",
                 cols=np.array(dump_c, dtype=object),
                 targets=np.array(dump_t, dtype=object))
        print(f"  dumped {len(dump_c)} columns x {len(dump_t)} targets -> "
              f"rangeprobe_cache.npz", flush=True)

    d3 = rep["targets"]["lead_range_m"]["columns"]["frozen DINOv3"]
    verdict = ("OUR ENCODER — frozen DINOv3 carries range and we do not; the fix "
               "is representational" if d3 > 0.05 else
               "MONOCULAR — frozen DINOv3 fails too, so metric range is not "
               "LINEARLY available from a single frame at all; the fix must be "
               "EXPLICIT supervision or temporal/motion-parallax cues, and a "
               "better trunk would not have helped")
    rep["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
