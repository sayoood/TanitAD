"""E-DEC-4: ENVIRONMENT decodability — other agents, not just ego motion.

⭐ WHY THIS EXISTS (PI, 2026-08-23): every decodability number so far has been
EGO state (speed, d_ego, yaw-rate). Ego motion is partly recoverable from optical
flow inside the 3-frame stack, so it is the EASY target. A world model has to
carry the ENVIRONMENT -- where the other vehicles are -- and nothing we have
measured tests that.

Targets, derived from the banked 3D agent cuboids (`sp2/lead130_agents.jsonl`,
per-frame ego-centric cx forward / cy lateral / yaw / class):

    lead_gap_m     min cx over in-lane agents (|cy| < 1.8 m, cx > 0)   LONGITUDINAL
    n_agents       how many agents are present                         SCENE DENSITY
    nearest_cy     lateral offset of the nearest agent                  LATERAL
    nearest_bear   atan2(cy, cx) of the nearest agent                   AZIMUTH

`nearest_bear` is the sharpest test of E-DEC-3: if azimuth pooling is what the
readout destroys, a BEARING target should be the most sensitive of the four.

⚠️⚠️ SCOPE CAVEAT, STATED NOT HIDDEN. Agent labels exist ONLY for the 130-clip
corpus, which is the corpus these arms TRAINED on. So:
  * the ENCODER saw these frames  -> absolute R2 is optimistic for OUR columns
  * frozen DINOv3 and the pixel floor did NOT train here -> comparisons against
    them FAVOUR us and must not be read as "we beat DINOv3"
  * the enc-vs-z_op contrast IS valid: same encoder, same exposure, the only
    difference is the pooling
The probe itself is cross-validated (LOEO over clips), so the RIDGE never sees
its scoring clip -- this is the standard frozen-encoder linear-probe protocol.

Controls: constant must read exactly 0.0000; raw-pixel floor included; n and d
printed. T0-DIAGNOSTIC.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
OUT = SP / "e_dec4_environment.json"
CACHE = SP / "sp2/cache/slotprobe-lead130-w120-256x640cyl"
LABELS = SP / "sp2/lead130_agents.jsonl"
FOV = 120.0
GRIDS = [(4, 4), (4, 10), (4, 20)]
N_CLIPS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
FRAMES = 100


def load_labels() -> dict:
    by = {}
    with LABELS.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            by.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r.get("agents", [])
    return by


def targets_for(agents_by_frame: dict, m: int) -> dict:
    lead, ncnt, ncy, nbear = [], [], [], []
    for i in range(m):
        ag = agents_by_frame.get(i, [])
        inlane = [a["cx"] for a in ag if abs(a.get("cy", 9e9)) < 1.8 and a.get("cx", -1) > 0]
        lead.append(min(inlane) if inlane else 80.0)      # 80 m = "clear road" sentinel
        ncnt.append(float(len(ag)))
        if ag:
            k = min(ag, key=lambda a: a["cx"] ** 2 + a["cy"] ** 2)
            ncy.append(float(k["cy"]))
            nbear.append(float(np.arctan2(k["cy"], max(k["cx"], 1e-3))))
        else:
            ncy.append(0.0)
            nbear.append(0.0)
    return {"lead_gap_m": np.array(lead)[:, None],
            "n_agents": np.array(ncnt)[:, None],
            "nearest_cy": np.array(ncy)[:, None],
            "nearest_bearing": np.array(nbear)[:, None]}


def pool(tok, rows, cols, gh, gw):
    f, n, d = tok.shape
    t = tok.reshape(f, gh, rows // gh, gw, cols // gw, d)
    return t.mean(axis=(2, 4)).reshape(f, gh * gw * d)


def main() -> int:
    import v7tiny_g2 as G
    import v7tiny_probe as P

    dev = torch.device("cuda")
    arm = sys.argv[1] if len(sys.argv) > 1 else "champ30k"
    LAB = load_labels()
    clips = [c for c in sorted(CACHE.glob("*.v2ep.pt"))][:N_CLIPS]
    world, st = G.load_arm(arm, dev)
    rows = 256 // world.stack.cfg.encoder.patch_size
    cols = 640 // world.stack.cfg.encoder.patch_size
    print(f"\n  E-DEC-4 ENVIRONMENT · {arm}@{st} · {len(clips)} clips · tokens {rows}x{cols}"
          f" ({FOV / cols:.2f} deg/col)")
    print("  ⚠️ IN-SAMPLE for our encoder (labels exist only on the training corpus);"
          " the enc-vs-z_op contrast is the readable part\n", flush=True)

    COLS = {f"{a}x{b}": [] for a, b in GRIDS}
    COLS["enc (full mean)"] = []
    ZOP, PIX, TG = [], [], {k: [] for k in
                            ("lead_gap_m", "n_agents", "nearest_cy", "nearest_bearing")}
    used = []
    for ci, c in enumerate(clips, 1):
        d = torch.load(c, map_location="cpu", weights_only=False)
        cid = d["clip_id"]
        if cid not in LAB:
            continue
        raw = d["jpeg_buf"].numpy().tobytes()
        off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(np.int64)
        m = min(FRAMES, len(off) - 1)
        imgs = [torch.from_numpy(np.asarray(Image.open(io.BytesIO(raw[off[i]:off[i + 1]]))
                                            .convert("RGB")).copy()).permute(2, 0, 1).float() / 255.0
                for i in range(m)]
        toks = []
        with torch.no_grad():
            for s in range(0, m, 8):
                chunk = [torch.cat([imgs[max(i - j, 0)] for j in range(2, -1, -1)], 0)
                         for i in range(s, min(s + 8, m))]
                x = torch.stack(chunk)[:, None].to(dev)
                _z, tk = world.stack.encode_window(x, return_tokens=True)
                toks.append(tk[:, 0].float().cpu().numpy())
        T = np.concatenate(toks).astype(np.float64)
        for a, b in GRIDS:
            COLS[f"{a}x{b}"].append(pool(T, rows, cols, a, b))
        COLS["enc (full mean)"].append(T.mean(1))
        z, _, _ = G.encode_clip(world, c, dev, m)
        ZOP.append(z.numpy().astype(np.float64)[:m])
        PIX.append(P.pooled_frames(c, m))
        for k, v in targets_for(LAB[cid], m).items():
            TG[k].append(v)
        used.append(cid)
        del T, toks, imgs
        if ci % 6 == 0:
            print(f"    [{ci}/{len(clips)}] {len(used)} usable", flush=True)
    DN = P.dinov3_encode(clips[:len(used)], FRAMES, dev)
    del world
    torch.cuda.empty_cache()

    COLS["z_op (incumbent readout)"] = ZOP
    COLS["pixel (floor)"] = PIX
    COLS["frozen DINOv3*"] = DN
    COLS["constant (control)"] = [np.ones((len(z), 1)) for z in ZOP]
    n = len(used)

    def loeo(X, Y):
        out = []
        for e in range(n):
            idx = [i for i in range(n) if i != e]
            Xf = [np.asarray(X[i])[:len(Y[i])] for i in idx]
            Yf = [Y[i][:len(np.asarray(X[i]))] for i in idx]
            Xs = [np.asarray(X[e])[:len(Y[e])]]
            Ys = [Y[e][:len(np.asarray(X[e]))]]
            fn = getattr(P, "probe_fit_score", None)
            out.append(fn(Xf, Yf, Xs, Ys, 128) if fn else P.probe(Xf + Xs, Yf + Ys, 128)[0])
        return np.array(out, dtype=np.float64)

    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)", "eval_tier": "T0-DIAGNOSTIC",
           "arm": arm, "step": int(st), "n_clips": n, "frames_per_clip": FRAMES,
           "scope_caveat": ("agent labels exist only on the 130-clip TRAINING corpus, so absolute "
                            "R2 is optimistic for our columns and comparisons against DINOv3/pixels "
                            "FAVOUR us; the enc-vs-z_op contrast is unaffected"),
           "estimator": "leave-one-clip-out, paired against the 4x4 incumbent pooling",
           "targets": {}}
    print(f"\n  {'target':<16}{'features':<26}{'R2':>9}{'vs 4x4':>9}{'t':>7}{'favours':>9}")
    print("  " + "-" * 78)
    for tn, Y in TG.items():
        R = {k: loeo(v, Y) for k, v in COLS.items()}
        base = R["4x4"]
        rep["targets"][tn] = {}
        for k in COLS:
            r = R[k]
            dd = r - base
            mm = float(dd.mean())
            se = float(dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else 0.0
            t = mm / max(se, 1e-12)
            rep["targets"][tn][k] = {"r2": round(float(r.mean()), 4),
                                     "delta_vs_4x4": round(mm, 4), "t": round(t, 2),
                                     "n_favouring": int((dd > 0).sum()), "n_clips": len(dd)}
            print(f"  {tn:<16}{k:<26}{float(r.mean()):>+9.4f}{mm:>+9.4f}{t:>7.2f}"
                  f"{int((dd > 0).sum()):>6}/{len(dd)}")
        print()

    ctrl = max(abs(rep["targets"][t]["constant (control)"]["r2"]) for t in TG)
    if ctrl > 1e-6:
        rep["verdict"] = f"⛔ VOID — constant control reads {ctrl}, not 0."
    else:
        wins = {t: [k for k in COLS if k.startswith("4x") and k != "4x4"
                    and rep["targets"][t][k]["t"] > 2.2] for t in TG}
        env_ok = [t for t in TG if rep["targets"][t]["enc (full mean)"]["r2"]
                  > rep["targets"][t]["pixel (floor)"]["r2"]]
        rep["verdict"] = (
            f"azimuth widening beats the 4x4 incumbent on: "
            f"{ {t: w for t, w in wins.items() if w} }. "
            f"Encoder beats the pixel floor on {env_ok or 'NO environment target'} "
            f"(in-sample, see caveat).")
    print(f"  VERDICT: {rep['verdict']}")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
