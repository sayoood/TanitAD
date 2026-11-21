"""E-DEC-8(a) READ-OUT: did an EXTERNAL target put the SCENE into the encoder?

The distilled arm was trained on NOTHING but MSE into frozen DINOv3's 4x8 cells.
It is deliberately NOT a world model (no O5, no O6, no actions) and must never be
quoted as one. It answers exactly one question: can an external target make THIS
encoder carry environment content, when every self-referential term has failed to?

Compared on identical probes against:
  * `rdw8`               our best two-term arm (the incumbent objective)
  * frozen DINOv3        the teacher -- an upper bound for distillation
  * raw pixels           the floor a learned representation must clear
  * constant             must read EXACTLY 0.0000 or the panel is void

EGO  -> physicalai-val, genuinely held out.
ENV  -> the 130-clip corpus (the only one with 3D agent cuboids). ⚠️ the distilled
        encoder AND rdw8 both trained there, so absolute R2 is optimistic for both
        and the ARM-vs-ARM contrast is the readable part. DINOv3 never trained on
        anything of ours, so it is not flattered.

⭐ THE DECISION THIS FEEDS: if environment decodability rises above the constant
control here, an external target is the missing ingredient and integrating it as
a trainer term (O7) is justified. If it does not rise even under PURE
distillation, the external-target route is refuted too -- cheaply, before any
trainer surgery.
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
VAL = SP / "sp2/cache/physicalai-val-w120-256x640cyl"
LEAD = SP / "sp2/cache/slotprobe-lead130-w120-256x640cyl"
LABELS = SP / "sp2/lead130_agents.jsonl"
DIST = SP / "v7tiny_distill/ckpt.pt"
OUT = SP / "e_dec8a_distill_readout.json"
GH, GW, H, W, PATCH = 4, 8, 256, 640, 16
F = 100


def env_targets(ag, m):
    lead, ncnt = [], []
    for i in range(m):
        a = ag.get(i, [])
        inl = [x["cx"] for x in a if abs(x.get("cy", 9e9)) < 1.8 and x.get("cx", -1) > 0]
        lead.append(min(inl) if inl else 80.0)
        ncnt.append(float(len(a)))
    return {"lead_gap_m": np.array(lead)[:, None], "n_agents": np.array(ncnt)[:, None]}


def main() -> int:
    import v7tiny_g2 as G
    import v7tiny_probe as P
    from tanitad.models.v6 import V6Stack

    dev = torch.device("cuda")
    assert DIST.is_file(), f"distilled ckpt missing: {DIST}"
    ck = torch.load(DIST, map_location="cpu", weights_only=False)
    dstack = V6Stack(ck["cfg"]).to(dev).eval()
    dstack.load_state_dict(ck["model"])
    print(f"\n  E-DEC-8(a) read-out · distilled arm @ step {ck['step']}\n", flush=True)

    LAB = {}
    with LABELS.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            LAB.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r.get("agents", [])

    def distill_cells(path, m):
        d = torch.load(path, map_location="cpu", weights_only=False)
        raw = d["jpeg_buf"].numpy().tobytes()
        off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(np.int64)
        mm = min(m, len(off) - 1)
        imgs = [torch.from_numpy(np.asarray(Image.open(io.BytesIO(raw[off[i]:off[i + 1]]))
                                            .convert("RGB")).copy()).permute(2, 0, 1).float() / 255.0
                for i in range(mm)]
        out = []
        rows, cols = H // PATCH, W // PATCH
        with torch.no_grad():
            for s in range(0, mm, 8):
                ch = [torch.cat([imgs[max(i - j, 0)] for j in (2, 1, 0)], 0)
                      for i in range(s, min(s + 8, mm))]
                x = torch.stack(ch)[:, None].to(dev)
                _z, tok = dstack.encode_window(x, return_tokens=True)
                t = tok[:, 0]
                b, _, dd = t.shape
                c = t.reshape(b, GH, rows // GH, GW, cols // GW, dd).mean(dim=(2, 4))
                out.append(c.reshape(b, GH * GW * dd).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)", "eval_tier": "T0-DIAGNOSTIC",
           "arm": "distill", "step": int(ck["step"]),
           "caveat": "distillation-only arm; NOT a world model. ENV block is in-sample for "
                     "distill and rdw8, not for DINOv3/pixels.",
           "blocks": {}}

    for block, cache, n_clips, held in (("EGO (held-out val)", VAL, 12, True),
                                        ("ENV (train corpus, in-sample)", LEAD, 24, False)):
        clips = sorted(cache.glob("*.v2ep.pt"))[:n_clips]
        if not held:
            clips = [c for c in clips
                     if torch.load(c, map_location="cpu", weights_only=False)["clip_id"] in LAB]
        DIS, RD8, PIX, PO, TG = [], [], [], [], {}
        w8, _ = G.load_arm("rdw8", dev)
        for c in clips:
            d = torch.load(c, map_location="cpu", weights_only=False)
            z, _, _ = G.encode_clip(w8, c, dev, F)
            m = len(z)
            RD8.append(z.numpy().astype(np.float64))
            DIS.append(distill_cells(c, m)[:m])
            PO.append(d["poses"].numpy().astype(np.float64)[:m])
            PIX.append(P.pooled_frames(c, m))
            if not held:
                for k, v in env_targets(LAB[d["clip_id"]], m).items():
                    TG.setdefault(k, []).append(v)
        del w8
        torch.cuda.empty_cache()
        if held:
            TG = {"speed": [p[:, 3:4] for p in PO],
                  "d_ego": [np.concatenate([np.diff(p[:, :2], axis=0), np.zeros((1, 2))])
                            for p in PO]}
        COLS = {"distill (DINOv3 target)": DIS, "rdw8 (two-term)": RD8,
                "frozen DINOv3 (teacher)": P.dinov3_encode(clips, F, dev),
                "pixel (floor)": PIX,
                "constant (control)": [np.ones((len(p), 1)) for p in PO]}
        n = len(PO)

        def loeo(X, Y):
            o = []
            for e in range(n):
                idx = [i for i in range(n) if i != e]
                Xf = [np.asarray(X[i])[:len(Y[i])] for i in idx]
                Yf = [Y[i][:len(np.asarray(X[i]))] for i in idx]
                Xs = [np.asarray(X[e])[:len(Y[e])]]
                Ys = [Y[e][:len(np.asarray(X[e]))]]
                fn = getattr(P, "probe_fit_score", None)
                o.append(fn(Xf, Yf, Xs, Ys, 128) if fn else P.probe(Xf + Xs, Yf + Ys, 128)[0])
            return np.array(o, dtype=np.float64)

        print(f"  === {block} · {n} clips ===")
        print(f"  {'target':<12}{'column':<28}{'R2':>9}{'vs rdw8':>10}{'t':>7}{'favours':>9}")
        print("  " + "-" * 78, flush=True)
        rep["blocks"][block] = {}
        for tn, Y in TG.items():
            R = {k: loeo(v, Y) for k, v in COLS.items()}
            base = R["rdw8 (two-term)"]
            rep["blocks"][block][tn] = {}
            for k in COLS:
                dd = R[k] - base
                mm = float(dd.mean())
                se = float(dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else 0.0
                t = mm / max(se, 1e-12)
                rep["blocks"][block][tn][k] = {"r2": round(float(R[k].mean()), 4),
                                               "delta_vs_rdw8": round(mm, 4), "t": round(t, 2),
                                               "n_favouring": int((dd > 0).sum()), "n": len(dd)}
                print(f"  {tn:<12}{k:<28}{float(R[k].mean()):>+9.4f}{mm:>+10.4f}{t:>7.2f}"
                      f"{int((dd > 0).sum()):>6}/{len(dd)}")
            print()

    env = rep["blocks"].get("ENV (train corpus, in-sample)", {})
    wins = [tn for tn, v in env.items()
            if v["distill (DINOv3 target)"]["r2"] > 0
            and v["distill (DINOv3 target)"]["r2"] > v["pixel (floor)"]["r2"]]
    rep["verdict"] = (
        f"distilled encoder clears BOTH zero and the pixel floor on environment target(s): "
        f"{wins or 'NONE'}. "
        + ("⭐ AN EXTERNAL TARGET PUTS THE SCENE IN — integrating it as a trainer term is "
           "justified, and E-DEC-7 is the right root cause."
           if wins else
           "⛔ Even PURE distillation into DINOv3 fails to give this encoder environment "
           "content ⇒ the external-target route is refuted too; the limit is the ENCODER "
           "(0.97M params, depth 3), not the objective."))
    print(f"  VERDICT: {rep['verdict']}")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
