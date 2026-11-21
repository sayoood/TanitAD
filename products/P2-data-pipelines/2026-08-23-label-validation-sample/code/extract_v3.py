"""Validation sample v3 — every layer, larger frames, full strategic horizon.

Adds over v2, all from PI feedback 2026-08-23:
 * the ego MANOEUVRE analysis (`ego_manoeuvre`): instantaneous-curvature turn
   detection, stop-type (CONTROLLED / QUEUE / YIELD / LAUNCH), decel cycles;
 * ALPAMAYO CoT SEMANTICS (`alpamayo_semantics`): referents, stop reason, and
   the tactical TOKENS the CoT unblocks;
 * the guard verdict per clip;
 * goal/action ARGS rendered in full (they were missing from the report);
 * larger frames (192 px source, 8 across the strategic horizon).
"""
from __future__ import annotations

import base64
import collections
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack"))
from tanitad.data import alpamayo_semantics as S      # noqa: E402
from tanitad.data import ego_manoeuvre as EM          # noqa: E402
from tanitad.data import label_guard as LG            # noqa: E402
from tanitad.refs import refc_tactical as TAC         # noqa: E402

ALPA = (REPO / "TanitAD Research Lab/Data Engineering/Implementation/incoming"
        / "2026-08-16-tactical-labels/raw/a1_alpamayo_taxonomy_per_clip.jsonl")
LBL = Path("C:/Users/Admin/tanitad-wt/_s2build/labels_v3")
OUT = Path("C:/Users/Admin/tanitad-wt/_s2build/validation")
EPC = glob.glob("C:/Users/Admin/tanitad-data/physicalai/_epcache/*/ep_*.pt")

HZ, KEY = 10.0, 78
OFFSETS = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 12.0]
STRAT_LO = 8.0


def b64(chw, up=2):
    from PIL import Image
    im = Image.fromarray(chw.permute(1, 2, 0).numpy())
    im = im.resize((im.width * up, im.height * up), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ci = json.loads((LBL / "clip_index.json").read_text(encoding="utf-8"))["clips"]
    leg = collections.defaultdict(list)
    for u, v in ci.items():
        leg[v["episode_id_legacy"]].append(u)
    amb = {k for k, v in leg.items() if len(v) > 1}

    alpa = {r["clip_id"]: r for r in
            map(json.loads, ALPA.read_text(encoding="utf-8").splitlines())}
    rows = {}
    for sp in ("aug120", "w120val"):
        for line in (LBL / f"s2_labels_{sp}.jsonl").read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["clip_id"]] = r

    cache = {}
    by_leg = collections.defaultdict(list)
    for p in sorted(EPC):
        d = torch.load(p, map_location="cpu", weights_only=False)
        cache[p] = d
        by_leg[int(d["episode_id"])].append(p)
    ambc = {k for k, v in by_leg.items() if len(v) > 1}

    out = []
    for p in sorted(EPC):
        d = cache[p]
        e = int(d["episode_id"])
        if e in amb or e in ambc or e not in leg:
            continue
        uid = leg[e][0]
        if uid not in rows:
            continue
        poses = np.asarray(d["poses"], dtype=np.float64)
        T = poses.shape[0]
        if KEY + 20 >= T:
            continue
        r = rows[uid]
        man = EM.analyse(poses, key=KEY, hz=HZ)

        # tactical factored pair
        pl = torch.tensor(poses[KEY:KEY + 1], dtype=torch.float32)
        fp = torch.tensor(poses[KEY + 1:KEY + 1 + TAC.LABEL_HORIZON][None],
                          dtype=torch.float32)
        lat2, lon2 = TAC.window_factored_labels_v2(pl, fp)
        tactical = {"lat": TAC.LAT_CLASSES[int(lat2)],
                    "lon": TAC.LON_CLASSES[int(lon2)],
                    "horizon_s": TAC.LABEL_HORIZON / HZ}

        a = alpa.get(uid)
        sem = S.extract(a["cot"]) if a else None
        toks = [t.as_dict() for t in S.propose_tokens(a["cot"])] if a else []
        resolved, prov = (S.reconcile(man.stop_type, sem) if sem
                          else (man.stop_type, "kinematic-only (no VLM)"))

        g, ab = r["g_str"], r["a_str"]
        rep = LG.check(clip_id=uid, g_str=g["token"],
                       a_str=(None if ab.get("abstain") else ab.get("token")),
                       peak_yaw_deg=man.peak_yaw_deg, v_at_key_ms=man.v_at_key,
                       v_end_ms=man.v_end, v_min_future_ms=man.v_min,
                       tac_lat=tactical["lat"], tac_lon=tactical["lon"],
                       lateral_class=man.lateral_class, stop_type=man.stop_type)

        frames, fmeta = {}, []
        for off in OFFSETS:
            i = int(round(KEY + off * HZ))
            clipped = not (0 <= i <= T - 1)
            i = max(0, min(T - 1, i))
            k = f"t{off:+.0f}"
            frames[k] = b64(d["frames_u8"][i][6:9])
            fmeta.append({"key": k, "offset_s": off, "clipped": clipped,
                          "strategic": off >= STRAT_LO})

        c, s = np.cos(-poses[KEY, 2]), np.sin(-poses[KEY, 2])
        dx, dy = poses[:, 0] - poses[KEY, 0], poses[:, 1] - poses[KEY, 1]
        ex, ey = c * dx - s * dy, s * dx + c * dy

        out.append({
            "clip_id": uid, "split": ci[uid]["label_split"], "t0_s": r["t0_s"],
            "g_str": {"token": g["token"], "args": g.get("args"),
                      "arg_mask": g.get("arg_mask"),
                      "confidence": g.get("confidence"),
                      "provenance": g.get("provenance"),
                      "sources": g.get("sources")},
            "a_str": {"token": (None if ab.get("abstain") else ab.get("token")),
                      "abstain": bool(ab.get("abstain")),
                      "args": ab.get("args"), "arg_mask": ab.get("arg_mask"),
                      "confidence": ab.get("confidence"),
                      "provenance": ab.get("provenance"),
                      "reason": ab.get("reason")},
            "tactical": tactical,
            "manoeuvre": man.as_dict(),
            "alpamayo": ({"lane": a["lane"], "lateral": a["lateral"],
                          "longitudinal": a["longitudinal"], "cot": a["cot"]}
                         if a else None),
            "semantics": (sem.as_dict() if sem else None),
            "tokens": toks,
            "stop_resolved": {"type": resolved, "provenance": prov},
            "guard": rep.as_dict(),
            "horizon": {"future_s": round((T - 1 - KEY) / HZ, 1),
                        "band_observable_s": round(
                            max(0.0, min((T - 1 - KEY) / HZ - STRAT_LO, 22.0)), 1)},
            "traj": {"px": [round(float(q), 2) for q in ex[max(0, KEY - 60):KEY + 1]],
                     "py": [round(float(q), 2) for q in ey[max(0, KEY - 60):KEY + 1]],
                     "fx": [round(float(q), 2) for q in ex[KEY:]],
                     "fy": [round(float(q), 2) for q in ey[KEY:]],
                     "fv": [round(float(q), 2) for q in poses[KEY:, 3]],
                     "band_i": int(STRAT_LO * HZ)},
            "frames": frames, "frame_meta": fmeta,
        })

    (OUT / "sample_v3.json").write_text(json.dumps(out), encoding="utf-8")
    slim = [{k: v for k, v in r.items() if k != "frames"} for r in out]
    (OUT / "sample_v3_slim.json").write_text(json.dumps(slim, indent=1),
                                             encoding="utf-8")
    n = len(out)
    print(f"[v3] {n} clips")
    print(f"[v3] alpamayo {sum(1 for r in out if r['alpamayo'])}/{n}, "
          f"tokens on {sum(1 for r in out if r['tokens'])}/{n}")
    print("[v3] lateral:", dict(collections.Counter(
        r["manoeuvre"]["lateral_class"] for r in out)))
    print("[v3] stop   :", dict(collections.Counter(
        r["manoeuvre"]["stop_type"] for r in out)))
    print("[v3] guard  :", f"{sum(1 for r in out if r['guard']['refused'])} refuse, "
          f"{sum(1 for r in out if r['guard']['findings'] and not r['guard']['refused'])} flag, "
          f"{sum(1 for r in out if not r['guard']['findings'])} clean")


if __name__ == "__main__":
    main()
