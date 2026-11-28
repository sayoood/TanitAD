"""The BLINDED adjudication pack for `H-DAC-DEF-1` §5 — 60 windows, no rule visible.

⛔⛔ **THE MAP IS NOT SHOWN, AND THAT IS THE POINT.** The class panel of the earlier renders
gives the answer away: a brown cell under a corner IS P2's verdict, a yellow or blue one IS
"V0 fires and P1 does not". Worse, judging the instrument from the instrument's own input is
circular — the adjudicator is establishing what "stayed on the drivable surface" MEANS, and the
only evidence that can settle that independently is the camera. So each sheet carries:

  A  the camera at t0 with the recorded footprint path projected into it (one neutral colour)
  B  a fixed ZOOM — the first tick whose footprint is fully inside the frame, never a tick chosen
     by where any rule fires
  C  the camera 2 s later, and  D  4 s later: the surface the car actually drove onto
  E  the path in metric top-down, WITHOUT any map under it

Nothing on a sheet names the window, the stratum, a candidate, or what any rule read. ⛔ sha12 is
omitted too: the landed `dac_windows.jsonl` is keyed by it and carries every candidate's verdict,
so printing it would hand over the key.

    python adjudication_pack.py --raw <raw dir> --out <media dir> --key <path OUTSIDE the repo>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                # noqa: E402
import numpy as np                             # noqa: E402
import torch                                   # noqa: E402

from dac_anatomy import CELL_M, DEFAULTS, THR, Y_HALF_M, _import_harness   # noqa: E402
from dac_project import controls                                           # noqa: E402

SEED = 20260920
V0 = "V0 current rule (any corner, any tick, drivable < 0.50)"
P1 = "P1 road surface (drivable + paint) < 0.50"
P2 = "P2 explicit off-road (edge + hatched + sidewalk) >= 0.50"
STRATA = {"A V0 fires, P1 does not": 20, "B P1 fires, P2 does not": 20,
          "C all three fire": 10, "D none fires": 10}
RUBRIC = {
    "on-surface": "every part of the car stayed on roadway a car may drive on — including paint, "
                  "markings and crosswalks painted ON the roadway",
    "over-boundary": "some part of the car crossed onto a surface that is not roadway — pavement, "
                     "verge, traffic island, or past a raised kerb",
    "cannot-tell": "the images do not show the surface under the relevant part of the car well "
                   "enough to decide",
}


def strata_of(r: dict) -> str | None:
    v0, p1, p2 = r["variants"][V0] == 0.0, r["variants"][P1] == 0.0, r["variants"][P2] == 0.0
    if v0 and not p1:
        return "A V0 fires, P1 does not"
    if p1 and not p2:
        return "B P1 fires, P2 does not"
    if v0 and p1 and p2:
        return "C all three fire"
    if not (v0 or p1 or p2):
        return "D none fires"
    return None


def tick0_diagnosis(vs: list[dict]) -> dict:
    """§6's tick-0 control, with amendment B's escape clause, as counts."""
    t0s = [s for s in vs if s["tick"] == 0]
    w_v0 = {(s["sha12"], s["t0"]) for s in t0s}
    w_p1 = {(s["sha12"], s["t0"]) for s in t0s if s["road_surface"] < THR}
    w_p2 = {(s["sha12"], s["t0"]) for s in t0s if s["explicit_off"] >= THR}
    no_class = {(s["sha12"], s["t0"]) for s in t0s
                if s["road_surface"] < THR and s["explicit_off"] < THR}
    return {"windows_failing_tick0_V0": len(w_v0), "under_P1": len(w_p1), "under_P2": len(w_p2),
            "excluded_by_amendment_B (t0 cell carries NO road class)": len(no_class),
            "excluded_share_of_736": round(len(no_class) / 736, 4),
            "cap": 0.05, "within_cap": len(no_class) / 736 <= 0.05,
            "remainder_after_exclusion": len(w_v0) - len(no_class)}


def sheet(img0, img2, img4, col, row, front, corners, blind_id, out_png,
          panel_a_note=""):
    # ⛔ `panel_a_note` defaults to "" so the LANDED pack reproduces byte-for-byte;
    # the 2026-09-21 sweep passes the depth-test disclosure through it.
    H, W = img0.shape[0], img0.shape[1]
    fig = plt.figure(figsize=(13.5, 11.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.0])
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(img0)

    def overlay(axis):
        for t in range(0, corners.shape[0], 5):
            if front[t].all():
                p = np.concatenate([np.stack([col[t], row[t]], -1), [[col[t, 0], row[t, 0]]]])
                axis.plot(p[:, 0], p[:, 1], lw=1.1, color="#1565c0", alpha=0.95)
        axis.scatter(col[front], row[front], s=5, color="#00bcd4", zorder=3)

    overlay(ax)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()
    ax.set_title("A — camera at t0, with the recorded path of the car's own footprint "
                 "projected in" + panel_a_note, fontsize=10)
    # B — the zoom: the FIRST tick whose footprint is fully inside the frame (never a rule's tick)
    vis = [t for t in range(corners.shape[0])
           if front[t].all() and (col[t] > 0).all() and (col[t] < W).all()
           and (row[t] > 0).all() and (row[t] < H).all()]
    axz = fig.add_subplot(gs[1, 0])
    axz.imshow(img0)
    overlay(axz)
    if vis:
        t = vis[0]
        cx0, cx1 = col[t].min(), col[t].max()
        cy0, cy1 = row[t].min(), row[t].max()
        mx, my = max(60.0, (cx1 - cx0) * 0.8), max(45.0, (cy1 - cy0) * 1.6)
        axz.set_xlim(max(0, cx0 - mx), min(W, cx1 + mx))
        axz.set_ylim(min(H, cy1 + my), max(0, cy0 - my))
    else:
        axz.set_xlim(0, W); axz.set_ylim(H, 0)
    axz.set_axis_off()
    axz.set_title("B — zoom on the nearest fully visible footprint", fontsize=10)
    for k, (im, lab) in enumerate(((img2, "C — the same camera 2 s later"),
                                   (img4, "D — the same camera 4 s later"))):
        a2 = fig.add_subplot(gs[1, 1] if k == 0 else gs[2, 1])
        a2.imshow(im); a2.set_axis_off(); a2.set_title(lab + " (no overlay)", fontsize=10)
    axb = fig.add_subplot(gs[2, 0])
    for t in range(0, corners.shape[0], 5):
        p = corners[t][[0, 1, 2, 3, 0]]
        axb.plot(p[:, 0], p[:, 1], lw=0.9, color="#1565c0", alpha=0.9)
    axb.plot(corners[:, :, 0].mean(1), corners[:, :, 1].mean(1), lw=1.0, color="#00bcd4")
    axb.set_aspect("equal"); axb.grid(True, lw=0.3, alpha=0.5)
    axb.set_xlabel("x forward (m)"); axb.set_ylabel("y lateral (m)")
    axb.set_title("E — the same path, metric top-down. NO MAP IS SHOWN", fontsize=10)
    fig.suptitle(blind_id, fontsize=13)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k, default=v)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", required=True, help="⛔ a path OUTSIDE the repo")
    ap.add_argument("--extrinsics",
                    default="D:/Projects/TanitAD-artifacts/refcv5v2_final/extrinsics141.json")
    a = ap.parse_args()
    raw, out, key_path = Path(a.raw), Path(a.out), Path(a.key)
    if str(key_path.resolve()).startswith(str(Path("D:/Projects/TanitAD").resolve())):
        raise SystemExit(f"⛔ the key would land inside the repo: {key_path}")
    out.mkdir(parents=True, exist_ok=True)
    S1, P, SMG = _import_harness()
    import refc_v3_train as V3
    from tanitad.data.rig_projection import RigCamera
    frame = V3._agent_cam_frames()[(416, 1024)]
    _, table = V3._read_rig_extrinsics(a.extrinsics)

    rows = [json.loads(l) for l in (raw / "dac_windows.jsonl").read_text(encoding="utf-8").splitlines()]
    vs = [json.loads(l) for l in (raw / "dac_violating_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    # the strata are disjoint only if P2 => P1 => V0; assert it rather than assume it
    bad = [r for r in rows if (r["variants"][P2] == 0.0 and r["variants"][P1] == 1.0)
           or (r["variants"][P1] == 0.0 and r["variants"][V0] == 1.0)]
    if bad:
        raise SystemExit(f"⛔ the candidate nesting P2 => P1 => V0 fails on {len(bad)} windows")
    pool: dict = {k: [] for k in STRATA}
    for r in rows:
        s = strata_of(r)
        if s:
            pool[s].append((r["sha12"], r["t0"]))
    rng = random.Random(SEED)
    draw = {}
    for s, n in STRATA.items():
        avail = sorted(pool[s])
        if len(avail) < n:
            raise SystemExit(f"⛔ stratum {s}: {len(avail)} available, {n} required")
        for w in rng.sample(avail, n):
            draw[w] = s
    ids = [f"W{i:02d}" for i in range(1, len(draw) + 1)]
    rng.shuffle(ids)
    blind = dict(zip(sorted(draw), ids))          # sorted -> the shuffle is the only ordering

    by_key = {(r["sha12"], r["t0"]): r for r in rows}
    corp = S1.Corpus(a.config, a.cache, a.labels, a.agents, a.maps, lru=2)
    ctrl, made = None, []
    todo = {}
    for wi in S1.trainer_windows(corp.ds, 1000):
        if corp.eligibility(wi) is not None:
            continue
        e_i, t = corp.ds.index[wi]
        k = (S1.sha12(str(corp.clip_ids[e_i])), int(t + corp.W - 1))
        if k in blind:
            if str(corp.clip_ids[e_i]) not in table:
                raise SystemExit(f"⛔ a drawn window's clip has no extrinsics entry; "
                                 f"its camera cannot be built and the draw would be silently short")
            todo[wi] = (k, e_i)
    if len(todo) != len(blind):
        raise SystemExit(f"⛔ {len(todo)} of {len(blind)} drawn windows were found in the "
                         f"eligible draw — the pack would be short and the strata unbalanced")
    # ⭐ render in EPISODE order: the payload LRU is 2, and a shuffled order would reload an
    # 81 MB clip for nearly every sheet while the S1 pass is reading the same disk.
    for wi in sorted(todo, key=lambda w: todo[w][1]):
        k, e_i = todo[wi]
        cid = str(corp.clip_ids[e_i])
        it = corp.light_item(wi)
        corners = P._ego_boxes(it["human"][None], P.PROXY)[0].numpy()
        cam = RigCamera.from_extrinsics(table[cid], frame)
        if ctrl is None:
            ctrl = controls(cam, frame, corners[0])
            if not ctrl["all_pass"]:
                (raw / "adjudication_controls_MISS.json").write_text(
                    json.dumps(ctrl, indent=1) + "\n", encoding="utf-8", newline="\n")
                raise SystemExit("⛔ a projection control missed; nothing rendered")
        pts = torch.tensor(np.concatenate([corners, np.zeros(corners.shape[:-1] + (1,))], -1),
                           dtype=torch.float64).reshape(-1, 3)
        col, row, _ = cam.project(pts)
        zc = cam.to_cam(pts)[..., 2].numpy()
        sh = corners.shape[:-1]
        col, row, front = col.numpy().reshape(sh), row.numpy().reshape(sh), (zc > 0).reshape(sh)
        fr = corp.eps[e_i].frames
        im = [fr[min(it["t0"] + d, fr.shape[0] - 1)][-3:].permute(1, 2, 0).numpy().astype(np.uint8)
              for d in (0, 20, 40)]
        png = out / f"{blind[k]}.jpg"
        sheet(im[0], im[1], im[2], col, row, front, corners, blind[k], png)
        made.append(blind[k])

    key = {"_what": "⛔ THE KEY to the blinded adjudication pack — do NOT land before the labels",
           "seed": SEED, "strata": STRATA,
           "rows": [{"blind_id": blind[k], "sha12": k[0], "t0": k[1], "stratum": draw[k],
                     "V0_fires": by_key[k]["variants"][V0] == 0.0,
                     "P1_fires": by_key[k]["variants"][P1] == 0.0,
                     "P2_fires": by_key[k]["variants"][P2] == 0.0}
                    for k in sorted(draw)]}
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8", newline="\n")
    key_sha = hashlib.sha256(key_path.read_bytes()).hexdigest()

    pack = {"_what": "blinded adjudication pack for H-DAC-DEF-1 §5",
            "_evidence_class": "MEASURED (ours; CPU, read-only)",
            "seed": SEED, "draw": "random.Random(SEED).sample over each stratum, sorted pool",
            "strata_requested": STRATA,
            "strata_available": {s: len(pool[s]) for s in STRATA},
            "sheets": len(made), "rubric": RUBRIC,
            "map_is_not_shown": "the class panel would BE the verdict (brown = P2 fires, "
                                "yellow/blue = V0 fires and P1 does not) and judging the "
                                "instrument from its own input is circular",
            "projection_controls": ctrl,
            "tick0_control_with_amendment_B": tick0_diagnosis(vs),
            "map_absent_control": {
                "bar": "with no map the term must not score at all (the multiplier defaults to 1)",
                "reads": S1._dac(torch.zeros(1, 2, 4), {"map_drivable": None, "map_seen": None}),
                "pass": S1._dac(torch.zeros(1, 2, 4), {"map_drivable": None, "map_seen": None}) is None},
            "key_sha256": key_sha,
            "key_location": "OUTSIDE the repo, held by the DataFlyWheel until the labels are landed",
            "key_is_reproducible": "re-running this script with the same seed rebuilds it exactly"}
    (raw / "adjudication_pack.json").write_text(json.dumps(pack, indent=1) + "\n",
                                                 encoding="utf-8", newline="\n")
    tmpl = ["# blind_id,label,note   (label: on-surface | over-boundary | cannot-tell)"]
    tmpl += [f"{i},," for i in sorted(made)]
    (out.parent / "LABELS_TEMPLATE.csv").write_text("\n".join(tmpl) + "\n",
                                                    encoding="utf-8", newline="\n")
    print(json.dumps({k: pack[k] for k in ("strata_available", "sheets", "key_sha256",
                                           "tick0_control_with_amendment_B",
                                           "map_absent_control")}, indent=1))
    print("projection controls all pass:", ctrl["all_pass"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
