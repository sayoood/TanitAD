"""PREREG_V8F_LATSEM2B label construction for the DECIDING set (WMACT TRAIN windows). Labels only: nothing here reads a latent statistic. Tier T0.

TRAIN: frame bank = LATSEM2's rule on theory/wmact/latents_train.pt; targets = LATSEM2's seven (+2 beside), same fields, units and NaN rules
(IMPL_CHOICES_LATSEM2.md section 3), read from the TRAIN counterparts of LATSEM2's sources:
  lead_present / lead_gap / lead_closing : D.lead_label_arrays on v9labels/inputs/lead_train.npz (the trainer's --lead-train), joined with D.build_label_join (the trainer's own call)
  lon7                                   : the v9 release row `lon_cls` of v9labels/v9_labels_train.npz (the trainer's --v9-train), same join
  agents30 / nearest / agents30_view     : the B1 TRAIN agent join (2-D) tanitad-data/joins/b1train3d/b1train_agents.jsonl.xz (cx, cy, occ are carried unchanged by the 3-D eval join)
  turn / ego speed                       : native_tracks.pt (columns 2 and 3 at raw row 2g)
Run from theory/latsem2b with WMACT_CODE_ROOT set.  `python labels_latsem2b.py` writes labels_latsem2b_train.npz and labels_latsem2b_train.report.json."""
import hashlib, json, lzma, os, sys
sys.dont_write_bytecode = True
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("../wmact"))
import wmact_lib as W  # noqa: E402,F401  (sets WMACT_CODE_ROOT onto sys.path)
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.data.v9_labels import load_v9_release  # noqa: E402

ART = "E:/Projects/TanitAD-artifacts"
TS = "../wmact/train_stream"
LATENTS = "../wmact/latents_train.pt"
V9 = f"{ART}/v9labels/v9_labels_train.npz"
V9_MD5 = "f63ece410b725febb8a5242cf2b01d3c"
LEAD = f"{ART}/v9labels/inputs/lead_train.npz"
AGENTS = "C:/Users/Admin/tanitad-data/joins/b1train3d/b1train_agents.jsonl.xz"
TRACKS = f"{ART}/sfresh_inputs/inputs/native_tracks.pt"
F = 16
DT_STEP, R_AGENT_M, TURN_ROWS, TURN_RAD, CLOSE_STEPS, CLOSE_CAP = 0.2, 30.0, 40, 0.3, 2, 20.0
TARGETS = ("lead_present", "lead_gap", "lead_closing", "agents30", "nearest", "lon7", "turn")
BESIDE = ("lead_closing_uncapped", "agents30_view")
KIND = {"lead_present": "bin", "lead_gap": "cont", "lead_closing": "cont", "agents30": "cont", "nearest": "cont", "lon7": "cls", "turn": "bin",
        "lead_closing_uncapped": "cont", "agents30_view": "cont"}
LON_NAMES = ("HOLD", "CREEP", "STOP", "FOLLOW", "DECELERATE", "ACCELERATE", "KEEP")
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def build_bank_train(latents=LATENTS):
    """LATSEM2's frame bank rule (labels_latsem2.build_bank) on the TRAIN windows. LATSEM2's 'episode has a frame file' filter is not applied: this read uses no camera frame (all 141 clips kept)."""
    L = torch.load(latents, weights_only=False)
    eid, tw, ok, A_raw = np.array(L["eid"]), np.asarray(L["t"]), L["ok"].numpy(), L["A_raw"]
    wkeep = np.arange(len(eid))
    G = tw[:, None] - 15 + np.arange(F)[None]
    key2b, occ_w, occ_f, occ_b = {}, [], [], []
    for i in wkeep:
        for f in range(F):
            if ok[i, f] and G[i, f] >= 0:
                k = (eid[i], int(G[i, f]))
                b = key2b.setdefault(k, len(key2b))
                occ_w.append(i); occ_f.append(f); occ_b.append(b)
    occ_w, occ_f, occ_b = map(np.asarray, (occ_w, occ_f, occ_b))
    first = np.full(len(key2b), -1)
    for j in range(len(occ_b) - 1, -1, -1):
        first[occ_b[j]] = j
    bw, bf = occ_w[first], occ_f[first]
    beid = np.array([k[0] for k in key2b]); bg = np.array([k[1] for k in key2b])
    return dict(L=L, eid=eid, tw=tw, ok=ok, A_raw=A_raw, wkeep=wkeep, G=G, key2b=key2b, occ_w=occ_w, occ_f=occ_f, occ_b=occ_b, bw=bw, bf=bf, beid=beid, bg=bg)


def load_agent_lines(path, clips):
    """{(clip_id, raw frame): (distance[m], occ)} for the lines of the given clips (centre distance in the ego frame of that frame)."""
    want, out = set(clips), {}
    for ln in lzma.open(path, "rt", encoding="utf-8"):
        if ln[12:48] not in want:
            continue
        r = json.loads(ln)
        a = r["agents"]
        d = np.hypot([x["cx"] for x in a], [x["cy"] for x in a]) if a else np.zeros(0)
        o = np.array([x["occ"] for x in a], np.int64) if a else np.zeros(0, np.int64)
        out[(r["clip_id"], int(r["frame"]))] = (d, o)
    return out


def build_join_train(clips, n_frames):
    """The trainer's own call (v8f_mini_train.py:268-272): load_v9_release(md5-checked) -> D.build_label_join(man, rel, lead_npz) on the 141 clips."""
    rel = load_v9_release(V9, expect_md5=V9_MD5)
    man = [(c, int(n_frames[c])) for c in clips]
    J = D.build_label_join(man, rel, lead_npz=LEAD, speed=None)
    return J, rel


def build_labels_train(beid, bg, J, AG, TR):
    ci = {str(c): i for i, c in enumerate(J["clip"])}
    lead = D.lead_label_arrays(J, np.load(LEAD, allow_pickle=True))
    nb = len(beid)
    Y = {q: np.full(nb, np.nan) for q in TARGETS}
    Y["agents30_view"] = np.full(nb, np.nan)
    Y["lead_closing_uncapped"] = np.full(nb, np.nan)
    speed = np.full(nb, np.nan)          # ego speed, native_tracks column 3 at raw row 2g (the field beside_gap_speed.py of LATSEM2 used)
    audit = dict(n_bank=nb, g_beyond_clip=0, no_join_clip=0, lead_has=0, lead_near=0, lead_valid_vs_near=[0, 0, 0, 0], agent_line_missing=0)
    for b, (e, g) in enumerate(zip(beid, bg)):
        g = int(g); c = ci.get(str(e))
        if c is None:
            audit["no_join_clip"] += 1; continue
        if g >= int(J["n"][c]):
            audit["g_beyond_clip"] += 1; continue
        row = int(J["off"][c]) + g
        tr = TR[str(e)]; r = 2 * g
        if r < len(tr):
            speed[b] = tr[r, 3]
        if lead["has"][row]:
            audit["lead_has"] += 1
            Y["lead_present"][b] = float(lead["near"][row])
            if lead["near"][row]:
                audit["lead_near"] += 1
                Y["lead_gap"][b] = float(D.LEAD_RANGE_M * lead["y"][row])
                if g >= CLOSE_STEPS and lead["has"][row - CLOSE_STEPS] and lead["near"][row - CLOSE_STEPS]:
                    cl = (D.LEAD_RANGE_M * float(lead["y"][row - CLOSE_STEPS]) - Y["lead_gap"][b]) / (CLOSE_STEPS * DT_STEP)
                    Y["lead_closing_uncapped"][b] = cl
                    Y["lead_closing"][b] = cl if abs(cl) <= CLOSE_CAP else np.nan
            vv = int(J["lead_valid"][row]); nn = int(lead["near"][row])
            audit["lead_valid_vs_near"][2 * vv + nn] += 1
        ln = AG.get((str(e), 2 * g))
        if ln is None:
            audit["agent_line_missing"] += 1
        else:
            d, o = ln
            Y["agents30"][b] = float((d <= R_AGENT_M).sum())
            Y["agents30_view"][b] = float(((d <= R_AGENT_M) & (o == 0)).sum())
            Y["nearest"][b] = float(d.min()) if len(d) else np.nan
        if J["has"][row] and int(J["lon_cls"][row]) >= 0:
            Y["lon7"][b] = float(J["lon_cls"][row])
        if r + TURN_ROWS < len(tr) and np.isfinite(tr[r, 2]) and np.isfinite(tr[r + TURN_ROWS, 2]):
            Y["turn"][b] = float(abs(wrap(tr[r + TURN_ROWS, 2] - tr[r, 2])) > TURN_RAD)
    return Y, speed, audit


if __name__ == "__main__":
    meta = json.load(open(f"{TS}/clips141.json"))
    clips, nfr = meta["clips"], meta["n_frames"]
    B = build_bank_train()
    assert sorted(set(B["beid"])) == sorted(clips)
    J, rel = build_join_train(clips, nfr)
    AG = load_agent_lines(AGENTS, clips)
    cs = set(clips)
    TR = {k: np.asarray(v, np.float64) for k, v in torch.load(TRACKS, weights_only=False)["tracks"].items() if k in cs}
    Y, speed, audit = build_labels_train(B["beid"], B["bg"], J, AG, TR)
    audit["clips_without_agent_line"] = sorted(cs - {k[0] for k in AG})
    beid = B["beid"]
    rep = {"n_bank_frames": int(len(beid)), "n_episodes": int(len(set(beid))), "audit": audit, "n": {}, "summary": {}, "n_clips_missing_v9_rows": int(J["n_clips_missing_labels"])}
    for q, y in Y.items():
        f = np.isfinite(y); rep["n"][q] = int(f.sum())
        if q == "lon7":
            rep["summary"][q] = {LON_NAMES[int(c)]: int((y[f] == c).sum()) for c in range(7)}
        elif KIND.get(q) == "bin":
            rep["summary"][q] = {"positive": int((y[f] == 1).sum()), "negative": int((y[f] == 0).sum())}
        else:
            rep["summary"][q] = {k: float(v) for k, v in zip(("min", "p5", "median", "p95", "max", "mean", "sd"), (y[f].min(), *np.percentile(y[f], [5, 50, 95]), y[f].max(), y[f].mean(), y[f].std()))}
    fold = np.array([int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % 2 for e in beid])
    rep["n_by_fold"] = {q: [int((np.isfinite(y) & (fold == k)).sum()) for k in (0, 1)] for q, y in Y.items()}
    rep["n_episodes_by_target"] = {q: int(len(set(beid[np.isfinite(y)]))) for q, y in Y.items()}
    rep["n_episodes_by_target_by_fold"] = {q: [int(len(set(beid[np.isfinite(y) & (fold == k)]))) for k in (0, 1)] for q, y in Y.items()}
    rep["n_episodes_by_fold"] = [int(len(set(beid[fold == k]))) for k in (0, 1)]
    sel = B["wkeep"][B["ok"][B["wkeep"]].all(1)]
    rep["n_windows_all_frames_ok"] = int(len(sel))
    key2b, G, eid = B["key2b"], B["G"], B["eid"]
    lp = np.array([Y["lead_present"][key2b[(eid[i], int(G[i, 0]))]] for i in sel])
    rep["n_windows_lead_near_at_frame0"] = int((lp == 1).sum())
    rep["n_episodes_windows_lead_near_at_frame0"] = int(len(set(eid[sel][lp == 1])))
    f = np.isfinite(Y["lead_gap"]) & np.isfinite(speed)
    rep["gap_speed_pearson"] = float(np.corrcoef(Y["lead_gap"][f], speed[f])[0, 1]); rep["n_gap_speed"] = int(f.sum())
    # alignment check, labels only (IMPL_CHOICES_LATSEM2 section 3): in-corridor agent distance (|cy| < 1.5 m, ahead) against the lead gap, at raw-frame offsets -1, 0, +1
    import lzma as _l
    corr = {}
    ahead = {}
    for ln in _l.open(AGENTS, "rt", encoding="utf-8"):
        if ln[12:48] not in cs:
            continue
        r_ = json.loads(ln)
        best = np.inf
        for a in r_["agents"]:
            if a["cx"] > 0 and abs(a["cy"]) < 1.5:
                best = min(best, a["cx"] - a["l"] / 2)
        ahead[(r_["clip_id"], int(r_["frame"]))] = best
    for off in (-1, 0, 1):
        xs, ys = [], []
        for b, (e, g) in enumerate(zip(beid, B["bg"])):
            if np.isfinite(Y["lead_gap"][b]):
                v = ahead.get((str(e), 2 * int(g) + off))
                if v is not None and np.isfinite(v):
                    xs.append(v); ys.append(Y["lead_gap"][b])
        corr[str(off)] = {"n": len(ys), "pearson": float(np.corrcoef(xs, ys)[0, 1]) if len(ys) > 3 else None}
    rep["alignment_check_corridor_distance_vs_lead_gap"] = corr
    rep["sources_sha256"] = {"v9_labels_train.npz": sha256(V9), "lead_train.npz": sha256(LEAD), "b1train_agents.jsonl.xz": sha256(AGENTS), "native_tracks.pt": sha256(TRACKS),
                             "latents_train.pt": sha256(LATENTS), "clips141.txt": sha256(f"{TS}/clips141.txt"), "clips141.json": sha256(f"{TS}/clips141.json"),
                             "box2_meta/manifest.jsonl": sha256(f"{TS}/box2_meta/manifest.jsonl")}
    rep["v9_md5"] = rel.md5
    np.savez_compressed("labels_latsem2b_train.npz", beid=beid.astype("U36"), bg=B["bg"], speed=speed, **{q: y for q, y in Y.items()})
    json.dump(rep, open("labels_latsem2b_train.report.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps(rep, indent=1))
