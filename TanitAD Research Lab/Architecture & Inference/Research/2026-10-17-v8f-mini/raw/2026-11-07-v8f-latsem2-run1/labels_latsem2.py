"""PREREG_V8F_LATSEM2 label construction (labels only: nothing here reads a latent statistic). Tier T0.

Builds LATSEM's frame bank (same rule as theory/latsem/run_latsem.py) and, per bank frame (episode e, cache step g), the targets of the card:
  lead_present, lead_gap, lead_closing, agents30, nearest, lon7, turn  (+ agents30_view, beside, deciding nothing).
Every field, unit and NaN rule is fixed in IMPL_CHOICES_LATSEM2.md. Run from theory/latsem2 with WMACT_CODE_ROOT set.
`python labels_latsem2.py` prints the n per target and writes labels_latsem2.npz."""
import hashlib, json, lzma, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("../wmact"))
import wmact_lib as W  # noqa: E402,F401  (sets WMACT_CODE_ROOT onto sys.path)
from tanitad.data import v8f_mini_data as D  # noqa: E402

ART = "E:/Projects/TanitAD-artifacts"
FR = f"{ART}/v2ep-eval139-416x1024cyl"
JOIN = f"{W.RAW}/join_eval141.npz"
LEAD = f"{ART}/v9labels/inputs/lead_eval139.npz"
AGENTS = f"{ART}/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
TRACKS = f"{ART}/sfresh_inputs/inputs/native_tracks.pt"
F = 16
DT_STEP = 0.2                      # one cache step = 0.2 s = 2 raw 10 Hz rows
R_AGENT_M = 30.0                   # "agents within 30 m": centre distance in the ego frame, <= 30
TURN_ROWS, TURN_RAD = 40, 0.3      # 4 s of raw rows; absolute heading change above 0.3 rad
CLOSE_STEPS = 2                    # 0.4 s
CLOSE_CAP = 20.0                   # m/s: |closing| above this is read as a lead-track switch (the lead file carries no track id) and set to NaN
LON_NAMES = ("HOLD", "CREEP", "STOP", "FOLLOW", "DECELERATE", "ACCELERATE", "KEEP")   # LON9 index order, v8f_mini_data.LON9
TARGETS = ("lead_present", "lead_gap", "lead_closing", "agents30", "nearest", "lon7", "turn")
BESIDE = ("lead_closing_uncapped", "agents30_view")        # computed and reported, decide nothing
KIND = {"lead_present": "bin", "lead_gap": "cont", "lead_closing": "cont", "agents30": "cont", "nearest": "cont", "lon7": "cls", "turn": "bin",
        "lead_closing_uncapped": "cont", "agents30_view": "cont"}
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi


def build_bank(latents="../wmact/latents.pt"):
    """LATSEM's frame bank, copied rule for rule: every distinct (episode, g) with frame_ok, restricted to the episodes with a frame file."""
    L = torch.load(latents, weights_only=False)
    eid, tw, ok, A_raw = np.array(L["eid"]), np.asarray(L["t"]), L["ok"].numpy(), L["A_raw"]
    have = {f.split(".")[0] for f in os.listdir(FR) if f.endswith(".v2ep.pt")}
    wkeep = np.flatnonzero(np.isin(eid, sorted(have)))
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
    return dict(L=L, eid=eid, tw=tw, ok=ok, A_raw=A_raw, wkeep=wkeep, G=G, key2b=key2b, occ_w=occ_w, occ_f=occ_f, occ_b=occ_b,
                bw=bw, bf=bf, beid=beid, bg=bg)


def load_agent_lines():
    """{(clip_id, raw frame): (distance[m], occ)} for every line of the B1-eval agent join (cuboid centre in the ego frame at that frame)."""
    out = {}
    for ln in lzma.open(AGENTS, "rt", encoding="utf-8"):
        r = json.loads(ln)
        a = r["agents"]
        d = np.hypot([x["cx"] for x in a], [x["cy"] for x in a]) if a else np.zeros(0)
        o = np.array([x["occ"] for x in a], np.int64) if a else np.zeros(0, np.int64)
        out[(r["clip_id"], int(r["frame"]))] = (d, o)
    return out


def build_labels(beid, bg):
    J = D.load_join(JOIN)
    ci = {str(c): i for i, c in enumerate(J["clip"])}
    lead = D.lead_label_arrays(J, np.load(LEAD, allow_pickle=True))
    AG = load_agent_lines()
    TR = {k: np.asarray(v, np.float64) for k, v in torch.load(TRACKS, weights_only=False)["tracks"].items()}
    nb = len(beid)
    Y = {q: np.full(nb, np.nan) for q in TARGETS}
    Y["agents30_view"] = np.full(nb, np.nan)
    Y["lead_closing_uncapped"] = np.full(nb, np.nan)
    audit = dict(n_bank=nb, g_beyond_clip=0, no_join_clip=0, lead_has=0, lead_near=0, lead_valid_vs_near=[0, 0, 0, 0], agent_line_missing=0)
    for b, (e, g) in enumerate(zip(beid, bg)):
        g = int(g)
        c = ci.get(str(e))
        if c is None:
            audit["no_join_clip"] += 1; continue
        if g >= int(J["n"][c]):
            audit["g_beyond_clip"] += 1; continue
        row = int(J["off"][c]) + g
        # --- lead (the trainer's lead_has / lead_near / lead_y, v8f_mini_data.lead_label_arrays on lead_eval139.npz)
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
            audit["lead_valid_vs_near"][2 * vv + nn] += 1        # [valid0&near0, valid0&near1, valid1&near0, valid1&near1]
        # --- agents (B1-eval agent join, raw frame 2g)
        ln = AG.get((str(e), 2 * g))
        if ln is None:
            audit["agent_line_missing"] += 1
        else:
            d, o = ln
            Y["agents30"][b] = float((d <= R_AGENT_M).sum())
            Y["agents30_view"][b] = float(((d <= R_AGENT_M) & (o == 0)).sum())
            Y["nearest"][b] = float(d.min()) if len(d) else np.nan
        # --- 7-class longitudinal label (v9 release row of the join: lon_cls, LON9 index; NaN where no row or ignore)
        if J["has"][row] and int(J["lon_cls"][row]) >= 0:
            Y["lon7"][b] = float(J["lon_cls"][row])
        # --- turn ahead (native_tracks heading, raw row 2g, 4 s = 40 rows)
        tr = TR[str(e)]; r = 2 * g
        if r + TURN_ROWS < len(tr) and np.isfinite(tr[r, 2]) and np.isfinite(tr[r + TURN_ROWS, 2]):
            Y["turn"][b] = float(abs(wrap(tr[r + TURN_ROWS, 2] - tr[r, 2])) > TURN_RAD)
    return Y, audit


if __name__ == "__main__":
    B = build_bank()
    Y, audit = build_labels(B["beid"], B["bg"])
    rep = {"n_bank_frames": int(len(B["beid"])), "n_episodes": int(len(set(B["beid"]))), "audit": audit, "n": {}, "summary": {}}
    for q, y in Y.items():
        f = np.isfinite(y)
        rep["n"][q] = int(f.sum())
        if q in ("lon7",):
            rep["summary"][q] = {LON_NAMES[int(c)]: int((y[f] == c).sum()) for c in range(7)}
        elif KIND.get(q) == "bin":
            rep["summary"][q] = {"positive": int((y[f] == 1).sum()), "negative": int((y[f] == 0).sum())}
        else:
            rep["summary"][q] = {"min": float(y[f].min()), "p5": float(np.percentile(y[f], 5)), "median": float(np.median(y[f])),
                                 "p95": float(np.percentile(y[f], 95)), "max": float(y[f].max()), "mean": float(y[f].mean()), "sd": float(y[f].std())}
    fold = np.array([int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % 2 for e in B["beid"]])
    rep["n_by_fold"] = {q: [int((np.isfinite(y) & (fold == k)).sum()) for k in (0, 1)] for q, y in Y.items()}
    rep["n_episodes_by_target"] = {q: int(len(set(B["beid"][np.isfinite(y)]))) for q, y in Y.items()}
    sel = B["wkeep"][B["ok"][B["wkeep"]].all(1)]
    rep["n_windows_all_frames_ok"] = int(len(sel))
    key2b, G, eid = B["key2b"], B["G"], B["eid"]
    lp = np.array([Y["lead_present"][key2b[(eid[i], int(G[i, 0]))]] for i in sel])
    rep["n_windows_lead_near_at_frame0"] = int((lp == 1).sum())
    rep["n_episodes_windows_lead_near_at_frame0"] = int(len(set(eid[sel][lp == 1])))
    print(json.dumps(rep, indent=1))
    np.savez_compressed("labels_latsem2.npz", beid=B["beid"].astype("U36"), bg=B["bg"], **{q: y for q, y in Y.items()})
    json.dump(rep, open("labels_latsem2.report.json", "w", encoding="utf-8", newline="\n"), indent=1)
