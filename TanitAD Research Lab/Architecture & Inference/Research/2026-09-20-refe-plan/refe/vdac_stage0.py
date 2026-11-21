#!/usr/bin/env python3
"""VDAC A5 Stage 0 (PREREG_VDAC_P6_A5 DRAFT, section 1): the in-sample read of HA-s0's head + the S1-F inputs.  A MEASURING tool, not a training run: needs no launch-gate token and is forbidden in a launch argv.
CPU only.  Tier T0-like (head perception on frames; never driving performance).

WHAT THIS TOOL IS AND IS NOT.  It is the DATA-INDEPENDENT frame draw, the read (the frozen S1a reader's OWN metric functions, sha asserted), the controls and the mechanical row.  It does NOT run the model: the head
forwards (the 2,048 in-sample frames, the 256 C0-ID frames, the seed-0 initial head on the extra-log frames) are produced by a SEPARATE forward step (READINESS.md item S0: a `--stage0-*` mode of the trainer, which
owns `fwd()`), which writes the frozen reader's npz schema (`log, token, step, logit, sdf_m`; logit / sdf_m [N,40,32] float32; step all 0).  This tool refuses to run on anything that is not exactly that.

Subcommands (every one has --dry-run: validate every input and print the plan, write nothing, no GPU):
  make-list  the in-sample frame list: per city (LV / BOS / PIT / SG) the 512 frames with the lowest sha256("<log>|<token>") among LABELLED bank rows at step 0, rank 0, whose log is in neither H48 nor the SFT
             held-out list.  Written WITH ITS sha256 before anything else exists (no forward is possible in this subcommand).  Also the 256 C0-ID frames (the lowest hashes among the banked H48 head rows).
  read       forwards in -> in-sample IoU (city-reweighted, [0,40) m, `logit > 0`, the SAME formula as S1-P1), controls C0-ID / C0-FLIP / C0-FLOOR, and the mechanical row S0-FIT (point >= 0.85) / S0-WEAK (< 0.85) /
             S0-INVALID (a control failed).  Writes in_sample_iou.json, the input of s1_read_v2.py's `in_sample_iou_file`.
  s1f        assembles the S1-F read (A5 draft 1): checks the rebuilt seed-0 initial head reproduces the pod's `head_pred_step0.npz` (sign agreement >= 99.9 %), writes the v2 manifest and runs the frozen reader v2 with the
             Stage 0 value supplied; compares its S1-FIT / S1-WEAK with this tool's row.  NOT BUILT: the 1,705-frame-subset branch (A5 1: reproduction fails) -> the tool stops and says so.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
import time

import numpy as np

TOOL = "vdac_stage0/1"
PER_CITY = 512
CITIES = ("LV", "BOS", "PIT", "SG")
N_REPRO = 256
FIT_BAR = 0.85                 # A5 draft 1 (= the card's S1-FIT in-sample bar, frozen reader FIT_IN_SAMPLE)
FLIP_MARGIN = 0.10             # C0-FLIP: the y-flipped in-sample IoU reads at least this far below the in-sample IoU
ID_SIGN_AGREE, ID_IOU_TOL = 0.999, 0.002
INIT_SIGN_AGREE = 0.999
MUTATIONS = ("include_h48", "flip_threshold", "no_reweight")      # deliberate regressions the tests switch on; a real run never does


# ======================================================================== the frozen readers, located and sha-asserted
def _find(env, rels):
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [os.environ.get(env)] + [os.path.join(here, r) for r in rels]:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise SystemExit(f"REFUSED: cannot find {rels[0]} (set {env}): no fallback")


def load_readers():
    """-> (v2 module, frozen v1 module).  v2 itself asserts the v1 sha at import; this tool asserts it again from the loaded file."""
    p2 = _find("S1_READ_V2", ["s1_read_v2.py", os.path.join("..", "eval", "s1_read_v2.py")])
    d2 = os.path.dirname(p2)
    if d2 not in sys.path:
        sys.path.insert(0, d2)
    spec = importlib.util.spec_from_file_location("s1_read_v2", p2)
    m = importlib.util.module_from_spec(spec)
    sys.modules["s1_read_v2"] = m
    spec.loader.exec_module(m)
    v1 = m._V1
    raw = hashlib.sha256(open(m.V1_PATH, "rb").read()).hexdigest()
    lf = hashlib.sha256(open(m.V1_PATH, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
    if m.V1_SHA256 not in (raw, lf):
        raise SystemExit("REFUSED: the frozen reader's sha256 does not match")
    return m, v1


V2, R = load_readers()


def sha256_file(p):
    return R.sha256_file(p)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ======================================================================== small readers
def read_lines(path):
    return open(path, "r", encoding="utf-8").read().split()


def require_file(p, what):
    if not p or not os.path.isfile(p):
        raise SystemExit(f"REFUSED: {what}: {p!r} is not a file (no fallback)")
    return p


def shard_header(path):
    """the shard header (log, entries) without the blobs: `aux_label_store.read_shard(with_blobs=False)`'s format."""
    with open(path, "rb") as f:
        if f.read(8) != b"AUXSDF1\n":
            raise SystemExit(f"REFUSED: {path}: bad shard magic")
        (hl,) = struct.unpack("<I", f.read(4))
        return json.loads(f.read(hl))


class Store:
    """the label store: manifest (pinned prefix), per-shard sha256 verified, city per log, labelled keys."""

    def __init__(self, root, sha_prefix=R.STORE_SHA_PREFIX):
        self.root = root
        mp = require_file(os.path.join(root, "manifest.json"), "label store manifest")
        self.man = json.load(open(mp))
        sha = str(self.man.get("store_sha256", ""))
        if not sha.startswith(sha_prefix):
            raise SystemExit(f"REFUSED: label store sha256 {sha[:16]} does not start with {sha_prefix}")
        again = hashlib.sha256("\n".join(f"{k}:{v['sha256']}" for k, v in sorted(self.man["files"].items())).encode()).hexdigest()
        if again != sha:
            raise SystemExit("REFUSED: label store manifest does not hash its own files map")
        self.sha = sha
        self.files = self.man["files"]
        self.city = {}
        self.keys = set()
        for rel, ent in sorted(self.files.items()):
            p = os.path.join(root, rel)
            if R.sha256_file(p) != ent["sha256"]:
                raise SystemExit(f"REFUSED: shard {rel} does not hash to the store manifest")
            head = shard_header(p)
            log = head["log"]
            if ent["map"] not in R.MAP_CITY:
                raise SystemExit(f"REFUSED: shard {rel} map {ent['map']!r} is not one of the four cities")
            self.city[log] = R.MAP_CITY[ent["map"]]
            self.keys |= {(log, e["t"]) for e in head["entries"]}

    def labels(self, keys):
        """[(log, token)] -> int8 [N,160,128]; a key without a label is a REFUSAL here (no silent valid=False)."""
        out = np.zeros((len(keys), R.NXF, R.NYF), np.int8)
        cache = {}
        for i, (lg, tk) in enumerate(keys):
            if lg not in cache:
                cache[lg] = R.read_shard(os.path.join(self.root, f"shards/{lg}.aux1"))[1]
            if tk not in cache[lg]:
                raise SystemExit(f"REFUSED: no label for ({lg}, {tk}) in the store")
            out[i] = cache[lg][tk]
        return out


def read_bank_keys(path):
    """bank rows -> list of (log, token, step, rank).  .npz (log, token, step, rank) or a TSV with that header."""
    require_file(path, "bank keys")
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        miss = [k for k in ("log", "token", "step", "rank") if k not in z.files]
        if miss:
            raise SystemExit(f"REFUSED: bank keys npz lacks {miss}")
        return list(zip([str(x) for x in z["log"]], [str(x) for x in z["token"]], [int(x) for x in z["step"]], [int(x) for x in z["rank"]]))
    rows = [ln.rstrip("\n").split("\t") for ln in open(path, encoding="utf-8") if ln.strip()]
    if not rows or rows[0] != ["log", "token", "step", "rank"]:
        raise SystemExit("REFUSED: bank keys TSV must start with the header log<TAB>token<TAB>step<TAB>rank")
    return [(r[0], r[1], int(r[2]), int(r[3])) for r in rows[1:]]


def key_hash(log, token):
    return hashlib.sha256(f"{log}|{token}".encode("utf-8")).hexdigest()


# ======================================================================== make-list: the data-independent draw
def draw_list(bank_rows, store, excluded_logs, per_city=PER_CITY, mutate=None):
    """-> rows [(log, token, city, hash)] ordered LV, BOS, PIT, SG then hash ascending.  Eligible = a labelled key at step 0 / rank 0 whose log is not excluded.  Pure function of its arguments
    (no model output).  mutate='include_h48' is the regression arm (drops the exclusion; the verifier must refuse its list)."""
    excl = set() if mutate == "include_h48" else set(excluded_logs)
    seen = {}
    for lg, tk, st, rk in bank_rows:
        if st != 0 or rk != 0 or (lg, tk) not in store.keys or lg in excl:
            continue
        if (lg, tk) in seen:
            raise SystemExit(f"REFUSED: ({lg}, {tk}) appears twice at step 0 rank 0 in the bank keys")
        seen[(lg, tk)] = key_hash(lg, tk)
    out = []
    for c in CITIES:
        pool = sorted(((h, lg, tk) for (lg, tk), h in seen.items() if store.city[lg] == c))
        if len(pool) < per_city:
            raise SystemExit(f"REFUSED: city {c}: {len(pool)} eligible frames < {per_city}")
        out += [(lg, tk, c, h) for h, lg, tk in pool[:per_city]]
    return out


def list_text(rows):
    return "log\ttoken\tcity\tsha256key\n" + "".join(f"{lg}\t{tk}\t{c}\t{h}\n" for lg, tk, c, h in rows)


def read_list(path):
    L = open(path, encoding="utf-8").read().split("\n")
    if not L or L[0] != "log\ttoken\tcity\tsha256key":
        raise SystemExit("REFUSED: frame list header")
    return [tuple(x.split("\t")) for x in L[1:] if x]


def verify_list(rows, store, excluded_logs, per_city=PER_CITY):
    """the list is what the rule says it is about: per-city quota, hashes right, no excluded log, every frame labelled, city = the store's city, no duplicate.  Raises SystemExit."""
    keys = [(r[0], r[1]) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("REFUSED: duplicate frame in the list")
    for c in CITIES:
        if sum(1 for r in rows if r[2] == c) != per_city:
            raise SystemExit(f"REFUSED: city {c} has {sum(1 for r in rows if r[2] == c)} frames != {per_city}")
    if len(rows) != per_city * len(CITIES):
        raise SystemExit("REFUSED: a frame of an unknown city is in the list")
    bad_ex = sorted({r[0] for r in rows} & set(excluded_logs))
    if bad_ex:
        raise SystemExit(f"REFUSED: the list contains {len(bad_ex)} log(s) of H48 / the SFT held-out set, e.g. {bad_ex[0]}")
    for lg, tk, c, h in rows:
        if h != key_hash(lg, tk):
            raise SystemExit(f"REFUSED: stored hash of ({lg}, {tk}) is not sha256 of '<log>|<token>'")
        if (lg, tk) not in store.keys:
            raise SystemExit(f"REFUSED: ({lg}, {tk}) has no label in the store")
        if store.city[lg] != c:
            raise SystemExit(f"REFUSED: city of {lg} is {store.city[lg]} in the store, {c} in the list")


def excluded_logs_from(h48_path, sft_path):
    with open(require_file(h48_path, "H48 list"), "rb") as f:
        h48b = f.read()
    if hashlib.sha256(h48b).hexdigest() != R.H48_SHA:
        raise SystemExit("REFUSED: H48 list sha256 is not the frozen 2ae13935...")
    h48 = h48b.decode().split()
    sft = read_lines(require_file(sft_path, "SFT held-out list"))
    return sorted(set(h48) | set(sft)), h48, sft


def cmd_make_list(a):
    excl, h48, sft = excluded_logs_from(a.h48_logs, a.sft_heldout)
    store = Store(a.store, a.store_sha256_prefix)
    bank = read_bank_keys(a.bank_keys)
    rows = draw_list(bank, store, excl, PER_CITY if not a.per_city_for_tests else a.per_city_for_tests)
    verify_list(rows, store, excl, PER_CITY if not a.per_city_for_tests else a.per_city_for_tests)
    txt = list_text(rows)
    sha = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    repro = None
    if a.banked_pred:
        bz = np.load(require_file(a.banked_pred, "banked head predictions"), allow_pickle=False)
        sft_set, n_rep = set(sft), (a.repro_n_for_tests or N_REPRO)
        bk = sorted({(str(l), str(t)) for l, t in zip(bz["log"], bz["token"]) if str(l) in sft_set}, key=lambda k: (key_hash(*k), k))
        if len(bk) < n_rep:
            raise SystemExit(f"REFUSED: the banked file holds {len(bk)} rows of the SFT-24 logs < {n_rep}")
        repro = "log\ttoken\tsha256key\n" + "".join(f"{l}\t{t}\t{key_hash(l, t)}\n" for l, t in bk[:n_rep])
    plan = {"frames": len(rows), "per_city": {c: sum(1 for r in rows if r[2] == c) for c in CITIES}, "frame_list_sha256": sha, "distinct_logs": len({r[0] for r in rows}),
            "repro_frames": None if repro is None else n_rep, "excluded_logs": len(excl), "bank_rows": len(bank), "store_sha256": store.sha}
    print("PLAN", json.dumps(plan))
    if a.dry_run:
        print("DRY-RUN: inputs valid, nothing written")
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    p_list = os.path.join(a.out_dir, "stage0_frames.tsv")
    open(p_list, "w", encoding="utf-8", newline="\n").write(txt)
    meta = {"tool": TOOL, "utc": utc(), "rule": "per city (LV, BOS, PIT, SG) the lowest-sha256('<log>|<token>') labelled bank rows at step 0 rank 0 whose log is in neither H48 nor sft_heldout_logs; 512 each; "
                                                "a pure function of the bank keys, the store manifest and the two exclusion lists; no model output is read",
            "frame_list_file": "stage0_frames.tsv", "frame_list_sha256": sha, **plan,
            "inputs": {"bank_keys": {"path": a.bank_keys, "sha256": sha256_file(a.bank_keys)}, "h48_logs_sha256": R.H48_SHA, "sft_heldout_sha256": sha256_file(a.sft_heldout), "store_sha256": store.sha},
            "tool_sha256": sha256_file(os.path.abspath(__file__)), "frozen_reader_sha256": V2.V1_SHA256}
    if repro is not None:
        open(os.path.join(a.out_dir, "stage0_repro256.tsv"), "w", encoding="utf-8", newline="\n").write(repro)
        meta["repro256_sha256"] = hashlib.sha256(repro.encode("utf-8")).hexdigest()
        meta["repro256_rule"] = "the 256 lowest sha256('<log>|<token>') keys of the SFT-24 logs among the rows of the banked head_pred_update10000.npz (keys only; values unread)"
    json.dump(meta, open(os.path.join(a.out_dir, "stage0_frames.json"), "w"), indent=1)
    print("WROTE", p_list, "sha256", sha)
    return 0


# ======================================================================== read
def load_pred(path, keys, nx=40, ny=32, what="predictions", exact=True):
    """npz of the reader's schema -> (logit, sdf) aligned to `keys` [(log, token)].  exact: the file must hold EXACTLY these keys (no extra, no missing, no duplicate)."""
    z = np.load(require_file(path, what), allow_pickle=False)
    miss = [k for k in ("log", "token", "step", "logit", "sdf_m") if k not in z.files]
    if miss:
        raise SystemExit(f"REFUSED: {what} lacks {miss}")
    if not (np.asarray(z["step"]) == 0).all():
        raise SystemExit(f"REFUSED: {what} holds rows with step != 0")
    if z["logit"].shape[1:] != (nx, ny) or z["sdf_m"].shape != z["logit"].shape:
        raise SystemExit(f"REFUSED: {what} logit shape {z['logit'].shape} is not [N,{nx},{ny}]")
    if not (np.isfinite(z["logit"]).all() and np.isfinite(z["sdf_m"]).all()):
        raise SystemExit(f"REFUSED: {what} holds non-finite outputs")
    fk = list(zip([str(x) for x in z["log"]], [str(x) for x in z["token"]]))
    if len(set(fk)) != len(fk):
        raise SystemExit(f"REFUSED: {what} holds duplicate keys")
    pos = {k: i for i, k in enumerate(fk)}
    if exact and set(fk) != set(keys):
        raise SystemExit(f"REFUSED: {what} keys differ from the frame list: {len(set(fk) - set(keys))} extra, {len(set(keys) - set(fk))} missing")
    idx = []
    for k in keys:
        if k not in pos:
            raise SystemExit(f"REFUSED: {what} lacks key {k}")
        idx.append(pos[k])
    return np.asarray(z["logit"], np.float32)[idx], np.asarray(z["sdf_m"], np.float32)[idx]


def decide(value, mutate=None):
    """the A5 draft's two rows, mechanically, both sides: point >= 0.85 -> S0-FIT, point < 0.85 -> S0-WEAK.  (mutate='flip_threshold' reads > instead: the regression arm.)"""
    if not math.isfinite(value):
        raise SystemExit("REFUSED: the in-sample IoU is not finite")
    ok = (value > FIT_BAR) if mutate == "flip_threshold" else (value >= FIT_BAR)
    return "S0-FIT" if ok else "S0-WEAK"


def sign_agreement(a, b):
    return float(((np.asarray(a) > 0) == (np.asarray(b) > 0)).mean())


def pooled_iou(logit, label, n=None, band="0-40"):
    st = R.frame_stats(logit, np.zeros_like(logit), label, np.ones(len(logit), bool))
    b = R.BAND_IDX[band]
    return float(st[f"inter{b}"].sum() / st[f"union{b}"].sum())


def stage0_read(rows, logit, sdf, label, store_city, banked, n_boot=R.N_BOOT, seed=R.SEED, mutate=None, s1a=None):
    """rows: the frame list; logit/sdf [N,40,32] aligned to rows; label [N,160,128]; store_city {log: city}; banked: (keys, logit_banked, logit_repro, label) for C0-ID or None -> result dict."""
    logs = np.asarray([r[0] for r in rows]).astype(str)
    grp = R.Groups(logs, {l: store_city[l] for l in set(logs)})
    W = R.boot_weights(grp.G, n_boot, seed)
    valid = np.ones(len(rows), bool)
    mut = "no_reweight" if mutate == "no_reweight" else None

    def p1(lg, band="0-40"):
        st = R.frame_stats(lg, sdf, label, valid)
        blk, reps, _ = R.p1_block(st, grp, W, band, mutate=mut)
        return blk, reps

    main_, _ = p1(logit)
    bands = {b: p1(logit, b)[0] for b in ("0-20", "20-40", "40-72")}
    flip, _ = p1(np.asarray(logit)[:, :, ::-1])
    floor, _ = p1(np.full_like(logit, 1.0))
    value = main_["point"]
    ctr = {"C0-FLIP": {"in_sample": value, "flipped": flip["point"], "margin": FLIP_MARGIN, "ok": bool(value - flip["point"] >= FLIP_MARGIN)},
           "C0-FLOOR": {"value": floor["point"], "note": "all-drivable constant head on the same frames (reported, not deciding)"}}
    if banked is not None:
        keys, lb, lr, labr = banked
        agree = sign_agreement(lb, lr)
        ib, ir = pooled_iou(lb, labr), pooled_iou(lr, labr)
        ctr["C0-ID"] = {"n_frames": len(keys), "sign_agreement": agree, "iou_banked": ib, "iou_repro": ir, "iou_diff": abs(ib - ir), "bars": {"sign_agreement": f">= {ID_SIGN_AGREE}", "iou_diff": f"<= {ID_IOU_TOL}"},
                        "ok": bool(agree >= ID_SIGN_AGREE and abs(ib - ir) <= ID_IOU_TOL), "note": "pooled [0,40) IoU over these frames (not city-reweighted)"}
    else:
        ctr["C0-ID"] = None
    invalid = [k for k in ("C0-ID", "C0-FLIP") if ctr[k] is None or not ctr[k]["ok"]]
    row = "S0-INVALID" if invalid else decide(value, mutate)
    # cross-check against the FROZEN reader's own table (an S1-P1/S1-Q failure with this in-sample value): both sides must agree
    if not invalid:
        v, _, _ = R.verdict(0.0, 0.0, None, 0.0, 0.0, 0.0, {}, value)
        if (v == "S1-FIT") != (row == "S0-FIT") and mutate is None:
            raise SystemExit(f"INTERNAL: the frozen reader's verdict() says {v} but this tool says {row}")
    res = {"tool": TOOL, "n_boot": n_boot, "seed": seed, "mutate": mutate, "n_frames": len(rows), "n_logs": grp.G, "frames_per_city": {c: int((np.asarray([r[2] for r in rows]) == c).sum()) for c in CITIES},
           "in_sample_iou": value, "ci95": main_["ci95"], "per_city": main_["per_city"], "bands": {"0-40": {"point": value, "ci95": main_["ci95"]}, **{b: {"point": v["point"], "ci95": v["ci95"]} for b, v in bands.items()}},
           "n_nan_resamples": main_["n_nan_resamples"], "controls": ctr, "invalid_controls": invalid, "row": row, "fit_bar": FIT_BAR,
           "row_rule": "S0-FIT iff the point >= 0.85 (the interval is reported, not deciding); S0-INVALID iff C0-ID or C0-FLIP fails; C0-FLOOR is reported",
           "next": {"S0-FIT": "HB alone (A5 section 2)", "S0-WEAK": "HB and HC as one registered pair (A5 sections 2, 3)", "S0-INVALID": "fix the tool, re-run; no HB / HC row starts"}[row]}
    if s1a:
        res["gap_vs_heldout"] = {"s1a_S1_P1_heldout": s1a, "in_sample_minus_heldout": value - s1a, "note": "reported, not deciding"}
    return res


def cmd_read(a):
    excl, h48, sft = excluded_logs_from(a.h48_logs, a.sft_heldout)
    store = Store(a.store, a.store_sha256_prefix)
    n_pc = a.per_city_for_tests or PER_CITY
    p_list = require_file(a.frame_list, "frame list")
    if sha256_file(p_list) != a.frame_list_sha256:
        raise SystemExit(f"REFUSED: the frame list's sha256 {sha256_file(p_list)} != the declared {a.frame_list_sha256}")
    rows = read_list(p_list)
    verify_list(rows, store, excl, n_pc)
    keys = [(r[0], r[1]) for r in rows]
    lg, sd = load_pred(a.preds_insample, keys, what="in-sample predictions")
    banked = None
    if a.banked_pred:
        bz = np.load(require_file(a.banked_pred, "banked head predictions"), allow_pickle=False)
        if a.banked_sha256 and sha256_file(a.banked_pred) != a.banked_sha256:
            raise SystemExit("REFUSED: banked head predictions sha256 differs from the declared one")
        rk = [(str(l), str(t)) for l, t in zip(bz["log"], bz["token"])]
        rz = np.load(require_file(a.preds_repro, "C0-ID re-forward"), allow_pickle=False)
        rkeys = list(zip([str(x) for x in rz["log"]], [str(x) for x in rz["token"]]))
        if len(rkeys) != (a.repro_n or N_REPRO) or len(set(rkeys)) != len(rkeys):
            raise SystemExit(f"REFUSED: the C0-ID re-forward holds {len(rkeys)} (distinct {len(set(rkeys))}) frames != {a.repro_n or N_REPRO}")
        if not set(rkeys) <= set(rk):
            raise SystemExit("REFUSED: a C0-ID frame is not in the banked export")
        if not {k[0] for k in rkeys} <= set(sft):
            raise SystemExit("REFUSED: a C0-ID frame is outside the 24 SFT held-out logs")
        pos = {k: i for i, k in enumerate(rk)}
        lb = np.asarray(bz["logit"], np.float32)[[pos[k] for k in rkeys]]
        lr, _ = load_pred(a.preds_repro, rkeys, what="C0-ID re-forward")
        city_of = {l: store.city[l] for l in {k[0] for k in rkeys} if l in store.city}
        if len(city_of) != len({k[0] for k in rkeys}):
            raise SystemExit("REFUSED: a C0-ID log is not in the label store")
        banked = (rkeys, lb, lr, store.labels(rkeys))
    elif a.preds_repro:
        raise SystemExit("REFUSED: --preds-repro without --banked-pred")
    s1a = None
    if a.s1a_json:
        s1a = float(json.load(open(require_file(a.s1a_json, "S1a json")))["bars"]["S1-P1"]["value"])
    plan = {"frames": len(rows), "logs": len({r[0] for r in rows}), "frame_list_sha256": a.frame_list_sha256, "c0_id_frames": None if banked is None else len(banked[0]), "n_boot": a.n_boot}
    print("PLAN", json.dumps(plan))
    if banked is None:
        raise SystemExit("REFUSED: C0-ID needs --banked-pred and --preds-repro (a tool that cannot reproduce S1a's rows has no authority over the in-sample number)")
    if a.dry_run:
        print("DRY-RUN: every input is present, hashed and shaped; nothing computed or written")
        return 0
    store_city = dict(store.city)
    res = stage0_read(rows, lg, sd, store.labels(keys), store_city, banked, n_boot=a.n_boot, s1a=s1a)
    res.update({"utc": utc(), "tool_sha256": sha256_file(os.path.abspath(__file__)), "frozen_reader_sha256": V2.V1_SHA256, "frame_list_sha256": a.frame_list_sha256,
                "inputs": {"preds_insample": sha256_file(a.preds_insample), "preds_repro": sha256_file(a.preds_repro), "banked_pred": sha256_file(a.banked_pred)}})
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(a.out_dir, "stage0_read.json"), "w"), indent=1, default=float)
    json.dump({"in_sample_iou": float(res["in_sample_iou"]), "row": res["row"], "frame_list_sha256": a.frame_list_sha256, "tool_sha256": res["tool_sha256"]}, open(os.path.join(a.out_dir, "in_sample_iou.json"), "w"), indent=1)
    print("STAGE0", res["row"], "in-sample IoU", res["in_sample_iou"], res["ci95"], "invalid controls:", res["invalid_controls"])
    return 0


# ======================================================================== s1f
def s1f_repro_check(pod_step0, rebuilt_step0):
    """the rebuilt seed-0 head must reproduce the pod's head_pred_step0.npz on its frames: -> (ok, sign agreement)."""
    ag = sign_agreement(pod_step0, rebuilt_step0)
    return bool(ag >= INIT_SIGN_AGREE), ag


def build_v2_manifest(base, init_a, init_b, in_sample_json):
    """the S1a manifest + the initial-head arm + the Stage 0 value -> a v2 manifest (HA-s0, HA-SHUF, HA-s0-initial)."""
    m = json.loads(json.dumps(base))
    for k, p in (("head_init0_a", init_a), ("head_init0_b", init_b), ("in_sample", in_sample_json)):
        m["files"][k] = {"path": p, "sha256": sha256_file(p)}
    m["arms"] = [{"name": "HA-s0", "role": "trained", "files": ["head_s0_a", "head_s0_b"], "cell": 2.0, "shuf": "HA-SHUF", "init": "HA-s0-initial", "in_sample_iou_file": "in_sample"},
                 {"name": "HA-SHUF", "role": "shuf", "files": ["head_shuf_a", "head_shuf_b"], "cell": 2.0},
                 {"name": "HA-s0-initial", "role": "init", "files": ["head_init0_a", "head_init0_b"], "cell": 2.0}]
    return m


def compare_rows(s0_row, reader_verdict):
    """A5 draft 1: 'if the Stage 0 tool and the reader disagree on the row, the card is read first and the discrepancy is reported'."""
    want = {"S0-FIT": "S1-FIT", "S0-WEAK": "S1-WEAK"}.get(s0_row)
    return {"stage0_row": s0_row, "reader_verdict": reader_verdict, "agree": reader_verdict == want, "note": "" if reader_verdict == want else "DISCREPANCY: read the card's S1 table first"}


def cmd_s1f(a):
    base = json.load(open(require_file(a.base_manifest, "S1a manifest")))
    sj = json.load(open(require_file(a.in_sample_json, "in_sample_iou.json")))
    if sj.get("row") == "S0-INVALID":
        raise SystemExit("REFUSED: Stage 0 is S0-INVALID; no S1-F read until it is fixed")
    pod = np.load(require_file(a.init_pod_step0, "the pod's head_pred_step0.npz"), allow_pickle=False)
    reb = np.load(require_file(a.init_rebuilt_a, "the rebuilt initial head on the pod's step-0 frames"), allow_pickle=False)
    pk = list(zip([str(x) for x in pod["log"]], [str(x) for x in pod["token"]]))
    rk = list(zip([str(x) for x in reb["log"]], [str(x) for x in reb["token"]]))
    if sorted(pk) != sorted(rk):
        raise SystemExit("REFUSED: the rebuilt initial head's frames differ from the pod's step-0 frames")
    pos = {k: i for i, k in enumerate(rk)}
    ok, ag = s1f_repro_check(np.asarray(pod["logit"]), np.asarray(reb["logit"])[[pos[k] for k in pk]])
    print("S1F-REPRO", json.dumps({"frames": len(pk), "sign_agreement": ag, "ok": ok}))
    if not ok:
        raise SystemExit(f"STOP: the rebuilt seed-0 head does not reproduce head_pred_step0.npz (sign agreement {ag:.6f} < {INIT_SIGN_AGREE}); the 1,705-frame-subset S1-F read of A5 section 1 is NOT BUILT in this tool")
    man = build_v2_manifest(base, a.init_pod_step0, a.init_rebuilt_b, a.in_sample_json)
    if a.dry_run:
        print("DRY-RUN: the v2 manifest builds; nothing written")
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    mp = os.path.join(a.out_dir, "manifest_s1f_v2.json")
    json.dump(man, open(mp, "w"), indent=1)
    d, audit = V2.load_real(man)
    r = V2.run_read(d, n_boot=a.n_boot, arms=d.pop("_spec"), pairs=d.pop("_pairs"))
    r["audit"], r["manifest_sha256"] = audit, sha256_file(mp)
    r["stage0_vs_reader"] = compare_rows(sj["row"], r["verdict"])
    json.dump(r, open(os.path.join(a.out_dir, "s1f_read_v2.json"), "w"), indent=1, default=float)
    open(os.path.join(a.out_dir, "s1f_read_v2_tables.md"), "w", encoding="utf-8", newline="\n").write(V2.tables_md(r))
    print("S1F", json.dumps(r["bars"]["S1-F"]), "| reader verdict", r["verdict"], "|", json.dumps(r["stage0_vs_reader"]))
    return 0


# ======================================================================== CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--dry-run", action="store_true", help="validate every input and print the plan; write nothing")
        p.add_argument("--out-dir", required=True)

    p = sub.add_parser("make-list")
    common(p)
    p.add_argument("--bank-keys", required=True, help="the bank's rows: .npz (log, token, step, rank) or TSV with that header (keys only)")
    p.add_argument("--store", required=True, help="the aux label store dir (manifest.json + shards/)")
    p.add_argument("--h48-logs", required=True, help="the H48 list (sha256 2ae13935... asserted)")
    p.add_argument("--sft-heldout", required=True, help="splits/sft_heldout_logs.txt")
    p.add_argument("--banked-pred", default="", help="HA-s0's head_pred_update10000.npz (keys only): also writes the 256 C0-ID frames")
    p.add_argument("--store-sha256-prefix", default=R.STORE_SHA_PREFIX)
    p.add_argument("--per-city-for-tests", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--repro-n-for-tests", type=int, default=0, help=argparse.SUPPRESS)
    p.set_defaults(fn=cmd_make_list)

    p = sub.add_parser("read")
    common(p)
    p.add_argument("--frame-list", required=True)
    p.add_argument("--frame-list-sha256", required=True, help="the sha256 written by make-list; refused on mismatch")
    p.add_argument("--preds-insample", required=True, help="npz (log, token, step, logit, sdf_m) of the head on EXACTLY the list's frames")
    p.add_argument("--banked-pred", required=True, help="HA-s0's head_pred_update10000.npz (C0-ID reference)")
    p.add_argument("--banked-sha256", default="", help="declared sha256 of --banked-pred (refused on mismatch)")
    p.add_argument("--preds-repro", required=True, help="the tool's re-forward of the 256 C0-ID frames (same schema)")
    p.add_argument("--repro-n", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--store", required=True)
    p.add_argument("--h48-logs", required=True)
    p.add_argument("--sft-heldout", required=True)
    p.add_argument("--s1a-json", default="", help="s1a_read.json (reports the in-sample minus held-out gap)")
    p.add_argument("--n-boot", type=int, default=R.N_BOOT)
    p.add_argument("--store-sha256-prefix", default=R.STORE_SHA_PREFIX)
    p.add_argument("--per-city-for-tests", type=int, default=0, help=argparse.SUPPRESS)
    p.set_defaults(fn=cmd_read)

    p = sub.add_parser("s1f")
    common(p)
    p.add_argument("--base-manifest", required=True, help="the S1a inputs manifest (paths must resolve)")
    p.add_argument("--in-sample-json", required=True, help="stage 0's in_sample_iou.json")
    p.add_argument("--init-pod-step0", required=True, help="the pod's head_pred_step0.npz (1,705 frames of the 24 SFT logs)")
    p.add_argument("--init-rebuilt-a", required=True, help="the rebuilt seed-0 initial head on the same 1,705 frames")
    p.add_argument("--init-rebuilt-b", required=True, help="the rebuilt seed-0 initial head on the 1,502 extra-log frames (head_init0_b.npz)")
    p.add_argument("--n-boot", type=int, default=R.N_BOOT)
    p.set_defaults(fn=cmd_s1f)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
