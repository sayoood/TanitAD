"""steer_common.py -- shared pieces of the kit of card PREREG_V8F_PROOF_ADD100 (STEERREAD). Tier T0. Imports the kit of card ADD99 (cropread/fit) and reuses it unchanged.

Layout (pod): run1 = /workspace/v8proof/cropread/run1 (card ADD99: tok/D, tok/B, lists, dump/D_H5, dump/B_H5), box files in enc_out/box,
S_ROOT = /workspace/v8proof/cropread/s_out (loc/, steer/, ctl/, tok/S, fits/, dump/, read/).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("STEERFIT_KIT"), os.path.join(HERE, "..", "fit")):
    if _p and os.path.isdir(_p) and os.path.abspath(_p) not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))
import crop_fit as CFT  # noqa: E402

CARD = "PREREG_V8F_PROOF_ADD100"
RUN1 = "/workspace/v8proof/cropread/run1"
BOX_DIR = "/workspace/v8proof/cropread/enc_out/box"
S_ROOT = "/workspace/v8proof/cropread/s_out"
N_FOLDS = 5
LOC_NAMES = ("L0", "L1", "L2", "L3", "L4", "LA")
ANG_LIMIT_DEG = 16.0                 # azimuth = 16 * tanh, elevation = 16 * tanh (card section 2)
DIR_SCALE = 16.0                     # the steering direction enters head S-H5d divided by 16
N_CTL_EVAL = 6                       # control s2: the first 6 evaluation clips by sha256 hex of the clip id
S2_TOL_DEG = 1e-3
W16_HALF_AZ_DEG, W16_HALF_EL_DEG = 8.0, 7.95
LOC_HPARAMS = {"lr": 1e-3, "weight_decay": 0.05, "batch": 256, "epochs": 20, "warmup_epochs": 1, "beta_deg": 0.5, "seed": 0}
SHEADS = ("H5", "H5d")


# ============================================================================================ small helpers
def sha256_file(path):
    return CFT.sha256_file(path)


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_json(path, obj):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1)


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def paths(run1=RUN1, s_root=S_ROOT):
    return {"train_list": os.path.join(run1, "lists", "train_final.txt"), "eval_list": os.path.join(run1, "lists", "eval141.txt")}


def read_lists(run1):
    """-> (rows [(clip, role)] in list order, eval ids). Refuses an evaluation clip in the train list (the existing kit's own check)."""
    p = paths(run1)
    rows, ev = CFT.read_train_list(p["train_list"]), CFT.read_ids(p["eval_list"])
    CFT.refuse_overlap(rows, ev)
    return rows, ev


# ============================================================================================ folds and who steers whom (card section 2)
def fit_clip_ids(rows):
    return [c for c, r in rows if r == "fit"]


def loc_train_clips(name, fit_ids):
    """clips a localiser reads: Lk (k 0..4) the fit clips whose position (list order) modulo 5 is not k; LA all fit clips."""
    if name == "LA":
        return list(fit_ids)
    if name in LOC_NAMES:
        k = int(name[1])
        return [c for i, c in enumerate(fit_ids) if i % N_FOLDS != k]
    raise SystemExit("REFUSED: unknown localiser set %r" % name)


def steer_assignment(rows, eval_ids):
    """-> ordered list of (clip, role, localiser): fit clip at position i -> L(i mod 5); the 50 selection clips and the evaluation clips -> LA."""
    out, pos = [], 0
    for c, r in rows:
        if r == "fit":
            out.append((c, "fit", "L%d" % (pos % N_FOLDS)))
            pos += 1
        else:
            out.append((c, "sel", "LA"))
    out += [(c, "eval", "LA") for c in eval_ids]
    return out


def audit_assignment(assign, records, rows, eval_ids):
    """control s1, localiser part. records {name: loc record dict}. -> list of violations (empty = pass).
    No clip is steered by a localiser that read it; no evaluation clip is read by a localiser; each record reads exactly the clips its name defines."""
    bad = []
    fit_ids = fit_clip_ids(rows)
    ev = set(eval_ids)
    for n in LOC_NAMES:
        if n not in records:
            bad.append("no record for localiser %s" % n)
            continue
        read = list(records[n].get("clips_read", []))
        if set(read) & ev:
            bad.append("%s read %d evaluation clip(s)" % (n, len(set(read) & ev)))
        if read != loc_train_clips(n, fit_ids):
            bad.append("%s clips_read differs from the clips its name defines" % n)
    for clip, role, loc in assign:
        if loc in records and clip in set(records[loc].get("clips_read", [])):
            bad.append("clip %s (%s) is steered by %s which read it" % (clip, role, loc))
    return bad


def first_eval_by_sha(eval_ids, n=N_CTL_EVAL):
    """the first n evaluation clips in the order of the sha256 hex of the clip id."""
    return [c for _h, c in sorted((sha256_text(c), c) for c in eval_ids)][:n]


# ============================================================================================ box labels
def load_box(box_dir, clip):
    """-> (az [n], el [n], ok [n]) of the centre of the lead's box; ok = has_lead and both angles finite. Refuses a missing file."""
    p = os.path.join(box_dir, clip + ".npz")
    if not os.path.isfile(p):
        raise SystemExit("REFUSED: no box file %s" % p)
    z = np.load(p, allow_pickle=False)
    az, el = np.asarray(z["ctr_az"], np.float64), np.asarray(z["ctr_el"], np.float64)
    ok = np.asarray(z["has_lead"]).astype(bool) & np.isfinite(az) & np.isfinite(el)
    return az, el, ok


# ============================================================================================ the localiser model (card section 2)
def make_loc_model():
    """the token reader of card ADD99 (crop_fit.make_model embed: layer norm, linear to 256, position vectors, one query, one cross-attention layer of 4 heads, token dropout 0.1),
    then an MLP 256-256-2; output = 16 * tanh (azimuth, elevation in degrees). The reader's own MLP is replaced by an identity so no parameter is left without a gradient."""
    import torch
    nn = torch.nn
    base = CFT.make_model("H1")
    base.mlp = nn.Identity()

    class Localiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.mlp = nn.Sequential(nn.Linear(CFT.D_MODEL, 256), nn.GELU(), nn.Linear(256, 2))

        def embed(self, x):
            return self.base.embed(x)

        def forward(self, x):
            """x [N, 130, 1024] float -> [N, 2] degrees (azimuth, elevation)"""
            return ANG_LIMIT_DEG * torch.tanh(self.mlp(self.embed(x)))
    return Localiser()


def load_localiser(loc_dir, dev):
    import torch
    ck = torch.load(os.path.join(loc_dir, "head.pt"), map_location="cpu")
    m = make_loc_model()
    m.load_state_dict(ck["state_dict"])
    m.to(dev)
    m.eval()
    return m, ck


def steer_clip(model, tokens, dev, chunk=64):
    """tokens [n, 2, 65, 1024] float16 (memmap ok) -> float32 [n, 2] (azimuth, elevation in degrees) on every frame."""
    import torch
    n = tokens.shape[0]
    out = np.empty((n, 2), np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, n, chunk):
            x = torch.from_numpy(np.array(tokens[i:i + chunk]).reshape(-1, CFT.N_TOK, CFT.WIDTH)).to(dev).float()
            out[i:i + len(x)] = model(x).double().cpu().numpy().astype(np.float32)
    return out


class SteerStore:
    """steering files <s_root>/steer/<clip>.npy float32 [n, 2], every frame finite; loaded whole on first use."""

    def __init__(self, s_root):
        self.dir = os.path.join(s_root, "steer")
        self.cache = {}

    def path(self, clip):
        return os.path.join(self.dir, clip + ".npy")

    def get(self, clip, n_frames=None):
        a = self.cache.get(clip)
        if a is None:
            p = self.path(clip)
            if not os.path.isfile(p):
                raise SystemExit("REFUSED: no steering file %s" % p)
            a = np.load(p)
            if a.ndim != 2 or a.shape[1] != 2 or a.dtype != np.float32 or not np.isfinite(a).all():
                raise SystemExit("REFUSED: steering file of %s has shape %s dtype %s or a non-finite value" % (clip, list(a.shape), a.dtype))
            self.cache[clip] = a
        if n_frames is not None and a.shape[0] != n_frames:
            raise SystemExit("REFUSED: steering file of %s has %d frames, the tokens have %d" % (clip, a.shape[0], n_frames))
        return a


def find_steer_shas(obj, clips):
    """steering sha256 map {clip: hex} found in a json record of the data side (ctl/S8.json or tok/S/MANIFEST.json). Accepted forms (the card names no key):
    a dict whose keys are clip ids and values 64-hex strings, under any key whose name contains 'steer'; or a dict clip -> {any key containing 'steer': hex};
    or a list of dicts carrying 'clip' and a key containing 'steer'. Searches the whole record. -> {} when none."""
    import re
    hexre = re.compile(r"[0-9a-f]{64}")
    cs = set(clips)
    found = {}

    def walk(o, under_steer):
        if isinstance(o, dict):
            if o and sum(1 for k in o if k in cs) >= max(1, len(o) // 2):
                for k, v in o.items():
                    if k not in cs:
                        continue
                    if isinstance(v, str) and hexre.fullmatch(v) and under_steer:
                        found[k] = v
                    elif isinstance(v, dict):
                        for kk, vv in v.items():
                            if "steer" in str(kk).lower() and isinstance(vv, str) and hexre.fullmatch(vv):
                                found[k] = vv
            if "clip" in o and isinstance(o.get("clip"), str):
                for kk, vv in o.items():
                    if "steer" in str(kk).lower() and isinstance(vv, str) and hexre.fullmatch(vv):
                        found[o["clip"]] = vv
            for k, v in o.items():
                walk(v, under_steer or "steer" in str(k).lower())
        elif isinstance(o, list):
            for v in o:
                walk(v, under_steer)
    walk(obj, False)
    return found
