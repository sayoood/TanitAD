#!/usr/bin/env python3
"""s1c_autoread.py -- the S1c auto-reader: a DETACHED dev-box job that reads arm `H1u10k-s0` BY ITSELF once the pod has pushed it, with no model turn in the loop.

    launch (detached, survives the session):  powershell -NoProfile -File launch_s1c_autoread.ps1
    dry run on S1's banked H1-s0 artifacts:   python s1c_autoread.py --dry-run          (reproduces the banked S1 reads end to end; writes only out/dryrun/)

THE TWO PRE-REGISTERED READS, applied MECHANICALLY (rows quoted verbatim from the prereg files; ambiguities resolved in QUESTIONS.md BEFORE any S1c number existed):
  * eval/PREREG_S1C.md   K2 on the 1,058 C3 tokens AND on the natural navtest population (nat_p1.py), the budget contrast vs S1's H1-s0 (paired log-cluster bootstrap), C1 identity, outcome O1-O4 / INVALID.
  * eval/PREREG_AUXVETO.md amendment A1: arm V-S1C = S1c's C3 rasters through the aux-veto rule (s1c_veto.py); its identity check is the harness's own C3 P1 of the same head.

PHASES (every wait is a script loop):
  1 POLL   Sayood/tanitad-refe-s1aux every --poll-min for `arms/H1u10k-s0/` COMPLETE (the pod pushes the arm in ONE commit: hf_egress.push -> create_commit): run_check.json, aux_drv.pt, sft.jsonl, the PASS token,
           PREFLIGHT_PASS.json, arms_manifest.jsonl, token_verify.json, dump/head_pred_{step0,update5000,update10000}.npz, dump/scores_update10000_B.npz and model_sft_B.pt (listed only: 1.29 GB, never fetched).
  2 FETCH  every needed file to the C: work dir, verified against the listing's size AND LFS sha256 / git blob id (a mismatch deletes the file and refuses).  D: has ~16 GB free; nothing big is written there.
  3 EVIDENCE  the arm's evidence must be the registered arm (token profile refe_aux_s1c, argv, sft.jsonl start/done/eval tags/lr 0, run_check ok) and C1 (step0 held-out IoU) must hold.
  4 FORWARD  the C3 forward of the S1c head and S1's H1-s0 head on the cached fp32 planner ctx (aux_navtest_forward.py, ONE call so the budget contrast is paired on one device).  Device `auto`: the dev-box GPU
           UNDER the shared lock first (the forward tool waits for the lock and for an idle GPU and exits 5 when busy: this script waits and retries, it never touches the lock file), then CPU.  The comparator H1-s0
           must reproduce S1's banked reads on whichever device ran (CPU: bit-exact; CUDA: 1e-4); a miss re-runs on CPU.  CPU is the precedent path of the banked S1 reads.
  5 READS  aux_s1_readout perception (navtest: P1/P2/P3/floor C4 + the budget contrast; held-out navtrain at update 5000 / 10000), nat_p1 (natural population), s1c_veto (V-S1C), s1c_desc_selection (arm B, descriptive).
  6 DECIDE s1c_rules: the PREREG_S1C row and the AUXVETO V-S1C row; RESULT.md; the marker.
MARKERS (out/): S1CREAD_DONE  = a RECORDED PREREG_S1C row (O1-O4); the AUXVETO row (or its INVALID) is in the payload.   S1CREAD_FAILED = anything else, INVALID included (no S1c read exists), with phase + reason.
SECURITY: the HF token is read IN PLACE from Keys.txt, kept in this process, never printed / written / put in argv / inherited by a child; every log line is scrubbed.  A re-run of the same command RESUMES
(completed stages are cached by input fingerprint).  Never `pgrep -f` / `pkill -f`: a single-instance lock.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")        # MUST precede any huggingface_hub import (dev-box transfers hang in CloseWait under hf_xet)

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PK = HERE.parent.parent
EVAL = PK / "eval"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EVAL))
import s1_read as S1R  # noqa: E402  -- the S1 read driver: fetch_verified / HFRemote / LocalRemote / argv helpers / floor_inputs / the harness command builders
import s1c_rules as R  # noqa: E402

TOOL = "s1c_autoread/1"
REPO = "Sayood/tanitad-refe-s1aux"
PY_T = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PY_EVAL = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"           # the venv of the banked S1 reads (numpy 1.26, torch 2.7.1)
ART = "D:/Projects/TanitAD-artifacts"
S1_CACHE = f"{ART}/s1_read_cache"
CTX_CACHE = f"{ART}/aux_navtest_forward/c3_full_ctx"
FLOOR_DIR = f"{ART}/aux_navtest_forward/p3_floor_full/heads"
HELDOUT_LOGS = f"{ART}/_tlane_work/ho/refe_heldout/heldout_logs.txt"
WORK_DEFAULT = "C:/Users/Admin/s1c_work"
KEYS_CANDS = ("D:/Projects/TanitAD/Keys.txt", "E:/Projects/TanitAD/Keys.txt")
S1_READ_DIR = PK / "raw/2026-10-15-s1-read"
BANKED_NAV = S1_READ_DIR / "instrument/perception_navtest.json"
BANKED_HO = S1_READ_DIR / "instrument/perception_heldout.json"
BANKED_SEL = S1_READ_DIR / "instrument/selection.json"
BANKED_HEADS = S1_READ_DIR / "instrument/heads/run.json"
ARGV_DIFF = PK / "raw/2026-10-16-s1c-build/ARGV_DIFF.json"
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{8,}")
COMPARATOR = "H1-s0"
N_ARMS = ("N-s0", "N-s1", "N-s2")
IDENT_TOL = {"cpu": 1e-9, "cuda": 1e-4}                 # the comparator's reproduction of S1's banked reads; CUDA kernels need not be bit-exact, 1e-4 is 350x below the smallest F
HEAD_PARAMS = 2440450
TRAIN_SETS, HELDOUT_SETS, HELDOUT_LOGS_N = 174912, 3137, 24
S_PER_UPDATE_RANGE = (1.24, 1.8)                         # PREREG_S1C section 7 operational check (NOT outcome-bearing)
TRAIN_HOURS_RANGE = (3.4, 4.7)


class Refuse(Exception):
    """a verification failed: write S1CREAD_FAILED and stop (never retried silently)"""


# ================================================================================================================ the arm under read
@dataclass(frozen=True)
class ArmSpec:
    name: str
    cls: str
    seed: int
    updates: int
    batch: int
    accum: int | None                       # None = the parser default (S1's argv leaves --batch/--accum unnamed: batch 8, accum 8)
    eval_every: int
    profile: str
    tags: tuple
    budget_flags_named: bool
    lr0_required: bool
    expected_argv_key: str
    extra_listed: tuple = ("model_sft_B.pt",)
    op_check: bool = True

    @property
    def final_tag(self) -> str:
        return self.tags[-1]

    @property
    def micro_per_update(self) -> int:
        return 8 if self.accum is None else self.accum


REAL = ArmSpec("H1u10k-s0", "H1u10k", 0, 10000, 8, 1, 5000, "refe_aux_s1c", ("step0", "update5000", "update10000"), True, True, "H1u10k-s0")
DRY = ArmSpec("H1-s0", "H1", 0, 400, 8, None, 400, "refe_aux_s1", ("step0", "update400"), False, False, "banked_H1-s0", extra_listed=(), op_check=False)
ARM_SMALL = ("arms_manifest.jsonl", "run_check.json", "PREFLIGHT_PASS.json", "sft.jsonl", "token_verify.json", "launch_gate.log")


def required_paths(spec: ArmSpec) -> list:
    a = f"arms/{spec.name}/"
    return [a + f for f in ARM_SMALL] + [a + "aux_drv.pt"] + [a + f"dump/head_pred_{t}.npz" for t in spec.tags] + [a + f"dump/scores_{spec.final_tag}_B.npz"]


def arm_status(listing: dict, spec: ArmSpec) -> dict:
    """The arm is READY iff every file the read needs (and the corroborating model_sft_B.pt) is in the listing, exactly one PASS token.  The pod's push is ONE commit, so presence == complete."""
    a = f"arms/{spec.name}/"
    miss = [p for p in required_paths(spec) + [a + f for f in spec.extra_listed] if p not in listing]
    tok = sorted(k for k in listing if re.fullmatch(rf"{re.escape(a)}PASS_[0-9a-f]+\.json", k))
    if len(tok) != 1:
        miss.append(f"{a}PASS_<id>.json (found {len(tok)})")
    return {"ready": not miss, "missing": miss, "token": tok[0] if len(tok) == 1 else None, "n_arm_files": sum(1 for k in listing if k.startswith(a))}


# ================================================================================================================ utilities
def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(p, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_token(cands) -> str | None:
    """the first `hf_...` token in the first readable Keys file, IN PLACE.  Returned to the caller; never printed or written here."""
    for c in cands:
        try:
            m = re.search(r"hf_[A-Za-z0-9]+", Path(c).read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if m:
            return m.group(0)
    return None


def pid_alive(pid) -> bool:
    try:
        r = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"], capture_output=True, timeout=30)
        return f'"{int(pid)}"' in r.stdout.decode("utf-8", "replace")
    except Exception:
        return True                                       # cannot tell = assume alive (never steal a lock on a failed probe)


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, p)


def fp_of(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


def numeric_leaves(a, b, path="", skip=("input", "note", "init_pred_sha256", "pred_sha256", "sha256", "estimator", "tool", "tier", "not_computed")):
    """Yield (path, x, y) for every leaf of two JSON-like trees; a structural mismatch yields (path, 'STRUCT', ...).  Numbers are compared by the caller, strings must be equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in skip:
                continue
            if k not in a or k not in b:
                yield (f"{path}/{k}", "STRUCT", "missing key in one tree")
                continue
            yield from numeric_leaves(a[k], b[k], f"{path}/{k}", skip)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield (f"{path}", "STRUCT", f"length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            yield from numeric_leaves(x, y, f"{path}[{i}]", skip)
    else:
        yield (path, a, b)


def deep_compare(a, b, tol: float) -> dict:
    """max |a - b| over the numeric leaves of two JSON trees (structure and string leaves must match exactly)."""
    n, worst, worst_at, bad = 0, 0.0, None, []
    for path, x, y in numeric_leaves(a, b):
        if x == "STRUCT":
            bad.append(f"{path}: {y}")
            continue
        if isinstance(x, bool) or isinstance(y, bool) or x is None or y is None or isinstance(x, str) or isinstance(y, str):
            if x != y:
                bad.append(f"{path}: {x!r} != {y!r}")
            continue
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if (isinstance(x, float) and math.isnan(x)) and (isinstance(y, float) and math.isnan(y)):
                n += 1
                continue
            d = abs(float(x) - float(y))
            n += 1
            if d > worst:
                worst, worst_at = d, path
    return {"ok": bool(not bad and worst <= tol), "n_numeric_leaves": n, "max_abs_diff": worst, "at": worst_at, "tol": tol, "structural_or_string_mismatches": bad[:10]}


# ================================================================================================================ evidence (the arm is the registered arm)
def check_evidence(spec: ArmSpec, ev: dict, expected_argv: list | None = None) -> tuple:
    """-> (problems, warnings, facts).  A problem means the arm's evidence contradicts its label or the registration: the read is INVALID (PREREG_S1C section 4, row INVALID)."""
    P, W, F = [], [], {}
    arm = spec.name
    rows = [r for r in ev["manifest"] if r.get("arm") == arm]
    if not rows:
        P.append(f"ID_MANIFEST_NO_ROW: arms_manifest.jsonl has no row for {arm}")
    elif ev["manifest"][-1].get("arm") != arm:
        P.append(f"ID_MANIFEST_LAST_ROW: the manifest's last row names {ev['manifest'][-1].get('arm')!r}, not {arm}")
    rc = ev["run_check"]
    if rc.get("arm") != arm or rc.get("ok") is not True or rc.get("problems"):
        P.append(f"ID_RUNCHECK: run_check arm={rc.get('arm')!r} ok={rc.get('ok')} problems={rc.get('problems')}")
    tk = ev.get("token")
    if tk is None:
        P.append("ID_TOKEN_MISSING: no single PASS_<id>.json")
    else:
        b = tk.get("binding", {})
        if tk.get("verdict") != "PASS" or b.get("profile") != spec.profile:
            P.append(f"ID_TOKEN_PROFILE: verdict={tk.get('verdict')} profile={b.get('profile')} (registered {spec.profile})")
        argv = b.get("argv", [])
        if S1R.argv_sha256(argv) != b.get("argv_sha256"):
            P.append("ID_ARGV_SHA: sha256(compact JSON of the token's argv) != the token's argv_sha256")
        if rows and not any(r.get("argv_sha256") == b.get("argv_sha256") and r.get("token_sha256") == ev.get("token_sha256") for r in rows):
            P.append("ID_TOKEN_MANIFEST: no manifest row of this arm carries this token's file sha256 and argv sha256")
        fl = S1R.argv_flags(argv)
        if S1R.argv_class(fl) != "H1":
            P.append(f"ID_ARGV_CLASS: the argv is an {S1R.argv_class(fl)} arm; {arm} is H1 (head on, feed OFF)")
        if fl.get("--seed") != str(spec.seed):
            P.append(f"ID_ARGV_SEED: argv --seed {fl.get('--seed')} != {spec.seed}")
        if fl.get("--out", "").rstrip("/").split("/")[-1] != arm:
            P.append(f"ID_ARGV_OUT: argv --out ends in {fl.get('--out', '').rstrip('/').split('/')[-1]!r}, not {arm}")
        if fl.get("--dump-scores", "").rstrip("/").split("/")[-2:-1] != [arm]:
            P.append(f"ID_ARGV_DUMP: argv --dump-scores {fl.get('--dump-scores')!r} is not under .../{arm}/dump")
        if fl.get("--max-updates") != str(spec.updates) or fl.get("--eval-every") != str(spec.eval_every):
            P.append(f"ID_ARGV_UPDATES: --max-updates {fl.get('--max-updates')} --eval-every {fl.get('--eval-every')} (registered {spec.updates} / {spec.eval_every})")
        if spec.budget_flags_named:
            if fl.get("--batch") != str(spec.batch) or fl.get("--accum") != str(spec.accum):
                P.append(f"ID_ARGV_BUDGET: --batch {fl.get('--batch')} --accum {fl.get('--accum')} (registered {spec.batch} / {spec.accum})")
        elif "--batch" in fl or "--accum" in fl:
            P.append(f"ID_ARGV_BUDGET: S1's budget leaves --batch/--accum to the parser defaults; the argv names {fl.get('--batch')}/{fl.get('--accum')}")
        if fl.get("--w-aux-drv") != "0.1" or fl.get("--aux-drv-lr") not in ("2e-4", "0.0002"):
            P.append(f"ID_ARGV_HEADFLAGS: --w-aux-drv {fl.get('--w-aux-drv')} --aux-drv-lr {fl.get('--aux-drv-lr')} (registered 0.1 / 2e-4)")
        if "--aux-drv" not in fl or "--aux-drv-feed" in fl:
            P.append("ID_ARGV_HEAD: an H1 arm names --aux-drv and NOT --aux-drv-feed")
        if expected_argv is not None and argv != expected_argv:
            diff = sorted(set(map(str, argv)) ^ set(map(str, expected_argv)))
            W.append(f"ARGV_EXPECTED_DIFF: the token's argv differs from the registered argv ({len(diff)} differing tokens, first {diff[:4]}); the substance flags above are enforced independently")
        F["token"] = {"file": ev.get("token_file"), "commit": b.get("commit"), "profile": b.get("profile"), "argv_sha256": b.get("argv_sha256")}
    sft = ev["sft"]
    by: dict = {}
    for r in sft:
        by.setdefault(r.get("event"), []).append(r)
    st, dn = by.get("start", []), by.get("done", [])
    if len(st) != 1 or st[0].get("updates") != spec.updates:
        P.append(f"ID_SFT_START: start events {[r.get('updates') for r in st]} (registered exactly one with updates == {spec.updates})")
    elif st[0].get("batch") != spec.batch or st[0].get("micro_per_update") != spec.micro_per_update:
        P.append(f"ID_SFT_BUDGET: start batch {st[0].get('batch')} micro_per_update {st[0].get('micro_per_update')} (registered {spec.batch} / {spec.micro_per_update})")
    if len(dn) != 1 or dn[0].get("updates") != spec.updates:
        P.append(f"ID_SFT_DONE: done events {[r.get('updates') for r in dn]} (registered exactly one with updates == {spec.updates})")
    for kind in ("eval", "aux_eval"):
        tags = {r.get("tag") for r in by.get(kind, [])}
        if not set(spec.tags) <= tags:
            P.append(f"ID_SFT_TAGS: {kind} tags {sorted(map(str, tags))} lack {sorted(set(spec.tags) - tags)}")
    data = (by.get("data") or [{}])[0]
    if data.get("train_sets") != TRAIN_SETS or data.get("heldout_sets") != HELDOUT_SETS or data.get("heldout_logs") != HELDOUT_LOGS_N or data.get("train_rows_in_heldout_logs") != 0:
        P.append(f"ID_DATA: train_sets {data.get('train_sets')} heldout_sets {data.get('heldout_sets')} heldout_logs {data.get('heldout_logs')} train_rows_in_heldout_logs {data.get('train_rows_in_heldout_logs')} "
                 f"(audited {TRAIN_SETS} / {HELDOUT_SETS} / {HELDOUT_LOGS_N} / 0)")
    ad = data.get("aux_drv") or {}
    if not (ad.get("head_built") is True and ad.get("feed_built") is False and ad.get("n_params") == HEAD_PARAMS and ad.get("tap") == "visual" and ad.get("feed") == "off"
            and ad.get("w_aux") == 0.1 and ad.get("lr") == 0.0002 and str(ad.get("store_sha256")) == S1R_LABEL_TRAIN_SHA):
        P.append(f"ID_HEAD_CONFIG: data.aux_drv {{head_built {ad.get('head_built')}, feed_built {ad.get('feed_built')}, n_params {ad.get('n_params')}, tap {ad.get('tap')}, feed {ad.get('feed')}, "
                 f"w_aux {ad.get('w_aux')}, lr {ad.get('lr')}, store {str(ad.get('store_sha256'))[:12]}}} is not the registered H1 head")
    train = [r for r in by.get("train", []) if isinstance(r, dict)]
    if spec.lr0_required and train:
        last = max(train, key=lambda r: r.get("update", 0))
        if last.get("update") != spec.updates or last.get("of") != spec.updates or not (isinstance(last.get("lr"), (int, float)) and abs(last["lr"]) <= 1e-12):
            P.append(f"ID_SFT_SCHEDULE: last train row is update {last.get('update')} of {last.get('of')} at lr {last.get('lr')}; want update {spec.updates} of {spec.updates} at lr 0.0")
    ab = [r.get("aux_bce") for r in train if "aux_bce" in r]
    if not ab or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in ab):
        P.append(f"ID_SFT_AUXBCE: aux_bce on {len(ab)} train rows, not all finite")
    ae = {r.get("tag"): r for r in by.get("aux_eval", [])}
    F["aux_eval_iou_by_tag"] = {t: ae[t].get("iou") for t in spec.tags if t in ae}
    F["step0_iou"] = ae.get("step0", {}).get("iou")
    F["aux_bce_first_last"] = [ab[0], ab[-1]] if ab else None
    spu = [r["s_per_update"] for r in train if isinstance(r.get("s_per_update"), (int, float))]
    F["s_per_update_mean"] = sum(spu) / len(spu) if spu else None
    F["train_hours"] = dn[0].get("hours") if dn else None
    if spec.op_check:                                                    # PREREG_S1C section 7: operational, NOT outcome-bearing
        lo, hi = S_PER_UPDATE_RANGE
        h0, h1 = TRAIN_HOURS_RANGE
        F["operational_check"] = {"s_per_update_in_range": bool(F["s_per_update_mean"] is not None and lo <= F["s_per_update_mean"] <= hi), "s_per_update_range": [lo, hi],
                                  "train_hours_in_range": bool(F["train_hours"] is not None and h0 <= F["train_hours"] <= h1), "train_hours_range": [h0, h1],
                                  "note": "a reading outside the range means the COST MODEL is wrong, not the claim (PREREG_S1C section 7)"}
    return P, W, F


S1R_LABEL_TRAIN_SHA = "4a17db673b1548ba35724ab848a7dc551e4b0bc1e49c57ae93e6357454bf287a"      # aux_s1_readout.LABEL_PINS["train"][0]


# ================================================================================================================ the instruments' command lines
def forward_cmd(py, heads: dict, out_dir, ctx_cache, device: str) -> list:
    cmd = [py, str(EVAL / "aux_navtest_forward.py"), "--out", str(out_dir), "--ctx-cache", ctx_cache, "--device", device]
    if device == "cpu":
        cmd.append("--no-gpu-lock")                       # CPU needs no lock (the tool takes it only for dev == cuda)
    return cmd + ["--heads"] + [f"{a}={p}" for a, p in sorted(heads.items())]


def nat_cmd(py, preds: dict, contrasts: list, out) -> list:
    return [py, "-I", "-X", "utf8", str(HERE / "nat_p1.py"), "--pred"] + [f"{a}={p}" for a, p in sorted(preds.items())] + (["--contrast"] + [f"{a}:{b}" for a, b in contrasts] if contrasts else []) + ["--out", str(out)]


def veto_cmd(py, arm_pred, label, expect_iou, out, with_vpred=False) -> list:
    return [py, "-I", "-X", "utf8", str(HERE / "s1c_veto.py"), "--head-pred", arm_pred, "--label", label, "--expect-iou", repr(float(expect_iou)), "--out", str(out)] + (["--with-vpred"] if with_vpred else [])


def desc_cmd(py, scores: dict, pdm_dir, out) -> list:
    return [py, "-I", "-X", "utf8", str(HERE / "s1c_desc_selection.py"), "--scores"] + [f"{a}={p}" for a, p in sorted(scores.items())] + ["--pdm", str(pdm_dir), "--out", str(out)]


# ================================================================================================================ remotes
class ScopedLocalRemote(S1R.LocalRemote):
    """A directory standing in for the repo (the dry run): only `arms/<arm>/` is listed, hashed on demand."""

    def __init__(self, root, arms):
        super().__init__(root)
        self.arms = list(arms)

    def listing(self) -> dict:
        out = {}
        for arm in self.arms:
            base = Path(self.root) / "arms" / arm
            for d, _dn, fs in os.walk(base):
                for f in fs:
                    p = Path(d) / f
                    rel = p.relative_to(self.root).as_posix()
                    big = p.stat().st_size > 4096 or rel.endswith((".npz", ".pt", ".tar"))
                    out[rel] = {"size": p.stat().st_size, "sha256": sha256_file(p) if big else None, "blob_id": S1R.git_blob_sha1(str(p))}
        return out


# ================================================================================================================ the job
class Job:
    def __init__(self, cfg: dict, remote, spec: ArmSpec, dry: bool = False, sleep=time.sleep, secrets=(), runner=None):
        self.c, self.remote, self.spec, self.dry, self.sleep = cfg, remote, spec, dry, sleep
        self.out = Path(cfg["out"])
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "stages").mkdir(exist_ok=True)
        self.work = Path(cfg["work"])
        self.cache = self.work / "cache"
        self.secrets = [s for s in secrets if s]
        self.runner = runner or self._run_child
        self.state = {"phase": "init", "started": utc(), "dry_run": dry, "pid": os.getpid(), "arm": spec.name}
        self.R: dict = {}

    # -- logging / state / markers
    def scrub(self, s) -> str:
        s = TOKEN_RE.sub("hf_***", str(s))
        for sec in self.secrets:
            s = s.replace(sec, "hf_***")
        return s

    def log(self, msg):
        line = f"{utc()} {self.scrub(msg)}"
        try:
            print(line, flush=True)
        except (UnicodeEncodeError, OSError):                       # a cp1252 / closed stdout must never kill a 48 h poll
            try:
                print(line.encode("ascii", "replace").decode("ascii"), flush=True)
            except Exception:
                pass
        try:
            with open(self.out / "autoread.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def set_state(self, **kw):
        self.state.update(kw, updated=utc())
        try:
            tmp = self.out / "autoread_state.json.tmp"
            tmp.write_text(self.scrub(json.dumps(self.state, indent=1, default=str)), encoding="utf-8")
            os.replace(tmp, self.out / "autoread_state.json")
        except OSError:
            pass

    def marker(self, kind: str, payload: dict):
        names = ("S1CREAD_DONE", "S1CREAD_FAILED", "S1CREAD_DRYRUN_OK")
        for other in names:
            if other != kind and (self.out / other).exists():
                (self.out / other).unlink()
        body = {"marker": kind, "when_utc": utc(), "repo": self.c["repo"], "arm": self.spec.name, **payload}
        (self.out / kind).write_text(self.scrub(json.dumps(body, indent=1, default=str)), encoding="utf-8")
        short = {k: v for k, v in payload.items() if k in ("s1c_row", "veto_row", "phase", "reason")}
        self.log(f"ZZ{kind} {json.dumps(short, default=str)}")

    # -- stage cache
    def stage(self, name: str, fp: str, fn):
        """run fn() once per input fingerprint.  Only a SUCCESSFUL result is cached: a 'busy' / 'NOT_READ' outcome is retried on the next launch."""
        p = self.out / "stages" / f"{name}.json"
        if p.is_file():
            try:
                d = jload(p)
                if d.get("fp") == fp:
                    self.log(f"stage {name}: cached ({d.get('utc')})")
                    return d["result"]
            except (OSError, ValueError):
                pass
        t0 = time.time()
        self.set_state(phase=name)
        res = fn()
        if not (isinstance(res, dict) and res.get("status") in ("busy", "NOT_READ")):
            jdump({"fp": fp, "utc": utc(), "seconds": round(time.time() - t0, 1), "result": res}, p)
        return res

    # -- child processes (stubbed in the tests)
    def child_env(self):
        env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN")}
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        return env

    def _run_child(self, label, argv, outfile, timeout_h) -> dict:
        assert not any(TOKEN_RE.search(str(a)) for a in argv), "a token-shaped string in argv"
        t0 = time.time()
        with open(outfile, "wb") as fh:
            p = subprocess.Popen([str(a) for a in argv], cwd=str(HERE), env=self.child_env(), stdout=fh, stderr=subprocess.STDOUT)
            limit = timeout_h * 3600
            while True:
                try:
                    rc = p.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    self.set_state(step=label, step_pid=p.pid, step_elapsed_s=round(time.time() - t0))
                    if time.time() - t0 > limit:
                        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
                        raise Refuse(f"step {label} exceeded {timeout_h} h and was killed")
        tail = [x.rstrip() for x in Path(outfile).read_text(encoding="utf-8", errors="replace").splitlines()[-3:]]
        return {"rc": rc, "seconds": round(time.time() - t0, 1), "tail": tail}

    def run_step(self, label, argv, ok_rcs=(0,), timeout_h=None) -> dict:
        self.log("RUN [" + label + "] " + " ".join(f'"{a}"' if " " in str(a) else str(a) for a in argv))
        r = self.runner(label, argv, str(self.out / f"{label}.out"), timeout_h or self.c["step_timeout_h"])
        self.log(f"[{label}] exit {r['rc']} after {r['seconds']} s :: {' | '.join(r['tail'])[:240]}")
        if r["rc"] not in ok_rcs:
            raise Refuse(f"step {label} exited {r['rc']} (see {label}.out): {' | '.join(r['tail'])[:200]}")
        return r

    # ------------------------------------------------------------------------------------------------ 1 poll
    def poll(self):
        c, spec = self.c, self.spec
        deadline = time.time() + c["max_wait_h"] * 3600
        n, errs = 0, 0
        while True:
            n += 1
            try:
                listing = self.remote.listing()
                errs = 0
                st = arm_status(listing, spec)
            except Exception as e:                                      # network / auth: counted, never silent
                errs += 1
                st = {"ready": False, "missing": [f"listing error {type(e).__name__}: {self.scrub(e)[:160]}"], "n_arm_files": 0}
                if errs >= c["max_errors"]:
                    raise Refuse(f"{errs} consecutive listing errors; last: {st['missing'][0]}")
            self.set_state(phase="poll", poll_n=n, last_poll=utc(), ready=st["ready"], n_arm_files=st["n_arm_files"], missing=st["missing"][:6])
            if st["ready"]:
                self.log(f"poll {n}: arms/{spec.name}/ is COMPLETE ({st['n_arm_files']} files; token {st['token']})")
                return listing, st
            self.log(f"poll {n}: not ready ({st['n_arm_files']} files under arms/{spec.name}/; missing {len(st['missing'])}: {st['missing'][:3]})")
            if time.time() > deadline:
                raise Refuse(f"arms/{spec.name}/ did not become complete within {c['max_wait_h']} h")
            self.sleep(c["poll_s"])

    # ------------------------------------------------------------------------------------------------ 2 fetch
    def fetch(self, listing: dict, st: dict) -> list:
        rels = required_paths(self.spec) + [st["token"]]
        rows = [S1R.fetch_verified(self.remote, listing, rel, str(self.cache)) for rel in rels]
        jdump(rows, self.out / "fetch_manifest.json")
        self.log(f"fetch: {len(rows)} files, {sum(r['size'] for r in rows)} B, all verified ({sum(1 for r in rows if r['cached'])} cached)")
        return rows

    def s1_file(self, rel: str, pin_sha: str) -> str:
        """a file of S1's cache (N arms' score dumps, the comparator's head), verified against the sha256 the S1 read banked; fetched from HF into the C: work dir when absent / wrong."""
        loc = Path(self.c["s1_cache"]) / rel
        if loc.is_file() and sha256_file(loc) == pin_sha:
            return str(loc)
        alt = self.cache / "s1" / rel
        if alt.is_file() and sha256_file(alt) == pin_sha:
            return str(alt)
        if self.dry:
            raise Refuse(f"{loc} is missing or not the banked file (sha256 pin {pin_sha[:12]}..)")
        listing = self.remote.listing() if not hasattr(self, "_s1_listing") else self._s1_listing
        self._s1_listing = listing
        row = S1R.fetch_verified(self.remote, listing, rel, str(self.cache / "s1"))
        if row["sha256"] != pin_sha:
            raise Refuse(f"{rel}: the HF copy's sha256 {row['sha256'][:12]}.. != the S1 read's banked {pin_sha[:12]}..")
        return str(self.cache / "s1" / rel)

    def pdm_dir(self) -> str:
        loc = Path(self.c["s1_cache"]) / "labels/x/heldout_pdm"
        if loc.is_dir():
            return str(loc)
        if self.dry:
            raise Refuse(f"{loc} missing")
        rows = [S1R.fetch_verified(self.remote, self._s1_listing_(), rel, str(self.cache / "s1")) for rel in S1R.LABEL_FILES]
        tar_sha = next(r["sha256"] for r in rows if r["path"].endswith("s1_heldout_pdm.tar"))
        mf = (self.cache / "s1/labels/MANIFEST_s1_labels_sha256.txt").read_text(encoding="utf-8")
        if f"{tar_sha}  s1_heldout_pdm.tar" not in mf:
            raise Refuse("s1_heldout_pdm.tar's sha256 is not the one in MANIFEST_s1_labels_sha256.txt")
        root = self.cache / "s1/labels/x"
        if not (root / "heldout_pdm").is_dir():
            S1R.safe_extract_tar(str(self.cache / "s1/labels/s1_heldout_pdm.tar"), str(root))
        return str(root / "heldout_pdm")

    def _s1_listing_(self):
        if not hasattr(self, "_s1_listing"):
            self._s1_listing = self.remote.listing()
        return self._s1_listing

    # ------------------------------------------------------------------------------------------------ 3 evidence
    def evidence(self) -> dict:
        spec = self.spec
        ev = S1R.load_evidence(spec.name, str(self.cache))
        exp = None
        try:
            exp = jload(ARGV_DIFF)[spec.expected_argv_key]["argv"]
        except (OSError, KeyError, ValueError):
            pass
        P, W, F = check_evidence(spec, ev, exp)
        c1 = R.c1_check(F["step0_iou"]) if F.get("step0_iou") is not None else {"ok": False, "step0_iou": None, "reason": "no aux_eval step0 row"}
        res = {"problems": P, "warnings": W, "facts": F, "c1": c1, "runcheck_ok": ev["run_check"].get("ok") is True and not ev["run_check"].get("problems") and ev["run_check"].get("arm") == spec.name,
               "evidence_ok": not P, "token_sha256": ev.get("token_sha256")}
        for w in W:
            self.log("WARN " + w)
        self.log(f"evidence: {'OK' if not P else 'PROBLEMS ' + '; '.join(P)[:300]}; C1 step0 IoU {F.get('step0_iou')} {'holds' if c1['ok'] else 'FAILS'}")
        return res

    # ------------------------------------------------------------------------------------------------ 4/5 the instruments
    def comparator_files(self) -> dict:
        """-> {arm: aux_drv.pt path} for the heads the forward reads; the comparator's head is S1's banked H1-s0 (sha256 pinned by the S1 read)."""
        pins = jload(BANKED_HEADS)["heads"]
        heads = {self.spec.name: str(self.cache / "arms" / self.spec.name / "aux_drv.pt")}
        if self.spec.name != COMPARATOR:
            heads[COMPARATOR] = self.s1_file(f"arms/{COMPARATOR}/aux_drv.pt", pins[COMPARATOR]["sha256"])
        for extra in self.c.get("extra_heads", ()):
            heads[extra] = self.s1_file(f"arms/{extra}/aux_drv.pt", pins[extra]["sha256"])
        return heads

    def forward(self, heads: dict, device: str) -> dict:
        d = self.out / f"forward_{device}"
        d.mkdir(exist_ok=True)
        deadline = time.time() + (self.c["gpu_wait_min"] * 60 if device == "cuda" else 0)
        backoff = 60
        while True:
            r = self.runner(f"forward_{device}", forward_cmd(self.c["py_eval"], heads, d, self.c["ctx_cache"], device), str(self.out / f"forward_{device}.out"), self.c["step_timeout_h"])
            self.log(f"[forward_{device}] exit {r['rc']} after {r['seconds']} s :: {' | '.join(r['tail'])[:200]}")
            if r["rc"] == 5 and device == "cuda":                       # GPU BUSY (nothing loaded, nothing written): the lock is somebody else's; wait, never break it
                if time.time() >= deadline:
                    return {"status": "busy", "why": f"the dev-box GPU lock/idle gate was not available within {self.c['gpu_wait_min']} min"}
                self.log(f"GPU busy (the shared lock is held or the GPU is not idle): retry in {backoff} s; the lock is never broken")
                self.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue
            break
        if r["rc"] != 0:
            raise Refuse(f"forward_{device} exited {r['rc']}: {' | '.join(r['tail'])[:200]}")
        ident = jload(d / "identity.json")
        if not ident.get("PASS"):
            raise Refuse(f"forward_{device}: the scorer-logit identity vs table.npz FAILED (max |d| {ident.get('max_abs_delta')}): the ctx cache is not the planner's")
        runj = jload(d / "run.json")
        preds = {}
        for a in heads:
            p = d / f"head_pred_{a}.npz"
            if not p.is_file():
                raise Refuse(f"forward_{device}: {p.name} was not written")
            preds[a] = str(p)
            info = runj["heads"][a]
            cfg = info["config"]
            if not (cfg.get("feed") == "off" and cfg.get("tap") == "visual" and info.get("n_params") == HEAD_PARAMS and not info.get("feed_present")):
                raise Refuse(f"forward_{device}: head {a} is not the H1 head (config {cfg})")
        spec_head = runj["heads"][self.spec.name]
        if spec_head.get("update") != self.spec.updates or cfg_seed(spec_head) != self.spec.seed:
            raise Refuse(f"forward_{device}: {self.spec.name}'s aux_drv.pt says update {spec_head.get('update')} seed {cfg_seed(spec_head)} (registered {self.spec.updates} / {self.spec.seed})")
        return {"status": "ok", "device": device, "preds": preds, "identity": {k: ident.get(k) for k in ("PASS", "n_tokens", "n_tokens_bit_identical", "max_abs_delta")},
                "head_sha256": {a: runj["heads"][a]["sha256"] for a in heads}, "seconds": runj.get("wall_s")}

    def perception_navtest(self, fw: dict, device: str) -> dict:
        spec = self.spec
        pkg = str(PK)
        fl = S1R.floor_inputs(self.c["floor_dir"], pkg)
        preds = dict(fw["preds"])
        keep = [spec.name] + ([COMPARATOR] if COMPARATOR in preds and COMPARATOR != spec.name else [])
        preds = {a: preds[a] for a in keep}
        inits = {a: fl[arm_seed(a)] for a in preds}
        out = self.out / f"perception_navtest_{device}.json"
        self.run_step(f"perception_navtest_{device}", S1R.cmd_perception_navtest(self.c["py_eval"], str(EVAL), pkg, preds, inits, str(out)))
        js = jload(out)
        S1R.check_harness_json(js, "perception_navtest")
        return {"json": str(out), "preds": preds, "inits": inits}

    def perception_heldout(self) -> dict:
        spec = self.spec
        if S1R.sha256_file(self.c["heldout_logs"]) != S1R.HELDOUT_LOGS_SHA256:
            raise Refuse("the held-out logs file is not the one the PASS tokens bind (sha256 b814998d...)")
        res = {}
        init = str(self.cache / "arms" / spec.name / "dump" / "head_pred_step0.npz")
        comp_pred = comp_init = None
        if spec.name != COMPARATOR:
            comp_pred = str(Path(self.c["s1_cache"]) / "arms" / COMPARATOR / "dump" / "head_pred_update400.npz")
            comp_init = str(Path(self.c["s1_cache"]) / "arms" / COMPARATOR / "dump" / "head_pred_step0.npz")
        for tag in self.spec.tags[1:]:
            preds = {spec.name: str(self.cache / "arms" / spec.name / "dump" / f"head_pred_{tag}.npz")}
            inits = {spec.name: init}
            if comp_pred and tag == spec.final_tag and Path(comp_pred).is_file() and Path(comp_init).is_file():       # informational: S1's H1-s0 held-out read, beside the S1c head
                preds[COMPARATOR], inits[COMPARATOR] = comp_pred, comp_init
            out = self.out / f"perception_heldout_{tag}.json"
            try:
                self.run_step(f"perception_heldout_{tag}", S1R.cmd_perception_heldout(self.c["py_eval"], str(EVAL), preds, inits, self.c["heldout_logs"], str(out)))
            except Refuse as e:
                if len(preds) < 2:
                    raise
                # the comparator beside the arm is informational (S1's H1-s0 held-out read): if the harness refuses the pair, read the arm ALONE (loudly) rather than lose its own held-out read
                self.log(f"WARN held-out read with the comparator beside the arm was refused ({str(e)[:160]}); re-reading the arm alone")
                preds, inits = {spec.name: preds[spec.name]}, {spec.name: inits[spec.name]}
                self.run_step(f"perception_heldout_{tag}_alone", S1R.cmd_perception_heldout(self.c["py_eval"], str(EVAL), preds, inits, self.c["heldout_logs"], str(out)))
            S1R.check_harness_json(jload(out), "perception_heldout")
            res[tag] = str(out)
        return res

    # ------------------------------------------------------------------------------------------------ the whole sequence
    def run(self) -> int:
        c, spec = self.c, self.spec
        try:
            self.log(f"start: repo {c['repo']} arm {spec.name} dry_run={self.dry} device={c['forward_device']} poll {c['poll_s']} s, pkg {PK}")
            if not self.dry and (self.out / "S1CREAD_DONE").is_file() and not c["force"]:
                self.log("S1CREAD_DONE already exists: nothing to do (--force re-runs)")
                return 0
            if not self.dry and (self.out / "S1CREAD_FAILED").exists():
                self.log("clearing the previous S1CREAD_FAILED (a relaunch is a retry; completed stages are reused)")
                (self.out / "S1CREAD_FAILED").unlink()
            try:
                self.R["prereg"] = {"s1c": R.guard_prereg_s1c(PK / "eval/PREREG_S1C.md"), "auxveto": R.guard_prereg_auxveto(PK / "eval/PREREG_AUXVETO.md")}
            except (ValueError, OSError) as e:
                raise Refuse(str(e))
            self.log("prereg guards: PREREG_S1C body and PREREG_AUXVETO body + amendment A1 are the registered texts" + (f"; AMENDMENTS PRESENT: {self.R['prereg']['s1c']['amendments']}" if self.R["prereg"]["s1c"]["amendments"] else ""))
            self.preflight()
            listing, st = self.poll()
            self.set_state(phase="fetch")
            self.R["fetch"] = self.fetch(listing, st)
            self.set_state(phase="evidence")
            ev = self.evidence()
            self.R["evidence"] = ev
            self.read_all(ev)
            return self.finish(ev)
        except (Refuse, S1R.FetchRefused, S1R.EvidenceRefused) as e:                 # a failed verification is a named refusal, never "unexpected"
            self.log(f"REFUSED/FAILED in phase {self.state.get('phase')}: {e}")
            self.marker("S1CREAD_FAILED", {"phase": self.state.get("phase"), "reason": str(e)})
            self.set_state(phase="failed")
            return 1
        except Exception as e:                                           # a bug is a failure with a name, never a silent death
            import traceback
            self.log("UNEXPECTED " + traceback.format_exc()[-1800:])
            self.marker("S1CREAD_FAILED", {"phase": self.state.get("phase"), "reason": f"unexpected {type(e).__name__}: {self.scrub(e)[:300]}"})
            self.set_state(phase="failed")
            return 1

    def preflight(self):
        c = self.c
        checks = {"driverl-eval python": Path(c["py_eval"]).is_file(), "tanitad python": Path(c["py_t"]).is_file(), "ctx cache (c3_full_ctx)": Path(c["ctx_cache"]).is_dir(),
                  "floor heads": Path(c["floor_dir"]).is_dir(), "held-out logs file": Path(c["heldout_logs"]).is_file(), "S1 cache arms": (Path(c["s1_cache"]) / "arms").is_dir(),
                  "instrument scripts": all((HERE / f).is_file() for f in ("nat_p1.py", "s1c_veto.py", "s1c_desc_selection.py", "s1c_rules.py")),
                  "banked S1 read json": BANKED_NAV.is_file() and BANKED_HO.is_file() and BANKED_SEL.is_file() and BANKED_HEADS.is_file()}
        self.work.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.work).free
        checks[f"work dir free >= {c['min_free_gib']} GiB"] = free >= c["min_free_gib"] * 2 ** 30
        for k, v in checks.items():
            self.log(f"preflight {'ok  ' if v else 'FAIL'} {k}")
        bad = [k for k, v in checks.items() if not v]
        if bad:
            raise Refuse("preflight failed: " + "; ".join(bad) + " (if the artifacts drive flipped letters, `subst D: E:/` restores D:; then relaunch: the same command resumes)")

    def read_all(self, ev: dict):
        c, spec = self.c, self.spec
        heads = self.comparator_files()
        mode = c["forward_device"]
        devices = {"cpu": ["cpu"], "cuda": ["cuda"], "auto": ["cuda", "cpu"]}[mode]
        chosen = None
        for i, dev in enumerate(devices):
            last = i == len(devices) - 1
            fp = fp_of("forward", dev, {a: sha256_file(p) for a, p in heads.items()})
            fw = self.stage(f"forward_{dev}", fp, lambda dev=dev: self.forward(heads, dev))
            if fw["status"] != "ok":
                self.log(f"forward on {dev}: {fw['why']}")
                if last:
                    raise Refuse(f"forward on {dev}: {fw['why']}")
                continue
            nav = self.stage(f"perception_navtest_{dev}", fp_of("pn", fp), lambda fw=fw, dev=dev: self.perception_navtest(fw, dev))
            nat = self.stage(f"nat_{dev}", fp_of("nat", fp), lambda fw=fw, dev=dev: self.nat(fw, dev))
            ident = self.comparator_identity(nav, nat, dev)
            self.log(f"comparator identity on {dev}: {'OK' if ident['ok'] else 'FAILED'} {json.dumps({k: v for k, v in ident.items() if k != 'detail'}, default=str)[:300]}")
            if ident["ok"]:
                chosen = {"device": dev, "forward": fw, "nav": nav, "nat": nat, "identity": ident}
                break
            if last:
                raise Refuse(f"instrument identity FAILED on {dev}: the comparator H1-s0 does not reproduce S1's banked reads ({ident['detail']})")
            self.log(f"the {dev} forward does not reproduce S1's banked H1-s0 reads: falling back to the next device")
        self.R["chosen"] = chosen
        dev = chosen["device"]
        self.R["heldout"] = self.stage("perception_heldout", fp_of("ho", ev["token_sha256"]), self.perception_heldout)
        nav_js = jload(chosen["nav"]["json"])
        p1_nav = nav_js["arms"][spec.name]["p1"]["all"]["0-40"]["est"]
        self.R["veto"] = self.stage("veto", fp_of("veto", chosen["forward"]["head_sha256"][spec.name], p1_nav), lambda: self.veto(chosen, p1_nav))
        self.R["desc"] = self.stage("desc_selection", fp_of("desc", ev["token_sha256"]), self.desc_selection)

    def nat(self, fw: dict, device: str) -> dict:
        spec = self.spec
        preds = {a: p for a, p in fw["preds"].items()}
        contrasts = []
        if spec.name != COMPARATOR:
            contrasts.append((spec.name, COMPARATOR))
        else:
            contrasts.append((spec.name, COMPARATOR))                      # the self-contrast: a ZERO control (exactly 0.0, CI [0, 0])
            if "H1-s1" in preds:
                contrasts.append(("H1-s1", COMPARATOR))
        out = self.out / f"nat_p1_{device}.json"
        r = self.run_step(f"nat_p1_{device}", nat_cmd(self.c["py_eval"], preds, contrasts, out), ok_rcs=(0, 6))
        js = jload(out) if out.is_file() else {}
        return {"json": str(out), "rc": r["rc"], "identity_ok": js.get("identity_ok")}

    def comparator_identity(self, nav: dict, nat: dict, device: str) -> dict:
        """The comparator H1-s0, read in the SAME forward call as the S1c head, must reproduce S1's banked reads: the harness's navtest block (deep compare, every numeric leaf) and the natural read."""
        tol = IDENT_TOL[device]
        nav_js, nat_js = jload(nav["json"]), jload(nat["json"])
        banked = jload(BANKED_NAV)["arms"][COMPARATOR]
        mine = nav_js["arms"][COMPARATOR]
        dc = deep_compare(mine, banked, tol)
        nat_id = nat_js["identity"].get(COMPARATOR)
        nat_ok = bool(nat_id and abs(nat_id["natural"] - nat_id["banked_natural"]) <= tol and abs(nat_id["c3"] - nat_id["banked_c3"]) <= tol and abs(nat_id["ess"] - nat_id["banked_ess"]) <= 1e-6)
        ok = bool(dc["ok"] and nat_ok)
        return {"ok": ok, "device": device, "tol": tol, "harness_block_vs_banked": dc, "natural_read": nat_id, "natural_ok": nat_ok, "detail": f"harness deep-compare max |d| {dc['max_abs_diff']:.3g} at {dc['at']}; natural ok {nat_ok}",
                "banked_source": "raw/2026-10-15-s1-read/instrument/perception_navtest.json arms.H1-s0 and raw/2026-10-16-aux-shift-locate/SHIFT_LOCATE.json"}

    def veto(self, chosen: dict, p1_nav: float) -> dict:
        spec = self.spec
        label = "V-S1C" if not self.dry else "V-DRYRUN-H1S0"
        out = self.out / "veto.json"
        self.run_step("veto", veto_cmd(self.c["py_t"], f"{spec.name}={chosen['forward']['preds'][spec.name]}", label, p1_nav, out, with_vpred=self.dry), ok_rcs=(0, 6))
        return {"json": str(out), "label": label}

    def desc_selection(self) -> dict:
        spec = self.spec
        pins = jload(BANKED_SEL)["inputs"]["scores_sha256"]
        scores = {spec.name: str(self.cache / "arms" / spec.name / "dump" / f"scores_{spec.final_tag}_B.npz")}
        for a in N_ARMS:
            scores[a] = self.s1_file(f"arms/{a}/dump/scores_update400_B.npz", pins[a])
        if spec.name != COMPARATOR:
            scores[COMPARATOR] = self.s1_file(f"arms/{COMPARATOR}/dump/scores_update400_B.npz", pins[COMPARATOR])
        out = self.out / "desc_selection.json"
        try:
            self.run_step("desc_selection", desc_cmd(self.c["py_eval"], scores, self.pdm_dir(), out))
            return {"status": "READ", "json": str(out)}
        except Refuse as e:                                              # descriptive, non-gating: recorded loudly, never fatal
            self.log(f"WARN descriptive selection NOT READ: {e}")
            return {"status": "NOT_READ", "why": str(e)}

    # ------------------------------------------------------------------------------------------------ decide + result
    def finish(self, ev: dict) -> int:
        spec = self.spec
        ch = self.R["chosen"]
        nav = jload(ch["nav"]["json"])
        nat = jload(ch["nat"]["json"])
        blk = nav["arms"][spec.name]
        p1_c3 = blk["p1"]["all"]["0-40"]["est"]
        a_nat = nat["arms"][spec.name]
        if abs(a_nat["c3"]["est"] - p1_c3) > 1e-12:
            raise Refuse(f"nat_p1's C3 read {a_nat['c3']['est']} != the harness's {p1_c3}: the two instruments disagree on the same frames")
        pair = f"{spec.name}-minus-{COMPARATOR}"
        d_nat = nat["contrasts"][pair]["natural"]
        d_c3 = nat["contrasts"][pair]["c3"]
        harness_pair = nav.get("head_fit_pairs", {}).get(pair)
        cross = None
        if harness_pair is not None:
            h = harness_pair["iou_0-40"]
            cross = {"est_abs_diff": abs(h["est"] - d_c3["est"]), "ci_abs_diff": max(abs(h["ci95"][0] - d_c3["ci95"][0]), abs(h["ci95"][1] - d_c3["ci95"][1]))}
            if cross["est_abs_diff"] > 1e-12 or cross["ci_abs_diff"] > 1e-12:
                raise Refuse(f"nat_p1's C3 contrast disagrees with the harness's head_fit_pairs[{pair}]: {cross}")
        s1c_text = R.read_lf(PK / "eval/PREREG_S1C.md")
        m = {"c1_ok": ev["c1"]["ok"], "runcheck_ok": ev["runcheck_ok"], "evidence_ok": ev["evidence_ok"], "p1_c3": p1_c3, "p1_nat": a_nat["natural"]["est"],
             "d_c3": {"est": d_c3["est"], "ci95": d_c3["ci95"]}, "d_nat": {"est": d_nat["est"], "ci95": d_nat["ci95"]}}
        dec = R.decide_s1c(m, s1c_text)
        sens = d_nat.get("sensitivity_rerake")
        dec["sensitivity_natural_contrast_rerake"] = {"ci95": sens["ci95"] if sens else None,
                                                      "separated_if_used": R.separated({"est": d_nat["est"], "ci95": sens["ci95"]}, R.F_NAT) if sens else None,
                                                      "same_class_as_primary": bool(sens and R.separated({"est": d_nat["est"], "ci95": sens["ci95"]}, R.F_NAT) == dec["separated"]["NAT"]),
                                                      "gating": False}
        vj = jload(self.R["veto"]["json"])
        if not vj["instrument_identity"]["ok"]:
            raise Refuse(f"s1c_veto does not reproduce the banked AUXVETO_READ.json (max |diff| {vj['instrument_identity']['max_abs_diff']}): the veto TOOL is not the one that made the banked read")
        self.R.update({"nav": nav, "nat": nat, "veto_json": vj, "decision": dec, "inputs_to_rule": m, "contrast_cross_check": cross})
        res_md = render_result(self)
        (self.out / "RESULT.md").write_text(res_md, encoding="utf-8", newline="\n")
        jdump({"decision": dec, "inputs_to_rule": m, "veto_decision": vj["decision"], "veto_validity": vj["validity"]["all_pass"], "chosen_device": ch["device"], "comparator_identity": ch["identity"],
               "evidence": {k: ev[k] for k in ("problems", "warnings", "facts", "c1")}, "prereg": self.R["prereg"]}, self.out / "RESULT.json")
        payload = {"s1c_row": dec["row"], "s1c_label": dec["label"], "veto_row": vj["decision"]["row"], "veto_valid": vj["validity"]["all_pass"], "P1_C3": p1_c3, "P1_NAT": m["p1_nat"],
                   "delta_C3": [d_c3["est"], d_c3["ci95"]], "delta_NAT": [d_nat["est"], d_nat["ci95"]], "separated": dec["separated"], "device": ch["device"], "result_md": str(self.out / "RESULT.md"),
                   "result_json": str(self.out / "RESULT.json")}
        if self.dry:
            self.marker("S1CREAD_DRYRUN_OK", dict(payload, note="DRY RUN on S1's H1-s0: the rows above are PLUMBING, not S1c results"))
            return 0
        if dec["row"] == "INVALID":
            self.marker("S1CREAD_FAILED", dict(payload, phase="decide", reason="INVALID read (PREREG_S1C row INVALID): " + "; ".join(ev["problems"] + ([] if ev["c1"]["ok"] else [f"C1 fails: step0 IoU {ev['c1'].get('step0_iou')}"]) +
                                                ([] if ev["runcheck_ok"] else ["run-check not ok"])) + " (no S1c read exists; RESULT.md carries the INVALID row and its next lever)"))
            return 1
        self.marker("S1CREAD_DONE", payload)
        return 0


def arm_seed(name: str) -> int:
    m = re.fullmatch(r"(?:N|H1|H2|H1u10k|H1u2k)-s(\d+)", name)
    if not m:
        raise Refuse(f"arm name {name!r} is not a registered arm")
    return int(m.group(1))


def cfg_seed(info: dict):
    return (info.get("config") or {}).get("seed")


# ================================================================================================================ RESULT.md
def f4(x, nd=4):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, float) else str(x))


def fci(ci, nd=4):
    return f"[{ci[0]:+.{nd}f}, {ci[1]:+.{nd}f}]"


def render_result(job: Job) -> str:
    R_, spec, dry = job.R, job.spec, job.dry
    dec, ev, ch = R_["decision"], R_["evidence"], R_["chosen"]
    nav, nat, vj = R_["nav"], R_["nat"], R_["veto_json"]
    blk, a_nat = nav["arms"][spec.name], nat["arms"][spec.name]
    pair = f"{spec.name}-minus-{COMPARATOR}"
    cn = nat["contrasts"][pair]
    L = []
    banner = ("> ⛔ **DRY RUN ON S1's `H1-s0` ARTIFACTS: every row below is PLUMBING. It is NOT an S1c result and must not be quoted as one.** The arm read is S1's 400-update head, "
              "the 'comparator' is itself (a ZERO control), and no S1c number exists.\n\n") if dry else ""
    L.append(f"# RESULT: S1c auto-read. PREREG_S1C row **{dec['label']}**; AUXVETO arm V-S1C row **{vj['decision']['row']}**\n")
    L.append(banner)
    L.append(f"> **Evidence class: MEASURED** (this read, `{TOOL}`, on the artifacts named in `RESULT.json`). Tier: **T0-like** for the K2 / budget reads (head perception on teacher-forced navtest frames, never driving "
             f"performance); the NAVSIM tier (table lookup, base u400) for the veto. Arm `{spec.name}`, repo `{job.c['repo']}`, forward device **{ch['device']}**. Read {utc()}. Nothing committed, nothing pushed.\n")
    L.append("## Headline\n")
    k2 = dec["k2"]
    inv_note = "**INVALID READ: the numbers below are INFORMATIONAL ONLY (PREREG_S1C section 2, C1 / section 4 INVALID row); no S1c verdict exists.** " if dec["row"] == "INVALID" else ""
    L.append(f"- {inv_note}**K2 (bar 0.80):** P1 [0, 40) m = **{blk['p1']['all']['0-40']['est']:.4f}** on the 1,058 C3 tokens (CI {fci(blk['p1']['all']['0-40']['ci95'])}) and **{a_nat['natural']['est']:.4f}** on the natural navtest population "
             f"(ESS {nat['population']['ess']:.1f}; CI {fci(a_nat['natural']['ci95'])}). The C3 read governs K2: **{k2['governing_verdict_C3']}**" + (" (the two populations DISAGREE on K2)" if k2["populations_disagree"] else "") + ".")
    L.append(f"- **Budget contrast vs S1's `H1-s0` (paired, same frames, same log resamples):** Delta_C3 = **{f4(cn['c3']['est'])}** {fci(cn['c3']['ci95'])}, Delta_NAT = **{f4(cn['natural']['est'])}** {fci(cn['natural']['ci95'])}; "
             f"separated (CI excludes 0 and |Delta| > F): C3 **{dec['separated']['C3']}** (F {dec['F']['C3']}), NAT **{dec['separated']['NAT']}** (F {dec['F']['NAT']}).")
    L.append(f"- **PREREG_S1C row: {dec['label']}.** " + (f"Stamp: *{dec['stamp']}*." if dec.get("stamp") else ""))
    vs = vj["arms"][vj["arm"]]["d_pdms"]
    L.append(f"- **AUXVETO V-S1C (stratified navtest-population dPDMS x100): {f4(vs['point'])} {fci([vs['lo'], vs['hi']])}** vs V-GT {f4(vj['arms']['V-GT']['d_pdms']['point'])} "
             f"{fci([vj['arms']['V-GT']['d_pdms']['lo'], vj['arms']['V-GT']['d_pdms']['hi']])}: row **{vj['decision']['row']}**; validity {'all ' + str(sum(1 for v in vj['validity'].values() if isinstance(v, dict) and 'ok' in v)) + ' checks PASS' if vj['validity']['all_pass'] else 'FAILED: ' + str(vj['decision'].get('invalid_checks'))}.")
    L.append("- **STRATEGIC: PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` ITEM 25): not a pass, not omitted, not extended to the other three families.\n")
    L.append("## 1. Validity (PREREG_S1C section 2 and 4 INVALID row)\n")
    c1 = ev["c1"]
    L.append(f"| check | result |\n|---|---|\n| C1 step0 held-out `aux_eval` IoU = S1 H1-s0's 0.1586791569036157 +-1e-3 | {'**holds**' if c1['ok'] else '**FAILS**'}: {c1.get('step0_iou')} (|diff| {c1.get('abs_diff')}) |")
    L.append(f"| the pod's run-check (`run_check.json`) | {'ok' if ev['runcheck_ok'] else '**NOT ok**'} |")
    L.append(f"| the arm's evidence is the registered arm (token profile `{spec.profile}`, argv, start/done/eval tags, lr 0 at the end, data audit, head config) | {'consistent' if ev['evidence_ok'] else '**CONTRADICTED**: ' + '; '.join(ev['problems'])} |")
    L.append(f"| forward: the scorer logits of the same pass are bit-identical to the banked C3 table | {'PASS' if ch['forward']['identity']['PASS'] else '**FAIL**'} ({ch['forward']['identity']['n_tokens_bit_identical']}/{ch['forward']['identity']['n_tokens']} tokens) |")
    ci_ = ch["identity"]
    L.append(f"| INSTRUMENT: the comparator `{COMPARATOR}` read in the same call reproduces S1's banked navtest block ({ci_['harness_block_vs_banked']['n_numeric_leaves']} numeric leaves) and the SHIFT_LOCATE natural read | "
             f"{'PASS' if ci_['ok'] else '**FAIL**'}, max |diff| {ci_['harness_block_vs_banked']['max_abs_diff']:.3g} on {ci_['device']} (tol {ci_['tol']:g}) |")
    if ev["warnings"]:
        L.append("\nWarnings: " + "; ".join(ev["warnings"]))
    op = ev["facts"].get("operational_check")
    if op:
        L.append(f"\nOperational check (PREREG_S1C section 7; NOT outcome-bearing): `s_per_update` mean {ev['facts'].get('s_per_update_mean')} (range {op['s_per_update_range']}: {'in' if op['s_per_update_in_range'] else 'OUT'}); "
                 f"train hours {ev['facts'].get('train_hours')} (range {op['train_hours_range']}: {'in' if op['train_hours_in_range'] else 'OUT'}). {op['note']}.")
    L.append("\n## 2. K2 and the perception reads (navtest, 1,058 C3 tokens; navtest governs)\n")
    L.append("| read | " + spec.name + " | " + (COMPARATOR if spec.name != COMPARATOR else "(comparator = self)") + " |\n|---|---|---|")
    cb = nav["arms"].get(COMPARATOR)

    def row(label, f):
        L.append(f"| {label} | {f(blk)} | {f(cb) if cb is not None else 'n/a'} |")
    for b in ("0-40", "0-20", "20-40", "40-72"):
        row(f"P1 IoU [{b.replace('-', ', ')}) m, all frames", lambda x, b=b: f"{x['p1']['all'][b]['est']:.4f} {fci(x['p1']['all'][b]['ci95'])}")
    row("P1 [0, 40) on the u400-DAC-zero tokens / the others", lambda x: f"{x['p1']['dac0']['0-40']['est']:.4f} / {x['p1']['dac_ok']['0-40']['est']:.4f}")
    nat_c = nat["arms"].get(COMPARATOR)
    L.append(f"| P1 [0, 40) NATURAL navtest population (nat_p1; weights fixed) {fci(a_nat['natural']['ci95'])} | **{a_nat['natural']['est']:.4f}** | " + (f"{nat_c['natural']['est']:.4f} {fci(nat_c['natural']['ci95'])}" if nat_c else "n/a") + " |")
    row("P2 near-boundary SDF MAE (m)", lambda x: f"{x['p2']['near_boundary_mae_m']['est']:.4f}")
    row("P3 pooled DAC AUC (head) vs the scorer's", lambda x: f"{x['p3']['pooled']['head']['est']:.4f} vs {x['p3']['pooled']['scorer']['est']:.4f}")
    fl = blk.get("floor", {})
    if fl.get("status") == "READ":
        fd = fl["trained_minus_init"]["iou_0-40"]
        L.append(f"\n**C4 init floor** (trained minus the seed-0 INITIAL head, same frames): P1 [0, 40) {f4(fd['est'])} {fci(fd['ci95'])}; 95 % lower bound {'> 0' if fd['ci95'][0] > 0 else '<= 0 (AT FLOOR)'}.")
    L.append("\nNatural population: " + f"{nat['population']['definition']}; ESS {nat['population']['ess']:.1f}; margins {json.dumps({k: round(v, 4) for k, v in nat['population']['spec']['share_map'].items()})}, "
             f"failure stratum {nat['population']['spec']['share_stratum']['zero']:.4f}. Per-city natural-weighted P1 ({spec.name}): " +
             ", ".join(f"{m} {v['p1_natural_weighted']:.3f} (n {v['n_frames']}, {v['n_logs']} logs)" for m, v in a_nat["per_city"].items()) + ".")
    L.append("\nHeld-out navtrain P1 [0, 40) (24 logs; informational, navtest governs): " + "; ".join(
        f"{tag}: " + ", ".join(f"{a} {b['p1']['all']['0-40']['est']:.4f} {fci(b['p1']['all']['0-40']['ci95'])}" for a, b in jload(path)['arms'].items()) for tag, path in R_["heldout"].items()) + ".")
    L.append(f"\naux_eval IoU by tag (the trainer's pooled held-out read, all cells): {json.dumps(ev['facts'].get('aux_eval_iou_by_tag'))}; aux_bce first -> last {ev['facts'].get('aux_bce_first_last')}.")
    L.append("\n## 3. The budget contrast (S1c minus S1's `H1-s0`)\n")
    L.append("| population | Delta | paired 95 % CI (primary: weights fixed) | separated? (F) | sensitivity: weights re-fitted per resample |\n|---|---|---|---|---|")
    L.append(f"| C3 (1,058 tokens) | {f4(cn['c3']['est'])} | {fci(cn['c3']['ci95'])} | **{dec['separated']['C3']}** ({dec['F']['C3']}) | n/a |")
    sn = cn["natural"].get("sensitivity_rerake")
    L.append(f"| natural navtest population | {f4(cn['natural']['est'])} | {fci(cn['natural']['ci95'])} | **{dec['separated']['NAT']}** ({dec['F']['NAT']}) | {fci(sn['ci95']) if sn else 'n/a'} "
             f"(class {'unchanged' if dec['sensitivity_natural_contrast_rerake']['same_class_as_primary'] else '**CHANGES**: ' + str(dec['sensitivity_natural_contrast_rerake']['separated_if_used'])}) |")
    L.append("\n*Estimator:* paired log-cluster bootstrap, 10,000 resamples, 95 %, seed 20260927, models held fixed, logs resampled. It answers **\"would another draw of LOGS say this?\" ONLY**: it is blind to training "
             "variance (ONE seed, `H-ESTIM-SEED-1`: every S1c-vs-S1 difference is **separated, single seed, not yet a lever claim**; F is the cheap proxy for 400-update seed noise, the variance at 10,000 updates is unknown) "
             "and to inference variance (the head is deterministic: ASSERTED).")
    L.append("\n## 4. PREREG_S1C section 4, applied mechanically\n")
    L.append("| condition | evaluates to |\n|---|---|")
    for k, v in dec["conditions"].items():
        L.append(f"| {k} | {v} |")
    L.append(f"\n**Row: {dec['label']}.** Remaining gap to the bar `0.80 - P1`: C3 {dec['remaining_gap_to_0.80']['C3']:+.4f}, NAT {dec['remaining_gap_to_0.80']['NAT']:+.4f} (worse population: {dec['remaining_gap_to_0.80']['worse_population']}).")
    if dec["overlap"]:
        L.append(f"\n⚠️ **OVERLAP (QUESTIONS Q4):** O3 and {', '.join(dec['overlap'])} both match (one population adverse, the other not separated). The row is O3 (table order); the other row and its committed lever are quoted below so that nothing is dropped.")
    q = dec["quoted_row"]
    L.append(f"\n**The row, verbatim from `eval/PREREG_S1C.md`:**\n\n> **Rule:** {q['rule']}\n>\n> **Meaning:** {q['meaning']}\n>\n> **NEXT lever, executed in the same run of work:** {q['next']}\n>\n> **What blocks it:** {q['blocks']}")
    for o, qq in (dec.get("overlap_rows_quoted") or {}).items():
        L.append(f"\n**Also matching, `{qq['label']}` (verbatim):**\n\n> **NEXT lever:** {qq['next']}\n>\n> **What blocks it:** {qq['blocks']}")
    L.append(f"\nMM note: Rule Zero applies. The O2 / O3 label-coverage lever is a 0-A40 CPU label build and does not wait for the PI (PREREG_S1C section 9, item 3); the replicate (`H1u10k-s1`), `N-u10k` and the trunk lever are NOT pre-authorised.")
    L.append("\n## 5. AUXVETO arm V-S1C (amendment A1), applied mechanically\n")
    val = vj["validity"]
    L.append("| validity (section 9; any failure => INVALID) | result | detail |\n|---|---|---|")
    for k in ("1_B_reproduces_pick", "0_population_identity", "2_A1_mask_iou_equals_harness_P1", "3_zero_control", "4_odac_bounds_rasters", "5_yflip_mutation_below_vgt"):
        v = val[k]
        det = {k2_: v[k2_] for k2_ in v if k2_ not in ("ok", "arms")}
        L.append(f"| {k} | {'PASS' if v['ok'] else '**FAIL**'} | {json.dumps(det, default=str)[:200]} |")
    ii = vj["instrument_identity"]
    L.append(f"\nInstrument identity (the tool reproduces the banked `AUXVETO_READ.json` for V-GT and O-DAC{' and V-PRED' if dry else ''}): {'PASS' if ii['ok'] else '**FAIL**'}, max |diff| {ii['max_abs_diff']:.3g}.")
    vd = vj["decision"]
    L.append("\n| decision list (section 6, read top-down, V-S1C in V-PRED's role) | evaluates to |\n|---|---|")
    for k, v in vd["conditions"].items():
        L.append(f"| {k} | {v} |")
    L.append(f"\n**Row: {vd['row']}**" + (f" (would be `{vd['row_if_valid']}` if valid)" if vd.get("row_if_valid") else "") + (f". {vd['gt_branch']['reading']}." if vd.get("gt_branch") else ".") +
             (f" Stamp: *{vd['stamp']}*." if vd.get("stamp") else ""))
    qv = vd["quoted_row"]
    if vd["row"] == "INVALID":
        L.append(f"\n**V-S1C is INVALID: this read commits NO next arm.** Find which validity check failed ({vd.get('invalid_checks')}) and fix the cause. The row it would have been, for the record only "
                 f"(`{vd.get('row_if_valid')}`, NOT in force), reads verbatim:\n\n> {qv['next']}")
    else:
        L.append(f"\n**The row's committed next arm, verbatim from `eval/PREREG_AUXVETO.md` section 6:**\n\n> {qv['next']}")
        L.append("\n*Note (QUESTIONS Q9): amendment A1 adds no separate next-arm cell for V-S1C; the cell of the row V-S1C lands in is quoted. An item in it that names \"S1c\" is the arm just read (satisfied by this read); "
                 "the remaining items are the committed next arms.*")
    if vd.get("gt_branch") and vd["gt_branch"].get("AV-NULL_next_arm_quoted"):
        L.append(f"\n> (V-GT not positive => AV-NULL's next arm, verbatim): {vd['gt_branch']['AV-NULL_next_arm_quoted']}")
    ab = vj["arms"]
    lab = vj["arm"]

    def sg(b):
        return f"{f4(b['point'])} {fci([b['lo'], b['hi']])}"
    L.append("\n### Arms (stratified navtest-population delta vs B, x100; Delta = 529/12,146 Delta_Z + 11,617/12,146 Delta_N)\n")
    L.append("| arm | dPDMS | Delta_Z / Delta_N | Z recovered (NC x DAC > 0) of 529 | B's pick falsely vetoed on N | every proposal vetoed (Z / N) |\n|---|---|---|---|---|---|")
    for a in (lab, "V-GT", "O-DAC", "V-GT-YFLIP") + tuple(k for k in ab if k not in (lab, "V-GT", "O-DAC", "V-GT-YFLIP")):
        if a not in ab:
            continue
        b = ab[a]
        L.append(f"| {a} | {sg(b['d_pdms'])} | {b['d_pdms']['z']:+.2f} / {b['d_pdms']['n']:+.2f} | {b['z_recovery']['ncdac_count']} | {b['false_veto_N']['b_pick_vetoed_count']}/{b['false_veto_N']['of']} | "
                 f"{b['all_vetoed_tokens']['Z']} / {b['all_vetoed_tokens']['N']} |")
    L.append("\n### The four families (AUXVETO section 8; per family, never pooled)\n")
    b = ab[lab]
    L.append(f"- **LONGITUDINAL:** dEP {sg(b['d_ep'])}; dTTC {sg(b['d_ttc'])}.")
    L.append(f"- **LATERAL:** dDAC {sg(b['d_dac'])}; dDDC {sg(b['d_ddc'])}; Z-stratum recovery {b['z_recovery']['ncdac_count']}/529 (DAC-only variant {b['z_recovery']['dac_only_count']}); false-veto rate on N {b['false_veto_N']['b_pick_vetoed_count']}/{b['false_veto_N']['of']}.")
    t = vj["tactical_levels"]
    L.append(f"- **TACTICAL (selection quality, PDMS x100, stratified):** random {t['random']['point']:.2f}, B {t['B']['point']:.2f}, best-of-64 {t['best_of_64']['point']:.2f}, {lab} {b['lvl_pdms']['point']:.2f}, V-GT {ab['V-GT']['lvl_pdms']['point']:.2f}, O-DAC {ab['O-DAC']['lvl_pdms']['point']:.2f}.")
    L.append("- **STRATEGIC: PAUSED** (PI 2026-09-19, ITEM 25).")
    L.append(f"- Also: dNC {sg(b['d_nc'])}; dcomfort {sg(b['d_comfort'])}.")
    L.append("\nMargin curve (reported, never gating): " + "; ".join(f"{m} m: dPDMS {sg(v['d_pdms'])}" for m, v in vj["margin_curve"][lab].items()) + ".")
    L.append("\nThe CI answers *another draw of logs*; it is blind to the head's TRAINING variance (one seed, `H-ESTIM-SEED-1`) and to the tracker. V-GT-TRACKED is reported-only in the AUXVETO read (section 7) and is not re-read here.")
    L.append("\n## 6. The four families for S1c itself (PREREG_S1C section 8)\n")
    L.append("- **LONGITUDINAL / LATERAL / TACTICAL: NOT COMPUTED as paired reads.** There is no budget-matched N arm (arm B's scorer trained 25x the updates at batch 8), so a difference is not attributable to the head. "
             "An `N-u10k` control (~4.2 A40-h) would unlock them and is a PI decision. The V-S1C table above is the NAVSIM-tier family read of the HEAD's raster, which needs no N arm (the base is the banked u400 scorer).")
    ds = R_["desc"]
    if ds["status"] == "READ":
        dj = jload(ds["json"])
        L.append(f"- **Arm B's pick, DESCRIPTIVE (levels on the {dj['n_sets']} held-out sets, 24 logs; {dj['stamp']}):**\n")
        L.append("| arm | pick PDMS x100 | DAC | EP | TTC | NC | DDC | zero picks |\n|---|---|---|---|---|---|---|---|")
        for a, s in dj["arms"].items():
            sub = s["subscores_x100"]
            L.append(f"| {a} | {s['PDMS_x100']:.3f} | {sub['DAC']:.3f} | {sub['EP']:.3f} | {sub['TTC']:.3f} | {sub['NC']:.3f} | {sub['DDC']:.3f} | {s['zero_picks']} |")
    else:
        L.append(f"- **Arm B's descriptive read: NOT READ** ({ds['why']}). Recorded loudly; non-gating.")
    L.append("- **STRATEGIC: PAUSED** (PI 2026-09-19, ITEM 25): not a pass, not omitted.")
    L.append("\n## 7. Provenance\n")
    pr = R_["prereg"]
    L.append(f"- PREREG_S1C body sha256 `{pr['s1c']['body_sha256'][:16]}...`; amendments: {pr['s1c']['amendments'] or 'none'}. PREREG_AUXVETO body (pre-A1, CRLF form) `{pr['auxveto']['body_sha256_crlf_form'][:16]}...`, A1 `{pr['auxveto']['a1_sha256'][:16]}...`; amendments: {pr['auxveto']['amendments']}.")
    L.append(f"- Estimator pins: {nav['estimator']}.")
    L.append(f"- Heads read (aux_drv.pt sha256): " + ", ".join(f"{a} `{s[:12]}...`" for a, s in ch["forward"]["head_sha256"].items()) + f". Token: `{ev['facts'].get('token', {}).get('file')}` (commit {str(ev['facts'].get('token', {}).get('commit'))[:12]}).")
    L.append(f"- Fetch: {len(R_['fetch'])} files, {sum(r['size'] for r in R_['fetch'])} B, every one verified (size + LFS sha256 / git blob id). Not fetched: model_sft_B.pt (1.29 GB; listed only), scorer_states_last.pt.")
    L.append("- Files: `RESULT.json`, `nat_p1_*.json`, `perception_navtest_*.json`, `perception_heldout_*.json`, `veto.json`, `desc_selection.json`, `forward_*/` (head_pred npz, identity.json, run.json), `fetch_manifest.json`, `autoread.log`.")
    return "\n".join(L) + "\n"


# ================================================================================================================ single instance + CLI
class SingleInstance:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, json.dumps({"pid": os.getpid(), "started": utc()}).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    holder = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    holder = {}
                if holder.get("pid") and pid_alive(holder["pid"]) and holder["pid"] != os.getpid():
                    raise SystemExit(f"s1c_autoread: another instance is running (pid {holder['pid']}, since {holder.get('started')}): {self.path}")
                try:
                    self.path.unlink()
                except OSError:
                    raise SystemExit(f"s1c_autoread: cannot clear the stale lock {self.path}")

    def __exit__(self, *a):
        try:
            self.path.unlink()
        except OSError:
            pass


def make_cfg(a, out: Path, dry: bool) -> dict:
    return {"repo": a.repo, "out": str(out), "work": a.work if not dry else str(out / "work"), "py_eval": PY_EVAL, "py_t": PY_T, "ctx_cache": CTX_CACHE, "floor_dir": FLOOR_DIR,
            "heldout_logs": HELDOUT_LOGS, "s1_cache": a.s1_cache, "poll_s": a.poll_min * 60, "max_wait_h": a.max_wait_h, "max_errors": a.max_errors, "step_timeout_h": a.step_timeout_h,
            "min_free_gib": a.min_free_gib, "force": a.force, "forward_device": a.forward_device if a.forward_device else ("cpu" if dry else "auto"), "gpu_wait_min": a.gpu_wait_min,
            "extra_heads": ("H1-s1", "H1-s2") if dry else ()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true", help="read S1's banked H1-s0 end to end from the local S1 cache (no HF poll); writes only out/dryrun/")
    ap.add_argument("--no-live", action="store_true", help="dry run: skip the one read-only live listing of the HF repo")
    ap.add_argument("--forward-device", choices=("auto", "cpu", "cuda"), default=None, help="default: auto (GPU under the shared lock, then CPU); the dry run defaults to cpu")
    ap.add_argument("--gpu-wait-min", type=float, default=30.0, help="auto/cuda: how long to wait for the shared GPU lock / an idle GPU before falling back (auto) or failing (cuda)")
    ap.add_argument("--work", default=WORK_DEFAULT)
    ap.add_argument("--s1-cache", default=S1_CACHE)
    ap.add_argument("--keys-file", default=None)
    ap.add_argument("--poll-min", type=float, default=10.0)
    ap.add_argument("--max-wait-h", type=float, default=48.0, help="give up (FAILED) if the arm is not complete on HF in this long")
    ap.add_argument("--max-errors", type=int, default=36)
    ap.add_argument("--step-timeout-h", type=float, default=6.0)
    ap.add_argument("--min-free-gib", type=float, default=6.0)
    ap.add_argument("--force", action="store_true", help="re-run even if S1CREAD_DONE exists")
    a = ap.parse_args(argv)
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass
    if a.dry_run:
        out = HERE / "out" / "dryrun"
        out.mkdir(parents=True, exist_ok=True)
        cfg = make_cfg(a, out, True)
        cfg.update(poll_s=0.05, max_wait_h=0.01)
        token = None if a.no_live else (os.environ.get("HF_TOKEN") or read_token([a.keys_file] if a.keys_file else KEYS_CANDS))
        job = Job(cfg, ScopedLocalRemote(a.s1_cache, [DRY.name]), DRY, dry=True, secrets=[token])
        rc = job.run()
        if rc == 0 and not a.no_live and token:
            try:                                                           # one real, read-only listing: auth + parse; the S1c arm is expected NOT to be there yet
                os.environ["HF_TOKEN"] = token
                lst = S1R.HFRemote(a.repo).listing()
                st = arm_status(lst, REAL)
                arms = sorted({k.split("/")[1] for k in lst if k.startswith("arms/")})
                job.log(f"LIVE read-only listing of {a.repo}: {len(lst)} files; arms present: {arms}; arms/{REAL.name}/ ready={st['ready']} ({st['n_arm_files']} files; missing {len(st['missing'])})")
            except Exception as e:
                job.log(f"LIVE listing failed ({type(e).__name__}: {job.scrub(e)[:160]}): informational in a dry run")
            finally:
                os.environ.pop("HF_TOKEN", None)
        print("ZZS1CAUTOREAD_DRYRUN_" + ("PASS" if rc == 0 else "FAIL"))
        return rc
    token = os.environ.get("HF_TOKEN") or read_token([a.keys_file] if a.keys_file else KEYS_CANDS)
    if not token:
        print("s1c_autoread: no HF token (Keys.txt unreadable or has no hf_ token)")
        return 3
    os.environ["HF_TOKEN"] = token                                         # this process only (S1R.HFRemote reads it); children get a sanitised environment
    out = HERE / "out"
    out.mkdir(parents=True, exist_ok=True)
    with SingleInstance(out / "autoread.lock"):
        return Job(make_cfg(a, out, False), S1R.HFRemote(a.repo), REAL, dry=False, secrets=[token]).run()


if __name__ == "__main__":
    sys.exit(main())
