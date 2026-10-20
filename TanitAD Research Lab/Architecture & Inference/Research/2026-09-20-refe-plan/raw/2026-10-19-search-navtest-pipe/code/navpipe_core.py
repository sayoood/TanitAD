#!/usr/bin/env python3
"""SEARCH-NAVTEST-PIPE core: the pure, torch-free parts of the chunked navtest pipeline for SEARCH-P2-CONFIRM READ C2.

Everything here is a plain function or a small class so that every gate has its own regression arm in test_navpipe.py.  NOTHING in this file computes a navtest label, score or search output;
the harness / GPU / search children are launched only by navpipe_chain.py, and only after G-SEAL.

Gates implemented here (each carries a `# @GUARD:<name>` marker that mutation_check.py disables in turn):
  G-SEAL      check_c2_seal        a sealed C2 amendment (sha bound three ways) AND the C1 LICENCE (MM_RULING_C2_LICENCE.md): C1's RESULT_SPC.json, sha-pinned, outcome O1/O2, every safety point >= -0.25, moved share >= 5 %
  G-DISK      check_disk           D: free (after the chunk's projected bytes) >= 6 GiB, else STOP
  G-IDENT     check_pick_identity  P2's pick in the fresh bundle == P2's BANKED navtest seam row, bit for bit, every token of the chunk
  G-BANK      bank_chunk / delete_ctx   a chunk's ctx is deleted only after its outputs and scores are banked, hashed and re-verified from disk
  G-STATE     State                a monotone, atomic, resumable per-chunk state file
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

# ------------------------------------------------------------------------------------------------------------------------------ constants (pinned by the seal where they matter)
MIN_FREE_GIB = 6.0                                  # the brief's disk guard: stop if D: free < 6 GB
CTX_BYTES_PER_TOKEN = 7680 * 256 * 2                # uint16 bf16 bits, [7680, 256]: 3,932,160 B = 3.75 MiB (measured in spc/raw/cost.json)
SUBS = ("nc", "dac", "ep", "ttc", "c", "ddc")
CSV_COLS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance")   # == run_v1.TERMS[:6]
HARNESS_ARG_BUDGET = 30000                          # Windows' CreateProcess limit is 32,767; W3's --tokens is ONE Hydra override (WinError 206 at 230,812 chars, RESULT_SFT4 Deviation 2)
STATUS_ORDER = ("NEW", "BUNDLED", "GATED", "SEARCHED", "SCORED", "BANKED", "DELETED")
EXIT = {"gate": 3, "timeout": 4, "seal": 5, "search": 6, "score": 7, "analysis": 8, "crash": 9}
# the sealed amendment must pin all of these (the seal check REFUSES on any missing one)
REQUIRED_PINS = ("token_file_sha256", "chunk_plan_sha256", "ckpt_sha256", "config_sha256", "head_s0_sha256", "banked_seam_sha256", "banked_csv_sha256", "c1_prereg_sha256", "licence_ruling_sha256",
                 "inference_cost_status", "bar_dpdms_points", "bar_safety_points", "harness_script_sha256", "seam_module_sha256")
OPERATIVE = ("navpipe_core.py", "navpipe_chain.py", "navpipe_launch.py", "navpipe_analyse.py", "make_navtest_bundle.py", "navpipe_bridge.py", "navpipe_gpu_stage.py")


class Refusal(Exception):
    """a refusal / stop with a class (EXIT key) and a reason; the chain prints it as ZZNAVPIPE_STOP and exits with the class's code"""

    def __init__(self, cls: str, reason: str, detail: str = ""):
        super().__init__(f"{reason} {detail}".strip())
        self.cls, self.reason, self.detail = cls, reason, detail

    @property
    def code(self) -> int:
        return EXIT[self.cls]


# ------------------------------------------------------------------------------------------------------------------------------ hashing
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_lf(path: str) -> str:
    """sha256 of a text file with CRLF folded to LF (a Windows checkout must not make the same source look like different code)"""
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def tokens_sha256(tokens) -> str:
    """== stepB.tokens_sha256 == ctx.json's `tokens_sha256`: sha256 of the tokens joined by newlines IN THE BUNDLE'S ORDER (the bundle's order is sorted)"""
    return hashlib.sha256("\n".join(str(t) for t in tokens).encode()).hexdigest()


def canon_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------ the chunk plan
def plan_chunks(token_doc: dict, cap: int = 1000) -> list:
    """Whole logs, in sorted log-name order, greedily packed up to `cap` tokens per chunk.  A chunk's tokens are SORTED (the bundle builder sorts its sample), so `tokens_sha256` is the data clause of that chunk."""
    toks, tl = token_doc["tokens"], token_doc["token_log"]
    if len(set(toks)) != len(toks):
        raise Refusal("gate", "PLAN_DUPLICATE_TOKENS")
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(tl[t], []).append(t)
    chunks, cur_logs, cur = [], [], []

    def flush():
        if cur_logs:
            ts = sorted(cur)
            chunks.append({"id": len(chunks), "logs": list(cur_logs), "n": len(ts), "tokens_sha256": tokens_sha256(ts)})

    for log in sorted(by_log):
        n = len(by_log[log])
        if n > cap:
            raise Refusal("gate", "PLAN_LOG_EXCEEDS_CAP", f"{log} has {n} tokens > cap {cap}")
        if cur and len(cur) + n > cap:
            flush()
            cur_logs, cur = [], []
        cur_logs.append(log)
        cur.extend(by_log[log])
    flush()
    if sum(c["n"] for c in chunks) != len(toks):
        raise Refusal("gate", "PLAN_COUNT_MISMATCH")
    return chunks


def plan_doc(token_doc: dict, token_file_sha256: str, cap: int = 1000) -> dict:
    ch = plan_chunks(token_doc, cap)
    d = {"schema": "tanitad.navpipe_plan/1", "token_file_sha256": token_file_sha256, "cap": int(cap), "n_tokens": len(token_doc["tokens"]), "n_logs": len({token_doc["token_log"][t] for t in token_doc["tokens"]}),
         "all_tokens_sha256": tokens_sha256(sorted(token_doc["tokens"])), "chunks": ch}
    d["plan_sha256"] = canon_sha256({k: v for k, v in d.items() if k != "plan_sha256"})
    return d


def chunk_tokens(token_doc: dict, chunk: dict) -> list:
    logs = set(chunk["logs"])
    ts = sorted(t for t in token_doc["tokens"] if token_doc["token_log"][t] in logs)
    if len(ts) != chunk["n"] or tokens_sha256(ts) != chunk["tokens_sha256"]:
        raise Refusal("gate", "PLAN_CHUNK_TOKENS_DIFFER", f"chunk {chunk['id']}: recomputed {len(ts)} tokens / {tokens_sha256(ts)[:12]}")
    return ts


def verify_plan(plan: dict, token_doc: dict, token_file_sha256: str) -> None:
    """the plan on disk equals the recomputation from the (pinned) token file: a plan is never trusted, always recomputed"""
    want = plan_doc(token_doc, token_file_sha256, plan["cap"])
    if want != plan:
        raise Refusal("gate", "PLAN_NOT_RECOMPUTED", "the chunk plan differs from its recomputation from the token file")


# ------------------------------------------------------------------------------------------------------------------------------ G-SEAL
def parse_pins(text: str) -> dict:
    if "<!-- PINS-BEGIN -->" not in text or "<!-- PINS-END -->" not in text:
        raise Refusal("seal", "SEAL_NO_PINS")
    blk = text.split("<!-- PINS-BEGIN -->")[1].split("<!-- PINS-END -->")[0]
    pins = {}
    for line in blk.splitlines():
        line = line.strip()
        if not line or line.startswith("`") or line.startswith("#"):
            continue
        k, _, v = line.partition(" ")
        pins[k] = v.strip()
    return pins


REQUIRED_PINS_S0 = ("mode", "token_file_sha256", "plan_json_sha256", "chunk_plan_sha256", "n_tokens", "n_logs", "ckpt_sha256", "config_sha256", "head_s0_sha256", "reference_sha256", "reference_kind", "fan_labels_sha256")


def check_c2_seal(amendment: str | None, amendment_sha256: str | None, c1_result: str | None, c1_result_sha256: str | None, code_dir: str = HERE, extra_dirs: tuple = (), operative: tuple = OPERATIVE) -> dict:
    return _check_seal(amendment, amendment_sha256, c1_result, c1_result_sha256, code_dir, extra_dirs, operative, REQUIRED_PINS, True)


def check_s0_seal(amendment: str | None, amendment_sha256: str | None, code_dir: str = HERE, extra_dirs: tuple = (), operative: tuple = OPERATIVE) -> dict:
    """G-SEAL for the S0-bank mode (MM scope addition 2026-10-19): the SAME three-way sha binding, SEALED status, pins and code pins as the C2 seal, with NO C1 licence (the S0 bank is not a navtest look)
    and the S0 pin set (token list, plan json, chunk plan, ckpt, config, head, the banked P2 reference and its kind).  It refuses without a sealed S0-bank amendment sha."""
    return _check_seal(amendment, amendment_sha256, None, None, code_dir, extra_dirs, operative, REQUIRED_PINS_S0, False)


# ------------------------------------------------------------------------------------------------------------------------------ the C1 licence (MM_RULING_C2_LICENCE.md, sealed 2026-10-20 02:56)
# It REPLACES the EP-ADMISSION record for `H-SEARCH-P2-1`'s arm only: the frozen gate cannot certify a search candidate at any geometry the programme has (that ruling, section 1).  The numbers are the ruling's and may not be loosened:
LICENCE_RULING_SHA256 = "fa7a26046a2fb5f8d8e69b2644c97a6cfce8907201d40d5d3c5be5358560ba6a"       # sha256 of MM_RULING_C2_LICENCE.md; the sealed amendment must pin this exact value
LICENCE_OUTCOMES = ("O1_CONFIRMED", "O2_DIRECTIONAL")                                              # TRADE, O3_NULL, NEGATIVE and O4_INERT do NOT license C2
LICENCE_SAFETY_TERMS = ("nc", "dac", "ttc", "c")                                                   # dNC, dDAC, dTTC, dC (== analyse_stepB.SAFETY_TERMS; a test pins the equality)
LICENCE_SAFETY_MIN = -0.25                                                                         # PDMS points, each safety POINT estimate (O2 too, not only O1)
LICENCE_MOVED_MIN = 0.05                                                                           # the moved share: `inert.frac_moved`, the quantity whose shortfall IS O4_INERT (tts_cem.inert_verdict)
C1_N_TOKENS, C1_N_LOGS = 905, 24                                                                   # B-REST (spc/raw/b_rest.json)
C1_STAGES = ("SEAL_OK", "DATA_OK", "GPU_STAGE_OK", "REACH_PASS", "CID_PASS", "LABEL_DONE")           # spc_chain.run_analysis's own G-ORDER `need` list, read back from the C1 chain's chain_state.json


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_c1_licence(c1_result: str | None, c1_result_sha256: str | None, pins: dict) -> dict:
    """The C1 licence (MM_RULING_C2_LICENCE.md section 2), READ BY CODE before any navtest token is touched.  Refuses (class `seal`, exit 5) unless ALL of:
      1. C1's `RESULT_SPC.json` exists and its sha256 equals the one the operator typed (and, when the amendment was sealed AFTER C1 read and pins `c1_result_sha256`, that pin too),
      2. it IS the C1 read: read == "C1", 905 tokens / 24 logs, arm P2HEAD, and the `chain_state.json` beside it was written under the prereg the amendment pins (`c1_prereg_sha256`) with every analysis precondition done
         (a carve-B read or the SEARCH-DOSE arm, both of which can read O1 on other tokens or another config, are not C1),
      3. its outcome is O1_CONFIRMED or O2_DIRECTIONAL (TRADE, O3_NULL, NEGATIVE and O4_INERT do not license),
      4. each of dNC, dDAC, dTTC, dC POINT estimates is finite and >= -0.25 PDMS points,
      5. the moved share (`inert.frac_moved`) is finite and >= 5 %, and the file does not say INERT.
    Returns the licence record the chain banks beside its state."""
    if not c1_result or not c1_result_sha256:                                               # @GUARD:G-LICENCE-ARGS
        raise Refusal("seal", "LICENCE_MISSING_ARGS", "--c1-result and --c1-result-sha256 are both required")
    if not os.path.isfile(c1_result):                                                       # @GUARD:G-LICENCE-EXISTS
        raise Refusal("seal", "LICENCE_RESULT_MISSING", c1_result)
    got = sha256_file(c1_result)
    if got != str(c1_result_sha256).strip().lower():                                        # @GUARD:G-LICENCE-SHA
        raise Refusal("seal", "LICENCE_RESULT_SHA_MISMATCH", f"file {got[:12]} / typed {str(c1_result_sha256)[:12]}")
    if "c1_result_sha256" in pins and str(pins["c1_result_sha256"]).strip().lower() != got:  # @GUARD:G-LICENCE-PIN
        raise Refusal("seal", "LICENCE_RESULT_PIN_MISMATCH", f"file {got[:12]} / amendment pin {str(pins['c1_result_sha256'])[:12]}")
    try:
        res = json.load(open(c1_result, encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refusal("seal", "LICENCE_RESULT_UNREADABLE", type(e).__name__)
    if not isinstance(res, dict):
        raise Refusal("seal", "LICENCE_RESULT_UNREADABLE", "not a JSON object")
    if (res.get("read"), res.get("n_tokens"), res.get("n_logs")) != ("C1", C1_N_TOKENS, C1_N_LOGS) or not str(res.get("arm", "")).startswith("P2HEAD"):   # @GUARD:G-LICENCE-IS-C1
        raise Refusal("seal", "LICENCE_NOT_C1", f"read={res.get('read')!r} n_tokens={res.get('n_tokens')!r} n_logs={res.get('n_logs')!r} arm={res.get('arm')!r}")
    try:
        cs = json.load(open(os.path.join(os.path.dirname(os.path.abspath(c1_result)), "chain_state.json"), encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refusal("seal", "LICENCE_CHAIN_STATE_MISSING", f"chain_state.json beside the C1 result: {type(e).__name__}")
    if not isinstance(cs, dict) or str(cs.get("prereg_sha256", "")).lower() != str(pins["c1_prereg_sha256"]).strip().lower():     # @GUARD:G-LICENCE-PREREG
        raise Refusal("seal", "LICENCE_PREREG_MISMATCH", f"the C1 chain ran under {str(cs.get('prereg_sha256') if isinstance(cs, dict) else None)[:12]}, the amendment pins {str(pins['c1_prereg_sha256'])[:12]}")
    missing = [s for s in C1_STAGES if s not in (cs.get("done") or [])]
    if missing:                                                                             # @GUARD:G-LICENCE-STAGES
        raise Refusal("seal", "LICENCE_C1_INCOMPLETE", ",".join(missing))
    outcome = res.get("outcome")
    if outcome not in LICENCE_OUTCOMES:                                                     # @GUARD:G-LICENCE-OUTCOME
        raise Refusal("seal", "LICENCE_OUTCOME", f"{outcome!r}: only {' / '.join(LICENCE_OUTCOMES)} license C2")
    sp = res.get("safety_points")
    safety = {k: (sp.get(k) if isinstance(sp, dict) else None) for k in LICENCE_SAFETY_TERMS}
    unreadable = [k for k, v in safety.items() if not _is_num(v)]
    if unreadable:                                                                          # @GUARD:G-LICENCE-SAFETY-READABLE
        raise Refusal("seal", "LICENCE_SAFETY_UNREADABLE", ",".join(unreadable))          # a NaN or null compares False against every bar, so it must be refused before the comparison
    low = {k: v for k, v in safety.items() if v < LICENCE_SAFETY_MIN}
    if low:                                                                                 # @GUARD:G-LICENCE-SAFETY
        raise Refusal("seal", "LICENCE_SAFETY_MISS", json.dumps(low) + f" < {LICENCE_SAFETY_MIN}")
    inert = res.get("inert")
    frac = inert.get("frac_moved") if isinstance(inert, dict) else None
    if not _is_num(frac):                                                                   # @GUARD:G-LICENCE-MOVED-READABLE
        raise Refusal("seal", "LICENCE_MOVED_UNREADABLE", repr(frac))
    if frac < LICENCE_MOVED_MIN or inert.get("inert") is not False:                         # @GUARD:G-LICENCE-MOVED
        raise Refusal("seal", "LICENCE_INERT", f"moved share {frac} (needs >= {LICENCE_MOVED_MIN}); inert flag {inert.get('inert')!r}")
    return {"c1_result_sha256": got, "outcome": outcome, "safety_points": {k: float(v) for k, v in safety.items()}, "moved_share_inert_frac_moved": float(frac), "c1_prereg_sha256": str(pins["c1_prereg_sha256"]),
            "licence_ruling_sha256": LICENCE_RULING_SHA256, "bars": {"safety_min": LICENCE_SAFETY_MIN, "moved_min": LICENCE_MOVED_MIN, "outcomes": list(LICENCE_OUTCOMES)}}


def _check_seal(amendment, amendment_sha256, c1_result, c1_result_sha256, code_dir, extra_dirs, operative, required, need_licence) -> dict:
    """G-SEAL.  Refuses (class `seal`, exit 5) unless ALL of:
      1. an amendment file exists, and its LF sha256 equals BOTH the `--amendment-sha256` the operator typed AND the sha in its sidecar `<name>.sha256` (three-way binding: a file edited after the seal is caught);
      2. its FIRST `Status:` line starts with `Status: **SEALED`;
      3. its PINS block carries every REQUIRED_PINS key, and every `code:<file>` pin equals the LF sha256 of that file (code_dir / extra_dirs), and every OPERATIVE file is pinned;
      4. (navtest mode only) the inference-cost status is stated, the amendment pins the sealed licence ruling, and `check_c1_licence` passes: C1's RESULT_SPC.json licenses the look (this REPLACES the EP-ADMISSION
         record, MM_RULING_C2_LICENCE.md); the S0-bank mode has no licence because it is not a navtest look.
    Returns the pins dict (the chain's proof of seal) and, in navtest mode, the licence record."""
    if not amendment or not amendment_sha256:                                               # @GUARD:G-SEAL-ARGS
        raise Refusal("seal", "SEAL_MISSING_ARGS", "--amendment and --amendment-sha256 are both required" + ("" if need_licence else " (the S0-bank mode has no C1 licence)"))
    if not os.path.isfile(amendment):                                                       # @GUARD:G-SEAL-AMEND-EXISTS
        raise Refusal("seal", "SEAL_AMENDMENT_MISSING", amendment)
    sidecar = os.path.splitext(amendment)[0] + ".sha256"
    if not os.path.isfile(sidecar):
        raise Refusal("seal", "SEAL_SIDECAR_MISSING", sidecar)
    text_b = open(amendment, "rb").read().replace(b"\r\n", b"\n")
    got = hashlib.sha256(text_b).hexdigest()
    side = open(sidecar, encoding="utf-8").read().split()[0].strip().lower()
    if not (got == amendment_sha256.strip().lower() == side):                               # @GUARD:G-SEAL-SHA3
        raise Refusal("seal", "SEAL_SHA_MISMATCH", f"file {got[:12]} / typed {amendment_sha256[:12]} / sidecar {side[:12]}")
    import re
    t = text_b.decode("utf-8")
    m = re.search(r"^Status: .*$", t, re.M)
    if not m or not m.group(0).startswith("Status: **SEALED"):                             # @GUARD:G-SEAL-STATUS
        raise Refusal("seal", "SEAL_NOT_SEALED", f"first status line: {(m.group(0)[:40] if m else None)!r}")
    pins = parse_pins(t)
    miss = [k for k in required if k not in pins]
    if miss:                                                                                # @GUARD:G-SEAL-PINS-PRESENT
        raise Refusal("seal", "SEAL_PIN_MISSING", ",".join(miss))
    if (pins.get("mode") == "s0bank") != (not need_licence):                                # @GUARD:G-SEAL-MODE
        raise Refusal("seal", "SEAL_WRONG_MODE", f"pin mode={pins.get('mode')!r}: an S0-bank amendment cannot unlock navtest, a C2 amendment cannot unlock the S0 bank")
    bad = []
    for f in operative:
        if f"code:{f}" not in pins:                                                         # @GUARD:G-SEAL-OPERATIVE-PINNED
            bad.append(f"operative file not pinned: {f}")
    for k, v in pins.items():
        if k.startswith("code:"):
            p = next((q for q in [os.path.join(d, k[5:]) for d in (code_dir, *extra_dirs)] if os.path.isfile(q)), None)
            if p is None or sha_lf(p) != v:                                                 # @GUARD:G-SEAL-CODE-PINS
                bad.append(k)
    if bad:
        raise Refusal("seal", "SEAL_PIN_MISMATCH", ";".join(bad))
    if not need_licence:
        return {"amendment_sha256": got, "pins": pins, "licence": None}
    if pins["inference_cost_status"] not in ("MEASURED", "UNMEASURED"):                     # @GUARD:G-SEAL-COST-STATED
        raise Refusal("seal", "SEAL_INFERENCE_COST_UNSTATED", pins["inference_cost_status"])  # DQ3: the search's inference cost per token, MEASURED or named UNMEASURED, before the look
    if str(pins["licence_ruling_sha256"]).strip().lower() != LICENCE_RULING_SHA256:         # @GUARD:G-SEAL-LICENCE-RULING
        raise Refusal("seal", "SEAL_LICENCE_RULING_PIN", f"the amendment pins {str(pins['licence_ruling_sha256'])[:12]}, the licence this code enforces is {LICENCE_RULING_SHA256[:12]}")
    licence = check_c1_licence(c1_result, c1_result_sha256, pins)
    return {"amendment_sha256": got, "pins": pins, "licence": licence}


# ------------------------------------------------------------------------------------------------------------------------------ G-DISK
def free_gib(path: str) -> float:
    return shutil.disk_usage(path).free / 2 ** 30


def check_disk(path: str, min_free_gib: float = MIN_FREE_GIB, need_bytes: int = 0, free_fn=None) -> float:
    """STOP unless free space on `path`'s volume, AFTER the bytes the next step will write, stays >= min_free_gib.  Called before every chunk (need = its projected ctx) and between every stage (need = 0)."""
    free = (free_fn or free_gib)(path)
    left = free - need_bytes / 2 ** 30
    if left < min_free_gib:                                                                 # @GUARD:G-DISK
        raise Refusal("gate", "DISK_LOW", f"{free:.2f} GiB free - {need_bytes / 2 ** 30:.2f} GiB needed = {left:.2f} GiB < {min_free_gib} GiB")
    return free


# ------------------------------------------------------------------------------------------------------------------------------ G-IDENT (the conversion is the seam's own `to_navsim`, vendored and sha-pinned)
SEAM_MODULE_SHA256_LF = "ceb2c0f84ea2f7dd45be0c7bb61d98e92ab4e113c53bffaf6235abdd9b8db070"    # eval/refe_navtest_seam.py on the mirror tip fb7ad9a2 (== the D: package copy)
_SEAM = None


def seam_module(path: str | None = None):
    """the seam script itself (`to_navsim`, the 20->8 conversion every banked navtest row went through), imported by path and sha-pinned.  The module imports numpy only; its heavy imports sit inside main()."""
    global _SEAM
    p = path or os.path.join(PKG, "ref", "refe_navtest_seam.py")
    if _SEAM is not None and _SEAM[0] == p:
        return _SEAM[1]
    if sha_lf(p) != SEAM_MODULE_SHA256_LF:                                                  # @GUARD:G-SEAM-PIN
        raise Refusal("gate", "SEAM_MODULE_NOT_PINNED", f"{p}: {sha_lf(p)[:12]} != {SEAM_MODULE_SHA256_LF[:12]}")
    spec = importlib.util.spec_from_file_location("refe_navtest_seam_pinned", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _SEAM = (p, m)
    return m


def repair_last_heading(x):
    """planner.repair_last_heading (SPEC_NAVTEST Amendment 7; planner.py:159): a COPY with the last pose's heading replaced by the previous pose's"""
    out = np.array(x, copy=True)
    out[..., -1, 2] = out[..., -2, 2]
    return out


def exec_to_poses(exec20, seam=None) -> np.ndarray:
    """[n, 20, 3] executed plans (ego frame, 0.2 s) -> [n, 8, 3] float32 on NAVSIM's 0.5 s grid, by the seam's own `to_navsim` and the seam's own cast (`.astype(np.float32)`, refe_navtest_seam.py main loop)"""
    m = seam or seam_module()
    a = np.asarray(exec20)
    return np.stack([m.to_navsim(a[i]).astype(np.float32) for i in range(a.shape[0])]) if a.shape[0] else np.zeros((0, 8, 3), np.float32)


def load_banked_seam(path: str) -> dict:
    """the banked navtest seam of P2 (stage2_p2_full.npz): token -> (poses [8,3] f32, fingerprint).  READ-ONLY."""
    z = np.load(path, allow_pickle=False)
    toks = [str(t) for t in z["token"]]
    if len(set(toks)) != len(toks):
        raise Refusal("gate", "BANKED_SEAM_DUPLICATES")
    poses, fps = z["poses"], z["fingerprint"]                       # an NpzFile RE-READS a member on EVERY `z[key]`: indexing `z["poses"][i]` inside the comprehension made 12,146 full copies, each kept alive by its row view (~14 GB)
    return {"poses": {t: poses[i] for i, t in enumerate(toks)}, "fp": {t: str(fps[i]) for i, t in enumerate(toks)}, "arm": str(z["arm"]) if "arm" in z.files else "?"}


def check_pick_identity(tokens, traj_raw, pick, banked: dict, seam=None) -> dict:
    """G-IDENT: for EVERY token of the chunk, to_navsim(repair(traj_raw[token, pick])) equals P2's banked navtest seam row BYTE FOR BYTE (so -0.0 vs 0.0 or a 1-ulp float noise is a failure).
    Reads no label: both sides are poses P2 already produced.  Returns the record; `ok` is False on any miss, any token the banked seam lacks, or an empty chunk."""
    n = len(tokens)
    raw = np.asarray(traj_raw)
    pk = np.asarray(pick, np.int64)
    ex = repair_last_heading(raw[np.arange(n), pk]) if n else np.zeros((0, 20, 3), np.float32)
    got = exec_to_poses(ex, seam)
    absent = [t for t in tokens if t not in banked["poses"]]
    bad, worst = [], 0.0
    for i, t in enumerate(tokens):
        if t in banked["poses"]:
            want = banked["poses"][t]
            if got[i].tobytes() != np.asarray(want, np.float32).tobytes():                  # @GUARD:G-IDENT-BYTES
                bad.append(t)
                worst = max(worst, float(np.max(np.abs(got[i].astype(np.float64) - np.asarray(want, np.float64)))))
    ok = bool(n > 0 and not bad and not absent)
    return {"ok": ok, "n": n, "n_bad": len(bad), "n_absent_from_banked": len(absent), "max_abs_diff_m": worst, "bad_examples": bad[:5], "absent_examples": absent[:5]}


def fan_dir_sha256(path: str) -> str:
    """sha256 over the sorted `<name> <sha256>` lines of every `*.fan.npz` in a directory (the S0 reference pin: a directory is not a file, so it is pinned by the hash of its members)"""
    import glob
    fs = sorted(glob.glob(os.path.join(path, "*.fan.npz")))
    return hashlib.sha256(chr(10).join(f"{os.path.basename(f)} {sha256_file(f)}" for f in fs).encode()).hexdigest()


def refuse_navtest_tokens(tokens, navtest_tokens) -> None:
    """The S0-bank and fixture modes can NEVER touch a navtest token (the label route would score a navtest label, which is forbidden before the sealed C2 look)."""
    hit = sorted(set(tokens) & set(navtest_tokens))
    if hit:                                                                                 # @GUARD:G-NOT-NAVTEST
        raise Refusal("seal", "REFUSED_NAVTEST_TOKEN", f"{len(hit)} tokens are navtest tokens, e.g. {hit[:2]}")


def join_fan_labels(bundle: dict, fan_labels: dict) -> dict:
    """s0bank: add P2's labelled fan (`sub` [n,64,6], `pdms` [n,64], `valid` [n,64]) to a chunk bundle by TOKEN.  The label route needs the fan's labels (the planted identity reads the pick's), and the EP-ADMISSION bank adapter
    reads them.  They are an INPUT here, pinned by the amendment (`fan_labels_sha256`), never computed: a token without a row is a refusal, a row order that differs from the bundle's is re-aligned by token."""
    toks = [str(t) for t in bundle["token"]]
    ix = {str(t): i for i, t in enumerate(fan_labels["token"])}
    miss = [t for t in toks if t not in ix]
    if miss:                                                                                # @GUARD:G-FANLABELS-COVER
        raise Refusal("gate", "FAN_LABELS_MISSING_TOKENS", f"{len(miss)} of {len(toks)} bundle tokens have no fan-label row, e.g. {miss[:2]}")
    rows = [ix[t] for t in toks]
    out = dict(bundle)
    for k in ("sub", "pdms", "valid"):
        if k not in fan_labels:
            raise Refusal("gate", "FAN_LABELS_MISSING_KEY", k)
        out[k] = np.asarray(fan_labels[k])[rows]
    if out["pdms"].shape[:2] != (len(toks), 64) or out["sub"].shape[:2] != (len(toks), 64):
        raise Refusal("gate", "FAN_LABELS_BAD_SHAPE", f"sub {out['sub'].shape} pdms {out['pdms'].shape}")
    return out


# ------------------------------------------------------------------------------------------------------------------------------ banked P2 references (token-list generic: the identity reference depends on the mode)
class SeamRef:
    """navtest: P2's banked seam (8 executed-pick poses per token, + the export's fingerprint).  Level of identity: the pick's 8 poses."""
    kind = "seam"

    def __init__(self, path: str):
        self.b = load_banked_seam(path)

    @classmethod
    def from_dict(cls, banked: dict):
        o = cls.__new__(cls)
        o.b = banked
        return o

    def has(self, t):
        return t in self.b["poses"]

    def fingerprints(self):
        return self.b["fp"]

    def equal(self, t, raw64, logits64, pick, seam=None) -> bool:
        got = exec_to_poses(repair_last_heading(np.asarray(raw64, np.float32)[int(pick)])[None], seam)[0]
        return got.tobytes() == np.asarray(self.b["poses"][t], np.float32).tobytes()


class TableRef:
    """navtest (optional, stronger): P2's banked 64-proposal TABLE (`p2_full_table/table_p2_full.npz`): all 64 converted proposals [64,8,3], the six logits per proposal [64,6] and the pick.  Reads token / proposals / logits / pick
    ONLY (never `pdms`, `sub`, `valid`: those are navtest labels).
    MEASURED 2026-10-20 (243 of 12,146 rows): the PICK's row equals the banked seam's repaired pick byte for byte.  UNVERIFIED: whether the 63 other rows were converted with or without the last-heading repair, so the
    comparison tries the three possible conventions on the first token it sees, LOCKS the one that matches, and requires that same convention for every later token (a mixed table is not an identity)."""
    kind = "table"
    VARIANTS = ("repaired", "raw", "pick_repaired_others_raw")

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        toks = [str(t) for t in z["token"]]
        self.ix = {t: i for i, t in enumerate(toks)}
        if len(self.ix) != len(toks):
            raise Refusal("gate", "TABLE_DUPLICATES")
        self.proposals, self.logits, self.pick = z["proposals"], z["logits"], z["pick"]
        self.variant = None

    def has(self, t):
        return t in self.ix

    def _convert(self, variant, r, k, seam):
        if variant == "repaired":
            return exec_to_poses(repair_last_heading(r), seam)
        if variant == "raw":
            return exec_to_poses(r, seam)
        out = exec_to_poses(r, seam)
        out[int(k)] = exec_to_poses(repair_last_heading(r[int(k)])[None], seam)[0]
        return out

    def equal(self, t, raw64, logits64, pick, seam=None) -> bool:
        i = self.ix[t]
        if int(pick) != int(self.pick[i]) or np.asarray(logits64, np.float32).tobytes() != np.asarray(self.logits[i], np.float32).tobytes():
            return False
        r = np.asarray(raw64, np.float32)
        want = np.asarray(self.proposals[i], np.float32).tobytes()
        for v in ([self.variant] if self.variant else self.VARIANTS):
            if self._convert(v, r, pick, seam).tobytes() == want:
                self.variant = v
                return True
        return False


class S0FanRef:
    """s0bank: P2's OWN fan files of the S0 bank (`<unit>.fan.npz`: token, logits [n,64,6], pick, fingerprint; the proposals are not stored).  Level of identity: the six logits of all 64 proposals and the pick, bytes."""
    kind = "s0fan"

    def __init__(self, path: str):
        import glob
        files = sorted(glob.glob(os.path.join(path, "*.fan.npz"))) if os.path.isdir(path) else [path]
        if not files:
            raise Refusal("gate", "S0FAN_NO_FILES", path)
        self.logits, self.pick, self.fp = {}, {}, {}
        for f in files:
            z = np.load(f, allow_pickle=False)
            lg, pk, fpr = z["logits"], z["pick"], z["fingerprint"]    # read each member ONCE (an NpzFile re-reads it on every `z[key]`)
            for i, t in enumerate(z["token"]):
                t = str(t)
                if t in self.logits:
                    raise Refusal("gate", "S0FAN_DUPLICATE_TOKEN", t)
                self.logits[t], self.pick[t], self.fp[t] = lg[i], int(pk[i]), str(fpr[i])

    def has(self, t):
        return t in self.logits

    def fingerprints(self):
        return self.fp

    def equal(self, t, raw64, logits64, pick, seam=None) -> bool:
        return bool(np.asarray(logits64, np.float32).tobytes() == np.asarray(self.logits[t], np.float32).tobytes() and int(pick) == self.pick[t])


def make_reference(kind: str, path: str):
    return {"seam": SeamRef, "table": TableRef, "s0fan": S0FanRef}[kind](path)


def check_reference_identity(ref, tokens, traj_raw, logits, pick, seam=None) -> dict:
    """G-IDENT, reference-generic: every token of the chunk must be byte-equal to P2's banked reference (kind-specific level, see the classes).  Absent tokens, an empty chunk, or any one difference is not ok."""
    n = len(tokens)
    absent = [t for t in tokens if not ref.has(t)]
    bad = [t for i, t in enumerate(tokens) if ref.has(t) and not ref.equal(t, traj_raw[i], logits[i], pick[i], seam)]      # @GUARD:G-IDENT-REF
    return {"ok": bool(n > 0 and not bad and not absent), "kind": ref.kind, "variant": getattr(ref, "variant", None), "n": n, "n_bad": len(bad), "n_absent_from_banked": len(absent), "bad_examples": bad[:5], "absent_examples": absent[:5]}


# ------------------------------------------------------------------------------------------------------------------------------ G-STATE
class State:
    """per-chunk state, a JSON file written atomically (tmp + fsync + replace).  Status is MONOTONE through STATUS_ORDER: a skip or a regress is refused (`reset` is explicit and logged)."""

    def __init__(self, path: str, plan_sha256: str | None = None):
        self.path = path
        if os.path.isfile(path):
            self.d = json.load(open(path, encoding="utf-8"))
            if plan_sha256 is not None and self.d.get("plan_sha256") != plan_sha256:        # @GUARD:G-STATE-PLAN
                raise Refusal("gate", "STATE_PLAN_DIFFERS", "the state file belongs to another chunk plan")
        else:
            self.d = {"schema": "tanitad.navpipe_state/1", "plan_sha256": plan_sha256, "chunks": {}, "events": []}

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + f".tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.d, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def chunk(self, cid: int) -> dict:
        return self.d["chunks"].setdefault(str(cid), {"status": "NEW", "artifacts": {}, "seconds": {}, "records": {}})

    def status(self, cid: int) -> str:
        return self.chunk(cid)["status"]

    def advance(self, cid: int, new: str, **rec):
        c = self.chunk(cid)
        i, j = STATUS_ORDER.index(c["status"]), STATUS_ORDER.index(new)
        if j != i + 1:                                                                      # @GUARD:G-STATE-MONOTONE
            raise Refusal("gate", "STATE_BAD_TRANSITION", f"chunk {cid}: {c['status']} -> {new}")
        c["status"] = new
        c["records"].update(rec)
        self.d["events"].append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "chunk": cid, "to": new})
        self.save()

    def artifact(self, cid: int, name: str, path: str, sha256: str | None = None):
        self.chunk(cid)["artifacts"][name] = {"path": path, "sha256": sha256 or sha256_file(path), "bytes": os.path.getsize(path)}
        self.save()

    def reset(self, cid: int, to: str, why: str):
        self.chunk(cid)["status"] = to
        self.d["events"].append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "chunk": cid, "reset_to": to, "why": why})
        self.save()


# ------------------------------------------------------------------------------------------------------------------------------ G-BANK
BANK_REQUIRED = ("bundle", "search", "scores", "identity")


def bank_chunk(state: State, cid: int, files: dict, ctx_json_path: str, manifest_path: str) -> dict:
    """hash every output of the chunk, write the manifest, record manifest + files in the state.  Needs ALL of BANK_REQUIRED; a missing or empty file is a refusal."""
    miss = [k for k in BANK_REQUIRED if k not in files]
    if miss:                                                                                # @GUARD:G-BANK-REQUIRED
        raise Refusal("gate", "BANK_INCOMPLETE", f"chunk {cid}: missing {miss}")
    man = {"schema": "tanitad.navpipe_bank/1", "chunk": cid, "files": {}}
    for k, p in sorted(files.items()):
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            raise Refusal("gate", "BANK_FILE_MISSING", f"{k}: {p}")
        man["files"][k] = {"path": os.path.abspath(p), "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    cj = json.load(open(ctx_json_path, encoding="utf-8"))
    man["ctx"] = {"ctx_sha256": cj["ctx_sha256"], "n": cj["n"], "tokens_sha256": cj["tokens_sha256"]}
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    for k, rec in man["files"].items():
        state.artifact(cid, k, rec["path"], rec["sha256"])
    state.artifact(cid, "manifest", manifest_path)
    return man


def verify_banked(state: State, cid: int) -> list:
    """re-hash every banked artifact FROM DISK and compare with the state (G-BANK-VERIFY); returns the list of problems (empty = all verified)"""
    arts = state.chunk(cid)["artifacts"]
    probs = [f"{k} not recorded" for k in (*BANK_REQUIRED, "manifest") if k not in arts]
    for k, rec in arts.items():
        if not os.path.isfile(rec["path"]):
            probs.append(f"{k}: file gone")
        elif sha256_file(rec["path"]) != rec["sha256"]:
            probs.append(f"{k}: bytes differ from the recorded sha256")
    return probs


def delete_ctx(state: State, cid: int, ctx_path: str) -> None:
    """THE ONLY DELETE: chunk status must be BANKED, every required artifact recorded AND re-verified from disk, the chunk's identity and score records must say ok, and only then is `ctx.npy` removed"""
    if state.status(cid) != "BANKED":                                                       # @GUARD:G-DELETE-STATUS
        raise Refusal("gate", "DELETE_REFUSED_NOT_BANKED", f"chunk {cid} is {state.status(cid)}")
    probs = verify_banked(state, cid)
    if probs:                                                                               # @GUARD:G-DELETE-VERIFY
        raise Refusal("gate", "DELETE_REFUSED_UNVERIFIED", "; ".join(probs))
    rec = state.chunk(cid)["records"]
    if not (rec.get("identity_ok") is True and rec.get("t1id_ok") is True):                 # @GUARD:G-DELETE-RECORDS
        raise Refusal("gate", "DELETE_REFUSED_RECORDS", f"identity_ok={rec.get('identity_ok')} t1id_ok={rec.get('t1id_ok')}")
    if os.path.isfile(ctx_path):
        os.remove(ctx_path)
    if os.path.exists(ctx_path):
        raise Refusal("gate", "DELETE_FAILED", ctx_path)
    state.advance(cid, "DELETED", ctx_deleted=True)


# ------------------------------------------------------------------------------------------------------------------------------ the T1 route (seam -> W3 harness -> CSV)
SEAM_SAMPLING = np.array([8, 0.5])


def write_seam(path: str, tokens, poses, fingerprints: dict, arm: str) -> str:
    """a W3 seam npz (SeamAgentV1's contract: token / fingerprint / poses [n,8,3] float32 / sampling [8, 0.5] / arm).  Fingerprints are the banked seam's (the W3 export's), looked up by token."""
    poses = np.asarray(poses, np.float32)
    if poses.shape != (len(tokens), 8, 3) or not np.isfinite(poses).all():
        raise Refusal("score", "SEAM_BAD_POSES", str(poses.shape))
    miss = [t for t in tokens if t not in fingerprints]
    if miss:
        raise Refusal("score", "SEAM_NO_FINGERPRINT", miss[0])
    np.savez(path, token=np.array(list(tokens)), fingerprint=np.array([fingerprints[t] for t in tokens]), poses=poses, sampling=SEAM_SAMPLING, arm=np.array(arm))
    return path


def estimate_override_chars(tokens, logs) -> int:
    """the two Hydra overrides run_v1.split_overrides builds, as one string (quotes + commas included)"""
    return len("train_test_split.scene_filter.log_names=[" + ",".join(f"'{x}'" for x in logs) + "]") + len("train_test_split.scene_filter.tokens=[" + ",".join(f"'{x}'" for x in tokens) + "]")


def write_tokens_file(path: str, tokens, token_log: dict) -> str:
    """W3-format subset file {"tokens": [...], "token_log": {...}}; REFUSES one whose override would exceed the Windows command-line budget (WinError 206)"""
    logs = sorted({token_log[t] for t in tokens})
    n = estimate_override_chars(tokens, logs)
    if n > HARNESS_ARG_BUDGET:                                                              # @GUARD:G-ARGLEN
        raise Refusal("score", "HARNESS_ARG_TOO_LONG", f"{n} chars > {HARNESS_ARG_BUDGET}: shrink the chunk cap")
    json.dump({"tokens": list(tokens), "token_log": {t: token_log[t] for t in tokens}}, open(path, "w", encoding="utf-8", newline="\n"))
    return path


def read_harness_csv(path: str, tokens) -> tuple:
    """-> (sub [n,6] in SUBS order, pdms [n]) for `tokens`, from a harness per-token CSV.  REFUSES a missing token, an invalid row, a non-finite value, or a PDMS that is not NC*DAC*(5EP+5TTC+2C)/12 (<= 1e-12)"""
    rows = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["token"] != "average":
            rows[r["token"]] = r
    miss = [t for t in tokens if t not in rows]
    if miss:
        raise Refusal("score", "CSV_TOKEN_MISSING", f"{len(miss)} missing, e.g. {miss[0]}")
    sub = np.zeros((len(tokens), 6))
    pdms = np.zeros(len(tokens))
    for i, t in enumerate(tokens):
        r = rows[t]
        if r["valid"] not in ("True", "true", "1"):
            raise Refusal("score", "CSV_ROW_INVALID", t)
        sub[i] = [float(r[c]) for c in CSV_COLS]
        pdms[i] = float(r["score"])
    if not (np.isfinite(sub).all() and np.isfinite(pdms).all()):
        raise Refusal("score", "CSV_NON_FINITE")
    f = sub[:, 0] * sub[:, 1] * (5 * sub[:, 2] + 5 * sub[:, 3] + 2 * sub[:, 4]) / 12.0
    if float(np.max(np.abs(f - pdms))) > 1e-12:                                             # @GUARD:G-CSV-FORMULA
        raise Refusal("score", "CSV_FORMULA_MISMATCH", f"max |formula - score| = {float(np.max(np.abs(f - pdms))):.3e}")
    return sub, pdms


def t1_planted_identity(tokens, pick_poses, banked_csv_rows: tuple, run_scores, tol: float = 1e-9, shift_m: float = 0.5) -> dict:
    """The label-path check of the T1 route, per chunk (the analogue of label_stepB's planted identity), on the K tokens given:
      * IDENTITY  the pick's own seam, put through the SAME harness route as every search output, must reproduce P2's BANKED per-token row (six sub-scores and PDMS) to <= tol;
      * CONTROL   the same plan shifted `shift_m` ahead must change at least one token's PDMS (so the identity is not a probe that cannot read a difference).
    `run_scores(poses [K,8,3]) -> (sub [K,6], pdms [K])` is the injected route (the real one writes a seam and runs the harness); `banked_csv_rows` = (sub, pdms) of the SAME K tokens from the banked CSV."""
    bsub, bpdms = banked_csv_rows
    s0, p0 = run_scores(np.asarray(pick_poses, np.float32))
    d = max(float(np.max(np.abs(np.asarray(s0) - np.asarray(bsub)))), float(np.max(np.abs(np.asarray(p0) - np.asarray(bpdms))))) if len(tokens) else float("nan")
    sh = np.array(pick_poses, np.float32, copy=True)
    sh[:, :, 0] += np.float32(shift_m)
    _, p1 = run_scores(sh)
    changed = int(np.sum(np.abs(np.asarray(p1) - np.asarray(p0)) > 1e-9))
    ident_ok = bool(len(tokens) > 0 and np.isfinite(d) and d <= tol)                        # @GUARD:G-T1ID-IDENTITY
    ctl_ok = bool(changed > 0)                                                              # @GUARD:G-T1ID-CONTROL
    return {"ok": bool(ident_ok and ctl_ok), "identity_ok": ident_ok, "control_ok": ctl_ok, "n_tokens": len(tokens), "max_abs_diff": d, "tol": tol, "control_tokens_changed": changed}


# ------------------------------------------------------------------------------------------------------------------------------ canonical-order merging (the analysis is order-sensitive in the last bits: always sorted-token order)
def canonical_merge(parts: list, token_key: str = "token", skip=("meta", "seeds", "sub_names", "identity", "search_sha256", "bundle_sha256")) -> dict:
    """concatenate per-chunk dicts of arrays along axis 0 and REORDER to the global sorted-token order (== the order of the single bundle an unchunked run would have used).
    Keys in `skip` are carried from the first part, not concatenated.  Duplicate tokens across parts are a refusal."""
    keys = [k for k in parts[0] if k not in skip]
    cat = {k: np.concatenate([np.asarray(p[k]) for p in parts], 0) for k in keys}
    toks = [str(t) for t in cat[token_key]]
    if len(set(toks)) != len(toks):
        raise Refusal("analysis", "MERGE_DUPLICATE_TOKENS")
    order = np.argsort(np.array(toks))
    out = {k: v[order] for k, v in cat.items()}
    for k in skip:
        if k in parts[0]:
            out[k] = parts[0][k]
    return out
