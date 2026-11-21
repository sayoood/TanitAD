"""PICKRULE-AUDIT-NC library: the planner's navsim_v1 pick rule (planner.py:558-561), the NC veto, the label-lookup PDMS, Q1 / Q2 / Q3, the registered decision functions, the controls.

REGISTERED SOURCE: PREREG_PICKRULE_AUDIT_NC.md (sha256 PREREG_SHA256 below).  Every constant cites its section.  No stage of this package computes a quantity unless the prereg file still hashes
to PREREG_SHA256 (`require_prereg`).  CPU only, numpy only except the pinned landed modules.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "pickrule-audit-nc-v1"
PREREG_SHA256 = "28947063086150582a7f0d94df26b815e790200832e25ad39ff24f255a953ea7"
AMENDMENT1_SHA256 = "28e349b8ffb7f5625cdb95544e3d6bece4e9aaddf27cd2e4b88600d2c3696522"     # PREREG_PICKRULE_AUDIT_NC_AMENDMENT1.md (sealed 09:57:42, before any real-data stage)
AMENDMENT2_SHA256 = "9dda8c58afd9d02aaca8887822e9a5c12214507497a57c6ea0ecd355e5f4c79d"                                                    # PREREG_PICKRULE_AUDIT_NC_AMENDMENT2.md (C-ID3 reach clause; sealed after the first gate stop, before any Q1/Q2/Q3)
EPR1_SHA256 = "a751c63eb8e38343a7ec1286114a158046ce4ec13390b2eea0a4b1d2237c38d8"             # prereg s2
FASTNEG_LIB_SHA256 = "0c0b399b37a47ee3e17f9d5484bc7f9521ff7dbe6bea37e8c7a38042668667ca"      # prereg s2
PLANNER_SHA256 = "91437cd09362ff872a861c6166b3c00eee98ce9b98c4e85b6a95ca133e29e211"          # prereg s4 (tip 993a78cf)
PLANNER_RULE_LINES = (558, 561)                                                                # prereg s4
PLANNER_RULE_LINES_SHA256 = "bcfaf349a155d881a3b35589bde3b1106f87d5e405d715be8e165625953c3574"
PLANNER_V1W_LINE = 520
V1_W = (5.0, 5.0, 2.0)                           # planner.py:520, NAVSIM v1 progress / TTC / comfort weights (sum 12); driving direction at weight 0
TAUS = (0.3, 0.5, 0.7, 0.9)                      # prereg s4 / s6 Q3
SIG_THR = 0.5                                    # prereg s6 Q1: "its OWN NC sigmoid is < 0.5"
BOOT_B, BOOT_SEED = 10000, 20260927              # prereg s2
N_LOGS = 117
SEEDS = (0, 1)                                   # prereg s2
SOURCES = ("OOF", "BANKED")                      # prereg s3.2: OOF is PRIMARY, BANKED is SECONDARY (no authority)
PRIMARY = "OOF"
CSTAR = {"id": "lr0.001_anchor0.1", "lr": 1e-3, "anchor": 0.1, "epochs": 10, "batch_sets": 256, "config_sha256": "ef1b12ee4d94d1e2ec118df4cb3a996dc575df4157d7368cd6226fb3540517d8"}
NULL_REPS, NULL_SALT, NULL_SEED_BASE = 5, 7919, 20261019      # prereg s8 C-NULL
NULL_VOID_FIRES = 2
MIN_AVAIL_GB = 3.0
ID2_TOL = 1e-9                                   # prereg s8 C-ID2
NC, DAC, EP, TTC, CMF, DDC = 0, 1, 2, 3, 4, 5


class PickruleError(RuntimeError):
    """An integrity check, a pin or a precondition failed."""


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# hashing, pins, pinned module loading
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def require_prereg(path: str | None = None) -> str:
    """Refuse unless the prereg file AND its amendment 1 still hash to their sealed values (a change comes only by a written amendment BEFORE the number it governs).  -> the prereg sha256."""
    p = path or os.path.join(HERE, "..", "PREREG_PICKRULE_AUDIT_NC.md")
    am = os.path.join(os.path.dirname(os.path.abspath(p)), "PREREG_PICKRULE_AUDIT_NC_AMENDMENT1.md")
    am2 = os.path.join(os.path.dirname(os.path.abspath(p)), "PREREG_PICKRULE_AUDIT_NC_AMENDMENT2.md")
    for f, want in ((p, PREREG_SHA256), (am, AMENDMENT1_SHA256), (am2, AMENDMENT2_SHA256)):
        if not os.path.exists(f):
            raise PickruleError(f"the sealed registration file {f} does not exist")
        sha = sha256_file(f)
        if sha != want:
            raise PickruleError(f"{f} sha256 {sha} != the sealed {want}: refuse (write an amendment BEFORE the number it governs)")
    return PREREG_SHA256


def load_pinned(name: str, candidates: list, want_sha: str):
    """Import a landed module by path after checking its sha256 against the pin; a tampered or missing copy is refused."""
    if name in sys.modules and getattr(sys.modules[name], "_PICKRULE_PINNED", None) == want_sha:
        return sys.modules[name]
    tried = []
    for c in candidates:
        if c and os.path.exists(c):
            sha = sha256_file(c)
            if sha != want_sha:
                raise PickruleError(f"{name} at {c} has sha256 {sha} != the pinned {want_sha}")
            if os.path.dirname(os.path.abspath(c)) not in sys.path:
                sys.path.insert(0, os.path.dirname(os.path.abspath(c)))
            spec = importlib.util.spec_from_file_location(name, c)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            mod._PICKRULE_PINNED = want_sha
            return mod
        tried.append(c)
    raise PickruleError(f"{name} not found; probed: {tried}")


REF = "C:/Users/Admin/par18/pickrule/ref"
_REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw"


def epr1_candidates(explicit: str | None = None) -> list:
    return [explicit, os.path.join(HERE, "..", "..", "2026-10-18-ep-r1", "code", "epr1.py"), f"{REF}/epr1/epr1.py"]


def fastneg_candidates(explicit: str | None = None) -> list:
    return [explicit, os.path.join(HERE, "..", "..", "2026-10-19-fastneg-census", "code", "fastneg_lib.py"),
            f"{REF}/{_REPO}/2026-10-19-fastneg-census/code/fastneg_lib.py"]


def load_epr1(explicit: str | None = None):
    return load_pinned("epr1", epr1_candidates(explicit), EPR1_SHA256)


def load_fastneg(explicit: str | None = None):
    return load_pinned("fastneg_lib", fastneg_candidates(explicit), FASTNEG_LIB_SHA256)


def check_planner_pin(path: str | None) -> dict:
    """C-ID1 source pin: lines 558-561 of the given planner.py hash to the registered value, and line 520 carries V1_W = (5.0, 5.0, 2.0).  A missing copy is NOT RUN, never PASS."""
    if not path or not os.path.exists(path):
        return {"status": "NOT_RUN", "reason": f"planner.py copy not found: {path}"}
    raw = open(path, "rb").read()
    lines = raw.split(b"\n")
    a, b = PLANNER_RULE_LINES
    chunk = b"\n".join(lines[a - 1:b]) + b"\n"
    sha = sha256_bytes(chunk)
    v1w = lines[PLANNER_V1W_LINE - 1].decode("utf-8", "replace")
    ok_rule = sha == PLANNER_RULE_LINES_SHA256
    ok_w = "V1_W = (5.0, 5.0, 2.0)" in v1w
    ok_file = sha256_bytes(raw) == PLANNER_SHA256
    return {"status": "PASS" if (ok_rule and ok_w) else "FAIL", "lines_sha256": sha, "lines_sha256_expected": PLANNER_RULE_LINES_SHA256, "v1w_line": v1w.strip()[:120],
            "file_sha256_equals_pin": ok_file, "rule_lines_text": chunk.decode("utf-8", "replace")}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the pick rule (planner.py:558-561), the veto, the PDMS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def sigmoid64(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def rule_aggregate(z6: np.ndarray, dtype=np.float64) -> np.ndarray:
    """planner.py:558-561 `navsim_v1`:  p[..., 0] * p[..., 1] * (p[..., 2] * w1[0] + p[..., 3] * w1[1] + p[..., 4] * w1[2]) / w1.sum()   with p = sigmoid(score), w1 = V1_W.
    float64 is the dtype of the bank's own pick (`epr1.aggregate_np`) and the one every control and outcome uses; float32 (torch sigmoid, the planner's runtime dtype) is a DIAGNOSTIC."""
    if dtype == np.float64:
        p = sigmoid64(z6)
        w1 = np.array(V1_W, np.float64)
    else:
        import torch
        p = torch.sigmoid(torch.as_tensor(np.asarray(z6, np.float32))).numpy()
        w1 = np.array(V1_W, np.float32)
    return p[..., 0] * p[..., 1] * (p[..., 2] * w1[0] + p[..., 3] * w1[1] + p[..., 4] * w1[2]) / w1.sum()


def first_argmax(a: np.ndarray) -> np.ndarray:
    """[N, 64] -> [N]; ties go to the LOWEST index (np.argmax; equal to the stable descending argsort's first element)."""
    return np.argmax(a, axis=1)


def rule_pick(z6: np.ndarray, dtype=np.float64) -> np.ndarray:
    return first_argmax(rule_aggregate(z6, dtype))


def veto_pick(z6: np.ndarray, tau: float) -> tuple:
    """prereg s4: exclude proposals with sigmoid(NC logit) < tau, first argmax of the SAME aggregate over the rest; a set with no eligible proposal falls back to the UNVETOED pick.
    -> (pick [N], fallback [N] bool).  Rows are unchanged."""
    agg = rule_aggregate(z6)
    elig = sigmoid64(z6[..., NC]) >= tau
    has = elig.any(axis=1)
    masked = np.where(elig, agg, -np.inf)
    pick = np.where(has, np.argmax(masked, axis=1), np.argmax(agg, axis=1))
    return pick.astype(np.int64), ~has


def pdms_from_labels(lb: np.ndarray) -> np.ndarray:
    """prereg s5: NC * DAC * (5 EP + 5 TTC + 2 C) / 12 of the labels6 entry, weights V1_W (planner.py:520), float32 like `epr1.pdms_np` (a control compares them bit for bit)."""
    w = V1_W
    return (lb[..., NC] * lb[..., DAC] * (w[0] * lb[..., EP] + w[1] * lb[..., TTC] + w[2] * lb[..., CMF]) / float(sum(w))).astype(np.float32)


def at(a: np.ndarray, pick: np.ndarray) -> np.ndarray:
    """a [N, 64, ...] at the pick of each set."""
    return a[np.arange(len(pick)), pick]


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the registered decision functions (pure; `branch_table_failures` is the table every mutation must break)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def share_class(ci: dict) -> str:
    """prereg s7, one seed / one source: Q1 lower CI > 0.5 -> COMPOSITION; upper CI < 0.5 -> ROW-MISS; else MIXED (NaN bounds -> MIXED)."""
    if ci["lo"] > 0.5:
        return "COMPOSITION"
    if ci["hi"] < 0.5:
        return "ROW-MISS"
    return "MIXED"


def outcome_class(classes: list) -> str:
    """prereg s7: COMPOSITION iff BOTH seeds read COMPOSITION; ROW-MISS iff BOTH read ROW-MISS; otherwise MIXED."""
    if all(c == "COMPOSITION" for c in classes):
        return "COMPOSITION"
    if all(c == "ROW-MISS" for c in classes):
        return "ROW-MISS"
    return "MIXED"


def tau_recovers(rate_points: list, p2_hi: float, dpdms_los: list) -> bool:
    """prereg s7: tau recovers iff, on BOTH seeds, (i) the vetoed composite's NC-unsafe pick rate <= P2's 97.5th percentile AND (ii) the lower CI of dPDMS x100 (vetoed composite - unvetoed P2) >= 0."""
    return all((r <= p2_hi) and (lo >= 0.0) for r, lo in zip(rate_points, dpdms_los))


def veto_recovers(flags_by_tau: dict) -> bool:
    return any(bool(v) for v in flags_by_tau.values())


NEXT_ARM = {
    ("COMPOSITION", True): "a 0-train pick-rule registration: a hard NC veto at a tau named by a rule fixed IN that registration (not by this arm), for the dev gate",
    ("COMPOSITION", False): "a soft gating rule (multiplicative NC power, p_NC^k in the product), registered",
    ("ROW-MISS", True): "NC calibration on the fast band (temperature or isotonic, out-of-fold), registered",
    ("ROW-MISS", False): "NC calibration on the fast band (temperature or isotonic, out-of-fold), registered",
    ("MIXED", True): "report both shares and stop",
    ("MIXED", False): "report both shares and stop",
}


def next_arm(outcome: str, recovers: bool) -> str:
    return NEXT_ARM[(outcome, bool(recovers))]


def null_fires(ci: dict) -> bool:
    """prereg s8 C-NULL: a replicate FIRES iff its Q1 LOWER CI > 0.5 (the COMPOSITION bar; an interval rule, R51)."""
    return bool(ci["lo"] > 0.5)


def null_void(fires_by_key: dict) -> bool:
    """VOID iff the null fires on >= 2 of its replicates for any (source, seed)."""
    return any(sum(1 for f in fires if f) >= NULL_VOID_FIRES for fires in fires_by_key.values())


def branch_table_failures(mod) -> list:
    """Every decision function on a synthetic branch table.  `mod` is this module or a MUTATED copy; returns the failing cases (empty = every branch right)."""
    bad = []

    def chk(name, got, want):
        if got != want:
            bad.append(f"{name}: got {got!r}, want {want!r}")
    ci = lambda lo, hi: {"lo": lo, "hi": hi}                                                          # noqa: E731
    chk("share lo>0.5", mod.share_class(ci(0.55, 0.9)), "COMPOSITION")
    chk("share hi<0.5", mod.share_class(ci(0.1, 0.45)), "ROW-MISS")
    chk("share straddle", mod.share_class(ci(0.3, 0.7)), "MIXED")
    chk("share lo=0.5 is not >", mod.share_class(ci(0.5, 0.9)), "MIXED")
    chk("share hi=0.5 is not <", mod.share_class(ci(0.1, 0.5)), "MIXED")
    chk("share nan bounds", mod.share_class(ci(float("nan"), float("nan"))), "MIXED")
    chk("share lo>0.5 AND hi<0.5 impossible but lo wins first", mod.share_class(ci(0.6, 0.7)), "COMPOSITION")
    chk("outcome both COMPOSITION", mod.outcome_class(["COMPOSITION", "COMPOSITION"]), "COMPOSITION")
    chk("outcome both ROW-MISS", mod.outcome_class(["ROW-MISS", "ROW-MISS"]), "ROW-MISS")
    chk("outcome seeds disagree C / M", mod.outcome_class(["COMPOSITION", "MIXED"]), "MIXED")
    chk("outcome seeds disagree C / R", mod.outcome_class(["COMPOSITION", "ROW-MISS"]), "MIXED")
    chk("outcome seeds disagree R / M", mod.outcome_class(["ROW-MISS", "MIXED"]), "MIXED")
    chk("outcome both MIXED", mod.outcome_class(["MIXED", "MIXED"]), "MIXED")
    chk("recovers both", mod.tau_recovers([0.01, 0.012], 0.02, [0.0, 0.3]), True)
    chk("recovers boundary point == p2_hi", mod.tau_recovers([0.02, 0.02], 0.02, [0.0, 0.0]), True)
    chk("not recovers: rate above p2_hi on one seed", mod.tau_recovers([0.01, 0.0201], 0.02, [0.0, 0.0]), False)
    chk("not recovers: dpdms lo < 0 on one seed", mod.tau_recovers([0.01, 0.01], 0.02, [0.0, -0.001]), False)
    chk("not recovers: nan rate", mod.tau_recovers([float("nan"), 0.01], 0.02, [0.0, 0.0]), False)
    chk("veto_recovers any", mod.veto_recovers({0.3: False, 0.5: True, 0.7: False}), True)
    chk("veto_recovers none", mod.veto_recovers({0.3: False, 0.5: False}), False)
    chk("veto_recovers empty", mod.veto_recovers({}), False)
    chk("next COMPOSITION+recovers", mod.next_arm("COMPOSITION", True).split(":")[0], "a 0-train pick-rule registration")
    chk("next COMPOSITION no recovery", mod.next_arm("COMPOSITION", False).split(" (")[0], "a soft gating rule")
    chk("next ROW-MISS", mod.next_arm("ROW-MISS", False).split(" (")[0], "NC calibration on the fast band")
    chk("next ROW-MISS even if veto recovers", mod.next_arm("ROW-MISS", True).split(" (")[0], "NC calibration on the fast band")
    chk("next MIXED", mod.next_arm("MIXED", True), "report both shares and stop")
    chk("next MIXED no recovery", mod.next_arm("MIXED", False), "report both shares and stop")
    chk("null fires lo>0.5", mod.null_fires(ci(0.51, 0.9)), True)
    chk("null does not fire on lo=0.5", mod.null_fires(ci(0.5, 0.9)), False)
    chk("null does not fire on a high POINT with a low bound", mod.null_fires({"lo": 0.2, "hi": 0.99, "point": 0.8}), False)
    chk("null void 2 of 5", mod.null_void({("OOF", 0): [True, True, False, False, False]}), True)
    chk("null void 1 of 5", mod.null_void({("OOF", 0): [True, False, False, False, False], ("OOF", 1): [False] * 5}), False)
    chk("null void on the other key", mod.null_void({("OOF", 0): [False] * 5, ("BANKED", 1): [False, True, False, True, False]}), True)
    return bad


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the estimator (the landed census helpers) and the per-quantity statistics
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def per_log(v: np.ndarray, log_idx: np.ndarray, n_logs: int = N_LOGS) -> np.ndarray:
    return np.bincount(log_idx, weights=np.asarray(v, np.float64), minlength=n_logs).astype(np.float64)


def scale_ci(ci: dict, k: float) -> dict:
    out = dict(ci)
    for key in ("point", "lo", "hi"):
        if key in out:
            out[key] = float(out[key]) * k
    return out


def jsonable(o):
    if isinstance(o, dict):
        return {(str(k) if not isinstance(k, str) else k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


class Stats:
    """The paired log-cluster bootstrap bound to one set of draws, one log assignment."""

    def __init__(self, log_idx: np.ndarray, draws: np.ndarray, F, n_logs: int = N_LOGS):
        self.log_idx, self.draws, self.F, self.n_logs = log_idx, draws, F, n_logs
        self.n_sets = per_log(np.ones(len(log_idx)), log_idx, n_logs)

    def pl(self, v):
        return per_log(v, self.log_idx, self.n_logs)

    def ratio(self, num_set, den_set) -> dict:
        return self.F.ratio_ci(self.pl(num_set), self.pl(den_set), self.draws)

    def rate(self, flag) -> dict:
        return self.F.ratio_ci(self.pl(flag), self.n_sets, self.draws)

    def rate_diff(self, flag_a, flag_b) -> dict:
        return self.F.diff_ci(self.pl(flag_a), self.n_sets, self.pl(flag_b), self.n_sets, self.draws)

    def mean(self, v) -> dict:
        return self.F.ratio_ci(self.pl(v), self.n_sets, self.draws)

    def ratio_diff(self, num_a, den_a, num_b, den_b) -> dict:
        return self.F.diff_ci(self.pl(num_a), self.pl(den_a), self.pl(num_b), self.pl(den_b), self.draws)


def q1_q2(lab_nc: np.ndarray, sig_nc: np.ndarray, pk_e: np.ndarray, pk_p: np.ndarray, S: Stats, thr: float = SIG_THR, unsafe_lt1: bool = False) -> dict:
    """Q1, Q2 (prereg s6).  lab_nc [N, 64] the NC label (possibly permuted), sig_nc [N, 64] the NC sigmoid of the rows (never permuted).
    unsafe_lt1: the descriptive sensitivity `NC label < 1` instead of `== 0`."""
    unsafe = (lab_nc < 1.0) if unsafe_lt1 else (lab_nc == 0.0)
    ue, up = at(unsafe, pk_e), at(unsafe, pk_p)
    se, sp = at(sig_nc, pk_e), at(sig_nc, pk_p)
    A = ue & ~up                                   # the EP-led pick is NC-unsafe and P2's is not
    over = A & (se < thr)                          # ... and the EP-led pick's own NC sigmoid flagged it
    B = up                                         # P2's own NC-unsafe picks
    flagged_p = B & (sp < thr)
    Bm = up & ~ue                                  # matched reference: P2 unsafe, EP-led pick not
    flagged_m = Bm & (sp < thr)
    q1 = S.ratio(over, A)
    q2 = S.ratio(flagged_p, B)
    q2m = S.ratio(flagged_m, Bm)
    contrast = S.ratio_diff(over, A, flagged_p, B)
    qs = lambda x: [float(v) for v in np.quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0])] if len(x) else []     # noqa: E731
    return {"q1": q1, "q2": q2, "q2_matched": q2m, "q1_minus_q2": contrast, "n_A": int(A.sum()), "n_over": int(over.sum()), "n_B": int(B.sum()), "n_B_matched": int(Bm.sum()),
            "sigmoid_nc_at_A_picks_quantiles_min_q25_med_q75_max": qs(se[A]), "sigmoid_nc_at_P2_unsafe_picks_quantiles": qs(sp[B]), "class": share_class(q1)}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# pick metrics, the sweep, the null, the analysis
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def pick_flags(lab: np.ndarray, pdms: np.ndarray, pk: np.ndarray) -> dict:
    nc, dac = at(lab[..., NC], pk), at(lab[..., DAC], pk)
    return {"nc0": nc == 0.0, "dac0": dac == 0.0, "nc_lt1": nc < 1.0, "dac_lt1": dac < 1.0, "unsafe_epr1": ~((nc >= 1.0) & (dac >= 1.0)), "pd": at(pdms, pk).astype(np.float64)}


def sweep_one(z6: np.ndarray, lab: np.ndarray, pdms: np.ndarray, pk_p: np.ndarray, S: Stats, tau) -> dict:
    """One row of Q3 for the rows z6 at tau (None = no veto), paired against P2's UNVETOED pick pk_p."""
    if tau is None:
        pk, fb = rule_pick(z6), np.zeros(len(z6), bool)
    else:
        pk, fb = veto_pick(z6, tau)
    f, f0 = pick_flags(lab, pdms, pk), pick_flags(lab, pdms, pk_p)
    unvetoed = rule_pick(z6)
    return {"tau": tau, "n_fallback": int(fb.sum()), "n_changed_vs_unvetoed": int((pk != unvetoed).sum()), "n_changed_vs_p2": int((pk != pk_p).sum()),
            "nc_unsafe_rate": S.rate(f["nc0"]), "dac_unsafe_rate": S.rate(f["dac0"]), "pdms_x100": scale_ci(S.mean(f["pd"]), 100.0),
            "d_nc_unsafe_rate": S.rate_diff(f["nc0"], f0["nc0"]), "d_dac_unsafe_rate": S.rate_diff(f["dac0"], f0["dac0"]),
            "d_pdms_x100": scale_ci(S.mean(f["pd"] - f0["pd"]), 100.0),
            "nc_lt1_rate": S.rate(f["nc_lt1"]), "unsafe_epr1_rate": S.rate(f["unsafe_epr1"])}


def run_analysis(arr: dict, ep_cols: dict, S: Stats, taus=TAUS) -> dict:
    """arr: z6 [N, 64, 6] f32 (P2's stored logits), lab [N, 64, 6] f32.  ep_cols: {(source, seed): [N, 64] EP logit}.  -> the full JSON-able result (without controls)."""
    z6, lab = arr["z6"], arr["lab"]
    pdms = pdms_from_labels(lab)
    pk_p = rule_pick(z6)
    sig_nc = sigmoid64(z6[..., NC])
    sig_dac = sigmoid64(z6[..., DAC])
    p2 = {"P2_unvetoed": sweep_one(z6, lab, pdms, pk_p, S, None), "P2_with_veto": {str(t): sweep_one(z6, lab, pdms, pk_p, S, t) for t in taus}}
    p2_hi = float(p2["P2_unvetoed"]["nc_unsafe_rate"]["hi"])
    out = {"p2": p2, "p2_nc_unsafe_rate_hi": p2_hi, "composites": {}}
    for (src, seed), ep in sorted(ep_cols.items()):
        zE = z6.copy()
        zE[..., EP] = ep
        pk_e = rule_pick(zE)
        q = q1_q2(lab[..., NC], sig_nc, pk_e, pk_p, S)
        q_lt1 = q1_q2(lab[..., NC], sig_nc, pk_e, pk_p, S, unsafe_lt1=True)
        dac_a = q1_q2(lab[..., DAC], sig_dac, pk_e, pk_p, S)          # the DAC analogue (descriptive)
        sweep = {"none": sweep_one(zE, lab, pdms, pk_p, S, None)}
        for t in taus:
            sweep[str(t)] = sweep_one(zE, lab, pdms, pk_p, S, t)
        rec = {}
        out.setdefault("composites", {})[f"{src}_s{seed}"] = {"source": src, "seed": seed, "q": q, "q_nc_lt1_sensitivity": {k: q_lt1[k] for k in ("q1", "q2", "n_A", "n_over", "n_B")},
                                                                  "dac_analogue": {k: dac_a[k] for k in ("q1", "q2", "n_A", "n_over", "n_B", "class")}, "sweep": sweep,
                                                                  "n_picks_changed_vs_p2": int((pk_e != pk_p).sum())}
    # outcome and veto decisions, per source
    out["by_source"] = {}
    for src in sorted({s for s, _ in ep_cols}):
        seeds = sorted(sd for s, sd in ep_cols if s == src)
        classes = [out["composites"][f"{src}_s{sd}"]["q"]["class"] for sd in seeds]
        flags = {}
        for t in taus:
            pts = [float(out["composites"][f"{src}_s{sd}"]["sweep"][str(t)]["nc_unsafe_rate"]["point"]) for sd in seeds]
            los = [float(out["composites"][f"{src}_s{sd}"]["sweep"][str(t)]["d_pdms_x100"]["lo"]) for sd in seeds]
            flags[str(t)] = {"recovers": tau_recovers(pts, p2_hi, los), "rate_points": pts, "p2_hi": p2_hi, "dpdms_lo_x100": los,
                             "cond_i_rate_le_p2_hi": [bool(p <= p2_hi) for p in pts], "cond_ii_dpdms_lo_ge_0": [bool(v >= 0.0) for v in los]}
        rec_any = veto_recovers({t: v["recovers"] for t, v in flags.items()})
        oc = outcome_class(classes)
        out["by_source"][src] = {"seeds": seeds, "classes": classes, "outcome": oc, "veto_flags": flags, "veto_recovers": rec_any,
                                 "recovering_taus": [t for t, v in flags.items() if v["recovers"]], "next_arm": next_arm(oc, rec_any)}
    return out


def run_null(arr: dict, ep_cols: dict, S: Stats, reps: int = NULL_REPS) -> dict:
    """prereg s8 C-NULL: permute the 64 proposals' LABELS within each set (all six columns jointly); rows, picks and sigmoids untouched; recompute Q1; a replicate fires iff its lower CI > 0.5."""
    z6, lab = arr["z6"], arr["lab"]
    pk_p = rule_pick(z6)
    sig_nc = sigmoid64(z6[..., NC])
    N = len(z6)
    res, fires_by_key = {}, {}
    perms = []
    for rep in range(reps):
        rng = np.random.default_rng(NULL_SEED_BASE + NULL_SALT * rep)
        perms.append(np.argsort(rng.random((N, z6.shape[1])), axis=1))
    for (src, seed), ep in sorted(ep_cols.items()):
        zE = z6.copy()
        zE[..., EP] = ep
        pk_e = rule_pick(zE)
        rows, fires = [], []
        for rep in range(reps):
            nc_perm = np.take_along_axis(lab[..., NC], perms[rep], axis=1)
            q = q1_q2(nc_perm, sig_nc, pk_e, pk_p, S)
            f = null_fires(q["q1"])
            fires.append(f)
            rows.append({"rep": rep, "perm_seed": NULL_SEED_BASE + NULL_SALT * rep, "q1": q["q1"], "n_A": q["n_A"], "n_over": q["n_over"], "fires": f})
        res[f"{src}_s{seed}"] = {"replicates": rows, "fires": int(sum(fires)), "of": reps}
        fires_by_key[(src, seed)] = fires
    return {"by_key": res, "void": null_void(fires_by_key), "fire_rate": float(np.mean([f for v in fires_by_key.values() for f in v])) if fires_by_key else float("nan")}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the gate: controls that must hold before any quantity is read
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def gate_checks(arr: dict, ep_cols: dict, E, ref_pick_level: dict, planner_path: str | None) -> dict:
    """C-ID1, C-ID2, C-ID3, C-COL, C-PDMS (C-JOIN is checked at load in `prepare` and recorded).  Returns {check: {pass, detail}} and ok.  ref_pick_level: {seed: epr1_result pick_level dict at c*}."""
    z6, lab, top8 = arr["z6"], arr["lab"], arr["top8"]
    N = len(z6)
    chk, ok = {}, True

    def rec(name, passed, detail, gating=True):
        nonlocal ok
        chk[name] = {"pass": bool(passed), "detail": detail, "gating": bool(gating)}
        if gating:
            ok = ok and bool(passed)
    pk_p = rule_pick(z6)
    bank = top8[:, 0].astype(np.int64)
    n_eq = int((pk_p == bank).sum())
    n_eq_epr1 = int((pk_p == E.picks_np(z6)).sum())
    n_eq32 = int((rule_pick(z6, np.float32) == bank).sum())
    pin = check_planner_pin(planner_path)
    rec("C-ID1", n_eq == N and n_eq_epr1 == N and pin["status"] != "FAIL",
        {"float64_rule_equals_banked_pick": f"{n_eq}/{N}", "equals_epr1_picks_np": f"{n_eq_epr1}/{N}", "float32_torch_diagnostic_equals_banked": f"{n_eq32}/{N}", "planner_pin": {k: v for k, v in pin.items() if k != "rule_lines_text"},
                   "planner_pin_note": "NOT RUN is reported as NOT RUN, not PASS" if pin["status"] == "NOT_RUN" else "checked"})
    # C-PDMS
    pd_mine, pd_epr1 = pdms_from_labels(lab), E.pdms_np(lab)
    rec("C-PDMS", bool(np.array_equal(pd_mine, pd_epr1)), {"bit_equal_to_epr1_pdms_np": bool(np.array_equal(pd_mine, pd_epr1)), "weights": list(V1_W), "source": "refe/planner.py:520 V1_W"})
    # C-ID3 (AMENDMENT 2): the identity clauses gate; the registered reach clause (tau = 0.5 on P2's own rows changes >= 1 set) is INFORMATIONAL because it presumed P2 picks below NC sigmoid 0.5;
    # the reach of the veto is gated by C-ID3b on a planted, label-free perturbation.
    z0, c0 = veto_pick(z6, 0.0)
    z1, c1 = veto_pick(z6, 1.01)
    z5, c5 = veto_pick(z6, 0.5)
    sig = sigmoid64(z6[..., NC])
    ok_ge = bool((at(sig, z5)[~c5] >= 0.5).all())
    rec("C-ID3", bool(np.array_equal(z0, pk_p) and np.array_equal(z1, pk_p) and c1.all() and not c0.any() and ok_ge),
        {"tau0_equals_unvetoed": bool(np.array_equal(z0, pk_p)), "tau1.01_equals_unvetoed_all_fallback": bool(np.array_equal(z1, pk_p) and c1.all()), "tau0.5_changed_sets": int((z5 != pk_p).sum()),
         "tau0.5_fallback_sets": int(c5.sum()), "tau0.5_vetoed_pick_nc_sigmoid_ge_tau": ok_ge})
    sig_at_p2_pick = at(sig, pk_p)
    rec("C-ID3-registered-reach-clause", 0 < int((z5 != pk_p).sum()) < N,
        {"registered_text": "veto at tau = 0.5 changes the pick on at least one set and on fewer than all", "tau0.5_changed_sets": int((z5 != pk_p).sum()), "min_nc_sigmoid_at_p2_picks": float(sig_at_p2_pick.min()),
         "p2_picks_below_tau": {str(t): int((sig_at_p2_pick < t).sum()) for t in TAUS}, "note": "INFORMATIONAL since Amendment 2 (amended after it failed; not a pre-registered pass)"}, gating=False)
    zs = z6.copy()
    zs[..., NC] = zs[..., NC] - 3.0
    pk_s = rule_pick(zs)
    v_s, f_s = veto_pick(zs, 0.5)
    ch_s = int((v_s != pk_s).sum())
    ge_s = bool((at(sigmoid64(zs[..., NC]), v_s)[~f_s] >= 0.5).all())
    rec("C-ID3b", 0 < ch_s < N and ge_s, {"shift_logits": -3.0, "tau0.5_changed_sets_under_shift": ch_s, "of": int(N), "tau0.5_fallback_sets_under_shift": int(f_s.sum()), "non_fallback_vetoed_pick_nc_sigmoid_ge_tau": ge_s})
    # C-COL + C-ID2
    safe = E.safe_np(lab)
    for (src, seed), ep in sorted(ep_cols.items()):
        zE = z6.copy()
        zE[..., EP] = ep
        other = [c for c in range(6) if c != EP]
        col_ok = bool(np.array_equal(zE[..., other], z6[..., other])) and bool((zE[..., EP] != z6[..., EP]).any())
        rec(f"C-COL[{src}_s{seed}]", col_ok, {"other_columns_bit_equal": bool(np.array_equal(zE[..., other], z6[..., other])), "ep_proposals_differing": int((zE[..., EP] != z6[..., EP]).sum())})
        if src == "OOF":
            pk_e = rule_pick(zE)
            rows = np.arange(N)
            pdn, pdp = pd_epr1[rows, pk_e].astype(np.float64), pd_epr1[rows, pk_p].astype(np.float64)
            got = {"new_pdms_x100": 100.0 * pdn.sum() / N, "p2_pdms_x100": 100.0 * pdp.sum() / N, "unsafe_rate_new": float((~safe[rows, pk_e]).sum() / N), "unsafe_rate_p2": float((~safe[rows, pk_p]).sum() / N)}
            ref = ref_pick_level[seed]
            diffs = {k: abs(got[k] - float(ref[k])) for k in got}
            changed = int((pk_e != pk_p).sum())
            rec(f"C-ID2[seed{seed}]", all(v <= ID2_TOL for v in diffs.values()) and changed == int(ref["oof_picks_changed"]),
                {"abs_diff_vs_banked": diffs, "tolerance": ID2_TOL, "picks_changed": changed, "picks_changed_banked": int(ref["oof_picks_changed"]), "got": got})
    return {"ok": ok, "checks": chk}
