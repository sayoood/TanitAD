#!/usr/bin/env python3
"""SEARCH-DOSE (MM_RULING_SPC.md DQ7): C1's chain with ONE lever moved, built WITHOUT editing a single sealed C1 file.

WHY A SUBCLASS AND NOT A FLAG IN spc_chain.py.  The C1 seal pins `spc_chain.py` (and 5 more operative files) by sha256 and the chain REFUSES (exit 5) when a pinned file differs.  A `--config` flag added there would
change its sha and void a seal the MM has already ruled on.  So the sealed chain is IMPORTED, never copied or edited; this file overrides only the hops that hard-wire the C8 config (MEASURED, spc_chain.py):
`CONFIG` (:45) is a `Cfg` default that `main` never exposes, and the C8 sha is read at `check_seal` (:159), `common` (:278), `stamp_ok` (:293) and `merged_valid` (:325).  Everything else (data gate, bundle gate, the
launcher, the labeller, the C1 outcome rules) is C1's own code, so "everything else identical to C1's arm" holds by construction.

What this arm does:  the pinned O4 config (stepB_config_O4.json, sha256 1ebd7196..., M = 256, elites 32, widen 2.0) on the SAME B-REST bundle C1 built.  CPU only: `gpu_stage` and `cid_gate` are overridden to REFUSE, so no
inherited path can start the GPU or build a bundle.  Its outcomes are C1's rules verbatim (`spc_analyse.analyse_c1`); the NEXT arm text is the dose amendment's; the contrast O4 minus C8 is descriptive and ungated.

    python dose_chain.py [--work D:/.../spc_dose] [--c1-work D:/.../spc] [--dose-pkg <dir with AMENDMENT_DOSE.md>] [--dry-run]
Markers are ZZDOSE_* at line start (START, SEAL_OK, AMEND_SEAL_OK, DATA_OK, C1_DONE, DISK_OK, BUNDLE_OK, SUBID_PASS, REACH_PASS, SEARCH_DONE, SELFREP_PASS, LABEL_DONE, ANALYSIS_DONE, DONE <outcome>, ENDPOINT, CONTRAST, FAMILY x4, C2CONFIG, NEXT, STAMPS)
and ZZDOSE_STOP <reason> with the C1 exit codes (3 gate, 5 seal/data/C1, 6 search, 7 label/disk, 8 analysis, 9 crash)."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPC_PKG = os.environ.get("DOSE_SPC_PKG", "C:/Users/Admin/par18/spc")
SPC_CODE = os.environ.get("DOSE_SPC_CODE", os.path.join(SPC_PKG, "code"))
if SPC_CODE not in sys.path:
    sys.path.insert(0, SPC_CODE)
import spc_analyse as AC  # noqa: E402  (sealed C1 code, imported)
import spc_chain as CH  # noqa: E402  (sealed C1 code, imported)
import spc_cid as CID  # noqa: E402  (sealed C1 code, imported: the bitwise row comparison)
import spc_launch as SL  # noqa: E402  (sealed C1 code, imported)

import analyse_stepB as AB  # noqa: E402  (Step B code, put on sys.path by spc_chain)
import stepA as A  # noqa: E402

AMEND_NAME = "AMENDMENT_DOSE.md"
AMEND_SHA = "AMENDMENT_DOSE.sha256"
RULING_NAME = "MM_RULING_DOSE.md"
CONFIG_C8 = CH.CONFIG
CONFIG_O4 = "C:/Users/Admin/par18/l2stepb/raw/stepB_config_O4.json"
O4_SHA256 = "1ebd719617e8978a03cea0254edd6f531504c1d070792d066736130ceb929013"      # also a pin of the C1 seal (`config_o4_sha256`); the two must agree with the file
O4_LEVERS = {"k_iters": 5, "m_pop": 256, "n_elite": 32, "widen": 2.0}               # what the pinned file says; C8 is 5 / 64 / 8 / 1.0
DOSE_WORK = "D:/Projects/TanitAD-artifacts/spc_dose"
DOSE_MIN_FREE_GIB = 0.25                                                             # the arm writes < 20 MB: shards, one merged file, labels, a stamp, logs
C1_STAGES = ("SEAL_OK", "DATA_OK", "GPU_STAGE_OK", "REACH_PASS", "CID_PASS", "SEARCH_DONE", "LABEL_DONE")
DOSE_STAGES = ("SEAL_OK", "AMEND_SEAL_OK", "DATA_OK", "C1_DONE", "BUNDLE_OK", "SUBID_PASS", "REACH_PASS", "SEARCH_DONE", "SELFREP_PASS", "LABEL_DONE")
SELFREP_N, SUBID_N = 8, 16                                                           # MM_RULING_DOSE.md DQ5: the first 8 / the first 16 B-REST tokens
ROW_KEYS = CID.SEARCH_KEYS + ("token", "bundle_index")                               # per-token arrays compared bit for bit; "seeds" is compared whole; the t_* timings and the meta are not
LICENCE_SAFETY_BAR, LICENCE_MOVED_BAR = AC.BAR_SAFETY, 0.05                          # MM_RULING_C2_LICENCE.md section 2: each safety point >= -0.25, moved share >= 5 %

STAMPS = ("MEASURED; SEEN-SCENE (all of B-REST is inside M-SCENE; M-PDM overlap 0); the SAME bundle and the SAME 905 tokens / 24 logs as C1, so this is a SECOND READ OF THE SAME TOKENS (not a fresh draw); "
          "SELECTION tier, label-only (not T1, not navtest); ONE training seed, no head substitution (H-ESTIM-SEED-1 open); the interval is the paired log-cluster bootstrap (B = 10,000, seed 20260927) = "
          "'another draw of LOGS' only, the 3-search-seed spread beside it; the contrast O4 minus C8 is DESCRIPTIVE and UNGATED and moves M x4, elites x4 AND widen x2 together (not a dose-response in M); "
          "no capability claim; STRATEGIC PAUSED")

_CLOSE = ("the search is CLOSED for P2's scorer on this footing and the report NAMES the next levers with what unblocks them: the P6 trunk (PI authorisation of trunk compute) and an L2-BRANCH with a TTC-protecting "
          "term (own amendment, a launch_gate.py PASS token, the GPU lock)")
DOSE_NEXT = {
    "O1_CONFIRMED": ("the config sent to the ONE navtest look (C2) is set mechanically by the committed table (section 5b) from C1's read and this read, and is printed as ZZDOSE_C2CONFIG: O4 if C1's C2 licence is NOT met "
                     "and this read is O1 with each safety point >= -0.25 and moved share >= 5 %; C8 if C1's licence is met and this read does not license; if both license, C8 unless the paired O4-minus-C8 contrast has its "
                     "lower CI > 0, then O4. H-SEARCH-P2-1 gets exactly ONE navtest look with ONE config, still behind EP-ADMISSION and C2's own amendment, which carries the per-token search cost of that config (about "
                     "2.9x for O4, ESTIMATED) MEASURED on the deployment target or named UNMEASURED. The M-only follow-up arm (DQ1) goes under its own amendment. No navtest token is read."),
    "O2_DIRECTIONAL": ("as O1_CONFIRMED through the table (section 5b), with one difference: O2 is NOT enough to send O4 when C1's C2 licence is not met (that would be the second look at the same tokens), so that path "
                       "yields no C2 and the committed next arms stand; if C1's licence is met and this read is O2 with each safety point >= -0.25 and moved share >= 5 %, C8 unless the paired O4-minus-C8 contrast has "
                       "its lower CI > 0, then O4. A safety miss is FLAGGED. The M-only follow-up arm (DQ1) goes under its own amendment, only because this read is O1 or O2."),
    "TRADE": ("the TRADE-restricted arm (Step B sealed text section 3: elites and pick restricted to candidates whose sigma(NC) and sigma(DAC) are >= the pick's inside the same set; no new head) on the O4 config, "
              "CPU only, own amendment sealed before launch."),
    "O3_NULL": ("the larger search adds nothing detectable on this footing. If C1 also read O3_NULL / NEGATIVE / O4_INERT: " + _CLOSE + ". If C1 read O1 / O2 the C2 candidate stays the C8 config (the dose licenses nothing)."),
    "NEGATIVE": ("the larger population HURTS: reported plainly (HYPOTHESIS, not measured here: scorer exploitation or the 256-set being outside the 64-set the scorer was trained on). If C1 licensed C2 the candidate "
                 "stays the C8 config; otherwise " + _CLOSE + "."),
    "O4_INERT": ("even M = 256 with widen 2.0 moves the pick on < 5 % of outputs: " + _CLOSE + "."),
}


def next_arm(outcome: str, c1_outcome: str) -> str:
    """the dose amendment's NEXT arm for this outcome; C1's INERT special case (this arm IS C1's committed raised-M arm, run ONCE) is stated, never silent"""
    txt = DOSE_NEXT[outcome]
    if c1_outcome == "O4_INERT":
        if outcome in ("O1_CONFIRMED", "O2_DIRECTIONAL", "TRADE"):
            return "C1 read INERT; this was its committed raised-M arm, run ONCE, and it MOVED the pick: " + txt
        return "C1 read INERT; this was its committed raised-M arm, run ONCE: " + txt
    return txt


def contrast(bundle: dict, so: dict, lab: dict, so_c1: dict, lab_c1: dict, B: int = A.N_BOOT, seed: int = A.SEED) -> dict:
    """DESCRIPTIVE, UNGATED: O4 minus C8 on the same tokens and the same search-seed ids, in PDMS points = 100 x the [0,1] difference, paired per token and seed, paired log-cluster bootstrap.
    Positive = the O4 config's search output has the higher true label.  The two arms differ in M, elites AND widen together (the pinned file), so this is not a dose-response in M."""
    tok = [str(t) for t in so["token"]]
    if tok != [str(t) for t in so_c1["token"]] or tok != [str(t) for t in lab["token"]] or tok != [str(t) for t in lab_c1["token"]]:
        raise ValueError("the dose and C1 search/label files are not for the same tokens in the same order")
    if not np_equal(so["seeds"], so_c1["seeds"]) or not np_equal(so["bundle_index"], so_c1["bundle_index"]):
        raise ValueError("the dose and C1 search files do not share the search-seed ids / bundle indices")
    li, n_logs, _ = AB.log_index(bundle, [int(i) for i in so["bundle_index"]])
    boot = A.boot_indices(n_logs, B, seed)
    pd, pc = _f64(lab["pdms"]), _f64(lab_c1["pdms"])
    d = AB.POINTS * (pd - pc)                                                       # [n,S]  O4 minus C8
    sd, sc = _f64(lab["sub"]), _f64(lab_c1["sub"])                                  # [n,S,6]
    S = d.shape[1]
    return {"definition": "mean over the 3 search seeds per token of 100 x [PDMS(O4 output) - PDMS(C8 output)], ratio of sums over WHOLE LOGS, paired log-cluster bootstrap; DESCRIPTIVE, UNGATED",
            "pooled": AB._ci(d.mean(1), li, n_logs, boot), "per_seed": [AB._ci(d[:, s], li, n_logs, boot) for s in range(S)],
            "sub_scores_points": {k: AB._ci(AB.POINTS * (sd[:, :, j] - sc[:, :, j]).mean(1), li, n_logs, boot) for j, k in enumerate(AB.SUBS)},
            "moved_share": {"O4": float(_bool(so["moved"]).mean()), "C8": float(_bool(so_c1["moved"]).mean())},
            "confound": "M x4, elites x4 and widen x2 move together in the pinned O4 file: the contrast is O4 minus C8, not a dose-response in M"}


def np_equal(a, b) -> bool:
    import numpy as np
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _f64(x):
    import numpy as np
    return np.asarray(x, np.float64)


def _bool(x):
    import numpy as np
    return np.asarray(x, bool)


def licensed(read: dict, strict: bool = False) -> bool:
    """the C2 licence terms of MM_RULING_C2_LICENCE.md section 2 applied to one read: O1 or O2 (only O1 when `strict`), each safety point >= -0.25, moved share >= 5 %.  A missing or NaN term is NOT a licence."""
    ok = ("O1_CONFIRMED",) if strict else ("O1_CONFIRMED", "O2_DIRECTIONAL")
    try:
        safety = all(float(read["safety_points"][k]) >= LICENCE_SAFETY_BAR for k in AC.SAFETY_TERMS)
        moved = float(read["moved_share"]) >= LICENCE_MOVED_BAR
    except (KeyError, TypeError, ValueError):
        return False
    return bool(read.get("outcome") in ok and safety and moved)


def c2_config(c1_read: dict, dose_read: dict, contrast_lo: float) -> dict:
    """MM_RULING_DOSE.md DQ4, MECHANICAL and committed before either read: H-SEARCH-P2-1 gets exactly ONE navtest look with ONE config.  Pure; no outcome of either read can reshape it.
      C1 licence met,     dose licence not met                           -> C8
      C1 licence not met, dose O1_CONFIRMED + safety + moved share        -> O4   (O2 is NOT enough on this path: it would be the second look at the same tokens)
      C1 licence met,     dose licence met (O1 or O2 + safety + moved)    -> C8 unless the paired O4-minus-C8 contrast has lower CI > 0, then O4
      C1 licence not met, dose not met (or O2 only)                       -> no C2 (config None); the committed next arms stand"""
    c1, dose_any, dose_strict = licensed(c1_read), licensed(dose_read), licensed(dose_read, strict=True)
    try:
        lo_pos = bool(float(contrast_lo) > 0.0)
    except (TypeError, ValueError):
        lo_pos = False
    if c1 and not dose_any:
        cfg, row = "C8", 1
    elif c1 and dose_any:
        cfg, row = ("O4" if lo_pos else "C8"), 3
    elif dose_strict:
        cfg, row = "O4", 2
    else:
        cfg, row = None, 4
    why = {1: "C1's licence met, the dose's not: C8", 2: "C1's licence not met, the dose O1_CONFIRMED with the safety and moved-share terms: O4",
           3: f"both licences met: C8 unless the paired O4-minus-C8 contrast has lower CI > 0 (lower CI > 0: {lo_pos}): {'O4' if lo_pos else 'C8'}",
           4: "C1's licence not met and the dose's not met (or O2 only): no C2, the committed next arms stand"}[row]
    return {"config": cfg, "row": row, "why": why, "c1_licensed": c1, "dose_licensed": dose_any, "dose_o1_strict": dose_strict, "contrast_lower_ci_gt_0": lo_pos}


def rows_identical(ref: dict, new: dict, n: int) -> dict:
    """BIT FOR BIT, by raw bytes (a -0.0 differs from 0.0; a NaN equals itself): the first `n` rows of `ref` against `new` (which must hold exactly n rows), every per-token search array plus the search seeds; the key sets
    must agree (the meta aside; the wall-clock t_* arrays must exist on both sides but their VALUES are never compared).  NON-VACUOUS only if the reference search actually moved its mean (hist_mu not all zero): an identity of two runs that did nothing proves nothing."""
    import numpy as np
    skip = {"meta"}
    fails = []
    if sorted(set(ref) - skip) != sorted(set(new) - skip):
        fails.append(f"key sets differ: {sorted((set(ref) ^ set(new)) - skip)}")
    for k in ROW_KEYS + ("seeds",):
        if k not in ref or k not in new:
            fails.append(f"{k}: missing")
            continue
        a = np.asarray(ref[k]) if k == "seeds" else np.asarray(ref[k])[:n]
        same, nbad, _ = CID._same_bytes(a, np.asarray(new[k]))
        if not same:
            fails.append(f"{k}: differs ({nbad} rows)" if k != "seeds" else f"{k}: differs")
    non_vacuous = bool(np.any(np.asarray(ref["hist_mu"])[:n] != 0)) if "hist_mu" in ref else False
    if not non_vacuous:
        fails.append("vacuous: the reference search never moved its mean")
    moved = int(np.asarray(ref["moved"])[:n].sum()) if "moved" in ref else None
    return {"pass": not fails, "failures": fails, "n_rows": int(n), "n_moved_outputs_in_reference": moved, "non_vacuous": non_vacuous}


def analyse_dose(bundle: dict, so: dict, lab: dict, so_c1: dict, lab_c1: dict, c1_read: dict, B: int = A.N_BOOT, seed: int = A.SEED, with_concordance: bool = True) -> dict:
    """C1's outcome rules VERBATIM (`spc_analyse.analyse_c1`: dPDMS vs P2's pick), the dose amendment's NEXT arm, the descriptive contrast, and the mechanical C2 config (`c2_config`).
    `c1_read` = C1's RESULT_SPC.json fields {outcome, safety_points, moved_share}."""
    res = AC.analyse_c1(bundle, so, lab, B=B, seed=seed, with_concordance=with_concordance)
    c1_outcome = c1_read["outcome"]
    res["read"] = "DOSE"
    res["arm"] = "P2HEAD search, O4 config (M 256, elites 32, widen 2.0), no head substitution"
    res["c1_outcome"] = c1_outcome
    res["next_arm"] = next_arm(res["outcome"], c1_outcome)
    res["contrast_O4_minus_C8"] = contrast(bundle, so, lab, so_c1, lab_c1, B=B, seed=seed)
    res["c2_config"] = c2_config(c1_read, {"outcome": res["outcome"], "safety_points": res["safety_points"], "moved_share": res["moved_share"]}, res["contrast_O4_minus_C8"]["pooled"]["lo"])
    res["stamps"] = STAMPS
    return res


class DoseChain(CH.Chain):
    def emit(self, marker, *parts):
        print("ZZDOSE_" + marker + (" " + " ".join(str(p) for p in parts) if parts else ""), file=self.out, flush=True)

    # ------------------------------------------------------------------------------------------------ the hops that are NOT C1's
    def bundle_dir(self, which):
        if which != "C1":
            self.stop(5, "DOSE_BUNDLE", f"the dose arm has exactly one bundle (C1's B-REST); asked for {which!r}")
        return os.path.join(self.cfg.c1_work, "B_REST", "bundle")                     # C1's bundle, never a copy: the SAME bytes

    def gpu_stage(self):
        self.stop(3, "DOSE_NEVER_BUILDS_A_BUNDLE", "CPU only: the dose arm reuses C1's B-REST bundle and starts no GPU job")

    def cid_gate(self, pins):
        self.stop(3, "DOSE_HAS_NO_CID", "C-ID is C1's, run on the C8 config against a banked C8 search; no banked O4 search exists")

    def check_seal(self):
        """C1's seal, byte for byte (the C8 config file included), THEN the O4 config against the seal's `config_o4_sha256` pin AND this file's constant.  Returns C1's pins with `config_sha256` REPLACED by the
        O4 sha: every inherited method that binds `pins["config_sha256"]` (`common`, `stamp_ok`, `merged_valid`) then binds the O4 file.  The C8 value is kept as `config_c8_sha256`."""
        c = self.cfg
        o4_path = c.config
        c.config = CONFIG_C8
        try:
            pins = super().check_seal()
        finally:
            c.config = o4_path
        want, got = pins.get("config_o4_sha256"), CH.sha_lf(o4_path)
        if not (want == got == O4_SHA256):                                             # @O4PIN
            self.stop(5, "SEAL_PIN_MISMATCH", f"config_o4_sha256: C1 seal {str(want)[:12]}, file {got[:12]}, constant {O4_SHA256[:12]}")
        return {**pins, "config_c8_sha256": pins["config_sha256"], "config_sha256": want}   # @SWAP

    def check_amendment(self, pins):
        """G-SEAL-DOSE: AMENDMENT_DOSE.md equals its .sha256, its first Status line is SEALED, and its pins (C1's prereg sha, the O4 sha, every `code:` file here) equal the files"""
        c = self.cfg
        p, sp = os.path.join(c.dose_pkg, AMEND_NAME), os.path.join(c.dose_pkg, AMEND_SHA)
        if not (os.path.isfile(p) and os.path.isfile(sp)):
            self.stop(5, "AMEND_MISSING", f"{p} / {sp}")
        text = open(p, "rb").read().replace(b"\r\n", b"\n")
        want = open(sp, encoding="utf-8").read().split()[0].strip().lower()
        if hashlib.sha256(text).hexdigest() != want:                                   # @AMENDSHA
            self.stop(5, "AMEND_SHA_MISMATCH", "the text differs from its .sha256")
        t = text.decode("utf-8")
        m = re.search(r"^Status: .*$", t, re.M)
        if not m or not m.group(0).startswith("Status: **SEALED"):                     # @AMENDSTATUS
            self.stop(5, "AMEND_NOT_SEALED", f"first status line: {(m.group(0)[:40] if m else None)!r}")
        ap = CH.parse_pins(t)
        c1_sha = open(c.sha_file, encoding="utf-8").read().split()[0].strip().lower()
        bad = []
        if ap.get("c1_prereg_sha256") != c1_sha:
            bad.append("c1_prereg_sha256")
        if ap.get("config_o4_sha256") != O4_SHA256 or ap.get("config_c8_sha256") != pins["config_c8_sha256"]:
            bad.append("config pins")
        rp = os.path.join(c.dose_pkg, RULING_NAME)
        if not os.path.isfile(rp) or ap.get("ruling_sha256") != CH.sha_lf(rp):          # @RULINGPIN
            bad.append("ruling_sha256")
        codes = {k[5:]: v for k, v in ap.items() if k.startswith("code:")}
        if "dose_chain.py" not in codes:
            bad.append("dose_chain.py not pinned")
        for f, v in codes.items():
            q = os.path.join(HERE, f)
            if not os.path.isfile(q) or CH.sha_lf(q) != v:
                bad.append(f"code:{f}")
        if bad:                                                                        # @AMENDPINS
            self.stop(5, "AMEND_PIN_MISMATCH", ";".join(bad))
        self.emit("AMEND_SEAL_OK", want[:12])
        self.state("AMEND_SEAL_OK")

    def check_c1_done(self, pins):
        """the dose runs after C1's DONE, whatever C1 read: its stages, RESULT_SPC.json, merged search and labels must exist and be valid under C1's OWN (C8) pins"""
        c = self.cfg
        if os.path.abspath(c.work) == os.path.abspath(c.c1_work):                      # @WORKGUARD
            self.stop(5, "DOSE_WORK_IS_C1_WORK", "the dose would overwrite C1's chain_state.json")
        try:
            st = json.load(open(os.path.join(c.c1_work, "chain_state.json")))
            res = json.load(open(os.path.join(c.c1_work, "RESULT_SPC.json")))
        except (OSError, ValueError) as e:
            self.stop(5, "C1_NOT_DONE", f"{type(e).__name__}: chain_state.json / RESULT_SPC.json")
        miss = [s for s in C1_STAGES if s not in st.get("done", [])]
        if miss:                                                                       # @C1STAGES
            self.stop(5, "C1_NOT_DONE", f"stages not done: {miss}")
        if res.get("outcome") not in AC.OUTCOMES:
            self.stop(5, "C1_NOT_DONE", f"RESULT_SPC.json outcome {res.get('outcome')!r}")
        if not isinstance(res.get("safety_points"), dict) or not all(k in res["safety_points"] for k in AC.SAFETY_TERMS) or not isinstance(res.get("moved_share"), (int, float)):      # @C1TERMS
            self.stop(5, "C1_NOT_DONE", "RESULT_SPC.json carries no safety_points / moved_share: C1's C2 licence cannot be read")
        merged, labels = os.path.join(c.c1_work, "search_C1_P2HEAD.npz"), os.path.join(c.c1_work, "labels_C1_P2HEAD.npz")
        c8_pins = {**pins, "config_sha256": pins["config_c8_sha256"]}
        ok, why = CH.Chain.merged_valid(self, merged, "C1", 905, c8_pins)               # C1's own file under C1's own (C8) config pin
        if ok:
            ok, why = self.labels_valid(labels, merged)
        if not ok:
            self.stop(5, "C1_NOT_DONE", f"C1's merged search / labels invalid: {why}")
        import numpy as np
        self.c1_bundle_sha = json.loads(str(np.load(merged, allow_pickle=False)["meta"]))["bundle_sha256"]
        if CH.sha256_file(os.path.join(self.bundle_dir("C1"), "bundle.npz")) != self.c1_bundle_sha:      # @SAMEBUNDLE
            self.stop(5, "C1_NOT_DONE", "the bundle on disk is not the bundle C1 searched")
        self.emit("C1_DONE", res["outcome"], "bundle", self.c1_bundle_sha[:12])
        self.state("C1_DONE")
        return {"outcome": res["outcome"], "safety_points": res["safety_points"], "moved_share": res["moved_share"], "result_sha256": CH.sha256_file(os.path.join(c.c1_work, "RESULT_SPC.json")), "merged": merged, "labels": labels}

    def check_bundle(self):
        ok, why = self.bundle_valid("C1", 905)
        if not ok:
            self.stop(3, "G_BUNDLE_INVALID", f"C1: {why}")
        self.emit("BUNDLE_OK")
        self.state("BUNDLE_OK")

    @contextlib.contextmanager
    def _as_c1_c8(self, pins):
        """argv-building context for G-SUBID: the SAME methods (`stamp_ok`, `search_args`, `common`) with C1's work dir (its gate stamp) and the C8 config.  Nothing is run or written inside it."""
        c = self.cfg
        old = (c.work, c.config)
        c.work, c.config = c.c1_work, CONFIG_C8
        try:
            yield {**pins, "config_sha256": pins["config_c8_sha256"]}
        finally:
            c.work, c.config = old

    def _replicate(self, argv, out_dir, n, ref_path, expect_cfg, tag):
        """run `argv` (a search over tokens 0..n-1) in a fresh child process into a fresh dir, then require its shard to equal the first n rows of `ref_path` BIT FOR BIT; -> the comparison dict (raises StopChain on any miss)"""
        import numpy as np
        shutil.rmtree(out_dir, ignore_errors=True)                                      # a replicate is always fresh: a stale shard is never accepted
        rc = self.child(argv + ["--start", "0", "--end", str(n), "--shard", str(n)], f"{tag.lower()}.log")
        if rc != 0:                                                                     # @REPRC
            self.stop(3, f"G_{tag}_NOT_PASSED", f"the replicate search exited rc={rc}")
        shard = os.path.join(out_dir, "shard_P2HEAD_0000.npz")
        if not os.path.isfile(shard):
            self.stop(3, f"G_{tag}_NOT_PASSED", "the replicate wrote no shard")
        ld = lambda p: {k: v for k, v in np.load(p, allow_pickle=False).items()}      # noqa: E731
        new, ref = ld(shard), ld(ref_path)
        meta = json.loads(str(new["meta"]))
        pins_ok = (meta.get("arm") == "P2HEAD" and meta.get("head_sha256") is None and meta.get("config_sha256") == expect_cfg and meta.get("bundle_sha256") == getattr(self, "c1_bundle_sha", None)
                   and (meta.get("data_clause") or {}).get("carve_sha256") == SL.READS["C1"] and (meta.get("lo"), meta.get("hi")) == (0, n))
        if not pins_ok:                                                                 # @REPMETA
            self.stop(3, f"G_{tag}_NOT_PASSED", "the replicate's meta (arm, head, config, bundle, data clause, range) is not the expected one")
        res = rows_identical(ref, new, n)
        if not res["pass"]:                                                             # @REPCMP
            self.stop(3, f"G_{tag}_NOT_PASSED", "; ".join(res["failures"])[:400])
        return res

    def subid_gate(self, pins, c1):
        """G-SUBID (MM_RULING_DOSE.md DQ5.2): THIS subclass run with the C8 config on the first 16 B-REST tokens reproduces C1's banked search outputs bit for bit, so the subclass does not change the search path"""
        out_dir = self.cfg.d("subid_C8")
        with self._as_c1_c8(pins) as p8:
            if not self.stamp_ok("C1", p8):                                             # C1's own C8 stamp, bound to this bundle
                self.stop(3, "G_SUBID_NOT_PASSED", "C1's G-REACH stamp for the C8 config is missing or does not bind this bundle")
            argv = self.search_args("C1", p8, out_dir)
        res = self._replicate(argv, out_dir, SUBID_N, c1["merged"], pins["config_c8_sha256"], "SUBID")
        self.emit("SUBID_PASS", res["n_rows"], "tokens", "moved_outputs", res["n_moved_outputs_in_reference"])
        self.state("SUBID_PASS")

    def selfrep_gate(self, pins, merged):
        """G-SELFREP (MM_RULING_DOSE.md DQ5.1): a second process re-runs the first 8 B-REST tokens with the O4 config; bit-equal to the merged output (thread-count nondeterminism, MM_RULING_SPC.md DQ8)"""
        out_dir = self.cfg.d("selfrep_O4")
        res = self._replicate(self.search_args("C1", pins, out_dir), out_dir, SELFREP_N, merged, pins["config_sha256"], "SELFREP")
        self.emit("SELFREP_PASS", res["n_rows"], "tokens", "moved_outputs", res["n_moved_outputs_in_reference"])
        self.state("SELFREP_PASS")

    def mint_gates(self, pins):
        """G-REACH-STEPB on the O4 config and C1's bundle (Step B's own gate; its stamp binds checkpoint, config, bundle and code, so C1's C8 stamp cannot stand in for it)"""
        c = self.cfg
        if not self.stamp_ok("C1", pins):
            argv = [c.py_cpu, "-u", os.path.join(c.code_dir, "spc_launch.py"), "gate", "--read", "C1", "--head", c.head_s0, "--head-sha256", pins["head_s0_sha256"], *self.common("C1", pins),
                    "--out-dir", c.d("gate_C1")]
            rc = self.child(argv, "gate_C1.log")
            if rc != 0 or not self.stamp_ok("C1", pins):
                self.stop(3, "G_REACH_NOT_PASSED", f"C1/O4: rc={rc}")
        self.emit("REACH_PASS", "C1", "O4", pins["config_sha256"][:12])
        self.state("REACH_PASS")

    def merged_valid(self, path, which, n, pins):
        ok, why = super().merged_valid(path, which, n, pins)                           # arm P2HEAD, no head, ckpt, CONFIG = the O4 sha (pins swapped), data clause
        if not ok:
            return ok, why
        import numpy as np
        meta = json.loads(str(np.load(path, allow_pickle=False)["meta"]))
        if getattr(self, "c1_bundle_sha", None) is None or meta.get("bundle_sha256") != self.c1_bundle_sha:      # @SAMEBUNDLE2
            return False, "G-SAMEBUNDLE: the search was not run on the bundle C1 searched"
        return True, ""

    def run_analysis(self, merged, labels, pins, c1):
        c = self.cfg
        miss = [s for s in DOSE_STAGES if s not in self.st["done"]]
        if miss:                                                                       # @DOSEORDER
            self.stop(8, "G_ORDER", f"analysis refused: stages not done: {miss}")
        import numpy as np
        ld = lambda p: {k: v for k, v in np.load(p, allow_pickle=False).items()}      # noqa: E731
        try:
            res = analyse_dose(A.load_npz(os.path.join(self.bundle_dir("C1"), "bundle.npz")), ld(merged), ld(labels), ld(c1["merged"]), ld(c1["labels"]), c1)
        except Exception as e:
            self.stop(8, "ANALYSIS_FAILED", f"{type(e).__name__}: {e}")
        res["c1"] = {"outcome": c1["outcome"], "result_sha256": c1["result_sha256"], "bundle_sha256": self.c1_bundle_sha, "config_c8_sha256": pins["config_c8_sha256"], "config_o4_sha256": pins["config_sha256"]}
        out = c.d("RESULT_DOSE.json")
        json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        self.emit("ANALYSIS_DONE")
        self.emit("DONE", res["outcome"])
        e = res["endpoint_dPDMS_points"]
        self.emit("ENDPOINT", f"dPDMS {e['point']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}] points vs P2's pick; seed range {res['per_seed_points']['range']:.4f}; safety {json.dumps(res['safety_points'])}")
        k = res["contrast_O4_minus_C8"]["pooled"]
        self.emit("CONTRAST", f"O4 minus C8 {k['point']:+.4f} [{k['lo']:+.4f}, {k['hi']:+.4f}] points (descriptive, ungated; M x4, elites x4, widen x2 together); C1 read {c1['outcome']}")
        for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL"):
            self.emit("FAMILY", fam, json.dumps(res["families"][fam])[:600])
        self.emit("FAMILY", "STRATEGIC", res["STRATEGIC"])
        k2 = res["c2_config"]
        self.emit("C2CONFIG", k2["config"] or "NONE", f"row {k2['row']}: {k2['why']}")
        self.emit("NEXT", res["next_arm"])
        self.emit("STAMPS", res["stamps"])

    def run(self):
        try:
            self.emit("START")
            pins = self.check_seal()
            self.check_amendment(pins)
            self.check_data(pins)
            c1 = self.check_c1_done(pins)
            self.check_disk()
            self.check_bundle()
            self.subid_gate(pins, c1)
            self.mint_gates(pins)
            merged = self.run_search("C1", pins, 905)
            self.emit("SEARCH_DONE")
            self.state("SEARCH_DONE")
            self.selfrep_gate(pins, merged)
            labels = self.run_labels(merged)
            self.run_analysis(merged, labels, pins, c1)
            return 0
        except CH.StopChain as e:
            self.emit("STOP", e.reason, e.detail)
            return e.code
        except Exception as e:  # a crash is a stop, not a pass
            self.emit("STOP", "CRASH", f"{type(e).__name__}: {e}")
            return 9


def make_cfg(pkg=SPC_PKG, work=DOSE_WORK, c1_work=CH.WORK, dose_pkg=None, config=CONFIG_O4, **kw):
    """the dose arm's Cfg: C1's pkg (its seal) and C1's defaults, the O4 config, the dose work dir; the C1 work dir holds the bundle it reuses"""
    cfg = CH.Cfg(pkg, work=work, config=config, min_free_gib=kw.pop("min_free_gib", DOSE_MIN_FREE_GIB), **kw)       # @CFGO4
    cfg.c1_work = c1_work
    cfg.dose_pkg = dose_pkg or os.path.dirname(HERE)
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=DOSE_WORK)
    ap.add_argument("--c1-work", default=CH.WORK)
    ap.add_argument("--dose-pkg", default=os.path.dirname(HERE))
    ap.add_argument("--dry-run", action="store_true", help="seal + amendment + data + C1-done + disk + bundle checks only; no child process")
    a = ap.parse_args(argv)
    ch = DoseChain(make_cfg(work=a.work, c1_work=a.c1_work, dose_pkg=a.dose_pkg))
    if a.dry_run:
        try:
            pins = ch.check_seal()
            ch.check_amendment(pins)
            ch.check_data(pins)
            ch.check_c1_done(pins)
            ch.check_disk()
            ch.check_bundle()
            ch.emit("DRYRUN_OK")
            return 0
        except CH.StopChain as e:
            ch.emit("STOP", e.reason, e.detail)
            return e.code
    return ch.run()


if __name__ == "__main__":
    sys.exit(main())
