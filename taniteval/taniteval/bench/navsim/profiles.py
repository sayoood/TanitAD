"""NavSim v2 runtime + split profiles — PROMOTED from E1 (``run_warmup_reference.sh``,
``run_navhard.sh``) and E2 (``score_arm.py`` ``ENV`` / ``SPLITS``), with the navhard cache pointed
at the agent-free waiter's C: build (``navhard_after_a8.sh`` → ``run_navhard_c1.sh``).

⛔ Every path here is a MEASURED fact of this dev box (2026-09-19), not a setting to tune:
* ``C:/Users/Admin/navsim-crun`` — E1's byte-verified C: RUNTIME (venv reinstalled offline, devkit
  copies blob-verified, data sha256-verified). The D:-resident venv is I/O-starved (E1: ``import
  numpy`` 23,350 ms vs 324 ms).
* ``C:/Users/Admin/navsim`` is a JUNCTION to ``D:/Archive/devbox-C/navsim`` — so the warmup metric
  cache (E1's, ``CACHE_DONE.json``) is read from D:; outputs of the suite go to C:
  (``EXP_ROOT``), never to D: (A7/A8 read D:; Master Mind request 2026-09-19).
* the MetricCacheLoader Windows defect (E1 finding 5) is fixed twice: in the C: devkit COPY by E1's
  ``apply_dataloader_patch.py`` (for process pools) and in-process by ``navsim_win.py
  --patch-loader``; both are recorded per run in ``devkit.patches``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

CR = Path("C:/Users/Admin/navsim-crun")
NAVSIM_JUNCTION = Path("C:/Users/Admin/navsim")          # -> D:/Archive/devbox-C/navsim
PY = CR / "venv" / "Scripts" / "python.exe"
DEVKIT = CR / "devkit"
NUPLAN_DEVKIT = CR / "nuplan-devkit"
DEVKIT_SIDE = Path(__file__).resolve().parent / "devkit_side"
WRAPPER = DEVKIT_SIDE / "navsim_win.py"
EXPORT_SCRIPT = DEVKIT_SIDE / "export_agent_inputs.py"
EXP_ROOT = CR / "exp" / "tanitad_bench"
TTS = DEVKIT / "navsim" / "planning" / "script" / "config" / "common" / "train_test_split"
DEVKIT_REPO = "autonomousvision/navsim"
DEVKIT_SHA = "0a380a9063d7162ec93d0f51e9990ebac585f720"
DEVKIT_GIT_SOURCE = Path("D:/Archive/devbox-C/navsim/devkit")   # the checkout the C: copy was verified against
DEVKIT_PIN_NOTE = ("autonomousvision/navsim@0a380a9 (2025-10-27; post-#151 fix, MEASURED by E1: README "
                   "changelog 2025/09/29 + pdm_score.py:194-219)")


#: the devkit's navsim_logs/test on C: holds only the 76 logs navhard needs (E1's mirror); the FULL
#: OpenScene test set (147 logs, every navtest log among them) is on D:. ⛔ W8 2026-09-26: the devkit's
#: ``filter_scenes`` (dataloader.py:34-36) iterates ``data_path.iterdir()`` and keeps the files whose
#: name is in ``scene_filter.log_names`` -- a log that is ABSENT is silently skipped, so pointing a
#: navtest run at the C: directory scores navhard's subset and reports success. Every profile names
#: its log directory, and preflight checks EVERY log the split needs is in it.
LOGS_C_NAVHARD = CR / "data" / "openscene" / "navsim_logs" / "test"
LOGS_D_FULL = Path("D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test")
#: the runner each stage count MUST use (navsim_win.py --script keys). ⛔ W8: a single-stage split
#: scored by the two-stage runner writes the THREE ``extended_pdm_score_*`` rows over a split that has
#: no stage 2 -- a number with the wrong definition, not a crash -- so the pairing is asserted, never
#: assumed (:func:`runner_for`, pinned by test_bench_navsim_navtest_single_stage.py).
RUNNER_FOR_STAGES = {1: "pdm_score_one_stage", 2: "pdm_score"}
#: W2's token -> log_name map for navtest (12,146 tokens, 136 logs, built on devkit 0a380a9): the
#: cluster unit of the interval AND the per-log expected counts the cache builder resumes against.
NAVTEST_CLUSTER_MAP = Path(__file__).resolve().parents[4] / (
    "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json")


@dataclass(frozen=True)
class SplitProfile:
    name: str
    protocol: str
    n_stage1: int
    n_stage2: int
    n_logs: int
    cache: Path
    cache_verified_by: str            # "CACHE_DONE.json" | "counts"
    syn_sensors: Path | None          # None on a single-stage split (no synthetic scenes exist)
    syn_scenes: Path | None
    no_interval_reason: str | None = None
    notes: tuple = field(default_factory=tuple)
    # ---- W8 2026-09-26: single-stage splits (navtest) -------------------------------------- #
    stages: int = 2                   # 1 = ONE-STAGE runner, original log frames only; 2 = pseudo-sim
    devkit_split: str | None = None   # train_test_split=<this> (None -> name); navtest_single_stage -> navtest
    logs_dir: Path = LOGS_C_NAVHARD   # the navsim_log_path the runner is pointed at
    traffic_agents: str | None = None  # one-stage runner only: "non_reactive" | "reactive" (passed EXPLICITLY)
    cluster_map: Path | None = None   # token -> log_name (single-stage interval + cache resume)

    @property
    def n_scenes(self) -> int:
        return self.n_stage1 + self.n_stage2

    @property
    def tts(self) -> str:
        """The devkit's ``train_test_split`` name (the suite's split name can differ from it)."""
        return self.devkit_split or self.name

    @property
    def runner(self) -> str:
        return runner_for(self)


SPLITS = {
    "warmup_two_stage": SplitProfile(
        name="warmup_two_stage", protocol="EPDMS_v2_warmup_two_stage", n_stage1=16, n_stage2=204, n_logs=7,
        cache=NAVSIM_JUNCTION / "exp" / "metric_cache_warmup_two_stage", cache_verified_by="CACHE_DONE.json",
        syn_sensors=NAVSIM_JUNCTION / "data" / "openscene" / "warmup_two_stage" / "sensor_blobs",
        syn_scenes=CR / "data" / "openscene" / "warmup_two_stage" / "synthetic_scene_pickles",
        no_interval_reason=("⛔ warmup NEVER carries a CI — D-BENCH-PORT (PI approval 2026-08-29): 7 log groups < "
                            "the RG-14 floor of 8 (products/P7-TanitEval/RELEASE_GATE.md); the 7 is re-read from "
                            "scene_filter/warmup_two_stage.yaml log_names on every run."),
        notes=("stage 1 has NO camera frames on this box (E2: 0/192 original jpgs) — a camera arm's stage 1 "
               "is a declared CV stand-in and its two-stage EPDMS is UNDEFINED",
               "EP == 1 for every proposal when the best compliant progress <= 5 m (pdm_scorer.py:231-236): "
               "STOP beats CV on warmup (E2: 0.3009 vs 0.1854)")),
    "navhard_two_stage": SplitProfile(
        name="navhard_two_stage", protocol="EPDMS_v2_navhard_two_stage", n_stage1=450, n_stage2=5462, n_logs=76,
        cache=CR / "exp" / "metric_cache_navhard_two_stage", cache_verified_by="counts",
        syn_sensors=CR / "data" / "openscene" / "navhard_two_stage" / "sensor_blobs",
        syn_scenes=CR / "data" / "openscene" / "navhard_two_stage" / "synthetic_scene_pickles",
        notes=("the metric cache is built by the agent-free waiter after A8 "
               "(FlyWheels/…/2026-09-19-navhard-download/code/navhard_after_a8.sh); the suite never rebuilds it",)),
    # ⭐ W8 2026-09-26 (PI: "wire the navtest single-stage split"). NAVSIM v2 EPDMS on navtest WITHOUT
    # pseudo-simulation: the devkit's ONE-STAGE runner (run_pdm_score_one_stage.py) over the 12,146
    # original navtest frames. The method papers' protocol (P4, NAVSIM_PROTOCOL.md §6.4) -- the
    # benchmark authors discourage it ([N2] §4.2); it is on the leaderboard for orientation, and it is
    # the only NavSim column where every token has BOTH a logged human future AND (via W3's bank) frames.
    "navtest_single_stage": SplitProfile(
        name="navtest_single_stage", protocol="EPDMS_v2_navtest_single_stage", n_stage1=12146, n_stage2=0,
        n_logs=136, cache=CR / "exp" / "metric_cache_navtest_v2", cache_verified_by="CACHE_DONE.json",
        syn_sensors=None, syn_scenes=None,
        stages=1, devkit_split="navtest", logs_dir=LOGS_D_FULL, traffic_agents="non_reactive",
        cluster_map=NAVTEST_CLUSTER_MAP,
        notes=("ONE-STAGE runner (run_pdm_score_one_stage.py @0a380a9): one query per original log frame; the "
               "summary row is `average_all_frames` (a plain skipna mean over tokens, L293) -- there is no "
               "stage 2, no synthetic scene and no mapping",
               "background traffic NON-REACTIVE (traffic_agents=non_reactive: log replay, 'identical to NAVSIM "
               "v1', docs/traffic_agents.md) -- the devkit DEFAULT for this runner (default_common.yaml), "
               "passed explicitly; E1's navhard N2b used traffic_agents=reactive, a DIFFERENT setting",
               "logs are read from D: (the full 147-log OpenScene test set); the C: navsim_logs/test holds only "
               "navhard's 76 and would silently score a subset (dataloader.py:34-36)",
               "EPDMS is POST-#151 (devkit 0a380a9); published rows of this column mix pre/post/unverified "
               "implementations (published_results.json harness.fix151) -- comparable ONLY to fix151=post")),
}


class Refusal(RuntimeError):
    """A precondition failed; the suite refuses before any compute (fail loud)."""


def runner_for(prof: "SplitProfile") -> str:
    """The devkit runner (``navsim_win.py --script`` key) a split MUST be scored with. Raises
    :class:`Refusal` for a stage count that has no runner -- never a default."""
    r = RUNNER_FOR_STAGES.get(int(prof.stages))
    if r is None:
        raise Refusal(f"{prof.name}: stages={prof.stages!r} has no runner (known {sorted(RUNNER_FOR_STAGES)})")
    return r


def check_runner(prof: "SplitProfile", script: str) -> str:
    """⛔ Refuse a runner that is not the one the split's stage count requires (W8 2026-09-26): the
    two-stage runner on a single-stage split does not crash -- it writes a three-row two-stage summary
    over a split with no stage 2. Returns ``script`` when it is the right one."""
    want = runner_for(prof)
    if script != want:
        raise Refusal(f"{prof.name} is a {prof.stages}-stage split and must be scored with --script {want}; "
                      f"got {script!r} (the two runners write DIFFERENT summary rows over different scene sets)")
    return script


#: ⛔ THE SUMMARY-ROW SHAPE BELONGS TO THE PROTOCOL, NOT TO THE CHECKER (W3, 2026-09-20). The three
#: ``extended_pdm_score_*`` rows are the v2 TWO-STAGE shape; NAVSIM v1.1 writes ONE row (``average``),
#: and the v2 ONE-STAGE runner writes ``average_all_frames`` (MEASURED by E1 on warmup A1b/A2b).
#: A protocol that is not declared here is REFUSED rather than assumed.
SUMMARY_ROW_SHAPE = {
    "EPDMS_v2_warmup_two_stage": {"rows": ("extended_pdm_score_stage_one", "extended_pdm_score_stage_two",
                                           "extended_pdm_score_combined"), "headline_row": "extended_pdm_score_combined",
                                  "evidence": "MEASURED (E1/E2/W1 runs on this box)"},
    "EPDMS_v2_navhard_two_stage": {"rows": ("extended_pdm_score_stage_one", "extended_pdm_score_stage_two",
                                            "extended_pdm_score_combined"), "headline_row": "extended_pdm_score_combined",
                                   "evidence": "same runner as warmup (run_pdm_score.py two-stage)"},
    "EPDMS_v2_private_test_hard_two_stage": {"rows": ("extended_pdm_score_stage_one", "extended_pdm_score_stage_two",
                                                      "extended_pdm_score_combined"),
                                             "headline_row": "extended_pdm_score_combined",
                                             "evidence": "same runner as warmup (run_pdm_score.py two-stage)"},
    "EPDMS_v2_navtest_single_stage": {"rows": ("average_all_frames",), "headline_row": "average_all_frames",
                                      "evidence": "MEASURED by E1 on the v2 ONE-STAGE runner (A1b/A2b warmup CSVs)"},
    "PDMS_v1_navtest": {"rows": ("average",), "headline_row": "average",
                        "evidence": ("MEASURED (W3, 2026-09-20), now at FULL SCALE: all three 12,147-row CSVs "
                                     "carry 12,146 hex token rows + EXACTLY ONE summary row, token verbatim "
                                     "'average', valid = True, on THREE independent arms (CV/STOP/HUMAN), with a "
                                     "per-file sha256. ⭐ This is the original 20-token smoke read REPRODUCED at "
                                     "12,146 tokens — the stamp did not change, its evidence got stronger, which "
                                     "is the outcome a pre-registered scale-up is supposed to have. Corroborated "
                                     "from the code that writes it: v1.1 @3e8291bf run_pdm_score.py:144-147, "
                                     "average_row[\"token\"] = \"average\" — independent of token count. Banked: "
                                     "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
                                     "W1_summary_row_shape_navsim_v1_FULLSPLIT.json (full split) and "
                                     "…/W1_summary_row_shape_navsim_v1.json (the original smoke read)"),
                        "reader_notes": (
                            "the CSV's first column is the UNNAMED pandas index (the header starts `,token,valid,…`); "
                            "the summary row's `valid` cell is the AND over ALL token rows (run_pdm_score.py:146) — a "
                            "GUARD SIGNAL, not decoration: one invalid token turns it False")},
}


def summary_row_shape(protocol: str) -> dict:
    """-> ``{rows, headline_row, n, evidence}`` for a protocol. Raises :class:`Refusal` for an
    undeclared protocol: a count assumed by the checker is how a v1 CSV fails a v2 guard."""
    shape = SUMMARY_ROW_SHAPE.get(protocol)
    if shape is None:
        raise Refusal(f"protocol {protocol!r} does not declare its summary-row shape "
                      f"(known: {sorted(SUMMARY_ROW_SHAPE)}) — refusing to assume one")
    return {**shape, "n": len(shape["rows"]), "protocol": protocol}


def scorer_env(exp_root: Path) -> dict:
    """The child environment for the NavSim venv. E2 ``score_arm.ENV`` + E1 ``run_*.sh`` exports.
    The caller's PYTHONPATH/PYTHONHOME/VIRTUAL_ENV are DROPPED (the TanitAD stack must never shadow a
    devkit module); only ``DEVKIT_SIDE`` is on the child's path (the seam agent's import target)."""
    env = {k: v for k, v in os.environ.items()
           if k.upper() not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP", "CONDA_PREFIX")}
    fixed = scorer_env_fixed(exp_root)
    env.update(fixed)
    return env


def scorer_env_fixed(exp_root: Path) -> dict:
    return {
        "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
        "NUPLAN_MAPS_ROOT": str(CR / "data" / "maps").replace(os.sep, "/"),
        "NAVSIM_EXP_ROOT": str(exp_root).replace(os.sep, "/"),
        "NAVSIM_DEVKIT_ROOT": str(DEVKIT).replace(os.sep, "/"),
        "OPENSCENE_DATA_ROOT": str(CR / "data" / "openscene").replace(os.sep, "/"),
        "PYTHONHASHSEED": "1",               # E1/E2: fixed seed => byte-identical CSVs (E1 C7)
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "CUDA_VISIBLE_DEVICES": "-1",        # the scorer never touches a GPU (Windows drops EMPTY vars)
        "PYTHONPATH": str(DEVKIT_SIDE).replace(os.sep, "/"),
    }


def git_blob(path: Path) -> str | None:
    try:
        b = Path(path).read_bytes()
    except OSError:
        return None
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def git_blob_normalised(path: Path) -> str | None:
    """``git hash-object``'s id: the blob AFTER git's text/eol conversion.

    ⚠️ PAIRED WITH :func:`git_blob` ON PURPOSE, because the two answer DIFFERENT questions and the
    asymmetry is ONE-DIRECTIONAL:

    * :func:`git_blob` (RAW bytes) answers *"did these bytes change at all?"* and is the **STRONGER**
      probe for that question;
    * this one answers *"would git store a different blob?"* and is **BLIND to a pure line-ending
      change** -- a CRLF->LF conversion leaves it identical while every byte on disk moved.

    MEASURED 2026-09-20 on the devkit's IDM file (100 % CRLF, 166/166): raw ``b95bcc7f…`` vs
    normalised ``214ef5ee…``. E1 met the same split from the other side, verifying the C: devkit copy
    against git blobs: two non-runtime files differed under RAW bytes and read CLEAN through git's
    filters. ⇒ record BOTH, labelled, or a correct file reads as a mismatch and a changed one reads
    as clean.
    """
    try:
        r = subprocess.run(["git", "-c", "safe.directory=*", "hash-object", str(path)],
                           capture_output=True, text=True, timeout=60)
        out = r.stdout.strip()
        return out if re.fullmatch(r"[0-9a-f]{40}", out) else None
    except Exception:                                                    # noqa: BLE001
        return None


#: what each hash field is, and what it CANNOT see. Carried in every patch entry so the reader never
#: has to guess which probe produced a number (⛔ the mismatch that cost a false alarm on 2026-09-20).
HASH_BASIS_NOTE = {
    "raw_bytes_git_blob_now": ("sha1('blob <len>\0' + RAW bytes), no eol conversion. STRONGER for "
                               "'did this file change at all'. This is what E1's apply_*_patch.py prints."),
    "normalised_git_blob_now": ("`git hash-object` = the blob AFTER git's text/eol conversion. "
                                "⛔ BLIND to a pure CRLF<->LF change. Quote it only against other "
                                "git-side ids, never against a raw-byte record."),
}


def _patch_hashes(path: Path) -> dict:
    return {"raw_bytes_git_blob_now": git_blob(path),
            "normalised_git_blob_now": git_blob_normalised(path),
            "hash_basis": HASH_BASIS_NOTE}


def devkit_patches() -> list:
    """The harness column (E1 RESULT §4) — with the CURRENT blobs of the files involved."""
    return [
        {"name": "navsim/common/dataclasses.py PosixPath unpickler (Windows)", "kind": "pre-existing",
         "evidence": "E1 RESULT.md §4 (blob 596cb7d in the source checkout)",
         **_patch_hashes(DEVKIT / "navsim" / "common" / "dataclasses.py")},
        {"name": "venv fcntl.py flock shim", "kind": "pre-existing",
         "evidence": "E1 RESULT.md §4 (sha256 75184a48...)",
         **_patch_hashes(CR / "venv" / "Lib" / "site-packages" / "fcntl.py")},
        {"name": "nuplan-devkit setup.py packaging-only change", "kind": "pre-existing",
         "evidence": "E1 RESULT.md §4"},
        {"name": "C: copy navsim/common/dataloader.py: MetricCacheLoader token split on either separator",
         "kind": "harness",
         "evidence": ("E1 code/patches/apply_dataloader_patch.py, applied 2026-09-19 13:04 "
                      "(raw/navhard/dataloader_copy_patch.json); needed for process-pool scoring"),
         **_patch_hashes(DEVKIT / "navsim" / "common" / "dataloader.py")},
        {"name": ("C: copy navsim/planning/simulation/observation/navsim_idm/navsim_idm_agent_manager.py: "
                  "IDM degenerate-path assert (a fully consumed baseline)"),
         "kind": "harness",
         "evidence": ("E1 code/patches/apply_idm_degenerate_path_patch.py, applied 2026-09-20 08:58 "
                      "(raw/navhard/idm_degenerate_path_patch.json). MECHANISM: a fully consumed IDM "
                      "baseline yields a ZERO-LENGTH path_to_go whose flat-cap buffer is an EMPTY polygon "
                      "under shapely 2.0.7, so the agent cannot intersect its own path and the assert at "
                      "navsim_idm_agent_manager.py:84 fires. Scope: the ASSERT CONDITION ONLY - it falls "
                      "through to the devkit's own free-road branch. Verified by E1 effective 3/3 and "
                      "INERT 4/4 (healthy tokens bit-identical)."),
         "why_it_is_a_precondition_not_a_footnote": ("MEASURED by E1: 166 of 5,462 stage-2 tokens (0 of 450 "
                                                     "stage-1) hit it, and ONE assert takes the whole run "
                                                     "down. Without this patch the navhard numbers DO NOT "
                                                     "EXIST - it is a precondition of the number, not a "
                                                     "caveat on it."),
         "blob_pre_patch": "1924b7c1e9ea2bd0436d475741e6b2e3a2f7b652",
         "blob_post_patch_e1": "b95bcc7f62246017ea9ab5dca92383c6b50d234b",
         "reader_note": ("⚠ VERIFY THIS ONE WITH RAW BYTES, NOT `git hash-object`. The file is 100% CRLF "
                         "(166/166), and `git hash-object` applies the text/eol conversion, returning "
                         "214ef5ee39944c3584cb16eaaffd53853fc9db85 for these same bytes - a DIFFERENT number "
                         "that looks exactly like a mismatch with E1's record. git_blob() below reads raw "
                         "bytes and agrees with E1 (MEASURED 2026-09-20). A pure line-ending change is "
                         "invisible to the normalising hash and visible to this one."),
         **_patch_hashes(DEVKIT / "navsim" / "planning" / "simulation" / "observation"
                         / "navsim_idm" / "navsim_idm_agent_manager.py")},
        {"name": "MetricCacheLoader._load_metric_cache_paths monkeypatch (navsim_win.py --patch-loader)",
         "kind": "in-process",
         "evidence": "E1 code/navsim_win.py, pinned by test_navsim_win_patch.py; separator only"},
    ]


def devkit_sha_measured() -> dict:
    """The SHA of the SOURCE checkout the C: copy was verified from (the copy has no .git)."""
    try:
        r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(DEVKIT_GIT_SOURCE), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=60)
        sha = r.stdout.strip()
        return {"sha": sha if re.fullmatch(r"[0-9a-f]{40}", sha) else "UNAVAILABLE",
                "matches_pin": sha == DEVKIT_SHA, "source": str(DEVKIT_GIT_SOURCE)}
    except Exception as e:                                               # noqa: BLE001
        return {"sha": "UNAVAILABLE", "matches_pin": False, "reason": f"{type(e).__name__}: {e}"}


def read_split_yaml(split: str) -> dict:
    """``{stage_one: [...], stage_two: [...], log_names: [...], mapping: [...]}`` from the devkit's own yaml.

    ⭐ W8 2026-09-26: a SINGLE-STAGE profile (``SPLITS[split].stages == 1``) is read from its DEVKIT split's
    scene filter (``navtest_single_stage`` -> ``scene_filter/navtest.yaml``) with ``stage_two`` and
    ``mapping`` EMPTY -- the split has no stage 2, and a two-stage key found in that filter is REFUSED."""
    prof = SPLITS.get(split)
    if prof is not None and prof.stages == 1:
        return read_single_stage_yaml(prof)
    import yaml
    sf = yaml.safe_load(open(TTS / "scene_filter" / f"{split}.yaml", encoding="utf-8"))
    tts = yaml.safe_load(open(TTS / f"{split}.yaml", encoding="utf-8"))
    return {"stage_one": list(sf["tokens"]), "stage_two": list(sf["reactive_synthetic_initial_tokens"]),
            "log_names": list(sf["log_names"]),
            "mapping": [[m[0], m[1], [list(p) for p in m[2]]] for m in tts["reactive_all_mapping"]]}


def read_single_stage_yaml(prof: "SplitProfile") -> dict:
    """The devkit's own scene filter of a single-stage split. Refuses a filter that carries synthetic
    (stage-2) content: scoring it with the one-stage runner would silently drop that content."""
    import yaml
    sf = yaml.safe_load(open(TTS / "scene_filter" / f"{prof.tts}.yaml", encoding="utf-8"))
    two_stage_keys = [k for k in ("reactive_synthetic_initial_tokens", "non_reactive_synthetic_initial_tokens",
                                  "synthetic_scene_tokens", "all_mapping") if sf.get(k)]
    if sf.get("include_synthetic_scenes") or two_stage_keys:
        raise Refusal(f"{prof.name}: scene_filter/{prof.tts}.yaml carries two-stage content "
                      f"({two_stage_keys or 'include_synthetic_scenes'}) — not a single-stage split")
    return {"stage_one": list(sf["tokens"]), "stage_two": [], "log_names": list(sf["log_names"]), "mapping": []}


def token_to_log(prof: "SplitProfile") -> dict:
    """``{token: log_name}`` for a single-stage split (W2's cluster map), VERIFIED against the devkit yaml:
    a map whose token set differs from the split's is REFUSED (a cluster is never guessed)."""
    if prof.cluster_map is None or not Path(prof.cluster_map).exists():
        raise Refusal(f"{prof.name}: no token -> log_name map at {prof.cluster_map}")
    m = json.loads(Path(prof.cluster_map).read_text(encoding="utf-8"))
    t2l = {str(k): str(v) for k, v in m["token_to_log_name"].items()}
    y = read_split_yaml(prof.name)
    if set(t2l) != set(y["stage_one"]) or set(t2l.values()) != set(y["log_names"]):
        raise Refusal(f"{prof.name}: the cluster map {prof.cluster_map} does not cover exactly the split "
                      f"({len(t2l)} tokens / {len(set(t2l.values()))} logs vs yaml {len(y['stage_one'])} / "
                      f"{len(y['log_names'])})")
    return t2l


def read_tokens_file(path) -> list:
    """A token SUBSET for a single-stage run: a JSON list, or ``{"tokens": [...]}`` (W3's format)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    toks = d.get("tokens") if isinstance(d, dict) else d
    if not isinstance(toks, list) or not toks or not all(isinstance(t, str) for t in toks):
        raise Refusal(f"--tokens-file {path}: no token list (want a JSON list or {{'tokens': [...]}})")
    if len(set(toks)) != len(toks):
        raise Refusal(f"--tokens-file {path}: {len(toks) - len(set(toks))} duplicate token(s)")
    return list(toks)


def verify_cache(prof: SplitProfile, expected: set | None = None) -> dict:
    """Content-level preconditions on the metric cache. Raises :class:`Refusal`.

    ``expected`` (single-stage SUBSET runs only, W8): the tokens this run will score. The cache must
    hold every one of them; the full-split count/equality rule applies when it is None."""
    meta = prof.cache / "metadata"
    if not meta.is_dir():
        raise Refusal(f"no metric cache at {prof.cache} (metadata/ absent)")
    csvs = sorted(p for p in meta.iterdir() if ".csv" in p.name)
    if not csvs:
        raise Refusal(f"metric cache {prof.cache}: no metadata CSV")
    if prof.stages == 1 and len(csvs) != 1:
        # MetricCacheLoader reads the FIRST csv ONLY (dataloader.py:313 @0a380a9): a second one is silently ignored
        raise Refusal(f"metric cache {prof.cache}: {len(csvs)} metadata CSVs — the devkit loader reads only the "
                      f"first ({csvs[0].name}); refuse rather than guess which one is complete")
    rows = [ln for ln in csvs[0].read_text(encoding="utf-8", errors="replace").splitlines()[1:] if ln.strip()]
    toks = {re.split(r"[\\/]", r.strip())[-2] for r in rows}
    res = {"cache": str(prof.cache).replace(os.sep, "/"), "metadata_csv": csvs[0].name,
           "n_metadata_rows": len(rows), "n_tokens": len(toks), "verified_by": prof.cache_verified_by}
    if prof.cache_verified_by == "CACHE_DONE.json":
        mk = prof.cache / "CACHE_DONE.json"
        if not mk.exists():
            raise Refusal(f"{mk} absent — the cache was never verified by content (E1 verify_cache.py)")
        m = json.loads(mk.read_text(encoding="utf-8"))
        res["cache_done"] = {k: m.get(k) for k in ("n_cached", "n_expected_from_yaml", "token_sets_equal_yaml",
                                                   "scene_types_match_yaml", "written_local", "token_set",
                                                   "tokens_sha256", "manifest_sha256")}
        certified = m.get("token_sets_equal_yaml") or (expected is not None and m.get("token_set") == "SUBSET"
                                                        and m.get("token_sets_equal_expected"))
        if not (certified and m.get("scene_types_match_yaml")):
            raise Refusal(f"{mk} does not certify the token sets / scene types: {res['cache_done']}")
    if expected is not None:
        miss = sorted(set(expected) - toks)
        if miss:
            raise Refusal(f"metric cache {prof.cache}: {len(miss)} of the {len(expected)} requested tokens are not "
                          f"cached (first {miss[0]}) — the runner would silently score only the intersection "
                          "(run_pdm_score_one_stage.py:250)")
        res["n_requested_tokens_cached"] = len(expected)
        return res
    if len(rows) != prof.n_scenes or len(toks) != prof.n_scenes:
        raise Refusal(f"metric cache {prof.cache}: {len(rows)} rows / {len(toks)} tokens != expected {prof.n_scenes} "
                      f"({prof.n_stage1} + {prof.n_stage2}) — incomplete or foreign cache")
    return res


def preflight(prof: SplitProfile, *, need_logs: bool = True, import_probe: bool = True,
              tokens: list | None = None) -> dict:
    """Every precondition, checked BEFORE any compute. Returns the evidence; raises :class:`Refusal`.

    ``tokens`` (single-stage only): a SUBSET of the split to score (W8 smoke / sub-split runs)."""
    ev = {"split": prof.name}
    need = [("navsim venv python", PY), ("wrapper", WRAPPER), ("devkit", DEVKIT),
            ("scene_filter yaml", TTS / "scene_filter" / f"{prof.tts}.yaml")]
    if prof.stages == 2:
        need.insert(3, ("synthetic scene pickles", prof.syn_scenes))
    for what, p in need:
        if p is None or not Path(p).exists():
            raise Refusal(f"{what} missing: {p}")
    if tokens is not None and prof.stages != 1:
        raise Refusal(f"{prof.name}: a token subset is only defined on a single-stage split (a two-stage "
                      "split's stage 2 is tied to its stage 1 by the mapping)")
    y = read_split_yaml(prof.name)
    ev["yaml_counts"] = {"stage_one": len(y["stage_one"]), "stage_two": len(y["stage_two"]),
                         "log_names": len(y["log_names"]), "mapping": len(y["mapping"])}
    want = {"stage_one": prof.n_stage1, "stage_two": prof.n_stage2, "log_names": prof.n_logs}
    bad = {k: (ev["yaml_counts"][k], v) for k, v in want.items() if ev["yaml_counts"][k] != v}
    if bad:
        raise Refusal(f"{prof.name}: yaml counts disagree with the profile (yaml, profile): {bad}")
    logs_needed = list(y["log_names"])
    if tokens is not None:
        foreign = sorted(set(tokens) - set(y["stage_one"]))
        if foreign:
            raise Refusal(f"{len(foreign)} requested token(s) are not {prof.tts} tokens (first {foreign[0]})")
        t2l = token_to_log(prof)
        logs_needed = sorted({t2l[t] for t in tokens})
        ev["subset"] = {"n_tokens": len(tokens), "n_logs": len(logs_needed)}
    ev["cache"] = verify_cache(prof, expected=set(tokens) if tokens is not None else None)
    if need_logs:
        logs_dir = prof.logs_dir
        missing = [ln for ln in logs_needed if not (logs_dir / f"{ln}.pkl").exists()]
        if missing:
            raise Refusal(f"{len(missing)}/{len(logs_needed)} log pickles missing under {logs_dir} "
                          f"(first: {missing[0]}) — the runner would silently skip them "
                          "(dataloader.py:34-36); run E1's mirror step first")
        ev["logs_present"] = len(logs_needed)
        ev["logs_dir"] = str(logs_dir).replace(os.sep, "/")
    if import_probe:
        r = subprocess.run([str(PY), "-c", "import navsim, nuplan, hydra, psutil; print(navsim.__file__)"],
                           capture_output=True, text=True, timeout=300, env=scorer_env(EXP_ROOT))
        where = r.stdout.strip().replace("\\", "/")
        if r.returncode != 0 or not where.lower().startswith(str(DEVKIT).replace(os.sep, "/").lower()):
            raise Refusal(f"navsim venv import probe failed or imported a foreign tree: rc={r.returncode} "
                          f"navsim={where!r} stderr={r.stderr.strip()[-400:]!r}")
        ev["import_probe"] = {"navsim": where}
    return ev
