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


@dataclass(frozen=True)
class SplitProfile:
    name: str
    protocol: str
    n_stage1: int
    n_stage2: int
    n_logs: int
    cache: Path
    cache_verified_by: str            # "CACHE_DONE.json" | "counts"
    syn_sensors: Path
    syn_scenes: Path
    no_interval_reason: str | None = None
    notes: tuple = field(default_factory=tuple)

    @property
    def n_scenes(self) -> int:
        return self.n_stage1 + self.n_stage2


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
}


class Refusal(RuntimeError):
    """A precondition failed; the suite refuses before any compute (fail loud)."""


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
    """``{stage_one: [...], stage_two: [...], log_names: [...], mapping: [...]}`` from the devkit's own yaml."""
    import yaml
    sf = yaml.safe_load(open(TTS / "scene_filter" / f"{split}.yaml", encoding="utf-8"))
    tts = yaml.safe_load(open(TTS / f"{split}.yaml", encoding="utf-8"))
    return {"stage_one": list(sf["tokens"]), "stage_two": list(sf["reactive_synthetic_initial_tokens"]),
            "log_names": list(sf["log_names"]),
            "mapping": [[m[0], m[1], [list(p) for p in m[2]]] for m in tts["reactive_all_mapping"]]}


def verify_cache(prof: SplitProfile) -> dict:
    """Content-level preconditions on the metric cache. Raises :class:`Refusal`."""
    meta = prof.cache / "metadata"
    if not meta.is_dir():
        raise Refusal(f"no metric cache at {prof.cache} (metadata/ absent)")
    csvs = sorted(p for p in meta.iterdir() if ".csv" in p.name)
    if not csvs:
        raise Refusal(f"metric cache {prof.cache}: no metadata CSV")
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
                                                   "scene_types_match_yaml", "written_local")}
        if not (m.get("token_sets_equal_yaml") and m.get("scene_types_match_yaml")):
            raise Refusal(f"{mk} does not certify the token sets / scene types: {res['cache_done']}")
    if len(rows) != prof.n_scenes or len(toks) != prof.n_scenes:
        raise Refusal(f"metric cache {prof.cache}: {len(rows)} rows / {len(toks)} tokens != expected {prof.n_scenes} "
                      f"({prof.n_stage1} + {prof.n_stage2}) — incomplete or foreign cache")
    return res


def preflight(prof: SplitProfile, *, need_logs: bool = True, import_probe: bool = True) -> dict:
    """Every precondition, checked BEFORE any compute. Returns the evidence; raises :class:`Refusal`."""
    ev = {"split": prof.name}
    for what, p in (("navsim venv python", PY), ("wrapper", WRAPPER), ("devkit", DEVKIT),
                    ("synthetic scene pickles", prof.syn_scenes), ("scene_filter yaml", TTS / "scene_filter" / f"{prof.name}.yaml")):
        if not Path(p).exists():
            raise Refusal(f"{what} missing: {p}")
    y = read_split_yaml(prof.name)
    ev["yaml_counts"] = {"stage_one": len(y["stage_one"]), "stage_two": len(y["stage_two"]),
                         "log_names": len(y["log_names"]), "mapping": len(y["mapping"])}
    want = {"stage_one": prof.n_stage1, "stage_two": prof.n_stage2, "log_names": prof.n_logs}
    bad = {k: (ev["yaml_counts"][k], v) for k, v in want.items() if ev["yaml_counts"][k] != v}
    if bad:
        raise Refusal(f"{prof.name}: yaml counts disagree with the profile (yaml, profile): {bad}")
    ev["cache"] = verify_cache(prof)
    if need_logs:
        logs_dir = CR / "data" / "openscene" / "navsim_logs" / "test"
        missing = [ln for ln in y["log_names"] if not (logs_dir / f"{ln}.pkl").exists()]
        if missing:
            raise Refusal(f"{len(missing)}/{len(y['log_names'])} stage-1 log pickles missing under {logs_dir} "
                          f"(first: {missing[0]}) — run E1's mirror step first")
        ev["logs_present"] = len(y["log_names"])
    if import_probe:
        r = subprocess.run([str(PY), "-c", "import navsim, nuplan, hydra, psutil; print(navsim.__file__)"],
                           capture_output=True, text=True, timeout=300, env=scorer_env(EXP_ROOT))
        where = r.stdout.strip().replace("\\", "/")
        if r.returncode != 0 or not where.lower().startswith(str(DEVKIT).replace(os.sep, "/").lower()):
            raise Refusal(f"navsim venv import probe failed or imported a foreign tree: rc={r.returncode} "
                          f"navsim={where!r} stderr={r.stderr.strip()[-400:]!r}")
        ev["import_probe"] = {"navsim": where}
    return ev
