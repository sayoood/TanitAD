"""Shared library for the Colab S2 label lab — auth, HF bank/resume, legs, fusion.

Both notebooks (`SAM3_BACKFILL_115.ipynb`, `STRATEGIC_LABEL_LAB.ipynb`) are thin
cell wrappers over this module, so the plumbing is testable OFF Colab (the CPU
smoke path runs these exact functions with stub GPU legs) and a fix lands in
one place.

Reuse contract (nothing re-invented):
  * VLM leg     — `stack/scripts/ph0_v2.py` (`ConstrainedVLM.ask`, `run_clip`,
                  the B1–B4 schemas); only the LOADER is swapped for unsloth.
  * SAM3 leg    — `stack/scripts/ph0_sam3.py` (`build_processor`,
                  `run_clip_frames`) with the per-clip block of its `main()`
                  mirrored here so the processor loads ONCE per session, and
                  ⛔ the clip count is ALWAYS explicit — the `--n` default of 4
                  is the measured root cause of the 115-clip gap
                  (AUG120_FUSION_RESULT.md §3).
  * ego leg     — `stack/scripts/ph0_pilot.py` engine A + `ph1_fuse.ego_from_npz`;
                  the yaw thresholds ported from
                  `stack/experiments/nurec-gsplat/strategic_gt.py` (25° / 150°).
  * fusion      — `stack/scripts/ph1_fuse.py` (`build_tracks`, `corroborate`,
                  `emit_vocab`, record shape `ph1-fused-v1`).
  * schema      — `colab/s2_schema.py` (PROVISIONAL; the one-file swap point).

⛔ HARD RULES enforced here:
  * The HF token is NEVER printed, logged, or written to disk/args. Colab
    Secrets first (`google.colab.userdata`), env second, Keys.txt read IN
    PLACE last (dev box only).
  * Banking is per clip, far-side verified by a byte round-trip (the silent-
    push-failure class), and every run starts by listing the far side —
    find-what-is-done-then-continue, never restart-from-zero.
  * Gap derivation COUNTS RECORDS, NOT FILES (C18): it reads each fused
    record's own `perception.absent`, then cross-checks two independent
    far-side sources and refuses on any disagreement.
"""
from __future__ import annotations

import gc
import io
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# repos (all datasets, all under the program's Sayood/ namespace)              #
# --------------------------------------------------------------------------- #
DS_LABELS = "Sayood/tanitad-ph0-aug120"            # v2 / sam3 / fused / ego npz
DS_VIDEO = "Sayood/tanitad-physicalai-w120-256x640cyl"   # <seg>/<clip>.v2ep.pt
DS_ALP = "Sayood/tanitad-alpamayo2-augmentation"   # records.parquet
DS_LAB = "Sayood/tanitad-s2-lab"                   # this lab's bank

FUSED_PREFIX = "fused_aug120/"
EGO_PREFIX = "bridged_w120train_2400/ego/"
SAM3_ABSENT_REASON = "AUG120_SAM3_STAGE_GAP"
BACKFILL_PREFIX = "sam3_backfill/"                 # in DS_LABELS (real runs)
#: ⛔ SCHEMA v2 GETS ITS OWN ADDRESS, AND THAT IS THE POINT — NOT TIDINESS.
#: The v1 prefix holds 83 records detected at `confidence_threshold=0.5` (the
#: vendor default nobody chose) plus 32 still carrying the C77 payload. v2 is
#: detected at **0.25** with contours and the scene channel. Writing v2 records
#: over v1 ones would make the corpus MIXED for the whole length of a run — and
#: free-Colab reclaimed the T4 three times during the last pass, so "the whole
#: length of a run" is not hypothetical. Every per-concept number computed over
#: a half-rewritten prefix would silently span two detection floors and be
#: unattributable, with nothing in the data to reveal it.
#: ⇒ v2 fills its own prefix; the v1 corpus stays intact and quotable (it is the
#: primary source of the concept-reliability study); consumers move when v2 is
#: complete, which is one decision at one moment instead of a race.
BACKFILL_V2_PREFIX = "sam3_backfill_v2/"           # in DS_LABELS (schema v2)
LAB_PREFIX = "lab_v0/"                             # in DS_LAB (real runs)
SMOKE_PREFIX = "smoke/"                            # in DS_LAB (smoke runs)

COLAB_REPO_ROOT = "/content/drive/MyDrive/SayBouBase/raw/Projects/TanitAD"
DEV_KEYS = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt"


# --------------------------------------------------------------------------- #
# environment                                                                  #
# --------------------------------------------------------------------------- #
def in_colab() -> bool:
    return "google.colab" in sys.modules or os.path.isdir("/content")


def repo_root() -> Path:
    """S2_REPO_ROOT env > Colab Drive mount > walk up from this file / cwd."""
    env = os.environ.get("S2_REPO_ROOT")
    if env and (Path(env) / "stack" / "scripts" / "ph0_v2.py").exists():
        return Path(env)
    if Path(COLAB_REPO_ROOT).exists():
        return Path(COLAB_REPO_ROOT)
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for cand in (start, *start.parents):
            if (cand / "stack" / "scripts" / "ph0_v2.py").exists():
                return cand
    raise SystemExit(
        "repo root not found — in Colab, run the Drive-mount cell first "
        f"(expects {COLAB_REPO_ROOT}); elsewhere set S2_REPO_ROOT")


def add_stack_paths(root: Path | None = None) -> Path:
    root = root or repo_root()
    for p in (root / "stack", root / "stack" / "scripts",
              Path(__file__).resolve().parent):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def mount_drive_if_colab() -> None:
    if in_colab() and not Path(COLAB_REPO_ROOT).exists():
        from google.colab import drive
        drive.mount("/content/drive")
        if not Path(COLAB_REPO_ROOT).exists():
            raise SystemExit(f"Drive mounted but {COLAB_REPO_ROOT} not found "
                             "— is this the PI's Google account?")


def pip_install_colab(smoke: bool) -> None:
    """Colab-only dependency install. ⛔ sam3's dependency closure includes
    torch — install it --no-deps so pip cannot silently replace Colab's torch
    with a wheel the driver cannot run (the uv-pip trap, CLAUDE.md), then
    verify with a REAL conv2d on CUDA, not with `import torch`."""
    if smoke or not in_colab():
        print("[setup] pip install skipped "
              f"(smoke={smoke}, colab={in_colab()})")
        return
    import subprocess
    pkgs = [
        ["unsloth"],
        ["lm-format-enforcer"],
        ["qwen-vl-utils"],
        ["--no-deps", "git+https://github.com/facebookresearch/sam3.git"],
        # sam3 arrives --no-deps (torch protection) so its torch-free runtime
        # deps come explicitly — MEASURED missing on the Colab T4 2026-08-16
        # (run 1: build_processor died ModuleNotFoundError iopath; timm/tqdm/
        # regex/typing_extensions/huggingface_hub were already present):
        ["iopath"],                              # pure-python dep closure
        ["--no-deps", "ftfy==6.1.1", "wcwidth"],  # sam3's CLIP text path
        ["--no-deps", "open_clip_torch"],       # ships the CLIP BPE vocab
        ["imageio", "imageio-ffmpeg"],
    ]
    for p in pkgs:
        rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q", *p])
        print(f"[setup] pip install {' '.join(p)} rc={rc}")
    assert_cuda_conv()


def assert_cuda_conv() -> None:
    """CLAUDE.md rule: verify torch with a real conv2d on CUDA — cuBLAS can
    succeed while cuDNN/conv is broken."""
    import torch
    if not torch.cuda.is_available():
        print("[setup] no CUDA — conv check skipped (CPU/smoke mode)")
        return
    x = torch.nn.functional.conv2d(torch.randn(1, 3, 16, 16, device="cuda"),
                                   torch.randn(4, 3, 3, 3, device="cuda"))
    assert x.shape == (1, 4, 14, 14)
    print(f"[setup] CUDA conv2d OK on {torch.cuda.get_device_name(0)}")


def gpu_mem_report(tag: str) -> float | None:
    """Print + return torch.cuda.max_memory_allocated (GB) and reset the peak.
    The ONLY admissible in-process memory probe (CLAUDE.md, Thor trap)."""
    try:
        import torch
        if not torch.cuda.is_available():
            print(f"[mem] {tag}: no CUDA (smoke/CPU mode)")
            return None
        gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[mem] {tag}: torch.cuda.max_memory_allocated = {gb:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
        return gb
    except Exception as e:                                   # noqa: BLE001
        print(f"[mem] {tag}: probe failed {type(e).__name__}: {e}")
        return None


def free_leg(*objs) -> None:
    """Explicit teardown between sequential legs — the T4 fits the legs one at
    a time, never together."""
    for o in objs:
        del o
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                        # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# auth                                                                         #
# --------------------------------------------------------------------------- #
def get_hf_token() -> str:
    """Colab Secrets -> env -> Keys.txt in place. NEVER printed or written."""
    if in_colab():
        try:
            from google.colab import userdata
            t = userdata.get("HF_TOKEN")
            if t:
                return t
        except Exception:                                    # noqa: BLE001
            pass
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        return t
    if os.path.exists(DEV_KEYS):
        m = re.search(r"hf_[A-Za-z0-9]+",
                      open(DEV_KEYS, encoding="utf-8",
                           errors="replace").read())
        if m:
            return m.group(0)
    raise SystemExit(
        "no HF token: in Colab add a secret named HF_TOKEN "
        "(key icon in the left sidebar) and grant this notebook access; "
        "elsewhere set the HF_TOKEN env var")


def hf_api():
    try:
        import truststore
        truststore.inject_into_ssl()      # dev box sits behind a TLS proxy
    except Exception:                                        # noqa: BLE001
        pass
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import HfApi
    return HfApi(token=get_hf_token())


def hf_download(repo: str, rf: str, repo_type: str = "dataset",
                force: bool = False) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, rf, repo_type=repo_type,
                           token=get_hf_token(), force_download=force)


# --------------------------------------------------------------------------- #
# banking + resume — the loop everything else hangs off                        #
# --------------------------------------------------------------------------- #
def bank_json(api, repo: str, rf: str, obj: dict,
              verify: bool = True, indent: int | None = 1) -> int:
    """Upload one JSON object and FAR-SIDE verify it by byte round-trip.

    The push log is never trusted (silent-push-failure class): on verify the
    file is re-downloaded with force_download=True and byte-compared.

    ⚠️ `indent=None` writes COMPACT JSON. That is not a style choice: with
    `indent=1` every element of every nested list gets its own line, and a
    schema-v2 record is mostly nested lists (RLE runs, contour points). MEASURED
    on the 5-clip pilot: 317 109 B/clip at `indent=1` against 120 005 B compact
    — **64 % of the file is whitespace**, and the 6.44x growth over the banked
    v1 records collapses to 2.44x for the SAME information. Compactness is the
    lossless lever; dropping detections would be the lossy one."""
    payload = json.dumps(obj, indent=indent,
                         separators=None if indent else (",", ":")
                         ).encode("utf-8")
    api.upload_file(path_or_fileobj=io.BytesIO(payload), path_in_repo=rf,
                    repo_id=repo, repo_type="dataset")
    if verify:
        back = open(hf_download(repo, rf, force=True), "rb").read()
        if back != payload:
            raise RuntimeError(
                f"FARSIDE VERIFY FAILED for {repo}/{rf}: "
                f"pushed {len(payload)} B, far side returned {len(back)} B "
                "or different bytes — do NOT mark this clip done")
    return len(payload)


def ensure_repo(api, repo: str) -> None:
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)


def list_far(api, repo: str, prefix: str) -> dict[str, int | None]:
    """{rfilename: size} for far-side files under prefix (fresh listing)."""
    info = api.dataset_info(repo, files_metadata=True)
    return {f.rfilename: f.size for f in info.siblings
            if f.rfilename.startswith(prefix)}


def done_set(api, repo: str, prefix: str, suffix: str = ".json",
             verify_sample: bool = True) -> set[str]:
    """Resume primitive: clip ids already banked under prefix.

    Stems are trusted only after a sample round-trip proves stem == the
    record's own clip_id (guards a filename/content drift ever creeping in),
    and zero-byte far-side files are refused rather than counted done."""
    far = list_far(api, repo, prefix)
    stems, bad = [], []
    for rf, size in far.items():
        if not rf.endswith(suffix) or "/_" in rf[len(prefix) - 1:]:
            continue
        if size == 0:
            bad.append(rf)
            continue
        stems.append(rf[len(prefix):-len(suffix)].split("/")[-1])
    if bad:
        raise RuntimeError(f"far side holds ZERO-BYTE files {bad[:3]} — a "
                           "silent push failure; delete/re-push before resume")
    if stems and verify_sample:
        probe = f"{prefix}{stems[0]}{suffix}"
        rec = json.load(open(hf_download(repo, probe, force=True)))
        got = rec.get("clip_id")
        if got != stems[0]:
            raise RuntimeError(
                f"resume refused: {probe} carries clip_id={got!r} — filename "
                "and content disagree, the done-set would be a lie")
    return set(stems)


def census_records(items, want: set[str] | None = None,
                   require_schema: int | None = None,
                   require_conf: float | None = None) -> dict:
    """⛔ THE COMPLETION CRITERION, AND IT IS NOT A FILE COUNT (C77).

    `done_set` answers *"does a non-empty file exist?"*. On 2026-08-16 the
    answer was yes for 115/115 clips whose entire payload was
    `RuntimeError: mat1 and mat2 must have the same dtype` — well-formed,
    correctly named, correctly counted, and empty. The verification enumerated
    CONTAINERS and never evaluated the quantity the artifact exists to produce.

    This reads every banked RECORD and returns what settles it: total
    detections, per-concept totals, the ERROR-STRING census, the clips with
    zero detections, and — the piece that makes a zero readable — whether the
    road/sky LIVENESS control also read zero (a dead engine) or only the agent
    concepts did (a legitimately empty scene).

    A run is complete when `pass_` is True. Nothing else is.

    ⭐ `require_schema` / `require_conf` (2026-08-16) extend the SAME idea one
    step: a record can be present, non-empty, error-free, live — and still be
    the WRONG record, because it was detected at a different confidence floor
    or under an older schema. That difference is invisible in the payload (a
    lower floor shows up only as rows that are not there), so it has to be
    checked against the stamped `engine` block or not at all. A record that
    fails either requirement is counted in `wrong_schema` / `wrong_conf`, is
    NOT complete, and will be re-run.

    ⭐ THIS IS THE PREDICATE WITH NO TRANSPORT ATTACHED (2026-08-17), and the
    split is the point. `items` is any iterable of `(clip_id, record)`, so the
    SAME rule judges a far-side HF corpus (`content_census`) and a local
    directory (`content_census_local`). A run whose durable bank is the dev box
    rather than HF would otherwise need a second completion rule, and two
    implementations of "is this clip done?" is how a corpus goes mixed while
    both checks report green — which is this census's own defect, one level up.
    """
    import collections
    per, sper, errs = (collections.Counter(), collections.Counter(),
                       collections.Counter())
    n_det = n_live = n_dead = n_nocontrol = 0
    n_scene = n_wrong_schema = n_wrong_conf = 0
    zero, seen, complete = [], set(), []
    n_records = 0
    for cid, rec in items:
        n_records += 1
        if rec.get("clip_id") != cid:
            raise RuntimeError(
                f"record for {cid!r} carries clip_id={rec.get('clip_id')!r} — "
                "filename and content disagree, the done-set would be a lie")
        seen.add(cid)
        nd = int(rec.get("n_det_total") or 0)
        n_det += nd
        n_scene += int(rec.get("n_scene_det_total") or 0)
        for k, v in (rec.get("per_concept_hits") or {}).items():
            per[k] += int(v)
        for k, v in (rec.get("per_scene_hits") or {}).items():
            sper[k] += int(v)
        eng = rec.get("engine") or {}
        bad_schema = (require_schema is not None
                      and int(rec.get("schema_version") or 0) < require_schema)
        got_conf = eng.get("confidence_threshold")
        bad_conf = (require_conf is not None
                    and (got_conf is None
                         or abs(float(got_conf) - float(require_conf)) > 1e-9))
        n_wrong_schema += int(bad_schema)
        n_wrong_conf += int(bad_conf)
        clip_err = 0
        if rec.get("err_kinds"):
            errs.update({k: int(v) for k, v in rec["err_kinds"].items()})
            clip_err = sum(int(v) for v in rec["err_kinds"].values())
        else:                                   # pre-census records
            for f in (rec.get("frames") or {}).values():
                for d in f.get("det", []):
                    if "error" in d:
                        errs[str(d["error"])[:60]] += 1
                        clip_err += 1
        # ⛔ RECOMPUTED FROM `n_det`, NOT READ FROM `live`. The stored boolean
        # is a DERIVED field and its rule changed once already (all -> any,
        # after an underpass gave `road 2 · sky 0` and was called dead). When
        # the inputs are banked, trusting the derivation is a needless
        # dependency on which version wrote the record.
        lv = rec.get("liveness")
        counts = (lv or {}).get("n_det") or {}
        if lv is None:
            n_nocontrol += 1
            alive = False
        else:
            alive = any(int(v) > 0 for v in counts.values())
            n_live += int(alive)
            n_dead += int(not alive)
        if nd == 0:
            zero.append({"clip_id": cid, "liveness_live": alive,
                         "liveness_n_det": counts})
        # ⛔ THE RESUME PREDICATE, AND IT IS NOT `n_det_total > 0`. A clip is
        # COMPLETE when the fixed engine produced it — i.e. the record carries
        # the liveness control AND holds no error entries. Keying on detections
        # instead would re-run a LEGITIMATELY EMPTY scene forever (its zero is
        # the right answer), and keying on file presence would skip the stale
        # BFloat16 records forever, which is C77 in the resume path.
        if lv is not None and clip_err == 0 and not bad_schema and not bad_conf:
            complete.append(cid)
    out = {"n_records": n_records, "n_det_total": n_det,
           "n_scene_det_total": n_scene,
           "complete_clips": sorted(complete),
           "n_complete": len(complete),
           "per_concept_totals": dict(per.most_common()),
           "per_scene_totals": dict(sper.most_common()),
           "error_census": dict(errs.most_common()),
           "clips_with_zero_det": len(zero), "zero_det_clips": zero,
           "liveness_live": n_live, "liveness_dead": n_dead,
           "records_without_control": n_nocontrol,
           "require_schema": require_schema, "require_conf": require_conf,
           "wrong_schema": n_wrong_schema, "wrong_conf": n_wrong_conf}
    if want is not None:
        out["coverage"] = f"{len(seen & want)}/{len(want)}"
        out["missing"] = sorted(want - seen)
        out["extra"] = sorted(seen - want)
    out["pass_"] = bool(n_det > 0 and not errs and n_dead == 0
                        and n_nocontrol == 0
                        and n_wrong_schema == 0 and n_wrong_conf == 0
                        and (want is None or not out["missing"]))
    return out


def content_census(api, repo: str, prefix: str,
                   want: set[str] | None = None,
                   require_schema: int | None = None,
                   require_conf: float | None = None) -> dict:
    """`census_records` over a far-side HF prefix. Transport only."""
    far = list_far(api, repo, prefix)
    rfs = sorted(rf for rf in far
                 if rf.endswith(".json") and "/_runs/" not in rf)

    def _items():
        for rf in rfs:
            yield (rf[len(prefix):-len(".json")],
                   json.load(open(hf_download(repo, rf, force=True))))

    return census_records(_items(), want, require_schema, require_conf)


def content_census_local(dirpath, want: set[str] | None = None,
                         require_schema: int | None = None,
                         require_conf: float | None = None) -> dict:
    """`census_records` over a local directory of `<clip_id>.json`.

    ⛔ THE RESUME AUTHORITY WHEN THE DURABLE BANK IS NOT HF. A Colab VM is
    reclaimed without warning and its `/content` goes with it, so a run that
    must not push to a public-facing platform has to keep its done-set on the
    dev box. This is that done-set — and it is the SAME predicate the far-side
    census applies, not a lookalike, because `census_records` is shared."""
    import glob
    import os as _os
    paths = sorted(glob.glob(_os.path.join(str(dirpath), "*.json")))

    def _items():
        for p in paths:
            stem = _os.path.basename(p)[:-len(".json")]
            if stem.startswith("_"):
                continue
            with open(p, encoding="utf-8") as fh:
                yield stem, json.load(fh)

    return census_records(_items(), want, require_schema, require_conf)


def run_manifest(api, repo: str, prefix: str, tag: str, extra: dict) -> str:
    """Bank a run manifest beside the outputs. One per (re)start."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    rf = f"{prefix}_runs/{ts}-{tag}.json"
    man = {"ts_utc": ts, "tag": tag, "host": platform.node(),
           "in_colab": in_colab(), "python": sys.version.split()[0],
           **extra}
    bank_json(api, repo, rf, man)
    return rf


# --------------------------------------------------------------------------- #
# gap derivation (C18: count records, not files)                               #
# --------------------------------------------------------------------------- #
def derive_sam3_gap(api, limit: int | None = None,
                    progress_every: int = 50) -> dict:
    """Read fused_aug120 RECORDS and split them by their own
    `perception.absent` marker. The unit is the RECORD — the 115-clip gap was
    invisible to every file count (25 sam3 files, all rc=0) and visible only
    in the records (AUG120_FUSION_RESULT.md §3)."""
    files = sorted(rf for rf in list_far(api, DS_LABELS, FUSED_PREFIX)
                   if not rf.startswith(FUSED_PREFIX + "_"))
    if limit:
        files = files[:limit]
    absent, covered, weird = [], [], []
    for i, rf in enumerate(files):
        rec = json.load(open(hf_download(DS_LABELS, rf)))
        cid = rec.get("clip_id") or Path(rf).stem
        reason = (rec.get("perception") or {}).get("absent")
        if reason == SAM3_ABSENT_REASON:
            absent.append(cid)
        elif reason is None:
            covered.append(cid)
        else:
            weird.append((cid, reason))
        if (i + 1) % progress_every == 0:
            print(f"[gap] {i + 1}/{len(files)} records read "
                  f"(absent so far: {len(absent)})", flush=True)
    if weird:
        raise RuntimeError(f"unexpected perception.absent values: {weird[:5]}")
    return {"absent": sorted(absent), "covered": sorted(covered),
            "n_records_checked": len(files),
            "source": "fused_aug120/<clip>.json perception.absent (records)",
            "evidence_class": "MEASURED"}


def cross_check_gap(api, gap: dict, partial: bool = False) -> None:
    """Second and third probes (absence needs >=2 probes): `_label_sources
    .json` per-clip sam3 source map, and `_summary.json` sam3_missing.
    Any disagreement refuses the run."""
    ls = json.load(open(hf_download(DS_LABELS,
                                    FUSED_PREFIX + "_label_sources.json")))
    src_absent = {c for c, s in (ls.get("sources") or {}).items()
                  if s.get("sam3") is None}
    mine = set(gap["absent"])
    if partial:
        universe = mine | set(gap["covered"])
        src_absent &= universe
    if mine != src_absent:
        raise RuntimeError(
            "gap derivation DISAGREES with _label_sources.json: "
            f"records say {len(mine)}, sources map says {len(src_absent)}; "
            f"only-in-records={sorted(mine - src_absent)[:3]} "
            f"only-in-sources={sorted(src_absent - mine)[:3]}")
    if not partial:
        summ = json.load(open(hf_download(DS_LABELS,
                                          FUSED_PREFIX + "_summary.json")))
        if summ.get("sam3_missing") != len(mine):
            raise RuntimeError(f"_summary.json sam3_missing="
                               f"{summ.get('sam3_missing')} != {len(mine)}")
    print(f"[gap] cross-check OK: {len(mine)} SAM3-absent clips agree across "
          f"records + _label_sources{'' if partial else ' + _summary'}")


def check_gap_fixture(gap: dict, root: Path) -> None:
    """Compare a FULL derivation against the banked dev-box fixture; loud diff
    on drift (the far side moved or the derivation changed — either matters)."""
    fx = root / "colab" / "fixtures" / "sam3_backfill_expected.json"
    if not fx.exists():
        print("[gap] no fixture banked — skipping fixture check")
        return
    exp = json.load(open(fx))
    if set(exp["clips"]) != set(gap["absent"]):
        only_e = sorted(set(exp["clips"]) - set(gap["absent"]))[:5]
        only_g = sorted(set(gap["absent"]) - set(exp["clips"]))[:5]
        raise RuntimeError(
            f"gap drifted vs fixture ({exp['derived_utc']}): fixture "
            f"{len(exp['clips'])} vs derived {len(gap['absent'])}; "
            f"fixture-only={only_e} derived-only={only_g}. If a backfill "
            "already re-fused some clips this is EXPECTED — re-derive the "
            "fixture, do not force past it.")
    print(f"[gap] fixture check OK ({len(exp['clips'])} clips, "
          f"derived {exp['derived_utc']})")


# --------------------------------------------------------------------------- #
# v2 record assembly (for the SAM3 backfill's B3 cross-check boxes)            #
# --------------------------------------------------------------------------- #
def load_v2_records(api, want: set[str]) -> dict[str, dict]:
    """Pull the far-side batch v2 JSONs and index records by clip_id,
    restricted to `want`. Duplicates (the 152 two-pass clips) are asserted
    content-identical modulo `_calls` — the declared merge policy of the
    fusion run (aug120_fuse_run.py); a substantive diff aborts."""
    files = sorted(rf for rf in list_far(api, DS_LABELS, "batch_")
                   if rf.endswith("/v2/ph0_v2.json"))
    out: dict[str, dict] = {}
    n_rec = 0
    for rf in files:
        d = json.load(open(hf_download(DS_LABELS, rf)))
        for rec in d.get("clips", []):
            cid = rec.get("clip_id")
            n_rec += 1
            if not cid or cid not in want or rec.get("fatal"):
                continue
            if cid in out:
                a = {k: v for k, v in out[cid].items() if k != "_calls"}
                b = {k: v for k, v in rec.items() if k != "_calls"}
                if a != b:
                    raise RuntimeError(
                        f"duplicate v2 records for {cid} DIFFER substantively "
                        f"({rf}) — refusing the silent-merge")
                continue
            out[cid] = rec
    print(f"[v2] {len(files)} far-side v2 files -> {n_rec} records -> "
          f"{len(out)}/{len(want)} wanted clips (records counted, not files)")
    missing = want - set(out)
    if missing:
        raise RuntimeError(f"{len(missing)} wanted clips have NO v2 record "
                           f"(e.g. {sorted(missing)[:3]}) — the gap list and "
                           "the v2 layer disagree; stop and look")
    return out


# --------------------------------------------------------------------------- #
# video: per-batch shard pull + bridge (mirrors aug120_pipeline.py)            #
# --------------------------------------------------------------------------- #
def w120_locations(api) -> dict[str, str]:
    info = api.dataset_info(DS_VIDEO)
    return {f.rfilename.split("/")[-1][: -len(".v2ep.pt")]: f.rfilename
            for f in info.siblings if f.rfilename.endswith(".v2ep.pt")}


def bridge_batch(batch: list[str], loc: dict[str, str], records_pq: str,
                 work: Path) -> Path:
    """Pull this batch's shards, bridge to mp4+ego via the repo's own
    v2_to_pilot.py, DELETE the shards (disk discipline: peak = one batch).
    Returns the bridge output dir (videos/ ego/ clips.json)."""
    import shutil
    cdir = work / "corpus"
    cdir.mkdir(parents=True, exist_ok=True)
    seg = loc[batch[0]].split("/")[0]
    for side in ("_geometry.json", "_v2manifest.pt"):
        try:
            shutil.copyfile(hf_download(DS_VIDEO, f"{seg}/{side}"),
                            cdir / side)
        except Exception as e:                               # noqa: BLE001
            print(f"[bridge] side MISS {side} {type(e).__name__}")
    for cid in batch:
        shutil.copyfile(hf_download(DS_VIDEO, loc[cid]),
                        cdir / f"{cid}.v2ep.pt")
    import v2_to_pilot
    rc = v2_to_pilot.main(["--corpus", str(cdir), "--records", records_pq,
                           "--out", str(work), "--n", str(len(batch))])
    shutil.rmtree(cdir, ignore_errors=True)
    if rc != 0:
        raise RuntimeError(f"bridge failed rc={rc} for batch {batch[:2]}…")
    written = set(json.load(open(work / "clips.json")))
    missing = [c for c in batch if c not in written]
    if missing:
        raise RuntimeError(f"bridge dropped {len(missing)} clips: "
                           f"{missing[:3]}")
    return work


# --------------------------------------------------------------------------- #
# VLM leg — unsloth loader around ph0_v2's ConstrainedVLM                      #
# --------------------------------------------------------------------------- #
#: Runtime-resolved candidates, newest first. MEASURED on HF 2026-08-16:
#: `Qwen/Qwen3.5-9B` and the `unsloth/Qwen3.5-9B` mirror EXIST (arch
#: Qwen3_5ForConditionalGeneration — the multimodal arch ph0_v2 drives via
#: AutoModelForImageTextToText; this IS the PI's "qwen3.5 9B VL", and the
#: production ph0 arm). No pre-quantized 4-bit of it exists yet, so 4-bit is
#: applied AT LOAD. The Qwen3-VL-8B ids are an OLDER GENERATION and load only
#: with allow_fallback=True.
VLM_CANDIDATES = ("unsloth/Qwen3.5-9B", "Qwen/Qwen3.5-9B")
VLM_FALLBACKS = ("unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
                 "Qwen/Qwen3-VL-8B-Instruct")


def resolve_vlm_model(api, allow_fallback: bool = False) -> dict:
    """Pick the first candidate that exists on HF; PRINT what resolved and
    its architecture; FAIL LOUD rather than silently substituting an older
    generation (fallbacks need allow_fallback=True)."""
    env = os.environ.get("S2_VLM_MODEL")
    order = ([env] if env else []) + list(VLM_CANDIDATES) + \
        (list(VLM_FALLBACKS) if allow_fallback else [])
    tried = []
    for mid in order:
        try:
            api.model_info(mid)
            arch = "?"
            try:
                cfg = json.load(open(hf_download(mid, "config.json",
                                                 repo_type="model")))
                arch = (cfg.get("architectures") or ["?"])[0]
            except Exception:                                # noqa: BLE001
                pass
            fam_ok = "3_5" in arch or "3.5" in mid or mid == env
            note = ("OK" if fam_ok else
                    "⚠️ FALLBACK GENERATION (not Qwen3.5) — explicitly "
                    "allowed by allow_fallback=True")
            print(f"[vlm] RESOLVED {mid} arch={arch} — {note}")
            return {"model_id": mid, "arch": arch, "fallback": not fam_ok,
                    "tried": tried}
        except Exception as e:                               # noqa: BLE001
            tried.append(f"{mid}: {type(e).__name__}")
            print(f"[vlm] miss {mid} ({type(e).__name__})")
    raise SystemExit(
        f"NO VLM candidate resolved (tried {tried}). Refusing to silently "
        "substitute — set S2_VLM_MODEL or allow_fallback=True knowingly.")


def load_vlm(model_id: str):
    """Load the VLM 4-bit and return a fully-armed ph0_v2.ConstrainedVLM.

    Loader order: unsloth FastVisionModel (the PI's tip — fits Qwen3.5-9B on
    the 16 GB T4 at ~7–8 GB) -> plain transformers + BitsAndBytesConfig 4-bit
    (same fit, no unsloth kernels) — WHICH loader won is printed and returned
    on the object. Everything downstream (`ask`, the JSON-schema FSM,
    `_build_tokenizer_data`) is ph0_v2's own code, inherited not copied."""
    import torch
    import ph0_v2
    vlm = ph0_v2.ConstrainedVLM.__new__(ph0_v2.ConstrainedVLM)
    model = processor = loader = None
    errs = []
    try:
        from unsloth import FastVisionModel
        model, processor = FastVisionModel.from_pretrained(
            model_id, load_in_4bit=True)
        FastVisionModel.for_inference(model)
        loader = "unsloth.FastVisionModel(load_in_4bit=True)"
    except Exception as e:                                   # noqa: BLE001
        errs.append(f"unsloth: {type(e).__name__}: {str(e)[:120]}")
        model = processor = None
    if model is None:
        import transformers
        from transformers import AutoProcessor, BitsAndBytesConfig
        processor = AutoProcessor.from_pretrained(model_id,
                                                  trust_remote_code=True)
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                     "AutoModelForCausalLM"):
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(model_id, quantization_config=bnb,
                                            device_map="auto",
                                            trust_remote_code=True)
                loader = f"transformers.{name} + bnb nf4 4-bit"
                break
            except Exception as e:                           # noqa: BLE001
                errs.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
    if model is None:
        raise RuntimeError(f"no loader could load {model_id}: {errs}")
    if not hasattr(processor, "apply_chat_template") or \
            not hasattr(processor, "image_processor"):
        # unsloth sometimes hands back a bare tokenizer for VL models — the
        # ask() path needs the full AutoProcessor (video support).
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id,
                                                  trust_remote_code=True)
    print(f"[vlm] LOADED {model_id} via {loader}"
          + (f" (unsloth path failed: {errs[0]})" if errs and
             loader.startswith("transformers") else ""))
    # ---- arm the object exactly as ph0_v2.ConstrainedVLM.__init__ does ----
    # (mirrored tail — the loader above replaces only the from_pretrained)
    vlm.processor, vlm.model, vlm.model_id = processor, model, model_id
    vlm.auto_class = loader
    vlm.model.eval()
    vlm._torch = torch
    vlm.tok = getattr(processor, "tokenizer", None) or processor
    from qwen_vl_utils import process_vision_info
    vlm._pvi = process_vision_info
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer
    from lmformatenforcer.characterlevelparser import \
        CharacterLevelParserConfig
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    vlm._JsonSchemaParser = JsonSchemaParser
    vlm._TokenEnforcer = TokenEnforcer
    vlm._CfgCls = CharacterLevelParserConfig
    print("[vlm] building token-enforcer vocab table (one-time full-vocab "
          "sweep, a few minutes on Colab CPU)…", flush=True)
    vlm._tok_data = vlm._build_tokenizer_data(TokenEnforcerTokenizerData)
    return vlm


def vlm_leg(vlm, frames, n_past: int, engine_a, ego_state,
            ego_mode: str = "past") -> dict:
    """One clip through ph0_v2.run_clip — the four constrained calls B1–B4."""
    import ph0_v2
    return ph0_v2.run_clip(vlm, frames, n_past, engine_a,
                           ego_state=ego_state, ego_mode=ego_mode)


# --------------------------------------------------------------------------- #
# SAM3 leg                                                                     #
# --------------------------------------------------------------------------- #
def load_sam3():
    """Processor once per session (facebook/sam3, gate GRANTED for this token
    — MEASURED config.json download 2026-08-16).

    ⛔ `build_processor` installs the C77 dtype fix
    (`ph0_sam3.install_dtype_agreement`) BEFORE the weights load. It is printed
    here because a run whose fix silently failed to apply is exactly the run
    that banks 115 records of `RuntimeError: mat1 and mat2 must have the same
    dtype`."""
    import ph0_sam3
    proc, meta = ph0_sam3.build_processor(None)
    print(f"[sam3] processor up · {meta['weights']} · "
          f"dtype_fix={meta.get('dtype_fix')}")
    return proc, meta


def sam3_leg(proc, frames, v2rec: dict, *, frame_stride: int = 8,
             min_score: float = 0.0, scene_concepts=None,
             scene_min_score: float | None = None, contours: bool = True,
             contour_tol_px: float | None = None,
             contour_max_pts: int | None = None,
             meta: dict | None = None) -> dict:
    """Per-clip block of ph0_sam3.main(), with the count ALWAYS explicit.

    Mirrors ph0_sam3.py:411-439 (B3 boxes -> vlm_boxes; run_clip_frames;
    record shape) so the processor is loaded once per session instead of once
    per invocation. Output shape == a `sam3.json` clips[] row, so ph1_fuse
    consumes it unchanged.

    ⚠️ `scene_concepts=None` means NO scene channel and a v1-shaped record —
    the default is deliberately the old behaviour, so a caller that has not
    been updated cannot silently start producing a different schema. Schema v2
    is something a caller ASKS for (pass `ph0_sam3.SCENE_CONCEPTS`)."""
    import ph0_sam3
    from ph0_v2 import norm_to_px
    fh, fw = int(frames[0].shape[0]), int(frames[0].shape[1])
    signs = (v2rec.get("signs") or {}).get("signs") or []
    vlm_boxes = []
    for i, g in enumerate(v2rec.get("grounding") or []):
        if not g or not g.get("visible") or not g.get("bbox"):
            continue
        vlm_boxes.append({"box_xyxy": norm_to_px(g["bbox"], fw, fh),
                          "frame_idx": int(g.get("frame_idx", 0)),
                          "label": signs[i].get("kind", "sign")
                          if i < len(signs) else "sign"})
    t0 = time.time()
    # ⭐ liveness=True is EXPLICIT here, not inherited: the road/sky positive
    # control is what distinguishes an empty scene from a dead engine, and
    # C77 is what a run without it banks.
    kw = {}
    if contour_tol_px is not None:
        kw["contour_tol_px"] = contour_tol_px
    if contour_max_pts is not None:
        kw["contour_max_pts"] = contour_max_pts
    out = ph0_sam3.run_clip_frames(proc, frames, ph0_sam3.AGENT_CONCEPTS,
                                   vlm_boxes, frame_stride=frame_stride,
                                   min_score=min_score, liveness=True,
                                   scene_concepts=scene_concepts,
                                   scene_min_score=scene_min_score,
                                   contours=contours, **kw)
    out.update({"clip_id": v2rec.get("clip_id"), "frame_wh": [fw, fh],
                "wall_s": round(time.time() - t0, 1)})
    if meta:
        # ⛔ THE DETECTION FLOOR TRAVELS WITH THE RECORD, OR IT IS UNRECOVERABLE.
        # `confidence_threshold` filters INSIDE the vendor's forward pass, so a
        # record detected at 0.5 and one at 0.25 are structurally identical —
        # the difference is only in what is ABSENT, which nothing downstream can
        # see. Stamping it here is what lets `content_census` refuse to call a
        # 0.5 record part of a 0.25 corpus.
        out["engine"] = {
            "confidence_threshold": meta.get("confidence_threshold"),
            "confidence_threshold_set_via":
                meta.get("confidence_threshold_set_via"),
            "schema_version": meta.get("schema_version"),
            "weights": meta.get("weights"),
            "dtype_fix_applied": bool((meta.get("dtype_fix") or {})
                                      .get("applied"))}
    return out


# --------------------------------------------------------------------------- #
# ego leg                                                                      #
# --------------------------------------------------------------------------- #
#: strategic_gt.py thresholds (nurec-gsplat), ported verbatim: |dyaw| < 25°
#: is road-following, >= 150° is a U-turn.
STRAIGHT_DEG = 25.0
UTURN_DEG = 150.0


def ego_leg(ego_npz_path: str, t0_s: float = 8.0) -> dict:
    """Engine-A numbers + spine + an ego-yaw g_str VOTE.

    ⚠️ The vote inherits strategic_gt.py's own caveat: a route label read off
    the ego's future yaw CANNOT distinguish 'took the left branch' from
    'drifted left on a curving road' — the map-derived option set fixes that
    only where map.xodr exists (NuRec scenes, provenance 'map'). On
    PhysicalAI clips this is therefore ONE VOTE among several, never the sole
    source; provenance 'ego' marks it privileged-labels-only."""
    import math
    import numpy as np
    import ph0_pilot
    import ph0_v2
    import ph1_fuse
    spine = ph1_fuse.ego_from_npz(ego_npz_path)              # speed_profile
    d = np.load(ego_npz_path)
    poses_np = np.asarray(d["poses"] if "poses" in d.files
                          else d[d.files[0]], dtype=np.float32)
    t0_idx = int(round(t0_s * ph0_pilot.POSE_HZ))
    engine_a = ego_state = None
    try:
        import torch
        poses_t = torch.as_tensor(poses_np)
        engine_a = ph0_pilot.engine_a_for_prompt(
            ph0_pilot.engine_a_summary(poses_t, t0_idx))
        ego_state = ph0_v2.ego_past_state(poses_t, t0_idx,
                                          dt=1.0 / ph0_pilot.POSE_HZ)
    except Exception as e:                                   # noqa: BLE001
        print(f"[ego] engine A unavailable ({type(e).__name__}: "
              f"{str(e)[:80]}) — spine + vote only")
    yaw = poses_np[:, 2].astype(float)
    t0c = min(max(t0_idx, 0), len(yaw) - 1)
    dyaw = math.degrees((yaw[-1] - yaw[t0c] + math.pi) % (2 * math.pi)
                        - math.pi)
    if abs(dyaw) >= UTURN_DEG:
        vote = {"token": "TURN", "args": {"direction": "left" if dyaw > 0
                                          else "right", "kind": "uturn"}}
    elif abs(dyaw) > STRAIGHT_DEG:
        vote = {"token": "TURN", "args": {"direction": "left" if dyaw > 0
                                          else "right", "kind": "turn"}}
    else:
        vote = {"token": "FOLLOW_MAIN_ROAD", "args": {}}
    vote.update({"net_dyaw_deg_from_t0": round(dyaw, 2), "src": "ego",
                 "caveat": "yaw-derived; cannot separate turn from curving "
                           "road (strategic_gt.py rule) — one vote, never "
                           "sole source"})
    return {"spine": spine, "engine_a": engine_a, "ego_state": ego_state,
            "g_str_vote": vote, "n_poses": int(len(poses_np))}


# --------------------------------------------------------------------------- #
# fusion (ph1_fuse building blocks) -> S2 record                               #
# --------------------------------------------------------------------------- #
def fuse_one(v2rec: dict, sam3rec: dict | None, ego_npz_path: str | None,
             alp: dict | None) -> dict:
    """One clip through ph1_fuse's building blocks; `ph1-fused-v1` shape.
    A missing SAM3 record degrades to the NAMED partial (absent marker +
    not_computable), exactly as the production fuser does."""
    import ph1_fuse
    absent = None if sam3rec else "LAB_SAM3_NOT_RUN"
    s3 = sam3rec or {}
    tracks = ph1_fuse.build_tracks(s3.get("frames") or {})
    r = dict(v2rec)
    if "speed_profile" not in r and ego_npz_path:
        spine = ph1_fuse.ego_from_npz(ego_npz_path)
        if spine:
            r.update(spine)
    cor, conf = ph1_fuse.corroborate(r, s3, tracks,
                                     sam3_absent=absent is not None)
    vocab, vconf = ph1_fuse.emit_vocab(r, alp)
    conf = conf + vconf
    return {
        "schema_version": "ph1-fused-v1", "clip_id": r.get("clip_id"),
        "geometry": {"frame_wh": r.get("_frame_wh")},
        "ego": {k: r.get(k) for k in
                ("ego_state", "route", "speed_profile", "speed_events",
                 "lane_change_events") if k in r},
        "perception": {"tracks": tracks,
                       "per_concept_hits": s3.get("per_concept_hits"),
                       "src": "sam3",
                       **({"absent": absent} if absent else {})},
        "semantics": {"scene": r.get("scene"), "signs": r.get("signs"),
                      "symbols": r.get("symbols"), "src": "vlm",
                      "sign_text_status": "pending_g1_gate"},
        "alpamayo": alp,
        "corroboration": cor, "vocab": vocab,
        "scenario_description": ph1_fuse.scenario_line(r, tracks),
        "_conflicts": conf,
        "inference_admissible": ["perception", "semantics"],
        "_provenance": {"ego": "privileged-labels-only", "sam3": "vision",
                        "vlm": "vision", "alpamayo": "external-labels-only"},
        "_lab": True,
    }


def to_s2(fused: dict, ego_extra: dict | None = None) -> dict:
    import s2_schema
    rec = s2_schema.from_fused(fused, ego_extra=ego_extra)
    rec["_v6_drift_check"] = s2_schema.check_v6_drift()
    return rec


def load_alpamayo(records_pq: str | None) -> dict[str, dict]:
    """{clip_id: {task: raw_json}} — the shape ph1_fuse.main builds."""
    if not records_pq:
        return {}
    import pandas as pd
    df = pd.read_parquet(records_pq)
    out = {}
    for cid, g in df.groupby("clip_id"):
        out[str(cid)] = {row["task"]: row.get("raw_json")
                        for _, row in g.iterrows()}
    return out


# --------------------------------------------------------------------------- #
# smoke stubs (CPU, no GPU models — plumbing only)                             #
# --------------------------------------------------------------------------- #
def stub_frames(n: int = 40, h: int = 64, w: int = 160) -> list:
    """Tiny synthetic uint8 frames with a moving rectangle (so the SAM3 stub's
    box drift and the review sheet have something to show)."""
    import numpy as np
    out = []
    for i in range(n):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:, :, 2] = np.linspace(40, 120, w, dtype=np.uint8)[None, :]
        x = 10 + int(i * (w - 40) / max(n - 1, 1))
        f[h // 3:h // 3 + 14, x:x + 20] = (200, 60, 60)
        out.append(f)
    return out


def stub_vlm_record(clip_id: str) -> dict:
    """Schema-valid ph0-v2.2 record (B1–B4 shapes) marked as a stub."""
    return {
        "clip_id": clip_id, "schema_stub_of": "ph0-v2.2",
        "scene": {"illumination": "day", "weather": "clear",
                  "road_type": "urban", "domain": "urban",
                  "lanes_visible": 2, "lane_ego": 0, "conf": "med"},
        "signs": {"n_signs": 1, "signs": [
            {"kind": "speed", "text": "50", "state": "none",
             "applies_to_ego": True}]},
        "grounding": [{"visible": True, "frame_idx": 4,
                       "bbox": [520, 120, 590, 200]}],
        "grounding_px": [[83, 7, 94, 12]],
        "symbols": {"goal_kind": "follow_main_road",
                    "goal_evidence_sign": None,
                    "actions": [{"verb": "reduce_to", "direction": "none"}],
                    "conf": "med"},
        "_frame_wh": [160, 64], "_all_valid": True, "_smoke_stub": True,
    }


def stub_sam3_record(clip_id: str, frame_wh=(160, 64)) -> dict:
    """A sam3.json clips[] row shaped like ph0_sam3's real output, with two
    concepts across two frames so build_tracks produces moving tracks."""
    def det(concept, x0, score):
        return {"concept": concept, "score": score,
                "box_xyxy": [x0, 20.0, x0 + 20.0, 36.0],
                "mask_area_px": 240}
    return {"clip_id": clip_id, "frame_wh": list(frame_wh),
            "frames": {
                "0": {"n_det": 2, "det": [det("car", 10.0, 0.91),
                                          det("traffic sign", 120.0, 0.83)]},
                "8": {"n_det": 2, "det": [det("car", 40.0, 0.90),
                                          det("traffic sign", 121.0, 0.82)]},
                "16": {"n_det": 1, "det": [det("car", 80.0, 0.88)]}},
            "per_concept_hits": {"car": 3, "traffic sign": 2},
            "n_frames_run": 3, "n_det_total": 5,
            # ⛔ the stub must carry the C77 census keys, or the smoke path
            # exercises a record shape the completion check would REJECT —
            # and the plumbing test would stop testing the plumbing.
            "n_err_total": 0, "err_kinds": {},
            # no `live` boolean: the counts are the primitive and the
            # verdict is ph0_sam3.is_live(), computed at read time
            "liveness": {"concepts": ["road", "sky"],
                         "n_det": {"road": 1, "sky": 1}, "frame_idx": 8},
            "vlm_cross_check": [], "wall_s": 0.0, "_smoke_stub": True}


def stub_alpamayo(clip_id: str) -> dict:
    return {"meta_action": json.dumps(
        {"speed": "decelerate", "lane": "keep", "cause": "speed sign"}),
        "_smoke_stub": True}


# --------------------------------------------------------------------------- #
# review sheet                                                                 #
# --------------------------------------------------------------------------- #
def _b64_png(frame) -> str:
    import base64
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def review_sheet_html(entries: list[dict], title: str = "S2 label lab") -> str:
    """Per clip: sampled frames + emitted tokens + PER-TOKEN PROVENANCE +
    confidence + votes + corroborations/conflicts — the artifact a human
    judges 'is the pipeline extracting the right information' from."""
    import s2_schema
    css = """<style>
    body{font-family:system-ui,sans-serif;margin:16px;background:#fafafa}
    .clip{border:1px solid #ccc;border-radius:8px;margin:14px 0;padding:12px;
          background:#fff}
    .frames img{height:96px;margin:2px;border:1px solid #ddd}
    .tok{display:inline-block;border-radius:6px;padding:3px 8px;margin:3px;
         font-weight:600}
    .g{background:#dbeafe}.a{background:#dcfce7}.t{background:#f3e8ff}
    .prov{font-weight:400;font-size:11px;color:#444}
    table{border-collapse:collapse;font-size:13px;margin:6px 0}
    td,th{border:1px solid #ddd;padding:3px 7px;text-align:left}
    .warn{color:#b45309}.bad{color:#b91c1c}.stub{color:#6b7280;
          font-style:italic}
    </style>"""
    head = (f"<h2>{title}</h2><p><b>schema:</b> {s2_schema.SCHEMA_VERSION} "
            f"(authoritative — spec: "
            f"<code>{s2_schema.AUTHORITATIVE_DOC}</code>)</p>")
    parts = [css, head]

    def tok_html(t, cls):
        if not t:
            return ""
        raw = t.get("args")
        if isinstance(raw, list):        # s2-strategic-v1: [8] slots + mask
            mask = t.get("arg_mask") or [0] * len(raw)
            args = ", ".join(
                f"{s2_schema.GOAL_ARG_NAMES[i]}={raw[i]}"
                for i in range(len(raw)) if i < len(mask) and mask[i])
        else:                            # legacy dict args / g_tac votes
            args = ", ".join(f"{k}={v}" for k, v in (raw or {}).items())
        prov = t.get("provenance")
        if not isinstance(prov, str):
            prov = "+".join(prov or []) or \
                "+".join(t.get("voters") or []) or "?"
        return (f"<span class='tok {cls}'>{t.get('token')}"
                f"{'(' + args + ')' if args else ''} "
                f"<span class='prov'>[{prov} · "
                f"{t.get('confidence', '·')}]</span></span>")

    for e in entries:
        s2, fused = e["s2"], e["fused"]
        cid = s2.get("clip_id", "?")
        stub = " <span class='stub'>(contains smoke stubs)</span>" if \
            e.get("smoke") else ""
        parts.append(f"<div class='clip'><h3>{cid}{stub}</h3>")
        frames = e.get("frames") or []
        if len(frames):
            idx = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
            imgs = "".join(f"<img src='data:image/png;base64,{_b64_png(frames[i])}' "
                           f"title='frame {i}'>" for i in sorted(set(idx)))
            parts.append(f"<div class='frames'>{imgs}</div>")
        a_str = s2.get("a_str")
        a_str = [a_str] if isinstance(a_str, dict) else (a_str or [])
        parts.append("<div><b>g_str</b> " + tok_html(s2.get("g_str"), "g")
                     + " <b>a_str</b> "
                     + ("".join(tok_html(a, "a") for a in a_str)
                        or "<i>none</i>")
                     + " <b>g_tac</b> "
                     + tok_html((s2.get("g_tac") or {}).get("lat"), "t")
                     + tok_html((s2.get("g_tac") or {}).get("lon"), "t")
                     + "</div>")
        ev = (s2.get("g_str") or {}).get("ego_vote")
        if ev:
            agree = (s2["g_str"].get("ego_agrees"))
            klass = "" if agree else "warn"
            parts.append(f"<div class='{klass}'>ego-yaw vote: "
                         f"{ev.get('token')} {ev.get('args')} "
                         f"(net dyaw {ev.get('net_dyaw_deg_from_t0')}°) — "
                         f"{'agrees' if agree else 'DISAGREES with g_str'}"
                         "</div>")
        rows = "".join(
            f"<tr><td>{k}</td><td>{v.get('verdict', '·')}</td>"
            f"<td>{'+'.join(v.get('src', []))}</td></tr>"
            for k, v in (fused.get("corroboration") or {}).items())
        if rows:
            parts.append("<table><tr><th>check</th><th>verdict</th>"
                         f"<th>sources</th></tr>{rows}</table>")
        confs = fused.get("_conflicts") or []
        if confs:
            parts.append(f"<div class='bad'>conflicts: {json.dumps(confs)[:400]}"
                         "</div>")
        lossy = s2.get("_mapping_lossy") or []
        if lossy:
            parts.append(f"<div class='warn'>lossy v6→S2 mapping: {lossy}"
                         "</div>")
        parts.append(f"<div>scenario: "
                     f"{fused.get('scenario_description', '')}</div>")
        parts.append("</div>")
    return "\n".join(parts)


def show_html(html: str, out_path: str | None = None) -> None:
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(html, encoding="utf-8")
        print(f"[sheet] wrote {out_path} ({len(html)} B)")
    try:
        from IPython.display import HTML, display
        display(HTML(html))
    except Exception:                                        # noqa: BLE001
        pass
