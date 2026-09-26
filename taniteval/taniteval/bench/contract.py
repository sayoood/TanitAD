"""The run-directory contract of the one-command suite (BUILD_PLAN.md §1) — shared by every benchmark.

    taniteval/results/bench/<benchmark>/<split>/<run_id>/
        bench_run.json        run identity + provenance        (schema/bench_run.schema.json)
        scores/<arm>.csv      the devkit's OWN per-token output, UNMODIFIED
        artifacts/<arm>.json  TanitEval artifact (criteria_check-shaped)
        criteria/<arm>.txt    tools/criteria_check.py output
        summary.json          per-arm result record                 (schema/summary.schema.json)
        report/               W5 (taniteval.benchreport)
        raw/                  wrapper manifests, hooks, logs, controls (the quotable layer)

Files > 20,000,000 bytes are moved OFF-REPO (``$TANITAD_BENCH_OFFREPO``, default
``C:/Users/Admin/tanitad-bench-offrepo``) and replaced by ``<name>.offrepo.json`` carrying the
sha256; the ``files`` manifest in ``bench_run.json`` lists every file with its location.

⛔ A run whose ``bench_run.json`` / ``summary.json`` does not validate is refused at write time
(:func:`validate_summary`, :func:`validate_bench_run`) — a contract that is only documented is
the failure class this programme keeps re-learning.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import schema_check

#: <repo>/taniteval/taniteval/bench/contract.py -> parents[3] = <repo>
REPO = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO / "taniteval" / "results" / "bench"
OFFREPO_ROOT = Path(os.environ.get("TANITAD_BENCH_OFFREPO", "C:/Users/Admin/tanitad-bench-offrepo"))
SIZE_LIMIT_BYTES = 20_000_000

#: ⛔ THE RESULTS TREE IS APPEND-ONLY (W4, 2026-09-20). MEASURED: four run directories became two in
#: ~20 minutes while the leaderboard was rendering, and a page cited a run directory that had been
#: DELETED — the visible symptom was a false "round-trip differs" alarm on a correct generator, the
#: invisible one a dangling citation. ⇒ a run directory is NEVER deleted or moved once written.
#: A superseded or scratch run either lives under ``_scratch/`` (consumers skip that subtree) or
#: keeps its directory with a ``TOMBSTONE.json`` naming what replaced it (consumers skip a directory
#: that has one — it carries no ``bench_run.json`` on purpose).
SCRATCH_DIR = "_scratch"
TOMBSTONE_FILE = "TOMBSTONE.json"
BENCHMARKS = ("navsim_v2", "navsim_v1", "nuscenes_ol", "internal_t1")
ARM_KINDS = ("model", "floor", "reference")
NAVSIM_FLOORS = ("STOP", "CV")

#: the closed protocol set (schema/bench_run.schema.json $defs.protocol) — read from the schema
#: so the list exists in ONE place.
PROTOCOLS = tuple(schema_check.load_schema("bench_run")["$defs"]["protocol"]["enum"])


class DeviceRefused(RuntimeError):
    """⛔ The box may not be loaded with our-model inference right now (the shared device gate)."""


class ContractError(ValueError):
    """The run directory would violate the §1 contract. Carries every violation."""

    def __init__(self, what: str, errors: list):
        self.errors = list(errors)
        super().__init__(f"{what}: {len(self.errors)} contract violation(s): " + "; ".join(self.errors[:12]))


# --------------------------------------------------------------------------- #
# identity                                                                     #
# --------------------------------------------------------------------------- #
def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ckpt_tag(ckpt: str | None, registry_key: str | None = None) -> str:
    """``none`` for no checkpoint, else the registry key or the file stem, lower-cased to
    ``[a-z0-9_.]`` (the run-id grammar)."""
    if ckpt is None:
        return "none"
    raw = registry_key or Path(ckpt).stem
    tag = re.sub(r"[^a-z0-9_.]+", "_", raw.lower()).strip("_.")
    return tag[:48] or "ckpt"


def make_run_id(benchmark: str, tag: str, now: _dt.datetime | None = None, rand: str | None = None) -> str:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    rand = rand if rand is not None else secrets.token_hex(3)
    if benchmark not in BENCHMARKS:
        raise ContractError("run_id", [f"benchmark {benchmark!r} not in {BENCHMARKS}"])
    if not re.fullmatch(r"[0-9a-f]{6}", rand):
        raise ContractError("run_id", [f"rand {rand!r} must be 6 hex"])
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{benchmark}-{tag}-{rand}"


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_head(repo: Path = REPO) -> dict:
    """``{"sha": 40-hex | "UNAVAILABLE", ...}``. ⛔ The shape is ASSERTED: an empty answer (the
    exFAT 'dubious ownership' refusal, a mount outage) is UNAVAILABLE, never a match."""
    out = {"sha": "UNAVAILABLE"}
    try:
        r = subprocess.run(["git", "-c", f"safe.directory={str(repo).replace(os.sep, '/')}",
                            "-C", str(repo), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=60)
        sha = r.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40}", sha):
            out["sha"] = sha
        else:
            out["reason"] = f"rev-parse rc={r.returncode} stdout={sha[:60]!r} stderr={r.stderr.strip()[:200]!r}"
        b = subprocess.run(["git", "-c", f"safe.directory={str(repo).replace(os.sep, '/')}",
                            "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
                           capture_output=True, text=True, timeout=60)
        out["branch"] = b.stdout.strip() or None
    except Exception as e:                                          # noqa: BLE001
        out["reason"] = f"{type(e).__name__}: {e}"
    return out


def code_blobs(paths) -> dict:
    """git blob ids of the suite's own code, computed LOCALLY (no git call): what ran is pinned
    even when the worktree is dirty."""
    res = {}
    for p in paths:
        p = Path(p)
        b = p.read_bytes()
        res[str(p.relative_to(REPO)).replace(os.sep, "/")] = hashlib.sha1(
            b"blob %d\0" % len(b) + b).hexdigest()
    return res


# --------------------------------------------------------------------------- #
# the run directory                                                            #
# --------------------------------------------------------------------------- #
class RunDir:
    SUBDIRS = ("scores", "artifacts", "criteria", "raw", "report", "plans")

    def __init__(self, benchmark: str, split: str, run_id: str, root: Path | None = None,
                 scratch: bool = False):
        if benchmark not in BENCHMARKS:
            raise ContractError("run dir", [f"benchmark {benchmark!r} not in {BENCHMARKS}"])
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", split):
            raise ContractError("run dir", [f"split {split!r} is not a safe path component"])
        self.benchmark, self.split, self.run_id = benchmark, split, run_id
        self.root = Path(root) if root is not None else RESULTS_ROOT
        self.scratch = bool(scratch)
        base = (self.root / SCRATCH_DIR) if self.scratch else self.root
        self.path = base / benchmark / split / run_id
        self.offrepo = OFFREPO_ROOT / benchmark / split / run_id

    def create(self) -> "RunDir":
        if self.path.exists():
            raise ContractError("run dir", [f"{self.path} already exists — run ids are never reused"])
        for s in self.SUBDIRS:
            (self.path / s).mkdir(parents=True, exist_ok=True)
        return self

    def p(self, rel: str) -> Path:
        return self.path / rel

    def write_json(self, rel: str, obj) -> Path:
        dst = self.p(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")
        os.replace(tmp, dst)
        return dst

    def copy_in(self, rel: str, src) -> Path:
        dst = self.p(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return dst

    def finalize_files(self, size_limit: int = SIZE_LIMIT_BYTES) -> dict:
        """sha256 + size of every file; files above ``size_limit`` move off-repo (stub left)."""
        files = {}
        for f in sorted(self.path.rglob("*")):
            if not f.is_file() or f.name.endswith(".tmp"):
                continue
            rel = str(f.relative_to(self.path)).replace(os.sep, "/")
            if rel == "bench_run.json":
                continue                     # written last; cannot list its own hash
            size = f.stat().st_size
            digest = sha256_file(f)
            if size > size_limit:
                dst = self.offrepo / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dst))
                stub = {"offrepo_path": str(dst).replace(os.sep, "/"), "bytes": size, "sha256": digest,
                        "why": f"> {size_limit} bytes: BUILD_PLAN.md §1 keeps large files off-repo"}
                (f.parent / (f.name + ".offrepo.json")).write_text(json.dumps(stub, indent=1), encoding="utf-8")
                files[rel] = {"bytes": size, "sha256": digest, "location": "offrepo",
                              "offrepo_path": stub["offrepo_path"]}
            else:
                files[rel] = {"bytes": size, "sha256": digest, "location": "repo"}
        return files


def write_tombstone(run_dir, *, superseded_by: str, why: str, evidence_kept: str = "") -> Path:
    """Withdraw a run WITHOUT deleting it (the append-only rule). The directory keeps only what it
    had plus ``TOMBSTONE.json``; consumers skip a directory that has one."""
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    rec = {"TOMBSTONE": True, "run_id": d.name, "removed_utc": utc_now(), "superseded_by": superseded_by,
           "why": why, "evidence_kept": evidence_kept,
           "rule": ("taniteval/results/bench/** is APPEND-ONLY (W4, 2026-09-20): a withdrawn run leaves this "
                    "tombstone instead of vanishing under a consumer that has already cited it")}
    p = d / TOMBSTONE_FILE
    p.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return p


def is_consumable_run(run_dir) -> tuple:
    """``(ok, why)`` — the rule a consumer (W4/W5) applies to a directory in the results tree."""
    d = Path(run_dir)
    if SCRATCH_DIR in d.parts:
        return False, f"under {SCRATCH_DIR}/ — a scratch run, never a result"
    if (d / TOMBSTONE_FILE).exists():
        return False, f"withdrawn: {TOMBSTONE_FILE} names what replaced it"
    if not (d / "bench_run.json").exists():
        # a run WRITES bench_run.json last, so a directory with a live log is IN FLIGHT, not dangling
        return False, ("IN FLIGHT (raw/bench.log exists, bench_run.json not written yet)"
                       if (d / "raw" / "bench.log").exists() else "no bench_run.json")
    return True, "ok"


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o).replace(os.sep, "/")
    if isinstance(o, float) and o != o:
        return None
    return str(o)


# --------------------------------------------------------------------------- #
# validation: schema + the cross-field rules a schema cannot express           #
# --------------------------------------------------------------------------- #
def validate_summary(summary: dict) -> list:
    errs = schema_check.validate(summary, schema_check.load_schema("summary"))
    arms = summary.get("arms") or {}
    floors = summary.get("floors") or []
    for f in floors:
        if f not in arms:
            errs.append(f"/floors: floor {f!r} is not an arm of this run")
        elif arms[f].get("kind") != "floor":
            errs.append(f"/arms/{f}/kind: a floor must have kind 'floor'")
    for name, a in arms.items():
        paired = a.get("paired") or {}
        for f in floors:
            if f not in paired:
                errs.append(f"/arms/{name}/paired: no entry for floor {f!r} (every arm is paired against every floor)")
            elif name == f and paired[f].get("status") != "SELF":
                errs.append(f"/arms/{name}/paired/{f}: an arm paired with itself must read SELF")
        hl = a.get("headline") or {}
        if "column" in hl and hl["column"] == "pdm_score":
            errs.append(f"/arms/{name}/headline/column: pdm_score is NEVER the headline")
    if str(summary.get("benchmark", "")).startswith("navsim"):
        for f in NAVSIM_FLOORS:
            if f not in arms:
                errs.append(f"/arms: mandatory NavSim floor {f!r} missing")
    errs += _ckpt_identified_for_model_arms(summary, arms)
    return errs


def ckpt_display(ck: dict | None) -> str:
    """What a leaderboard cell shows for "which checkpoint" -- NEVER a blank.

    ⛔ Orchestrator ruling 2026-09-20: a row without a ``registry_key`` must show the **sha256
    prefix in the key's place**, so a reader can still identify the checkpoint at a glance. A blank
    cell reads as "no checkpoint" when the truth is "identified, but not cross-referenced" -- two
    very different claims, and only one of them is a problem.

    ⭐ It lives HERE, computed once and written into the artifact, rather than being re-derived by
    each renderer: two producers formatting the same row independently are two things that can
    disagree, which is the same reasoning that made ``provenance.ckpt`` the SAME object
    ``bench_run.json`` carries rather than a second triple.
    """
    ck = ck or {}
    key = str(ck.get("registry_key") or "").strip()
    if key:
        return key
    sha = str(ck.get("sha256") or "").strip()
    if sha:
        return f"sha256:{sha[:12]}"          # identified, just not named
    if str(ck.get("path") or "").strip():
        return "UNIDENTIFIED (path only, no sha256)"
    return "no checkpoint (devkit floors only)"


def _ckpt_identified_for_model_arms(summary: dict, arms: dict) -> list:
    """⛔ A MODEL ROW MUST NAME ITS CHECKPOINT, OR IT IS AN ANONYMOUS RESULT (W4, 2026-09-20).

    On a devkit-floor run (``--ckpt none``) there is no checkpoint and the page correctly says so, so
    this is silent. The moment an arm of kind ``model`` appears, ``provenance.ckpt`` must carry the
    SAME triple ``bench_run.json`` does -- path, sha256 and registry key -- because a leaderboard row
    that cannot say WHICH checkpoint produced it is not a result anyone can act on or reproduce.

    ⚠ Structural on purpose: this runs inside ``validate_summary``, which ``write_summary``
    raises on, so such a run FAILS its contract instead of publishing the row. A documented
    convention would be a request; this is a refusal.
    """
    model_arms = sorted(n for n, a in arms.items() if str(a.get("kind", "")).lower() == "model")
    if not model_arms:
        return []
    ck = (summary.get("provenance") or {}).get("ckpt")
    if not isinstance(ck, dict):
        return [f"/provenance/ckpt: MISSING, but {model_arms} are model arms — a model row must name its "
                f"checkpoint (path + sha256 + registry_key), never publish anonymously"]
    out = []
    # IDENTITY is a hard refusal: without these the row cannot be tied to any artifact at all.
    for field in ("path", "sha256"):
        if not str(ck.get(field) or "").strip():
            out.append(f"/provenance/ckpt/{field}: empty, but {model_arms} are model arms — "
                       f"the leaderboard cannot identify the checkpoint that produced the row")
    # ⚠ The REGISTRY KEY is a different claim: it LINKS an identified checkpoint to
    # MODEL_REGISTRY.md. A run identified by sha256 is not anonymous, so refusing outright would
    # reject a reproducible result over a missing cross-reference. It must therefore be either
    # PRESENT or EXPLICITLY ACCOUNTED FOR -- `cli.py` already writes `registry_key_status` when the
    # operator omits `--registry-key`. What is forbidden is SILENCE.
    if not str(ck.get("registry_key") or "").strip() and not str(ck.get("registry_key_status") or "").strip():
        out.append(f"/provenance/ckpt/registry_key: absent with no `registry_key_status` explaining why, "
                   f"but {model_arms} are model arms — state the key or state why it is missing")
    return out


def validate_bench_run(rec: dict) -> list:
    errs = schema_check.validate(rec, schema_check.load_schema("bench_run"))
    names = [a.get("name") for a in rec.get("arms", [])]
    if len(names) != len(set(names)):
        errs.append("/arms: duplicate arm names")
    if str(rec.get("benchmark", "")).startswith("navsim") and rec.get("status") == "COMPLETE":
        for f in NAVSIM_FLOORS:
            if f not in names:
                errs.append(f"/arms: mandatory NavSim floor {f!r} missing from a COMPLETE run")
    return errs


def require_valid(what: str, errs: list) -> None:
    if errs:
        raise ContractError(what, errs)


# --------------------------------------------------------------------------- #
# the context a benchmark implementation (built-in or plugin) runs against     #
# --------------------------------------------------------------------------- #
class BenchContext:
    """What ``run_benchmark(ctx)`` receives. A plugin (W3 ``navsim_v1``, W6 ``nuscenes_ol``) uses
    ONLY this surface; the core writes ``bench_run.json`` after the plugin returns.

    Read: ``ctx.args`` (argparse namespace: benchmark, ckpt, split, arms, device, …),
    ``ctx.run`` (:class:`RunDir`), ``ctx.run_id``.
    Declare (all REQUIRED before return): ``ctx.set_protocol(tag)``,
    ``ctx.set_devkit(repo, sha, patches)``, ``ctx.set_split(name, n_scenes, n_logs, **extra)``,
    ``ctx.add_arm(name, kind, declared_inputs, **extra)`` per arm, ``ctx.set_stamps(tier, loop, …)``,
    ``ctx.set_claim_bearing(bool)``.
    Write: ``ctx.write_scores(arm, src_csv)``, ``ctx.write_artifact(arm, dict)``,
    ``ctx.run_criteria(arm)``, ``ctx.write_summary(dict)`` (schema-validated; raises
    :class:`ContractError`). GPU: ``ctx.gpu_device()`` — the gap launcher's decision.
    """

    def __init__(self, args, run: RunDir, log=print):
        self.args, self.run, self.run_id, self.log = args, run, run.run_id, log
        self.t0 = time.time()
        self.rec: dict = {
            "schema": "taniteval.bench.bench_run/1", "run_id": run.run_id, "utc": utc_now(),
            "utc_end": None, "benchmark": run.benchmark, "arms": [], "status": "FAILED",
            "report": {"status": "SKIPPED"}, "command": [sys.executable] + list(sys.argv),
        }
        self.summary_written = False
        self.gpu = None

    # -- declarations ------------------------------------------------------ #
    def set_protocol(self, tag: str):
        if tag not in PROTOCOLS:
            raise ContractError("protocol", [f"{tag!r} is not in the closed set {PROTOCOLS}"])
        self.rec["protocol"] = tag

    def set_devkit(self, repo: str, sha: str, patches: list, **extra):
        self.rec["devkit"] = {"repo": repo, "sha": sha, "patches": list(patches), **extra}

    def set_split(self, name: str, n_scenes, n_logs, **extra):
        self.rec["split"] = {"name": name, "n_scenes": n_scenes, "n_logs": n_logs, **extra}

    def add_arm(self, name: str, kind: str, declared_inputs, **extra):
        if kind not in ARM_KINDS:
            raise ContractError("arm", [f"{name}: kind {kind!r} not in {ARM_KINDS}"])
        self.rec["arms"].append({"name": name, "kind": kind, "declared_inputs": list(declared_inputs), **extra})

    def arm_rec(self, name: str) -> dict:
        for a in self.rec["arms"]:
            if a["name"] == name:
                return a
        raise KeyError(name)

    def set_stamps(self, tier: str, loop: dict, evidence_class: str = "MEASURED", **extra):
        self.rec["stamps"] = {"tier": tier, "loop": dict(loop), "evidence_class": evidence_class, **extra}

    def set_claim_bearing(self, flag: bool):
        self.rec["claim_bearing"] = bool(flag)

    # -- outputs ----------------------------------------------------------- #
    def write_scores(self, arm: str, src) -> Path:
        return self.run.copy_in(f"scores/{arm}.csv", src)

    def write_artifact(self, arm: str, art: dict) -> Path:
        return self.run.write_json(f"artifacts/{arm}.json", art)

    def run_criteria(self, arm: str) -> dict:
        """``tools/criteria_check.py`` over ``artifacts/<arm>.json`` -> ``criteria/<arm>.txt``
        (+ the full JSON report in ``raw/criteria_<arm>.json``)."""
        art = self.run.p(f"artifacts/{arm}.json")
        js = self.run.p(f"raw/criteria_{arm}.json")
        r = subprocess.run([sys.executable, str(REPO / "tools" / "criteria_check.py"), str(art), "--json", str(js)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(REPO))
        self.run.p(f"criteria/{arm}.txt").write_text(
            f"rc={r.returncode}\n{r.stdout}\n[stderr]\n{r.stderr}", encoding="utf-8")
        out = {"rc": r.returncode, "json": str(js.relative_to(self.run.path)).replace(os.sep, "/"),
               "json_written": js.exists()}
        if js.exists():
            try:
                rep = json.loads(js.read_text(encoding="utf-8"))
                arts = list((rep.get("artifacts") or {}).values())
                out["registry_version"] = rep.get("registry_version")
                out["unreadable"] = rep.get("unreadable")
                out["n_violations"] = sum(int(a.get("n_violations", 0)) for a in arts) if arts else None
                out["n_work_items"] = sum(int(a.get("n_work_items", 0)) for a in arts) if arts else None
                out["scope"] = [a.get("scope") for a in arts]
            except Exception as e:                                  # noqa: BLE001
                out["parse_error"] = f"{type(e).__name__}: {e}"
        return out

    def write_summary(self, summary: dict) -> Path:
        require_valid("summary.json", validate_summary(summary))
        p = self.run.write_json("summary.json", summary)
        self.summary_written = True
        return p

    def gpu_device(self, *, queue_hint: str = "", **kw) -> str:
        """The device for OUR-MODEL inference, through the SHARED gate (orchestrator arbitration
        2026-09-20). Raises :class:`DeviceRefused` when the box may not be loaded; the refusal text
        carries the queue command. ⛔ Floors / reference agents do not call this: the devkit scorer
        is CPU by construction and is not our-model inference."""
        from .gpu_gap import GpuGapLauncher, device_policy
        # ⛔ limit_mib MUST be threaded from the caller: it was reachable in every signature and passed
        # by nobody, so the gate ran at a hardcoded 1024 MiB that a desktop box can never satisfy.
        lim = getattr(self.args, "gpu_mem_limit_mib", None)
        pol = device_policy(getattr(self.args, "device", "auto"),
                            accept_training_box_load=bool(getattr(self.args, "accept_training_box_load", False)),
                            limit_mib=lim, queue_hint=queue_hint)
        self.rec["device_policy"] = {**pol["record"], "decision": pol["decision"], "reason": pol["reason"]}
        self.log(f"[device] {pol['decision']}: {pol['reason']}")
        if pol["decision"] == "REFUSE":
            raise DeviceRefused(pol["reason"])
        if pol["decision"] == "CPU":
            return "cpu"
        if self.gpu is None:
            kw.setdefault("limit_mib", lim)          # setdefault: a caller's explicit kw still wins
            self.gpu = GpuGapLauncher(requested="cuda", log=self.log, **kw)
        return self.gpu.acquire()
