#!/usr/bin/env python3
"""corpus_census — how many copies does each load-bearing artifact have?

Why this exists
---------------
On 2026-08-03 two independent streams hit the same wall on the same day:

  * the raw parity training corpus (``physicalai-train-e438721ae894``, 2376
    episodes, skip-hash ``f09e44db``, 278.78 GB) was probed for and reported
    **"not found on any live machine"**; and
  * the 256 px REF-C val raster that every published REF-C number is computed
    on was reported to have **one reachable copy**, after one clip had already
    produced a transient unreadable load.

Neither of those was a sudden event. The copy count had been drifting toward 1
for weeks — pod2 terminated, pod1/pod3/eval stopped — and **nothing was
watching the count**. The failure this module prevents is not "a disk died";
it is "the count silently reached 1 and nobody noticed".

(For the record, the first claim was also **wrong**: Thor was holding 439 train
episodes and all 40 eval val episodes the whole time. It had been probed at one
location. Which is the other half of why this exists — see below.)

The four rules this module encodes
----------------------------------
1. **ABSENT and UNKNOWN are different facts.** A listing that fails returns
   ``UNKNOWN``, never ``0``. A chunked relay once stalled reporting ``have=0``
   while 29 files existed, and that reading is precisely how a
   "single-copy" risk gets manufactured out of a network hiccup. Every
   location carries a ``status`` and only ``PRESENT``/``ABSENT`` are
   conclusions; ``UNKNOWN``/``UNREACHABLE`` are the absence of one.

2. **Absence found at ONE location is not absence** (CLAUDE.md, operating
   standard #2). Every artifact declares *several* candidate paths and the
   census probes all of them on every host. A "our pods cannot render" claim
   once stood for 12 days on a single probe.

3. **An ssh config is a cache of what somebody wrote down, not the fleet.**
   Host discovery is delegated to :mod:`tools.fleet_probe`, which derives the
   fleet from the config *and reports what it could not confirm*. A live pod
   was once missed entirely because it was not in the config.

4. **Copy count is over DISTINCT machines.** Two paths on the same host are one
   copy: they die together. ``epcache`` and ``epcache_prefix`` on Thor are one
   disk, not two.

Usage
-----
    python tools/corpus_census.py                     # table
    python tools/corpus_census.py --json out.json     # machine-readable
    python tools/corpus_census.py --no-hf             # skip the HF probe
    python tools/corpus_census.py --hosts thor pod5   # subset

Exit codes: ``0`` every artifact has >= MIN_COPIES; ``1`` something is at
exactly one copy (or is unresolvable); ``2`` something is at ZERO copies or the
census could not be completed. Non-zero is the point — this is a guard, and it
is meant to be run from CI.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

# Minimum copies before an artifact is considered durable. Two distinct
# machines. Deliberately not 3: we are guarding against "silently reached 1",
# and a bar nobody can meet gets muted, which is worse than no bar.
MIN_COPIES = 2

# ---------------------------------------------------------------------------
# Location status vocabulary. The whole point of the module is that these are
# not collapsible into a boolean.
# ---------------------------------------------------------------------------
PRESENT = "PRESENT"          # probed, found, count/size recorded
ABSENT = "ABSENT"            # probed successfully, genuinely not there
PARTIAL = "PARTIAL"          # found, but fewer members than the contract
UNKNOWN = "UNKNOWN"          # probe ran but could not decide
UNREACHABLE = "UNREACHABLE"  # host/endpoint did not answer at all

CONCLUSIVE = (PRESENT, ABSENT, PARTIAL)

#: Statuses that count toward durability. PARTIAL does NOT: half a corpus is
#: not a copy of the corpus, and calling it one is how "we have a backup"
#: becomes false.
COUNTS_AS_COPY = (PRESENT,)


# ---------------------------------------------------------------------------
# Not every copy is equally durable, and counting them as if they were is how
# "we have two copies" becomes false overnight.
#
#   durable  — survives a pod being terminated: HuggingFace, GitHub, hardware
#              we own outright (Thor), and the dev-box working tree.
#   volatile — a RENTED RunPod pod. pod2 was terminated on 2026-08-03 with the
#              programme's arms on it, and pod1/pod3/eval went to `Connection
#              refused` the same week. A pod is not storage.
#
# An artifact whose only non-HF copy is on a pod is ONE provider outage from a
# single copy, so it is reported separately rather than folded into `copies`.
# ---------------------------------------------------------------------------
def is_durable_host(host: str) -> bool:
    return host.startswith("hf:") or host in ("github", "repo", "thor")


# ---------------------------------------------------------------------------
# The artifact contract.
#
# `members` is the number of files the artifact MUST have to be a complete
# copy. `parity` records the committed manifest fact so a re-selection is
# detectable: parity is sacred and anything that re-selects episodes must be
# refused, so the census reports member counts rather than trusting a name.
# ---------------------------------------------------------------------------
@dataclass
class Artifact:
    key: str
    kind: str                       # corpus | raster | anchors | ckpt | evaldump
    desc: str
    members: int | None = None      # expected file count (None = single file)
    parity: str | None = None
    # Candidate locations: (host, path). host "repo" = this git checkout,
    # "hf:<repo_id>" = a HuggingFace repo prefix.
    candidates: list[tuple[str, str]] = field(default_factory=list)
    pattern: str = "ep_*.pt"        # glob used to count members on a host


ARTIFACTS: list[Artifact] = [
    Artifact(
        key="raw-train-epcache-256px",
        kind="corpus",
        desc="Raw parity TRAIN epcache, 256 px — the canonical train corpus",
        members=2376,
        parity="physicalai-train-e438721ae894 / skip-hash f09e44db / 2376 eps",
        candidates=[
            ("hf:Sayood/tanitad-physicalai-w120-256x640cyl",
             "epcache-256px-phase0/physicalai-train-e438721ae894"),
            ("thor", "/home/nvidia/epcache/epcache-256px-phase0/"
                     "physicalai-train-e438721ae894"),
            ("thor", "/home/nvidia/epcache_prefix/physicalai-train-e438721ae894"),
            ("pod5", "/workspace/data/epcache-256px-phase0/"
                     "physicalai-train-e438721ae894"),
            ("pod5", "/workspace/data/_stage/physicalai-train-e438721ae894"),
            ("pod4", "/workspace/data/physicalai-train-e438721ae894"),
            ("pod4", "/workspace/rescue/data/physicalai_v2"),
        ],
    ),
    Artifact(
        key="raw-val-epcache-256px",
        kind="raster",
        desc="Raw parity VAL epcache, 256 px — the REF-C val raster; every "
             "published REF-C number is computed on the first 40 episodes",
        members=600,
        parity="physicalai-val-0c5f7dac3b11 / 600 clips",
        candidates=[
            ("hf:Sayood/tanitad-physicalai-w120-256x640cyl",
             "epcache-256px-phase0/physicalai-val-0c5f7dac3b11"),
            ("thor", "/home/nvidia/valdata/physicalai-val-0c5f7dac3b11"),
            ("pod5", "/workspace/data/epcache-256px-phase0/"
                     "physicalai-val-0c5f7dac3b11"),
            ("pod4", "/workspace/data/physicalai-val-0c5f7dac3b11"),
        ],
    ),
    Artifact(
        key="raw-val-epcache-256px-eval40",
        kind="raster",
        desc="The 40 val episodes the canonical eval actually reads "
             "(ep_00000..ep_00039). Tracked separately from the 600 because "
             "it is the subset that decides published numbers.",
        members=40,
        parity="first 40 of physicalai-val-0c5f7dac3b11",
        candidates=[
            ("hf:Sayood/tanitad-physicalai-w120-256x640cyl",
             "epcache-256px-phase0/physicalai-val-0c5f7dac3b11"),
            ("thor", "/home/nvidia/valdata/physicalai-val-0c5f7dac3b11"),
            ("pod5", "/workspace/data/epcache-256px-phase0/"
                     "physicalai-val-0c5f7dac3b11"),
        ],
        pattern="ep_0000*.pt ep_0001*.pt ep_0002*.pt ep_0003*.pt",
    ),
    Artifact(
        key="w120-train-cyl",
        kind="corpus",
        desc="w120 256x640 cylindrical TRAIN cache (v2 corpus)",
        members=2400,
        parity="physicalai-train-e438721ae894-w120-256x640cyl",
        candidates=[
            ("hf:Sayood/tanitad-physicalai-w120-256x640cyl",
             "physicalai-train-e438721ae894-w120-256x640cyl"),
            ("pod5", "/workspace/data/physicalai-train-e438721ae894-w120-256x640cyl"),
            ("pod4", "/workspace/data/physicalai-train-e438721ae894-w120-256x640cyl"),
            ("thor", "/home/nvidia/traindata/"
                     "physicalai-train-e438721ae894-w120-256x640cyl"),
        ],
        pattern="*.v2ep.pt",
    ),
    Artifact(
        key="w120-val-cyl",
        kind="raster",
        desc="w120 256x640 cylindrical VAL cache (v2 corpus)",
        members=600,
        parity="physicalai-val-0c5f7dac3b11-w120-256x640cyl",
        candidates=[
            ("hf:Sayood/tanitad-physicalai-w120-256x640cyl",
             "physicalai-val-0c5f7dac3b11-w120-256x640cyl"),
            ("pod5", "/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl"),
            ("thor", "/home/nvidia/valdata/physicalai-val-0c5f7dac3b11-w120-256x640cyl"),
        ],
        pattern="*.v2ep.pt",
    ),
    Artifact(
        key="anchors-flagship-v4-dense",
        kind="anchors",
        desc="flagship_v4_anchors_dense.pt — the 256x20x2 dense anchor set the "
             "v4/v5 anchored-diffusion planner is defined against",
        candidates=[
            ("repo", "TanitAD Research Hub/Architecture & Inference/"
                     "Implementation/incoming/2026-07-28-pod-migration-rescue/"
                     "flagship_v4_anchors_dense.pt"),
            ("github", "TanitAD Research Hub/Architecture & Inference/"
                       "Implementation/incoming/2026-07-28-pod-migration-rescue/"
                       "flagship_v4_anchors_dense.pt"),
            ("pod5", "/workspace/experiments/flagship_v4_anchors_dense.pt"),
            ("pod4", "/workspace/experiments/flagship_v4_anchors_dense.pt"),
            ("thor", "/home/nvidia/models/flagship_v4_anchors_dense.pt"),
        ],
    ),
    Artifact(
        key="anchors-refc-small64",
        kind="anchors",
        desc="refc_anchors_small64.pt — REF-C small anchor set",
        candidates=[
            ("repo", "TanitAD Research Hub/Benchmarks & Eval/Implementation/"
                     "incoming/2026-07-22-refc-small-30k/refc_anchors_small64.pt"),
            ("github", "TanitAD Research Hub/Benchmarks & Eval/Implementation/"
                       "incoming/2026-07-22-refc-small-30k/refc_anchors_small64.pt"),
            ("pod5", "/workspace/experiments/refc_anchors_small64.pt"),
            ("thor", "/home/nvidia/models/refc_anchors_small64.pt"),
        ],
    ),
    Artifact(
        key="anchors-dev256",
        kind="anchors",
        desc="anchors_dev256.pt — per-candidate label anchor set",
        candidates=[
            ("repo", "TanitAD Research Hub/Architecture & Inference/"
                     "Implementation/incoming/2026-07-27-percandidate-labels/"
                     "raw/anchors_dev256.pt"),
            ("github", "TanitAD Research Hub/Architecture & Inference/"
                       "Implementation/incoming/2026-07-27-percandidate-labels/"
                       "raw/anchors_dev256.pt"),
            ("thor", "/home/nvidia/models/anchors_dev256.pt"),
        ],
    ),
    Artifact(
        key="evaldumps-windows-fan",
        kind="evaldump",
        desc="Banked per-window eval dumps (windows_*.pt / fan_*.pt) — the "
             "0-GPU re-analysis surface behind the CI recomputes",
        members=29,
        candidates=[
            ("repo", "taniteval/results"),
            ("github", "taniteval/results"),
            ("pod5", "/workspace/TanitAD/taniteval/results"),
            ("pod4", "/workspace/TanitAD/taniteval/results"),
            ("thor", "/home/nvidia/TanitAD/taniteval/results"),
        ],
        pattern="windows_*.pt fan_*.pt",
    ),
]


# ---------------------------------------------------------------------------
# Remote probe. One ssh per host, all paths in a single payload — a per-path
# ssh would be N round trips and would make a partial network failure look like
# selective absence.
#
# `ssh -n` is mandatory: a nested ssh inside a piped script EATS THE REST OF
# THE SCRIPT'S STDIN and the tail silently never runs (CLAUDE.md).
# ---------------------------------------------------------------------------
REMOTE_TEMPLATE = r"""
for spec in {specs}; do
  d=${{spec%%::*}}; pat=${{spec#*::}}
  if [ -e "$d" ]; then
    if [ -d "$d" ]; then
      n=0
      for g in $pat; do
        c=$(ls -1 "$d"/$g 2>/dev/null | wc -l); n=$((n+c))
      done
      b=$(du -sb "$d" 2>/dev/null | cut -f1)
      echo "OK|$d|dir|$n|${{b:-NA}}"
    else
      b=$(stat -c%s "$d" 2>/dev/null)
      echo "OK|$d|file|1|${{b:-NA}}"
    fi
  else
    echo "MISS|$d|-|0|0"
  fi
done
echo "__CENSUS_DONE__"
"""


def ssh_client() -> str:
    """Native OpenSSH on win32 — the MSYS one deadlocks under subprocess pipes
    against *busy* hosts, which reads exactly like a fleet outage and is not
    one (see :func:`tools.fleet_probe.ssh_client` for the measurement)."""
    if sys.platform == "win32":
        native = Path(r"C:\Windows\System32\OpenSSH\ssh.exe")
        if native.exists():
            return str(native)
    return "ssh"


def build_remote_script(specs: list[tuple[str, str]]) -> str:
    """specs = [(path, glob_pattern)] -> a single POSIX sh payload."""
    quoted = " ".join(
        "'" + f"{path}::{pattern}".replace("'", "'\\''") + "'"
        for path, pattern in specs
    )
    return REMOTE_TEMPLATE.format(specs=quoted)


def parse_remote_output(text: str) -> tuple[dict[str, dict], bool]:
    """-> ({path: {status,count,bytes}}, completed).

    ``completed`` is False unless the sentinel arrived. A truncated payload
    MUST NOT be read as "the rest was absent" — that is the have=0 defect, and
    it is the single most dangerous misreading in this whole file.
    """
    out: dict[str, dict] = {}
    completed = "__CENSUS_DONE__" in text
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 5 or parts[0] not in ("OK", "MISS"):
            continue
        tag, path, node, count, nbytes = parts
        try:
            cnt = int(count)
        except ValueError:
            cnt = None
        try:
            nb = int(nbytes)
        except ValueError:
            nb = None
        out[path] = {
            "status": PRESENT if tag == "OK" else ABSENT,
            "count": cnt,
            "bytes": nb,
            "node": node,
        }
    return out, completed


def probe_host(alias: str, specs: list[tuple[str, str]], timeout: int,
               ssh_bin: str | None = None) -> tuple[dict[str, dict], str | None]:
    """Probe one host. Returns (results, error). On error every path is
    UNKNOWN — never ABSENT."""
    script = build_remote_script(specs)
    # The payload travels in ARGV, not stdin, and stdin is explicitly
    # /dev/null. Measured 2026-08-03: piping the script to `ssh -n ... sh`
    # returns an empty stream every time, because `-n` *is* "redirect stdin
    # from /dev/null" — the script never reaches the remote `sh`. It presents
    # as "incomplete payload (no sentinel and no stderr)" on every host at
    # once, i.e. it looks exactly like a fleet-wide outage. Keeping stdin at
    # /dev/null also honours the standing rule that a nested ssh inside a piped
    # script eats the rest of that script's stdin.
    cmd = [ssh_bin or ssh_client(), "-o", "ConnectTimeout=12",
           "-o", "BatchMode=yes", alias, script]
    try:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                              capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {}, f"ssh timeout after {timeout}s"
    except OSError as exc:
        return {}, f"ssh failed: {exc}"
    text = proc.stdout.decode("utf-8", "replace")
    results, completed = parse_remote_output(text)
    if not completed:
        err = (proc.stderr.decode("utf-8", "replace").strip().splitlines()
               or ["no sentinel and no stderr"])[-1]
        # Partial output is retained but every path is downgraded: we cannot
        # tell a real MISS from a truncated stream.
        for v in results.values():
            if v["status"] == ABSENT:
                v["status"] = UNKNOWN
        return results, f"incomplete payload ({err})"
    return results, None


# ---------------------------------------------------------------------------
# HuggingFace probe
# ---------------------------------------------------------------------------
def probe_hf(repo_ids: set[str], keys_path: Path | None = None
             ) -> tuple[dict[str, dict], dict[str, str]]:
    """-> ({repo_id: {path: {size, sha256}}}, {repo_id: error}).

    The token is read in place and never printed, echoed, or placed in argv.
    """
    trees: dict[str, dict] = {}
    errors: dict[str, str] = {}
    keys = keys_path or (REPO_ROOT / "Keys.txt")
    try:
        import truststore  # certifi fails behind this box's TLS proxy
        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001  (optional dependency)
        pass
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        return {}, {r: f"huggingface_hub unavailable: {exc}" for r in repo_ids}
    try:
        match = re.search(r"hf_[A-Za-z0-9]+",
                          keys.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        return {}, {r: f"cannot read token file: {exc}" for r in repo_ids}
    if not match:
        return {}, {r: "no hf_ token in Keys.txt" for r in repo_ids}
    token = match.group(0)
    api = HfApi(token=token)
    for rid in sorted(repo_ids):
        kind = "dataset" if "physicalai" in rid or "dataset" in rid else "model"
        try:
            entries = api.list_repo_tree(rid, repo_type=kind, recursive=True,
                                         expand=True, token=token)
            tree = {}
            for f in entries:
                if getattr(f, "size", None) is None:
                    continue
                lfs = getattr(f, "lfs", None)
                tree[f.path] = {
                    "size": f.size,
                    "sha256": getattr(lfs, "sha256", None) if lfs else None,
                }
            trees[rid] = tree
        except Exception as exc:  # noqa: BLE001
            errors[rid] = f"{type(exc).__name__}: {exc}"
    return trees, errors


def count_hf_members(tree: dict[str, dict], prefix: str, pattern: str
                     ) -> tuple[int, int]:
    """Count files under `prefix` matching any glob in `pattern`."""
    import fnmatch
    globs = pattern.split()
    n = 0
    total = 0
    pre = prefix.rstrip("/") + "/"
    for path, meta in tree.items():
        if not path.startswith(pre):
            continue
        name = path[len(pre):]
        if "/" in name:
            continue
        if any(fnmatch.fnmatch(name, g) for g in globs):
            n += 1
            total += meta["size"]
    return n, total


# ---------------------------------------------------------------------------
# Local repo probe
# ---------------------------------------------------------------------------
def probe_git_remote(path: str, pattern: str, root: Path | None = None,
                     runner=None) -> dict:
    """Is this path present in a blob on a remote-tracking branch?

    A git-tracked file that has been PUSHED lives on GitHub as well as on this
    disk, and that is a genuine second machine. The first cut of this census
    counted only the working tree and therefore reported four artifacts as
    SINGLE_COPY that were in fact mirrored on origin — the same
    absence-at-one-location error the module exists to prevent, committed by
    the module itself.

    Honest limits, both recorded rather than papered over:
      * ``origin/*`` refs are a LOCAL CACHE of the remote. This proves the blob
        was pushed as of the last fetch, not that it is on the remote right
        now.
      * With no remote-tracking refs at all we return UNKNOWN, never ABSENT —
        "we never fetched" is not "it was never pushed".
    """
    run = runner or (lambda args: subprocess.run(
        args, cwd=str(root or REPO_ROOT), capture_output=True,
        stdin=subprocess.DEVNULL, timeout=60))
    try:
        proc = run(["git", "branch", "-r", "--format=%(refname:short)"])
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": UNKNOWN, "count": None, "bytes": None,
                "error": f"git unavailable: {exc}"}
    refs = [r.strip() for r in proc.stdout.decode("utf-8", "replace").splitlines()
            if r.strip() and "->" not in r]
    if not refs:
        return {"status": UNKNOWN, "count": None, "bytes": None,
                "error": "no remote-tracking refs — never fetched?"}
    globs = pattern.split()
    for ref in refs:
        try:
            p = run(["git", "ls-tree", "-r", "--name-only", ref, "--", path])
        except (OSError, subprocess.SubprocessError):
            continue
        names = [n for n in p.stdout.decode("utf-8", "replace").splitlines()
                 if n.strip()]
        if not names:
            continue
        import fnmatch
        if len(names) == 1 and names[0] == path:
            return {"status": PRESENT, "count": 1, "bytes": None, "ref": ref}
        pre = path.rstrip("/") + "/"
        n = sum(1 for x in names
                if x.startswith(pre) and "/" not in x[len(pre):]
                and any(fnmatch.fnmatch(x[len(pre):], g) for g in globs))
        if n:
            return {"status": PRESENT, "count": n, "bytes": None, "ref": ref}
    return {"status": ABSENT, "count": 0, "bytes": 0}


def probe_repo(path: str, pattern: str, root: Path | None = None) -> dict:
    base = (root or REPO_ROOT) / path
    if not base.exists():
        return {"status": ABSENT, "count": 0, "bytes": 0}
    if base.is_file():
        return {"status": PRESENT, "count": 1, "bytes": base.stat().st_size}
    n = 0
    total = 0
    for glob in pattern.split():
        for f in base.glob(glob):
            if f.is_file():
                n += 1
                total += f.stat().st_size
    return {"status": PRESENT if n else ABSENT, "count": n, "bytes": total}


# ---------------------------------------------------------------------------
# Census assembly
# ---------------------------------------------------------------------------
def classify(status: str, count: int | None, expected: int | None) -> str:
    """PRESENT only when the member contract is met. A short directory is
    PARTIAL and does NOT count as a copy."""
    if status != PRESENT:
        return status
    if expected is None:
        return PRESENT if (count or 0) >= 1 else ABSENT
    if count is None:
        return UNKNOWN
    if count == 0:
        return ABSENT
    return PRESENT if count >= expected else PARTIAL


def build_census(host_results: dict[str, dict],
                 host_errors: dict[str, str],
                 hf_trees: dict[str, dict],
                 hf_errors: dict[str, str],
                 repo_root: Path | None = None,
                 artifacts: list[Artifact] | None = None) -> dict:
    arts = artifacts if artifacts is not None else ARTIFACTS
    out: dict = {"min_copies": MIN_COPIES, "artifacts": {}, "warnings": []}

    for art in arts:
        locs = []
        for host, path in art.candidates:
            if host == "repo":
                r = probe_repo(path, art.pattern, repo_root)
                st = classify(r["status"], r["count"], art.members)
                locs.append({"host": "repo", "path": path, "status": st,
                             "count": r["count"], "bytes": r["bytes"]})
            elif host == "github":
                r = probe_git_remote(path, art.pattern, repo_root)
                st = classify(r["status"], r["count"], art.members)
                loc = {"host": "github", "path": path, "status": st,
                       "count": r["count"], "bytes": r["bytes"]}
                if r.get("ref"):
                    loc["ref"] = r["ref"]
                if r.get("error"):
                    loc["error"] = r["error"]
                locs.append(loc)
            elif host.startswith("hf:"):
                rid = host[3:]
                if rid in hf_errors:
                    locs.append({"host": host, "path": path,
                                 "status": UNREACHABLE, "count": None,
                                 "bytes": None, "error": hf_errors[rid]})
                    continue
                tree = hf_trees.get(rid)
                if tree is None:
                    locs.append({"host": host, "path": path, "status": UNKNOWN,
                                 "count": None, "bytes": None,
                                 "error": "repo not probed"})
                    continue
                n, b = count_hf_members(tree, path, art.pattern)
                # A single-file artifact is addressed exactly, not by glob.
                if art.members is None and n == 0 and path in tree:
                    n, b = 1, tree[path]["size"]
                st = classify(PRESENT if n else ABSENT, n, art.members)
                locs.append({"host": host, "path": path, "status": st,
                             "count": n, "bytes": b})
            else:
                if host in host_errors:
                    locs.append({"host": host, "path": path,
                                 "status": UNREACHABLE, "count": None,
                                 "bytes": None, "error": host_errors[host]})
                    continue
                res = host_results.get(host, {}).get(path)
                if res is None:
                    locs.append({"host": host, "path": path, "status": UNKNOWN,
                                 "count": None, "bytes": None,
                                 "error": "path not in probe output"})
                    continue
                st = classify(res["status"], res["count"], art.members)
                locs.append({"host": host, "path": path, "status": st,
                             "count": res["count"], "bytes": res["bytes"]})

        # Copies are counted over DISTINCT machines: two paths on one host die
        # together and are one copy.
        machines = {l["host"] for l in locs if l["status"] in COUNTS_AS_COPY}
        unknown = [l for l in locs if l["status"] not in CONCLUSIVE]
        partial = [l for l in locs if l["status"] == PARTIAL]

        durable = {m for m in machines if is_durable_host(m)}
        entry = {
            "kind": art.kind,
            "desc": art.desc,
            "expected_members": art.members,
            "parity": art.parity,
            "copies": len(machines),
            "durable_copies": len(durable),
            "volatile_machines": sorted(machines - durable),
            "copy_machines": sorted(machines),
            "locations": locs,
            "unknown_locations": len(unknown),
            "partial_locations": [f'{l["host"]}:{l["path"]}' for l in partial],
        }
        if len(machines) == 0:
            entry["verdict"] = "ZERO_COPIES" if not unknown else "UNRESOLVED"
        elif len(machines) < MIN_COPIES:
            entry["verdict"] = "SINGLE_COPY"
        else:
            entry["verdict"] = "OK"
        # An artifact that reaches the bar only because some location could not
        # be probed is still OK, but the uncertainty must be visible.
        if unknown and entry["verdict"] == "OK":
            entry["note"] = (f"{len(unknown)} location(s) UNKNOWN/UNREACHABLE "
                             f"— copy count is a LOWER BOUND")
        # Reaching MIN_COPIES only by counting a RENTED pod is not durability.
        # This does not change the verdict (the copies are real today) but it
        # must be visible, because the pod can be gone tomorrow.
        if entry["verdict"] == "OK" and len(durable) < MIN_COPIES:
            entry["volatility_warning"] = (
                f"only {len(durable)} durable copy/copies; the rest are on "
                f"rented pods {entry['volatile_machines']} — one termination "
                f"from SINGLE_COPY")
        out["artifacts"][art.key] = entry

    for host, err in sorted(host_errors.items()):
        out["warnings"].append(
            f"UNREACHABLE {host}: {err} — its paths are UNKNOWN, not absent")
    for rid, err in sorted(hf_errors.items()):
        out["warnings"].append(
            f"UNREACHABLE hf:{rid}: {err} — its paths are UNKNOWN, not absent")
    return out


def census_exit_code(census: dict) -> int:
    verdicts = [a["verdict"] for a in census["artifacts"].values()]
    if any(v in ("ZERO_COPIES", "UNRESOLVED") for v in verdicts):
        return 2
    if any(v == "SINGLE_COPY" for v in verdicts):
        return 1
    return 0


def format_table(census: dict) -> str:
    rows = ["", f"{'artifact':34s} {'kind':9s} {'copies':>6s}{'dur':>4s}  "
                f"{'verdict':12s} where", "-" * 122]
    for key, a in sorted(census["artifacts"].items(),
                         key=lambda kv: (kv[1]["copies"], kv[0])):
        mark = {"OK": "  ", "SINGLE_COPY": "!!", "ZERO_COPIES": "XX",
                "UNRESOLVED": "??"}[a["verdict"]]
        where = ",".join(a["copy_machines"]) or "-"
        if a["unknown_locations"]:
            where += f"  (+{a['unknown_locations']} UNKNOWN)"
        if a.get("volatility_warning"):
            mark = "~~"
            where += "  [VOLATILE]"
        rows.append(f"{mark}{key:32s} {a['kind']:9s} {a['copies']:6d}"
                    f"{a.get('durable_copies', 0):4d}  "
                    f"{a['verdict']:12s} {where}")
    if census["warnings"]:
        rows += ["", "WARNINGS (these are UNKNOWNs, not absences):"]
        rows += [f"  - {w}" for w in census["warnings"]]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", metavar="PATH", help="write the census JSON here")
    ap.add_argument("--no-hf", action="store_true", help="skip the HF probe")
    ap.add_argument("--hosts", nargs="*", help="probe only these hosts")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args(argv)

    wanted_hosts: dict[str, list[tuple[str, str]]] = {}
    hf_repos: set[str] = set()
    for art in ARTIFACTS:
        for host, path in art.candidates:
            if host in ("repo", "github"):
                continue
            if host.startswith("hf:"):
                hf_repos.add(host[3:])
                continue
            wanted_hosts.setdefault(host, []).append((path, art.pattern))

    # Map census host names -> ssh aliases via the discovered fleet, so the
    # host list is never a second hardcoded cache of the ssh config.
    try:
        from fleet_probe import load_fleet
        fleet, fleet_warnings = load_fleet()
    except Exception as exc:  # noqa: BLE001
        fleet, fleet_warnings = {}, [f"fleet discovery failed: {exc}"]

    host_results: dict[str, dict] = {}
    host_errors: dict[str, str] = {}
    for host, specs in sorted(wanted_hosts.items()):
        if args.hosts and host not in args.hosts:
            host_errors[host] = "not selected (--hosts)"
            continue
        alias = fleet.get(host, {}).get("ssh") or f"tanitad-{host}"
        res, err = probe_host(alias, specs, args.timeout)
        if err and not res:
            host_errors[host] = err
        else:
            host_results[host] = res
            if err:
                host_errors[host] = err
    hf_trees: dict[str, dict] = {}
    hf_errors: dict[str, str] = {}
    if args.no_hf:
        hf_errors = {r: "skipped (--no-hf)" for r in hf_repos}
    else:
        hf_trees, hf_errors = probe_hf(hf_repos)

    census = build_census(host_results, host_errors, hf_trees, hf_errors)
    census["fleet_warnings"] = fleet_warnings
    print(format_table(census))
    if args.json:
        Path(args.json).write_text(json.dumps(census, indent=1),
                                   encoding="utf-8")
        print(f"\nwrote {args.json}")
    return census_exit_code(census)


if __name__ == "__main__":
    raise SystemExit(main())
