"""secret_scan -- the credential gate C111 asked for, as a MECHANISM.

WHY THIS EXISTS
---------------
C111 (2026-08-18): a live Hugging Face token with WRITE access to the ``Sayood/``
namespace sat on line 11 of a rescued plaintext run log. Our procedure staged it,
committed it, and would have pushed it. **GitHub's push protection caught it --
nothing of ours did.** C111's rule was written the same day: *"any bulk import
from a machine -- rescue, pull, backup -- is SCANNED FOR CREDENTIAL PATTERNS
BEFORE IT IS STAGED."*

C117 then recorded, at three probes, that **no scanner exists**. That absence
claim is WRONG and this module's first job is to say so: ``tools/safe_commit.py``
has carried a six-pattern content scan since 2026-07-25 (commit ``37158f7``), and
MEASURED here it catches the exact C111 shape. All three C117 probes were
structurally unable to see it -- they searched script/test NAMES, grepped for the
THIRD-PARTY tool names ``detect-secrets``/``trufflehog``/``gitleaks``, and read
``pre-commit`` + ``.github/workflows/`` + the operating standard. A home-grown
scanner living inside a commit wrapper is invisible to all three.

**So C111's root cause is not a missing scanner. It is a scanner NOTHING CALLS
-- C108's class exactly, and the 2026-07-26 program harvest had already written
down that ``safe_commit.py`` is "imported_by: Nothing" and "not referenced by
CLAUDE.md at all".** The failure was 24 days old and documented.

⇒ This module therefore ships two things the old one could not:

1. **A scan that binds.** See ``--install-hook``: a ``pre-commit`` hook so the
   scan runs on EVERY commit whether or not anyone remembers a procedure, plus
   ``stack/tests/test_secret_scan.py`` which the mandated ``pytest``/``ci_gate``
   gate already runs and which fails when the hook is missing or stale.
2. **Scopes the old one structurally could not reach.** ``safe_commit`` scanned
   ``git diff --cached``, i.e. *diff text*: a token inside a blob git classifies
   as BINARY produces "Binary files differ" and no content lines at all, so it
   was invisible (MEASURED). This module reads staged BLOBS via
   ``git cat-file``, walks arbitrary TREES before anything is staged (C111's
   literal rule), and audits HISTORY.

WHAT IT NEVER DOES
------------------
⛔ **It never prints a matched value.** Every finding is (path, line, pattern
name, redacted length). ``Keys.txt`` is scanned like any other file and reported
by path alone -- its contents are never read into an argument, echoed, or logged.
That is how C111 itself was handled, deliberately.

FALSE POSITIVES ARE A DESIGN PARAMETER
--------------------------------------
``pod_git_drift.py``'s first widening produced 63.6 % artifact rows and had to be
narrowed with reasons (C110). A gate nobody can live with gets switched off
(C118). So:

* provider shapes carry **length floors** -- the brief's loose ``hf_[A-Za-z0-9]+``
  matches this repo's own legitimate ``hf_export.py`` / ``hf_relay`` identifiers,
  so Tier A demands the real token length and the loose form is ADVISORY only;
* the generic ``token=``/``secret=``/``password=``/``api_key=`` rule is gated on
  **Shannon entropy**, a mixed-alphabet requirement, a placeholder deny-list, and
  a hex-digest exclusion -- this repo commits thousands of md5/sha256 digests;
* every filter is printed as DATA next to its counts (C110: a count is a claim
  about the filter until the filter is stated), and skipped files are COUNTED,
  never silently dropped.

MODES
-----
::

    python tools/secret_scan.py --staged            # the gate: staged blobs
    python tools/secret_scan.py --tree <dir>        # C111: BEFORE staging
    python tools/secret_scan.py --tracked           # every tracked file
    python tools/secret_scan.py --history           # every blob in the object DB
    python tools/secret_scan.py --install-hook      # bind it to git
    python tools/secret_scan.py --check-hook        # is the gate installed?
    python tools/secret_scan.py --tree X --json out.json

Exit codes: 0 = clean · 1 = BLOCKING findings or SCAN UNUSABLE · 2 = usage error
3 = git unavailable / not a repo.

Stdlib-only, ASCII-clean stdout (the cp1252 console lesson), OS-agnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from fnmatch import fnmatch
from pathlib import Path

HOOK_MARKER = "TANITAD-SECRET-GATE v1"

# --------------------------------------------------------------------- patterns
#
# Tier A -- provider token SHAPES. A hit is BLOCKING and no flag overrides it.
# Each carries the length floor that separates a real credential from an ordinary
# identifier. These are byte-oriented (ASCII), so they run on binary blobs too.
TOKEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # hf_ + 34 alnum. The floor is what keeps `hf_export.py`, `hf_relay`,
    # `hf_repo_state_2026-07-25.json` out of the blocking tier.
    ("huggingface", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    # OpenAI / Anthropic style, including the `sk-ant-` and `sk-proj-` variants.
    ("openai-anthropic", re.compile(r"\bsk-(?:ant-|proj-|or-)?[A-Za-z0-9_\-]{20,}\b")),
    # GitHub classic PATs / OAuth / server / refresh tokens: ghp_ gho_ ghu_ ghs_ ghr_
    ("github-pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    # GitHub fine-grained PAT.
    ("github-fine", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b")),
    # AWS long-term (AKIA) and temporary (ASIA) access key ids.
    ("aws-akid", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("slack", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("google-api", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]

# Tier A2 -- the brief's "generic high-entropy assignment". This is where every
# false positive comes from, so it is the most heavily gated rule in the file.
ASSIGN_RE = re.compile(
    r"""(?ix)
    \b(?P<key>
        api[_\-]?keys? | secret(?:[_\-]?key)? | tokens? | passwd | password |
        access[_\-]?key | auth[_\-]?token | client[_\-]?secret | bearer
    )\b
    \s* (?: [:=] | \s*=>\s* ) \s*
    (?P<q>["']?)
    (?P<val>[^\s"'`,;)}\]]{20,200})
    (?P=q)
    """
)

# Tier C -- the brief's loose hf_ form. ADVISORY only: it fires on this repo's
# own committed `hf_*` filenames and symbols. Never dropped silently -- a count
# is always printed so an operator can look.
LOOSE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("loose-hf", re.compile(r"\bhf_[A-Za-z0-9]{8,29}\b")),
]

# Tier B -- credential-shaped PATHS. Reported by path only; the file is never
# opened for this finding. `gotty_url.txt` is here because .gitignore records
# that a RunPod web-terminal URL carries a root credential.
SECRET_PATH_GLOBS = (
    "Keys.txt", "keys.txt", "KEYS.txt",
    "*.pem", "*.key", "*.pfx", "*.p12",
    ".env", ".env.*",
    "*_token.txt", "*token.json",
    "id_rsa", "id_ed25519", "*.ppk",
    ".netrc", "_netrc",
    "gotty_url.txt",
)

# ⛔ SUBSTRING globs are heuristics, not declarations, and they SELF-MATCHED.
# MEASURED 2026-08-18: `*secret*` fired on `tools/secret_scan.py` and
# `stack/tests/test_secret_scan.py` -- THIS SCANNER AND ITS OWN TESTS -- which
# made the repo-wide gate permanently red on the day it was written. That is the
# polling-monitor trap in the path tier: a filter matching the very thing that
# names it. ⇒ A substring glob does not apply to a SOURCE or DOC file, whose
# content is scanned anyway; it applies to plausible credential CONTAINERS.
# Exact names and extensions above stay unconditional -- those are declarations.
SECRET_NAME_SUBSTRING_GLOBS = ("*credential*", "*secret*")
# ⚠️ `.txt` is deliberately NOT here. It is the shape of `Keys.txt` -- this
# repo's actual key store -- and C111's token lived in a plaintext log. The
# self-match that forced this exemption was `.py`/`.md`; widening it to `.txt`
# would have exempted the very file class the incident came from. (Verified:
# no tracked basename carries "secret"/"credential" -- the 11 that look like it
# have the word in their DIRECTORY, and this rule matches the basename only.)
SOURCE_DOC_SUFFIXES = {
    ".py", ".pyi", ".md", ".rst", ".sh", ".bash", ".ps1", ".bat",
    ".js", ".ts", ".c", ".h", ".cc", ".cpp", ".hpp", ".go", ".rs", ".java",
    ".ipynb", ".html", ".css", ".sql", ".toml", ".cfg", ".ini", ".xml",
    # 2026-08-23: banked literature. arXiv 2305.18290 ("...Your Language Model
    # is SECRETLY a Reward Model") matched `*secret*` and blocked a 5,400-path
    # commit. A paper TITLE is not a credential file; its CONTENT is still
    # scanned like any other blob (Tier A unchanged).
    ".pdf",
}

# ⚠️ NARROWED WITH A REASON (MEASURED 2026-08-18, whole-repo run). The obvious
# glob `*.env` produced **4 of 7 blocking findings and 4 of 4 were artifacts**:
# in this programme `*.env` is the SUPERVISOR RUN MANIFEST convention
# (`stack/ops/runs.d/flagship-v5f-w120-30k.env`, `stack/scripts/pai_build.run.env`,
# ...), not a dotenv credential store. A gate that refuses a normal deliverable
# gets switched off inside a week (C118), so `*.env` is ADVISORY.
# ⭐ Nothing is lost that matters: these files are still CONTENT-scanned by
# Tier A/A2 like everything else -- the path rule is a redundant belt, the
# content scan is the braces. The dotenv shapes proper (`.env`, `.env.*`) stay
# BLOCKING because there the name really does declare the contents.
ADVISORY_PATH_GLOBS = ("*.env",)

# --- Tier A2 gates, each with the false positive it exists to stop -------------

# A value that is one of these is a placeholder, not a credential. Substring
# match, case-insensitive.
PLACEHOLDERS = (
    "xxx", "***", "redacted", "your_", "your-", "changeme", "change_me",
    "example", "dummy", "placeholder", "insert", "fill_me", "todo", "fixme",
    "<", ">", "${", "$(", "%(", "{{", "os.environ", "getenv", "env[",
    "none", "null", "true", "false", "...", "abcdef", "123456", "aaaa",
    "sample", "test_", "fake", "notasecret", "redact",
)

# 32/40/64 hex is md5/sha1/sha256. This repo commits thousands of digests and
# they are high-entropy by construction -- without this the rule would be noise.
HEXDIGEST_RE = re.compile(r"(?i)^[0-9a-f]{32}$|^[0-9a-f]{40}$|^[0-9a-f]{64}$")

# A value that is really a path, URL, or version spec.
NONSECRET_SHAPE_RE = re.compile(r"^(?:https?://|[A-Za-z]:[\\/]|[./~]|\w+/\w+/)")

# ⚠️ NARROWED WITH A REASON (MEASURED 2026-08-18, whole-repo run). Two of the
# seven blocking findings were the SAME shape: ``tokens = head.build_tokens(st4``
# -- a CODE EXPRESSION assigned to a variable called `tokens`, 21 chars at 4.01
# bits/char, sailing through the entropy gate. Real credentials come from a
# narrow alphabet (base64/base64url/hex plus the provider's separators), and
# code expressions do not: they carry brackets, dots-with-calls, commas.
# ⇒ Every character of the value must be in the credential alphabet.
CREDENTIAL_ALPHABET_RE = re.compile(r"^[A-Za-z0-9+/=_\-.:]+$")
# ...and a plain dotted identifier (`head.build_tokens`, `cfg.model.name`) is a
# symbol, never a credential, even though it satisfies the alphabet.
DOTTED_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")

MIN_ENTROPY_BITS = 3.4        # Shannon bits/char over the value
MIN_ASSIGN_LEN = 20

# --- walk filters -------------------------------------------------------------

DEFAULT_MAX_BYTES = 25 * 1024 * 1024
# Directories never worth walking. `.git` is excluded by default because its
# objects are zlib-compressed (a plaintext pattern cannot match) and it is the
# largest thing in the tree -- use --history for the object database instead.
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}
# Suffixes skipped, and the ONLY admissible reason to skip one: the bytes are
# ENTROPY-CODED, so a plaintext credential cannot be present as plaintext and no
# pattern could match it anyway.
#
# ⚠️ This list was NARROWED after a measurement. The first cut also carried
# `.bin`, `.npy`, `.idx`, `.h5`, `.parquet`, `.so`, `.dll`, `.exe`, `.pt` -- and
# the planted-token panel then showed ``--tree`` MISSING a token inside a `.bin`
# that ``--staged`` caught. Those containers are UNCOMPRESSED: an ASCII token
# survives verbatim inside them. Skipping them was the `pod_git_drift` mistake in
# miniature -- a filter chosen for byte volume rather than for whether the answer
# could be in there. Volume is handled by --max-bytes, which COUNTS what it skips.
SKIP_SUFFIXES = {
    # compressed archives
    ".zip", ".gz", ".tgz", ".xz", ".bz2", ".7z", ".rar", ".whl", ".jar",
    ".npz", ".pack",
    # compressed media
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".mp3", ".wav", ".flac",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf",
    ".woff", ".woff2", ".ttf", ".otf",
}


def filter_rule(max_bytes: int, include_git: bool) -> dict:
    """The filter, as DATA -- C110: a count is a claim about the filter until the
    filter is stated next to it."""
    return {
        "max_bytes": max_bytes,
        "skip_dirs": sorted(SKIP_DIRS - ({".git"} if include_git else set())),
        "skip_suffixes": sorted(SKIP_SUFFIXES),
        "tier_a_patterns": [n for n, _ in TOKEN_PATTERNS],
        "tier_a2_generic_assignment": {
            "min_len": MIN_ASSIGN_LEN,
            "min_entropy_bits_per_char": MIN_ENTROPY_BITS,
            "gates": ["placeholder-denylist", "must-mix-letters-and-digits",
                      "not-a-hex-digest(32/40/64)", "not-a-path-or-url",
                      "credential-alphabet-only", "not-a-dotted-identifier",
                      "text-files-only"],
        },
        "tier_b_path_globs": list(SECRET_PATH_GLOBS),
        "tier_b_advisory_path_globs": list(ADVISORY_PATH_GLOBS),
        "tier_c_advisory": [n for n, _ in LOOSE_PATTERNS],
        "_warning": (
            "THIS IS A FILTERED VIEW, NOT A CENSUS. A file outside this rule is "
            "invisible here BY CONSTRUCTION. Binary-suffixed and oversized files "
            "are COUNTED in 'skipped', never silently dropped."),
    }


# ------------------------------------------------------------------- redaction


def redact(match: str) -> str:
    """Never echo a credential. Scheme prefix + length only.

    C111 was handled exactly this way on purpose: the token was named by file and
    line and never reproduced."""
    head = match[:4] if len(match) > 4 else "?"
    return f"{head}***<{len(match)} chars, redacted>"


@dataclass
class Finding:
    kind: str        # "token" | "generic-assign" | "secret-path" | "loose"
    pattern: str     # pattern NAME, never the value
    path: str
    line: int        # 1-based; 0 when the finding is about the path itself
    detail: str      # ALWAYS redacted
    blocking: bool = True


# --------------------------------------------------------------------- scanning


def shannon_bits_per_char(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _assign_is_credential_shaped(val: str) -> bool:
    """All the Tier A2 gates, in the cheap-first order."""
    if len(val) < MIN_ASSIGN_LEN:
        return False
    low = val.lower()
    if any(p in low for p in PLACEHOLDERS):
        return False
    if HEXDIGEST_RE.match(val):
        return False
    if NONSECRET_SHAPE_RE.match(val):
        return False
    if not (any(c.isdigit() for c in val) and any(c.isalpha() for c in val)):
        return False
    # A sentence or an English-ish phrase: credentials do not carry these.
    if any(c in val for c in " \t"):
        return False
    if not CREDENTIAL_ALPHABET_RE.match(val):
        return False
    if DOTTED_IDENT_RE.match(val):
        return False
    return shannon_bits_per_char(val) >= MIN_ENTROPY_BITS


def looks_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


# ⚠️ SPEED IS A CORRECTNESS PROPERTY HERE, and it took two measurements to get.
# Cut 1 ran all 8 Tier A patterns per LINE: 661 MB in 106 s. Cut 2 merged them
# into one alternation and scanned whole-text: **still 6 MB/s** -- because every
# branch starts with `\b` and Python's `re` has no literal prefix to fast-scan
# on, so it tries all 8 alternatives at every one of 661 million positions.
# ⇒ Cut 3 gates each pattern behind its own MANDATORY LITERAL, tested with
# ``str.__contains__`` (C-speed two-way search, ~GB/s). A file with no `hf_` in
# it never pays for the Hugging Face regex.
# ⭐ This is not a heuristic and it drops nothing: every Tier A pattern provably
# REQUIRES one of its literals to match, so a file failing the pre-filter cannot
# contain a match. The `gh[pousr]_` class is enumerated rather than gated on
# "gh", which occurs in ordinary English ("through", "right") and would have
# made the pre-filter useless on prose.
PATTERN_LITERALS: dict[str, tuple[str, ...]] = {
    "huggingface": ("hf_",),
    "openai-anthropic": ("sk-",),
    "github-pat": ("ghp_", "gho_", "ghu_", "ghs_", "ghr_"),
    "github-fine": ("github_pat_",),
    "aws-akid": ("AKIA", "ASIA"),
    "slack": ("xox",),
    "google-api": ("AIza",),
    "private-key": ("-----BEGIN ",),
}

# The Tier A2 rule only has anything to say about text that mentions one of its
# key words, and most files mention none. Cheap substring gate before the regex.
ASSIGN_KEY_HINTS = ("api_key", "apikey", "api-key", "secret", "token", "passwd",
                    "password", "access_key", "access-key", "auth_token",
                    "client_secret", "bearer")
LOOSE_HINTS = ("hf_",)


def _line_of(text: str, offsets: list[int] | None, pos: int) -> int:
    """1-based line number for a match offset, via the newline index."""
    if offsets is None:
        return text.count("\n", 0, pos) + 1
    import bisect
    return bisect.bisect_right(offsets, pos) + 1


def scan_bytes(data: bytes, path: str) -> list[Finding]:
    """Scan one blob. Tier A runs on everything (token shapes are ASCII, so they
    survive inside a binary container -- that is the gap the diff-text scan had).
    Tier A2 runs on text only, because entropy on binary is pure noise."""
    out: list[Finding] = []
    binary = looks_binary(data)
    # latin-1 never raises and maps bytes 1:1, so ASCII patterns are exact and
    # line numbers stay true. Decoding as utf-8 would drop whole blobs on error.
    text = data.decode("latin-1", errors="replace")

    offsets: list[int] | None = None

    def line_at(pos: int) -> int:
        nonlocal offsets
        if offsets is None:
            offsets = []
            start = text.find("\n")
            while start != -1:
                offsets.append(start)
                start = text.find("\n", start + 1)
        return _line_of(text, offsets, pos)

    hit_lines: set[int] = set()
    for name, pat in TOKEN_PATTERNS:
        lits = PATTERN_LITERALS.get(name)
        if lits and not any(lit in text for lit in lits):
            continue                      # cannot match; proven, not guessed
        for m in pat.finditer(text):
            ln = line_at(m.start())
            hit_lines.add(ln)
            out.append(Finding("token", name, path, ln,
                               f"{name} token shape: {redact(m.group(0))}"))

    if not binary:
        low = text.lower()
        if any(h in low for h in ASSIGN_KEY_HINTS):
            for m in ASSIGN_RE.finditer(text):
                val = m.group("val")
                if _assign_is_credential_shaped(val):
                    ln = line_at(m.start())
                    hit_lines.add(ln)
                    out.append(Finding(
                        "generic-assign", "generic-assignment", path, ln,
                        f"high-entropy value assigned to '{m.group('key')}': "
                        f"{redact(val)} "
                        f"(entropy {shannon_bits_per_char(val):.2f} bits/char)"))

    if any(h in text for h in LOOSE_HINTS):
        for name, pat in LOOSE_PATTERNS:
            for m in pat.finditer(text):
                ln = line_at(m.start())
                if ln in hit_lines:      # already blocked on this line; not news
                    continue
                out.append(Finding(
                    "loose", name, path, ln,
                    f"loose {name} identifier {redact(m.group(0))} "
                    f"(usually a legitimate filename/symbol here)",
                    blocking=False))
    return out


def scan_path_shape(path: str) -> list[Finding]:
    """Tier B. Path only -- the file is NEVER opened for this finding."""
    base = path.replace("\\", "/").rsplit("/", 1)[-1]
    for g in SECRET_PATH_GLOBS:
        if fnmatch(base, g):
            return [Finding("secret-path", "credential-filename", path, 0,
                            f"filename matches a credential glob ({g})")]
    if Path(base).suffix.lower() not in SOURCE_DOC_SUFFIXES:
        for g in SECRET_NAME_SUBSTRING_GLOBS:
            if fnmatch(base, g):
                return [Finding("secret-path", "credential-filename", path, 0,
                                f"filename matches a credential glob ({g})")]
    for g in ADVISORY_PATH_GLOBS:
        if fnmatch(base, g):
            return [Finding("secret-path", "credential-filename-advisory", path, 0,
                            f"filename matches an ADVISORY glob ({g}); in this "
                            f"repo these are supervisor run manifests, and the "
                            f"content was scanned like any other file",
                            blocking=False)]
    return []


@dataclass
class Report:
    scope: str
    files_scanned: int = 0
    bytes_scanned: int = 0
    # How many files/blobs the scan was ASKED to read. If this is
    # positive and files_scanned is zero, the run read nothing and
    # its 'clean' verdict is worthless -- see .unusable below.
    candidates: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    # ⚠️ COUNTING a skip is not enough to keep the "filtered view, not a
    # census" promise honest -- an operator must be able to SEE which files
    # the answer could not have come from. Capped so a huge tree cannot
    # explode the report; the cap itself is recorded.
    skipped_paths: dict[str, list[str]] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    filter_rule: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    # ⛔ PARTIAL-failure accounting -- the counterpart of the 0-read guard.
    # MEASURED 2026-08-18 (the Drive mount flapping): --history read SOME
    # objects, printed rc=128 batch failures over ~1,200 others, and still
    # summarised "BLOCKING (0) -- clean", exit 0. The 0-read guard could not
    # see it because files_scanned was positive. objects_unread counts every
    # object the scan was ASKED to read and could not (batch rc!=0, an
    # enumerated object coming back 'missing', a short/truncated read);
    # batch_failures counts failed git invocations. A POLICY skip (too-large,
    # compressed-suffix) is never counted here -- those are stated filters.
    objects_unread: int = 0
    batch_failures: int = 0

    @property
    def partial(self) -> bool:
        """Some of what the scan was asked to read was never read."""
        return bool(self.objects_unread or self.batch_failures)

    @property
    def unusable_label(self) -> str:
        return "SCAN UNUSABLE (partial)" if self.partial else "SCAN UNUSABLE"

    @property
    def unusable_reason(self) -> str | None:
        """⛔ Why this report must NOT be read as a pass."""
        if self.partial:
            return (f"read {self.files_scanned} of {self.candidates} objects, "
                    f"{self.batch_failures} batch failures")
        if self.errors:
            return f"{len(self.errors)} read/git error(s) -- the scan is INCOMPLETE"
        if self.candidates and not self.files_scanned:
            return (f"{self.candidates} candidate(s) but ZERO were read -- "
                    f"a 'clean' verdict here would be an artifact")
        return None

    @property
    def unusable(self) -> bool:
        return self.unusable_reason is not None

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.blocking]

    @property
    def advisory(self) -> list[Finding]:
        return [f for f in self.findings if not f.blocking]

    def to_json(self) -> dict:
        return {
            "scope": self.scope,
            "candidates": self.candidates,
            "files_scanned": self.files_scanned,
            "unusable": self.unusable,
            "unusable_reason": self.unusable_reason,
            "partial": self.partial,
            "objects_unread": self.objects_unread,
            "batch_failures": self.batch_failures,
            "bytes_scanned": self.bytes_scanned,
            "skipped": self.skipped,
            "skipped_paths": self.skipped_paths,
            "skipped_paths_cap": SKIPPED_PATHS_CAP,
            "blocking_count": len(self.blocking),
            "advisory_count": len(self.advisory),
            "findings": [asdict(f) for f in self.findings],
            "filter_rule": self.filter_rule,
            "errors": self.errors,
        }


SKIPPED_PATHS_CAP = 200


def _bump(d: dict[str, int], k: str) -> None:
    d[k] = d.get(k, 0) + 1


def _skip(rep: "Report", k: str, path: str) -> None:
    """Count the skip AND name the file (up to a cap)."""
    _bump(rep.skipped, k)
    lst = rep.skipped_paths.setdefault(k, [])
    if len(lst) < SKIPPED_PATHS_CAP:
        lst.append(path)


def scan_tree(root: Path, max_bytes: int = DEFAULT_MAX_BYTES,
              include_git: bool = False, scope: str = "tree") -> Report:
    """C111's literal rule: scan an imported tree BEFORE anything is staged."""
    rep = Report(scope=f"{scope}:{root}", filter_rule=filter_rule(max_bytes, include_git))
    skip_dirs = set(SKIP_DIRS)
    if include_git:
        skip_dirs.discard(".git")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:                       # pragma: no cover
                rel = str(p)
            rep.candidates += 1
            rep.findings.extend(scan_path_shape(rel))
            if p.suffix.lower() in SKIP_SUFFIXES:
                _skip(rep, "compressed-suffix", rel)
                continue
            try:
                size = p.stat().st_size
            except OSError as exc:
                rep.errors.append(f"stat {rel}: {exc.__class__.__name__}")
                _skip(rep, "stat-error", rel)
                continue
            if size > max_bytes:
                _skip(rep, "too-large", rel)
                continue
            try:
                data = p.read_bytes()
            except OSError as exc:
                rep.errors.append(f"read {rel}: {exc.__class__.__name__}")
                _skip(rep, "read-error", rel)
                continue
            rep.files_scanned += 1
            rep.bytes_scanned += len(data)
            rep.findings.extend(scan_bytes(data, rel))
    return rep


# ------------------------------------------------------------------------- git


class GitError(RuntimeError):
    pass


def git(repo: Path, *args: str, check: bool = True,
        binary: bool = False, stdin: bytes | None = None):
    proc = subprocess.run(["git", *args], cwd=str(repo), input=stdin,
                          capture_output=True)
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} -> {proc.returncode}: "
                       f"{proc.stderr.decode('utf-8', 'replace').strip()}")
    if binary:
        return proc
    return proc.stdout.decode("utf-8", "replace")


def repo_root(start: Path) -> Path:
    return Path(git(start, "rev-parse", "--show-toplevel").strip())


def _cat_file_batch(repo: Path, revs: list[str], max_bytes: int
                    ) -> tuple[dict[str, bytes], dict[str, int], list[str], bool]:
    """Read many blobs in ONE git process. Binary-safe: this is what closes the
    'Binary files differ' hole in the old diff-text scan.

    ⛔ A FAILED git call is an ERROR, never a skip. MEASURED 2026-08-18: the
    Google-Drive-backed volume this repo lives on dropped mid-session, every
    ``git cat-file`` returned 128 ("not a git repository"), and ``--history``
    printed **"files scanned = 0 ... BLOCKING (0) -- clean", exit 0** -- a
    perfect all-clear produced by reading NOTHING. That is the fleet_probe rule
    in its purest form: absence of evidence is an ALARM, not an all-clear, and a
    guard that cannot see its subject manufactures assurance.

    Returns ``(got, skipped, errors, batch_failed)``. ``batch_failed`` is True
    when the git INVOCATION itself failed -- nonzero rc, a short/truncated
    stream, or an unparseable header (stream desync) -- so callers COUNT failed
    batches instead of inferring them from error strings. The same 2026-08-18
    flap also produced the PARTIAL variant of the incident above: some batches
    read, some failed rc=128, failures printed in scrollback, and the run still
    ended "BLOCKING (0) -- clean". A short body is never handed back as a
    scanned blob: the missing half is exactly where a credential could hide."""
    got: dict[str, bytes] = {}
    skipped: dict[str, int] = {}
    errors: list[str] = []
    if not revs:
        return got, skipped, errors, False
    payload = ("\n".join(revs) + "\n").encode("utf-8")
    proc = git(repo, "cat-file", "--batch", check=False, binary=True, stdin=payload)
    if proc.returncode != 0:
        errors.append(
            f"git cat-file --batch failed (rc={proc.returncode}) over "
            f"{len(revs)} object(s): "
            f"{proc.stderr.decode('utf-8', 'replace').strip()[:200]}")
        return got, skipped, errors, True
    batch_failed = False
    buf = proc.stdout
    pos = 0
    for rev in revs:
        nl = buf.find(b"\n", pos)
        if nl < 0:
            errors.append(f"git cat-file output truncated at object {rev} "
                          f"({len(revs)} requested) -- the scan is INCOMPLETE")
            batch_failed = True
            break
        header = buf[pos:nl].decode("utf-8", "replace")
        pos = nl + 1
        parts = header.rsplit(" ", 2)
        if len(parts) != 3 or parts[1] not in ("blob", "tree", "commit", "tag"):
            # "<rev> missing": no body follows. (Anything unrecognisable lands
            # here too; --history counts every unreturned rev as UNREAD.)
            _bump(skipped, "missing-or-not-blob")
            continue
        try:
            size = int(parts[2])
        except ValueError:
            errors.append(f"unparseable cat-file header at object {rev} -- "
                          f"stream desync, the rest of this batch is unreadable")
            batch_failed = True
            break
        body = buf[pos:pos + size]
        if len(body) < size:
            errors.append(f"git cat-file returned a SHORT body for {rev} "
                          f"({len(body)} of {size} bytes) -- the scan is "
                          f"INCOMPLETE")
            batch_failed = True
            break
        pos += size + 1                      # git appends a newline after body
        if parts[1] != "blob":
            # A tree/commit/tag STREAMS a body in --batch output too -- it must
            # be consumed (above) or every later header in this batch is read
            # from inside the previous body: a silent stream desync. The old
            # parser skipped without consuming.
            _bump(skipped, "missing-or-not-blob")
            continue
        if size > max_bytes:
            _bump(skipped, "too-large")
            continue
        got[rev] = body
    return got, skipped, errors, batch_failed


def staged_paths(repo: Path) -> list[str]:
    """The index, as repo-relative POSIX paths.

    ``-z`` because git C-quotes any path with a space otherwise -- and this repo
    is full of them (``TanitAD Research Lab/...``). An unquoted read here is the
    trap that has now caught three separate streams."""
    out = git(repo, "diff", "--cached", "--name-only", "-z",
              "--diff-filter=ACMR")
    return [p for p in out.split("\0") if p]


def scan_staged(repo: Path, max_bytes: int = DEFAULT_MAX_BYTES) -> Report:
    """The gate. Reads staged BLOBS, not the diff text."""
    rep = Report(scope="staged", filter_rule=filter_rule(max_bytes, False))
    paths = staged_paths(repo)
    rep.candidates = len(paths)
    for p in paths:
        rep.findings.extend(scan_path_shape(p))

    # git's OWN ignore verdict -- the tool that owns the fact (operating standard
    # 2). --no-index makes check-ignore answer for ALREADY-staged paths, which is
    # exactly the `git add -f Keys.txt` case; without it the probe is vacuous.
    # ⭐ 2026-08-23: a RENAME (R) of an already-tracked path is NOT a new
    # `git add -f` — that force-add was committed before. The Hub->Lab migration
    # re-staged 5,342 renames, 42 of them force-added media (the tracked
    # showcase-video corpus), and each was reported as fresh exposure. Only this
    # PATH probe is narrowed to A/C/M; content scanning below still covers
    # every staged blob, renames included.
    new_paths = [q for q in git(repo, "diff", "--cached", "--name-only", "-z",
                                 "--diff-filter=ACM").split("\0") if q]
    if new_paths:
        proc = subprocess.run(
            ["git", "check-ignore", "--no-index", "-z", "--stdin"],
            cwd=str(repo), input=("\0".join(new_paths) + "\0").encode("utf-8"),
            capture_output=True)
        for raw in proc.stdout.decode("utf-8", "replace").split("\0"):
            if raw:
                rep.findings.append(Finding(
                    "secret-path", "git-ignored-but-staged", raw, 0,
                    "staged despite being git-IGNORED (only 'git add -f' does this)"))

    revs = [f":{p}" for p in paths]
    # Staged keeps the errors->UNUSABLE path (any errs already invalidate the
    # verdict, non-zero exit). Strict per-object accounting is history-only:
    # here 'missing-or-not-blob' is a LEGITIMATE skip -- a staged submodule
    # gitlink has no blob behind it -- so counting it as a read failure would
    # turn a normal index into a false alarm.
    blobs, skipped, errs, _batch_failed = _cat_file_batch(repo, revs, max_bytes)
    rep.skipped.update(skipped)
    rep.errors.extend(errs)
    for p in paths:
        data = blobs.get(f":{p}")
        if data is None:
            continue
        rep.files_scanned += 1
        rep.bytes_scanned += len(data)
        rep.findings.extend(scan_bytes(data, p))
    return rep


def scan_tracked(repo: Path, max_bytes: int = DEFAULT_MAX_BYTES) -> Report:
    """Every tracked file, read from the working tree. This is the scope whose
    blocking count MUST be zero -- it is what git would publish."""
    rep = Report(scope="tracked", filter_rule=filter_rule(max_bytes, False))
    out = git(repo, "ls-files", "-z")
    paths = [p for p in out.split("\0") if p]
    for rel in paths:
        rep.findings.extend(scan_path_shape(rel))
        p = repo / rel
        if p.suffix.lower() in SKIP_SUFFIXES:
            _skip(rep, "compressed-suffix", rel)
            continue
        try:
            size = p.stat().st_size
        except OSError:
            _skip(rep, "missing-in-worktree", rel)
            continue
        if size > max_bytes:
            _skip(rep, "too-large", rel)
            continue
        try:
            data = p.read_bytes()
        except OSError as exc:
            rep.errors.append(f"read {rel}: {exc.__class__.__name__}")
            _skip(rep, "read-error", rel)
            continue
        rep.files_scanned += 1
        rep.bytes_scanned += len(data)
        rep.findings.extend(scan_bytes(data, rel))
    return rep


def scan_history(repo: Path, max_bytes: int = 2 * 1024 * 1024,
                 batch: int = 400) -> Report:
    """Every blob in the object database, REACHABLE and UNREACHABLE.

    The distinction is the whole point of the C111 audit: a pattern in a
    reachable blob means a credential is in committed history and is the PI's
    call (⛔ do NOT rewrite history from an agent). A pattern in an UNREACHABLE
    blob is debris from an undone commit -- C111 records exactly one such blob,
    from the reset-away commit -- and it disappears on ``git gc --prune=now``.

    ⛔ STRICT ACCOUNTING (the 2026-08-18 partial-failure hole). Every sha this
    function hands a batch was certified ``blob``, within the size cap, by the
    enumeration moments earlier -- so anything a batch does not hand back is an
    OBJECT-READ FAILURE (flapping mount, concurrent prune), never a policy
    skip, and it makes the whole report ``SCAN UNUSABLE (partial)`` with a
    non-zero exit. 'clean' is admissible ONLY when every enumerated object was
    actually scanned. The 0-read guard alone could not see this: with SOME
    objects read, ``files_scanned`` looked healthy while ~1,200 objects went
    unread behind printed-and-then-ignored rc=128 batch failures."""
    rep = Report(scope="history", filter_rule={
        "max_bytes": max_bytes,
        "objects": "git cat-file --batch-all-objects (reachable AND unreachable)",
        "tier_a_patterns": [n for n, _ in TOKEN_PATTERNS],
        "_note": "Tier A shapes only; the generic-assignment rule is not run "
                 "over history (its FP budget is tuned for the live tree).",
    })
    proc = git(repo, "cat-file", "--batch-all-objects", "--batch-check",
               "--buffer", check=False, binary=True)
    if proc.returncode != 0:
        # The ENUMERATION failed: candidates=0 used to sail past every guard
        # and report a perfect clean over an object DB it never saw.
        rep.batch_failures += 1
        rep.errors.append(
            f"git cat-file --batch-all-objects --batch-check failed "
            f"(rc={proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace').strip()[:200]} -- "
            f"the object database could not be enumerated")
    listing = proc.stdout.decode("utf-8", "replace")
    blob_shas: list[str] = []
    for ln in listing.splitlines():
        parts = ln.split()
        if len(parts) >= 3 and parts[1] == "blob":
            if int(parts[2]) > max_bytes:
                _bump(rep.skipped, "too-large")
                continue
            blob_shas.append(parts[0])
    rep.candidates = len(blob_shas)

    # ONE rev-list pass feeds both the REACHABLE set and the sha->path names.
    # Its rc is checked because these labels set f.blocking: a failed rev-list
    # used to silently mark EVERY blob UNREACHABLE, demoting a committed
    # credential to non-blocking -- exit 0 with the finding buried as advisory.
    proc = git(repo, "rev-list", "--objects", "--all", check=False, binary=True)
    if proc.returncode != 0:
        rep.batch_failures += 1
        rep.errors.append(
            f"git rev-list --objects --all failed (rc={proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace').strip()[:200]} -- "
            f"REACHABLE/UNREACHABLE labels (which set the blocking flag) "
            f"cannot be trusted")
    reachable: set[str] = set()
    path_of: dict[str, str] = {}
    for ln in proc.stdout.decode("utf-8", "replace").splitlines():
        bits = ln.split(" ", 1)
        sha = bits[0].strip()
        if not sha:
            continue
        reachable.add(sha)
        if len(bits) == 2 and bits[1]:
            path_of.setdefault(sha, bits[1])

    for i in range(0, len(blob_shas), batch):
        chunk = blob_shas[i:i + batch]
        blobs, skipped, errs, batch_failed = _cat_file_batch(repo, chunk, max_bytes)
        rep.errors.extend(errs)
        if batch_failed:
            rep.batch_failures += 1
        # 'missing' for a sha the enumeration JUST certified is a read failure
        # here, not a skip. (In --staged it stays a legitimate skip: a
        # submodule gitlink really has no blob behind it.)
        missing = skipped.pop("missing-or-not-blob", 0)
        if missing:
            rep.errors.append(
                f"{missing} enumerated object(s) came back missing/non-blob "
                f"from git cat-file --batch -- an object-read failure, not a "
                f"skip")
        for k, v in skipped.items():
            rep.skipped[k] = rep.skipped.get(k, 0) + v
        # Requested minus scanned minus stated-policy skips = objects the scan
        # could not read. Any nonzero count makes the verdict inadmissible.
        unread = len(chunk) - len(blobs) - skipped.get("too-large", 0)
        if unread > 0:
            rep.objects_unread += unread
        for sha, data in blobs.items():
            rep.files_scanned += 1
            rep.bytes_scanned += len(data)
            where = "REACHABLE" if sha in reachable else "UNREACHABLE"
            name = path_of.get(sha, "<no path in any tree>")
            label = f"{where} blob {sha[:12]} ({name})"
            for f in scan_bytes(data, label):
                if f.kind == "loose":
                    continue                      # advisory noise at this scale
                f.blocking = (where == "REACHABLE")
                rep.findings.append(f)
    return rep


# --------------------------------------------------------------------- the hook

HOOK_BODY = f"""#!/bin/sh
# {HOOK_MARKER} -- installed by: python tools/secret_scan.py --install-hook
#
# C111: a live HF token reached a local commit and only GitHub's push protection
# stopped it. This hook is the control that should have. It scans the STAGED
# BLOBS (binary-safe) and refuses the commit on any provider-token shape,
# credential-shaped filename, git-ignored-but-staged path, or high-entropy
# credential assignment. It never prints a matched value.
#
# To bypass in a genuine emergency you must say so out loud:
#   SECRET_SCAN_SKIP=1 git commit ...     (and then explain it in the message)
set -e
if [ -n "$SECRET_SCAN_SKIP" ]; then
    echo "[secret_scan] SKIPPED via SECRET_SCAN_SKIP -- say why in the commit message"
    exit 0
fi
root=$(git rev-parse --show-toplevel)
# The interpreter is probed by RUNNING it, never by `command -v` alone.
# MEASURED 2026-08-18: on Windows `command -v python3` succeeds because it finds
# the Microsoft Store ALIAS STUB, which then prints "Python wurde nicht gefunden"
# and exits 1 -- so the hook REFUSED EVERY COMMIT, including clean ones, on the
# dev box. Presence of a binary is not the ability to run it, the same way an
# exit code is not evidence. `-c ''` separates a real interpreter from a stub.
for py in python3 python py; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c "" >/dev/null 2>&1; then
        exec "$py" "$root/tools/secret_scan.py" --staged --hook
    fi
done
echo "[secret_scan] FATAL: no python on PATH -- the credential gate could NOT run."
echo "[secret_scan] Refusing the commit: an unrunnable gate must not read as a pass."
exit 1
"""


def hooks_dir(repo: Path) -> Path:
    """Ask git where hooks live. In a worktree this resolves to the COMMON dir,
    so one install covers every worktree -- and `core.hooksPath` is honoured."""
    p = Path(git(repo, "rev-parse", "--git-path", "hooks").strip())
    return p if p.is_absolute() else (repo / p)


def hook_status(repo: Path) -> tuple[str, Path]:
    """-> ("missing" | "foreign" | "stale" | "current", path)."""
    path = hooks_dir(repo) / "pre-commit"
    if not path.exists():
        return "missing", path
    try:
        body = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "foreign", path
    if HOOK_MARKER not in body:
        return "foreign", path
    return ("current" if body == HOOK_BODY else "stale"), path


def install_hook(repo: Path, force: bool = False, log=print) -> tuple[bool, str]:
    state, path = hook_status(repo)
    if state == "current":
        return True, f"already current: {path}"
    if state == "foreign" and not force:
        return False, (f"REFUSING to overwrite a pre-existing, non-TanitAD hook at "
                       f"{path} -- inspect it, then re-run with --force-hook")
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF newlines: this runs under sh even on Windows (git-bash), and CRLF here
    # produces the classic 'bad interpreter' failure.
    with open(path, "wb") as fh:
        fh.write(HOOK_BODY.encode("utf-8"))
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass
    return True, f"installed ({state} -> current): {path}"


# ------------------------------------------------------------------------- CLI


def render(rep: Report, show_advisory: bool = True) -> str:
    lines = [f"[secret_scan] scope={rep.scope}",
             f"[secret_scan] candidates = {rep.candidates}  "
             f"files scanned = {rep.files_scanned}  bytes = {rep.bytes_scanned}"]
    if rep.unusable:
        lines.append("[secret_scan] " + "=" * 62)
        lines.append(f"[secret_scan] *** {rep.unusable_label}: {rep.unusable_reason}")
        lines.append("[secret_scan] Do NOT read the verdict below as a pass. A "
                     "scan that read nothing")
        lines.append("[secret_scan] reports zero findings for the same reason a "
                     "dead monitor reports no")
        lines.append("[secret_scan] alarm. Fix the cause and re-run.")
        lines.append("[secret_scan] " + "=" * 62)
    if rep.skipped:
        lines.append("[secret_scan] skipped (COUNTED, not dropped): "
                     + ", ".join(f"{k}={v}" for k, v in sorted(rep.skipped.items())))
    if rep.errors:
        lines.append(f"[secret_scan] errors = {len(rep.errors)}")
        for e in rep.errors[:10]:
            lines.append(f"    ! {e}")
    blocking, advisory = rep.blocking, rep.advisory
    if blocking:
        lines.append(f"[secret_scan] BLOCKING ({len(blocking)}) -- values are "
                     f"NEVER printed:")
        for f in blocking:
            loc = f"{f.path}:{f.line}" if f.line else f.path
            lines.append(f"    X [{f.pattern}] {loc}")
            lines.append(f"        {f.detail}")
    if rep.unusable:
        # ⛔ The verdict line itself carries the failure. MEASURED 2026-08-18:
        # the partial run printed its batch failures in scrollback and then
        # summarised "BLOCKING (0) -- clean" anyway -- and the summary is what
        # got read. An unusable scan may NEVER end on the word 'clean'.
        lines.append(f"[secret_scan] {rep.unusable_label}: "
                     f"{rep.unusable_reason} -- NOT clean")
    elif not blocking:
        lines.append("[secret_scan] BLOCKING (0) -- clean")
    if show_advisory:
        lines.append(f"[secret_scan] advisory ({len(advisory)}) -- not a refusal")
        for f in advisory[:15]:
            loc = f"{f.path}:{f.line}" if f.line else f.path
            lines.append(f"    . [{f.pattern}] {loc}")
        if len(advisory) > 15:
            lines.append(f"    . ... {len(advisory) - 15} more")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="secret_scan",
        description="Credential gate (C111). Never prints a matched value.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--staged", action="store_true",
                      help="scan the staged blobs (the commit gate)")
    mode.add_argument("--tree", metavar="DIR",
                      help="scan an arbitrary directory BEFORE staging (C111)")
    mode.add_argument("--tracked", action="store_true",
                      help="scan every tracked file in the working tree")
    mode.add_argument("--history", action="store_true",
                      help="scan every blob in the object database")
    mode.add_argument("--install-hook", action="store_true",
                      help="install the pre-commit gate")
    mode.add_argument("--check-hook", action="store_true",
                      help="report whether the pre-commit gate is installed")
    ap.add_argument("--repo", default=".", help="repo root (default: cwd)")
    ap.add_argument("--json", metavar="FILE", help="write the report as JSON")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--include-git", action="store_true",
                    help="do not skip .git/ when walking a tree")
    ap.add_argument("--force-hook", action="store_true",
                    help="overwrite a pre-existing foreign pre-commit hook")
    ap.add_argument("--hook", action="store_true",
                    help="terser output; used by the installed hook")
    ap.add_argument("--no-advisory", action="store_true")
    args = ap.parse_args(argv)

    start = Path(args.repo).resolve()

    if args.install_hook or args.check_hook:
        try:
            root = repo_root(start)
        except (GitError, OSError, FileNotFoundError) as exc:
            print(f"[secret_scan] not a git repo / git unavailable: {exc}")
            return 3
        if args.check_hook:
            state, path = hook_status(root)
            print(f"[secret_scan] pre-commit hook: {state}  ({path})")
            if state != "current":
                print("[secret_scan] install it:  python tools/secret_scan.py "
                      "--install-hook")
            return 0 if state == "current" else 1
        ok, msg = install_hook(root, force=args.force_hook)
        print(f"[secret_scan] {msg}")
        return 0 if ok else 1

    if args.tree:
        rep = scan_tree(Path(args.tree).resolve(), args.max_bytes, args.include_git)
    else:
        try:
            root = repo_root(start)
        except (GitError, OSError, FileNotFoundError) as exc:
            print(f"[secret_scan] not a git repo / git unavailable: {exc}")
            return 3
        if args.tracked:
            rep = scan_tracked(root, args.max_bytes)
        elif args.history:
            rep = scan_history(root)
        else:
            rep = scan_staged(root, args.max_bytes)   # --staged is the default

    if args.json:
        Path(args.json).write_text(json.dumps(rep.to_json(), indent=2),
                                   encoding="utf-8")
    blocking = rep.blocking
    if rep.unusable:
        print(render(rep, show_advisory=False))
        return 1
    if args.hook and not blocking:
        print(f"[secret_scan] OK -- {rep.files_scanned} staged blob(s) clean")
        return 0
    print(render(rep, show_advisory=not args.no_advisory))
    if blocking and args.hook:
        print("")
        print("[secret_scan] COMMIT REFUSED. Remove the credential, then rotate it")
        print("[secret_scan] -- a secret that reached a file is exposed, and")
        print("[secret_scan] redacting our copy does nothing to the machine's.")
        print("[secret_scan] Never use a host's 'allow this secret' unblock link.")
    return 1 if blocking else 0


if __name__ == "__main__":     # pragma: no cover
    sys.exit(main())
