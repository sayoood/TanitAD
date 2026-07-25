"""registry_lint — keep `Project Steering/MODEL_REGISTRY.md` honest.

The registry is the ONLY quotable source for model facts (CLAUDE.md section
"Source of truth"), but every number in it is **hand-transcribed from raw eval
JSON** and every headline is hand-written. Both drift, and the drift is not
theoretical:

  * **The 4-day stale headline.** Section 1.4b's header read *"flagship-v1.6 =
    best ADE in the program"* for **four days after its own body retracted the
    claim** (RETRACTION_LOG 07-25, class C4). The 07-21 retraction edited the
    prose and left the HEADER standing -- so the source of truth headlined a
    claim it refuted 14 lines later. Two binding lessons came out of it:
    a retraction sweep must re-read **section HEADERS**, and it must be
    **MULTILINE**, because a second instance wrapped across a newline
    (``the best`` / ``ADE in the program``) and walked straight through a
    line-based grep.

  * **Transcription drift.** A number copied by hand from
    ``taniteval/results/*.json`` has no link back to its source, so nothing
    notices when the JSON is re-run and the prose is not.

Two checks, therefore:

  CHECK 1 -- POINTER DRIFT. A row may carry a machine-readable pointer at the
  raw JSON that produced its number::

      <!-- src: taniteval/results/driving_flagship-30k.json#headline.ade_0_2s.mean -->
      | 1= | Flagship v1 ... | ... 0.4271 ... |

  The linter re-reads the JSON, extracts every number from the target line, and
  fails unless one of them equals the JSON value **to the precision it was
  written at** (``0.4271`` tolerates 5e-5, ``0.452`` tolerates 5e-4) -- which is
  exactly the rounding a human transcriber applies. Pointers may also live in a
  SIDECAR (``tools/registry_pointers.jsonl``) keyed by an anchor regex, so the
  mechanism can be seeded without editing a file six agents have open.

  CHECK 2 -- RETRACTED-CLAIM SWEEP. Every claim quoted in ``RETRACTION_LOG.md``
  is shredded into word-shingles and searched against a **whitespace-collapsed
  token stream of the whole document**, with a token->line map. Because the
  stream ignores line boundaries, a claim that wraps across a newline is found
  exactly like one that does not. A hit whose span touches a markdown header is
  an ERROR (the highest-visibility surface, and the one the 07-25 retraction was
  about); a hit in body prose is a WARNING, because bodies legitimately QUOTE the
  claims they retract. Hits near a retraction marker (``retracted``,
  ``corrected``, ``NOT``, ``superseded``, ...) are suppressed, as is any line
  carrying ``<!-- lint-ok: reason -->``.

Exit codes: 0 = clean · 1 = a finding · 2 = usage/IO error.

Usage::

    python tools/registry_lint.py                        # the standard sweep
    python tools/registry_lint.py --strict                # body hits also fail
    python tools/registry_lint.py --file "Project Steering/LEADERBOARD.md"
    python tools/registry_lint.py --self-test             # red/green falsifiers
    python tools/registry_lint.py --json report.json

Stdlib-only, ASCII-clean stdout, OS-agnostic.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

def ascii_safe(s: str) -> str:
    """The Windows cp1252 console lesson (tools/README.md): this repo's docs are
    full of em dashes, arrows and emoji, and printing an excerpt of them to a
    cp1252 console raises UnicodeEncodeError -- i.e. the LINTER crashes instead
    of reporting the finding it just made. Measured 2026-07-25 on the real
    registry before this guard existed."""
    return s.encode("ascii", "replace").decode("ascii")


DEFAULT_REGISTRY = "Project Steering/MODEL_REGISTRY.md"
DEFAULT_RETRACTIONS = "Project Steering/RETRACTION_LOG.md"
DEFAULT_SIDECAR = "tools/registry_pointers.jsonl"

# ---------------------------------------------------------------- number extraction

# Numbers as a human writes them in this registry: `0.4271`, `-0.0340`, `1.96`,
# `286,339,251`, `29 999` (space-grouped), `1e-3`. The grouped alternative comes
# FIRST so `29 999` is read as one number, not two.
_SEP = r"[,    ]"
NUM_RE = re.compile(
    rf"[-+−]?(?:\d{{1,3}}(?:{_SEP}\d{{3}})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass
class Quoted:
    """A number as it appears in the prose, with the tolerance its own written
    precision implies."""
    raw: str
    value: float
    tol: float
    col: int

    @classmethod
    def parse(cls, raw: str, col: int) -> "Quoted | None":
        clean = re.sub(_SEP, "", raw).replace("−", "-")
        try:
            value = float(clean)
        except ValueError:
            return None
        # Tolerance = half of the last written decimal place. A transcriber who
        # writes 0.4271 has rounded to 1e-4, so 5e-5 is the honest band; one who
        # writes 0.452 has rounded to 1e-3. An integer tolerates 0.5.
        if "e" in clean.lower():
            tol = abs(value) * 5e-4 or 5e-4
        elif "." in clean:
            tol = 0.5 * 10 ** (-len(clean.split(".", 1)[1]))
        else:
            tol = 0.5
        return cls(raw=raw, value=value, tol=tol, col=col)


def implied_tol(value: float) -> float:
    """The rounding band the SOURCE value was itself written at.

    Both sides of the comparison are rounded: `driving_flagship-v16-ab-ft.json`
    stores ``0.4375`` (4 dp) while the registry quotes the CI recompute's
    ``0.43746`` (5 dp). Neither is wrong, and a tolerance taken only from the
    prose (5e-6) would flag that honest pair as drift. The admissible band is
    therefore the COARSER of the two written precisions."""
    text = repr(float(value))
    if "e" in text.lower():
        return abs(value) * 5e-4 or 5e-4
    if "." in text:
        return 0.5 * 10 ** (-len(text.split(".", 1)[1]))
    return 0.5


def numbers_in(text: str, after: str | None = None) -> list[Quoted]:
    start = 0
    if after:
        idx = text.find(after)
        if idx < 0:
            return []
        start = idx + len(after)
    out: list[Quoted] = []
    for m in NUM_RE.finditer(text, start):
        q = Quoted.parse(m.group(0), m.start())
        if q is not None:
            out.append(q)
    return out


# ------------------------------------------------------------------ pointer parsing

POINTER_RE = re.compile(r"<!--\s*src:\s*(?P<spec>.*?)\s*-->", re.DOTALL)
KV_RE = re.compile(r"""(\w+)=("([^"]*)"|'([^']*)'|\S+)""")
LINT_OK_RE = re.compile(r"<!--\s*lint-ok\b")


@dataclass
class Pointer:
    src: str                 # repo-relative json path
    field_path: str          # dotted path inside the json
    line: int                # 1-based line the quoted number lives on
    origin: str              # "inline" or "sidecar:<anchor>"
    near: str | None = None  # only look at numbers after this literal
    tol: float | None = None # explicit override
    label: str = ""


def parse_spec(spec: str) -> tuple[str, str, dict[str, str]]:
    """``path.json#a.b[0].c near="full-set" tol=1e-3`` -> (path, field, opts)."""
    opts = {m.group(1): (m.group(3) if m.group(3) is not None
                         else m.group(4) if m.group(4) is not None
                         else m.group(2))
            for m in KV_RE.finditer(spec)}
    head = KV_RE.sub("", spec).strip()
    if "#" not in head:
        raise ValueError(f"pointer needs 'path#field.path', got {head!r}")
    src, _, fieldp = head.partition("#")
    return src.strip(), fieldp.strip(), opts


_IDX_RE = re.compile(r"\[(\d+)\]")


def resolve_field(doc, field_path: str):
    """``a.b[0].c`` into nested dicts/lists. Raises KeyError/IndexError/TypeError."""
    cur = doc
    for part in field_path.split("."):
        if not part:
            continue
        name = _IDX_RE.split(part)[0]
        if name:
            cur = cur[name]
        for idx in _IDX_RE.findall(part):
            cur = cur[int(idx)]
    return cur


def collect_inline_pointers(lines: list[str], lookahead: int = 8) -> list[Pointer]:
    """An inline pointer governs the next line that actually QUOTES A NUMBER.

    If the comment shares its line with other text (the common table-row case,
    ``| ... | <!-- src: ... -->``) it governs that same line. On its own line it
    scans forward -- past blank lines, past the ``|---|---|`` rule, past a table
    header row -- for the first line carrying a numeral, because that is what a
    numeric pointer means. ``line=+N`` overrides the search explicitly."""
    out: list[Pointer] = []
    for i, line in enumerate(lines):
        for m in POINTER_RE.finditer(line):
            residue = POINTER_RE.sub("", line).strip()
            target = i + 1                      # 1-based; same line
            if not residue:                     # comment-only line
                target = i + 2                  # fallback: the very next line
                for j in range(i + 1, min(len(lines), i + 1 + lookahead)):
                    if lines[j].strip() and numbers_in(lines[j]):
                        target = j + 1
                        break
            try:
                src, fieldp, opts = parse_spec(m.group("spec"))
            except ValueError as exc:
                out.append(Pointer(src="", field_path=str(exc), line=i + 1,
                                   origin="inline"))
                continue
            if "line" in opts:
                off = opts["line"]
                target = (i + 1 + int(off)) if off.startswith(("+", "-")) else int(off)
            out.append(Pointer(
                src=src, field_path=fieldp, line=target, origin="inline",
                near=opts.get("near"),
                tol=float(opts["tol"]) if "tol" in opts else None,
                label=opts.get("label", "")))
    return out


def collect_sidecar_pointers(sidecar: Path, lines: list[str],
                             doc_rel: str) -> tuple[list[Pointer], list[str]]:
    """Sidecar rows are JSONL: ``{"file":..., "anchor":<regex>, "src":..., "field":...}``.

    The anchor is matched against each line of the document; exactly one match is
    required, so an anchor that goes ambiguous after an edit is itself a finding
    (a silently-relocated pointer is worse than a missing one)."""
    ptrs: list[Pointer] = []
    errs: list[str] = []
    if not sidecar.is_file():
        return ptrs, errs
    for n, raw in enumerate(sidecar.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            errs.append(f"{sidecar}:{n}: unreadable JSONL row: {exc}")
            continue
        if row.get("file") and row["file"].replace("\\", "/") != doc_rel:
            continue
        try:
            pat = re.compile(row["anchor"])
        except (re.error, KeyError) as exc:
            errs.append(f"{sidecar}:{n}: bad anchor: {exc}")
            continue
        hits = [i + 1 for i, ln in enumerate(lines) if pat.search(ln)]
        if len(hits) != 1:
            errs.append(f"{sidecar}:{n}: anchor {row['anchor']!r} matched "
                        f"{len(hits)} line(s), need exactly 1 "
                        f"(the row moved or the anchor went ambiguous)")
            continue
        ptrs.append(Pointer(src=row["src"], field_path=row["field"],
                            line=hits[0], origin=f"sidecar:{sidecar.name}:{n}",
                            near=row.get("near"), tol=row.get("tol"),
                            label=row.get("label", row.get("why", ""))))
    return ptrs, errs


# ------------------------------------------------------------------- retracted claims

# Words that carry no discriminating power on their own; a shingle made only of
# these is not evidence of anything.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "then",
    "this", "to", "was", "were", "with", "not", "no", "our", "we", "i", "my",
}

# Presence of any of these near a hit means the document is TALKING ABOUT the
# retraction rather than asserting the claim. Over-suppression is the failure
# mode to avoid, so the list is deliberately about retraction vocabulary, not
# general hedging.
RETRACTION_MARKERS = {
    "retracted", "retraction", "retract", "corrected", "correction", "wrong",
    "refuted", "superseded", "deprecated", "false", "mistaken", "stale",
    "formerly", "previously", "read", "not", "never", "unresolved", "tied",
    "withdrawn", "obsolete", "invalid", "unquotable", "quotable",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s")
# Markdown furniture that must not become tokens.
_STRIP_RE = re.compile(r"[*_`~>|#\\]")


def tokenize_stream(lines: list[str]) -> tuple[list[str], list[int]]:
    """Whole document -> (tokens, line_of_token).

    THIS is what makes the sweep multiline: markdown line breaks vanish, so a
    claim split as ``the best`` / ``ADE in the program`` is adjacent in the
    stream and matches the same shingle as the unwrapped form. Emphasis markers,
    table pipes and blockquote arrows are stripped first so ``**best**`` and
    ``best`` tokenize identically."""
    toks: list[str] = []
    lines_of: list[int] = []
    for i, raw in enumerate(lines, 1):
        clean = _STRIP_RE.sub(" ", raw).lower()
        for m in TOKEN_RE.finditer(clean):
            toks.append(m.group(0))
            lines_of.append(i)
    return toks, lines_of


def normalize_claim(text: str) -> list[str]:
    return TOKEN_RE.findall(_STRIP_RE.sub(" ", text).lower())


_CLAIM_QUOTE_RE = re.compile(r"[\"“”]([^\"“”]{12,400})"
                             r"[\"“”]")


def load_retracted_claims(path: Path) -> list[str]:
    """Every quoted claim in the retraction log's entry table.

    The log's shape is ``| date | *"claim"* | class | cost |``; the claim cell is
    the only place a retracted assertion is quoted verbatim, and the surrounding
    cells are commentary we must NOT mine (they contain the corrections)."""
    claims: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        if not re.match(r"^\d{2}-\d{2}$", cells[0]):   # skip header/legend rows
            continue
        for m in _CLAIM_QUOTE_RE.finditer(cells[1]):
            claims.append(m.group(1))
    return claims


def shingles(claim: str, n: int) -> list[tuple[str, ...]]:
    """Distinctive n-word phrases of a claim.

    A shingle must START on a content word: it is the key the stream index is
    built on, and indexing on ``the`` would make every article in the document a
    candidate. Windows carrying fewer than two content words are dropped -- they
    match everything and therefore mean nothing."""
    words = normalize_claim(claim)
    out = []
    for i in range(len(words) - n + 1):
        window = tuple(words[i:i + n])
        if window[0] in STOPWORDS:
            continue
        if sum(1 for w in window if w not in STOPWORDS) < 2:
            continue
        out.append(window)
    return out


def match_gapped(toks: list[str], i: int, shingle: tuple[str, ...],
                 gap: int) -> int | None:
    """Match ``shingle`` at ``i`` as an ORDERED SUBSEQUENCE with <=``gap`` extra
    tokens spliced in. Returns the end index (exclusive) or None.

    This is not decoration -- it is the difference between catching the 4-day
    stale headline and missing it. RETRACTION_LOG's 07-21 entry quotes the claim
    as *"best in the program"*; the header that survived said *"best **ADE** in
    the program"*. A strict n-gram sweep run on 07-21 would have reported clean.
    One inserted token is all the drift there ever was."""
    n = len(shingle)
    k, j, budget = 1, i + 1, gap
    limit = min(len(toks), i + n + gap)
    while k < n and j < limit:
        if toks[j] == shingle[k]:
            k += 1
        else:
            budget -= 1
            if budget < 0:
                return None
        j += 1
    return j if k == n else None


@dataclass
class Finding:
    kind: str            # "pointer" | "claim" | "pointer-error"
    severity: str        # "ERROR" | "WARN"
    file: str
    line: int
    message: str
    detail: str = ""


# --------------------------------------------------------------------------- checks


def check_pointers(doc: Path, repo: Path, lines: list[str],
                   pointers: list[Pointer]) -> list[Finding]:
    out: list[Finding] = []
    cache: dict[str, object] = {}
    rel = doc.relative_to(repo).as_posix() if doc.is_relative_to(repo) else str(doc)
    for p in pointers:
        if not p.src:
            out.append(Finding("pointer-error", "ERROR", rel, p.line,
                               f"malformed pointer: {p.field_path}"))
            continue
        src_path = (repo / p.src)
        if p.src not in cache:
            try:
                cache[p.src] = json.loads(src_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                cache[p.src] = exc
        doc_json = cache[p.src]
        if isinstance(doc_json, Exception):
            out.append(Finding("pointer-error", "ERROR", rel, p.line,
                               f"pointer source unreadable: {p.src}",
                               str(doc_json)))
            continue
        try:
            value = resolve_field(doc_json, p.field_path)
        except (KeyError, IndexError, TypeError) as exc:
            out.append(Finding("pointer-error", "ERROR", rel, p.line,
                               f"field '{p.field_path}' not in {p.src}",
                               f"{type(exc).__name__}: {exc}"))
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            out.append(Finding("pointer-error", "ERROR", rel, p.line,
                               f"field '{p.field_path}' in {p.src} is not numeric",
                               repr(value)[:120]))
            continue
        if not (1 <= p.line <= len(lines)):
            out.append(Finding("pointer-error", "ERROR", rel, p.line,
                               "pointer target line is out of range"))
            continue
        target = lines[p.line - 1]
        cands = numbers_in(target, after=p.near)
        src_tol = implied_tol(float(value))
        ok = any(abs(c.value - float(value))
                 <= (p.tol if p.tol is not None else max(c.tol, src_tol))
                 for c in cands)
        if not ok:
            shown = ", ".join(c.raw for c in cands[:8]) or "(no number on the line)"
            out.append(Finding(
                "pointer", "ERROR", rel, p.line,
                f"DRIFT: {p.src}#{p.field_path} = {value!r} but the line quotes "
                f"[{shown}]",
                f"pointer origin {p.origin}"
                + (f", near={p.near!r}" if p.near else "")
                + f"\n    line: {target.strip()[:200]}"))
    return out


def check_retracted_claims(doc: Path, repo: Path, lines: list[str],
                           claims: list[str], shingle_n: int,
                           context: int, gap: int = 1,
                           rare_max: int = 25) -> list[Finding]:
    out: list[Finding] = []
    rel = doc.relative_to(repo).as_posix() if doc.is_relative_to(repo) else str(doc)
    toks, lines_of = tokenize_stream(lines)
    if not toks:
        return out
    # Document term frequencies drive the ERROR/WARN split. A short phrase built
    # entirely from this document's own house vocabulary is a section title, not
    # a propagated claim -- MEASURED on the registry (26,459 tokens, 2026-07-25):
    # the false positive `### 4.4 REF-C CLOSED-LOOP ...` matches the retracted
    # "flagship v1 beats REF-C closed-loop" on ref=122 / c=81 / loop=60 /
    # closed=32, while the REAL stale headline carries best=23 and program=13.
    # Nothing is hidden either way: a boilerplate hit is still reported, just as
    # a warning, so --strict surfaces it.
    tf = {}
    for t in toks:
        tf[t] = tf.get(t, 0) + 1
    header_lines = {i for i, ln in enumerate(lines, 1) if HEADER_RE.match(ln)}
    lint_ok_lines = {i for i, ln in enumerate(lines, 1) if LINT_OK_RE.search(ln)}
    # lint-ok also covers the immediately following line, so a marker can sit
    # above the row it exempts.
    lint_ok_lines |= {i + 1 for i in lint_ok_lines}

    # Index by the shingle's first (content) token so the stream is scanned once.
    by_head: dict[str, list[tuple[tuple[str, ...], str]]] = {}
    for claim in claims:
        for sh in shingles(claim, shingle_n):
            by_head.setdefault(sh[0], []).append((sh, claim))
    if not by_head:
        return out

    # Overlapping shingles of one claim all fire on the same passage; collapse
    # them into ONE finding per (passage, claim) so an operator reads a finding,
    # not a wall.
    hits: dict[tuple[int, str], dict] = {}
    for i, head in enumerate(toks):
        for sh, claim in by_head.get(head, ()):
            end = match_gapped(toks, i, sh, gap)
            if end is None:
                continue
            span_lines = sorted(set(lines_of[i:end]))
            lo = max(0, i - context)
            hi = min(len(toks), end + context)
            near_tokens = set(toks[lo:i]) | set(toks[end:hi])
            if near_tokens & RETRACTION_MARKERS:
                continue
            if any(ln in lint_ok_lines for ln in span_lines):
                continue
            key = (span_lines[0], claim)
            agg = hits.setdefault(key, {"lines": set(), "tok_lo": i,
                                        "tok_hi": end, "claim": claim})
            agg["lines"] |= set(span_lines)
            agg["tok_lo"] = min(agg["tok_lo"], i)
            agg["tok_hi"] = max(agg["tok_hi"], end)

    for (first_line, _claim), agg in sorted(hits.items()):
        span_lines = sorted(agg["lines"])
        in_header = any(ln in header_lines for ln in span_lines)
        wrapped = len(span_lines) > 1
        span = toks[agg["tok_lo"]:agg["tok_hi"]]
        phrase = " ".join(span)
        rare = [t for t in span if t not in STOPWORDS and tf.get(t, 0) <= rare_max]
        boilerplate = rare_max > 0 and not rare
        excerpt = " / ".join(lines[ln - 1].strip()[:110] for ln in span_lines)
        if in_header and not boilerplate:
            head, sev = "RETRACTED CLAIM IN A SECTION HEADER", "ERROR"
        elif in_header:
            head, sev = ("header matches retracted-claim vocabulary, but every "
                         "word is house boilerplate (likely a plain section "
                         "title)"), "WARN"
        else:
            head, sev = "retracted claim restated in body prose", "WARN"
        out.append(Finding(
            "claim", sev, rel, first_line,
            head + (" (WRAPPED ACROSS A NEWLINE)" if wrapped else "")
            + f": \"{phrase}\"",
            f"matches retracted claim: \"{agg['claim'][:160]}\"\n    "
            f"rare tokens in the match: {rare or '(none)'}\n    lines "
            f"{span_lines}: {excerpt}"))
    return out


# --------------------------------------------------------------------------- driver


@dataclass
class LintReport:
    files: list[str] = field(default_factory=list)
    n_pointers: int = 0
    n_claims: int = 0
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"files": self.files, "n_pointers": self.n_pointers,
                "n_claims": self.n_claims,
                "findings": [asdict(f) for f in self.findings]}


def lint(repo: Path, doc_paths: list[Path], retractions: Path,
         sidecar: Path | None, shingle_n: int, context: int,
         gap: int = 1, rare_max: int = 25) -> LintReport:
    rep = LintReport()
    claims = load_retracted_claims(retractions) if retractions.is_file() else []
    rep.n_claims = len(claims)
    for doc in doc_paths:
        rel = doc.relative_to(repo).as_posix() if doc.is_relative_to(repo) else str(doc)
        rep.files.append(rel)
        lines = doc.read_text(encoding="utf-8").splitlines()
        ptrs = collect_inline_pointers(lines)
        if sidecar is not None:
            side, errs = collect_sidecar_pointers(sidecar, lines, rel)
            ptrs += side
            for e in errs:
                rep.findings.append(Finding("pointer-error", "ERROR", rel, 0, e))
        rep.n_pointers += len(ptrs)
        rep.findings += check_pointers(doc, repo, lines, ptrs)
        rep.findings += check_retracted_claims(doc, repo, lines, claims,
                                               shingle_n, context, gap, rare_max)
    rep.findings.sort(key=lambda f: (f.file, f.line, f.kind))
    return rep


def render(rep: LintReport, strict: bool) -> tuple[str, int]:
    out = [f"registry_lint: {len(rep.files)} file(s), {rep.n_pointers} pointer(s), "
           f"{rep.n_claims} retracted claim(s) loaded"]
    errors = [f for f in rep.findings if f.severity == "ERROR"]
    warns = [f for f in rep.findings if f.severity == "WARN"]
    for f in rep.findings:
        tag = "[ERROR]" if f.severity == "ERROR" else "[warn] "
        out.append("")
        out.append(f"{tag} {f.file}:{f.line}: {f.message}")
        if f.detail:
            for ln in f.detail.splitlines():
                out.append(f"        {ln}")
    out.append("")
    fail = bool(errors) or (strict and bool(warns))
    out.append(f"RESULT: {'FAIL' if fail else 'PASS'} "
               f"({len(errors)} error(s), {len(warns)} warning(s)"
               f"{'; --strict makes warnings fatal' if strict else ''})")
    return ascii_safe("\n".join(out)), (1 if fail else 0)


# ------------------------------------------------------------------------ self-test

_CLEAN_DOC = """# Registry

## 1.4b flagship-v1.6 -- TIED with the deployed v1

<!-- src: {json}#headline.ade_0_2s.mean -->
| arm | ADE |
|---|---|
| v1 | 0.4271 |
"""

_STALE_HEADER_DOC = """# Registry

## 1.4b flagship-v1.6 -- the best
ADE in the program

<!-- src: {json}#headline.ade_0_2s.mean -->
| arm | ADE |
|---|---|
| v1 | 0.4271 |
"""

_DRIFTED_DOC = """# Registry

## 1.4b flagship-v1.6 -- TIED with the deployed v1

<!-- src: {json}#headline.ade_0_2s.mean -->
| arm | ADE |
|---|---|
| v1 | 0.4420 |
"""

# The REAL 2026-07-21..25 header, verbatim from git blob c5e5d5f.
_REAL_STALE_HEADER_DOC = """# Registry

### 1.4b flagship-v1.6 -- `flagship-v16-ab-ft` -- COMPLETE at 5,999 - best ADE in the program

<!-- src: {json}#headline.ade_0_2s.mean -->
| arm | ADE |
|---|---|
| v1 | 0.4271 |
"""

_SELFTEST_RETRACTIONS = """# RETRACTION LOG

| date | retracted claim | class | cost |
|---|---|---|---|
| 07-25 | *"flagship-v1.6 = the best ADE in the program"* | C4 | four days |
"""

# The retraction log as it stood on 07-21, BEFORE the 07-25 entry existed. Its
# wording differs from the surviving header by one inserted word.
_SELFTEST_RETRACTIONS_0721 = """# RETRACTION LOG

| date | retracted claim | class | cost |
|---|---|---|---|
| 07-21 | *"v1.6 ADE 0.4420 - best in the program"* | C1 | trainer in-loop val |
"""


def self_test(tmp_root: Path) -> tuple[str, int]:
    """Falsifiers, run against throwaway files. RED cases must fail, the GREEN
    case must pass -- a linter nobody has watched fail is not evidence."""
    tmp_root.mkdir(parents=True, exist_ok=True)
    src = tmp_root / "eval.json"
    src.write_text(json.dumps({"headline": {"ade_0_2s": {"mean": 0.4271}}}),
                   encoding="utf-8")
    retr = tmp_root / "RETRACTION_LOG.md"
    retr.write_text(_SELFTEST_RETRACTIONS, encoding="utf-8")
    retr21 = tmp_root / "RETRACTION_LOG_0721.md"
    retr21.write_text(_SELFTEST_RETRACTIONS_0721, encoding="utf-8")

    rows, rc = [], 0
    # (name, doc template, retraction log, gap, expected exit)
    cases = [
        ("GREEN clean doc (correct number, corrected header)",
         _CLEAN_DOC, retr, 1, 0),
        ("RED   stale headline WRAPPED ACROSS A NEWLINE",
         _STALE_HEADER_DOC, retr, 1, 1),
        ("RED   pointer drift (0.4420 vs JSON 0.4271)",
         _DRIFTED_DOC, retr, 1, 1),
        ("RED   REAL c5e5d5f header vs the 07-21 log wording (needs gap=1)",
         _REAL_STALE_HEADER_DOC, retr21, 1, 1),
        ("CTRL  same doc+log at gap=0 -- reproduces the 4-day MISS",
         _REAL_STALE_HEADER_DOC, retr21, 0, 0),
    ]
    for name, tpl, log, gap, want in cases:
        doc = tmp_root / (re.sub(r"\W+", "_", name) + ".md")
        doc.write_text(tpl.format(json="eval.json"), encoding="utf-8")
        rep = lint(tmp_root, [doc], log, None, 4, 25, gap, 0)
        _, got = render(rep, strict=False)
        ok = got == want
        rc |= 0 if ok else 1
        rows.append(f"  [{'ok' if ok else 'FAIL'}] {name}: exit {got} "
                    f"(expected {want})")
        for f in rep.findings:
            rows.append(f"         -> {f.severity} {f.file}:{f.line} {f.message}")
    rows.insert(0, "registry_lint --self-test")
    rows.append(f"SELF-TEST: {'PASS' if rc == 0 else 'FAIL'}")
    return ascii_safe("\n".join(rows)), rc


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Drift + retracted-claim linter for "
                                             "the TanitAD model registry.")
    ap.add_argument("--repo", default=".", help="repo root (default: cwd)")
    ap.add_argument("--file", action="append", default=[], dest="files",
                    help=f"document to lint (repeatable; default {DEFAULT_REGISTRY})")
    ap.add_argument("--retractions", default=DEFAULT_RETRACTIONS)
    ap.add_argument("--sidecar", default=DEFAULT_SIDECAR,
                    help="anchor-keyed pointer sidecar JSONL "
                         f"(default {DEFAULT_SIDECAR}; pass '' to disable)")
    ap.add_argument("--shingle", type=int, default=4,
                    help="claim phrase length in words (default 4)")
    ap.add_argument("--gap", type=int, default=1,
                    help="extra tokens tolerated INSIDE a matched phrase "
                         "(default 1). The 4-day stale header differed from the "
                         "logged claim by exactly one inserted word ('best ADE "
                         "in the program' vs 'best in the program'), so gap=0 "
                         "reproduces the miss.")
    ap.add_argument("--context", type=int, default=25,
                    help="tokens around a hit searched for a retraction marker "
                         "before suppressing it (default 25)")
    ap.add_argument("--rare-max", type=int, default=25,
                    help="a header hit is an ERROR only if the matched phrase "
                         "holds a token occurring <= this many times in the "
                         "document; otherwise it is house boilerplate and is "
                         "reported as a WARN (default 25, measured). 0 disables.")
    ap.add_argument("--strict", action="store_true",
                    help="body-prose and boilerplate hits also fail "
                         "(header hits always fail)")
    ap.add_argument("--json", default=None, help="write the report as JSON")
    ap.add_argument("--self-test", action="store_true",
                    help="run the built-in red/green falsifiers and exit")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    if args.self_test:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            text, rc = self_test(Path(td) / "selftest")
            print(text)
            return rc

    docs = [repo / f for f in (args.files or [DEFAULT_REGISTRY])]
    missing = [d for d in docs if not d.is_file()]
    if missing:
        for d in missing:
            print(f"registry_lint: no such file: {d}", file=sys.stderr)
        return 2
    retr = repo / args.retractions
    sidecar = (repo / args.sidecar) if args.sidecar else None

    rep = lint(repo, docs, retr, sidecar, args.shingle, args.context,
               args.gap, args.rare_max)
    text, rc = render(rep, args.strict)
    print(text)
    if args.json:
        Path(args.json).write_text(json.dumps(rep.to_dict(), indent=2),
                                   encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
