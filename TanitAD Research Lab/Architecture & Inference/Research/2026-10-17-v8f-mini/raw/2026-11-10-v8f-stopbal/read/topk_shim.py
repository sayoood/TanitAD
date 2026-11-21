"""topk_shim.py -- the two EVAL-ONLY helpers the STOPDEC instrument calls, taken verbatim from the commit that introduced them.

`theory/stopdec/stopdec_read.py` (via speedsrc_read / topk_read) calls `model.forward_topk(batch, K)`.  That method and its module function `topk_codes` were
added by 8e9499e0 ("v8F O/T band split", PREREG_V8F_OTBAND) on the OTBAND line; the STOPBAL code of record 715ef8fc is on another line and has neither
(MEASURED 2026-11-11 18:05: `grep -c "def topk_codes"` 0 in the STOPBAL export, 1 in the OTBAND export; the read stopped with AttributeError before any output).

Both are eval-only by their own docstrings ("No new head, no parameter, never called by training"): forward_topk = forward_infer + forward_diag per top-K
(lat, lon) code.  forward_infer, forward_diag, N_GOAL / N_LAT / N_LON and `F` exist in the STOPBAL module with the same signatures and values (MEASURED).

attach() reads the two function sources out of the 8e9499e0 blob (git, core.autocrlf=false), pins their sha256, executes `topk_codes` in the STOPBAL module's
own namespace (so it uses STOPBAL's constants) and sets `V8FMini.forward_topk` ONLY if the class lacks it.  No file of the tree is edited.
"""
import ast
import hashlib
import subprocess
import textwrap

SRC_COMMIT = "8e9499e0f7c201945a856b51e698b465767b6d01"
SRC_PATH = "stack/tanitad/models/v8f_mini.py"
MIRROR = "C:/Users/Admin/tanitad-push"
PINS = {"topk_codes": "f201ea0430faf578034ca72106770e119389d098437a8dfd1c2424965ae3398f",      # MEASURED 2026-11-11: equal at 8e9499e0, at 202849da
        "forward_topk": "1e7eaa0ae8513f61f9aecf8de6ce3b43d5514827c5dcb5d1ce191fe2be81f4e2"}   # and in the OTBAND export that rolled the DAG-OACC baseline


def source_text(mirror=MIRROR):
    return subprocess.run(["git", "-C", mirror, "-c", "core.autocrlf=false", "show", f"{SRC_COMMIT}:{SRC_PATH}"],
                          capture_output=True, check=True).stdout.decode("utf-8")


def segments(text):
    """(topk_codes source, forward_topk source dedented) out of the module text, by AST position."""
    tree = ast.parse(text)
    tc = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "topk_codes"]
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "V8FMini"]
    assert len(tc) == 1 and len(cls) == 1, (len(tc), len(cls))
    ft = [n for n in cls[0].body if isinstance(n, ast.FunctionDef) and n.name == "forward_topk"]
    assert len(ft) == 1, len(ft)
    return ast.get_source_segment(text, tc[0]), textwrap.dedent(ast.get_source_segment(text, ft[0], padded=True))


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def attach(vm, text=None):
    """vm = the STOPBAL `tanitad.models.v8f_mini` module.  Returns a record of what was attached."""
    if hasattr(vm.V8FMini, "forward_topk"):
        return {"attached": False, "reason": "V8FMini already has forward_topk"}
    seg_tc, seg_ft = segments(text if text is not None else source_text())
    for k, s in (("topk_codes", seg_tc), ("forward_topk", seg_ft)):
        if PINS[k] is not None:
            assert sha(s) == PINS[k], (k, sha(s))
    ns = vm.__dict__
    assert "topk_codes" not in ns, "the STOPBAL module already defines topk_codes"
    exec(compile(seg_tc, f"<{SRC_COMMIT[:8]}:topk_codes>", "exec"), ns)
    loc = {}
    exec(compile(seg_ft, f"<{SRC_COMMIT[:8]}:V8FMini.forward_topk>", "exec"), ns, loc)
    vm.V8FMini.forward_topk = loc["forward_topk"]
    return {"attached": True, "src_commit": SRC_COMMIT, "sha256": {"topk_codes": sha(seg_tc), "forward_topk": sha(seg_ft)}, "module": vm.__file__}
