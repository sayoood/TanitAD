"""Kernel bring-up for the SAM3 dtype-fix session (no secrets in this file).

Mirrors the MEASURED bring-up of SAM3_BACKFILL_RUN.md §2:
  1. import closure from a shipped tar -> /content/repo, S2_REPO_ROOT set
  2. HF token moved from a shipped file into the kernel env, file DELETED on
     the VM in this same step (local copy is deleted by the caller)
  3. google.colab.userdata stubbed to raise instantly (Secrets time out
     headlessly and get_hf_token() retries Secrets on EVERY call)
  4. sam3 + torch-free deps installed --no-deps where torch is in the closure
  5. CLIP BPE vocab copied to /content so ph0_sam3.find_bpe()'s cwd-relative
     glob finds it
"""
import os
import subprocess
import sys
import tarfile
import types

# ---- 1. import closure ----------------------------------------------------
os.makedirs("/content/repo", exist_ok=True)
with tarfile.open("/content/repo_closure.tgz") as tf:
    tf.extractall("/content/repo")
# ⛔ THE KERNEL PERSISTS ACROSS `colab exec`, SO A RE-SHIP IS A NO-OP WITHOUT
# THIS. Extracting a newer file over an ALREADY-IMPORTED module changes nothing
# — `import ph0_sam3` returns the cached one, the verify-gate below happily
# passes on the OLD code, and the run silently uses the version the fix was
# meant to replace. That is the pod-checkout-drift trap with a Colab accent,
# and the only reason it is visible here is that this session shipped twice.
for _m in [k for k in list(sys.modules)
           if k.split(".")[0] in ("ph0_sam3", "ph0_v2", "ph0_pilot",
                                  "ph1_fuse", "ph0_rich_overlay",
                                  "s2_lab_lib", "s2_schema", "v2_to_pilot",
                                  "tanitad")]:
    del sys.modules[_m]
os.environ["S2_REPO_ROOT"] = "/content/repo"
for p in ("/content/repo/colab", "/content/repo/stack",
          "/content/repo/stack/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
print("[bs] closure files:",
      sum(len(f) for _, _, f in os.walk("/content/repo")))

# ---- 2. token -> env, file deleted here ------------------------------------
TOKF = "/content/hf_tok.txt"
if os.path.exists(TOKF):
    with open(TOKF) as fh:
        os.environ["HF_TOKEN"] = fh.read().strip()
    os.remove(TOKF)
print("[bs] token_in_env:", bool(os.environ.get("HF_TOKEN")),
      "| token_file_deleted:", not os.path.exists(TOKF))

# ---- 3. userdata stub ------------------------------------------------------
try:
    import google.colab as _gc
    _ud = types.ModuleType("google.colab.userdata")

    class SecretNotFoundError(Exception):
        pass

    class NotebookAccessError(Exception):
        pass

    def get(name):
        raise SecretNotFoundError(name)

    _ud.get = get
    _ud.SecretNotFoundError = SecretNotFoundError
    _ud.NotebookAccessError = NotebookAccessError
    sys.modules["google.colab.userdata"] = _ud
    _gc.userdata = _ud
    print("[bs] userdata stubbed")
except Exception as e:                                    # pragma: no cover
    print("[bs] userdata stub skipped:", type(e).__name__, e)

# ---- 4. installs (sam3 leg only; unsloth is not needed for this session) ----
PKGS = [
    ["--no-deps", "git+https://github.com/facebookresearch/sam3.git"],
    ["iopath"],
    ["--no-deps", "ftfy==6.1.1", "wcwidth"],
    ["--no-deps", "open_clip_torch"],
    ["imageio", "imageio-ffmpeg"],
]
for p in PKGS:
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q", *p])
    print(f"[bs] pip {' '.join(p)} rc={rc}", flush=True)

import torch  # noqa: E402
print("[bs] torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
x = torch.randn(1, 3, 16, 16, device="cuda")
w = torch.randn(4, 3, 3, 3, device="cuda")
print("[bs] conv2d OK", tuple(torch.nn.functional.conv2d(x, w).shape))

# ---- 5. BPE vocab ----------------------------------------------------------
import glob  # noqa: E402
hits = glob.glob("/usr/local/lib/python*/dist-packages/open_clip/"
                 "bpe_simple_vocab_16e6.txt.gz")
if hits:
    import shutil
    shutil.copyfile(hits[0], "/content/bpe_simple_vocab_16e6.txt.gz")
os.chdir("/content")
print("[bs] bpe at /content:",
      os.path.exists("/content/bpe_simple_vocab_16e6.txt.gz"))

import ph0_sam3  # noqa: E402
print("[bs] find_bpe ->", ph0_sam3.find_bpe())

# ---- 6. ⛔ VERIFY-GATE: the SHIPPED code, not whatever the VM had ------------
# A pod/VM whose checkout drifted and a launch from it resurrects fixed bugs is
# a MEASURED failure mode in this programme, and the fix is always the same —
# grep the specific change out of the loaded module before spending GPU on it.
# Here that means: the v2 symbols must exist, and the SCENE list must not have
# swallowed the liveness control.
assert ph0_sam3.SCHEMA_VERSION >= 2, "shipped ph0_sam3 is pre-v2"
assert "lane marking" in ph0_sam3.SCENE_CONCEPTS
assert not (set(ph0_sam3.SCENE_CONCEPTS) & set(ph0_sam3.LIVENESS_CONCEPTS))
assert hasattr(ph0_sam3, "contour_of_mask") and hasattr(ph0_sam3,
                                                        "derive_ego_lane")
import numpy as _np  # noqa: E402
_m = _np.zeros((9, 9), bool)
_m[3:6, 3:6] = True
_c = ph0_sam3.contour_of_mask(_m, tol_px=0.0)
assert _c["contour_area_px"] == 9, _c          # the corner-lattice invariant
# the (1, H, W) fix — the vendor's REAL mask shape, MEASURED on this T4
assert ph0_sam3._rows_rle(_m[None, ...]) == ph0_sam3._rows_rle(_m), \
    "shipped ph0_sam3 still flattens a (1, H, W) mask"
print("[bs] v2 verify-gate OK ·", ph0_sam3.SCENE_CONCEPTS)
print("BOOTSTRAP_A_OK")
