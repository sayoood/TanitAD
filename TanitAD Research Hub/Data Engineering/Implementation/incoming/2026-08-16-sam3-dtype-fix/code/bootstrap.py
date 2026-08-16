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
print("BOOTSTRAP_A_OK")
