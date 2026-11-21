"""Model-side helpers for IMAG1 (loads the tip's code through wmact_lib; WMACT_CODE_ROOT must be set)."""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "wmact"))
import torch  # noqa: E402
import wmact_lib as W  # noqa: E402

M = W.M


def make_random_model(r, cfg):
    """The same architecture as M2b with untrained weights (default torch init, torch.manual_seed(5000 + r))."""
    torch.manual_seed(5000 + int(r))
    return M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30))).eval()
