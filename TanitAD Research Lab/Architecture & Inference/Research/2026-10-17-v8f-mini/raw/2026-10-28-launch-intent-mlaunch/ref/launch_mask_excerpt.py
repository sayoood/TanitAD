# VERBATIM excerpt of stack/tanitad/models/v8f_mini.py lines 396 and 401-406 at commit d7dc9f11 (git blob c381172e3b713d830626e68375240a3846c9bafe,
# file sha256 b337ccecd3abfa4250a77e233502d9d3464a280dbafdc86b1e94ecbeef682721). Only the two comment lines are added by mlaunch. Used by control (c): an independent recount.
from torch import Tensor  # noqa: F401
import torch  # noqa: F401
LAUNCH_V0_MAX, LAUNCH_DV, LAUNCH_DT = 6.0, 1.0, 0.2


def launch_mask(v0: Tensor, gt: Tensor, gok: Tensor) -> Tensor:
    """``[B]`` bool: the GT-launch samples. ``v0`` [B] is the LOGGED ego speed at NOW (``kin3[:, 0]``), ``gt`` [B, 30, 2] the GT waypoints in the ego frame at NOW, ``gok`` [B] the GT-valid mask. A function of the
    logged state and the GT future ONLY (the trainer calls it BEFORE any ego stage rebuilds ``kin3``). A NaN v0 gives False. ``>=`` is the registered inequality (a chord speed exactly v0 + 1 is a launch)."""
    v6 = torch.linalg.norm((gt[:, 29] - gt[:, 28]).to(torch.float64), dim=-1) / LAUNCH_DT
    v0 = v0.to(torch.float64)
    return (gok.to(torch.float64) > 0.0) & (v0 < LAUNCH_V0_MAX) & (v6 >= v0 + LAUNCH_DV)
