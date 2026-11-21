import sys, argparse
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/stack", sys.argv[1] + "/stack/scripts"]
import v8f_dagger_bank as B
from tanitad.data.v8f_dagger import readout_token
for name, ns in (("causal_g1x", dict(readout="causal_g", acc_gain=1.0, path_exec=True, kappa_max="0.2")), ("causal_g1", dict(readout="causal_g", acc_gain=1.0, path_exec=False, kappa_max="0.2"))):
    g = B.generator_record(argparse.Namespace(**ns))
    print(name, g, readout_token(g["readout"], g["acc_gain"], g["path_exec"]), B.is_default_generator(g))
