"""N4-DAC geometry + scoring context (navtrain ONLY; CPU). Reuses W-TRANSP / TF-N2 machinery unchanged:
  - scenery = nuPlan-DB rebuild (navsim_pdm_targets.metric_cache_parts); DAC reads only the ego corners and the drivable-area map, never the
    objects or the route, and WTRANSP_DIAG.md section 2c / line 85 measured on navtest that DAC, DDC, C agree between this rebuild and the harness
    cache on all 12,146 tokens (ego state, centerline and drivable map identical; only the object positions differ by 1-15 cm).
  - labels: C = iterate20(t20, cubic), R = P5-R (lambda 3.0, warm start C), exactly tfn3_score.py.
Run with the NAVSIM venv python:  PYTHONPATH=<navsim devkit>;<w>/refe  and  W=<export with refe/ eval/>."""
import math
import os
import sys

os.environ.setdefault("REFE_SIM_HZ", "10")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

W = os.environ.get("N4_W", "F:/tanitad_scratch/n4dac/w")
P5R = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5r"
DB_ROOT = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
LAM = 3.0
for _p in (W + "/refe", W + "/eval", W + "/eval/instruments", P5R):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["P5C_PKG"] = W

import navsim_dac as ND  # noqa: E402
import navtrain_scenarios as NS  # noqa: E402
import navsim_pdm_targets as PT  # noqa: E402
import transp_score as TS  # noqa: E402
import p5r_labels as R  # noqa: E402
from p4_tracker_comp import tracked8  # noqa: E402
from p5_target_labels import iterate20  # noqa: E402
from navsim_pdm.pdm_array_representation import ego_state_to_state_array, state_array_to_coords_array  # noqa: E402


def wrap(a):
    return (np.asarray(a, np.float64) + np.pi) % (2 * np.pi) - np.pi


class Ctx:
    """One token: scenery parts, ego, drivable union polygon, and a batch scorer for 20-pose plans."""

    def __init__(self, sc):
        from shapely.ops import unary_union
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        self.sc = sc
        self.parts = PT.metric_cache_parts(sc)
        self.ego = self.parts[1]
        self.s0 = ego_state_to_state_array(self.ego)
        dam = self.parts[5]
        idx = dam.get_indices_of_map_type([SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.DRIVABLE_AREA,
                                           SemanticMapLayer.CARPARK_AREA])
        self.drivable = unary_union([dam._geometries[i] for i in idx])
        self.vp = self.ego.car_footprint.vehicle_parameters

    def labels(self, t20):
        ctl = {"knot_exact": True, "knot_checks": 0}
        tc = iterate20(ND, t20, ND.to_navsim(t20), self.ego, ctl, None, "cubic")[0]
        track = lambda Pb: tracked8(ND, Pb, self.ego)                    # noqa: E731
        rl = R.solve(t20, tc - t20, [LAM], track, ND.to_navsim)[LAM]["label"]
        return {"T": t20, "C": tc, "R": rl}

    def score(self, plans20):
        """tracked-only scores of a list of 20-pose plans in ONE scorer call -> (list of metric dicts, sim states [n,41,11])"""
        res, sim = TS.score_arms(self.parts, [ND.to_navsim(p) for p in plans20], [])
        return res, sim

    def transparent(self, plan20):
        res, _ = TS.score_arms(self.parts, [], [TS.transparent_states(plan20, self.ego, self.s0, 0.2)])
        return res[0]

    def ext_dist(self, states):
        """[41] max over the 4 ego corners of the distance (m) to the drivable union (0 if all 4 inside) for one state array [41,11]."""
        from shapely.geometry import Point
        c = state_array_to_coords_array(states[None], self.vp)[0]          # [41, 5, 2]  corners + centre
        out = np.zeros(41)
        for k in range(41):
            out[k] = max(0.0 if self.drivable.contains(Point(*pt)) or self.drivable.touches(Point(*pt)) else self.drivable.distance(Point(*pt))
                         for pt in c[k, :4])
        return out

    def ego_frame(self, xy_world):
        ra = self.ego.rear_axle
        c, s = math.cos(-ra.heading), math.sin(-ra.heading)
        d = np.asarray(xy_world) - np.array([ra.x, ra.y])
        return np.stack([d[..., 0] * c - d[..., 1] * s, d[..., 0] * s + d[..., 1] * c], -1)


def contexts(tokens_by_log):
    """yield (log, token, Ctx) for {log: [tokens]}"""
    dbs = NS.index_dbs(db_root=DB_ROOT)
    for lg in sorted(tokens_by_log):
        for sc in NS.build_scenarios_for_log(dbs[lg], tokens_by_log[lg]):
            yield lg, sc._initial_lidar_token, Ctx(sc)
