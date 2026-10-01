"""Two helpers from nuplan-devkit `nuplan/planning/simulation/planner/ml_planner/transform_utils.py`,
VERBATIM (Apache-2.0, Motional). DriveRL's nuplan fork on the pod lacks `ml_planner`; NAVSIM's
`transform_trajectory` imports exactly these two. Source: the official nuplan-devkit the NAVSIM harness
runs with (navsim-crun), file sha256 f133c81ccb6c3bdb."""
from typing import List

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters


def _se2_vel_acc_to_ego_state(
    state: StateSE2,
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32],
    timestamp: float,
    vehicle: VehicleParameters,
) -> EgoState:
    """
    Convert StateSE2, velocity and acceleration to EgoState given a timestamp.

    :param state: input SE2 state
    :param velocity: [m/s] longitudinal velocity, lateral velocity
    :param acceleration: [m/s^2] longitudinal acceleration, lateral acceleration
    :param timestamp: [s] timestamp of state
    :return: output agent state
    """
    return EgoState.build_from_rear_axle(
        rear_axle_pose=state,
        rear_axle_velocity_2d=StateVector2D(*velocity),
        rear_axle_acceleration_2d=StateVector2D(*acceleration),
        tire_steering_angle=0.0,
        time_point=TimePoint(int(timestamp * 1e6)),
        vehicle_parameters=vehicle,
        is_in_auto_mode=True,
    )


def _get_fixed_timesteps(state: EgoState, future_horizon: float, step_interval: float) -> List[float]:
    """
    Get a fixed array of timesteps starting from a state's time.

    :param state: input state
    :param future_horizon: [s] future time horizon
    :param step_interval: [s] interval between steps in the array
    :return: constructed timestep list
    """
    timesteps = np.arange(0.0, future_horizon, step_interval) + step_interval
    timesteps += state.time_point.time_s

    return list(timesteps.tolist())
