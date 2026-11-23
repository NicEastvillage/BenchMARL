from dataclasses import dataclass, MISSING
from functools import lru_cache
from typing import Optional, Tuple

from imashielding.roads import RoadNetwork, RoadNetworkState
from imashielding.roads_reduction import AbstractRoundaboutSystem, roundabout_safety
from imashielding.shield import DGShield
from imashielding.systems import LabelledTransitionSystem

import torch
from torch import Tensor

from benchmarl.environments.shieldable.road_network import (
    ShieldableRoadNetworkEnv,
    build_model_and_shield,
)

OBS_SIZE = 3


@dataclass
class TaskConfig:
    n_agents: int = MISSING
    max_steps: int = MISSING
    road_per_circ: int = MISSING
    max_vel: int = MISSING
    high_acceleration_chance: float = MISSING
    min_dist: int = MISSING
    do_shielding: bool = MISSING
    resynthesize_shield: bool = MISSING
    vel_reward: float = MISSING
    violation_penalty: float = MISSING


@lru_cache(maxsize=16)
def get_shared_roundabout_model_and_shield(
    car_count: int,
    road_per_circ: int,
    max_vel: int,
    high_acceleration_chance: float,
    min_dist: int,
    do_shielding: bool,
    resynthesize_shield: bool = False,
) -> Tuple[RoadNetwork, LabelledTransitionSystem, object, Optional[DGShield]]:
    """Build the road network, abstraction and, optionally, the shield for one configuration.

    Cached because synthesis gets expensive as the road grows.  The synthesized
    shield is also cached on disk (see ``imashielding.shield_cache``), so separate
    runs reuse it unless ``resynthesize_shield`` is set.
    """
    network = RoadNetwork.create_roundabout_scenario(
        car_count=car_count,
        road_per_circ=road_per_circ,
        max_vel=max_vel,
        high_acceleration_chance=high_acceleration_chance,
    )
    abstraction = AbstractRoundaboutSystem(network)
    safety = roundabout_safety(min_dist)
    return build_model_and_shield(
        network, abstraction, safety, do_shielding, resynthesize_shield
    )


class RoundaboutEnv(ShieldableRoadNetworkEnv):
    """Cars circulating on a single-lane roundabout, driven by discrete accelerations.

        The safety objective is the invariant property `Box (AND_i p_i)`, where `p_i` is
        "the gap to the car in front is at least min_dist". An episode terminates as soon as
        any agent violates its safety property.

        With `shielding=True` every agent is restricted to the actions that keep its own
        clause `p_i` true.
        """

    OBSERVATION_SIZE = OBS_SIZE

    def __init__(
        self,
        n_agents: int = 6,
        max_steps: int = 100,
        road_per_circ: int = 32,
        max_vel: int = 4,
        high_acceleration_chance: float = 0.2,
        min_dist: int = 4,
        do_shielding: bool = True,
        resynthesize_shield: bool = False,
        vel_reward: float = 1.0,
        violation_penalty: float = 60.0,
        device: str = "cpu",
        seed: int = 0,
        **kwargs,
    ):
        self.road_per_circ = road_per_circ
        self.max_vel = max_vel
        self.high_acceleration_chance = high_acceleration_chance
        self.min_dist = min_dist
        self.resynthesize_shield = resynthesize_shield

        network, abstraction, safety, shield = get_shared_roundabout_model_and_shield(
            car_count=n_agents,
            road_per_circ=road_per_circ,
            max_vel=max_vel,
            high_acceleration_chance=high_acceleration_chance,
            min_dist=min_dist,
            do_shielding=do_shielding,
            resynthesize_shield=resynthesize_shield,
        )

        super().__init__(
            n_agents=n_agents,
            max_steps=max_steps,
            network=network,
            abstraction=abstraction,
            safety=safety,
            shield=shield,
            do_shielding=do_shielding,
            vel_reward=vel_reward,
            violation_penalty=violation_penalty,
            device=device,
            seed=seed,
            **kwargs,
        )

    def _observations(self, povs: Tuple[RoadNetworkState, ...]) -> Tensor:
        rows = []
        for pov in povs:
            ego_vel = pov.cars[0].vel
            if len(pov.cars) > 1:
                # normalize() places the ego at position 0, so dist is the position of car 1.
                front_dist, front_vel = pov.cars[1].pos, pov.cars[1].vel
            else:
                # Lone car: the car "in front" is itself one full lap ahead.
                front_dist, front_vel = self.road_per_circ, 0
            rows.append([ego_vel, front_dist, front_vel])
        return torch.tensor(rows, dtype=torch.float32, device=self.device)
