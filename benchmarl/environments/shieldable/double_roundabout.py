from dataclasses import dataclass, MISSING
from functools import lru_cache
from typing import Optional, Tuple

from imashielding.roads import RoadNetwork
from imashielding.roads_reduction import AbstractRoadNetworkSystem, road_safety
from imashielding.shield import DGShield
from imashielding.systems import LabelledTransitionSystem

from benchmarl.environments.shieldable.road_network import (
    ShieldableRoadNetworkEnv,
    build_model_and_shield,
)


@dataclass
class TaskConfig:
    n_agents: int = MISSING
    max_steps: int = MISSING
    road_per_circ: int = MISSING
    max_vel: int = MISSING
    high_acceleration_chance: float = MISSING
    min_dist: int = MISSING
    merge_space: int = MISSING
    do_shielding: bool = MISSING
    resynthesize_shield: bool = MISSING
    vel_reward: float = MISSING
    violation_penalty: float = MISSING


@lru_cache(maxsize=16)
def get_shared_double_roundabout_model_and_shield(
    car_count: int,
    road_per_circ: int,
    max_vel: int,
    high_acceleration_chance: float,
    min_dist: int,
    merge_space: int,
    do_shielding: bool,
    resynthesize_shield: bool = False,
) -> Tuple[RoadNetwork, LabelledTransitionSystem, object, Optional[DGShield]]:
    """Build the double-roundabout network, abstraction and, optionally, the shield.

    Two roundabouts are joined by two one-way connecting roads, so cars either
    circulate on one roundabout or drive over to the other one.  Shield synthesis
    scales badly with the number of cars (two cars take seconds to a minute,
    three cars can take hours).  The result is cached on disk by
    :func:`imashielding.shield.compute_shield`, so keep ``resynthesize_shield``
    off unless the configuration changed.
    """
    network = RoadNetwork.create_double_roundabout_scenario(
        car_count=car_count,
        road_per_circ=road_per_circ,
        max_vel=max_vel,
        high_acceleration_chance=high_acceleration_chance,
    )
    abstraction = AbstractRoadNetworkSystem(network)
    safety = road_safety(network, min_dist=min_dist, merge_space=merge_space)
    return build_model_and_shield(
        network, abstraction, safety, do_shielding, resynthesize_shield
    )


class DoubleRoundaboutEnv(ShieldableRoadNetworkEnv):
    """Two roundabouts connected by a pair of straight roads.

    Cars circulate on either roundabout and can switch between them through the
    connecting roads.  The safety property is
    ``imashielding.roads_reduction.road_safety``: keep ``min_dist`` cells to the
    car ahead and leave ``merge_space`` cells behind when merging onto another
    road.  Episodes terminate when an agent violates its clause.

    Each agent observes its full normalized point of view (see
    :class:`ShieldableRoadNetworkEnv`), so it sees the other cars, the road they
    are on and the gaps to them.
    """

    def __init__(
        self,
        n_agents: int = 2,
        max_steps: int = 200,
        road_per_circ: int = 32,
        max_vel: int = 4,
        high_acceleration_chance: float = 0.2,
        min_dist: int = 3,
        merge_space: int = 8,
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
        self.merge_space = merge_space
        self.resynthesize_shield = resynthesize_shield

        network, abstraction, safety, shield = get_shared_double_roundabout_model_and_shield(
            car_count=n_agents,
            road_per_circ=road_per_circ,
            max_vel=max_vel,
            high_acceleration_chance=high_acceleration_chance,
            min_dist=min_dist,
            merge_space=merge_space,
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
