from dataclasses import dataclass, MISSING
from functools import lru_cache
from typing import Optional, Tuple

from imashielding.roads import RoadNetwork
from imashielding.roads_egocentric import EgoCentricRoadSystem
from imashielding.roads_reduction import AbstractRoadNetworkSystem, road_safety
from imashielding.shield import Shield
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
def get_shared_crossing_roads_model_and_shield(
    car_count: int,
    road_per_circ: int,
    max_vel: int,
    high_acceleration_chance: float,
    min_dist: int,
    merge_space: int,
    do_shielding: bool,
    resynthesize_shield: bool = False,
) -> Tuple[RoadNetwork, LabelledTransitionSystem, object, Optional[Shield]]:
    """Build the crossing-roads network, abstraction and, optionally, the shield.

    The shield is synthesized for the ego-centric abstraction
    (:class:`imashielding.roads_egocentric.EgoCentricRoadSystem`), whose state space does
    not depend on the number of cars; observations still use the full point of view
    (:class:`imashielding.roads_reduction.AbstractRoadNetworkSystem`).  The shield is
    cached on disk by :func:`imashielding.shield.compute_shield`, so keep
    ``resynthesize_shield`` off unless the configuration changed.
    """
    network = RoadNetwork.create_crossing_roads_scenario(
        car_count=car_count,
        road_per_circ=road_per_circ,
        max_vel=max_vel,
        high_acceleration_chance=high_acceleration_chance,
    )
    abstraction = AbstractRoadNetworkSystem(network)
    safety = road_safety(network, min_dist=min_dist, merge_space=merge_space)
    shield_system = EgoCentricRoadSystem(network, min_dist=min_dist, merge_space=merge_space)
    return build_model_and_shield(
        network, abstraction, safety, do_shielding, resynthesize_shield,
        shield_system=shield_system,
    )


class CrossingRoadsEnv(ShieldableRoadNetworkEnv):
    """Two crossing roads, driven by discrete accelerations.

    Cars circulate on two arcs joined by a straight segment.  One arc is a
    priority road; the other must yield when it merges back into the straight.
    The safety property is ``imashielding.roads_reduction.road_safety``: keep
    ``min_dist`` cells to the car ahead and leave ``merge_space`` cells behind
    when merging.  Episodes terminate when an agent violates its clause.

    Each agent observes its full normalized point of view (see
    :class:`ShieldableRoadNetworkEnv`), so it sees the other cars, their roads
    and the gaps to them.
    """

    def __init__(
        self,
        n_agents: int = 5,
        max_steps: int = 200,
        road_per_circ: int = 50,
        max_vel: int = 3,
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

        network, abstraction, safety, shield = get_shared_crossing_roads_model_and_shield(
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
