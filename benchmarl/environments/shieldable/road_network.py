"""Shared machinery for the shieldable road-network environments.

All shieldable tasks simulate a :class:`imashielding.roads.RoadNetwork` with
discrete acceleration actions and expose, for every agent, the safe actions that
the synthesized shield allows.  The scenarios differ in their road topology, in
the abstraction used for synthesis, and in their safety property; the episode
bookkeeping, reward and action-mask logic is identical.

Two abstraction flavours exist:

* :class:`~imashielding.roads_reduction.AbstractRoundaboutSystem` reduces the
  single roundabout to "ego + the car directly ahead" (see ``roundabout.py``).
* :class:`~imashielding.roads_reduction.AbstractRoadNetworkSystem` keeps every
  car in the agent's point of view, ordered by distance ahead.  This is what the
  crossing-roads and double-roundabout scenarios use.

The generic observation in this module encodes that full point of view, so the
policy sees exactly the state the shield is queried about.
"""

from __future__ import annotations

import heapq
import math
from typing import Callable, List, Optional, Tuple

import torch
from imashielding.roads import RoadNetwork, RoadNetworkState
from imashielding.shield import Shield, compute_shield
from imashielding.systems import LabelledTransitionSystem, StateSafety

from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torchrl.data import Categorical, Composite, UnboundedContinuous, UnboundedDiscrete
from torchrl.envs import EnvBase

BRAKE_ACTION = 0
COAST_ACTION = 1
ACT_SIZE = 3
ALL_ACTIONS = tuple(range(ACT_SIZE))

# Shield statuses for logging.  They allow detecting faulty shields and/or
# abstractions instead of silently masking (or not masking) an agent.
SHIELD_STATUS_OK = 0          # Shield working as expected
SHIELD_STATUS_MISSING = 1     # Shield was queried about a state not explored in synthesis
SHIELD_STATUS_UNSAFE = 2      # Shield was queried about a state with no safe actions
SHIELD_STATUS_OFF = 3         # Shielding disabled for this run
SHIELD_STATUS_RELEASED = 4    # Another agent violated its clause, so this agent is released


def build_model_and_shield(
    network: RoadNetwork,
    abstraction: LabelledTransitionSystem,
    safety: Callable[[RoadNetworkState], StateSafety],
    do_shielding: bool,
    resynthesize_shield: bool,
    shield_system=None,
) -> Tuple[
    RoadNetwork,
    LabelledTransitionSystem,
    Callable[[RoadNetworkState], StateSafety],
    Optional[Shield],
]:
    """Build the transition system, safety property and (optionally) the shield.

    Shield synthesis and loading from the on-disk cache are delegated to
    :func:`imashielding.shield.compute_shield`.  When shielding is enabled we
    additionally verify that the synthesized shield covers every agent's initial
    point of view, because an uncovered initial state would be reported as a
    missing shield at runtime instead of failing here with a useful message.

    ``abstraction`` produces the points of view used for observations and for
    detecting violations.  By default the shield is synthesized for it too.  If
    ``shield_system`` is given (e.g. an ``EgoCentricRoadSystem``), the shield is
    synthesized for that system instead and wrapped so that it is queried with
    the same points of view: a point of view is a concrete state with the agent
    as car 0, which ``shield_system.normalize(pov, 0)`` maps to its own states.
    """
    if not do_shielding:
        return network, abstraction, safety, None

    if shield_system is None:
        shield = compute_shield(abstraction, safety, force_recompute=resynthesize_shield)
    else:
        synthesized = compute_shield(
            shield_system, shield_system.safety(), force_recompute=resynthesize_shield
        )
        shield = synthesized.with_normalizer(lambda pov: shield_system.normalize(pov, 0))

    init_state = network.get_init_state()
    problems = []
    for car in init_state.cars:
        pov = abstraction.normalize(init_state, car.id)
        if not shield.contains(pov):
            problems.append(f"agent {car.id}'s initial point of view is not covered")
        elif shield.is_released(pov):
            problems.append(f"agent {car.id}'s initial point of view is a release state")
    if problems:
        raise RuntimeError(
            "The synthesized shield cannot be used safely for this configuration: "
            + "; ".join(problems)
            + ". The abstraction does not cover the initial state."
        )

    return network, abstraction, safety, shield


class ShieldableRoadNetworkEnv(EnvBase):
    """Base environment for cars on an ``imashielding`` road network.

    Subclasses provide the concrete road network, abstraction, safety property
    and shield (usually through a cached factory) and may override
    :meth:`_observations`.  Unless overridden, the observation is the agent's
    full point of view under ``AbstractRoadNetworkSystem``, with one block of
    ``num_roads + 4`` features per car:

    ``[velocity, distance_ahead / network_length, road_one_hot..., position / segment_length, just_merged]``

    Cars are ordered by distance ahead and the ego is always the first car; its
    distance-ahead entry is zero.  The shield is queried with the same point of
    view (possibly through a normalizing wrapper, see :func:`build_model_and_shield`),
    so it never sees more than the observation does.
    """

    #: Fixed observation size, or ``None`` to derive it from the number of cars
    #: and roads (the generic point-of-view encoding).
    OBSERVATION_SIZE: Optional[int] = None

    def __init__(
        self,
        n_agents: int,
        max_steps: int,
        network: RoadNetwork,
        abstraction,
        safety,
        shield: Optional[Shield],
        do_shielding: bool,
        vel_reward: float,
        violation_penalty: float,
        device: str = "cpu",
        seed: int = 0,
        **kwargs,
    ):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.network = network
        self.abstraction = abstraction
        self.safety = safety
        self.shield = shield
        self.do_shielding = do_shielding
        self.vel_reward = vel_reward
        self.violation_penalty = violation_penalty

        if n_agents != len(network.cars):
            raise ValueError(
                f"n_agents={n_agents} does not match the network's {len(network.cars)} cars"
            )

        self.group_map = {"agents": [f"agent_{i}" for i in range(n_agents)]}

        self.cur_state: RoadNetworkState = network.get_init_state()
        self._step_count = 0
        self._travelled = torch.zeros(n_agents, dtype=torch.float32, device=device)
        self._generator = torch.Generator(device="cpu")
        self._seed = seed
        self._generator.manual_seed(seed)

        self.observation_size = (
            self.OBSERVATION_SIZE
            if self.OBSERVATION_SIZE is not None
            else n_agents * (len(network.roads) + 4)
        )

        super().__init__(device=device, batch_size=torch.Size([]))

        self._make_specs()

    # ------------------------------------------------------------------ specs

    def _make_specs(self):
        n, device = self.n_agents, self.device

        observation_spec = UnboundedContinuous(
            shape=torch.Size((n, self.observation_size)), device=device
        )
        action_mask_spec = Categorical(
            n=2, shape=torch.Size((n, ACT_SIZE)), dtype=torch.bool, device=device
        )
        shield_status_spec = UnboundedDiscrete(shape=torch.Size((n,)), device=device)   # Diagnostics
        travelled_spec = UnboundedContinuous(shape=torch.Size((n,)), device=device)     # Diagnostics
        gap_spec = UnboundedContinuous(shape=torch.Size((n,)), device=device)           # Diagnostics
        episode_length_spec = UnboundedContinuous(shape=torch.Size((n,)), device=device)  # Diagnostics

        self.observation_spec = Composite(
            {
                "agents": Composite(
                    {
                        "observation": observation_spec,
                        "action_mask": action_mask_spec,
                        "shield_status": shield_status_spec,
                        "travelled_distance": travelled_spec,
                        "gap_ahead": gap_spec,
                        "episode_length": episode_length_spec,
                    },
                    shape=torch.Size((n,)),
                )
            }
        )

        self.action_spec = Composite(
            {
                "agents": Composite(
                    {"action": Categorical(n=ACT_SIZE, shape=torch.Size((n,)), device=device)},
                    shape=torch.Size((n,)),
                )
            }
        )

        self.reward_spec = Composite(
            {
                "agents": Composite(
                    {"reward": UnboundedContinuous(shape=torch.Size((n, 1)), device=device)},
                    shape=torch.Size((n,)),
                )
            }
        )

        def flags(shape):
            return {
                name: Categorical(n=2, shape=torch.Size(shape), dtype=torch.bool, device=device)
                for name in ("done", "terminated", "truncated")
            }

        self.done_spec = Composite(
            {
                **flags((1,)),
                "agents": Composite({**flags((n, 1))}, shape=torch.Size((n,))),
            }
        )

    # ------------------------------------------------------------------ transitions

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        if tensordict is not None:
            reset_flag = tensordict.get("_reset", None)
            if reset_flag is not None and not reset_flag.all():
                raise RuntimeError(
                    f"{type(self)}._reset was called without a reset signal, "
                    f"got '_reset' = {reset_flag}"
                )

        self.cur_state = self.network.get_init_state()
        self._step_count = 0
        self._travelled = torch.zeros(self.n_agents, dtype=torch.float32, device=self.device)
        povs = tuple(
            self.abstraction.normalize(self.cur_state, car.id)
            for car in self.cur_state.cars
        )
        return self._write_output(povs, reward=None, terminated=False, truncated=False)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action_tensor = tensordict["agents", "action"]
        actions = tuple(int(a) for a in action_tensor)

        self.cur_state = self._sample_successor(actions)
        self._step_count += 1
        self._travelled = self._travelled + torch.tensor(
            [car.vel for car in self.cur_state.cars], dtype=torch.float32, device=self.device
        )

        povs = tuple(
            self.abstraction.normalize(self.cur_state, car.id)
            for car in self.cur_state.cars
        )

        violated = torch.tensor(
            [self.safety(pov) == StateSafety.UNSAFE for pov in povs],
            dtype=torch.bool,
            device=self.device,
        )
        terminated = bool(violated.any())
        truncated = self._step_count >= self.max_steps

        return self._write_output(
            povs,
            reward=self._reward(povs, action_tensor, violated),
            terminated=terminated,
            truncated=truncated,
        )

    def _sample_successor(self, actions: Tuple[int, ...]) -> RoadNetworkState:
        successors = self.network.get_successors(self.cur_state, actions)
        chances = torch.tensor([s.chance for s in successors], dtype=torch.float64)
        index = int(
            torch.multinomial(
                chances / chances.sum(), 1, generator=self._generator
            ).item()
        )
        return successors[index].state

    # ------------------------------------------------------------------ outputs

    def _write_output(
        self,
        povs: Tuple[RoadNetworkState, ...],
        reward: Optional[Tensor],
        terminated: bool,
        truncated: bool,
    ) -> TensorDictBase:
        n = self.n_agents
        root_flags = {
            "done": torch.full((1,), terminated or truncated, dtype=torch.bool, device=self.device),
            "terminated": torch.full((1,), terminated, dtype=torch.bool, device=self.device),
            "truncated": torch.full((1,), truncated, dtype=torch.bool, device=self.device),
        }

        action_mask, shield_status = self._action_mask(povs)

        tensordict_out = TensorDict(batch_size=self.batch_size, device=self.device)
        tensordict_out["agents", "observation"] = self._observations(povs)
        tensordict_out["agents", "action_mask"] = action_mask
        tensordict_out["agents", "shield_status"] = shield_status
        tensordict_out["agents", "travelled_distance"] = self._travelled
        tensordict_out["agents", "gap_ahead"] = self._gaps_ahead(povs)
        # Steps taken so far in this episode (same for every agent); read at episode end.
        tensordict_out["agents", "episode_length"] = torch.full(
            (n,), float(self._step_count), dtype=torch.float32, device=self.device
        )
        if reward is not None:
            tensordict_out["agents", "reward"] = reward
        for name, flag in root_flags.items():
            tensordict_out[name] = flag
            tensordict_out["agents", name] = flag.repeat(n, 1)
        return tensordict_out

    def _observations(self, povs: Tuple[RoadNetworkState, ...]) -> Tensor:
        """Encode every car in the agent's point of view (see the class docstring)."""
        n_roads = len(self.network.roads)
        lengths = [float(road.length) for road in self.network.roads]
        total_length = float(sum(lengths))

        rows: List[List[float]] = []
        for pov in povs:
            distances = self._distances_ahead(pov)
            features: List[float] = []
            for index, car in enumerate(pov.cars):
                features.append(float(car.vel))
                if index == 0:
                    features.append(0.0)
                else:
                    distance = min(distances[index - 1], total_length)
                    features.append(distance / total_length)
                features.extend(1.0 if road == car.road else 0.0 for road in range(n_roads))
                features.append(car.pos / lengths[car.road])
                features.append(1.0 if car.just_merged else 0.0)
            rows.append(features)

        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def _gaps_ahead(self, povs: Tuple[RoadNetworkState, ...]) -> Tensor:
        """Distance to the nearest car ahead, used for logging only."""
        total_length = float(sum(road.length for road in self.network.roads))
        gaps = []
        for pov in povs:
            distances = self._distances_ahead(pov)
            gaps.append(min(distances[0], total_length) if distances else total_length)
        return torch.tensor(gaps, dtype=torch.float32, device=self.device)

    def _distances_ahead(self, state: RoadNetworkState) -> List[float]:
        """Distance from ``state.cars[0]`` to every other car, following the roads.

        This mirrors ``AbstractRoadNetworkSystem.normalize``: run a Dijkstra over
        the road segments to find how far the ego is from the start of each
        segment, then add the position of the car on its own segment.
        """
        roads = self.network.roads
        ego = state.cars[0]
        dist_start = [math.inf] * len(roads)
        queue = []

        def relax(segment: int, distance: float) -> None:
            if distance < dist_start[segment]:
                dist_start[segment] = distance
                heapq.heappush(queue, (distance, segment))

        ego_segment = roads[ego.road]
        for end in {ego_segment.end_left, ego_segment.end_right}:
            relax(end, ego_segment.length - ego.pos)

        while queue:
            distance, segment_id = heapq.heappop(queue)
            if distance > dist_start[segment_id]:
                continue
            segment = roads[segment_id]
            for end in {segment.end_left, segment.end_right}:
                relax(end, distance + segment.length)

        distances = []
        for car in state.cars[1:]:
            if car.road == ego.road and car.pos >= ego.pos:
                distances.append(float(car.pos - ego.pos))
            else:
                distances.append(dist_start[car.road] + car.pos)
        return distances

    # ------------------------------------------------------------------ shield

    def _action_mask(self, povs: Tuple[RoadNetworkState, ...]) -> Tuple[Tensor, Tensor]:
        """The safe actions each agent may take, plus a per-agent shield status code."""
        n = self.n_agents
        mask = torch.zeros(n, ACT_SIZE, dtype=torch.bool, device=self.device)
        shield_status = torch.full((n,), SHIELD_STATUS_OFF, dtype=torch.long, device=self.device)

        if self.shield is None:
            mask[:] = True
            return mask, shield_status

        for agent, pov in enumerate(povs):
            actions, agent_status = self._safe_actions(pov)
            shield_status[agent] = agent_status
            for action in actions:
                mask[agent, action] = True
        return mask, shield_status

    def _safe_actions(self, pov: RoadNetworkState) -> Tuple[Tuple[int, ...], int]:
        if not self.shield.contains(pov):
            # Unreachable for a policy that respects the mask, so this signals a bug or an
            # unsound abstraction. Be permissive: an all-masked row makes MaskedCategorical
            # produce NaN.
            return ALL_ACTIONS, SHIELD_STATUS_MISSING

        if self.shield.is_released(pov):
            # Another agent already violated its clause, so this agent's obligation is
            # released. The at-fault agent is unsafe and ends the episode this step; do
            # not mask this agent, which would otherwise leave an empty action row.
            return ALL_ACTIONS, SHIELD_STATUS_RELEASED

        safe = self.shield.get_safe_actions(pov)
        if not safe:
            # The state is already unsafe, so no action restores the invariant.
            return (BRAKE_ACTION,), SHIELD_STATUS_UNSAFE
        return safe, SHIELD_STATUS_OK

    def _reward(self, povs, action_tensor: Tensor, violated: Tensor) -> Tensor:
        vel = torch.tensor([float(pov.cars[0].vel) for pov in povs], dtype=torch.float32, device=self.device)
        accel_cost = (action_tensor != COAST_ACTION).to(torch.float32)
        reward = self.vel_reward * vel - accel_cost - self.violation_penalty * violated.to(torch.float32)
        return reward.unsqueeze(-1)

    def _set_seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            self._seed = seed
            self._generator.manual_seed(seed)
