from dataclasses import dataclass, MISSING
from functools import lru_cache
from typing import Callable, Optional, Tuple

import torch
from imashielding.roads import RoadNetwork, RoadNetworkState
from imashielding.roads_reduction import AbstractRoundaboutSystem, roundabout_safety
from imashielding.shield import DGShield, compute_shield
from imashielding.systems import StateSafety

from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torchrl.data import Categorical, Composite, UnboundedContinuous, UnboundedDiscrete
from torchrl.envs import EnvBase

OBS_SIZE = 3
ACT_SIZE = 3
BRAKE_ACTION = 0
COAST_ACTION = 1
ALL_ACTIONS = tuple(range(ACT_SIZE))

# Shield statuses for logging. Allows us to detect faulty shields and/or abstractions
SHIELD_STATUS_OK = 0        # Shield working as expected
SHIELD_STATUS_MISSING = 1   # Shield was queried about a state not explored in synthesis
SHIELD_STATUS_UNSAFE = 2    # Shield was queried about state with no safe actions
SHIELD_STATUS_OFF = 3       # Shielding disabled for this run


@dataclass
class TaskConfig:
    n_agents: int = MISSING
    max_steps: int = MISSING
    road_length: int = MISSING
    max_vel: int = MISSING
    min_dist: int = MISSING
    do_shielding: bool = MISSING
    vel_reward: float = MISSING
    violation_penalty: float = MISSING


@lru_cache(maxsize=16)
def get_shared_roundabout_model_and_shield(
    car_count: int, road_length: int, max_vel: int, min_dist: int, do_shielding: bool
) -> Tuple[RoadNetwork, AbstractRoundaboutSystem, Optional[DGShield]]:
    """Build the road network, abstraction and, optionally, the shield for one configuration.

    Cached because synthesis gets expensive as the road grows.
    """
    network = RoadNetwork.create_roundabout_scenario(
        car_count=car_count, road_length=road_length, max_vel=max_vel
    )
    abstraction = AbstractRoundaboutSystem(network, 0)

    if not do_shielding:
        return network, abstraction, None

    shield = compute_shield(abstraction, roundabout_safety(min_dist))

    # Check all agent's normalized initial state is defined in the shield
    init_state = network.get_init_state()
    uncovered = [
        car.id
        for car in init_state.cars
        if not shield.contains(abstraction.normalize(init_state, car.id))
    ]
    if uncovered:
        raise RuntimeError(
            f"The synthesized shield does not cover the initial point of view of agent(s) "
            f"{uncovered}; the abstraction cannot be used safely for this configuration."
        )
    return network, abstraction, shield


class RoundaboutEnv(EnvBase):
    """Cars circulating on a single-lane roundabout, driven by discrete accelerations.

        The safety objective is the invariant property `Box (AND_i p_i)`, where `p_i` is
        "the gap to the car in front is at least min_dist". An episode terminates as soon as
        any agent violates its safety property.

        With `shielding=True` every agent is restricted to the actions that keep its own
        clause `p_i` true.
        """

    def __init__(
        self,
        n_agents: int = 6,
        max_steps: int = 100,
        road_length: int = 32,
        max_vel: int = 4,
        min_dist: int = 4,
        do_shielding: bool = True,
        vel_reward: float = 1.0,
        violation_penalty: float = 60.0,
        device: str = "cpu",
        seed: int = 0,
        **kwargs,
    ):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.road_length = road_length
        self.min_dist = min_dist
        self.do_shielding = do_shielding
        self.vel_reward = vel_reward
        self.violation_penalty = violation_penalty

        self.roundabout, self.abstraction, self.shield = get_shared_roundabout_model_and_shield(
            car_count=n_agents, road_length=road_length, max_vel=max_vel, min_dist=min_dist, do_shielding=do_shielding
        )

        self.safety = roundabout_safety(min_dist)

        self.group_map = {"agents": [f"agent_{i}" for i in range(n_agents)]}

        self.cur_state: RoadNetworkState = self.roundabout.get_init_state()
        self._step_count = 0
        self._generator = torch.Generator(device="cpu")
        self._seed = seed
        self._generator.manual_seed(seed)

        super().__init__(device=device, batch_size=torch.Size([]))

        self._make_specs()

    # ------------------------------------------------------------------ specs

    def _make_specs(self):
        n, device = self.n_agents, self.device

        observation_spec = UnboundedContinuous(
            shape=torch.Size((n, OBS_SIZE)), device=device
        )
        action_mask_spec = Categorical(
            n=2, shape=torch.Size((n, ACT_SIZE)), dtype=torch.bool, device=device
        )
        shield_status_spec = UnboundedDiscrete(shape=torch.Size((n,)), device=device)   # Diagnostics

        self.observation_spec = Composite(
            {
                "agents": Composite(
                    {
                        "observation": observation_spec,
                        "action_mask": action_mask_spec,
                        "shield_status": shield_status_spec,
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

        self.cur_state = self.roundabout.get_init_state()
        self._step_count = 0
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
        successors = self.roundabout.get_successors(self.cur_state, actions)
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
        if reward is not None:
            tensordict_out["agents", "reward"] = reward
        for name, flag in root_flags.items():
            tensordict_out[name] = flag
            tensordict_out["agents", name] = flag.repeat(n, 1)
        return tensordict_out

    def _observations(self, povs: Tuple[RoadNetworkState, ...]) -> Tensor:
        rows = []
        for pov in povs:
            ego_vel = pov.cars[0].vel
            if len(pov.cars) > 1:
                # normalize() places the ego at position 0, so dist is the position of car 1.
                front_dist, front_vel = pov.cars[1].pos, pov.cars[1].vel
            else:
                # Lone car: the car "in front" is itself one full lap ahead.
                front_dist, front_vel = self.road_length, 0
            rows.append([ego_vel, front_dist, front_vel])
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def _action_mask(self, povs: Tuple[RoadNetworkState, ...]) -> Tuple[Tensor, Tensor]:
        """The safe actions each agent may take, plus a per-agent shield status code."""
        n = self.n_agents
        mask = torch.zeros(n, ACT_SIZE, dtype=torch.bool, device=self.device)
        shield_status = torch.full((n,), SHIELD_STATUS_OFF, dtype=torch.long, device=self.device)

        if self.shield is None:
            mask[:] = True
            return mask, shield_status

        for agent, pov in enumerate(povs):
            actions, agent_status = self._safe_actions(agent, pov)
            shield_status[agent] = agent_status
            for action in actions:
                mask[agent, action] = True
        return mask, shield_status

    def _safe_actions(self, agent: int, pov: RoadNetworkState) -> Tuple[Tuple[int, ...], int]:
        if not self.shield.contains(pov):
            # Unreachable for a policy that respects the mask, so this signals a bug or an
            # unsound abstraction. Be permissive: an all-masked row makes MaskedCategorical
            # produce NaN.
            return ALL_ACTIONS, SHIELD_STATUS_MISSING

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