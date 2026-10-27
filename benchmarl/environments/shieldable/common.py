import copy
from typing import Optional, Dict, List, Callable

import torch
from tensordict import TensorDictBase
from torchrl.data import Composite
from torchrl.envs import EnvBase

from benchmarl.environments.common import Task, TaskClass
from benchmarl.environments.shieldable.roundabout import (
    SHIELD_STATUS_OFF,
    SHIELD_STATUS_MISSING,
    SHIELD_STATUS_UNSAFE,
    RoundaboutEnv,
)
from benchmarl.utils import DEVICE_TYPING


class ShieldableTaskClass(TaskClass):
    def get_env_fun(self, num_envs: int, continuous_actions: bool, seed: Optional[int], device: DEVICE_TYPING) -> \
    Callable[[], EnvBase]:

        config = copy.deepcopy(self.config)

        self.name = self.name.lower()

        if self.name == "roundabout":
            return lambda: RoundaboutEnv(
                device=device,
                seed=seed,
                **config
            )
        elif self.name == "double_roundabout":
            raise NotImplementedError("Double Roundabout task has not been implemented yet")

        raise AssertionError(f"Unknown scenario name for ShieldableTaskClass: '{self.name}'")

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_steps"]

    def has_render(self, env: EnvBase) -> bool:
        return False   # TODO

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def observation_spec(self, env: EnvBase) -> Composite:
        return self._split_observation_spec(env, "observation")

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None   # Not used for anything. Maybe rendering or statistics?

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None   # Used for critic input spec. If None, obs spec is used instead

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return self._split_observation_spec(env, "action_mask")

    def _split_observation_spec(self, env: EnvBase, keep: str) -> Optional[Composite]:
        """Carve a single per-group entry out of the env observation spec."""
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_spec = observation_spec[group]
            for key in list(group_spec.keys()):
                if key != keep:
                    del group_spec[key]
            if group_spec.is_empty():
                del observation_spec[group]
        return observation_spec if not observation_spec.is_empty() else None

    @staticmethod
    def log_info(batch: TensorDictBase) -> Dict[str, float]:

        # An episode terminates only when the safety property is violated.
        terminated = batch.get(("next", "agents", "terminated"), None)
        if terminated is None:
            return {}

        to_log = {
            "safety/violation_rate": terminated.any(-1).to(torch.float).mean().item(),
        }

        action_mask = batch.get(("agents", "action_mask"), None)
        if action_mask is not None:
            to_log["safety/legal_action_rate"] = action_mask.to(torch.float).mean().item()

        # How well the shield was able to restrict the agents. A shield that silently
        # falls back to "allow everything" would otherwise look like a working shield.
        shield_status = batch.get(("next", "agents", "shield_status"), None)
        if shield_status is not None:
            to_log["safety/shield_missing_rate"] = (shield_status == SHIELD_STATUS_MISSING).to(torch.float).mean().item()
            to_log["safety/shield_no_safe_action_rate"] = (
                (shield_status == SHIELD_STATUS_UNSAFE).to(torch.float).mean().item()
            )
            to_log["safety/shield_active_rate"] = (shield_status != SHIELD_STATUS_OFF).to(torch.float).mean().item()

        # Environment statistics. Observation columns are (ego_vel, front_dist, front_vel), and
        # front_dist is the gap to the car ahead, so min/max over agents are the extreme
        # gaps between consecutive cars.
        observation = batch.get(("next", "agents", "observation"), None)
        if observation is not None and observation.shape[-1] >= 2:
            to_log["env/average_velocity"] = observation[..., 0].mean().item()
            to_log["env/min_gap"] = observation[..., 1].min().item()
            to_log["env/max_gap"] = observation[..., 1].max().item()

        # Travelled distance is cumulative per episode, so read it at episode boundaries.
        travelled = batch.get(("next", "agents", "travelled_distance"), None)
        done = batch.get(("next", "done"), None)
        if travelled is not None and done is not None:
            episode_distance = travelled.reshape(-1, travelled.shape[-1])[done.squeeze(-1).reshape(-1).bool()]
            if episode_distance.numel() > 0:
                to_log["env/average_distance_travelled"] = episode_distance.mean().item()

        return to_log

    @staticmethod
    def env_name() -> str:
        return "shieldable"


class ShieldableTask(Task):

    ROUNDABOUT = None
    DOUBLE_ROUNDABOUT = None

    @staticmethod
    def associated_class():
        return ShieldableTaskClass