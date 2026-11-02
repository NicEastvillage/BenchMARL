"""Replay a trained BenchMARL policy in the ``imashielding`` pygame visualizer.

The script rebuilds the experiment from the run's ``config.pkl`` (two levels above the
checkpoint), restores the policy weights from the checkpoint, reconstructs the
road network for the task configuration, and then hands a ``policy(shields, state)``
callable to :func:`imashielding.visualize.visualize`.

Usage
-----
    # A checkpoint file
    python visualize_shieldable.py multirun/2026-10-29/15-40-28/0/<run>/checkpoints/checkpoint_66000.pt

    # The run folder (the newest checkpoint is picked automatically)
    python visualize_shieldable.py multirun/2026-10-29/15-40-28/0/<run>

    # Start paused, step with `e`, and ignore the shield
    python visualize_shieldable.py <run> --steps 0 --no-shield

Controls: space pauses, `e` single-steps, `r` resets, `esc` quits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.environments.shieldable.roundabout import RoundaboutEnv
from benchmarl.experiment import Experiment
from imashielding.visualize import visualize


def _checkpoint_step(path: Path) -> int:
    """Sort key for ``checkpoint_<frames>.pt`` files (numeric, not lexicographic)."""
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def resolve_checkpoint(raw: str) -> Path:
    """Turn a checkpoint file, checkpoints dir, or run dir into a checkpoint path."""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    if path.is_file():
        if path.suffix != ".pt":
            raise ValueError(
                f"{path} is not a checkpoint (.pt) file. Pass a checkpoint or a run folder."
            )
        return path.resolve()

    if not path.is_dir():
        raise FileNotFoundError(f"No checkpoint or directory found at {path}")

    # Direct checkpoint files, in the folder or in its checkpoints/ subfolder.
    candidates = sorted(path.glob("checkpoint_*.pt")) + sorted(
        path.glob("checkpoints/checkpoint_*.pt")
    )
    if not candidates:
        # A multirun arm directory contains a single run directory.
        candidates = sorted(path.glob("*/checkpoint_*.pt")) + sorted(
            path.glob("*/checkpoints/checkpoint_*.pt")
        )
    if not candidates:
        # Last resort: search nested run folders.
        candidates = sorted(path.glob("**/checkpoint_*.pt"))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint_*.pt found under {path}. Pass the .pt file explicitly."
        )

    return max(candidates, key=_checkpoint_step).resolve()


def load_experiment(checkpoint: Path) -> Experiment:
    """Rebuild an experiment from the run's config and restore the checkpoint."""
    # Loggers are disabled: replaying a policy must not start a new wandb/csv run.
    return Experiment.reload_from_file(
        str(checkpoint), experiment_patch={"loggers": []}
    )


def build_environment(
    experiment: Experiment, shield_override: Optional[bool] = None
) -> RoundaboutEnv:
    """Build the road network/environment for the checkpoint's task config."""
    config = dict(experiment.task.config)
    if shield_override is not None:
        config["do_shielding"] = shield_override
    return RoundaboutEnv(**config)


def make_policy(
    experiment: Experiment,
    env: RoundaboutEnv,
    stochastic: bool = False,
) -> Callable:
    """Build the ``policy(shields, state) -> actions`` callable expected by visualize."""
    exploration = ExplorationType.RANDOM if stochastic else ExplorationType.DETERMINISTIC

    def policy(_shields, state) -> tuple[int, ...]:
        # Exactly the observation + action mask the environment gives the policy.
        povs = tuple(
            env.abstraction.normalize(state, car.id) for car in state.cars
        )
        observation = env._observations(povs)
        action_mask, _ = env._action_mask(povs)

        tensordict = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": observation,
                        "action_mask": action_mask,
                    },
                    batch_size=(env.n_agents,),
                )
            },
            batch_size=(),
        )

        with torch.no_grad(), set_exploration_type(exploration):
            tensordict = experiment.policy(tensordict)

        return tuple(int(action) for action in tensordict["agents", "action"])

    return policy


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained BenchMARL shieldable policy with pygame."
    )
    parser.add_argument(
        "checkpoint",
        help="Path to a checkpoint_*.pt, a checkpoints/ folder, or a run folder.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Pause after N steps. 0 starts paused (default: run continuously).",
    )
    parser.add_argument("--fps", type=int, default=8, help="Frames per second (default: 8).")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy instead of using its deterministic/mode action.",
    )
    shield_group = parser.add_mutually_exclusive_group()
    shield_group.add_argument(
        "--shield",
        dest="shield",
        action="store_true",
        help="Force shielding on (default: use the value stored in the run config).",
    )
    shield_group.add_argument(
        "--no-shield",
        dest="shield",
        action="store_false",
        help="Force shielding off (default: use the value stored in the run config).",
    )
    parser.set_defaults(shield=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    try:
        checkpoint = resolve_checkpoint(args.checkpoint)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

    print(f"Loading checkpoint: {checkpoint}")
    experiment = load_experiment(checkpoint)
    env = build_environment(experiment, shield_override=args.shield)

    frames = experiment.total_frames
    print(f"  algorithm: {type(experiment.algorithm_config).__name__}")
    print(f"  frames:    {frames}")
    print(f"  task:      {experiment.task.name}")
    for key, value in dict(experiment.task.config).items():
        print(f"    {key}: {value}")
    print(f"  shield:    {'on' if env.shield is not None else 'off'}")
    print(f"  actions:   {'stochastic' if args.stochastic else 'deterministic'}")

    policy = make_policy(experiment, env, stochastic=args.stochastic)
    visualize(
        env.roundabout,
        steps=args.steps,
        fps=args.fps,
        policy=policy,
    )


if __name__ == "__main__":
    main()
