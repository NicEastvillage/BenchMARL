"""A/B comparison: shielded vs unshielded learning on the roundabout task.

Runs the same algorithm, seed and frame budget with and without the shield. Both arms
can be run concurrently (they write to different folders), which is much faster than
running them back to back.

    python ab_compare.py <seed> <max_frames> <out_dir> [shielded|unshielded|both]
"""
import sys
from pathlib import Path

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import ShieldableTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

SEED = int(sys.argv[1])
MAX_FRAMES = int(sys.argv[2])
OUT = Path(sys.argv[3])
ARM = sys.argv[4] if len(sys.argv) > 4 else "both"


def run(shielding: bool):
    task_config = dict(ShieldableTask.ROUNDABOUT.get_from_yaml().config)
    task_config["shielding"] = shielding
    task = ShieldableTask.ROUNDABOUT.get_task(task_config)

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.loggers = ["csv"]
    experiment_config.create_json = False
    experiment_config.max_n_frames = MAX_FRAMES
    experiment_config.evaluation_interval = MAX_FRAMES // 2
    save_folder = OUT / ("shielded" if shielding else "unshielded")
    save_folder.mkdir(parents=True, exist_ok=True)
    experiment_config.save_folder = str(save_folder)

    experiment = Experiment(
        task=task,
        algorithm_config=MappoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=SEED,
        config=experiment_config,
    )
    experiment.run()


arms = ["shielded", "unshielded"] if ARM == "both" else [ARM]
for arm in arms:
    print(f"\n===== {arm.upper()} (seed={SEED}) =====", flush=True)
    run(arm == "shielded")
