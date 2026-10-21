"""Summarize an A/B run: read the csv scalars of both arms and print a comparison."""
import sys
from pathlib import Path

OUT = Path(sys.argv[1])

METRICS = [
    ("safety/violation_rate", "violation rate"),
    ("safety/legal_action_rate", "legal action rate"),
    ("safety/shield_missing_rate", "shield missing"),
    ("safety/shield_no_safe_action_rate", "shield no-safe-action"),
    ("safety/shield_active_rate", "shield active"),
    ("collection/agents/reward/episode_reward/mean", "ep reward (collect)"),
    ("eval/reward/episode_reward/mean", "ep reward (eval)"),
    ("eval/reward/episode_len_mean", "eval episode len"),
]


def read(folder, key):
    files = sorted(folder.rglob(f"{key.replace('/', '_')}.csv"))
    if not files:
        return None
    rows = [float(line.split(",")[1]) for line in files[0].read_text().strip().splitlines()]
    return rows


def fmt(values):
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4g}"
    return f"{values[0]:.4g} -> {values[-1]:.4g}"


arms = {}
for arm in ("shielded", "unshielded"):
    folder = OUT / arm
    if folder.exists():
        arms[arm] = folder

if not arms:
    raise SystemExit(f"no arm folders under {OUT}")

width = max(len(label) for _, label in METRICS)
print(f"{'metric'.ljust(width)}  " + "  ".join(a.ljust(26) for a in arms))
for key, label in METRICS:
    cells = []
    for arm in arms:
        values = read(arms[arm], key)
        cells.append(fmt(values).ljust(26))
    print(f"{label.ljust(width)}  " + "  ".join(cells))
