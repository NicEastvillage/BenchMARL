"""Plot a hydra multirun sweep: shielded vs unshielded, one figure per metric.

Point it at a sweep folder produced by `python benchmarl/run.py -m ...`, e.g.

    python plot_multirun.py multirun/2026-10-22/11-33-43

Every run under the sweep is discovered, its algorithm (parsed from the suffix of the
experiment folder name), its `do_shielding` flag and its seed (parsed from
`.hydra/overrides.yaml`) are read, and the csv scalars are turned into curves.

Runs are grouped by `(algorithm, shielded)`; multiple seeds are aggregated into a mean
line. One PNG per metric is written to `<sweep>/plots/`.

Band: a metric ending in `/mean` is drawn with a band from its `.../min` and `.../max`
sibling scalars (the per-episode return spread), taking the lowest min and highest max
across seeds. Metrics without such siblings fall back to a min/max band across seeds.
"""
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (csv key, plot title, y axis label). The csv key is the metric name with '/' -> '_',
# matching the file name under the run's `scalars/` folder.
METRICS: List[Tuple[str, str, str]] = [
    ("safety/violation_rate", "Safety violation rate", "fraction of agent-steps"),
    ("safety/legal_action_rate", "Legal action rate", "fraction of actions"),
    ("safety/shield_no_safe_action_rate", "Shield no-safe-action rate", "fraction of queries"),
    ("collection/agents/reward/episode_reward/mean", "Episode reward (sampled policy)", "reward"),
    ("eval/reward/episode_reward/mean", "Episode reward (deterministic policy)", "reward"),
    ("env/average_velocity", "Average velocity", "cells / step"),
    ("env/min_gap", "Minimum gap between consecutive cars", "cells"),
    ("env/max_gap", "Maximum gap between consecutive cars", "cells"),
    ("env/average_distance_travelled", "Average distance travelled", "cells per episode"),
]

# Columns used as the x axis (in preference order). These are logged per iteration.
X_CANDIDATES = ["counters/total_frames", "counters/iter"]

# Fallback algorithm names, so a folder name that does not start with a known algorithm
# still yields something readable.
KNOWN_ALGORITHMS = [
    "mappo", "ippo", "iql", "qmix", "vdn", "isac", "masac", "iddpg", "maddpg", "ensemble",
]


def series_key(metric: str) -> str:
    return metric.replace("/", "_")


def find_runs(sweep: Path) -> List[Path]:
    """Return the experiment folders (the ones containing a `scalars/` dir) under a sweep."""
    return sorted({scalars.parent for scalars in sweep.rglob("scalars") if scalars.is_dir()})


def parse_algorithm(run: Path) -> str:
    """Algorithm is the leading token of the experiment folder name, e.g. mappo_roundabout_mlp__.."""
    base = run.name
    # Nested logging writes `run/<same name>`, so strip a duplicated trailing component.
    if run.parent.name == base:
        base = run.parent.name
    for algo in KNOWN_ALGORITHMS:
        if base.startswith(algo + "_"):
            return algo
    return base.split("_", 1)[0]


def parse_overrides(run: Path) -> Dict[str, str]:
    """Read the run's `.hydra/overrides.yaml` into a {key: lowercased value} dict."""
    for candidate in (run / ".hydra" / "overrides.yaml", *(p / ".hydra" / "overrides.yaml" for p in run.parents)):
        if candidate.exists():
            overrides = {}
            for line in candidate.read_text().splitlines():
                line = line.strip().lstrip("-").strip()
                if "=" in line:
                    key, _, value = line.partition("=")
                    overrides[key.strip()] = value.strip().lower()
            return overrides
    return {}


def parse_shielding(overrides: Dict[str, str]) -> Optional[bool]:
    value = overrides.get("task.do_shielding") or overrides.get("task.shielding")
    if value is None:
        return None
    return value in ("true", "1", "yes")


def parse_seed(overrides: Dict[str, str]) -> str:
    seed = overrides.get("seed")
    return seed if seed is not None else "?"


def safe_label(algorithm: str, shielded: Optional[bool]) -> str:
    if shielded is None:
        return algorithm
    return f"{algorithm} ({'shielded' if shielded else 'unshielded'})"


def load_metric(run: Path, metric: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (x, y) for one metric, or None if the csv is absent."""
    csv_path = run / "scalars" / f"{series_key(metric)}.csv"
    if not csv_path.exists():
        return None

    indices: List[int] = []
    values: List[float] = []
    for line in csv_path.read_text().strip().splitlines():
        if not line:
            continue
        index, _, value = line.partition(",")
        indices.append(int(index))
        values.append(float(value))

    # Metrics are logged on a subset of iterations (e.g. eval runs every other iteration),
    # so map each row's own index through the x axis instead of assuming a full grid.
    x_index = load_x_index(run)
    x = np.array([x_index.get(i, float(i)) for i in indices], dtype=float)
    return x, np.asarray(values, dtype=float)


def load_x_index(run: Path) -> Dict[int, float]:
    """Map row index -> x value (frames preferred, then iter). Empty if neither exists."""
    for key in X_CANDIDATES:
        csv_path = run / "scalars" / f"{series_key(key)}.csv"
        if not csv_path.exists():
            continue
        index: Dict[int, float] = {}
        for line in csv_path.read_text().strip().splitlines():
            if not line:
                continue
            row, _, value = line.partition(",")
            index[int(row)] = float(value)
        return index
    return {}


def stack_curves(
    curves: List[Tuple[np.ndarray, np.ndarray]], grid_x: np.ndarray
) -> np.ndarray:
    """Reindex curves onto `grid_x`, returning a (len(curves), len(grid_x)) array with NaNs."""
    grid = np.full((len(curves), len(grid_x)), np.nan)
    for row, (x, y) in enumerate(curves):
        grid[row, np.searchsorted(grid_x, x)] = y
    return grid


def mean_band_metrics(metric: str) -> Optional[Tuple[str, str]]:
    """For a `/mean` metric, the `.../min` and `.../max` sibling keys; else None."""
    if not metric.endswith("/mean"):
        return None
    base = metric[: -len("mean")]
    return base + "min", base + "max"


def plot_metric(
    groups: Dict[str, List[Path]],
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> bool:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    siblings = mean_band_metrics(metric)

    for label in sorted(groups):
        runs = groups[label]
        triples = []  # (mean curve, min curve, max curve) per run, min/max may be None
        complete_band = siblings is not None

        for run in runs:
            mean_curve = load_metric(run, metric)
            if mean_curve is None:
                continue
            if siblings is None:
                triples.append((mean_curve, None, None))
                continue
            min_key, max_key = siblings
            min_curve = load_metric(run, min_key)
            max_curve = load_metric(run, max_key)
            if min_curve is None or max_curve is None:
                complete_band = False
            triples.append((mean_curve, min_curve, max_curve))

        if not triples:
            continue

        mean_curves = [t[0] for t in triples]
        grid_x = np.unique(np.concatenate([x for x, _ in mean_curves]))
        mean_grid = stack_curves(mean_curves, grid_x)

        band = None
        if complete_band:
            min_grid = stack_curves([t[1] for t in triples], grid_x)
            max_grid = stack_curves([t[2] for t in triples], grid_x)
            band = (np.nanmin(min_grid, axis=0), np.nanmax(max_grid, axis=0))

        if band is None:
            # No sibling min/max scalars: fall back to the spread across seeds.
            band = (np.nanmin(mean_grid, axis=0), np.nanmax(mean_grid, axis=0))

        (line,) = ax.plot(grid_x, np.nanmean(mean_grid, axis=0), label=label, linewidth=2)
        ax.fill_between(grid_x, band[0], band[1], color=line.get_color(), alpha=0.2)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_title(title)
    ax.set_xlabel("collected frames")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_sweep(sweep: Path) -> None:
    runs = find_runs(sweep)
    if not runs:
        raise SystemExit(f"no runs found under {sweep}")

    groups: Dict[str, List[Path]] = defaultdict(list)
    for run in runs:
        overrides = parse_overrides(run)
        label = safe_label(parse_algorithm(run), parse_shielding(overrides))
        groups[label].append(run)
        print(f"  {run.relative_to(sweep)}  ->  {label} (seed {parse_seed(overrides)})")

    out_dir = sweep / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for metric, title, ylabel in METRICS:
        out_path = out_dir / f"{series_key(metric)}.png"
        if plot_metric(groups, metric, title, ylabel, out_path):
            written.append(out_path.name)

    print(f"\nwrote {len(written)} plot(s) to {out_dir}")
    for name in written:
        print(f"  {name}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    for argument in sys.argv[1:]:
        sweep = Path(argument)
        if not sweep.exists():
            raise SystemExit(f"sweep folder does not exist: {sweep}")
        print(f"===== {sweep} =====")
        plot_sweep(sweep)
        print()


if __name__ == "__main__":
    main()
