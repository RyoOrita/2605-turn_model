from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import colors, cm
from matplotlib import pyplot as plt

from analyze_turn_trend import (
    PLOT_MIN_LEVER,
    build_paths,
    compute_axis_limits,
    find_constant_lever_segments,
    group_samples_by_lever,
    load_yaw_aligned_samples,
)
from output_paths import COMPARE_ROLL_PITCH_YAW_3D_FIGURE_DIR, ensure_parent_dir


DEFAULT_PREFIXES = ("lever_01-05", "lever_05-10")
DEFAULT_OUTPUT = COMPARE_ROLL_PITCH_YAW_3D_FIGURE_DIR / "lever_01-05_vs_05-10_roll_pitch_yaw_3d.png"
LEVER_LEVELS = [round(value, 1) for value in np.arange(0.1, 1.01, 0.1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a single comparison figure for the 3D roll-pitch-yaw plots.")
    parser.add_argument("prefixes", nargs="*", default=list(DEFAULT_PREFIXES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def load_grouped_samples(prefix: str) -> dict[float, np.ndarray]:
    paths = build_paths(prefix)
    _, samples = load_yaw_aligned_samples(paths["input_csv"])
    filtered = samples[samples[:, 0] >= PLOT_MIN_LEVER]
    segments = find_constant_lever_segments(filtered)
    return dict(group_samples_by_lever(filtered, segments))


def main() -> None:
    args = parse_args()
    output_path = ensure_parent_dir(Path(args.output))
    grouped_by_prefix = {prefix: load_grouped_samples(prefix) for prefix in args.prefixes}

    available_groups = [group for grouped in grouped_by_prefix.values() for group in grouped.values()]
    if not available_groups:
        raise RuntimeError("No constant-lever samples were found for comparison.")

    z_limits = compute_axis_limits(available_groups, 3)
    color_min = float(min(group[:, 4].min() for group in available_groups))
    color_max = float(max(group[:, 4].max() for group in available_groups))
    color_norm = colors.Normalize(vmin=color_min, vmax=color_max)

    figure = plt.figure(figsize=(6.6 * len(args.prefixes), 3.3 * len(LEVER_LEVELS)), constrained_layout=True)

    for row_index, lever_value in enumerate(LEVER_LEVELS):
        for column_index, prefix in enumerate(args.prefixes):
            axis = figure.add_subplot(len(LEVER_LEVELS), len(args.prefixes), row_index * len(args.prefixes) + column_index + 1, projection="3d")
            segment = grouped_by_prefix[prefix].get(lever_value)
            axis.view_init(elev=22, azim=-128)
            axis.set_zlim(*z_limits)
            axis.set_title(f"{prefix} lever={lever_value:.1f}")
            axis.set_xlabel("roll angle [deg]")
            axis.set_ylabel("pitch angle [deg]")
            axis.set_zlabel("yaw rate")

            if segment is None:
                axis.text2D(0.34, 0.5, "no data", transform=axis.transAxes)
                continue

            roll = segment[:, 1]
            pitch = segment[:, 2]
            yaw = segment[:, 3]
            abs_yaw = segment[:, 4]

            if segment.shape[0] > 1200:
                sample_index = np.linspace(0, segment.shape[0] - 1, 1200, dtype=int)
                roll_plot = roll[sample_index]
                pitch_plot = pitch[sample_index]
                yaw_plot = yaw[sample_index]
                color_plot = abs_yaw[sample_index]
            else:
                roll_plot = roll
                pitch_plot = pitch
                yaw_plot = yaw
                color_plot = abs_yaw

            axis.scatter(
                roll_plot,
                pitch_plot,
                yaw_plot,
                c=color_plot,
                cmap="viridis",
                norm=color_norm,
                s=8,
                alpha=0.55,
            )

            design = np.column_stack([np.ones(segment.shape[0]), roll, pitch])
            beta, _, _, _ = np.linalg.lstsq(design, yaw, rcond=None)
            roll_grid = np.linspace(float(roll.min()), float(roll.max()), 18)
            pitch_grid = np.linspace(float(pitch.min()), float(pitch.max()), 18)
            roll_mesh, pitch_mesh = np.meshgrid(roll_grid, pitch_grid)
            yaw_mesh = beta[0] + beta[1] * roll_mesh + beta[2] * pitch_mesh
            axis.plot_surface(roll_mesh, pitch_mesh, yaw_mesh, color="#f97316", alpha=0.2, linewidth=0)

    scalar_mappable = cm.ScalarMappable(norm=color_norm, cmap="viridis")
    scalar_mappable.set_array([])
    figure.colorbar(scalar_mappable, ax=figure.axes, shrink=0.35, pad=0.02, label="abs yaw rate")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()