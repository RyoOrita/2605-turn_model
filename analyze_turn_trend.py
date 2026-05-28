from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from output_paths import ANALYZE_TURN_TREND_OUTPUT_DIR, EXTRACT_TARGET_TIMESERIES_OUTPUT_DIR, ensure_output_dir


MIN_SEGMENT_SAMPLES = 30
LEVER_TOLERANCE = 1e-9
STEADY_STATE_TAIL_SAMPLES = 50
DECAY_ANALYSIS_WINDOW_SEC = 6.0
PLOT_MIN_LEVER = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze constant-lever turn trends from extracted CSV data.")
    parser.add_argument("prefix", nargs="?", default="lever_05-10")
    return parser.parse_args()


def build_paths(prefix: str) -> dict[str, Path]:
    return {
        "input_csv": EXTRACT_TARGET_TIMESERIES_OUTPUT_DIR / f"{prefix}_target_timeseries_wide.csv",
        "aligned_csv": ANALYZE_TURN_TREND_OUTPUT_DIR / f"{prefix}_yaw_aligned_samples.csv",
        "summary_csv": ANALYZE_TURN_TREND_OUTPUT_DIR / f"{prefix}_turn_trend_summary.csv",
        "segment_summary_csv": ANALYZE_TURN_TREND_OUTPUT_DIR / f"{prefix}_constant_lever_segments.csv",
        "relationship_figure": ANALYZE_TURN_TREND_OUTPUT_DIR / f"{prefix}_constant_lever_relationships.png",
        "summary_figure": ANALYZE_TURN_TREND_OUTPUT_DIR / f"{prefix}_constant_lever_summary.png",
        "surface_figure": ANALYZE_TURN_TREND_OUTPUT_DIR / f"{prefix}_roll_pitch_yaw_3d.png",
        "decay_summary_csv": ANALYZE_TURN_TREND_OUTPUT_DIR / f"{prefix}_zero_drop_decay_summary.csv",
        "decay_curves_figure": ANALYZE_TURN_TREND_OUTPUT_DIR / f"{prefix}_zero_drop_decay_curves.png",
        "decay_metrics_figure": ANALYZE_TURN_TREND_OUTPUT_DIR / f"{prefix}_zero_drop_decay_metrics.png",
    }


def parse_timestamp_ns(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return int(float(value))


def load_yaw_aligned_samples(input_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    lever = None
    roll = None
    pitch = None
    timestamps: list[int] = []
    samples: list[tuple[float, float, float, float, float]] = []

    with input_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row["lever"]:
                lever = float(row["lever"])
            if row["roll_angle"]:
                roll = float(row["roll_angle"])
            if row["pitch_angle"]:
                pitch = float(row["pitch_angle"])

            if not row["yaw_rate"]:
                continue

            yaw_rate = float(row["yaw_rate"])
            if lever is None or roll is None or pitch is None:
                continue

            timestamps.append(parse_timestamp_ns(row["timestamp_ns"]))
            samples.append((lever, roll, pitch, yaw_rate, abs(yaw_rate)))

    return np.array(timestamps, dtype=np.int64), np.array(samples, dtype=float)


def corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def group_samples_by_lever(
    samples: np.ndarray,
    segments: list[tuple[int, int]],
) -> list[tuple[float, np.ndarray]]:
    grouped_segments: dict[float, list[np.ndarray]] = {}
    for start, end in segments:
        segment = samples[start:end]
        lever_value = round(float(segment[0, 0]), 1)
        grouped_segments.setdefault(lever_value, []).append(segment)

    return [(lever_value, np.vstack(segment_group)) for lever_value, segment_group in sorted(grouped_segments.items())]


def compute_axis_limits(grouped_samples: list[np.ndarray], axis_index: int) -> tuple[float, float]:
    lower = float(min(group[:, axis_index].min() for group in grouped_samples))
    upper = float(max(group[:, axis_index].max() for group in grouped_samples))
    if np.isclose(lower, upper):
        margin = 0.1 if np.isclose(lower, 0.0) else abs(lower) * 0.1
    else:
        margin = (upper - lower) * 0.05
    return lower - margin, upper + margin


def write_aligned_csv(output_path: Path, timestamps: np.ndarray, samples: np.ndarray) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_ns", "lever", "roll_angle", "pitch_angle", "yaw_rate", "abs_yaw_rate"])
        for timestamp_ns, sample in zip(timestamps, samples, strict=True):
            writer.writerow([int(timestamp_ns), *sample.tolist()])


def write_summary_csv(output_path: Path, filtered: np.ndarray) -> None:
    lever = filtered[:, 0]
    roll = filtered[:, 1]
    pitch = filtered[:, 2]
    abs_yaw = filtered[:, 4]

    rows: list[list[object]] = [["metric", "value"]]
    rows.extend(
        [
            ["filtered_samples", filtered.shape[0]],
            ["corr_abs_yaw_vs_lever", corrcoef(lever, abs_yaw)],
            ["corr_abs_yaw_vs_roll", corrcoef(roll, abs_yaw)],
            ["corr_abs_yaw_vs_pitch", corrcoef(pitch, abs_yaw)],
        ]
    )

    design = np.column_stack([np.ones(filtered.shape[0]), lever, roll, pitch])
    beta, _, _, _ = np.linalg.lstsq(design, abs_yaw, rcond=None)
    predicted = design @ beta
    residual_sum = float(np.square(abs_yaw - predicted).sum())
    total_sum = float(np.square(abs_yaw - abs_yaw.mean()).sum())
    rows.extend(
        [
            ["ols_abs_yaw_intercept", beta[0]],
            ["ols_abs_yaw_lever_coef", beta[1]],
            ["ols_abs_yaw_roll_coef", beta[2]],
            ["ols_abs_yaw_pitch_coef", beta[3]],
            ["ols_abs_yaw_r2", 1.0 - residual_sum / total_sum],
        ]
    )

    design_lever = np.column_stack([np.ones(filtered.shape[0]), lever])
    lever_beta, _, _, _ = np.linalg.lstsq(design_lever, abs_yaw, rcond=None)
    lever_residual = abs_yaw - design_lever @ lever_beta
    rows.extend(
        [
            ["residual_corr_roll_after_lever", corrcoef(roll, lever_residual)],
            ["residual_corr_pitch_after_lever", corrcoef(pitch, lever_residual)],
        ]
    )

    lever_bins = [(lower, min(lower + 0.1, 1.01)) for lower in np.arange(PLOT_MIN_LEVER, 1.0, 0.1)]
    rows.append([])
    rows.append(["lever_bin", "sample_count", "mean_abs_yaw", "corr_roll", "corr_pitch"])
    for lower, upper in lever_bins:
        mask = (lever >= lower) & (lever < upper)
        rows.append(
            [
                f"[{lower},{upper})",
                int(mask.sum()),
                safe_mean(abs_yaw[mask]),
                corrcoef(roll[mask], abs_yaw[mask]),
                corrcoef(pitch[mask], abs_yaw[mask]),
            ]
        )

    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows(rows)


def find_constant_lever_segments(samples: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = 0
    sample_count = samples.shape[0]

    for index in range(1, sample_count + 1):
        is_break = index == sample_count
        if not is_break:
            is_break = abs(samples[index, 0] - samples[start, 0]) > LEVER_TOLERANCE

        if not is_break:
            continue

        if index - start >= MIN_SEGMENT_SAMPLES:
            segments.append((start, index))
        start = index

    return segments


def write_segment_summary_csv(
    output_path: Path,
    timestamps: np.ndarray,
    samples: np.ndarray,
    segments: list[tuple[int, int]],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "segment_id",
                "start_timestamp_ns",
                "end_timestamp_ns",
                "duration_sec",
                "lever",
                "sample_count",
                "mean_roll_angle",
                "mean_pitch_angle",
                "mean_abs_yaw_rate",
                "corr_roll_vs_abs_yaw",
                "corr_pitch_vs_abs_yaw",
                "ols_abs_yaw_intercept",
                "ols_abs_yaw_roll_coef",
                "ols_abs_yaw_pitch_coef",
                "ols_abs_yaw_r2",
            ]
        )

        for segment_id, (start, end) in enumerate(segments, start=1):
            segment = samples[start:end]
            segment_timestamps = timestamps[start:end]
            roll = segment[:, 1]
            pitch = segment[:, 2]
            abs_yaw = segment[:, 4]
            design = np.column_stack([np.ones(segment.shape[0]), roll, pitch])
            beta, _, _, _ = np.linalg.lstsq(design, abs_yaw, rcond=None)
            predicted = design @ beta
            residual_sum = float(np.square(abs_yaw - predicted).sum())
            total_sum = float(np.square(abs_yaw - abs_yaw.mean()).sum())
            r2 = float("nan") if total_sum == 0.0 else 1.0 - residual_sum / total_sum

            writer.writerow(
                [
                    segment_id,
                    int(segment_timestamps[0]),
                    int(segment_timestamps[-1]),
                    (int(segment_timestamps[-1]) - int(segment_timestamps[0])) / 1e9,
                    segment[0, 0],
                    segment.shape[0],
                    float(roll.mean()),
                    float(pitch.mean()),
                    float(abs_yaw.mean()),
                    corrcoef(roll, abs_yaw),
                    corrcoef(pitch, abs_yaw),
                    beta[0],
                    beta[1],
                    beta[2],
                    r2,
                ]
            )


def plot_regression_line(axis: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(float(x.min()), float(x.max()), 100)
    axis.plot(x_line, slope * x_line + intercept, color=color, linewidth=2)


def plot_segment_relationships(output_path: Path, samples: np.ndarray, segments: list[tuple[int, int]]) -> None:
    figure, axes = plt.subplots(len(segments), 2, figsize=(12, 3.4 * len(segments)), constrained_layout=True)
    if len(segments) == 1:
        axes = np.array([axes])

    for row_index, (start, end) in enumerate(segments):
        segment = samples[start:end]
        lever_value = segment[0, 0]
        roll = segment[:, 1]
        pitch = segment[:, 2]
        abs_yaw = segment[:, 4]

        roll_axis = axes[row_index, 0]
        pitch_axis = axes[row_index, 1]

        roll_axis.scatter(roll, abs_yaw, s=8, alpha=0.35, color="#1f77b4")
        plot_regression_line(roll_axis, roll, abs_yaw, "#0d3b66")
        roll_axis.set_title(f"lever={lever_value:.1f} roll vs abs yaw")
        roll_axis.set_xlabel("roll angle [deg]")
        roll_axis.set_ylabel("abs yaw rate")
        roll_axis.grid(alpha=0.2)

        pitch_axis.scatter(pitch, abs_yaw, s=8, alpha=0.35, color="#ff7f0e")
        plot_regression_line(pitch_axis, pitch, abs_yaw, "#9a3412")
        pitch_axis.set_title(f"lever={lever_value:.1f} pitch vs abs yaw")
        pitch_axis.set_xlabel("pitch angle [deg]")
        pitch_axis.set_ylabel("abs yaw rate")
        pitch_axis.grid(alpha=0.2)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_segment_summary(output_path: Path, samples: np.ndarray, segments: list[tuple[int, int]]) -> None:
    lever_values: list[float] = []
    mean_abs_yaw_values: list[float] = []
    corr_roll_values: list[float] = []
    corr_pitch_values: list[float] = []

    for start, end in segments:
        segment = samples[start:end]
        lever_values.append(float(segment[0, 0]))
        mean_abs_yaw_values.append(float(segment[:, 4].mean()))
        corr_roll_values.append(corrcoef(segment[:, 1], segment[:, 4]))
        corr_pitch_values.append(corrcoef(segment[:, 2], segment[:, 4]))

    x = np.arange(len(segments))
    labels = [f"{lever_value:.1f}" for lever_value in lever_values]
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].bar(x, mean_abs_yaw_values, color="#2563eb")
    axes[0].set_title("Mean abs yaw rate by constant lever segment")
    axes[0].set_ylabel("mean abs yaw rate")
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("lever")
    axes[0].grid(axis="y", alpha=0.2)

    width = 0.35
    axes[1].bar(x - width / 2, corr_roll_values, width=width, label="roll", color="#059669")
    axes[1].bar(x + width / 2, corr_pitch_values, width=width, label="pitch", color="#dc2626")
    axes[1].set_title("Correlation with abs yaw rate by constant lever segment")
    axes[1].set_ylabel("correlation")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("lever")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_roll_pitch_yaw_3d(output_path: Path, samples: np.ndarray, segments: list[tuple[int, int]]) -> None:
    lever_groups = group_samples_by_lever(samples, segments)
    figure = plt.figure(figsize=(14, 3.8 * len(lever_groups)), constrained_layout=True)
    grouped_samples = [np.vstack(segment_group) for _, segment_group in lever_groups]
    z_limits = compute_axis_limits(grouped_samples, 3)

    for plot_index, ((lever_value, _), segment) in enumerate(zip(lever_groups, grouped_samples, strict=True), start=1):
        roll = segment[:, 1]
        pitch = segment[:, 2]
        yaw = segment[:, 3]
        abs_yaw = segment[:, 4]

        if segment.shape[0] > 1500:
            sample_index = np.linspace(0, segment.shape[0] - 1, 1500, dtype=int)
            roll_plot = roll[sample_index]
            pitch_plot = pitch[sample_index]
            yaw_plot = yaw[sample_index]
            color_plot = abs_yaw[sample_index]
        else:
            roll_plot = roll
            pitch_plot = pitch
            yaw_plot = yaw
            color_plot = abs_yaw

        axis = figure.add_subplot(len(lever_groups), 1, plot_index, projection="3d")
        scatter = axis.scatter(
            roll_plot,
            pitch_plot,
            yaw_plot,
            c=color_plot,
            cmap="viridis",
            s=8,
            alpha=0.55,
        )

        if segment.shape[0] >= 3:
            design = np.column_stack([np.ones(segment.shape[0]), roll, pitch])
            beta, _, _, _ = np.linalg.lstsq(design, yaw, rcond=None)
            roll_grid = np.linspace(float(roll.min()), float(roll.max()), 18)
            pitch_grid = np.linspace(float(pitch.min()), float(pitch.max()), 18)
            roll_mesh, pitch_mesh = np.meshgrid(roll_grid, pitch_grid)
            yaw_mesh = beta[0] + beta[1] * roll_mesh + beta[2] * pitch_mesh
            axis.plot_surface(roll_mesh, pitch_mesh, yaw_mesh, color="#f97316", alpha=0.25, linewidth=0)

        axis.set_title(f"lever={lever_value:.1f} roll-pitch-yaw")
        axis.set_xlabel("roll angle [deg]")
        axis.set_ylabel("pitch angle [deg]")
        axis.set_zlabel("yaw rate")
        axis.set_zlim(*z_limits)
        axis.view_init(elev=22, azim=-128)
        figure.colorbar(scatter, ax=axis, shrink=0.72, pad=0.08, label="abs yaw rate")

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def safe_threshold_time(times_sec: np.ndarray, values: np.ndarray, threshold: float) -> float:
    hits = np.where(values <= threshold)[0]
    if hits.size == 0:
        return float("nan")
    return float(times_sec[hits[0]])


def find_drop_to_zero_events(samples: np.ndarray, segments: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    events: list[tuple[int, int, int, int]] = []
    for index in range(len(segments) - 1):
        pre_start, pre_end = segments[index]
        post_start, post_end = segments[index + 1]
        pre_level = samples[pre_start, 0]
        post_level = samples[post_start, 0]
        if pre_level >= 0.5 and abs(post_level) <= LEVER_TOLERANCE:
            events.append((pre_start, pre_end, post_start, post_end))
    return events


def write_decay_summary_csv(
    output_path: Path,
    timestamps: np.ndarray,
    samples: np.ndarray,
    events: list[tuple[int, int, int, int]],
) -> list[dict[str, float | int]]:
    metric_rows: list[dict[str, float | int]] = []
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "event_id",
                "lever_before_drop",
                "drop_timestamp_ns",
                "pre_duration_sec",
                "post_duration_sec",
                "steady_state_abs_yaw",
                "first_post_abs_yaw",
                "last_post_abs_yaw",
                "time_to_50pct_sec",
                "time_to_10pct_sec",
                "time_to_5pct_sec",
                "mean_roll_post",
                "mean_pitch_post",
            ]
        )

        for event_id, (pre_start, pre_end, post_start, post_end) in enumerate(events, start=1):
            pre_segment = samples[pre_start:pre_end]
            post_segment = samples[post_start:post_end]
            post_times_sec = (timestamps[post_start:post_end] - timestamps[post_start]) / 1e9
            tail_count = min(STEADY_STATE_TAIL_SAMPLES, pre_segment.shape[0])
            steady_state_abs_yaw = float(np.median(pre_segment[-tail_count:, 4]))
            post_abs_yaw = post_segment[:, 4]
            row = {
                "event_id": event_id,
                "lever_before_drop": float(pre_segment[0, 0]),
                "drop_timestamp_ns": int(timestamps[post_start]),
                "pre_duration_sec": (int(timestamps[pre_end - 1]) - int(timestamps[pre_start])) / 1e9,
                "post_duration_sec": (int(timestamps[post_end - 1]) - int(timestamps[post_start])) / 1e9,
                "steady_state_abs_yaw": steady_state_abs_yaw,
                "first_post_abs_yaw": float(post_abs_yaw[0]),
                "last_post_abs_yaw": float(post_abs_yaw[-1]),
                "time_to_50pct_sec": safe_threshold_time(post_times_sec, post_abs_yaw, 0.5 * steady_state_abs_yaw),
                "time_to_10pct_sec": safe_threshold_time(post_times_sec, post_abs_yaw, 0.1 * steady_state_abs_yaw),
                "time_to_5pct_sec": safe_threshold_time(post_times_sec, post_abs_yaw, 0.05 * steady_state_abs_yaw),
                "mean_roll_post": float(post_segment[:, 1].mean()),
                "mean_pitch_post": float(post_segment[:, 2].mean()),
            }
            metric_rows.append(row)

            writer.writerow(
                [
                    row["event_id"],
                    row["lever_before_drop"],
                    row["drop_timestamp_ns"],
                    row["pre_duration_sec"],
                    row["post_duration_sec"],
                    row["steady_state_abs_yaw"],
                    row["first_post_abs_yaw"],
                    row["last_post_abs_yaw"],
                    row["time_to_50pct_sec"],
                    row["time_to_10pct_sec"],
                    row["time_to_5pct_sec"],
                    row["mean_roll_post"],
                    row["mean_pitch_post"],
                ]
            )

    return metric_rows


def plot_decay_curves(
    output_path: Path,
    timestamps: np.ndarray,
    samples: np.ndarray,
    events: list[tuple[int, int, int, int]],
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    has_curve = False

    for pre_start, pre_end, post_start, post_end in events:
        pre_segment = samples[pre_start:pre_end]
        post_segment = samples[post_start:post_end]
        post_times_sec = (timestamps[post_start:post_end] - timestamps[post_start]) / 1e9
        window_mask = post_times_sec <= DECAY_ANALYSIS_WINDOW_SEC
        if not np.any(window_mask):
            continue

        tail_count = min(STEADY_STATE_TAIL_SAMPLES, pre_segment.shape[0])
        steady_state_abs_yaw = float(np.median(pre_segment[-tail_count:, 4]))
        clipped_times = post_times_sec[window_mask]
        clipped_abs_yaw = post_segment[window_mask, 4]
        label = f"lever {pre_segment[0, 0]:.1f}"

        axes[0].plot(clipped_times, clipped_abs_yaw, linewidth=2, label=label)
        if steady_state_abs_yaw > 0.0:
            axes[1].plot(clipped_times, clipped_abs_yaw / steady_state_abs_yaw, linewidth=2, label=label)
        has_curve = True

    axes[0].set_title("Abs yaw-rate decay after lever drops to zero")
    axes[0].set_xlabel("time after drop [s]")
    axes[0].set_ylabel("abs yaw rate")
    axes[0].grid(alpha=0.2)
    if has_curve:
        axes[0].legend()

    axes[1].set_title("Normalized decay after lever drops to zero")
    axes[1].set_xlabel("time after drop [s]")
    axes[1].set_ylabel("abs yaw rate / steady-state")
    axes[1].axhline(0.5, color="#6b7280", linewidth=1, linestyle="--")
    axes[1].axhline(0.1, color="#9ca3af", linewidth=1, linestyle=":")
    axes[1].grid(alpha=0.2)
    if has_curve:
        axes[1].legend()

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_decay_metrics(output_path: Path, rows: list[dict[str, float | int]]) -> None:
    lever = np.array([float(row["lever_before_drop"]) for row in rows])
    time_to_50 = np.array([float(row["time_to_50pct_sec"]) for row in rows])
    time_to_10 = np.array([float(row["time_to_10pct_sec"]) for row in rows])
    steady_state = np.array([float(row["steady_state_abs_yaw"]) for row in rows])

    x = np.arange(lever.size)
    labels = [f"{value:.1f}" for value in lever]
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].bar(x, steady_state, color="#2563eb")
    axes[0].set_title("Steady-state abs yaw before lever drops to zero")
    axes[0].set_ylabel("abs yaw rate")
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("lever before drop")
    axes[0].grid(axis="y", alpha=0.2)

    width = 0.35
    axes[1].bar(x - width / 2, time_to_50, width=width, label="50%", color="#059669")
    axes[1].bar(x + width / 2, time_to_10, width=width, label="10%", color="#dc2626")
    axes[1].set_title("Decay time after lever drops to zero")
    axes[1].set_ylabel("time [s]")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("lever before drop")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ensure_output_dir(ANALYZE_TURN_TREND_OUTPUT_DIR)
    paths = build_paths(args.prefix)

    timestamps, samples = load_yaw_aligned_samples(paths["input_csv"])
    write_aligned_csv(paths["aligned_csv"], timestamps, samples)
    all_segments = find_constant_lever_segments(samples)
    filtered_mask = samples[:, 0] >= PLOT_MIN_LEVER
    filtered = samples[filtered_mask]
    filtered_timestamps = timestamps[filtered_mask]
    write_summary_csv(paths["summary_csv"], filtered)
    segments = find_constant_lever_segments(filtered)
    write_segment_summary_csv(paths["segment_summary_csv"], filtered_timestamps, filtered, segments)
    plot_segment_relationships(paths["relationship_figure"], filtered, segments)
    plot_segment_summary(paths["summary_figure"], filtered, segments)
    plot_roll_pitch_yaw_3d(paths["surface_figure"], filtered, segments)
    decay_events = find_drop_to_zero_events(samples, all_segments)
    decay_metric_rows = write_decay_summary_csv(paths["decay_summary_csv"], timestamps, samples, decay_events)
    plot_decay_curves(paths["decay_curves_figure"], timestamps, samples, decay_events)
    plot_decay_metrics(paths["decay_metrics_figure"], decay_metric_rows)


if __name__ == "__main__":
    main()