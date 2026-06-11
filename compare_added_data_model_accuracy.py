from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from build_generalized_turn_model import (
    DATASET_CONFIG_CSV,
    build_model_rows,
    build_prediction_rows,
    build_training_rows,
    compute_validation_metrics,
    load_dataset_config,
)
from output_paths import (
    BUILD_GENERALIZED_TURN_MODEL_ADDED_DATA_COMPARISON_CSV_DIR,
    BUILD_GENERALIZED_TURN_MODEL_ADDED_DATA_COMPARISON_FIGURE_DIR,
    ensure_parent_dir,
)


BASELINE_PREFIXES = ("lever_01-05", "lever_05-10")
METRICS_CSV = BUILD_GENERALIZED_TURN_MODEL_ADDED_DATA_COMPARISON_CSV_DIR / "turn_model_added_data_accuracy_comparison.csv"
METRIC_FIGURE = BUILD_GENERALIZED_TURN_MODEL_ADDED_DATA_COMPARISON_FIGURE_DIR / "turn_model_added_data_metric_comparison.png"
PREDICTION_FIGURE = BUILD_GENERALIZED_TURN_MODEL_ADDED_DATA_COMPARISON_FIGURE_DIR / "turn_model_added_data_prediction_comparison.png"
MAX_SCATTER_POINTS = 60000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare turn model accuracy before and after adding extra logs.")
    parser.add_argument("--config", default=str(DATASET_CONFIG_CSV))
    parser.add_argument(
        "--baseline-prefix",
        action="append",
        dest="baseline_prefixes",
        help="Prefix used in the original baseline model. Can be specified multiple times.",
    )
    return parser.parse_args()


def filter_config_rows(config_rows: list[dict[str, str]], prefixes: tuple[str, ...]) -> list[dict[str, str]]:
    prefix_set = set(prefixes)
    selected_rows = [row for row in config_rows if row["prefix"] in prefix_set]
    found_prefixes = {row["prefix"] for row in selected_rows}
    missing_prefixes = sorted(prefix_set - found_prefixes)
    if missing_prefixes:
        raise ValueError(f"baseline prefix not found in config: {missing_prefixes}")
    return selected_rows


def build_predictions_for_training_set(
    training_config_rows: list[dict[str, str]],
    validation_training_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    model_training_rows = build_training_rows(training_config_rows)
    model_rows = build_model_rows(model_training_rows)
    return build_prediction_rows(validation_training_rows, model_rows)


def filter_prediction_rows_by_prefix(
    prediction_rows: list[dict[str, object]],
    prefixes: tuple[str, ...],
) -> list[dict[str, object]]:
    prefix_set = set(prefixes)
    return [row for row in prediction_rows if str(row["prefix"]) in prefix_set]


def build_metric_rows(
    variant_predictions: dict[str, list[dict[str, object]]],
    variant_training_prefixes: dict[str, tuple[str, ...]],
    baseline_prefixes: tuple[str, ...],
    additional_prefixes: tuple[str, ...],
    all_prefixes: tuple[str, ...],
) -> list[dict[str, object]]:
    scope_specs = [
        ("all_logs", "all logs", all_prefixes),
        ("original_logs", "original 2 logs", baseline_prefixes),
    ]
    if additional_prefixes:
        scope_specs.append(("added_logs", f"added {len(additional_prefixes)} logs", additional_prefixes))

    metric_rows: list[dict[str, object]] = []
    for variant_name, prediction_rows in variant_predictions.items():
        training_prefixes = variant_training_prefixes[variant_name]
        for scope_key, scope_label, scope_prefixes in scope_specs:
            scoped_rows = filter_prediction_rows_by_prefix(prediction_rows, scope_prefixes)
            if not scoped_rows:
                continue
            metrics = compute_validation_metrics(scoped_rows)
            metric_rows.append(
                {
                    "validation_scope": scope_key,
                    "validation_label": scope_label,
                    "model_variant": variant_name,
                    "training_log_count": len(training_prefixes),
                    "validation_log_count": len(scope_prefixes),
                    "training_prefixes": ";".join(training_prefixes),
                    "validation_prefixes": ";".join(scope_prefixes),
                    **metrics,
                }
            )
    return metric_rows


def write_metric_rows(metric_rows: list[dict[str, object]]) -> None:
    ensure_parent_dir(METRICS_CSV)
    with METRICS_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "validation_scope",
                "validation_label",
                "model_variant",
                "training_log_count",
                "validation_log_count",
                "training_prefixes",
                "validation_prefixes",
                "sample_count",
                "mae",
                "rmse",
                "bias",
                "r2",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)


def get_metric(metric_rows: list[dict[str, object]], variant_name: str, scope_key: str, metric_name: str) -> float:
    for row in metric_rows:
        if row["model_variant"] == variant_name and row["validation_scope"] == scope_key:
            return float(row[metric_name])
    return float("nan")


def plot_metric_comparison(metric_rows: list[dict[str, object]]) -> None:
    ensure_parent_dir(METRIC_FIGURE)
    scope_keys = []
    scope_labels = []
    for row in metric_rows:
        scope_key = str(row["validation_scope"])
        if scope_key not in scope_keys:
            scope_keys.append(scope_key)
            scope_labels.append(str(row["validation_label"]))

    variant_names = []
    for row in metric_rows:
        variant_name = str(row["model_variant"])
        if variant_name not in variant_names:
            variant_names.append(variant_name)

    metric_specs = [("rmse", "RMSE lower is better"), ("mae", "MAE lower is better"), ("r2", "R2 higher is better")]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    x_positions = np.arange(len(scope_keys), dtype=float)
    bar_width = 0.35
    colors = ["#64748b", "#0f766e"]

    for axis, (metric_name, title) in zip(axes, metric_specs, strict=True):
        for variant_index, variant_name in enumerate(variant_names):
            values = [get_metric(metric_rows, variant_name, scope_key, metric_name) for scope_key in scope_keys]
            offset = (variant_index - (len(variant_names) - 1) / 2.0) * bar_width
            axis.bar(x_positions + offset, values, width=bar_width, label=variant_name, color=colors[variant_index % len(colors)])

        axis.set_title(title)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(scope_labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if metric_name == "r2":
            axis.set_ylim(0.0, 1.02)

    axes[0].set_ylabel("metric value")
    axes[0].legend(loc="best")
    figure.suptitle("Turn model accuracy before/after adding 9 logs")
    figure.savefig(METRIC_FIGURE, dpi=180)
    plt.close(figure)


def downsample_prediction_rows(prediction_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(prediction_rows) <= MAX_SCATTER_POINTS:
        return prediction_rows
    indices = np.linspace(0, len(prediction_rows) - 1, MAX_SCATTER_POINTS, dtype=int)
    return [prediction_rows[int(index)] for index in indices]


def plot_prediction_comparison(
    variant_predictions: dict[str, list[dict[str, object]]],
    metric_rows: list[dict[str, object]],
    baseline_prefixes: tuple[str, ...],
) -> None:
    ensure_parent_dir(PREDICTION_FIGURE)
    variant_names = list(variant_predictions.keys())
    figure, axes = plt.subplots(2, len(variant_names), figsize=(7.0 * len(variant_names), 9.5), constrained_layout=True)
    axes_array = np.atleast_2d(axes)

    all_actual_values: list[float] = []
    all_predicted_values: list[float] = []
    for prediction_rows in variant_predictions.values():
        sampled_rows = downsample_prediction_rows(prediction_rows)
        all_actual_values.extend(float(row["abs_yaw_rate"]) for row in sampled_rows)
        all_predicted_values.extend(float(row["predicted_abs_yaw_rate"]) for row in sampled_rows)

    axis_min = float(min(all_actual_values + all_predicted_values))
    axis_max = float(max(all_actual_values + all_predicted_values))
    baseline_prefix_set = set(baseline_prefixes)

    for variant_index, variant_name in enumerate(variant_names):
        sampled_rows = downsample_prediction_rows(variant_predictions[variant_name])
        actual = np.array([float(row["abs_yaw_rate"]) for row in sampled_rows])
        predicted = np.array([float(row["predicted_abs_yaw_rate"]) for row in sampled_rows])
        lever = np.array([float(row["lever"]) for row in sampled_rows])
        residual = actual - predicted
        is_baseline_log = np.array([str(row["prefix"]) in baseline_prefix_set for row in sampled_rows], dtype=bool)
        title_rmse = get_metric(metric_rows, variant_name, "all_logs", "rmse")
        title_r2 = get_metric(metric_rows, variant_name, "all_logs", "r2")

        scatter_axis = axes_array[0, variant_index]
        scatter_axis.scatter(actual[is_baseline_log], predicted[is_baseline_log], s=8, alpha=0.22, color="#2563eb", label="original logs")
        scatter_axis.scatter(actual[~is_baseline_log], predicted[~is_baseline_log], s=8, alpha=0.22, color="#f97316", label="added logs")
        scatter_axis.plot([axis_min, axis_max], [axis_min, axis_max], color="#111827", linewidth=1.3, linestyle="--")
        scatter_axis.set_title(f"{variant_name}\nR2={title_r2:.3f}, RMSE={title_rmse:.4f}")
        scatter_axis.set_xlabel("actual abs yaw rate")
        scatter_axis.set_ylabel("predicted abs yaw rate")
        scatter_axis.grid(alpha=0.2)
        scatter_axis.legend(loc="best")

        residual_axis = axes_array[1, variant_index]
        residual_axis.scatter(lever[is_baseline_log], residual[is_baseline_log], s=8, alpha=0.22, color="#2563eb", label="original logs")
        residual_axis.scatter(lever[~is_baseline_log], residual[~is_baseline_log], s=8, alpha=0.22, color="#f97316", label="added logs")
        residual_axis.axhline(0.0, color="#111827", linewidth=1.3, linestyle="--")
        residual_axis.set_title("Residual vs lever")
        residual_axis.set_xlabel("lever")
        residual_axis.set_ylabel("actual - predicted")
        residual_axis.grid(alpha=0.2)

    figure.suptitle("Prediction comparison on the same 11-log validation set")
    figure.savefig(PREDICTION_FIGURE, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    baseline_prefixes = tuple(args.baseline_prefixes or BASELINE_PREFIXES)
    config_rows = load_dataset_config(Path(args.config))
    all_prefixes = tuple(row["prefix"] for row in config_rows)
    baseline_config_rows = filter_config_rows(config_rows, baseline_prefixes)
    additional_prefixes = tuple(prefix for prefix in all_prefixes if prefix not in set(baseline_prefixes))
    validation_training_rows = build_training_rows(config_rows)

    variant_training_prefixes = {
        "Original 2 logs": baseline_prefixes,
        f"Original + {len(additional_prefixes)} logs": all_prefixes,
    }
    variant_predictions = {
        "Original 2 logs": build_predictions_for_training_set(baseline_config_rows, validation_training_rows),
        f"Original + {len(additional_prefixes)} logs": build_predictions_for_training_set(config_rows, validation_training_rows),
    }

    metric_rows = build_metric_rows(
        variant_predictions,
        variant_training_prefixes,
        baseline_prefixes,
        additional_prefixes,
        all_prefixes,
    )
    write_metric_rows(metric_rows)
    plot_metric_comparison(metric_rows)
    plot_prediction_comparison(variant_predictions, metric_rows, baseline_prefixes)

    print("Saved:")
    print(f"  {METRICS_CSV}")
    print(f"  {METRIC_FIGURE}")
    print(f"  {PREDICTION_FIGURE}")


if __name__ == "__main__":
    main()