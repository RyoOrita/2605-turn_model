from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from analyze_turn_trend import PLOT_MIN_LEVER, build_paths, load_yaw_aligned_samples
from output_paths import (
    BUILD_GENERALIZED_TURN_MODEL_GRID_CSV_DIR,
    BUILD_GENERALIZED_TURN_MODEL_MOMENT_FIGURE_DIR,
    BUILD_GENERALIZED_TURN_MODEL_OVERVIEW_FIGURE_DIR,
    BUILD_GENERALIZED_TURN_MODEL_PREDICTIONS_CSV_DIR,
    BUILD_GENERALIZED_TURN_MODEL_TRAINING_CSV_DIR,
    BUILD_GENERALIZED_TURN_MODEL_VALIDATION_CSV_DIR,
    BUILD_GENERALIZED_TURN_MODEL_VALIDATION_FIGURE_DIR,
    TURN_MODEL_COEFFICIENTS_CSV,
    ensure_parent_dir,
)


DATASET_CONFIG_CSV = Path("turn_model_datasets.csv")
TRAINING_SAMPLES_CSV = BUILD_GENERALIZED_TURN_MODEL_TRAINING_CSV_DIR / "turn_model_training_samples.csv"
MODEL_COEFFICIENTS_CSV = TURN_MODEL_COEFFICIENTS_CSV
MODEL_GRID_CSV = BUILD_GENERALIZED_TURN_MODEL_GRID_CSV_DIR / "turn_model_grid.csv"
MODEL_FIGURE = BUILD_GENERALIZED_TURN_MODEL_OVERVIEW_FIGURE_DIR / "turn_model_overview.png"
MODEL_PREDICTIONS_CSV = BUILD_GENERALIZED_TURN_MODEL_PREDICTIONS_CSV_DIR / "turn_model_predictions.csv"
MODEL_VALIDATION_CSV = BUILD_GENERALIZED_TURN_MODEL_VALIDATION_CSV_DIR / "turn_model_validation_summary.csv"
MODEL_VALIDATION_FIGURE = BUILD_GENERALIZED_TURN_MODEL_VALIDATION_FIGURE_DIR / "turn_model_validation.png"
MOMENT_ABS_YAW_FIGURE = BUILD_GENERALIZED_TURN_MODEL_MOMENT_FIGURE_DIR / "turn_model_moment_vs_abs_yaw.png"
MOMENT_ABS_YAW_BY_LEVER_FIGURE = BUILD_GENERALIZED_TURN_MODEL_MOMENT_FIGURE_DIR / "turn_model_moment_vs_abs_yaw_by_lever.png"
GRAVITY = 9.80665
MODEL_BASE_MIN_LEVER = 0.4
LOW_LEVER_LEVELS = (0.1, 0.2, 0.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a generic load-aware turn model.")
    parser.add_argument("--config", default=str(DATASET_CONFIG_CSV))
    return parser.parse_args()


def load_dataset_config(config_path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(config_path.open("r", encoding="utf-8", newline="")))
    return [row for row in rows if row.get("enabled", "1") == "1"]


def build_training_rows(config_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    training_rows: list[dict[str, object]] = []

    for config in config_rows:
        prefix = config["prefix"]
        load_state = config["load_state"]
        mass_kg = float(config["mass_kg"])
        x_cg_m = float(config["x_cg_mm"]) / 1000.0
        y_cg_m = float(config["y_cg_mm"]) / 1000.0
        z_cg_m = float(config["z_cg_mm"]) / 1000.0

        _, samples = load_yaw_aligned_samples(build_paths(prefix)["input_csv"])
        filtered = samples[samples[:, 0] >= PLOT_MIN_LEVER]

        for sample in filtered:
            lever = float(sample[0])
            roll_deg = float(sample[1])
            pitch_deg = float(sample[2])
            yaw_rate = float(sample[3])
            abs_yaw_rate = float(sample[4])
            roll_rad = np.deg2rad(roll_deg)
            pitch_rad = np.deg2rad(pitch_deg)
            sin_roll = float(np.sin(roll_rad))
            sin_pitch = float(np.sin(pitch_rad))
            moment_proxy_nm = mass_kg * GRAVITY * (x_cg_m * sin_roll + y_cg_m * sin_pitch)

            training_rows.append(
                {
                    "prefix": prefix,
                    "load_state": load_state,
                    "mass_kg": mass_kg,
                    "x_cg_mm": float(config["x_cg_mm"]),
                    "y_cg_mm": float(config["y_cg_mm"]),
                    "z_cg_mm": float(config["z_cg_mm"]),
                    "lever": lever,
                    "roll_deg": roll_deg,
                    "pitch_deg": pitch_deg,
                    "yaw_rate": yaw_rate,
                    "abs_yaw_rate": abs_yaw_rate,
                    "sin_roll": sin_roll,
                    "sin_pitch": sin_pitch,
                    "moment_proxy_nm": moment_proxy_nm,
                }
            )

    return training_rows


def write_training_samples(rows: list[dict[str, object]]) -> None:
    with TRAINING_SAMPLES_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "prefix",
                "load_state",
                "mass_kg",
                "x_cg_mm",
                "y_cg_mm",
                "z_cg_mm",
                "lever",
                "roll_deg",
                "pitch_deg",
                "yaw_rate",
                "abs_yaw_rate",
                "sin_roll",
                "sin_pitch",
                "moment_proxy_nm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def fit_group_model(rows: list[dict[str, object]]) -> dict[str, float]:
    abs_yaw = np.array([float(row["abs_yaw_rate"]) for row in rows])
    moment = np.array([float(row["moment_proxy_nm"]) for row in rows])
    design = np.column_stack([np.ones(abs_yaw.size), moment])
    beta, _, _, _ = np.linalg.lstsq(design, abs_yaw, rcond=None)
    predicted = design @ beta
    residual_sum = float(np.square(abs_yaw - predicted).sum())
    total_sum = float(np.square(abs_yaw - abs_yaw.mean()).sum())
    r2 = float("nan") if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    return {
        "baseline_abs_yaw": float(beta[0]),
        "moment_gain_per_nm": float(beta[1]),
        "r2": r2,
        "sample_count": float(abs_yaw.size),
    }


def predict_abs_yaw_rate(row: dict[str, object], model_row: dict[str, object]) -> float:
    moment = float(row["moment_proxy_nm"])
    baseline = float(model_row["baseline_abs_yaw"])
    moment_gain = float(model_row["moment_gain_per_nm"])
    return baseline + moment_gain * moment


def build_model_rows(training_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped_rows: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in training_rows:
        load_state = str(row["load_state"])
        lever = round(float(row["lever"]), 1)
        grouped_rows.setdefault((load_state, lever), []).append(row)

    model_rows: list[dict[str, object]] = []
    for (load_state, lever), rows in sorted(grouped_rows.items()):
        if lever < MODEL_BASE_MIN_LEVER:
            continue
        coefficients = fit_group_model(rows)
        first_row = rows[0]
        model_rows.append(
            {
                "load_state": load_state,
                "lever": lever,
                "mass_kg": first_row["mass_kg"],
                "x_cg_mm": first_row["x_cg_mm"],
                "y_cg_mm": first_row["y_cg_mm"],
                "z_cg_mm": first_row["z_cg_mm"],
                "source": "fitted",
                **coefficients,
            }
        )

    anchor_map = {str(row["load_state"]): row for row in model_rows if np.isclose(float(row["lever"]), MODEL_BASE_MIN_LEVER)}
    for load_state, anchor_row in sorted(anchor_map.items()):
        for lever in LOW_LEVER_LEVELS:
            scale = lever / MODEL_BASE_MIN_LEVER
            raw_rows = grouped_rows.get((load_state, lever), [])
            model_rows.append(
                {
                    "load_state": load_state,
                    "lever": lever,
                    "mass_kg": anchor_row["mass_kg"],
                    "x_cg_mm": anchor_row["x_cg_mm"],
                    "y_cg_mm": anchor_row["y_cg_mm"],
                    "z_cg_mm": anchor_row["z_cg_mm"],
                    "source": "interpolated_from_0.4",
                    "baseline_abs_yaw": float(anchor_row["baseline_abs_yaw"]) * scale,
                    "moment_gain_per_nm": float(anchor_row["moment_gain_per_nm"]) * scale,
                    "r2": float("nan"),
                    "sample_count": float(len(raw_rows)),
                }
            )

    model_rows.sort(key=lambda row: (str(row["load_state"]), float(row["lever"])))

    return model_rows


def write_model_coefficients(rows: list[dict[str, object]]) -> None:
    with MODEL_COEFFICIENTS_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "load_state",
                "lever",
                "source",
                "mass_kg",
                "x_cg_mm",
                "y_cg_mm",
                "z_cg_mm",
                "baseline_abs_yaw",
                "moment_gain_per_nm",
                "r2",
                "sample_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_prediction_rows(
    training_rows: list[dict[str, object]],
    model_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    model_map = {
        (str(row["load_state"]), round(float(row["lever"]), 1)): row
        for row in model_rows
    }
    prediction_rows: list[dict[str, object]] = []

    for row in training_rows:
        model_row = model_map[(str(row["load_state"]), round(float(row["lever"]), 1))]
        predicted_abs_yaw_rate = predict_abs_yaw_rate(row, model_row)
        actual_abs_yaw_rate = float(row["abs_yaw_rate"])
        residual = actual_abs_yaw_rate - predicted_abs_yaw_rate
        prediction_rows.append(
            {
                **row,
                "predicted_abs_yaw_rate": predicted_abs_yaw_rate,
                "residual_abs_yaw_rate": residual,
            }
        )

    return prediction_rows


def write_prediction_rows(rows: list[dict[str, object]]) -> None:
    with MODEL_PREDICTIONS_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "prefix",
                "load_state",
                "mass_kg",
                "x_cg_mm",
                "y_cg_mm",
                "z_cg_mm",
                "lever",
                "roll_deg",
                "pitch_deg",
                "yaw_rate",
                "abs_yaw_rate",
                "sin_roll",
                "sin_pitch",
                "moment_proxy_nm",
                "predicted_abs_yaw_rate",
                "residual_abs_yaw_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def compute_validation_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    actual = np.array([float(row["abs_yaw_rate"]) for row in rows])
    predicted = np.array([float(row["predicted_abs_yaw_rate"]) for row in rows])
    residual = actual - predicted
    abs_error = np.abs(residual)
    squared_error = np.square(residual)
    total_sum = float(np.square(actual - actual.mean()).sum())

    return {
        "sample_count": float(actual.size),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "bias": float(residual.mean()),
        "r2": float("nan") if total_sum == 0.0 else 1.0 - float(squared_error.sum()) / total_sum,
    }


def build_validation_rows(prediction_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped_rows: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in prediction_rows:
        grouped_rows.setdefault((str(row["load_state"]), str(row["prefix"])), []).append(row)

    validation_rows: list[dict[str, object]] = []
    overall_metrics = compute_validation_metrics(prediction_rows)
    validation_rows.append({"scope": "overall", "load_state": "all", "prefix": "all", **overall_metrics})

    by_load_state: dict[str, list[dict[str, object]]] = {}
    for row in prediction_rows:
        by_load_state.setdefault(str(row["load_state"]), []).append(row)
    for load_state, rows in sorted(by_load_state.items()):
        validation_rows.append(
            {"scope": "load_state", "load_state": load_state, "prefix": "all", **compute_validation_metrics(rows)}
        )

    for (load_state, prefix), rows in sorted(grouped_rows.items()):
        validation_rows.append(
            {"scope": "dataset", "load_state": load_state, "prefix": prefix, **compute_validation_metrics(rows)}
        )

    return validation_rows


def write_validation_rows(rows: list[dict[str, object]]) -> None:
    with MODEL_VALIDATION_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["scope", "load_state", "prefix", "sample_count", "mae", "rmse", "bias", "r2"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_grid_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grid_rows: list[dict[str, object]] = []

    for row in rows:
        grid_rows.append(
            {
                "load_state": row["load_state"],
                "lever": row["lever"],
                "baseline_abs_yaw_rate": row["baseline_abs_yaw"],
                "moment_gain_per_nm": row["moment_gain_per_nm"],
            }
        )

    return grid_rows


def write_model_grid(rows: list[dict[str, object]]) -> None:
    with MODEL_GRID_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "load_state",
                "lever",
                "baseline_abs_yaw_rate",
                "moment_gain_per_nm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_model_overview(model_rows: list[dict[str, object]], grid_rows: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    load_states = sorted({str(row["load_state"]) for row in model_rows})

    for load_state in load_states:
        group_rows = [row for row in grid_rows if row["load_state"] == load_state]
        lever = np.array([float(row["lever"]) for row in group_rows])
        baseline = np.array([float(row["baseline_abs_yaw_rate"]) for row in group_rows])
        moment_gain = np.array([float(row["moment_gain_per_nm"]) for row in group_rows])

        axes[0].plot(lever, baseline, linewidth=2, marker="o", label=load_state)
        axes[1].plot(lever, moment_gain, linewidth=2, marker="o", label=load_state)

    axes[0].set_title("Baseline abs yaw model")
    axes[0].set_xlabel("lever")
    axes[0].set_ylabel("omega_0(u)")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].set_title("Moment gain model")
    axes[1].set_xlabel("lever")
    axes[1].set_ylabel("k(u)")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    figure.savefig(MODEL_FIGURE, dpi=180)
    plt.close(figure)


def plot_validation(prediction_rows: list[dict[str, object]], validation_rows: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
    prefixes = sorted({str(row["prefix"]) for row in prediction_rows})
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(prefixes), 1)))
    all_actual = np.array([float(row["abs_yaw_rate"]) for row in prediction_rows])
    all_predicted = np.array([float(row["predicted_abs_yaw_rate"]) for row in prediction_rows])
    axis_min = float(min(all_actual.min(), all_predicted.min()))
    axis_max = float(max(all_actual.max(), all_predicted.max()))

    for color, prefix in zip(colors, prefixes, strict=True):
        group_rows = [row for row in prediction_rows if str(row["prefix"]) == prefix]
        actual = np.array([float(row["abs_yaw_rate"]) for row in group_rows])
        predicted = np.array([float(row["predicted_abs_yaw_rate"]) for row in group_rows])
        lever = np.array([float(row["lever"]) for row in group_rows])
        residual = actual - predicted

        axes[0].scatter(actual, predicted, s=10, alpha=0.35, color=color, label=prefix)
        axes[1].scatter(lever, residual, s=10, alpha=0.35, color=color, label=prefix)

    axes[0].plot([axis_min, axis_max], [axis_min, axis_max], color="#111827", linewidth=1.5, linestyle="--")
    overall_row = next(row for row in validation_rows if row["scope"] == "overall")
    axes[0].set_title(
        f"Actual vs predicted abs yaw rate (R2={float(overall_row['r2']):.3f}, RMSE={float(overall_row['rmse']):.4f})"
    )
    axes[0].set_xlabel("actual abs yaw rate")
    axes[0].set_ylabel("predicted abs yaw rate")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].axhline(0.0, color="#111827", linewidth=1.5, linestyle="--")
    axes[1].set_title("Residual vs lever")
    axes[1].set_xlabel("lever")
    axes[1].set_ylabel("actual - predicted")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    figure.savefig(MODEL_VALIDATION_FIGURE, dpi=180)
    plt.close(figure)


def plot_moment_vs_abs_yaw(training_rows: list[dict[str, object]]) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    prefixes = sorted({str(row["prefix"]) for row in training_rows})
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(prefixes), 1)))

    for color, prefix in zip(colors, prefixes, strict=True):
        group_rows = [row for row in training_rows if str(row["prefix"]) == prefix]
        moment = np.array([float(row["moment_proxy_nm"]) for row in group_rows])
        abs_yaw = np.array([float(row["abs_yaw_rate"]) for row in group_rows])
        axis.scatter(moment, abs_yaw, s=10, alpha=0.35, color=color, label=prefix)

    axis.set_title("Gravity moment vs abs yaw rate")
    axis.set_xlabel("gravity moment [Nm]")
    axis.set_ylabel("abs yaw rate")
    axis.grid(alpha=0.2)
    axis.legend()

    figure.savefig(MOMENT_ABS_YAW_FIGURE, dpi=180)
    plt.close(figure)


def plot_moment_vs_abs_yaw_by_lever(training_rows: list[dict[str, object]]) -> None:
    lever_levels = sorted({round(float(row["lever"]), 1) for row in training_rows})
    if not lever_levels:
        return

    columns = 2
    rows = int(np.ceil(len(lever_levels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    prefixes = sorted({str(row["prefix"]) for row in training_rows})
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(prefixes), 1)))
    all_abs_yaw = np.array([float(row["abs_yaw_rate"]) for row in training_rows])
    y_max = float(all_abs_yaw.max()) if all_abs_yaw.size else 1.0
    y_upper = y_max * 1.05 if y_max > 0.0 else 1.0

    for axis, lever_level in zip(axes_array, lever_levels, strict=False):
        lever_rows = [row for row in training_rows if np.isclose(round(float(row["lever"]), 1), lever_level)]

        for color, prefix in zip(colors, prefixes, strict=True):
            group_rows = [row for row in lever_rows if str(row["prefix"]) == prefix]
            if not group_rows:
                continue
            moment = np.array([float(row["moment_proxy_nm"]) for row in group_rows])
            abs_yaw = np.array([float(row["abs_yaw_rate"]) for row in group_rows])
            axis.scatter(moment, abs_yaw, s=10, alpha=0.35, color=color, label=prefix)

        axis.set_title(f"lever={lever_level:.1f}")
        axis.set_xlabel("gravity moment [Nm]")
        axis.set_ylabel("abs yaw rate")
        axis.set_ylim(0.0, y_upper)
        axis.grid(alpha=0.2)

    for axis in axes_array[len(lever_levels):]:
        axis.set_visible(False)

    handles, labels = axes_array[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))

    figure.savefig(MOMENT_ABS_YAW_BY_LEVER_FIGURE, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    for output_path in (
        TRAINING_SAMPLES_CSV,
        MODEL_COEFFICIENTS_CSV,
        MODEL_GRID_CSV,
        MODEL_FIGURE,
        MODEL_PREDICTIONS_CSV,
        MODEL_VALIDATION_CSV,
        MODEL_VALIDATION_FIGURE,
        MOMENT_ABS_YAW_FIGURE,
        MOMENT_ABS_YAW_BY_LEVER_FIGURE,
    ):
        ensure_parent_dir(output_path)
    config_rows = load_dataset_config(Path(args.config))
    training_rows = build_training_rows(config_rows)
    write_training_samples(training_rows)
    model_rows = build_model_rows(training_rows)
    write_model_coefficients(model_rows)
    prediction_rows = build_prediction_rows(training_rows, model_rows)
    write_prediction_rows(prediction_rows)
    validation_rows = build_validation_rows(prediction_rows)
    write_validation_rows(validation_rows)
    grid_rows = build_grid_rows(model_rows)
    write_model_grid(grid_rows)
    plot_model_overview(model_rows, grid_rows)
    plot_validation(prediction_rows, validation_rows)
    plot_moment_vs_abs_yaw(training_rows)
    plot_moment_vs_abs_yaw_by_lever(training_rows)


if __name__ == "__main__":
    main()