from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from output_paths import ANALYZE_TURN_TREND_OUTPUT_DIR, BUILD_STATIC_ZERO_TILT_MODEL_OUTPUT_DIR, ensure_output_dir


SEGMENT_SUMMARY_CSV = ANALYZE_TURN_TREND_OUTPUT_DIR / "lever_05-10_constant_lever_segments.csv"
STATIC_MAP_CSV = BUILD_STATIC_ZERO_TILT_MODEL_OUTPUT_DIR / "lever_05-10_static_zero_tilt_map.csv"
STATIC_MODEL_CSV = BUILD_STATIC_ZERO_TILT_MODEL_OUTPUT_DIR / "lever_05-10_static_zero_tilt_model.csv"
STATIC_MODEL_FIGURE = BUILD_STATIC_ZERO_TILT_MODEL_OUTPUT_DIR / "lever_05-10_static_zero_tilt_model.png"

STATIC_THRESHOLD = 0.6


def load_zero_tilt_points() -> np.ndarray:
    rows = list(csv.DictReader(SEGMENT_SUMMARY_CSV.open("r", encoding="utf-8", newline="")))
    points: list[tuple[float, float, float, float, float]] = []

    for row in rows:
        lever = float(row["lever"])
        if lever < STATIC_THRESHOLD:
            continue
        points.append(
            (
                lever,
                float(row["ols_abs_yaw_intercept"]),
                float(row["ols_abs_yaw_roll_coef"]),
                float(row["ols_abs_yaw_pitch_coef"]),
                float(row["ols_abs_yaw_r2"]),
            )
        )

    return np.array(points, dtype=float)


def write_static_map(points: np.ndarray) -> None:
    with STATIC_MAP_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "lever",
                "zero_tilt_abs_yaw_static",
                "roll_coef_reference",
                "pitch_coef_reference",
                "segment_r2",
            ]
        )
        writer.writerow([0.0, 0.0, "", "", ""])
        writer.writerow([0.5, 0.0, "", "", ""])
        writer.writerows(points.tolist())


def write_static_model(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lever = points[:, 0]
    zero_tilt = points[:, 1]

    quadratic_coeffs = np.polyfit(lever, zero_tilt, deg=2)
    grid = np.linspace(0.0, 1.0, 101)
    quadratic_pred = np.polyval(quadratic_coeffs, grid)
    piecewise_pred = np.interp(grid, np.concatenate(([0.0, 0.5], lever)), np.concatenate(([0.0, 0.0], zero_tilt)))
    piecewise_pred[grid < STATIC_THRESHOLD] = 0.0

    with STATIC_MODEL_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "parameter", "value"])
        writer.writerow(["rule", "static_threshold", STATIC_THRESHOLD])
        writer.writerow(["rule", "output_below_threshold", 0.0])
        writer.writerow(["quadratic", "a2", quadratic_coeffs[0]])
        writer.writerow(["quadratic", "a1", quadratic_coeffs[1]])
        writer.writerow(["quadratic", "a0", quadratic_coeffs[2]])
        writer.writerow([])
        writer.writerow(["lever", "piecewise_linear_abs_yaw", "quadratic_abs_yaw"])
        for lever_value, piecewise_value, quadratic_value in zip(grid, piecewise_pred, quadratic_pred, strict=True):
            writer.writerow([lever_value, max(piecewise_value, 0.0), max(quadratic_value, 0.0)])

    return grid, piecewise_pred


def plot_static_model(points: np.ndarray, grid: np.ndarray, piecewise_pred: np.ndarray) -> None:
    lever = points[:, 0]
    zero_tilt = points[:, 1]

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.scatter(lever, zero_tilt, s=60, color="#1d4ed8", label="zero-tilt anchor")
    axis.plot(grid, piecewise_pred, color="#0f766e", linewidth=2.5, label="piecewise static model")
    axis.axvline(STATIC_THRESHOLD, color="#6b7280", linestyle="--", linewidth=1.5, label="threshold 0.6")
    axis.set_title("Static zero-tilt yaw-rate surrogate")
    axis.set_xlabel("lever")
    axis.set_ylabel("abs yaw rate")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(STATIC_MODEL_FIGURE, dpi=180)
    plt.close(figure)


def main() -> None:
    ensure_output_dir(BUILD_STATIC_ZERO_TILT_MODEL_OUTPUT_DIR)
    points = load_zero_tilt_points()
    write_static_map(points)
    grid, piecewise_pred = write_static_model(points)
    plot_static_model(points, grid, piecewise_pred)


if __name__ == "__main__":
    main()