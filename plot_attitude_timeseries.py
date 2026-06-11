from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore

from output_paths import EXTRACT_TARGET_TIMESERIES_ATTITUDE_FIGURE_DIR, ensure_output_dir


ANGLE_TOPICS = {
    "/cd110r_1/trek/upper_body_simple_turn/euler_y_median": ("std_msgs/msg/Float64", "roll"),
    "/cd110r_1/trek/upper_body_simple_turn/euler_x_median": ("std_msgs/msg/Float64", "pitch"),
    "/cd110r_1/trek/upper_body_simple_turn/upper_body_angle_deg": ("std_msgs/msg/Float64", "yaw"),
}

SIGNAL_ORDER = ("roll", "pitch", "yaw")
SIGNAL_STYLES = {
    "roll": {"color": "#2563eb", "ylabel": "roll angle [deg]"},
    "pitch": {"color": "#059669", "ylabel": "pitch angle [deg]"},
    "yaw": {"color": "#dc2626", "ylabel": "yaw angle [deg]"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot roll, pitch, and yaw angle time series for MCAP logs.")
    parser.add_argument("input_mcaps", nargs="*", type=Path, help="MCAP files to plot. Defaults to all *.mcap files.")
    parser.add_argument("--output-dir", type=Path, default=EXTRACT_TARGET_TIMESERIES_ATTITUDE_FIGURE_DIR)
    return parser.parse_args()


def unwrap_degrees(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    return np.rad2deg(np.unwrap(np.deg2rad(values)))


def extract_angles(input_mcap: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    timestamps_by_signal: dict[str, list[int]] = {signal: [] for signal in SIGNAL_ORDER}
    values_by_signal: dict[str, list[float]] = {signal: [] for signal in SIGNAL_ORDER}

    with input_mcap.open("rb") as stream:
        reader = make_reader(stream)
        for schema, channel, message in reader.iter_messages(topics=ANGLE_TOPICS.keys()):
            msgtype, signal = ANGLE_TOPICS[channel.topic]
            decoded = typestore.deserialize_cdr(message.data, msgtype)
            timestamps_by_signal[signal].append(int(message.log_time))
            values_by_signal[signal].append(float(decoded.data))

    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    first_timestamp_ns = min(
        timestamps[0]
        for timestamps in timestamps_by_signal.values()
        if timestamps
    )

    for signal in SIGNAL_ORDER:
        timestamps = np.array(timestamps_by_signal[signal], dtype=np.int64)
        values = np.array(values_by_signal[signal], dtype=float)
        if signal == "yaw":
            values = unwrap_degrees(values)
        times_sec = (timestamps - first_timestamp_ns) / 1e9
        series[signal] = (times_sec, values)

    return series


def plot_angles(input_mcap: Path, output_dir: Path) -> Path:
    series = extract_angles(input_mcap)
    missing_signals = [signal for signal, (_, values) in series.items() if values.size == 0]
    if missing_signals:
        raise RuntimeError(f"{input_mcap} is missing angle signals: {', '.join(missing_signals)}")

    output_path = output_dir / f"{input_mcap.stem}_roll_pitch_yaw_timeseries.png"
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    figure.suptitle(f"{input_mcap.stem} roll / pitch / yaw angle time series")

    for axis, signal in zip(axes, SIGNAL_ORDER, strict=True):
        times_sec, values = series[signal]
        style = SIGNAL_STYLES[signal]
        axis.plot(times_sec, values, linewidth=1.4, color=style["color"])
        axis.set_ylabel(style["ylabel"])
        axis.grid(alpha=0.25)

    axes[-1].set_xlabel("time [s]")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    input_mcaps = args.input_mcaps or sorted(Path(".").glob("*.mcap"))
    if not input_mcaps:
        raise RuntimeError("No MCAP files were found.")

    output_dir = ensure_output_dir(args.output_dir)
    for input_mcap in input_mcaps:
        output_path = plot_angles(input_mcap, output_dir)
        print(output_path)


if __name__ == "__main__":
    main()