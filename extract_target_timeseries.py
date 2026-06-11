from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore

from output_paths import EXTRACT_TARGET_TIMESERIES_CSV_DIR, ensure_output_dir


TOPICS = {
    "/cd110r_1/trek/joy_control_command": ("sensor_msgs/msg/Joy", "lever"),
    "/cd110r_1/trek/mtlt335_can_parser/imu_129": ("sensor_msgs/msg/Imu", "yaw_rate"),
    "/cd110r_1/trek/upper_body_simple_turn/euler_y_median": ("std_msgs/msg/Float64", "roll_angle"),
    "/cd110r_1/trek/upper_body_simple_turn/euler_x_median": ("std_msgs/msg/Float64", "pitch_angle"),
}


def build_output_paths(input_mcap: Path) -> tuple[Path, Path]:
    stem = input_mcap.stem
    return (
        EXTRACT_TARGET_TIMESERIES_CSV_DIR / f"{stem}_target_timeseries.csv",
        EXTRACT_TARGET_TIMESERIES_CSV_DIR / f"{stem}_target_timeseries_wide.csv",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract target time series from an MCAP log.")
    parser.add_argument("input_mcap", nargs="?", default="lever_05-10.mcap")
    return parser.parse_args()


def extract_value(topic: str, message: object) -> float | None:
    if topic == "/cd110r_1/trek/joy_control_command":
        axes = getattr(message, "axes", None)
        if axes is None or len(axes) == 0:
            return None
        return float(axes[0])

    if topic == "/cd110r_1/trek/mtlt335_can_parser/imu_129":
        angular_velocity = getattr(message, "angular_velocity", None)
        if angular_velocity is None:
            return None
        return float(angular_velocity.z)

    return float(message.data)


def main() -> None:
    args = parse_args()
    input_mcap = Path(args.input_mcap)
    ensure_output_dir(EXTRACT_TARGET_TIMESERIES_CSV_DIR)
    output_csv, output_wide_csv = build_output_paths(input_mcap)
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with (
        input_mcap.open("rb") as stream,
        output_csv.open("w", newline="", encoding="utf-8") as output,
        output_wide_csv.open("w", newline="", encoding="utf-8") as wide_output,
    ):
        reader = make_reader(stream)
        writer = csv.DictWriter(
            output,
            fieldnames=["timestamp_ns", "topic", "signal", "value"],
        )
        wide_writer = csv.DictWriter(
            wide_output,
            fieldnames=["timestamp_ns", "lever", "yaw_rate", "roll_angle", "pitch_angle"],
        )
        writer.writeheader()
        wide_writer.writeheader()

        for schema, channel, message in reader.iter_messages(topics=TOPICS.keys()):
            msgtype = TOPICS[channel.topic][0]
            decoded = typestore.deserialize_cdr(message.data, msgtype)
            value = extract_value(channel.topic, decoded)
            signal = TOPICS[channel.topic][1]
            writer.writerow(
                {
                    "timestamp_ns": str(int(message.log_time)),
                    "topic": channel.topic,
                    "signal": signal,
                    "value": value,
                }
            )
            wide_writer.writerow(
                {
                    "timestamp_ns": str(int(message.log_time)),
                    "lever": value if signal == "lever" else "",
                    "yaw_rate": value if signal == "yaw_rate" else "",
                    "roll_angle": value if signal == "roll_angle" else "",
                    "pitch_angle": value if signal == "pitch_angle" else "",
                }
            )


if __name__ == "__main__":
    main()