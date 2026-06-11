"""
turn_mpc_static_demo.py

定常旋回モデルだけを使った、クローラダンプ上部旋回体の簡易MPC確認プログラム。

モデル:
    omega = sign(u) * omega0(|u|, load_state) + k(|u|, load_state) * M

    psi[k+1] = psi[k] + dt * omega[k]

ここでは一次遅れ tau や慣性 J は使わない。
その代わり、定常モデルだけでは表現しにくい以下の懸念を評価関数で扱う。

- 停止近傍のオーバーシュート懸念
    -> 目標近傍で yaw rate 罰則を強める
    -> 終端角度誤差を強める
- 急なレバー操作による実機遅れ・ショック
    -> 入力変化率 du 罰則
    -> du 制約
- 重力モーメントが大きいときの予測不確かさ
    -> yaw rate 罰則、du 罰則、終端罰則を強める
- 目標を通過した後も回り続ける懸念
    -> 角度誤差の符号反転ペナルティ
    -> 目標近傍での低速化ペナルティ

必要パッケージ:
    pip install numpy pandas scipy matplotlib

使い方:
    python turn_mpc_static_demo.py

同じフォルダに turn_model_coefficients.csv があれば読み込む。
ない場合はデモ用係数で実行する。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import messagebox

from output_paths import (
    TURN_MODEL_COEFFICIENTS_CSV,
    TURN_MPC_STATIC_DEMO_OUTPUT_DIR,
    ensure_parent_dir,
    resolve_prefixed_output,
)


RESULT_CSV_SUBDIR = Path("CSV") / "時系列"
ANGLE_FIGURE_SUBDIR = Path("画像") / "角度"
OMEGA_FIGURE_SUBDIR = Path("画像") / "角速度"
INPUT_FIGURE_SUBDIR = Path("画像") / "レバー入力"
MOMENT_FIGURE_SUBDIR = Path("画像") / "モーメント"
HORIZON_FIGURE_SUBDIR = Path("画像") / "予測ホライズン"
STORYBOARD_FIGURE_SUBDIR = Path("画像") / "絵コンテ"
ANIMATION_SUBDIR = Path("アニメーション")


@dataclass
class MachineParams:
    # empty の例。積荷対応時は load_state ごとに別管理する。
    m: float = 7785.0
    g: float = 9.80665
    xcg: float = -0.587
    ycg: float = 0.030


@dataclass
class MPCParams:
    dt: float = 0.1

    # ここでは符号付き yaw rate モデルを想定し、左右両方向の旋回を扱う。
    u_min: float = -1.0
    u_max: float = 1.0
    du_max: float = 0.06
    u_grid_step: float = 0.02

    min_horizon: int = 8
    max_horizon: int = 50

    # 基本重み
    q_angle: float = 8.0
    q_terminal: float = 120.0
    q_omega: float = 0.8
    r_u: float = 0.03
    r_du: float = 3.0

    # 停止・通過対策
    stop_band_rad: float = math.radians(6.0)
    q_stop_omega: float = 18.0
    q_overshoot: float = 250.0
    q_reverse_error: float = 80.0

    # 外乱モーメントが大きい時の安全側補正
    moment_norm_nm: float = 30000.0
    moment_horizon_gain: float = 0.35
    moment_weight_gain: float = 2.0


@dataclass
class ScenarioParams:
    target_angle_deg: float = 90.0
    sim_time: float = 30.0
    load_state: str = "empty"
    roll_offset_deg: float = 2.0
    roll_amp_deg: float = 4.0
    roll_rate_rad_s: float = 0.10
    pitch_offset_deg: float = 0.0
    pitch_amp_deg: float = 2.0
    pitch_rate_rad_s: float = 0.07
    output_prefix: str = str(TURN_MPC_STATIC_DEMO_OUTPUT_DIR / "turn_mpc_static")


def parse_args() -> ScenarioParams:
    parser = argparse.ArgumentParser(description="任意の目標角度とロール・ピッチ条件で旋回を試す簡易MPCシミュレータ")
    parser.add_argument("--gui", action="store_true", help="簡単な入力フォームを表示する")
    parser.add_argument("--target-angle-deg", type=float, default=90.0)
    parser.add_argument("--sim-time", type=float, default=30.0)
    parser.add_argument("--load-state", default="empty")
    parser.add_argument("--roll-deg", type=float, default=2.0, help="ロール角オフセット [deg]")
    parser.add_argument("--roll-amp-deg", type=float, default=4.0, help="ロール角振幅 [deg]")
    parser.add_argument("--roll-rate-rad-s", type=float, default=0.10, help="ロール角波形の角周波数 [rad/s]")
    parser.add_argument("--pitch-deg", type=float, default=0.0, help="ピッチ角オフセット [deg]")
    parser.add_argument("--pitch-amp-deg", type=float, default=2.0, help="ピッチ角振幅 [deg]")
    parser.add_argument("--pitch-rate-rad-s", type=float, default=0.07, help="ピッチ角波形の角周波数 [rad/s]")
    parser.add_argument("--output-prefix", default="turn_mpc_static")
    args = parser.parse_args()
    scenario = ScenarioParams(
        target_angle_deg=args.target_angle_deg,
        sim_time=args.sim_time,
        load_state=args.load_state,
        roll_offset_deg=args.roll_deg,
        roll_amp_deg=args.roll_amp_deg,
        roll_rate_rad_s=args.roll_rate_rad_s,
        pitch_offset_deg=args.pitch_deg,
        pitch_amp_deg=args.pitch_amp_deg,
        pitch_rate_rad_s=args.pitch_rate_rad_s,
        output_prefix=resolve_prefixed_output(args.output_prefix, TURN_MPC_STATIC_DEMO_OUTPUT_DIR),
    )
    return args.gui, scenario


def resolve_scenario_output_prefix(scenario: ScenarioParams) -> Path:
    return Path(resolve_prefixed_output(scenario.output_prefix, TURN_MPC_STATIC_DEMO_OUTPUT_DIR))


def build_scenario_output_path(scenario: ScenarioParams, suffix: str, output_subdir: Path) -> Path:
    output_prefix = resolve_scenario_output_prefix(scenario)
    return ensure_parent_dir(output_prefix.parent / output_subdir / f"{output_prefix.name}_{suffix}")


def create_file_list(scenario: ScenarioParams) -> list[str]:
    return [
        str(build_scenario_output_path(scenario, "result.csv", RESULT_CSV_SUBDIR)),
        str(build_scenario_output_path(scenario, "angle.png", ANGLE_FIGURE_SUBDIR)),
        str(build_scenario_output_path(scenario, "omega.png", OMEGA_FIGURE_SUBDIR)),
        str(build_scenario_output_path(scenario, "input.png", INPUT_FIGURE_SUBDIR)),
        str(build_scenario_output_path(scenario, "moment.png", MOMENT_FIGURE_SUBDIR)),
        str(build_scenario_output_path(scenario, "horizon.png", HORIZON_FIGURE_SUBDIR)),
        str(build_scenario_output_path(scenario, "storyboard.png", STORYBOARD_FIGURE_SUBDIR)),
        str(build_scenario_output_path(scenario, "animation.gif", ANIMATION_SUBDIR)),
    ]


def save_simulation_outputs(rows: list[dict[str, float | bool]], scenario: ScenarioParams) -> list[str]:
    if not rows:
        return []

    fieldnames = list(rows[0].keys()) if rows else []
    result_csv = build_scenario_output_path(scenario, "result.csv", RESULT_CSV_SUBDIR)
    with result_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    times = np.array([row["time"] for row in rows], dtype=float)
    psi_deg = np.array([row["psi_deg"] for row in rows], dtype=float)
    target_psi_deg = np.array([row["target_psi_deg"] for row in rows], dtype=float)
    omega_rad_s = np.array([row["omega_rad_s"] for row in rows], dtype=float)
    u_cmd = np.array([row["u_cmd"] for row in rows], dtype=float)
    moment_nm = np.array([row["moment_nm"] for row in rows], dtype=float)
    horizon_hist = np.array([row["horizon"] for row in rows], dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(times, psi_deg, label="psi")
    plt.plot(times, target_psi_deg, "--", label="target")
    plt.xlabel("time [s]")
    plt.ylabel("angle [deg]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(build_scenario_output_path(scenario, "angle.png", ANGLE_FIGURE_SUBDIR), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(times, omega_rad_s, label="omega")
    plt.xlabel("time [s]")
    plt.ylabel("yaw rate [rad/s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(build_scenario_output_path(scenario, "omega.png", OMEGA_FIGURE_SUBDIR), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(times, u_cmd, label="u")
    plt.xlabel("time [s]")
    plt.ylabel("lever command")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(build_scenario_output_path(scenario, "input.png", INPUT_FIGURE_SUBDIR), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(times, moment_nm, label="gravity yaw moment")
    plt.xlabel("time [s]")
    plt.ylabel("moment [Nm]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(build_scenario_output_path(scenario, "moment.png", MOMENT_FIGURE_SUBDIR), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(times, horizon_hist, label="horizon")
    plt.xlabel("time [s]")
    plt.ylabel("prediction horizon")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(build_scenario_output_path(scenario, "horizon.png", HORIZON_FIGURE_SUBDIR), dpi=150)
    plt.close()

    save_turn_storyboard(rows, scenario)
    save_turn_animation(rows, scenario)
    return create_file_list(scenario)


def rotate_points(points: np.ndarray, angle_rad: float) -> np.ndarray:
    rotation = np.array(
        [
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)],
        ],
        dtype=float,
    )
    return points @ rotation.T


def draw_machine_top_view(axis: plt.Axes, yaw_rad: float, target_yaw_rad: float, title: str) -> None:
    track_width = 3.6
    track_length = 6.0
    upper_width = 2.6
    upper_length = 4.8
    cab_length = 1.2

    track_outline = np.array(
        [
            [-track_length / 2, -track_width / 2],
            [track_length / 2, -track_width / 2],
            [track_length / 2, track_width / 2],
            [-track_length / 2, track_width / 2],
        ],
        dtype=float,
    )
    upper_outline = np.array(
        [
            [-upper_length / 2, -upper_width / 2],
            [upper_length / 2, -upper_width / 2],
            [upper_length / 2, upper_width / 2],
            [-upper_length / 2, upper_width / 2],
        ],
        dtype=float,
    )
    cab_outline = np.array(
        [
            [-upper_length / 2, -upper_width / 2],
            [(-upper_length / 2) + cab_length, -upper_width / 2],
            [(-upper_length / 2) + cab_length, upper_width / 2],
            [-upper_length / 2, upper_width / 2],
        ],
        dtype=float,
    )
    body_arrow = np.array(
        [
            [upper_length / 2 - 0.3, 0.0],
            [upper_length / 2 - 1.0, 0.35],
            [upper_length / 2 - 1.0, -0.35],
        ],
        dtype=float,
    )

    rotated_upper = rotate_points(upper_outline, yaw_rad)
    rotated_cab = rotate_points(cab_outline, yaw_rad)
    rotated_arrow = rotate_points(body_arrow, yaw_rad)

    axis.add_patch(patches.Polygon(track_outline, closed=True, facecolor="#d1d5db", edgecolor="#4b5563", linewidth=1.5))
    axis.add_patch(patches.Polygon(rotated_upper, closed=True, facecolor="#eab308", edgecolor="#854d0e", linewidth=1.8))
    axis.add_patch(patches.Polygon(rotated_cab, closed=True, facecolor="#0f766e", edgecolor="#134e4a", linewidth=1.4))
    axis.add_patch(patches.Polygon(rotated_arrow, closed=True, facecolor="#1d4ed8", edgecolor="#1e3a8a", linewidth=1.0))

    target_tip = np.array([[0.0, 3.1], [0.22, 2.5], [-0.22, 2.5]], dtype=float)
    target_arrow = rotate_points(target_tip, target_yaw_rad)
    axis.add_patch(patches.Polygon(target_arrow, closed=True, facecolor="#dc2626", edgecolor="#7f1d1d", linewidth=1.0))
    axis.plot([0.0, 2.9 * math.cos(target_yaw_rad)], [0.0, 2.9 * math.sin(target_yaw_rad)], color="#f87171", linestyle="--", linewidth=1.2)

    axis.set_title(title)
    axis.set_aspect("equal")
    axis.set_xlim(-4.0, 4.0)
    axis.set_ylim(-4.0, 4.0)
    axis.grid(alpha=0.18)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")


def save_turn_storyboard(rows: list[dict[str, float | bool]], scenario: ScenarioParams) -> Path:
    storyboard_path = build_scenario_output_path(scenario, "storyboard.png", STORYBOARD_FIGURE_SUBDIR)
    if not rows:
        return storyboard_path

    frame_count = min(8, len(rows))
    frame_indices = np.linspace(0, len(rows) - 1, frame_count, dtype=int)
    columns = 4
    rows_count = int(math.ceil(frame_count / columns))
    figure, axes = plt.subplots(rows_count, columns, figsize=(14, 3.8 * rows_count), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(rows_count, columns)
    target_yaw_rad = math.radians(scenario.target_angle_deg)

    for axis in axes_array.ravel():
        axis.set_visible(False)

    for axis, sample_index in zip(axes_array.ravel(), frame_indices):
        axis.set_visible(True)
        sample = rows[int(sample_index)]
        yaw_rad = math.radians(float(sample["psi_deg"]))
        title = (
            f"t={float(sample['time']):.1f}s / yaw={float(sample['psi_deg']):.1f}deg\n"
            f"u={float(sample['u_cmd']):.2f}"
        )
        draw_machine_top_view(axis, yaw_rad, target_yaw_rad, title)

    figure.suptitle("Turn MPC storyboard", fontsize=14)
    figure.savefig(storyboard_path, dpi=160)
    plt.close(figure)
    return storyboard_path


def save_turn_animation(rows: list[dict[str, float | bool]], scenario: ScenarioParams) -> Path:
    animation_path = build_scenario_output_path(scenario, "animation.gif", ANIMATION_SUBDIR)
    if not rows:
        return animation_path

    target_yaw_rad = math.radians(scenario.target_angle_deg)
    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)

    def update(frame_index: int):
        axis.clear()
        sample = rows[frame_index]
        yaw_rad = math.radians(float(sample["psi_deg"]))
        title = (
            f"t={float(sample['time']):.1f}s / yaw={float(sample['psi_deg']):.1f}deg\n"
            f"u={float(sample['u_cmd']):.2f}, omega={float(sample['omega_rad_s']):.3f}rad/s"
        )
        draw_machine_top_view(axis, yaw_rad, target_yaw_rad, title)

    animation = FuncAnimation(
        figure,
        update,
        frames=len(rows),
        interval=max(int(1000 * 0.08), 1),
        repeat=True,
    )
    animation.save(animation_path, writer=PillowWriter(fps=12))
    plt.close(figure)
    return animation_path


def build_horizon_formula_text(mpc: MPCParams) -> str:
    return (
        "予測ホライズン:\n"
        "H = clip(round(Hmin + (Hmax - Hmin) * ((1 - a) * clip(|angle_error| / 90deg, 0, 1) "
        "+ a * clip(|M| / Mnorm, 0, 1))), Hmin, Hmax)\n"
        "M = m g (xcg sin(phi) + ycg sin(theta))\n"
        f"現在値: dt={mpc.dt:.2f} s, Hmin={mpc.min_horizon}, Hmax={mpc.max_horizon}, "
        f"a={mpc.moment_horizon_gain:.2f}, Mnorm={mpc.moment_norm_nm:.0f} Nm"
    )


def launch_gui(default_scenario: ScenarioParams) -> None:
    root = tk.Tk()
    root.title("Turn MPC Static Simulator")
    root.resizable(False, False)
    mpc = MPCParams()

    fields = [
        ("目標角度 [deg]", "target_angle_deg"),
        ("シミュレーション時間 [s]", "sim_time"),
        ("積荷状態", "load_state"),
        ("ロールオフセット [deg]", "roll_offset_deg"),
        ("ロール振幅 [deg]", "roll_amp_deg"),
        ("ロール角周波数 [rad/s]", "roll_rate_rad_s"),
        ("ピッチオフセット [deg]", "pitch_offset_deg"),
        ("ピッチ振幅 [deg]", "pitch_amp_deg"),
        ("ピッチ角周波数 [rad/s]", "pitch_rate_rad_s"),
        ("出力プレフィックス", "output_prefix"),
    ]

    variables: dict[str, tk.StringVar] = {}
    for row_index, (label, attr) in enumerate(fields):
        tk.Label(root, text=label, anchor="w", width=24).grid(row=row_index, column=0, padx=8, pady=4, sticky="w")
        value = getattr(default_scenario, attr)
        variable = tk.StringVar(value=str(value))
        variables[attr] = variable
        tk.Entry(root, textvariable=variable, width=28).grid(row=row_index, column=1, padx=8, pady=4)

    status_var = tk.StringVar(value="条件を入力して [実行] を押してください。")
    tk.Label(root, textvariable=status_var, anchor="w", width=58, justify="left").grid(
        row=len(fields), column=0, columnspan=2, padx=8, pady=(8, 4), sticky="w"
    )
    tk.Label(root, text=build_horizon_formula_text(mpc), anchor="w", justify="left", wraplength=520).grid(
        row=len(fields) + 1, column=0, columnspan=2, padx=8, pady=(0, 8), sticky="w"
    )

    control_frame = tk.Frame(root)
    control_frame.grid(row=len(fields) + 2, column=0, columnspan=2, padx=8, pady=(0, 8), sticky="w")

    animation_frame = tk.LabelFrame(root, text="ライブアニメーション", padx=6, pady=6)
    animation_frame.grid(row=len(fields) + 3, column=0, columnspan=2, padx=8, pady=(0, 8), sticky="w")

    animation_figure = Figure(figsize=(7.4, 7.0), constrained_layout=True)
    animation_grid = animation_figure.add_gridspec(2, 1, height_ratios=[3.2, 1.3])
    animation_top_axis = animation_figure.add_subplot(animation_grid[0])
    animation_input_axis = animation_figure.add_subplot(animation_grid[1])
    animation_canvas = FigureCanvasTkAgg(animation_figure, master=animation_frame)
    animation_canvas.get_tk_widget().grid(row=0, column=0, columnspan=3, padx=4, pady=4)

    animation_status_var = tk.StringVar(value="実行後にこの領域でアニメーションを再生します。")
    tk.Label(animation_frame, textvariable=animation_status_var, anchor="w", justify="left", width=84).grid(
        row=1, column=0, columnspan=3, padx=4, pady=(0, 6), sticky="w"
    )

    play_pause_var = tk.StringVar(value="再生")
    animation_state = {
        "rows": [],
        "scenario": None,
        "dt": mpc.dt,
        "index": 0,
        "job": None,
        "playing": False,
    }
    latest_result_state = {"rows": [], "scenario": None}

    def draw_animation_placeholder() -> None:
        animation_top_axis.clear()
        animation_top_axis.text(0.5, 0.5, "ここに旋回アニメーションを表示します", ha="center", va="center")
        animation_top_axis.set_axis_off()

        animation_input_axis.clear()
        animation_input_axis.text(0.5, 0.5, "実行後にレバー推移を重ねて表示します", ha="center", va="center")
        animation_input_axis.set_axis_off()
        animation_canvas.draw_idle()

    def cancel_animation_job() -> None:
        job = animation_state["job"]
        if job is not None:
            root.after_cancel(job)
            animation_state["job"] = None

    def render_animation_frame() -> None:
        rows = animation_state["rows"]
        if not rows:
            draw_animation_placeholder()
            return

        sample = rows[int(animation_state["index"])]
        scenario = animation_state["scenario"]
        dt_local = float(animation_state["dt"])
        target_yaw_rad = math.radians(float(scenario.target_angle_deg))
        times = np.array([float(row["time"]) for row in rows], dtype=float)
        lever = np.array([float(row["u_cmd"]) for row in rows], dtype=float)

        animation_top_axis.clear()
        draw_machine_top_view(
            animation_top_axis,
            math.radians(float(sample["psi_deg"])),
            target_yaw_rad,
            (
                f"t={float(sample['time']):.1f}s / yaw={float(sample['psi_deg']):.1f}deg\n"
                f"u={float(sample['u_cmd']):.2f}, omega={float(sample['omega_rad_s']):.3f}rad/s"
            ),
        )

        animation_input_axis.clear()
        animation_input_axis.plot(times, lever, color="#1d4ed8", linewidth=1.8, label="u_cmd")
        animation_input_axis.axvline(float(sample["time"]), color="#dc2626", linestyle="--", linewidth=1.2)
        animation_input_axis.scatter([float(sample["time"])], [float(sample["u_cmd"])], color="#dc2626", s=36, zorder=3)
        x_max = float(times[-1]) if len(times) > 1 else max(float(times[0]), dt_local)
        animation_input_axis.set_xlim(0.0, x_max)
        animation_input_axis.set_ylim(-0.02, max(1.02, float(np.max(lever)) + 0.05))
        animation_input_axis.set_xlabel("time [s]")
        animation_input_axis.set_ylabel("lever")
        animation_input_axis.grid(True, alpha=0.3)
        animation_input_axis.legend(loc="upper right")

        animation_status_var.set(
            f"t={float(sample['time']):.1f} s, yaw={float(sample['psi_deg']):.1f} deg, "
            f"target={float(sample['target_psi_deg']):.1f} deg, u={float(sample['u_cmd']):.2f}, "
            f"omega={float(sample['omega_rad_s']):.3f} rad/s, horizon={int(sample['horizon'])}, dt={dt_local:.2f} s"
        )
        animation_canvas.draw_idle()

    def schedule_next_frame() -> None:
        cancel_animation_job()
        rows = animation_state["rows"]
        if not animation_state["playing"] or not rows:
            return
        if int(animation_state["index"]) >= len(rows) - 1:
            animation_state["playing"] = False
            play_pause_var.set("再生")
            return
        delay_ms = max(int(round(float(animation_state["dt"]) * 1000.0)), 1)
        animation_state["job"] = root.after(delay_ms, advance_animation_frame)

    def advance_animation_frame() -> None:
        animation_state["job"] = None
        rows = animation_state["rows"]
        if rows and int(animation_state["index"]) < len(rows) - 1:
            animation_state["index"] = int(animation_state["index"]) + 1
        render_animation_frame()
        schedule_next_frame()

    def toggle_play_pause() -> None:
        rows = animation_state["rows"]
        if not rows:
            return
        if animation_state["playing"]:
            animation_state["playing"] = False
            play_pause_var.set("再生")
            cancel_animation_job()
            return
        if int(animation_state["index"]) >= len(rows) - 1:
            animation_state["index"] = 0
        animation_state["playing"] = True
        play_pause_var.set("一時停止")
        render_animation_frame()
        schedule_next_frame()

    def restart_animation() -> None:
        if not animation_state["rows"]:
            return
        cancel_animation_job()
        animation_state["index"] = 0
        render_animation_frame()
        if animation_state["playing"]:
            schedule_next_frame()

    play_pause_button = tk.Button(animation_frame, textvariable=play_pause_var, width=14, command=toggle_play_pause, state="disabled")
    play_pause_button.grid(row=2, column=0, padx=4, pady=(0, 4))
    restart_button = tk.Button(animation_frame, text="先頭から", width=14, command=restart_animation, state="disabled")
    restart_button.grid(row=2, column=1, padx=4, pady=(0, 4))
    tk.Button(animation_frame, text="表示更新", width=14, command=render_animation_frame).grid(row=2, column=2, padx=4, pady=(0, 4))

    draw_animation_placeholder()

    def build_scenario_from_form() -> ScenarioParams:
        return ScenarioParams(
            target_angle_deg=float(variables["target_angle_deg"].get()),
            sim_time=float(variables["sim_time"].get()),
            load_state=variables["load_state"].get().strip() or "empty",
            roll_offset_deg=float(variables["roll_offset_deg"].get()),
            roll_amp_deg=float(variables["roll_amp_deg"].get()),
            roll_rate_rad_s=float(variables["roll_rate_rad_s"].get()),
            pitch_offset_deg=float(variables["pitch_offset_deg"].get()),
            pitch_amp_deg=float(variables["pitch_amp_deg"].get()),
            pitch_rate_rad_s=float(variables["pitch_rate_rad_s"].get()),
            output_prefix=resolve_prefixed_output(
                variables["output_prefix"].get().strip() or "turn_mpc_static",
                TURN_MPC_STATIC_DEMO_OUTPUT_DIR,
            ),
        )

    def on_run() -> None:
        try:
            scenario = build_scenario_from_form()
            rows = run_simulation(scenario, save_outputs=False)
        except Exception as error:
            status_var.set(f"実行失敗: {error}")
            messagebox.showerror("Turn MPC Simulator", str(error))
            return

        cancel_animation_job()
        animation_state["rows"] = rows
        animation_state["scenario"] = scenario
        animation_state["dt"] = mpc.dt
        animation_state["index"] = 0
        animation_state["playing"] = True
        latest_result_state["rows"] = rows
        latest_result_state["scenario"] = scenario
        play_pause_var.set("一時停止")
        play_pause_button.configure(state="normal")
        restart_button.configure(state="normal")
        export_button.configure(state="normal")
        render_animation_frame()
        schedule_next_frame()
        status_var.set("実行完了: GUI 内ライブ再生を開始しました。保存が必要なら [出力保存] を押してください。")

    def on_export() -> None:
        rows = latest_result_state["rows"]
        scenario = latest_result_state["scenario"]
        if not rows or scenario is None:
            status_var.set("先に [実行] して結果を作成してください。")
            return
        try:
            file_list = save_simulation_outputs(rows, scenario)
        except Exception as error:
            status_var.set(f"保存失敗: {error}")
            messagebox.showerror("Turn MPC Simulator", str(error))
            return
        status_var.set("保存完了: " + ", ".join(file_list))

    def on_close_root() -> None:
        cancel_animation_job()
        root.destroy()

    tk.Button(control_frame, text="実行", width=16, command=on_run).grid(row=0, column=0, padx=(0, 8), pady=0)
    export_button = tk.Button(control_frame, text="出力保存", width=16, command=on_export, state="disabled")
    export_button.grid(row=0, column=1, padx=8, pady=0)
    tk.Button(control_frame, text="閉じる", width=16, command=on_close_root).grid(row=0, column=2, padx=(8, 0), pady=0)

    root.protocol("WM_DELETE_WINDOW", on_close_root)
    root.mainloop()


class TurnModel:
    def __init__(self, coefficient_csv: str | Path | None = None, load_state: str = "empty"):
        self.coeff = self._load_coefficients(coefficient_csv, load_state)

    def _load_coefficients(self, coefficient_csv, load_state):
        if coefficient_csv is not None and Path(coefficient_csv).exists():
            rows = list(csv.DictReader(Path(coefficient_csv).open("r", encoding="utf-8", newline="")))
            if rows and "load_state" in rows[0]:
                filtered_rows = [row for row in rows if str(row.get("load_state", "")).lower() == load_state.lower()]
                if filtered_rows:
                    rows = filtered_rows

            required = {"lever", "baseline_abs_yaw", "moment_gain_per_nm"}
            missing = required - set(rows[0].keys())
            if missing:
                raise ValueError(f"CSVに必要列がありません: {missing}")

            return sorted(
                [
                    {
                        "lever": float(row["lever"]),
                        "baseline_abs_yaw": float(row["baseline_abs_yaw"]),
                        "moment_gain_per_nm": float(row["moment_gain_per_nm"]),
                    }
                    for row in rows
                ]
                + [{"lever": 0.0, "baseline_abs_yaw": 0.0, "moment_gain_per_nm": 0.0}],
                key=lambda row: row["lever"],
            )

        # デモ用係数。実評価では出力/05_汎用旋回モデル/CSV/モデル係数/turn_model_coefficients.csvを使う。
        lever = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        baseline = [0.0, 0.015, 0.030, 0.045, 0.065, 0.090, 0.130, 0.180, 0.270, 0.420, 0.480]
        gain = [0.0, 1.0e-6, 1.4e-6, 1.8e-6, 2.0e-6, 2.2e-6, 2.4e-6, 1.8e-6, 1.2e-6, 0.5e-6, 0.3e-6]
        return [
            {"lever": lever_value, "baseline_abs_yaw": baseline_value, "moment_gain_per_nm": gain_value}
            for lever_value, baseline_value, gain_value in zip(lever, baseline, gain, strict=True)
        ]

    def omega0(self, u: float) -> float:
        lever = np.array([row["lever"] for row in self.coeff], dtype=float)
        baseline = np.array([row["baseline_abs_yaw"] for row in self.coeff], dtype=float)
        return float(np.interp(u, lever, baseline))

    def moment_gain(self, u: float) -> float:
        lever = np.array([row["lever"] for row in self.coeff], dtype=float)
        gain = np.array([row["moment_gain_per_nm"] for row in self.coeff], dtype=float)
        return float(np.interp(u, lever, gain))

    def omega_ss(self, u: float, moment_nm: float) -> float:
        lever_abs = abs(float(u))
        baseline = math.copysign(self.omega0(lever_abs), float(u))
        omega = baseline + self.moment_gain(lever_abs) * moment_nm
        return float(omega)

    def discrete_lever_candidates(self, u_prev: float, du_max: float, u_min: float, u_max: float, u_grid_step: float) -> np.ndarray:
        lower = max(u_min, u_prev - du_max)
        upper = min(u_max, u_prev + du_max)
        discrete_grid = np.arange(u_min, u_max + 0.5 * u_grid_step, u_grid_step, dtype=float)
        mask = (discrete_grid >= lower - 1e-9) & (discrete_grid <= upper + 1e-9)
        candidates = discrete_grid[mask]
        if candidates.size == 0:
            clipped = float(np.clip(u_prev, u_min, u_max))
            return np.array([clipped])
        return candidates


def gravity_yaw_moment(machine: MachineParams, roll_rad: float, pitch_rad: float) -> float:
    return machine.m * machine.g * (
        machine.xcg * math.sin(roll_rad) + machine.ycg * math.sin(pitch_rad)
    )


def moment_scale(moment_nm: float, mpc: MPCParams) -> float:
    return float(np.clip(abs(moment_nm) / mpc.moment_norm_nm, 0.0, 1.0))


def choose_horizon(angle_error: float, moment_nm: float, mpc: MPCParams) -> int:
    # 残り角度が大きいほど長く見る。
    # 重力モーメントが大きい場合も、予測不確かさ・停止余裕を見て少し長くする。
    angle_scale = float(np.clip(abs(angle_error) / math.radians(90.0), 0.0, 1.0))
    m_scale = moment_scale(moment_nm, mpc)

    h = mpc.min_horizon + (mpc.max_horizon - mpc.min_horizon) * (
        (1.0 - mpc.moment_horizon_gain) * angle_scale
        + mpc.moment_horizon_gain * m_scale
    )

    return int(np.clip(round(h), mpc.min_horizon, mpc.max_horizon))


def choose_weights(angle_error: float, moment_nm: float, mpc: MPCParams) -> dict[str, float]:
    m_scale = moment_scale(moment_nm, mpc)

    # 目標近傍では速度をより強く抑える。
    near_stop = abs(angle_error) < mpc.stop_band_rad
    near_stop_scale = 1.0 - float(np.clip(abs(angle_error) / mpc.stop_band_rad, 0.0, 1.0))

    # 重力モーメントが大きいほど安全側に、速度と入力変化を抑える。
    uncertainty_gain = 1.0 + mpc.moment_weight_gain * m_scale

    return {
        "q_angle": mpc.q_angle * (1.0 + 0.5 * m_scale),
        "q_terminal": mpc.q_terminal * (1.0 + 1.5 * m_scale),
        "q_omega": mpc.q_omega * uncertainty_gain + mpc.q_stop_omega * near_stop_scale,
        "r_u": mpc.r_u * (1.0 + 0.5 * m_scale),
        "r_du": mpc.r_du * uncertainty_gain,
        "q_overshoot": mpc.q_overshoot * (1.0 + 1.0 * m_scale),
        "q_reverse_error": mpc.q_reverse_error,
    }


def build_u_sequence(u_prev: float, u_seq: np.ndarray, mpc: MPCParams) -> np.ndarray:
    clipped = []
    last_u = u_prev
    for u in u_seq:
        u = float(np.clip(u, max(mpc.u_min, last_u - mpc.du_max), min(mpc.u_max, last_u + mpc.du_max)))
        clipped.append(u)
        last_u = u
    return np.array(clipped)


def predict_static(
    psi0: float,
    u_seq: np.ndarray,
    moment_seq: np.ndarray,
    model: TurnModel,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    psi = psi0
    psi_list = []
    omega_list = []

    for u, m in zip(u_seq, moment_seq):
        omega = model.omega_ss(float(u), float(m))
        psi = psi + dt * omega

        psi_list.append(psi)
        omega_list.append(omega)

    return np.array(psi_list), np.array(omega_list)


def solve_static_mpc(
    psi: float,
    target_psi: float,
    u_prev: float,
    moment_now: float,
    model: TurnModel,
    mpc: MPCParams,
):
    angle_error = target_psi - psi
    horizon = choose_horizon(angle_error, moment_now, mpc)
    weights = choose_weights(angle_error, moment_now, mpc)

    # まずはホライズン内でモーメント一定と仮定。
    # 実ログ/予測姿勢がある場合はここを moment_seq に置き換える。
    moment_seq = np.full(horizon, moment_now)

    initial_error_sign = np.sign(angle_error) if abs(angle_error) > 1e-9 else 0.0

    candidates = np.arange(mpc.u_min, mpc.u_max + 0.5 * mpc.u_grid_step, mpc.u_grid_step, dtype=float)

    def ramp_to_target_sequence(target_u: float) -> np.ndarray:
        # 総当たり対象はホライズン内で目指す目標レバー値とし、du 制約の範囲で漸近的に到達する列を評価する。
        return np.full(horizon, target_u, dtype=float)

    def objective(u_seq: np.ndarray) -> float:
        u_seq = build_u_sequence(u_prev, u_seq, mpc)
        psi_pred, omega_pred = predict_static(psi, u_seq, moment_seq, model, mpc.dt)

        cost = 0.0
        last_u = u_prev

        for i in range(horizon):
            e = target_psi - psi_pred[i]
            u = u_seq[i]
            omega = omega_pred[i]
            du = u - last_u

            # 先の時刻ほど角度誤差を少し重くする。
            progress = i / max(horizon - 1, 1)
            q_angle_i = weights["q_angle"] * (1.0 + 0.7 * progress)

            # 目標に近いほど yaw rate を強く抑える。
            stop_scale = 1.0 - float(np.clip(abs(e) / mpc.stop_band_rad, 0.0, 1.0))
            q_omega_i = weights["q_omega"] * (1.0 + 3.0 * stop_scale)

            cost += q_angle_i * e * e
            cost += q_omega_i * omega * omega
            cost += weights["r_u"] * u * u
            cost += weights["r_du"] * du * du

            # 目標角を通過した場合の罰則。
            if initial_error_sign != 0.0 and np.sign(e) != 0.0 and np.sign(e) != initial_error_sign:
                cost += weights["q_overshoot"] * e * e

            # 通過後も速度があることを強く嫌う。
            if initial_error_sign != 0.0 and np.sign(e) != 0.0 and np.sign(e) != initial_error_sign:
                cost += weights["q_reverse_error"] * omega * omega

            last_u = u

        e_terminal = target_psi - psi_pred[-1]
        cost += weights["q_terminal"] * e_terminal * e_terminal

        return float(cost)

    best_cost = float("inf")
    best_u = float(np.clip(u_prev, mpc.u_min, mpc.u_max))
    for candidate_u in candidates:
        u_seq = ramp_to_target_sequence(float(candidate_u))
        cost = objective(u_seq)
        if cost < best_cost:
            best_cost = cost
            best_u = float(build_u_sequence(u_prev, u_seq, mpc)[0])

    return best_u, horizon, weights, True


def body_attitude_profile(time_sec: float, scenario: ScenarioParams) -> tuple[float, float]:
    body_roll = math.radians(scenario.roll_offset_deg) + math.radians(scenario.roll_amp_deg) * math.sin(
        scenario.roll_rate_rad_s * time_sec
    )
    body_pitch = math.radians(scenario.pitch_offset_deg) + math.radians(scenario.pitch_amp_deg) * math.cos(
        scenario.pitch_rate_rad_s * time_sec
    )
    return body_roll, body_pitch


def acceleration_from_body_euler(body_roll_rad: float, body_pitch_rad: float) -> tuple[float, float, float]:
    cos_body_roll = math.cos(body_roll_rad)
    acceleration_x = math.sin(body_roll_rad)
    acceleration_y = -cos_body_roll * math.sin(body_pitch_rad)
    acceleration_z = -cos_body_roll * math.cos(body_pitch_rad)
    return acceleration_x, acceleration_y, acceleration_z


def rotate_acceleration_to_upper_body(
    acceleration_x: float,
    acceleration_y: float,
    acceleration_z: float,
    yaw_rad: float,
) -> tuple[float, float, float]:
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    upper_acceleration_x = cos_yaw * acceleration_x + sin_yaw * acceleration_y
    upper_acceleration_y = -sin_yaw * acceleration_x + cos_yaw * acceleration_y
    return upper_acceleration_x, upper_acceleration_y, acceleration_z


def roll_pitch_from_acceleration(
    acceleration_x: float,
    acceleration_y: float,
    acceleration_z: float,
) -> tuple[float, float]:
    euler_x_rad = math.atan2(-acceleration_y, -acceleration_z)
    euler_y_rad = math.atan2(acceleration_x, math.sqrt(acceleration_y * acceleration_y + acceleration_z * acceleration_z))
    return euler_y_rad, euler_x_rad


def attitude_profile(time_sec: float, yaw_rad: float, scenario: ScenarioParams) -> tuple[float, float]:
    body_roll, body_pitch = body_attitude_profile(time_sec, scenario)
    body_acceleration = acceleration_from_body_euler(body_roll, body_pitch)
    upper_body_acceleration = rotate_acceleration_to_upper_body(*body_acceleration, yaw_rad)
    return roll_pitch_from_acceleration(*upper_body_acceleration)


def run_simulation(scenario: ScenarioParams, save_outputs: bool = True) -> list[dict[str, float | bool]]:
    machine = MachineParams()
    mpc = MPCParams()

    coeff_candidates = [TURN_MODEL_COEFFICIENTS_CSV, Path("turn_model_coefficients.csv")]
    coeff_path = next((path for path in coeff_candidates if path.exists()), None)
    model = TurnModel(coeff_path, load_state=scenario.load_state)

    dt = mpc.dt
    sim_time = scenario.sim_time
    n_steps = int(sim_time / dt)

    psi = 0.0
    target_psi = math.radians(scenario.target_angle_deg)
    u_prev = 0.0

    rows = []

    for k in range(n_steps):
        time_sec = k * dt

        roll, pitch = attitude_profile(time_sec, psi, scenario)
        moment_nm = gravity_yaw_moment(machine, roll, pitch)

        u_cmd, horizon, weights, ok = solve_static_mpc(
            psi=psi,
            target_psi=target_psi,
            u_prev=u_prev,
            moment_now=moment_nm,
            model=model,
            mpc=mpc,
        )

        omega = model.omega_ss(u_cmd, moment_nm)
        psi = psi + dt * omega

        rows.append(
            {
                "time": time_sec,
                "psi_deg": math.degrees(psi),
                "target_psi_deg": math.degrees(target_psi),
                "angle_error_deg": math.degrees(target_psi - psi),
                "omega_rad_s": omega,
                "u_cmd": u_cmd,
                "roll_deg": math.degrees(roll),
                "pitch_deg": math.degrees(pitch),
                "moment_nm": moment_nm,
                "horizon": horizon,
                "q_angle": weights["q_angle"],
                "q_terminal": weights["q_terminal"],
                "q_omega": weights["q_omega"],
                "r_u": weights["r_u"],
                "r_du": weights["r_du"],
                "solver_success": ok,
            }
        )

        u_prev = u_cmd

        if abs(target_psi - psi) < math.radians(0.3) and abs(omega) < 0.005:
            break

    if save_outputs:
        print("Saved:")
        for file_name in save_simulation_outputs(rows, scenario):
            print(f"  {file_name}")

    return rows


if __name__ == "__main__":
    use_gui, scenario = parse_args()
    if use_gui:
        launch_gui(scenario)
    else:
        run_simulation(scenario)
