#!/usr/bin/env python3

import argparse
import csv
import math
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
VENV_DIR = REPO_ROOT / ".venv-linux"
VENV_PYTHON = VENV_DIR / "bin/python"

# 直接実行やVS Codeの実行ボタンでも、このリポジトリ用のPython環境を優先する。
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV_DIR.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve())] + sys.argv[1:])

import matplotlib
import numpy as np

SHOW_3D_REQUESTED = "--show-3d" in sys.argv
if not SHOW_3D_REQUESTED:
    matplotlib.use("Agg")

from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# このファイルは移植用の1ファイル版。classは使わず、辞書と関数だけで構成する。
DEFAULT_COEFFICIENT_CSV = REPO_ROOT / "出力/05_汎用旋回モデル/CSV/モデル係数/turn_model_coefficients.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def clamp(value, lower, upper):
    return min(max(value, lower), upper)


def gravity_yaw_moment(machine, roll_rad, pitch_rad):
    # 姿勢と重心ずれから、旋回方向に効く重力モーメントを計算する。
    return machine["mass_kg"] * machine["gravity_mps2"] * (
        machine["x_cg_m"] * math.sin(roll_rad) + machine["y_cg_m"] * math.sin(pitch_rad)
    )


def embedded_turn_model_coefficients():
    # 実データ由来の係数CSVと同じ値を埋め込む。
    # 0.4-1.0 は実測フィット値、0.1-0.3 は build_generalized_turn_model.py で生成した補間値。
    return [
        {"lever": 0.1, "omega0_abs_rad_s": 0.00985588651312668, "moment_gain_per_nm": -1.996305880760116e-06},
        {"lever": 0.2, "omega0_abs_rad_s": 0.01971177302625336, "moment_gain_per_nm": -3.992611761520232e-06},
        {"lever": 0.3, "omega0_abs_rad_s": 0.029567659539380033, "moment_gain_per_nm": -5.9889176422803465e-06},
        {"lever": 0.4, "omega0_abs_rad_s": 0.03942354605250672, "moment_gain_per_nm": -7.985223523040464e-06},
        {"lever": 0.5, "omega0_abs_rad_s": 0.06042287111130664, "moment_gain_per_nm": -8.19346341937846e-06},
        {"lever": 0.6, "omega0_abs_rad_s": 0.09877176122853194, "moment_gain_per_nm": -6.9413257682949955e-06},
        {"lever": 0.7, "omega0_abs_rad_s": 0.16452881686114862, "moment_gain_per_nm": -4.229310177135325e-06},
        {"lever": 0.8, "omega0_abs_rad_s": 0.28406221050337704, "moment_gain_per_nm": -4.364388958651549e-06},
        {"lever": 0.9, "omega0_abs_rad_s": 0.4193285215897151, "moment_gain_per_nm": -1.8456287833239893e-06},
        {"lever": 1.0, "omega0_abs_rad_s": 0.45168397980926456, "moment_gain_per_nm": -7.318519443458326e-06},
    ]


def normalize_coefficients(coeffs):
    # u=0を補間の基準点として追加する。同じleverがあれば後勝ちにする。
    rows = [{"lever": 0.0, "omega0_abs_rad_s": 0.0, "moment_gain_per_nm": 0.0}, *coeffs]
    unique_rows = {round(row["lever"], 9): row for row in rows}
    return sorted(unique_rows.values(), key=lambda row: row["lever"])


def load_turn_model_coefficients(coefficient_csv, load_state):
    # CSVの baseline_abs_yaw が omega0、moment_gain_per_nm が k に対応する。
    if not coefficient_csv.exists():
        return normalize_coefficients(embedded_turn_model_coefficients())

    rows = list(csv.DictReader(coefficient_csv.open("r", encoding="utf-8", newline="")))
    if rows and "load_state" in rows[0]:
        matched_rows = [row for row in rows if row.get("load_state", "").lower() == load_state.lower()]
        if matched_rows:
            rows = matched_rows

    coeffs = [
        {
            "lever": float(row["lever"]),
            "omega0_abs_rad_s": float(row["baseline_abs_yaw"]),
            "moment_gain_per_nm": float(row["moment_gain_per_nm"]),
        }
        for row in rows
    ]
    return normalize_coefficients(coeffs)


def interp_by_lever(coeffs, lever_abs, value_key):
    levers = [row["lever"] for row in coeffs]
    values = [row[value_key] for row in coeffs]
    return float(np.interp(clamp(abs(lever_abs), 0.0, 1.0), levers, values))


def omega0(coeffs, lever_abs):
    # レバー量だけで決まる基準旋回速度の大きさ。
    return interp_by_lever(coeffs, lever_abs, "omega0_abs_rad_s")


def moment_gain(coeffs, lever_abs):
    # 重力モーメントMがyaw rateにどれだけ効くかを表す感度k。
    return interp_by_lever(coeffs, lever_abs, "moment_gain_per_nm")


def turn_omega_ss(coeffs, u, moment_nm):
    # 旋回モデル: omega = sign(u) * omega0(|u|) + k(|u|) * M
    lever_abs = abs(float(u))
    lever_component = math.copysign(omega0(coeffs, lever_abs), float(u))
    moment_component = moment_gain(coeffs, lever_abs) * moment_nm
    return lever_component + moment_component


def moment_scale(moment_nm, mpc):
    return clamp(abs(moment_nm) / mpc["moment_norm_nm"], 0.0, 1.0)


def choose_horizon(angle_error_rad, moment_nm, mpc):
    # 残り角度と重力モーメントが大きいほど、少し先まで見る。
    angle_scale = clamp(abs(angle_error_rad) / math.radians(90.0), 0.0, 1.0)
    m_scale = moment_scale(moment_nm, mpc)
    ratio = (1.0 - mpc["moment_horizon_gain"]) * angle_scale + mpc["moment_horizon_gain"] * m_scale
    horizon = mpc["min_horizon"] + (mpc["max_horizon"] - mpc["min_horizon"]) * ratio
    return int(clamp(round(horizon), mpc["min_horizon"], mpc["max_horizon"]))


def choose_weights(angle_error_rad, moment_nm, mpc):
    # 状況に応じてJの重みを変える。目標近傍では速度を強く抑える。
    m_scale = moment_scale(moment_nm, mpc)
    near_stop_scale = 0.0
    if abs(angle_error_rad) < mpc["stop_band_rad"]:
        near_stop_scale = 1.0 - clamp(abs(angle_error_rad) / mpc["stop_band_rad"], 0.0, 1.0)

    uncertainty_gain = 1.0 + mpc["moment_weight_gain"] * m_scale
    return {
        "q_angle": mpc["q_angle"] * (1.0 + 0.5 * m_scale),
        "q_terminal": mpc["q_terminal"] * (1.0 + 1.5 * m_scale),
        "q_omega": mpc["q_omega"] * uncertainty_gain + mpc["q_stop_omega"] * near_stop_scale,
        "r_u": mpc["r_u"] * (1.0 + 0.5 * m_scale),
        "r_du": mpc["r_du"] * uncertainty_gain,
        "q_overshoot": mpc["q_overshoot"] * (1.0 + m_scale),
        "q_reverse_error": mpc["q_reverse_error"],
    }


def build_rate_limited_u_sequence(u_prev, u_seq, mpc):
    # 実機で急にレバーが飛ばないよう、du制約をかける。
    clipped_values = []
    last_u = u_prev
    for u in u_seq:
        lower = max(mpc["u_min"], last_u - mpc["du_max"])
        upper = min(mpc["u_max"], last_u + mpc["du_max"])
        clipped_u = clamp(float(u), lower, upper)
        clipped_values.append(clipped_u)
        last_u = clipped_u
    return np.array(clipped_values, dtype=float)


def predict_horizon(
    psi0_rad,
    u_seq,
    moment_seq,
    coeffs,
    mpc,
):
    # 旋回モデルで未来の角度psiと角速度omegaを予測する。
    psi = psi0_rad
    psi_values = []
    omega_values = []
    for u, moment_nm in zip(u_seq, moment_seq, strict=True):
        omega = turn_omega_ss(coeffs, float(u), float(moment_nm))
        psi = psi + mpc["dt"] * omega
        psi_values.append(psi)
        omega_values.append(omega)
    return np.array(psi_values, dtype=float), np.array(omega_values, dtype=float)


def evaluate_mpc_cost(
    psi_rad,
    target_psi_rad,
    u_prev,
    u_seq_raw,
    moment_seq,
    coeffs,
    mpc,
    weights,
):
    # J = sum(q_angle e^2 + q_omega omega^2 + r_u u^2 + r_du du^2) + terminal + overshoot
    u_seq = build_rate_limited_u_sequence(u_prev, u_seq_raw, mpc)
    psi_pred, omega_pred = predict_horizon(psi_rad, u_seq, moment_seq, coeffs, mpc)

    initial_error = target_psi_rad - psi_rad
    initial_sign = np.sign(initial_error) if abs(initial_error) > 1e-9 else 0.0
    cost = 0.0
    last_u = u_prev

    for index, (psi_i, omega_i, u_i) in enumerate(zip(psi_pred, omega_pred, u_seq, strict=True)):
        error_i = target_psi_rad - float(psi_i)
        du_i = float(u_i) - last_u
        progress = index / max(len(u_seq) - 1, 1)
        q_angle_i = weights["q_angle"] * (1.0 + 0.7 * progress)

        stop_scale = 0.0
        if abs(error_i) < mpc["stop_band_rad"]:
            stop_scale = 1.0 - clamp(abs(error_i) / mpc["stop_band_rad"], 0.0, 1.0)
        q_omega_i = weights["q_omega"] * (1.0 + 3.0 * stop_scale)

        cost += q_angle_i * error_i * error_i
        cost += q_omega_i * float(omega_i) * float(omega_i)
        cost += weights["r_u"] * float(u_i) * float(u_i)
        cost += weights["r_du"] * du_i * du_i

        crossed_target = initial_sign != 0.0 and np.sign(error_i) != 0.0 and np.sign(error_i) != initial_sign
        if crossed_target:
            cost += weights["q_overshoot"] * error_i * error_i
            cost += weights["q_reverse_error"] * float(omega_i) * float(omega_i)

        last_u = float(u_i)

    terminal_error = target_psi_rad - float(psi_pred[-1])
    cost += weights["q_terminal"] * terminal_error * terminal_error
    return float(cost), u_seq


def solve_mpc(
    psi_rad,
    target_psi_rad,
    u_prev,
    moment_now_nm,
    coeffs,
    mpc,
):
    # 候補レバーを総当たりし、Jが最小の最初の入力だけを採用する。
    angle_error = target_psi_rad - psi_rad
    horizon = choose_horizon(angle_error, moment_now_nm, mpc)
    weights = choose_weights(angle_error, moment_now_nm, mpc)
    moment_seq = np.full(horizon, moment_now_nm, dtype=float)
    candidates = np.arange(mpc["u_min"], mpc["u_max"] + 0.5 * mpc["u_grid_step"], mpc["u_grid_step"], dtype=float)

    best_cost = float("inf")
    best_u = clamp(u_prev, mpc["u_min"], mpc["u_max"])
    for candidate_u in candidates:
        candidate_seq = np.full(horizon, float(candidate_u), dtype=float)
        cost, u_seq = evaluate_mpc_cost(psi_rad, target_psi_rad, u_prev, candidate_seq, moment_seq, coeffs, mpc, weights)
        if cost < best_cost:
            best_cost = cost
            best_u = float(u_seq[0])

    return best_u, horizon, weights, best_cost


def body_attitude_profile(time_sec, scenario):
    # テスト用の車体roll/pitch波形。実機ではIMU由来の車体角に置き換える。
    body_roll = math.radians(float(scenario["roll_offset_deg"])) + math.radians(float(scenario["roll_amp_deg"])) * math.sin(
        float(scenario["roll_rate_rad_s"]) * time_sec
    )
    body_pitch = math.radians(float(scenario["pitch_offset_deg"])) + math.radians(float(scenario["pitch_amp_deg"])) * math.cos(
        float(scenario["pitch_rate_rad_s"]) * time_sec
    )
    return body_roll, body_pitch


def acceleration_from_body_euler(body_roll_rad, body_pitch_rad):
    cos_body_roll = math.cos(body_roll_rad)
    acceleration_x = math.sin(body_roll_rad)
    acceleration_y = -cos_body_roll * math.sin(body_pitch_rad)
    acceleration_z = -cos_body_roll * math.cos(body_pitch_rad)
    return acceleration_x, acceleration_y, acceleration_z


def rotate_acceleration_to_upper_body(acceleration_x, acceleration_y, acceleration_z, yaw_rad):
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    upper_acceleration_x = cos_yaw * acceleration_x + sin_yaw * acceleration_y
    upper_acceleration_y = -sin_yaw * acceleration_x + cos_yaw * acceleration_y
    return upper_acceleration_x, upper_acceleration_y, acceleration_z


def roll_pitch_from_acceleration(acceleration_x, acceleration_y, acceleration_z):
    euler_x_rad = math.atan2(-acceleration_y, -acceleration_z)
    euler_y_rad = math.atan2(acceleration_x, math.sqrt(acceleration_y * acceleration_y + acceleration_z * acceleration_z))
    return euler_y_rad, euler_x_rad


def upper_body_attitude_from_body(body_roll, body_pitch, yaw_rad):
    body_acceleration = acceleration_from_body_euler(body_roll, body_pitch)
    upper_body_acceleration = rotate_acceleration_to_upper_body(*body_acceleration, yaw_rad)
    return roll_pitch_from_acceleration(*upper_body_acceleration)


def attitude_profile(time_sec, yaw_rad, scenario):
    body_roll, body_pitch = body_attitude_profile(time_sec, scenario)
    return upper_body_attitude_from_body(body_roll, body_pitch, yaw_rad)


def run_simulation(
    coeffs,
    machine,
    mpc,
    scenario,
):
    psi = 0.0
    u_prev = 0.0
    target_psi = math.radians(float(scenario["target_angle_deg"]))
    rows = []

    step_count = int(round(float(scenario["sim_time_sec"]) / mpc["dt"]))
    for step_index in range(step_count):
        control_time_sec = step_index * mpc["dt"]
        record_time_sec = (step_index + 1) * mpc["dt"]
        body_roll, body_pitch = body_attitude_profile(control_time_sec, scenario)
        roll, pitch = upper_body_attitude_from_body(body_roll, body_pitch, psi)
        moment_nm = gravity_yaw_moment(machine, roll, pitch)

        u_cmd, horizon, weights, cost = solve_mpc(psi, target_psi, u_prev, moment_nm, coeffs, mpc)
        omega = turn_omega_ss(coeffs, u_cmd, moment_nm)
        psi = psi + mpc["dt"] * omega
        settled = (
            abs(target_psi - psi) < math.radians(mpc["settle_angle_deg"])
            and abs(omega) < mpc["settle_omega_rad_s"]
        )

        rows.append(
            {
                "time_sec": record_time_sec,
                "psi_deg": math.degrees(psi),
                "target_psi_deg": math.degrees(target_psi),
                "angle_error_deg": math.degrees(target_psi - psi),
                "omega_rad_s": omega,
                "u_cmd": u_cmd,
                "roll_deg": math.degrees(roll),
                "pitch_deg": math.degrees(pitch),
                "body_roll_deg": math.degrees(body_roll),
                "body_pitch_deg": math.degrees(body_pitch),
                "moment_nm": moment_nm,
                "horizon": float(horizon),
                "mpc_cost": cost,
                "q_angle": weights["q_angle"],
                "q_terminal": weights["q_terminal"],
                "q_omega": weights["q_omega"],
                "r_u": weights["r_u"],
                "r_du": weights["r_du"],
                "settled": int(settled),
            }
        )

        u_prev = u_cmd

    return rows


def write_result_csv(rows, output_dir):
    csv_path = ensure_parent(output_dir / "csv" / "turn_mpc_result.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def plot_series(rows, output_dir, y_key, y_label, file_name):
    path = ensure_parent(output_dir / "plots" / file_name)
    times = np.array([row["time_sec"] for row in rows], dtype=float)
    values = np.array([row[y_key] for row in rows], dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(times, values, linewidth=2, label=y_key)
    plt.xlabel("time [s]")
    plt.ylabel(y_label)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_angle(rows, output_dir):
    path = ensure_parent(output_dir / "plots" / "angle.png")
    times = np.array([row["time_sec"] for row in rows], dtype=float)
    yaw = np.array([row["psi_deg"] for row in rows], dtype=float)
    target = np.array([row["target_psi_deg"] for row in rows], dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(times, yaw, linewidth=2, label="yaw")
    plt.plot(times, target, "--", linewidth=2, label="target")
    plt.xlabel("time [s]")
    plt.ylabel("angle [deg]")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def draw_machine(axis, yaw_rad, target_yaw_rad, title):
    # GIF用の簡易上面図。実機形状ではなく、向きが分かることを優先する。
    base = np.array([[-3.0, -1.8], [3.0, -1.8], [3.0, 1.8], [-3.0, 1.8]], dtype=float)
    upper = np.array([[-2.4, -1.2], [2.4, -1.2], [2.4, 1.2], [-2.4, 1.2]], dtype=float)
    arrow = np.array([[2.1, 0.0], [1.2, 0.45], [1.2, -0.45]], dtype=float)
    rotation = np.array(
        [[math.cos(yaw_rad), -math.sin(yaw_rad)], [math.sin(yaw_rad), math.cos(yaw_rad)]],
        dtype=float,
    )

    axis.add_patch(patches.Polygon(base, closed=True, facecolor="#d1d5db", edgecolor="#4b5563", linewidth=1.4))
    axis.add_patch(patches.Polygon(upper @ rotation.T, closed=True, facecolor="#facc15", edgecolor="#854d0e", linewidth=1.8))
    axis.add_patch(patches.Polygon(arrow @ rotation.T, closed=True, facecolor="#2563eb", edgecolor="#1e3a8a", linewidth=1.0))
    axis.plot([0.0, 3.3 * math.cos(target_yaw_rad)], [0.0, 3.3 * math.sin(target_yaw_rad)], "--", color="#dc2626", linewidth=1.5)
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.set_xlim(-4.0, 4.0)
    axis.set_ylim(-4.0, 4.0)
    axis.grid(alpha=0.18)


def save_turn_gif(rows, output_dir):
    gif_path = ensure_parent(output_dir / "gif" / "turn_mpc_animation.gif")
    target_yaw_rad = math.radians(rows[0]["target_psi_deg"])
    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)

    def update(frame_index):
        axis.clear()
        row = rows[frame_index]
        title = (
            f"t={row['time_sec']:.1f}s yaw={row['psi_deg']:.1f}deg\n"
            f"u={row['u_cmd']:.2f} omega={row['omega_rad_s']:.3f}rad/s"
        )
        draw_machine(axis, math.radians(row["psi_deg"]), target_yaw_rad, title)

    animation = FuncAnimation(figure, update, frames=len(rows), interval=80, repeat=True)
    animation.save(gif_path, writer=PillowWriter(fps=12))
    plt.close(figure)
    return gif_path


def rotation_x(angle_rad):
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle_rad), -math.sin(angle_rad)],
            [0.0, math.sin(angle_rad), math.cos(angle_rad)],
        ],
        dtype=float,
    )


def rotation_y(angle_rad):
    return np.array(
        [
            [math.cos(angle_rad), 0.0, math.sin(angle_rad)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle_rad), 0.0, math.cos(angle_rad)],
        ],
        dtype=float,
    )


def rotation_z(angle_rad):
    return np.array(
        [
            [math.cos(angle_rad), -math.sin(angle_rad), 0.0],
            [math.sin(angle_rad), math.cos(angle_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def transform_points(points, rotation):
    return np.asarray(points, dtype=float) @ rotation.T


def add_poly(axis, vertices, face_indices, color, edgecolor, alpha=1.0, linewidth=0.8):
    faces = [[vertices[index] for index in face] for face in face_indices]
    collection = Poly3DCollection(faces, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
    axis.add_collection3d(collection)
    return collection


def add_box(axis, rotation, length, width, height, bottom_z, color, edgecolor, alpha=1.0):
    x0 = -length / 2.0
    x1 = length / 2.0
    y0 = -width / 2.0
    y1 = width / 2.0
    z0 = bottom_z
    z1 = bottom_z + height
    points = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    return add_poly(axis, transform_points(points, rotation), faces, color, edgecolor, alpha=alpha)


def draw_tilted_turn_model(axis, row, target_yaw_rad):
    body_roll_rad = math.radians(float(row.get("body_roll_deg", row["roll_deg"])))
    body_pitch_rad = math.radians(float(row.get("body_pitch_deg", row["pitch_deg"])))
    yaw_rad = math.radians(float(row["psi_deg"]))
    body_rotation = rotation_y(body_roll_rad) @ rotation_x(body_pitch_rad)
    upper_rotation = body_rotation @ rotation_z(yaw_rad)
    target_rotation = body_rotation @ rotation_z(target_yaw_rad)

    plane_points = transform_points(
        np.array(
            [
                [-5.0, -4.0, 0.0],
                [5.0, -4.0, 0.0],
                [5.0, 4.0, 0.0],
                [-5.0, 4.0, 0.0],
            ],
            dtype=float,
        ),
        body_rotation,
    )
    add_poly(axis, plane_points, [[0, 1, 2, 3]], "#dbeafe", "#2563eb", alpha=0.36, linewidth=0.9)
    for grid_value in np.linspace(-4.0, 4.0, 5):
        x_line = transform_points(np.array([[-5.0, grid_value, 0.006], [5.0, grid_value, 0.006]], dtype=float), body_rotation)
        y_line = transform_points(np.array([[grid_value, -4.0, 0.006], [grid_value, 4.0, 0.006]], dtype=float), body_rotation)
        axis.plot(x_line[:, 0], x_line[:, 1], x_line[:, 2], color="#60a5fa", linewidth=0.7, alpha=0.8)
        axis.plot(y_line[:, 0], y_line[:, 1], y_line[:, 2], color="#60a5fa", linewidth=0.7, alpha=0.8)

    footprint = transform_points(
        np.array(
            [
                [-2.2, -1.05, 0.015],
                [2.2, -1.05, 0.015],
                [2.2, 1.05, 0.015],
                [-2.2, 1.05, 0.015],
                [-2.2, -1.05, 0.015],
            ],
            dtype=float,
        ),
        upper_rotation,
    )
    axis.plot(footprint[:, 0], footprint[:, 1], footprint[:, 2], color="#111827", linewidth=1.6)
    add_box(axis, upper_rotation, 4.4, 2.1, 0.6, 0.0, "#facc15", "#854d0e", alpha=0.96)
    add_box(axis, upper_rotation, 1.2, 1.7, 0.55, 0.6, "#0f766e", "#134e4a", alpha=0.96)

    origin = upper_rotation @ np.array([0.0, 0.0, 1.25])
    upper_forward = upper_rotation @ np.array([2.6, 0.0, 0.0])
    target_forward = target_rotation @ np.array([3.3, 0.0, 0.0])
    body_forward = body_rotation @ np.array([3.0, 0.0, 0.0])
    body_left = body_rotation @ np.array([0.0, 2.2, 0.0])

    axis.quiver(*origin, *upper_forward, color="#1d4ed8", linewidth=2.4, arrow_length_ratio=0.18)
    axis.plot([0.0, target_forward[0]], [0.0, target_forward[1]], [0.12, target_forward[2] + 0.12], "--", color="#dc2626", linewidth=1.6)
    axis.quiver(0.0, 0.0, 0.08, *body_forward, color="#16a34a", linewidth=1.5, arrow_length_ratio=0.12)
    axis.quiver(0.0, 0.0, 0.08, *body_left, color="#f97316", linewidth=1.5, arrow_length_ratio=0.12)
    axis.quiver(3.9, 3.9, 2.4, 0.0, 0.0, -1.7, color="#111827", linewidth=1.6, arrow_length_ratio=0.18)

    axis.text2D(
        0.03,
        0.96,
        (
            f"t={float(row['time_sec']):.1f}s yaw={float(row['psi_deg']):.1f}deg  u={float(row['u_cmd']):.2f}\n"
            f"body roll={math.degrees(body_roll_rad):.1f}deg body pitch={math.degrees(body_pitch_rad):.1f}deg\n"
            f"upper roll={float(row['roll_deg']):.1f}deg upper pitch={float(row['pitch_deg']):.1f}deg"
        ),
        transform=axis.transAxes,
        fontsize=9,
    )
    axis.text(3.95, 3.9, 0.5, "gravity", color="#111827", fontsize=8)
    axis.text(*(body_forward * 1.05 + np.array([0.0, 0.0, 0.08])), "vehicle x", color="#166534", fontsize=8)
    axis.text(*(body_left * 1.05 + np.array([0.0, 0.0, 0.08])), "vehicle y", color="#9a3412", fontsize=8)

    axis.set_xlim(-4.8, 4.8)
    axis.set_ylim(-4.8, 4.8)
    axis.set_zlim(-1.2, 2.6)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_box_aspect((1.0, 1.0, 0.48))
    axis.view_init(elev=24.0, azim=-55.0)
    axis.grid(alpha=0.22)


def save_turn_3d_gif(rows, output_dir):
    gif_path = ensure_parent(output_dir / "gif" / "turn_mpc_3d_tilt_animation.gif")
    target_yaw_rad = math.radians(float(rows[0]["target_psi_deg"]))
    max_frames = 140
    frame_indices = np.linspace(0, len(rows) - 1, min(len(rows), max_frames), dtype=int)
    figure = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")

    def update(frame_index):
        axis.clear()
        draw_tilted_turn_model(axis, rows[int(frame_indices[frame_index])], target_yaw_rad)

    animation = FuncAnimation(figure, update, frames=len(frame_indices), interval=90, repeat=True)
    animation.save(gif_path, writer=PillowWriter(fps=12))
    plt.close(figure)
    return gif_path


def show_turn_3d_plot(rows):
    if not rows:
        return
    target_yaw_rad = math.radians(float(rows[0]["target_psi_deg"]))
    times = np.array([float(row["time_sec"]) for row in rows], dtype=float)
    figure = plt.figure(figsize=(8.4, 7.2))
    axis = figure.add_subplot(111, projection="3d")
    figure.subplots_adjust(bottom=0.14)
    slider_axis = figure.add_axes([0.18, 0.045, 0.64, 0.028])
    time_slider = Slider(
        slider_axis,
        "time [s]",
        float(times[0]),
        float(times[-1]),
        valinit=float(times[0]),
        valstep=times,
    )

    def redraw(row_index):
        elev = axis.elev
        azim = axis.azim
        axis.clear()
        draw_tilted_turn_model(axis, rows[int(row_index)], target_yaw_rad)
        axis.view_init(elev=elev, azim=azim)
        figure.canvas.draw_idle()

    def on_slider_change(value):
        row_index = int(np.argmin(np.abs(times - float(value))))
        redraw(row_index)

    time_slider.on_changed(on_slider_change)
    redraw(0)
    print("3D plot: drag with the mouse to rotate, use the time slider to move through the simulation.")
    plt.show()


def save_outputs(rows, output_dir):
    # CSV、角度、角速度、レバー、姿勢、モーメント、ホライズン、GIFをまとめて出す。
    if not rows:
        return []
    return [
        write_result_csv(rows, output_dir),
        plot_angle(rows, output_dir),
        plot_series(rows, output_dir, "omega_rad_s", "yaw rate [rad/s]", "omega.png"),
        plot_series(rows, output_dir, "u_cmd", "lever command", "lever.png"),
        plot_series(rows, output_dir, "roll_deg", "roll angle [deg]", "roll.png"),
        plot_series(rows, output_dir, "pitch_deg", "pitch angle [deg]", "pitch.png"),
        plot_series(rows, output_dir, "moment_nm", "gravity moment [Nm]", "moment.png"),
        plot_series(rows, output_dir, "horizon", "prediction horizon", "horizon.png"),
        save_turn_3d_gif(rows, output_dir),
        # save_turn_gif(rows, output_dir),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="旋回モデルとMPC評価関数を1ファイルで確認する移植用サンプル")
    parser.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--load-state", default="empty")
    parser.add_argument("--target-angle-deg", type=float, default=90.0)
    parser.add_argument("--sim-time-sec", type=float, default=100.0)
    parser.add_argument("--roll-deg", type=float, default=2.0)
    parser.add_argument("--roll-amp-deg", type=float, default=4.0)
    parser.add_argument("--roll-rate-rad-s", type=float, default=0.10)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--pitch-amp-deg", type=float, default=2.0)
    parser.add_argument("--pitch-rate-rad-s", type=float, default=0.07)
    parser.add_argument("--show-3d", action="store_true", help="Matplotlibの3Dプロットを表示して視点を自由に回転する")
    return parser.parse_args()


def main():
    args = parse_args()

    # 重力モーメント M = m g (xcg sin(phi) + ycg sin(theta)) に使う機体定数。
    machine = {
        "mass_kg": 7785.0,
        "gravity_mps2": 9.80665,
        "x_cg_m": -0.587,
        "y_cg_m": 0.030,
    }

    # MPCの制約、予測ホライズン、評価関数Jの重み。移植先ではまずここを調整する。
    mpc = {
        "dt": 0.1,
        "u_min": -1.0,
        "u_max": 1.0,
        "du_max": 0.06,
        "u_grid_step": 0.02,
        "min_horizon": 8.0,
        "max_horizon": 50.0,
        "q_angle": 8.0,
        "q_terminal": 120.0,
        "q_omega": 0.8,
        "r_u": 0.03,
        "r_du": 3.0,
        "stop_band_rad": math.radians(6.0),
        "q_stop_omega": 18.0,
        "q_overshoot": 250.0,
        "q_reverse_error": 80.0,
        "moment_norm_nm": 30000.0,
        "moment_horizon_gain": 0.35,
        "moment_weight_gain": 2.0,
        "settle_angle_deg": 0.3,
        "settle_omega_rad_s": 0.005,
    }

    # 実機ではroll/pitchをセンサ値に置き換える。ここでは確認用の波形を使う。
    scenario = {
        "target_angle_deg": args.target_angle_deg,
        "sim_time_sec": args.sim_time_sec,
        "load_state": args.load_state,
        "roll_offset_deg": args.roll_deg,
        "roll_amp_deg": args.roll_amp_deg,
        "roll_rate_rad_s": args.roll_rate_rad_s,
        "pitch_offset_deg": args.pitch_deg,
        "pitch_amp_deg": args.pitch_amp_deg,
        "pitch_rate_rad_s": args.pitch_rate_rad_s,
    }

    coeffs = load_turn_model_coefficients(args.coefficients, str(scenario["load_state"]))
    rows = run_simulation(coeffs, machine, mpc, scenario)
    output_paths = save_outputs(rows, args.output_dir)

    print("Saved:")
    for path in output_paths:
        print(f"  {path}")

    if args.show_3d:
        show_turn_3d_plot(rows)


if __name__ == "__main__":
    main()