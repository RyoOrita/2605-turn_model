from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


DEFAULT_COEFFICIENT_CSV = Path("出力/05_汎用旋回モデル/CSV/モデル係数/turn_model_coefficients.csv")


@dataclass(frozen=True)
class MachineParams:
    mass_kg: float = 7785.0
    gravity_mps2: float = 9.80665
    x_cg_m: float = -0.587
    y_cg_m: float = 0.030


@dataclass(frozen=True)
class TurnModelCoefficient:
    lever: float
    baseline_abs_yaw: float
    moment_gain_per_nm: float


@dataclass
class PIDGains:
    kp: float = 1.2
    ki: float = 0.0
    kd: float = 0.08
    u_min: float = -1.0
    u_max: float = 1.0
    du_max: float = 0.06
    integral_min: float = -1.0
    integral_max: float = 1.0


@dataclass
class PIDState:
    previous_error_rad: float = 0.0
    previous_u: float = 0.0
    integral_error_rad_s: float = 0.0
    initialized: bool = False


@dataclass(frozen=True)
class TurnControlOutput:
    u_cmd: float
    angle_error_rad: float
    moment_nm: float
    predicted_omega_rad_s: float


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def wrap_angle_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def gravity_yaw_moment(machine: MachineParams, roll_rad: float, pitch_rad: float) -> float:
    return machine.mass_kg * machine.gravity_mps2 * (
        machine.x_cg_m * math.sin(roll_rad) + machine.y_cg_m * math.sin(pitch_rad)
    )


class TurnModel:
    def __init__(self, coefficients: list[TurnModelCoefficient]):
        if not coefficients:
            raise ValueError("coefficients must not be empty")
        zero_row = TurnModelCoefficient(0.0, 0.0, 0.0)
        rows = [zero_row, *coefficients]
        unique_rows = {round(row.lever, 9): row for row in rows}
        self.coefficients = sorted(unique_rows.values(), key=lambda row: row.lever)

    @classmethod
    def from_csv(cls, coefficient_csv: Path, load_state: str = "empty") -> TurnModel:
        rows = list(csv.DictReader(coefficient_csv.open("r", encoding="utf-8", newline="")))
        if not rows:
            raise ValueError(f"coefficient CSV is empty: {coefficient_csv}")
        if "load_state" in rows[0]:
            filtered_rows = [row for row in rows if str(row.get("load_state", "")).lower() == load_state.lower()]
            if filtered_rows:
                rows = filtered_rows
        return cls(
            [
                TurnModelCoefficient(
                    lever=float(row["lever"]),
                    baseline_abs_yaw=float(row["baseline_abs_yaw"]),
                    moment_gain_per_nm=float(row["moment_gain_per_nm"]),
                )
                for row in rows
            ]
        )

    @classmethod
    def demo(cls) -> TurnModel:
        lever = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        baseline = [0.015, 0.030, 0.045, 0.065, 0.090, 0.130, 0.180, 0.270, 0.420, 0.480]
        gain = [1.0e-6, 1.4e-6, 1.8e-6, 2.0e-6, 2.2e-6, 2.4e-6, 1.8e-6, 1.2e-6, 0.5e-6, 0.3e-6]
        return cls(
            [
                TurnModelCoefficient(lever_value, baseline_value, gain_value)
                for lever_value, baseline_value, gain_value in zip(lever, baseline, gain, strict=True)
            ]
        )

    def _interp(self, lever_abs: float, values: list[float]) -> float:
        x = clamp(abs(lever_abs), 0.0, 1.0)
        levers = [row.lever for row in self.coefficients]
        if x <= levers[0]:
            return values[0]
        if x >= levers[-1]:
            return values[-1]
        for index in range(1, len(levers)):
            left = levers[index - 1]
            right = levers[index]
            if left <= x <= right:
                ratio = (x - left) / (right - left)
                return values[index - 1] + ratio * (values[index] - values[index - 1])
        return values[-1]

    def omega0(self, lever_abs: float) -> float:
        return self._interp(lever_abs, [row.baseline_abs_yaw for row in self.coefficients])

    def moment_gain(self, lever_abs: float) -> float:
        return self._interp(lever_abs, [row.moment_gain_per_nm for row in self.coefficients])

    def omega_ss_from_moment(self, u: float, moment_nm: float) -> float:
        lever_abs = abs(float(u))
        lever_component = math.copysign(self.omega0(lever_abs), float(u))
        moment_component = self.moment_gain(lever_abs) * moment_nm
        return lever_component + moment_component

    def omega_ss(self, u: float, roll_rad: float, pitch_rad: float, machine: MachineParams) -> float:
        moment_nm = gravity_yaw_moment(machine, roll_rad, pitch_rad)
        return self.omega_ss_from_moment(u, moment_nm)


class TurnPIDController:
    def __init__(self, model: TurnModel, machine: MachineParams, gains: PIDGains):
        self.model = model
        self.machine = machine
        self.gains = gains
        self.state = PIDState()

    def reset(self) -> None:
        self.state = PIDState()

    def step(
        self,
        current_yaw_rad: float,
        target_yaw_rad: float,
        roll_rad: float,
        pitch_rad: float,
        dt: float,
    ) -> TurnControlOutput:
        error = wrap_angle_rad(target_yaw_rad - current_yaw_rad)
        if self.state.initialized:
            derivative = (error - self.state.previous_error_rad) / dt
        else:
            derivative = 0.0
            self.state.initialized = True

        self.state.integral_error_rad_s = clamp(
            self.state.integral_error_rad_s + error * dt,
            self.gains.integral_min,
            self.gains.integral_max,
        )

        raw_u = (
            self.gains.kp * error
            + self.gains.ki * self.state.integral_error_rad_s
            + self.gains.kd * derivative
        )
        rate_limited_u = clamp(
            raw_u,
            self.state.previous_u - self.gains.du_max,
            self.state.previous_u + self.gains.du_max,
        )
        u_cmd = clamp(rate_limited_u, self.gains.u_min, self.gains.u_max)

        moment_nm = gravity_yaw_moment(self.machine, roll_rad, pitch_rad)
        predicted_omega = self.model.omega_ss_from_moment(u_cmd, moment_nm)

        self.state.previous_error_rad = error
        self.state.previous_u = u_cmd

        return TurnControlOutput(
            u_cmd=u_cmd,
            angle_error_rad=error,
            moment_nm=moment_nm,
            predicted_omega_rad_s=predicted_omega,
        )


def load_model(coefficient_csv: Path, load_state: str) -> TurnModel:
    if coefficient_csv.exists():
        return TurnModel.from_csv(coefficient_csv, load_state=load_state)
    return TurnModel.demo()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable turn model and PID-control reference implementation.")
    parser.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENT_CSV)
    parser.add_argument("--load-state", default="empty")
    parser.add_argument("--target-angle-deg", type=float, default=90.0)
    parser.add_argument("--current-angle-deg", type=float, default=0.0)
    parser.add_argument("--roll-deg", type=float, default=2.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    machine = MachineParams()
    model = load_model(args.coefficients, args.load_state)
    controller = TurnPIDController(model, machine, PIDGains())
    output = controller.step(
        current_yaw_rad=math.radians(args.current_angle_deg),
        target_yaw_rad=math.radians(args.target_angle_deg),
        roll_rad=math.radians(args.roll_deg),
        pitch_rad=math.radians(args.pitch_deg),
        dt=args.dt,
    )
    print(f"u_cmd={output.u_cmd:.6f}")
    print(f"angle_error_rad={output.angle_error_rad:.6f}")
    print(f"moment_nm={output.moment_nm:.6f}")
    print(f"predicted_omega_rad_s={output.predicted_omega_rad_s:.6f}")


if __name__ == "__main__":
    main()