from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from output_paths import VALIDATE_TURN_MPC_CASES_CASES_DIR, VALIDATE_TURN_MPC_CASES_SUMMARY_CSV_DIR, ensure_parent_dir
from turn_mpc_static_demo import ScenarioParams, run_simulation, save_simulation_outputs


OUTPUT_CSV = VALIDATE_TURN_MPC_CASES_SUMMARY_CSV_DIR / "turn_mpc_case_validation.csv"
OUTPUT_ROOT_DIR = VALIDATE_TURN_MPC_CASES_CASES_DIR

STRICT_STOP_ERROR_DEG = 0.3
STRICT_STOP_OMEGA_RAD_S = 0.005
PRACTICAL_STOP_ERROR_DEG = 0.5
PRACTICAL_STOP_OMEGA_RAD_S = 0.01
PRACTICAL_STOP_U = 0.04
MAX_ACCEPTABLE_OVERSHOOT_DEG = 1.0


@dataclass(frozen=True)
class TestCase:
    name: str
    target_angle_deg: float
    sim_time: float
    roll_offset_deg: float
    roll_amp_deg: float
    roll_rate_rad_s: float
    pitch_offset_deg: float
    pitch_amp_deg: float
    pitch_rate_rad_s: float
    notes: str

    def scenario(self, output_root_dir: Path) -> ScenarioParams:
        case_dir = output_root_dir / self.name
        return ScenarioParams(
            target_angle_deg=self.target_angle_deg,
            sim_time=self.sim_time,
            load_state="empty",
            roll_offset_deg=self.roll_offset_deg,
            roll_amp_deg=self.roll_amp_deg,
            roll_rate_rad_s=self.roll_rate_rad_s,
            pitch_offset_deg=self.pitch_offset_deg,
            pitch_amp_deg=self.pitch_amp_deg,
            pitch_rate_rad_s=self.pitch_rate_rad_s,
            output_prefix=str(case_dir / self.name),
        )


def build_test_cases() -> list[TestCase]:
    return [
        TestCase("flat_30", 30.0, 30.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.07, "小角度・平坦"),
        TestCase("flat_90", 90.0, 40.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.07, "代表角度・平坦"),
        TestCase("flat_150", 150.0, 55.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.07, "大角度・平坦"),
        TestCase("flat_neg_60", -60.0, 35.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.07, "逆旋回・平坦"),
        TestCase("flat_neg_120", -120.0, 50.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.07, "逆旋回・大角度"),
        TestCase("roll_pos_90", 90.0, 45.0, 6.0, 0.0, 0.10, 0.0, 0.0, 0.07, "正ロール定常"),
        TestCase("roll_neg_90", 90.0, 45.0, -6.0, 0.0, 0.10, 0.0, 0.0, 0.07, "負ロール定常"),
        TestCase("pitch_pos_90", 90.0, 40.0, 0.0, 0.0, 0.10, 4.0, 0.0, 0.07, "正ピッチ定常"),
        TestCase("pitch_neg_90", 90.0, 40.0, 0.0, 0.0, 0.10, -4.0, 0.0, 0.07, "負ピッチ定常"),
        TestCase("dynamic_roll_90", 90.0, 40.0, 2.0, 4.0, 0.10, 0.0, 0.0, 0.07, "実機寄りロール変動"),
        TestCase("dynamic_pitch_90", 90.0, 40.0, 0.0, 0.0, 0.10, 1.0, 3.0, 0.10, "ピッチ変動"),
        TestCase("combined_120", 120.0, 50.0, 5.0, 2.0, 0.10, 3.0, 1.0, 0.07, "ロール・ピッチ複合"),
        TestCase("combined_neg_120", -120.0, 50.0, -5.0, 2.0, 0.10, -3.0, 1.0, 0.07, "逆旋回・複合"),
        TestCase("dynamic_combined_90", 90.0, 45.0, 3.0, 5.0, 0.12, 1.0, 3.0, 0.09, "複合変動"),
        TestCase("counter_bias_90", 90.0, 45.0, 5.0, 0.0, 0.10, -3.0, 0.0, 0.07, "符号逆向きバイアス"),
        TestCase("counter_bias_neg_90", -90.0, 45.0, -5.0, 0.0, 0.10, 3.0, 0.0, 0.07, "逆旋回・符号逆向きバイアス"),
    ]


def max_signed_overshoot_deg(rows: list[dict[str, float | bool]]) -> float:
    target = float(rows[0]["target_psi_deg"])
    direction = 1.0 if target >= 0.0 else -1.0
    overshoots = []
    for row in rows:
        signed_progress = direction * float(row["psi_deg"])
        signed_target = direction * float(row["target_psi_deg"])
        overshoots.append(max(signed_progress - signed_target, 0.0))
    return max(overshoots) if overshoots else 0.0


def find_first_stop_time(
    rows: list[dict[str, float | bool]],
    error_threshold_deg: float,
    omega_threshold_rad_s: float,
    u_threshold: float | None = None,
) -> float | None:
    for row in rows:
        angle_error_deg = abs(float(row["target_psi_deg"]) - float(row["psi_deg"]))
        omega = abs(float(row["omega_rad_s"]))
        u_cmd = abs(float(row["u_cmd"]))
        if angle_error_deg <= error_threshold_deg and omega <= omega_threshold_rad_s:
            if u_threshold is None or u_cmd <= u_threshold:
                return float(row["time"])
    return None


def write_case_summary(case_dir: Path, summary_row: dict[str, object]) -> Path:
    summary_path = ensure_parent_dir(case_dir / "CSV" / "判定要約" / f"{case_dir.name}_summary.csv")
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)
    return summary_path


def evaluate_case(test_case: TestCase, output_root_dir: Path) -> dict[str, object]:
    scenario = test_case.scenario(output_root_dir)
    case_dir = Path(scenario.output_prefix).parent
    case_dir.mkdir(parents=True, exist_ok=True)

    rows = run_simulation(scenario, save_outputs=False)
    final_row = rows[-1]
    final_error_deg = abs(float(final_row["target_psi_deg"]) - float(final_row["psi_deg"]))
    final_omega = abs(float(final_row["omega_rad_s"]))
    final_u = abs(float(final_row["u_cmd"]))
    overshoot_deg = max_signed_overshoot_deg(rows)
    strict_stop_time = find_first_stop_time(rows, STRICT_STOP_ERROR_DEG, STRICT_STOP_OMEGA_RAD_S)
    practical_stop_time = find_first_stop_time(
        rows,
        PRACTICAL_STOP_ERROR_DEG,
        PRACTICAL_STOP_OMEGA_RAD_S,
        PRACTICAL_STOP_U,
    )
    timed_out = len(rows) >= int(test_case.sim_time / 0.1)
    practical_ok = (
        practical_stop_time is not None
        and final_error_deg <= PRACTICAL_STOP_ERROR_DEG
        and final_omega <= PRACTICAL_STOP_OMEGA_RAD_S
        and final_u <= PRACTICAL_STOP_U
        and overshoot_deg <= MAX_ACCEPTABLE_OVERSHOOT_DEG
    )

    saved_files = save_simulation_outputs(rows, scenario)
    summary_row = {
        **asdict(test_case),
        "steps": len(rows),
        "timed_out": timed_out,
        "strict_stop_achieved": strict_stop_time is not None,
        "strict_stop_time_s": "" if strict_stop_time is None else round(strict_stop_time, 3),
        "practical_stop_achieved": practical_stop_time is not None,
        "practical_stop_time_s": "" if practical_stop_time is None else round(practical_stop_time, 3),
        "final_angle_error_deg": round(final_error_deg, 4),
        "final_omega_rad_s": round(final_omega, 6),
        "final_u_cmd": round(final_u, 4),
        "max_overshoot_deg": round(overshoot_deg, 4),
        "overall_pass": practical_ok,
        "case_output_dir": str(case_dir),
        "saved_file_count": len(saved_files),
    }
    summary_path = write_case_summary(case_dir, summary_row)
    summary_row["case_summary_csv"] = str(summary_path)
    return summary_row


def write_results(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with ensure_parent_dir(OUTPUT_CSV).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    passed = sum(1 for row in rows if bool(row["overall_pass"]))
    strict = sum(1 for row in rows if bool(row["strict_stop_achieved"]))
    practical = sum(1 for row in rows if bool(row["practical_stop_achieved"]))
    print(f"cases={len(rows)} strict_stop={strict} practical_stop={practical} overall_pass={passed}")
    for row in rows:
        print(
            f"{row['name']}: pass={row['overall_pass']} final_err_deg={row['final_angle_error_deg']} "
            f"final_omega={row['final_omega_rad_s']} overshoot_deg={row['max_overshoot_deg']} "
            f"practical_stop_time_s={row['practical_stop_time_s']}"
        )


def main() -> None:
    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [evaluate_case(test_case, OUTPUT_ROOT_DIR) for test_case in build_test_cases()]
    write_results(rows)
    print_summary(rows)
    print(f"saved={OUTPUT_CSV}")
    print(f"case_output_root={OUTPUT_ROOT_DIR}")


if __name__ == "__main__":
    main()