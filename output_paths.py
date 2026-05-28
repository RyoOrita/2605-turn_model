from __future__ import annotations

from pathlib import Path


OUTPUT_ROOT = Path("outputs")

EXTRACT_TARGET_TIMESERIES_OUTPUT_DIR = OUTPUT_ROOT / "extract_target_timeseries"
ANALYZE_TURN_TREND_OUTPUT_DIR = OUTPUT_ROOT / "analyze_turn_trend"
COMPARE_ROLL_PITCH_YAW_3D_OUTPUT_DIR = OUTPUT_ROOT / "compare_roll_pitch_yaw_3d"
BUILD_STATIC_ZERO_TILT_MODEL_OUTPUT_DIR = OUTPUT_ROOT / "build_static_zero_tilt_model"
BUILD_GENERALIZED_TURN_MODEL_OUTPUT_DIR = OUTPUT_ROOT / "build_generalized_turn_model"
TURN_MPC_STATIC_DEMO_OUTPUT_DIR = OUTPUT_ROOT / "turn_mpc_static_demo"
VALIDATE_TURN_MPC_CASES_OUTPUT_DIR = OUTPUT_ROOT / "validate_turn_mpc_cases"

TURN_MODEL_COEFFICIENTS_CSV = BUILD_GENERALIZED_TURN_MODEL_OUTPUT_DIR / "turn_model_coefficients.csv"


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_prefixed_output(output_prefix: str | Path, output_dir: Path) -> str:
    prefix_path = Path(output_prefix)
    if prefix_path.is_absolute() or prefix_path.parent != Path("."):
        return str(prefix_path)
    return str(output_dir / prefix_path)