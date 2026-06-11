# プロット計算式

## 共通

- 横軸: `t = (i + 1) dt` [s]
- 車体ロール角: `phi_body(t) = roll_offset + roll_amp * sin(roll_rate * t)` [rad]
- 車体ピッチ角: `theta_body(t) = pitch_offset + pitch_amp * cos(pitch_rate * t)` [rad]
- 車体加速度方向: `a_body = [sin(phi_body), -cos(phi_body) sin(theta_body), -cos(phi_body) cos(theta_body)]`
- 上部旋回体加速度方向: `a_upper = Rz(-psi) a_body`
- ロール角: `phi = atan2(a_upper_x, sqrt(a_upper_y^2 + a_upper_z^2))` [rad]
- ピッチ角: `theta = atan2(-a_upper_y, -a_upper_z)` [rad]
- 重力モーメント: `M = m g (x_cg sin(phi) + y_cg sin(theta))` [Nm]
- 旋回角速度: `omega = sign(u) * omega0(|u|) + k(|u|) * M` [rad/s]
- 旋回角: `psi_next = psi + dt * omega` [rad]
- 角度誤差: `e = psi_target - psi` [rad]

## 出力グラフ

- `angle.png`: `x = t`, `y = rad2deg(psi)`, `target = rad2deg(psi_target)`
- `omega.png`: `x = t`, `y = omega`
- `lever.png`: `x = t`, `y = u`
- `roll.png`: `x = t`, `y = rad2deg(phi)`
- `pitch.png`: `x = t`, `y = rad2deg(theta)`
- `moment.png`: `x = t`, `y = M`
- `horizon.png`: `x = t`, `y = H`
- `turn_mpc_3d_tilt_animation.gif`: 傾斜地面と上部旋回体の3D GIF

## 3D表示

GIFを保存したうえで、Matplotlibの3Dプロットを表示して視点を自由に回転するには `--show-3d` を付けます。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-linux/bin/python pid_mpc_porting_reference/turn_model_mpc_reference.py --roll-deg 2 --roll-amp-deg 0 --pitch-deg 0 --pitch-amp-deg 0 --sim-time-sec 30 --show-3d
```

表示されたウィンドウでは、マウスドラッグで視点を回転し、下部のスライダーで時刻を変更できます。

## ホライズン

- `angle_scale = clamp(|e| / 90deg, 0, 1)`
- `moment_scale = clamp(|M| / moment_norm_nm, 0, 1)`
- `ratio = (1 - moment_horizon_gain) * angle_scale + moment_horizon_gain * moment_scale`
- `H = round(min_horizon + (max_horizon - min_horizon) * ratio)`

## 変数定義

- `t`: 時刻 [s]
- `dt`: 制御周期 [s]
- `phi_body`: 絶対座標系で一定とみなす車体ロール角 [rad]
- `theta_body`: 絶対座標系で一定とみなす車体ピッチ角 [rad]
- `phi`: 上部旋回体座標へ変換したロール角 [rad]
- `theta`: 上部旋回体座標へ変換したピッチ角 [rad]
- `m`: 機体質量 [kg]
- `g`: 重力加速度 [m/s^2]
- `x_cg`: 重心x位置 [m]
- `y_cg`: 重心y位置 [m]
- `M`: 重力モーメント [Nm]
- `u`: レバー指令 [-1から1]
- `omega0(|u|)`: レバー量だけで決まる基準旋回角速度 [rad/s]
- `k(|u|)`: 重力モーメント感度 [rad/s/Nm]
- `omega`: 旋回角速度 [rad/s]
- `psi`: 旋回角 [rad]
- `psi_target`: 目標旋回角 [rad]
- `e`: 角度誤差 [rad]
- `H`: MPC予測ホライズン [step]