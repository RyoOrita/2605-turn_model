# turn_model how to

## 目的

このフォルダは、MCAP ログから lever / roll / pitch / yaw を抽出し、傾向分析、汎用旋回モデル構築、簡易 MPC シミュレーション、複数ケース検証までを一通り行うためのものです。

現状の旋回モデルと MPC は、実機ログから作成した係数を使っています。
ただし、いまのモデルは実機そのものの全動特性ではなく、実機ログから同定した定常近似モデルです。

## 実機ログ由来の旋回モデル

現在の旋回モデルは、レバー入力だけでなく、上部旋回体の重心とロール・ピッチによる重力モーメントも含めて次の形で整理しています。

$$
\omega = \operatorname{sign}(u)\,\omega_0(|u|, \lambda) + k(|u|, \lambda) mg\left(x_{cg}(\lambda)\sin\phi + y_{cg}(\lambda)\sin\theta\right)
$$

ここで、

- $u$: レバー量。現在の MPC では $-1.0 \le u \le 1.0$
- $\lambda$: 積荷状態。現在は empty のみ
- $\omega$: 予測する旋回角速度
- $\omega_0(|u|, \lambda)$: 姿勢影響がないときの基準旋回速度
- $k(|u|, \lambda)$: 重力モーメントの影響ゲイン
- $m$: 上部旋回体の質量
- $x_{cg}(\lambda)$: 前後方向の重心位置
- $y_{cg}(\lambda)$: 左右方向の重心位置
- $\phi$: ロール角
- $\theta$: ピッチ角

を表します。

実装上は

$$
M = mg\left(x_{cg}(\lambda)\sin\phi + y_{cg}(\lambda)\sin\theta\right)
$$

を重力モーメントとして計算し、各レバー段ごとに

$$
\omega \approx \operatorname{sign}(u)\,\omega_0(|u|, \lambda) + k(|u|, \lambda) M
$$

として使っています。

現状のモデル係数は [outputs/build_generalized_turn_model/turn_model_coefficients.csv](outputs/build_generalized_turn_model/turn_model_coefficients.csv) に出力されます。

- `baseline_abs_yaw`: $\omega_0(|u|, \lambda)$
- `moment_gain_per_nm`: $k(|u|, \lambda)$

この係数は、実機ログ由来の [lever_01-05.mcap](lever_01-05.mcap) と [lever_05-10.mcap](lever_05-10.mcap) から生成した学習データに基づいています。現在は空荷 `empty` のみです。

なお、現時点では 0.4 から 1.0 を学習対象のベース領域とし、0.1 から 0.3 は 0.4 の係数から線形補間して滑らかにつないでいます。

## MPC の定式化

### 状態更新

MPC デモでは一次遅れや慣性 $J$ は入れず、定常モデルを使って以下で更新します。

$$
\psi_{k+1} = \psi_k + dt\,\omega_k
$$

ここで、

- $\psi$: 旋回角
- $dt = 0.1$ s
- $\omega_k$: 上の旋回モデルで求めた角速度

です。

### 予測ホライズン

予測ホライズンは固定ではなく、残り角度と重力モーメントの大きさで可変にしています。

$$
\\text{angle\_scale} = \operatorname{clip}\left(\frac{|\psi_{target}-\psi|}{90^\circ}, 0, 1\right)
$$

$$
\\text{m\_scale} = \operatorname{clip}\left(\frac{|M|}{M_{norm}}, 0, 1\right)
$$

$$
H = \operatorname{clip}\left(\operatorname{round}\left(H_{min} + (H_{max}-H_{min})\left[(1-a)\,\text{angle\_scale} + a\,\text{m\_scale}\right]\right), H_{min}, H_{max}\right)
$$

現在値は以下です。

- $H_{min} = 8$
- $H_{max} = 50$
- $a = 0.35$
- $M_{norm} = 30000$ Nm

### 評価関数

MPC のコストは、角度誤差、角速度、レバー量、レバー変化率、オーバーシュート対策、終端誤差で構成しています。

概念的には次の形です。

$$
J = \sum_{i=0}^{H-1}\left(
q_{angle,i} e_i^2 + q_{\omega,i} \omega_i^2 + r_u u_i^2 + r_{du}(u_i-u_{i-1})^2
\right)
+ q_{terminal} e_H^2
+ J_{overshoot}
$$

ここで、

- $e_i = \psi_{target} - \psi_i$
- 目標近傍では $q_{\omega}$ を強める
- 重力モーメントが大きいときは $q_{\omega}$, $r_{du}$, $q_{terminal}$ を強める
- 目標通過後はオーバーシュート罰則を加える

としています。

### 制約と探索方法

現在の制約は以下です。

- $-1.0 \le u \le 1.0$
- $|\Delta u| \le 0.06$
- レバー探索刻みは 0.02

最適化は連続最適化ではなく、離散候補の総当たりです。
ただし 1 ステップだけのホールド評価ではなく、ホライズン内で目標レバーへ向かうランプ列を評価しています。

これにより、不利モーメント下で 0 近傍の小レバーでは回り出せないケースでも、将来の踏み増しを見込んで最初の一手を選べます。

## 前提

- Windows PowerShell を使う
- 仮想環境 .venv が作成済みで、必要な Python パッケージが入っている
- 入力 MCAP は同じフォルダに置く

## 実行手順

### 1. 仮想環境を有効化する

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& ".\.venv\Scripts\Activate.ps1")
```

### 2. MCAP から対象信号を抽出する

```powershell
python .\extract_target_timeseries.py .\lever_01-05.mcap
python .\extract_target_timeseries.py .\lever_05-10.mcap
```

出力は [outputs/extract_target_timeseries](outputs/extract_target_timeseries) に保存されます。

### 3. 各データセットの傾向解析を行う

```powershell
python .\analyze_turn_trend.py lever_01-05
python .\analyze_turn_trend.py lever_05-10
```

出力は [outputs/analyze_turn_trend](outputs/analyze_turn_trend) に保存されます。

### 4. 2 つの 3D 図を 1 枚の比較図にまとめる

```powershell
python .\compare_roll_pitch_yaw_3d.py
```

出力は [outputs/compare_roll_pitch_yaw_3d](outputs/compare_roll_pitch_yaw_3d) に保存されます。

### 5. 必要なら static zero-tilt model を作る

```powershell
python .\build_static_zero_tilt_model.py
```

出力は [outputs/build_static_zero_tilt_model](outputs/build_static_zero_tilt_model) に保存されます。

### 6. 汎用旋回モデルを作る

```powershell
python .\build_generalized_turn_model.py
```

出力は [outputs/build_generalized_turn_model](outputs/build_generalized_turn_model) に保存されます。

### 7. GUI で簡易 MPC シミュレータを使う

```powershell
python .\turn_mpc_static_demo.py --gui
```

フォーム上で以下を入力して実行できます。

- 目標角度 [deg]
- シミュレーション時間 [s]
- 積荷状態
- ロールのオフセット、振幅、角周波数
- ピッチのオフセット、振幅、角周波数
- 出力プレフィックス

GUI では、

- 実行: シミュレーション計算と画面内ライブ再生
- 出力保存: CSV / PNG / GIF の保存

を分けています。

### 8. CLI で簡易 MPC シミュレータを使う

```powershell
python .\turn_mpc_static_demo.py --target-angle-deg 60 --roll-deg 3 --roll-amp-deg 0 --pitch-deg 1 --pitch-amp-deg 0 --output-prefix turn_mpc_test_60deg
```

`--output-prefix` にファイル名だけを指定した場合、出力は [outputs/turn_mpc_static_demo](outputs/turn_mpc_static_demo) に保存されます。ディレクトリ付きのパスを指定した場合は、そのパスをそのまま使います。

## 複数ケース検証

複数ケースをまとめて検証するには以下を実行します。

```powershell
python .\validate_turn_mpc_cases.py
```

このスクリプトでは、

- 平坦路での小角度 / 中角度 / 大角度
- 正負の旋回方向
- 正負ロールバイアス
- 正負ピッチバイアス
- ロール変動
- ピッチ変動
- ロール・ピッチ複合
- 符号逆向きバイアス

を含む 16 ケースを一括評価します。

評価は 2 段です。

- strict stop: 元の終了条件に到達したか
- practical stop: 実用上停止と見なせるか

### ケース別出力

各ケースの出力は [outputs/validate_turn_mpc_cases/turn_mpc_case_runs](outputs/validate_turn_mpc_cases/turn_mpc_case_runs) 配下にケースごとのフォルダで保存されます。

例:

- [outputs/validate_turn_mpc_cases/turn_mpc_case_runs/flat_30](outputs/validate_turn_mpc_cases/turn_mpc_case_runs/flat_30)
- [outputs/validate_turn_mpc_cases/turn_mpc_case_runs/flat_90](outputs/validate_turn_mpc_cases/turn_mpc_case_runs/flat_90)

各ケースフォルダには以下が入ります。

- `*_result.csv`
- `*_angle.png`
- `*_omega.png`
- `*_input.png`
- `*_moment.png`
- `*_horizon.png`
- `*_storyboard.png`
- `*_animation.gif`
- `*_summary.csv`

ケース横断の要約は [outputs/validate_turn_mpc_cases/turn_mpc_case_validation.csv](outputs/validate_turn_mpc_cases/turn_mpc_case_validation.csv) に保存されます。

### 現時点の検証結果

現時点の一括検証結果は次の通りです。

- strict stop: 16 / 16
- practical stop: 15 / 16
- overall pass: 15 / 16

残っている 1 ケースは `roll_neg_90` です。
このケースも角度誤差と角速度は十分小さく、オーバーシュートも 0 ですが、最終レバー量がまだやや大きく、実用停止判定の `u` 閾値を満たしていません。

## 主な入力

- [lever_01-05.mcap](lever_01-05.mcap)
- [lever_05-10.mcap](lever_05-10.mcap)
- [turn_model_datasets.csv](turn_model_datasets.csv)

## 出力フォルダ

- [outputs/extract_target_timeseries](outputs/extract_target_timeseries)
- [outputs/analyze_turn_trend](outputs/analyze_turn_trend)
- [outputs/compare_roll_pitch_yaw_3d](outputs/compare_roll_pitch_yaw_3d)
- [outputs/build_static_zero_tilt_model](outputs/build_static_zero_tilt_model)
- [outputs/build_generalized_turn_model](outputs/build_generalized_turn_model)
- [outputs/turn_mpc_static_demo](outputs/turn_mpc_static_demo)
- [outputs/validate_turn_mpc_cases](outputs/validate_turn_mpc_cases)

## 主な出力

- [outputs/extract_target_timeseries/lever_01-05_target_timeseries.csv](outputs/extract_target_timeseries/lever_01-05_target_timeseries.csv)
- [outputs/extract_target_timeseries/lever_01-05_target_timeseries_wide.csv](outputs/extract_target_timeseries/lever_01-05_target_timeseries_wide.csv)
- [outputs/analyze_turn_trend/lever_01-05_turn_trend_summary.csv](outputs/analyze_turn_trend/lever_01-05_turn_trend_summary.csv)
- [outputs/analyze_turn_trend/lever_01-05_constant_lever_segments.csv](outputs/analyze_turn_trend/lever_01-05_constant_lever_segments.csv)
- [outputs/extract_target_timeseries/lever_05-10_target_timeseries.csv](outputs/extract_target_timeseries/lever_05-10_target_timeseries.csv)
- [outputs/extract_target_timeseries/lever_05-10_target_timeseries_wide.csv](outputs/extract_target_timeseries/lever_05-10_target_timeseries_wide.csv)
- [outputs/analyze_turn_trend/lever_05-10_turn_trend_summary.csv](outputs/analyze_turn_trend/lever_05-10_turn_trend_summary.csv)
- [outputs/analyze_turn_trend/lever_05-10_constant_lever_segments.csv](outputs/analyze_turn_trend/lever_05-10_constant_lever_segments.csv)
- [outputs/compare_roll_pitch_yaw_3d/lever_01-05_vs_05-10_roll_pitch_yaw_3d.png](outputs/compare_roll_pitch_yaw_3d/lever_01-05_vs_05-10_roll_pitch_yaw_3d.png)
- [outputs/build_static_zero_tilt_model/lever_05-10_static_zero_tilt_model.csv](outputs/build_static_zero_tilt_model/lever_05-10_static_zero_tilt_model.csv)
- [outputs/build_generalized_turn_model/turn_model_training_samples.csv](outputs/build_generalized_turn_model/turn_model_training_samples.csv)
- [outputs/build_generalized_turn_model/turn_model_coefficients.csv](outputs/build_generalized_turn_model/turn_model_coefficients.csv)
- [outputs/build_generalized_turn_model/turn_model_grid.csv](outputs/build_generalized_turn_model/turn_model_grid.csv)
- [outputs/build_generalized_turn_model/turn_model_overview.png](outputs/build_generalized_turn_model/turn_model_overview.png)
- [outputs/build_generalized_turn_model/turn_model_validation_summary.csv](outputs/build_generalized_turn_model/turn_model_validation_summary.csv)
- [outputs/build_generalized_turn_model/turn_model_validation.png](outputs/build_generalized_turn_model/turn_model_validation.png)
- [outputs/validate_turn_mpc_cases/turn_mpc_case_validation.csv](outputs/validate_turn_mpc_cases/turn_mpc_case_validation.csv)
- [outputs/validate_turn_mpc_cases/turn_mpc_case_runs](outputs/validate_turn_mpc_cases/turn_mpc_case_runs)

## 出力の見方

- `target_timeseries_wide.csv`: 時刻ごとの lever / yaw_rate / roll_angle / pitch_angle
- `turn_trend_summary.csv`: 全体相関と回帰の要約
- `constant_lever_segments.csv`: レバー一定区間ごとの統計
- `lever_01-05_vs_05-10_roll_pitch_yaw_3d.png`: 0.1 刻みのレバー値で左右比較した 3D 図
- `turn_model_coefficients.csv`: load_state ごとの汎用旋回モデル係数
- `turn_model_grid.csv`: レバーごとの baseline yaw と moment gain の一覧
- `turn_model_overview.png`: baseline yaw と moment gain の可視化
- `turn_model_moment_vs_abs_yaw.png`: 重力モーメントと絶対旋回速度の散布図
- `turn_model_moment_vs_abs_yaw_by_lever.png`: レバー量ごとに分け、縦軸を統一した重力モーメントと絶対旋回速度の散布図
- `turn_model_validation_summary.csv`: 実測 vs 予測の妥当性指標
- `turn_model_validation.png`: 実測 vs 予測と残差の可視化
- `*_result.csv`: MPC の時系列出力
- `*_horizon.png`: 可変ホライズンの推移
- `*_storyboard.png`: 旋回の代表コマ
- `*_animation.gif`: 旋回アニメーション
- `*_summary.csv`: ケースごとの停止判定要約

## 追加メモ

- 出力は実行ファイルごとの [outputs](outputs) 配下に分けて保存される
- 3D 比較図は z 軸を全サブプロットで統一している
- データが存在しないレバー値は no data と表示される
- `analyze_turn_trend.py` 単体でも各系列の 3D 図を出力する
- `turn_model_datasets.csv` に積荷ログを追加すれば、同じスクリプトで load_state ごとのモデルを再推定できる
- 現在の汎用旋回モデルは空荷データのみで推定している
- `turn_mpc_static_demo.py` は GUI と CLI の両方で実行できる
- 現在は符号付きレバーと符号付き旋回速度に対応している
- 逆方向ケースの改善のため、ホライズン内で目標レバーに向かうランプ列を評価している
