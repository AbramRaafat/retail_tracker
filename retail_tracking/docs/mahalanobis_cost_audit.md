# Mahalanobis Cost Audit

## Current Best Baseline

The baseline to protect is TrackTrack + YOLO11-JDE with the MOT20-like original-pool config and no relaxed detections:

- MOTA: 75.36
- IDF1: 67.39
- HOTA: 56.97
- DetA: 62.80
- AssA: 51.97
- Precision: 91.82
- Recall: 83.09
- FP: 11451
- FN: 26169
- IDs: 511
- FM: 1470
- FPS: about 28.49

Use `retail_tracking/configs/tracktrack_mot20_static_best.yaml` for this baseline. It keeps `cost_mode: static`, `relaxed_association_mode: original_pool`, `match_thr: 0.55`, `init_thr: 0.40`, and the existing MOT20-like settings.

## Static Cost

TrackTrack's static association cost is still:

```text
0.50 * iou_dist
+ 0.50 * cosine_distance
+ 0.10 * confidence_distance
+ 0.05 * angle_distance
```

After that, TrackTrack applies low/deleted penalties, the hard IoU gate (`iou_sim <= 0.10`), clipping to `[0, 1]`, and the existing iterative mutual-nearest matching loop.

`cost_mode: static` preserves this behavior.

## Mahalanobis Computation

Mahalanobis distance is computed in Kalman measurement space using each track's existing state:

```text
z = detection.cxcywh
projected_mean, projected_cov = track.kalman_filter.project(track.mean, track.covariance, detection.score)
y = z - projected_mean
mahalanobis_squared = y.T * inv(projected_cov) * y
```

The implementation uses Cholesky solve, not an explicit inverse. If the projected covariance is not positive definite, it retries with diagonal jitter. If it still fails, the pair receives a large finite diagnostic value; behavior-changing modes honor `mahalanobis_fail_open`.

Supported gates:

- 4D `cx, cy, w, h` default
- 2D `cx, cy`
- confidence thresholds: `0.95`, `0.99`, `0.999`
- optional explicit `mahalanobis_gate_threshold`

## Modes

- `static`: production baseline, no Mahalanobis behavior.
- `static_shadow_mahalanobis`: computes Mahalanobis diagnostics when audit CSV is enabled, but still matches with the static cost.
- `static_mahalanobis_gate`: computes static cost, then rejects applicable candidates that fail the Mahalanobis gate.
- `static_mahalanobis_blend`: adds `mahalanobis_weight * normalized_mahalanobis_cost` to applicable static costs.

Gate and blend modes are ablations only. Their default application is conservative: `lost_only` states and all detection tiers.

## Experiment Commands

Baseline:

```powershell
python -m retail_tracking.eval.run_mot_video_eval `
  --video "data\MOT20-02_package\MOT20-02\MOT20-02.mp4" `
  --model "weights\YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt" `
  --tracker tracktrack `
  --config "retail_tracking/configs/tracktrack_mot20_static_best.yaml" `
  --mot-out "outputs\MOT20-02_pred_static_best.txt" `
  --device cuda:0 `
  --appearance-mode jde `
  --imgsz 1280 `
  --conf 0.1 `
  --normal-iou 0.70
```

Evaluate:

```powershell
python -m retail_tracking.eval.eval_mot_single `
  --gt "data\MOT20-02_package\MOT20-02\gt\gt.txt" `
  --pred "outputs\MOT20-02_pred_static_best.txt" `
  --iou 0.5
```

Shadow Mahalanobis audit:

```powershell
python -m retail_tracking.eval.run_mot_video_eval `
  --video "data\MOT20-02_package\MOT20-02\MOT20-02.mp4" `
  --model "weights\YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt" `
  --tracker tracktrack `
  --config "retail_tracking/configs/tracktrack_mot20_shadow_mahalanobis.yaml" `
  --mot-out "outputs\MOT20-02_pred_shadow_mahalanobis.txt" `
  --device cuda:0 `
  --appearance-mode jde `
  --imgsz 1280 `
  --conf 0.1 `
  --normal-iou 0.70 `
  --assoc-debug-csv "outputs\MOT20-02_assoc_shadow_mahalanobis.csv" `
  --assoc-debug-summary "outputs\MOT20-02_assoc_shadow_mahalanobis_summary.json"
```

Analyze audit:

```powershell
python -m retail_tracking.eval.analyze_assoc_audit `
  --csv "outputs\MOT20-02_assoc_shadow_mahalanobis.csv" `
  --out-json "outputs\MOT20-02_assoc_shadow_mahalanobis_analysis.json" `
  --out-md "outputs\MOT20-02_assoc_shadow_mahalanobis_analysis.md"
```

Gate ablation:

```powershell
python -m retail_tracking.eval.run_mot_video_eval `
  --video "data\MOT20-02_package\MOT20-02\MOT20-02.mp4" `
  --model "weights\YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt" `
  --tracker tracktrack `
  --config "retail_tracking/configs/tracktrack_mot20_mahalanobis_gate.yaml" `
  --mot-out "outputs\MOT20-02_pred_mahalanobis_gate.txt" `
  --device cuda:0 `
  --appearance-mode jde `
  --imgsz 1280 `
  --conf 0.1 `
  --normal-iou 0.70 `
  --assoc-debug-csv "outputs\MOT20-02_assoc_mahalanobis_gate.csv" `
  --assoc-debug-summary "outputs\MOT20-02_assoc_mahalanobis_gate_summary.json"
```

Blend ablation:

```powershell
python -m retail_tracking.eval.run_mot_video_eval `
  --video "data\MOT20-02_package\MOT20-02\MOT20-02.mp4" `
  --model "weights\YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt" `
  --tracker tracktrack `
  --config "retail_tracking/configs/tracktrack_mot20_mahalanobis_blend.yaml" `
  --mot-out "outputs\MOT20-02_pred_mahalanobis_blend.txt" `
  --device cuda:0 `
  --appearance-mode jde `
  --imgsz 1280 `
  --conf 0.1 `
  --normal-iou 0.70 `
  --assoc-debug-csv "outputs\MOT20-02_assoc_mahalanobis_blend.csv" `
  --assoc-debug-summary "outputs\MOT20-02_assoc_mahalanobis_blend_summary.json"
```

## Audit Interpretation

Mahalanobis is promising if:

- many unmatched tracks have at least one Mahalanobis-valid candidate,
- many IoU-blocked candidates are Mahalanobis-valid,
- the benefit is concentrated in Lost tracks,
- matched successful pairs usually pass the Mahalanobis gate.

Mahalanobis is risky if:

- many successful matched pairs fail the gate,
- matched Tracked tracks have a high rejection rate,
- gate/blend improves one metric but hurts HOTA, IDF1, or AssA beyond the acceptance bounds.

## Known Limitations

- Shadow mode proves only what Mahalanobis would have done locally per pair; it does not account for downstream ID effects unless gate/blend ablations are evaluated.
- Pair-level audit CSVs can be large on dense scenes. Keep `--assoc-debug-max-frames` for quick diagnosis.
- The conservative `lost_only` gate default may miss some active-track benefits, but it reduces risk to the current best baseline.

## Recommendation

Keep `tracktrack_mot20_static_best.yaml` as production until the shadow audit shows a clear Mahalanobis opportunity. Promote gate or blend only if full MOT evaluation preserves or improves HOTA, IDF1, AssA, IDs, and FPS relative to the static baseline.
