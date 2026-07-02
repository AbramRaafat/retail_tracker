# Relaxed Recovery Runtime Optimization

## Diagnosis

The recovery-tuned TrackTrack + YOLO11-JDE path was spending most of its extra runtime in detector inference, not in the final MOT writer or visualization path. With `--relaxed-dets`, the adapter ran YOLO twice per frame:

1. a normal pass at the requested confidence and normal NMS IoU,
2. a relaxed pass at lower confidence and higher NMS IoU to collect recovery candidates.

That preserves recovery accuracy, but it roughly doubles detector forward/NMS work before TrackTrack sees the frame. The relaxed pass can also increase tracker-side duplicate filtering and association cost because more candidates flow into `find_deleted_detections`.

## Implementation

The optimized mode is configurable with `--relaxed-source single-pass`. In that mode the detector runs one relaxed superset forward, then derives the normal detection set in software:

1. keep relaxed detections and JDE embeddings as the recovery superset,
2. select detections above the normal confidence threshold,
3. apply class-aware software NMS with the normal IoU threshold,
4. gather the same embedding rows for the resulting normal detections.

The default remains `--relaxed-source two-pass` so the current recovery-tuned reference is still reproducible. Non-relaxed runs still use the original single normal detector pass.

The tracker IoU overlap helper was also vectorized with NumPy while preserving its existing `+1` pixel box convention. This keeps association and relaxed duplicate filtering behavior equivalent, but avoids nested Python loops when relaxed candidates are numerous.

## Profiling

`run_mot_video_eval.py` now accepts `--timing-profile <csv>`. Profiling is disabled by default. When enabled, it records per-frame detector normal/relaxed/superset time, software split/NMS time, tracker update time, total frame time, normal detection count, relaxed detection count, and output track count.
