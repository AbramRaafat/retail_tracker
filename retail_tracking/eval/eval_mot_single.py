"""Single-sequence local MOT evaluator.

Computes:
- MOTA / MOTP / IDF1 using motmetrics
- HOTA-style metrics using local implementation:
  HOTA, DetA, AssA, DetRe, DetPr, AssRe, AssPr, LocA, OWTA

Example:
    python -m retail_tracking.eval.eval_mot_single \
      --gt data\\MOT20-02_package\\MOT20-02\\gt\\gt.txt \
      --pred outputs\\MOT20-02_pred.txt \
      --iou 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MOT_COLUMNS = [
    "frame",
    "id",
    "x",
    "y",
    "w",
    "h",
    "conf",
    "class",
    "visibility",
    "ignored",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single Sequence MOT Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth gt.txt file.")
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted MOT text file.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for motmetrics matching.")
    parser.add_argument("--no-hota-table", action="store_true", help="Hide per-alpha HOTA table.")
    return parser.parse_args()


def load_mot_file(filepath: str | Path, is_gt: bool) -> pd.DataFrame:
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(filepath)

    if filepath.stat().st_size == 0:
        return pd.DataFrame(columns=MOT_COLUMNS)

    try:
        df = pd.read_csv(
            filepath,
            header=None,
            sep=r"\s*,\s*|\s+",
            engine="python",
        )
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        sys.exit(1)

    original_cols = df.shape[1]

    while df.shape[1] < 10:
        df[df.shape[1]] = -1

    df = df.iloc[:, :10]
    df.columns = MOT_COLUMNS
    df.attrs["orig_cols"] = original_cols

    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)

    for col in ["x", "y", "w", "h", "conf", "class", "visibility"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)

    if is_gt:
        # MOT GT format:
        # frame,id,x,y,w,h,mark,class,visibility
        # mark/conf == 1 means valid annotation.
        if original_cols >= 7:
            df = df[df["conf"] == 1]

        # Class 1 is pedestrian. Keep pedestrian rows if the class exists.
        if original_cols >= 8:
            if 1 in set(df["class"].astype(int).unique().tolist()):
                df = df[df["class"].astype(int) == 1]

    return df.reset_index(drop=True)


def box_iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float64)

    gt = np.asarray(gt_boxes, dtype=np.float64)
    pr = np.asarray(pred_boxes, dtype=np.float64)

    gt_x1 = gt[:, 0]
    gt_y1 = gt[:, 1]
    gt_x2 = gt[:, 0] + np.maximum(0.0, gt[:, 2])
    gt_y2 = gt[:, 1] + np.maximum(0.0, gt[:, 3])

    pr_x1 = pr[:, 0]
    pr_y1 = pr[:, 1]
    pr_x2 = pr[:, 0] + np.maximum(0.0, pr[:, 2])
    pr_y2 = pr[:, 1] + np.maximum(0.0, pr[:, 3])

    inter_x1 = np.maximum(gt_x1[:, None], pr_x1[None, :])
    inter_y1 = np.maximum(gt_y1[:, None], pr_y1[None, :])
    inter_x2 = np.minimum(gt_x2[:, None], pr_x2[None, :])
    inter_y2 = np.minimum(gt_y2[:, None], pr_y2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    gt_area = np.maximum(0.0, gt[:, 2]) * np.maximum(0.0, gt[:, 3])
    pr_area = np.maximum(0.0, pr[:, 2]) * np.maximum(0.0, pr[:, 3])

    union = gt_area[:, None] + pr_area[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def get_frame_data(df: pd.DataFrame, frame: int) -> tuple[np.ndarray, np.ndarray]:
    fdf = df[df["frame"] == frame]
    ids = fdf["id"].to_numpy(dtype=int)
    boxes = fdf[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
    return ids, boxes


def linear_assignment_max(score_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if score_matrix.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError(
            "scipy is required for HOTA matching. Install it with:\n"
            "pip install scipy numpy pandas"
        ) from exc

    rows, cols = linear_sum_assignment(-score_matrix)
    return rows.astype(int), cols.astype(int)


def compute_hota(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    eps = 1e-12
    alphas = np.arange(0.05, 1.0, 0.05, dtype=np.float64)
    alphas = np.round(alphas, 10)

    gt_ids_all = sorted(gt_df["id"].unique().tolist())
    pred_ids_all = sorted(pred_df["id"].unique().tolist())

    gt_id_map = {int(track_id): idx for idx, track_id in enumerate(gt_ids_all)}
    pred_id_map = {int(track_id): idx for idx, track_id in enumerate(pred_ids_all)}

    num_gt_ids = len(gt_id_map)
    num_pred_ids = len(pred_id_map)

    gt_id_count = np.zeros(num_gt_ids, dtype=np.float64)
    pred_id_count = np.zeros(num_pred_ids, dtype=np.float64)

    for track_id, count in gt_df["id"].value_counts().items():
        gt_id_count[gt_id_map[int(track_id)]] = float(count)

    for track_id, count in pred_df["id"].value_counts().items():
        pred_id_count[pred_id_map[int(track_id)]] = float(count)

    frames = sorted(set(gt_df["frame"].unique()).union(set(pred_df["frame"].unique())))

    potential_matches_count = np.zeros((num_gt_ids, num_pred_ids), dtype=np.float64)

    # First pass: global alignment score.
    for frame in frames:
        gt_ids, gt_boxes = get_frame_data(gt_df, frame)
        pred_ids, pred_boxes = get_frame_data(pred_df, frame)

        if len(gt_ids) == 0 or len(pred_ids) == 0:
            continue

        sim = box_iou_matrix(gt_boxes, pred_boxes)

        row_sum = sim.sum(axis=1, keepdims=True)
        col_sum = sim.sum(axis=0, keepdims=True)
        sim_iou = sim / np.maximum(row_sum + col_sum - sim, eps)

        gt_indices = np.array([gt_id_map[int(i)] for i in gt_ids], dtype=int)
        pred_indices = np.array([pred_id_map[int(i)] for i in pred_ids], dtype=int)

        potential_matches_count[np.ix_(gt_indices, pred_indices)] += sim_iou

    global_alignment = potential_matches_count / np.maximum(
        gt_id_count[:, None] + pred_id_count[None, :] - potential_matches_count,
        eps,
    )

    n_alpha = len(alphas)

    hota_tp = np.zeros(n_alpha, dtype=np.float64)
    hota_fp = np.zeros(n_alpha, dtype=np.float64)
    hota_fn = np.zeros(n_alpha, dtype=np.float64)
    loc_sum = np.zeros(n_alpha, dtype=np.float64)

    matches_count = np.zeros((n_alpha, num_gt_ids, num_pred_ids), dtype=np.float64)

    # Second pass: alpha-wise matching.
    for frame in frames:
        gt_ids, gt_boxes = get_frame_data(gt_df, frame)
        pred_ids, pred_boxes = get_frame_data(pred_df, frame)

        num_gt = len(gt_ids)
        num_pred = len(pred_ids)

        if num_gt == 0 and num_pred == 0:
            continue

        if num_gt == 0:
            hota_fp += num_pred
            continue

        if num_pred == 0:
            hota_fn += num_gt
            continue

        sim = box_iou_matrix(gt_boxes, pred_boxes)

        gt_indices = np.array([gt_id_map[int(i)] for i in gt_ids], dtype=int)
        pred_indices = np.array([pred_id_map[int(i)] for i in pred_ids], dtype=int)

        alignment = global_alignment[np.ix_(gt_indices, pred_indices)]
        score_matrix = alignment * sim

        row_ind, col_ind = linear_assignment_max(score_matrix)

        matched_sims = sim[row_ind, col_ind] if len(row_ind) else np.array([])
        matched_gt_global = gt_indices[row_ind] if len(row_ind) else np.array([], dtype=int)
        matched_pred_global = pred_indices[col_ind] if len(col_ind) else np.array([], dtype=int)

        for a_idx, alpha in enumerate(alphas):
            valid = matched_sims >= alpha

            tp = int(valid.sum())
            fp = num_pred - tp
            fn = num_gt - tp

            hota_tp[a_idx] += tp
            hota_fp[a_idx] += fp
            hota_fn[a_idx] += fn

            if tp > 0:
                loc_sum[a_idx] += matched_sims[valid].sum()

                for gi, pi in zip(matched_gt_global[valid], matched_pred_global[valid]):
                    matches_count[a_idx, gi, pi] += 1.0

    rows = []

    for a_idx, alpha in enumerate(alphas):
        tp = hota_tp[a_idx]
        fp = hota_fp[a_idx]
        fn = hota_fn[a_idx]

        det_re = tp / max(tp + fn, eps)
        det_pr = tp / max(tp + fp, eps)
        det_a = tp / max(tp + fn + fp, eps)
        loc_a = loc_sum[a_idx] / max(tp, eps)

        mc = matches_count[a_idx]

        if tp > 0:
            ass_denom = np.maximum(gt_id_count[:, None] + pred_id_count[None, :] - mc, eps)
            ass_a_pair = mc / ass_denom
            ass_a = float((mc * ass_a_pair).sum() / max(tp, eps))

            ass_re_pair = mc / np.maximum(gt_id_count[:, None], eps)
            ass_re = float((mc * ass_re_pair).sum() / max(tp, eps))

            ass_pr_pair = mc / np.maximum(pred_id_count[None, :], eps)
            ass_pr = float((mc * ass_pr_pair).sum() / max(tp, eps))
        else:
            ass_a = 0.0
            ass_re = 0.0
            ass_pr = 0.0

        hota = float(np.sqrt(det_a * ass_a))
        owta = float(np.sqrt(det_re * ass_a))

        rows.append(
            {
                "alpha": float(alpha),
                "HOTA": hota,
                "DetA": det_a,
                "AssA": ass_a,
                "DetRe": det_re,
                "DetPr": det_pr,
                "AssRe": ass_re,
                "AssPr": ass_pr,
                "LocA": loc_a,
                "OWTA": owta,
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
            }
        )

    per_alpha = pd.DataFrame(rows)

    summary = {
        "HOTA": float(per_alpha["HOTA"].mean()),
        "DetA": float(per_alpha["DetA"].mean()),
        "AssA": float(per_alpha["AssA"].mean()),
        "DetRe": float(per_alpha["DetRe"].mean()),
        "DetPr": float(per_alpha["DetPr"].mean()),
        "AssRe": float(per_alpha["AssRe"].mean()),
        "AssPr": float(per_alpha["AssPr"].mean()),
        "LocA": float(per_alpha["LocA"].mean()),
        "OWTA": float(per_alpha["OWTA"].mean()),
        "HOTA(0)": float(per_alpha.iloc[0]["HOTA"]),
        "LocA(0)": float(per_alpha.iloc[0]["LocA"]),
    }

    summary["HOTALocA(0)"] = summary["HOTA(0)"] * summary["LocA(0)"]

    return summary, per_alpha


def run_motmetrics(gt_df: pd.DataFrame, pred_df: pd.DataFrame, iou_threshold: float):
    # NumPy 2.0 compatibility monkey-patch for older motmetrics.
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda val, dtype=np.float64: np.asarray(val, dtype=dtype)

    try:
        import motmetrics as mm
    except ImportError:
        raise ImportError(
            "motmetrics is missing. Please install it by running:\n"
            "pip install motmetrics pandas numpy scipy"
        )

    acc = mm.MOTAccumulator(auto_id=True)

    frames = sorted(set(gt_df["frame"].unique()).union(set(pred_df["frame"].unique())))

    for frame in frames:
        gt_frame = gt_df[gt_df["frame"] == frame]
        pred_frame = pred_df[pred_df["frame"] == frame]

        gt_ids = gt_frame["id"].tolist()
        pred_ids = pred_frame["id"].tolist()

        gt_boxes = gt_frame[["x", "y", "w", "h"]].values
        pred_boxes = pred_frame[["x", "y", "w", "h"]].values

        distances = mm.distances.iou_matrix(
            gt_boxes,
            pred_boxes,
            max_iou=1.0 - iou_threshold,
        )

        acc.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()

    metrics_list = [
        "num_frames",
        "mota",
        "motp",
        "idf1",
        "idp",
        "idr",
        "precision",
        "recall",
        "num_objects",
        "num_predictions",
        "num_matches",
        "num_false_positives",
        "num_misses",
        "num_switches",
        "num_fragmentations",
    ]

    summary = mh.compute(acc, metrics=metrics_list, name="sequence_eval")
    return summary, mh


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def print_hota_summary(hota_summary: dict, per_alpha: pd.DataFrame, show_table: bool) -> None:
    print("-" * 50)
    print("HOTA-style metrics")
    print("-" * 50)
    print(f"  {'hota':<22}: {pct(hota_summary['HOTA'])}")
    print(f"  {'deta':<22}: {pct(hota_summary['DetA'])}")
    print(f"  {'assa':<22}: {pct(hota_summary['AssA'])}")
    print(f"  {'det_re':<22}: {pct(hota_summary['DetRe'])}")
    print(f"  {'det_pr':<22}: {pct(hota_summary['DetPr'])}")
    print(f"  {'ass_re':<22}: {pct(hota_summary['AssRe'])}")
    print(f"  {'ass_pr':<22}: {pct(hota_summary['AssPr'])}")
    print(f"  {'loca':<22}: {pct(hota_summary['LocA'])}")
    print(f"  {'owta':<22}: {pct(hota_summary['OWTA'])}")
    print(f"  {'hota(0)':<22}: {pct(hota_summary['HOTA(0)'])}")
    print(f"  {'loca(0)':<22}: {pct(hota_summary['LocA(0)'])}")
    print(f"  {'hotaloca(0)':<22}: {pct(hota_summary['HOTALocA(0)'])}")

    if show_table:
        print("-" * 50)
        print("Per-alpha HOTA table")
        table = per_alpha.copy()

        for col in ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA"]:
            table[col] = table[col].map(lambda v: f"{100.0 * v:.2f}")

        table["alpha"] = table["alpha"].map(lambda v: f"{v:.2f}")

        print(
            table[
                [
                    "alpha",
                    "HOTA",
                    "DetA",
                    "AssA",
                    "DetRe",
                    "DetPr",
                    "AssRe",
                    "AssPr",
                    "LocA",
                    "TP",
                    "FP",
                    "FN",
                ]
            ].to_string(index=False)
        )


def run_evaluation() -> None:
    args = parse_arguments()

    gt_df = load_mot_file(args.gt, is_gt=True)
    pred_df = load_mot_file(args.pred, is_gt=False)

    mot_summary, mh = run_motmetrics(gt_df, pred_df, args.iou)
    hota_summary, per_alpha = compute_hota(gt_df, pred_df)

    print("\n" + "=" * 50)
    print("MOT Challenge Single-Sequence Evaluation Results")
    print("=" * 50)

    strsummary = mh.io.render_summary(
        mot_summary,
        formatters=mh.formatters,
        namemap=mh.io.motchallenge_metric_names,
    ) if hasattr(mh, "io") else None

    # motmetrics exposes io at module level, not always from mh.
    if strsummary is None:
        import motmetrics as mm
        strsummary = mm.io.render_summary(
            mot_summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names,
        )

    print(strsummary)

    print("-" * 50)
    print("Detailed MOT metrics list:")

    metrics_list = [
        "num_frames",
        "mota",
        "motp",
        "idf1",
        "idp",
        "idr",
        "precision",
        "recall",
        "num_objects",
        "num_predictions",
        "num_matches",
        "num_false_positives",
        "num_misses",
        "num_switches",
        "num_fragmentations",
    ]

    for metric in metrics_list:
        val = mot_summary.loc["sequence_eval", metric]
        if isinstance(val, (float, np.float32, np.float64)):
            if metric in ["mota", "idf1", "idp", "idr", "precision", "recall"]:
                print(f"  {metric:<22}: {val * 100.0:.2f}%")
            else:
                print(f"  {metric:<22}: {val:.4f}")
        else:
            print(f"  {metric:<22}: {val}")

    print_hota_summary(
        hota_summary=hota_summary,
        per_alpha=per_alpha,
        show_table=not args.no_hota_table,
    )

    print("=" * 50)
    print("NOTE: This is a single-sequence local evaluator.")
    print("HOTA here is a local HOTA-style implementation, not full TrackEval.")
    print("=" * 50)


if __name__ == "__main__":
    run_evaluation()