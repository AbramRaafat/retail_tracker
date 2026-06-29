"""Single-sequence local MOT evaluator comparing one prediction file against ground truth.

Example usage:
    python -m retail_tracking.eval.eval_mot_single \
      --gt C:\\Users\\abraa\\datasets\\MOT20-02_package\\MOT20-02\\gt\\gt.txt \
      --pred outputs\\MOT20-02_pred.txt \
      --iou 0.5
"""

import argparse
import sys
import pandas as pd
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single Sequence MOT Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth gt.txt file.")
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted MOT text file.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for evaluation.")
    return parser.parse_args()


def load_mot_file(filepath: str) -> pd.DataFrame:
    try:
        # Load the space/comma delimited data
        df = pd.read_csv(filepath, header=None, sep=None, engine='python')
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        sys.exit(1)
        
    orig_cols = df.shape[1]
    # Pad columns to at least 10 columns with -1
    for i in range(orig_cols, 10):
        df[i] = -1
        
    # Standard MOT Challenge column names
    df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility', 'ignored']
    
    # Store original column count as metadata attribute
    df.attrs['orig_cols'] = orig_cols
    return df


def run_evaluation() -> None:
    # NumPy 2.0 compatibility monkey-patch for older packages like motmetrics
    import numpy as np
    if not hasattr(np, 'asfarray'):
        np.asfarray = lambda val, dtype=np.float64: np.asarray(val, dtype=dtype)

    try:
        import motmetrics as mm
    except ImportError:
        raise ImportError(
            "motmetrics is missing. Please install it by running:\n"
            "pip install motmetrics pandas numpy scipy"
        )

    args = parse_arguments()

    # Load GT and Pred files
    gt_df = load_mot_file(args.gt)
    pred_df = load_mot_file(args.pred)

    orig_gt_cols = gt_df.attrs['orig_cols']

    # Filter GT file:
    # 1. Keep rows where column 7 (conf/mark) equals 1 when present.
    if orig_gt_cols >= 7:
        gt_df = gt_df[gt_df['conf'] == 1]
    
    # 2. Keep class 1 pedestrian rows if class column contains 1.
    if orig_gt_cols >= 8:
        if 1 in gt_df['class'].values:
            gt_df = gt_df[gt_df['class'] == 1]

    # Initialize py-motmetrics accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Get union of all frame IDs
    all_frames = sorted(list(set(gt_df['frame'].unique()).union(set(pred_df['frame'].unique()))))

    for f in all_frames:
        gt_frame = gt_df[gt_df['frame'] == f]
        pred_frame = pred_df[pred_df['frame'] == f]

        gt_ids = gt_frame['id'].tolist()
        pred_ids = pred_frame['id'].tolist()

        gt_boxes = gt_frame[['x', 'y', 'w', 'h']].values
        pred_boxes = pred_frame[['x', 'y', 'w', 'h']].values

        # Compute IoU distance matrix (max distance is 1.0 - iou_threshold)
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1.0 - args.iou)

        acc.update(gt_ids, pred_ids, distances)

    # Compute and print metrics
    mh = mm.metrics.create()
    metrics_list = [
        'num_frames',
        'mota',
        'motp',
        'idf1',
        'idp',
        'idr',
        'precision',
        'recall',
        'num_objects',
        'num_predictions',
        'num_matches',
        'num_false_positives',
        'num_misses',
        'num_switches',
        'num_fragmentations'
    ]

    summary = mh.compute(acc, metrics=metrics_list, name='sequence_eval')

    print("\n" + "=" * 50)
    print("MOT Challenge Single-Sequence Evaluation Results")
    print("=" * 50)
    
    # Print the clean summary from motmetrics
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    
    print("-" * 50)
    print("Detailed Metrics list:")
    for metric in metrics_list:
        val = summary.loc['sequence_eval', metric]
        if isinstance(val, (float, np.float32, np.float64)):
            # Formats percentage metrics cleanly or formats float values
            if metric in ['mota', 'motp', 'idf1', 'idp', 'idr', 'precision', 'recall']:
                print(f"  {metric:<22}: {val * 100.0:.2f}%")
            else:
                print(f"  {metric:<22}: {val:.4f}")
        else:
            print(f"  {metric:<22}: {val}")
    print("=" * 50)
    print("NOTE: This is a single-sequence local evaluator and not a full official TrackEval benchmark run.")
    print("=" * 50)


if __name__ == "__main__":
    run_evaluation()
