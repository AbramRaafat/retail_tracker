"""Evaluation runner to run tracking on a single MOT video sequence and write MOTChallenge predictions.

Example usage:
    python -m retail_tracking.eval.run_mot_video_eval \
      --video C:\\Users\\abraa\\datasets\\MOT20-02_package\\MOT20-02\\MOT20-02.mp4 \
      --model retail_tracking\\weights\\YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt \
      --tracker tracktrack \
      --mot-out outputs\\MOT20-02_pred.txt \
      --video-out outputs\\MOT20-02_vis.mp4 \
      --verbose
"""

import argparse
import logging
import time
from pathlib import Path
import sys
import cv2
import numpy as np

# Ensure boxmot is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOXMOT_ROOT = PROJECT_ROOT / "boxmot"
if str(BOXMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOXMOT_ROOT))

from retail_tracking.src.core.tracker import RetailTracker
from retail_tracking.src.detection.adapters import UltralyticsJDEAdapter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MOT Single Video Evaluation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--video", type=str, required=True, help="Input MOT video sequence (.mp4).")
    parser.add_argument("--model", type=str, required=True, help="YOLO model path.")
    parser.add_argument("--tracker", type=str, default="tracktrack", help="BoxMOT tracker type.")
    parser.add_argument("--mot-out", type=str, required=True, help="Output file path for MOTChallenge predictions.")
    parser.add_argument("--config", type=str, default=None, help="Optional tracker YAML configuration path.")
    parser.add_argument("--reid", type=str, default=None, help="Optional ReID model path.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device (e.g. cuda:0, cpu).")
    parser.add_argument("--video-out", type=str, default=None, help="Optional visualization video output path.")
    parser.add_argument("--display", action="store_true", help="Display annotated tracking live.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose diagnostics.")
    return parser.parse_args()


def run_eval() -> None:
    args = parse_arguments()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("Initializing evaluation run...")
    logger.info("Video path: %s", args.video)
    logger.info("Model path: %s", args.model)
    logger.info("Tracker type: %s", args.tracker)
    logger.info("MOT output path: %s", args.mot_out)

    # Ensure parent directory for MOT output exists
    Path(args.mot_out).parent.mkdir(parents=True, exist_ok=True)
    if args.video_out:
        Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)

    # Setup JDE Adapter and RetailTracker
    try:
        model_adapter = UltralyticsJDEAdapter(
            weights_path=args.model,
            conf_threshold=0.1,  # Keep low to let tracker gate detections
            classes=[0]          # Default to person
        )
        tracker_system = RetailTracker(
            detector=model_adapter,
            tracker_type=args.tracker,
            tracker_config=args.config,
            reid_weights=args.reid,
            device=args.device
        )
    except Exception as e:
        logger.error("Initialization failure: %s", e)
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error("Failed to open video source: %s", args.video)
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    out = None
    if args.video_out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.video_out, fourcc, int(fps), (width, height))
        if not out.isOpened():
            cap.release()
            logger.error("Failed to open output video writer: %s", args.video_out)
            sys.exit(1)

    visualize = (args.video_out is not None) or args.display

    frame_count = 0
    total_processing_time = 0.0
    predictions = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            start_time = time.time()

            # Process frame using the RetailTracker wrapper
            tracks = tracker_system.process_frame(frame)

            processing_time = time.time() - start_time
            total_processing_time += processing_time

            # Format MOTChallenge outputs: frame,id,x,y,w,h,score,-1,-1,-1
            # tracks format: [x1, y1, x2, y2, track_id, score, cls, extra]
            for track in tracks:
                x1, y1, x2, y2 = track[0], track[1], track[2], track[3]
                track_id = int(track[4])
                score = track[5]

                x = x1
                y = y1
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)

                predictions.append(f"{frame_count},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1\n")

            if visualize:
                annotated_frame = tracker_system.draw_tracks(frame.copy(), tracks)
                if out is not None:
                    out.write(annotated_frame)
                if args.display:
                    cv2.imshow("MOT Single Video Evaluation", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Evaluation visualization stopped by user.")
                        break

            if args.verbose:
                logger.debug("Frame %d | Detections & Tracks processed. Latency: %.3fs", frame_count, processing_time)
            elif frame_count % 20 == 0:
                print(f"\rProcessing frame {frame_count}...", end="", flush=True)

    except KeyboardInterrupt:
        logger.warning("\nUser interrupted execution.")
    finally:
        cap.release()
        if out is not None:
            out.release()
        if args.display:
            cv2.destroyAllWindows()
        print()

    # Save prediction outputs
    with open(args.mot_out, "w") as f:
        f.writelines(predictions)

    avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0.0
    logger.info("Evaluation sequence finalized.")
    logger.info("Frame count processed: %d", frame_count)
    logger.info("Average FPS: %.2f", avg_fps)
    logger.info("MOT Challenge format predictions written to: %s", args.mot_out)
    if args.video_out:
        logger.info("Visualization video saved to: %s", args.video_out)


if __name__ == "__main__":
    run_eval()
