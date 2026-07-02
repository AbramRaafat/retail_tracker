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


def _load_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _requests_cuda(device: str) -> bool:
    return device.lower().startswith("cuda")


def _cuda_available(torch_module) -> bool:
    return bool(torch_module is not None and torch_module.cuda.is_available())


def _synchronize_cuda(torch_module, enabled: bool) -> None:
    if enabled:
        torch_module.cuda.synchronize()


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
    parser.add_argument("--conf", type=float, default=0.1, help="Detector confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Detector inference image size.")
    parser.add_argument("--half", dest="half", action="store_true", default=None, help="Enable half precision.")
    parser.add_argument("--no-half", dest="half", action="store_false", help="Disable half precision.")
    parser.add_argument("--warmup", dest="warmup", action="store_true", default=True, help="Run one detector warmup pass.")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", help="Skip detector warmup.")
    parser.add_argument("--progress-every", type=int, default=50, help="Console progress update interval in frames.")
    parser.add_argument("--debug-routing", action="store_true", help="Print early-frame detector/embedding routing metadata.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional limit for quick first-N-frame runs.")
    parser.add_argument("--video-out", type=str, default=None, help="Optional visualization video output path.")
    parser.add_argument("--display", action="store_true", help="Display annotated tracking live.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose diagnostics.")
    return parser.parse_args()


def run_eval() -> None:
    args = parse_arguments()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    torch_module = _load_torch()
    requested_cuda = _requests_cuda(args.device)
    use_half = args.half if args.half is not None else requested_cuda
    if not requested_cuda and use_half:
        logger.warning("Half precision was requested on CPU; disabling half precision.")
        use_half = False

    torch_version = getattr(torch_module, "__version__", "not installed") if torch_module else "not installed"
    torch_cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None) if torch_module else None
    cuda_available = _cuda_available(torch_module)
    cuda_device_name = None
    if cuda_available:
        try:
            cuda_device_name = torch_module.cuda.get_device_name(args.device if requested_cuda else 0)
        except Exception as exc:
            cuda_device_name = f"unavailable ({exc})"

    logger.info("Python executable: %s", sys.executable)
    logger.info("torch version: %s", torch_version)
    logger.info("torch CUDA version: %s", torch_cuda_version)
    logger.info("torch.cuda.is_available(): %s", cuda_available)
    logger.info("CUDA device name: %s", cuda_device_name)
    logger.info("Requested device: %s", args.device)
    logger.info("Effective half precision: %s", use_half)
    logger.info("Detector imgsz: %s", args.imgsz)
    logger.info("Detector confidence threshold: %.3f", args.conf)

    if requested_cuda and not cuda_available:
        logger.error(
            "Requested device '%s', but CUDA is not visible to torch. "
            "Exiting because inference would silently fall back to CPU-slow execution. "
            "Use --device cpu to run CPU intentionally.",
            args.device,
        )
        sys.exit(1)

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
            conf_threshold=args.conf,
            classes=[0],
            device=args.device,
            half=use_half,
            imgsz=args.imgsz,
        )
        tracker_system = RetailTracker(
            detector=model_adapter,
            tracker_type=args.tracker,
            tracker_config=args.config,
            reid_weights=args.reid,
            device=args.device,
            half=use_half,
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

    if args.warmup:
        logger.info("Running detector warmup...")
        model_adapter.warmup((height, width, 3))
        if requested_cuda:
            _synchronize_cuda(torch_module, enabled=True)

    frame_count = 0
    total_wall_start = time.perf_counter()
    total_processing_time = 0.0
    interval_processing_time = 0.0
    cuda_timing_active = requested_cuda and cuda_available

    try:
        with open(args.mot_out, "w", encoding="utf-8") as mot_file:
            while True:
                if args.max_frames is not None and frame_count >= args.max_frames:
                    logger.info("Reached max frame limit: %d", args.max_frames)
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                _synchronize_cuda(torch_module, cuda_timing_active)
                start_time = time.perf_counter()

                # Process frame using the RetailTracker wrapper
                tracks = tracker_system.process_frame(frame)

                _synchronize_cuda(torch_module, cuda_timing_active)
                processing_time = time.perf_counter() - start_time
                total_processing_time += processing_time
                interval_processing_time += processing_time

                if args.debug_routing and frame_count <= 20:
                    metadata = getattr(tracker_system, "last_jde_metadata", {}) or {}
                    logger.info(
                        "Routing frame %d | detections=%s | tracks=%d | has_embeddings=%s | "
                        "embedding_shape=%s | embedding_norm_mean=%s",
                        frame_count,
                        metadata.get("num_detections"),
                        len(tracks),
                        metadata.get("has_embeddings"),
                        metadata.get("embedding_shape"),
                        metadata.get("embedding_norm_mean"),
                    )

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

                    mot_file.write(f"{frame_count},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1\n")

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
                elif args.progress_every > 0 and frame_count % args.progress_every == 0:
                    interval_fps = args.progress_every / interval_processing_time if interval_processing_time > 0 else 0.0
                    avg_fps_so_far = frame_count / total_processing_time if total_processing_time > 0 else 0.0
                    print(
                        f"\rProcessing frame {frame_count} | "
                        f"avg FPS {avg_fps_so_far:.2f} | interval FPS {interval_fps:.2f}",
                        end="",
                        flush=True,
                    )
                    interval_processing_time = 0.0

    except KeyboardInterrupt:
        logger.warning("\nUser interrupted execution.")
    finally:
        cap.release()
        if out is not None:
            out.release()
        if args.display:
            cv2.destroyAllWindows()
        print()

    total_wall_time = time.perf_counter() - total_wall_start
    avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0.0
    logger.info("Evaluation sequence finalized.")
    logger.info("Frame count processed: %d", frame_count)
    logger.info("Total processing time: %.3fs", total_processing_time)
    logger.info("Total wall time: %.3fs", total_wall_time)
    logger.info("Average FPS: %.2f", avg_fps)
    logger.info("MOT Challenge format predictions written to: %s", args.mot_out)
    if args.video_out:
        logger.info("Visualization video saved to: %s", args.video_out)


if __name__ == "__main__":
    run_eval()
