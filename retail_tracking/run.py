import cv2
import sys
import argparse
import logging
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOXMOT_ROOT = PROJECT_ROOT / "boxmot"
if str(BOXMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOXMOT_ROOT))

try:
    from retail_tracking.src.core.tracker import RetailTracker
    from retail_tracking.src.detection.adapters import UltralyticsJDEAdapter
    from retail_tracking.src.config import PipelineConfig
except ImportError:
    from src.core.tracker import RetailTracker
    from src.detection.adapters import UltralyticsJDEAdapter
    from src.config import PipelineConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retail Tracking Pipeline Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # I/O Paths
    parser.add_argument("-i", "--input", type=str, required=True, help="Input video source.")
    parser.add_argument(
        "-o", "--output", "--video-out",
        dest="video_out",
        type=str,
        default=None,
        help="Optional annotated output video path."
    )
    parser.add_argument("--display", action="store_true", help="Display annotated tracking output in a window.")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for rendering.")
    
    # Model Zoo Routing
    parser.add_argument("-m", "--model", type=str, default="yolo11n.pt",
                        help="YOLO model path. Can be standard or JDE fork.")
    parser.add_argument("--reid", type=str, default=None,
                        help="External ReID model path (e.g., 'osnet_x1_0_msmt17.pt'). "
                             "Omit if using a JDE model.")
                             
    # Tracker Configuration
    parser.add_argument("--tracker", type=str, default="tracktrack", 
                        help="Target BoxMOT algorithm (tracktrack, botsort, boosttrack, etc.).")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML configuration override path.")
    
    # Debug
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable performance diagnostics.")
    
    return parser.parse_args()


def run_pipeline(config: PipelineConfig) -> None:
    if config.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        # Note: Conf threshold is maintained low here to allow the tracker's YAML 
        # configuration (e.g., det_thresh: 0.4) to govern the gating logic.
        model_adapter = UltralyticsJDEAdapter(
            weights_path=config.yolo_model,
            conf_threshold=0.1, 
            classes=[0]
        )
        
        tracker_system = RetailTracker(
            detector=model_adapter,
            tracker_type=config.tracker_type,
            tracker_config=config.tracker_config,
            reid_weights=config.reid_model,
            device="cuda:0"
        )
    except Exception as e:
        logger.error("Pipeline initialization abort: %s", e)
        sys.exit(1)
    
    cap = cv2.VideoCapture(config.input_video)
    if not cap.isOpened():
        logger.error("Failed to open stream: %s", config.input_video)
        sys.exit(1)
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Dynamically read the native FPS of the source video
    source_fps = cap.get(cv2.CAP_PROP_FPS)

    # Use native FPS. Fallback to CLI target_fps ONLY if OpenCV fails to read the metadata (returns 0)
    fps = source_fps if source_fps > 0 else config.target_fps

    out = None
    if config.video_out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(config.video_out, fourcc, int(fps), (width, height))
        if not out.isOpened():
            cap.release()
            logger.error("Failed to open output writer: %s", config.video_out)
            sys.exit(1)

    visualize = out is not None or config.display

    logger.info("Pipeline Ready: %dx%d @ %.2f FPS", width, height, fps)
    logger.info("Detector: %s", config.yolo_model)
    logger.info(
        "Run mode: %s",
        "visualization mode" if visualize else "production inference mode"
    )
    if config.video_out:
        logger.info("Video output enabled: %s", config.video_out)
    else:
        logger.info("Video output disabled")
    if config.reid_model:
        logger.info("ReID mode: two-stage mode with external model %s", config.reid_model)
    else:
        logger.info("ReID mode: JDE mode / internal embeddings when provided by detector")

    frame_count = 0
    total_processing_time = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            start_time = time.time()

            tracks = tracker_system.process_frame(frame)

            if visualize:
                annotated_frame = tracker_system.draw_tracks(frame.copy(), tracks)
                if out is not None:
                    out.write(annotated_frame)
                if config.display:
                    cv2.imshow("Retail Tracking", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Display stopped by user.")
                        break

            processing_time = time.time() - start_time
            total_processing_time += processing_time

            if config.verbose:
                logger.debug("Frame %05d | Active Tracks: %03d | Latency: %.3fs", 
                             frame_count, len(tracks), processing_time)
            elif frame_count % 20 == 0: 
                print(f"\rProcessing frame {frame_count}...", end="", flush=True)
            
    except KeyboardInterrupt:
        logger.warning("\nUser interruption detected.")
    finally:
        cap.release()
        if out is not None:
            out.release()
        if config.display:
            cv2.destroyAllWindows()
        print()
        avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0.0
        logger.info("Execution finalized. Frames: %d | Average FPS: %.2f", frame_count, avg_fps)
        if config.video_out:
            logger.info("Artifact: %s", config.video_out)


if __name__ == "__main__":
    args = parse_arguments()
    try:
        runtime_config = PipelineConfig.from_args(args)
    except Exception as e:
        logger.error("Configuration validation failed: %s", e)
        sys.exit(1)
        
    run_pipeline(runtime_config)
