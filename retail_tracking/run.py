import cv2
import sys
import argparse
import logging
import time
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
    parser.add_argument("-o", "--output", type=str, required=True, help="Output video destination.")
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
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config.output_video, fourcc, int(fps), (width, height))
    
    logger.info("Pipeline Ready: %dx%d @ %d FPS", width, height, fps)
    logger.info("Detector: %s", config.yolo_model)
    if config.reid_model:
        logger.info("ReID Extractor: %s (Two-Stage Architecture Active)", config.reid_model)
    else:
        logger.info("ReID Extractor: Internal (JDE Architecture Active)")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            start_time = time.time()
            
            tracks = tracker_system.process_frame(frame)
            annotated_frame = tracker_system.draw_tracks(frame, tracks)
            out.write(annotated_frame)
            
            processing_time = time.time() - start_time
            
            if config.verbose:
                logger.debug("Frame %05d | Active Tracks: %03d | Latency: %.3fs", 
                             frame_count, len(tracks), processing_time)
            elif frame_count % 20 == 0: 
                print(f"\rProcessing frame {frame_count}...", end="", flush=True)
            
    except KeyboardInterrupt:
        logger.warning("\nUser interruption detected.")
    finally:
        cap.release()
        out.release()
        print()
        logger.info("Execution finalized. Artifact: %s", config.output_video)


if __name__ == "__main__":
    args = parse_arguments()
    try:
        runtime_config = PipelineConfig.from_args(args)
    except Exception as e:
        logger.error("Configuration validation failed: %s", e)
        sys.exit(1)
        
    run_pipeline(runtime_config)