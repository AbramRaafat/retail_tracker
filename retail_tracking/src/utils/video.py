import cv2
import os
import argparse
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def downsample_video(input_path: str, output_path: str, target_fps: int = 30) -> None:
    """
    Reduces the frame rate of a video by dropping frames uniformly.

    Args:
        input_path (str): Path to the source video file.
        output_path (str): Path where the downsampled video will be saved.
        target_fps (int): Desired frame rate. Must be greater than 0.

    Raises:
        FileNotFoundError: If the input video path does not exist.
        IOError: If the video source cannot be opened by OpenCV.
        ValueError: If target_fps is less than or equal to 0.
        
    Side Effects:
        Writes a new video file to the disk. Logs progress to standard output.
    """
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Could not open video source.")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info("Original Video: %dx%d @ %.2f FPS (%d frames)", width, height, orig_fps, total_frames)

    if target_fps >= orig_fps:
        logger.info("Target FPS (%d) >= Original FPS. Proceeding with step size 1.", target_fps)
        step = 1
    else:
        step = max(1, int(round(orig_fps / target_fps)))

    real_output_fps = orig_fps / step
    logger.info("Downsampling factor: %d. Effective Output FPS: %.2f", step, real_output_fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, real_output_fps, (width, height))

    count = 0
    saved_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % step == 0:
                out.write(frame)
                saved_count += 1

            count += 1
            
            if count % 50 == 0 and total_frames > 0:
                percent = (count / total_frames) * 100
                print(f"\rProcessing: {count}/{total_frames} ({percent:.1f}%)", end="", flush=True)

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Finalizing file write operations.")
        
    finally:
        cap.release()
        out.release()
        print() 
        logger.info("Saved to: %s. Wrote %d frames @ %.2f FPS", output_path, saved_count, real_output_fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample high-FPS video for tracking.")
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default=None, help="Path to output video (Optional)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (Default: 30)")

    args = parser.parse_args()

    out_path = args.output
    if out_path is None:
        base_name, ext = os.path.splitext(args.source)
        out_path = f"{base_name}_{args.fps}fps{ext}"

    try:
        downsample_video(args.source, out_path, args.fps)
    except Exception as e:
        logger.error("Execution failed: %s", e)