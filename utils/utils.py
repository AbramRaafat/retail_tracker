import cv2
import os
import argparse
import sys
print("ok")
def downsample_video(input_path, output_path, target_fps=30):
    # Safety Checks
    if not os.path.exists(input_path):
        print(f" Error: Input file not found at: {input_path}")
        return

    # Open the video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f" Error: Could not open video source.")
        return

    # Get original properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f" Original: {width}x{height} @ {orig_fps:.2f} FPS ({total_frames} frames)")

    # Calculate Step Size (Skip Factor)
    if target_fps >= orig_fps:
        print(f"Target FPS ({target_fps}) is >= Original FPS. No downsampling needed.")
        step = 1
    else:
        step = int(round(orig_fps / target_fps))
        if step < 1: step = 1
    

    real_output_fps = orig_fps / step
    
    print(f" Downsampling by factor of {step} (Keep 1, Skip {step-1})")
    print(f" Effective Output FPS: {real_output_fps:.2f}")

    # Video Writer
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
            
            if count % 50 == 0:
                percent = (count / total_frames) * 100 if total_frames > 0 else 0
                sys.stdout.write(f"\r Processing: {count}/{total_frames} ({percent:.1f}%)")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n Interrupted by user. Saving what we have...")
        
    finally:
        cap.release()
        out.release()
        print(f"\n\Done! Saved to: {output_path}")
        print(f"Stats: {saved_count} frames written @ {real_output_fps:.2f} FPS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample high-FPS video for tracking.")
    
    # Arguments
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default=None, help="Path to output video (Optional)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (Default: 30)")

    args = parser.parse_args()

    # Auto-generate output filename if not provided
    # e.g., "video.mp4" -> "video_30fps.mp4"
    if args.output is None:
        base_name, ext = os.path.splitext(args.source)
        args.output = f"{base_name}_{args.fps}fps{ext}"

    downsample_video(args.source, args.output, args.fps)