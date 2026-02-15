#run.py
from tracker import RetailTracker
import cv2
from config import INPUT_VIDEO, OUTPUT_VIDEO, YOLO_MODEL

from tracker import RetailTracker
import cv2
from config import INPUT_VIDEO, OUTPUT_VIDEO, YOLO_MODEL
print("Starting Retail Tracking Pipeline...")
def run_pipeline():
    # tracking algorithm 
    # 'DeepOcSort' (good occlusion recovery)
    # 'BoostTrack' (SOTA model in MOT benchmark)
    CURRENT_ALGORITHM = 'BoostTrack' 

    tracker_system = RetailTracker(
        tracker_type=CURRENT_ALGORITHM,
        yolo_weights=YOLO_MODEL,
        det_thresh=0.4,   
        max_age=60    
    )
    
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Could not open video at {INPUT_VIDEO}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    print(f" Processing video: {width}x{height} @ {fps}fps")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 20 == 0: print(f"Processing frame {frame_count}...", end='\r')
        
        tracks = tracker_system.process_frame(frame)
        annotated_frame = tracker_system.draw_tracks(frame, tracks)
        
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"\n Done! Video saved to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    run_pipeline()
