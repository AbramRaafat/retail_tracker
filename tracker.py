#tracker.py
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import DeepOcSort, BoostTrack


class RetailTracker:
    def __init__(self, 
                 tracker_type='DeepOcSort', 
                 yolo_weights='yolo11n.pt', 
                 reid_weights='osnet_x1_0_msmt17.pt', 
                 device=0,
                 #tracker global Hyperparameters
                 det_thresh=0.4,
                 max_age=60,       
                 min_hits=3,         
                 iou_threshold=0.3):

        self.device = device
        self.tracker_type = tracker_type
        
        # Initialize YOLO
        print(f"Loading Detector: {yolo_weights}...")
        self.detector = YOLO(yolo_weights)
        
        # Prepare Paths & Config
        reid_path = Path(reid_weights)
        print(f"Loading Tracker: {tracker_type} with {reid_weights}...")
        
        # Tracker Initialization
        if tracker_type == 'DeepOcSort':
            
            self.tracker = DeepOcSort(
                reid_weights=reid_path,
                device=device,
                half=True,
                det_thresh=det_thresh,
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold,
                delta_t=3,           # Velocity calculation window
                inertia=0.2,         # Low inertia allows quick turns in aisles
                asso_func="giou"     # Generalized IoU helps with non-overlapping boxes
            )
        
        elif tracker_type == 'BoostTrack':

            self.tracker = BoostTrack(
                reid_weights=reid_path,
                device=device,
                half=True,
                det_thresh=det_thresh,
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold,
                with_reid=True,      
                use_ecc=True,       
                lambda_iou=0.5,    
                use_dlo_boost=True, 
                asso_func="iou" 
            )
            
        else:
            raise ValueError(f"Unknown tracker type: {tracker_type}")
            
        print(f"System Initialized: {tracker_type}")

    def process_frame(self, frame):
        """
        Standardized processing pipeline
        """
        # Detect (YOLO)
        # Use slightly lower conf (0.35) than tracker (0.4) to allow tracker to filter
        results = self.detector.predict(frame, classes=[0], conf=0.35, verbose=False)
        
        dets = results[0].boxes.data.cpu().numpy()
        
        if len(dets) == 0:
            dets = np.empty((0, 6))

        tracker_outputs = self.tracker.update(dets, frame)
        
        if len(tracker_outputs) == 0:
            return np.empty((0, 8))
            
        return tracker_outputs

    def draw_tracks(self, frame, tracks):
        """
        Visualizes the tracks.
        BoxMOT Output Format: [x1, y1, x2, y2, id, conf, cls, det_ind]
        """
        for t in tracks:
            # Extract coordinates and ID
            x1, y1, x2, y2 = map(int, t[:4])
            id = int(t[4])

            color = (0, 255, 0) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID: {id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)

            cv2.putText(frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame