import cv2
import logging
import numpy as np
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BOXMOT_ROOT = PROJECT_ROOT / "boxmot"
if str(BOXMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOXMOT_ROOT))

try:
    from retail_tracking.src.detection.adapters import BaseJDEAdapter
except ImportError:
    from src.detection.adapters import BaseJDEAdapter

logger = logging.getLogger(__name__)


class RetailTracker:
    """
    Orchestrates Multi-Object Tracking algorithms via BoxMOT.
    Supports dynamic module loading and hybrid JDE/Two-Stage feature extraction.
    """

    def __init__(self, 
                 detector: BaseJDEAdapter,
                 tracker_type: str = 'tracktrack', 
                 tracker_config: Optional[str] = None,
                 reid_weights: Optional[str] = None,
                 device: str = 'cuda:0'):
        """
        Initializes the tracking ecosystem via BoxMOT factory methods.
        
        Args:
            detector: Adapter handling YOLO inference.
            tracker_type: Target algorithm (e.g., 'tracktrack', 'botsort').
            tracker_config: Path to custom YAML. If None, BoxMOT loads framework defaults.
            reid_weights: Path to external ReID model. Required if utilizing a standard 
                          YOLO detector instead of a JDE model.
            device: Compute device mapping for ReID tensor operations.
        """
        self.tracker_type = tracker_type
        self.detector = detector
        
        # BoxMOT Factory: Automatically loads default YAMLs if tracker_config is None.
        # Warms up ReID models internally if reid_weights is provided.
        try:
            from boxmot.trackers.tracker_zoo import create_tracker

            self.tracker = create_tracker(
                tracker_type=tracker_type,
                tracker_config=tracker_config,
                reid_weights=Path(reid_weights) if reid_weights else None,
                device=device,
                half=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate tracker '{tracker_type}': {e}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Executes detector inference and routes state updates.
        """
        jde_result = self.detector.predict(frame)

        dets = jde_result.detections
        if dets is None or dets.shape[0] == 0:
            dets = np.empty((0, 6), dtype=np.float32)

        if hasattr(self.tracker, "set_frame_aux"):
            self.tracker.set_frame_aux(
                relaxed_detections=jde_result.relaxed_detections,
                relaxed_embeddings=jde_result.relaxed_embeddings,
                metadata=jde_result.metadata,
            )

        embs = jde_result.embeddings

        # Dynamic Routing: 
        # If JDE embeddings exist, inject them directly to bypass BoxMOT ReID.
        # If None, BoxMOT utilizes its initialized `reid_weights` to extract them natively.
        if embs is not None:
            tracker_outputs = self.tracker.update(dets, frame, embs=embs)
        else:
            tracker_outputs = self.tracker.update(dets, frame)
        
        return tracker_outputs if len(tracker_outputs) > 0 else np.empty((0, 8))

    def draw_tracks(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        for t in tracks:
            x1, y1, x2, y2 = map(int, t[:4])
            track_id = int(t[4])

            color = (0, 255, 0) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID: {track_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame
