import cv2
import logging
import numpy as np
import sys
import time
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
                 device: str = 'cuda:0',
                 half: bool = True,
                 appearance_mode: str = "auto",
                 allow_zero_embs: bool = False,
                 assoc_debug_csv: Optional[str] = None,
                 assoc_debug_max_frames: Optional[int] = None,
                 assoc_debug_summary: Optional[str] = None,
                 profile_timing: bool = False):
        """
        Initializes the tracking ecosystem via BoxMOT factory methods.
        
        Args:
            detector: Adapter handling YOLO inference.
            tracker_type: Target algorithm (e.g., 'tracktrack', 'botsort').
            tracker_config: Path to custom YAML. If None, BoxMOT loads framework defaults.
            reid_weights: Path to external ReID model. Required if utilizing a standard 
                          YOLO detector instead of a JDE model.
            device: Compute device mapping for ReID tensor operations.
            half: Whether tracker-owned ReID tensor operations may use half precision.
            appearance_mode: Appearance routing mode ('auto', 'jde', 'external', 'none').
            allow_zero_embs: Allow using zero embeddings if JDE/external ReID is unavailable.
            assoc_debug_csv: Optional path for association debugging output.
            assoc_debug_max_frames: Max frames to record for association debugging.
            assoc_debug_summary: Optional path for association summary output.
            profile_timing: Record lightweight per-frame detector/tracker timings.
        """
        if appearance_mode == "external" and not reid_weights:
            raise RuntimeError("--appearance-mode external requires --reid.")

        self.tracker_type = tracker_type
        self.detector = detector
        self.appearance_mode = appearance_mode
        self.allow_zero_embs = allow_zero_embs
        self.profile_timing = profile_timing
        self.last_jde_metadata = {}
        self.last_route_metadata = {}
        self.last_timing_metadata = {}
        
        # BoxMOT Factory: Automatically loads default YAMLs if tracker_config is None.
        # Warms up ReID models internally if reid_weights is provided.
        try:
            from boxmot.trackers.tracker_zoo import create_tracker

            self.tracker = create_tracker(
                tracker_type=tracker_type,
                tracker_config=tracker_config,
                reid_weights=Path(reid_weights) if reid_weights else None,
                device=device,
                half=half
            )

            # Thread debug audit parameters to the tracker if it supports it
            if hasattr(self.tracker, "set_audit_params"):
                self.tracker.set_audit_params(
                    assoc_debug_csv=assoc_debug_csv,
                    assoc_debug_max_frames=assoc_debug_max_frames,
                    assoc_debug_summary=assoc_debug_summary,
                )

            # Print relaxed tracking configuration for reproducibility
            logger.info("==================================================")
            logger.info("TrackTrack Tracker Startup Configuration:")
            logger.info("  tracker_config: %s", tracker_config)
            logger.info("  relaxed_dets_enabled: %s", getattr(self.detector, "relaxed_enabled", "N/A"))
            logger.info("  relaxed_source: %s", getattr(self.detector, "relaxed_source", "N/A"))
            core_args = getattr(getattr(self.tracker, "tracker", None), "args", None)
            if core_args is not None:
                cost_mode = getattr(core_args, "cost_mode", "static")
                if cost_mode == "static_shadow_mahalanobis":
                    behavior_mode = "shadow-only"
                elif cost_mode == "static_mahalanobis_gate":
                    behavior_mode = "mahalanobis-gate"
                elif cost_mode == "static_mahalanobis_blend":
                    behavior_mode = "mahalanobis-blend"
                else:
                    behavior_mode = "static"
                logger.info("  cost_mode: %s", cost_mode)
                logger.info("  behavior_mode: %s", behavior_mode)
                logger.info("  relaxed_association_mode: %s", getattr(core_args, "relaxed_association_mode", "N/A"))
                logger.info("  relaxed_recovery_enabled: %s", getattr(core_args, "relaxed_recovery_enabled", "N/A"))
                logger.info("  relaxed_recovery_for_lost: %s", getattr(core_args, "relaxed_recovery_for_lost", "N/A"))
                logger.info("  relaxed_recovery_for_unmatched_tracked: %s", getattr(core_args, "relaxed_recovery_for_unmatched_tracked", "N/A"))
                logger.info("  relaxed_recovery_freeze_feature_update: %s", getattr(core_args, "relaxed_recovery_freeze_feature_update", "N/A"))
                logger.info("  mahalanobis_enabled: %s", getattr(core_args, "mahalanobis_enabled", "N/A"))
                logger.info("  mahalanobis_gate_dim: %s", getattr(core_args, "mahalanobis_gate_dim", "N/A"))
                logger.info("  mahalanobis_gate_confidence: %s", getattr(core_args, "mahalanobis_gate_confidence", "N/A"))
                logger.info("  mahalanobis_gate_threshold: %s", getattr(core_args, "mahalanobis_gate_threshold", "N/A"))
                logger.info("  mahalanobis_apply_to_states: %s", getattr(core_args, "mahalanobis_apply_to_states", "N/A"))
                logger.info("  mahalanobis_apply_to_tiers: %s", getattr(core_args, "mahalanobis_apply_to_tiers", "N/A"))
                logger.info("  mahalanobis_fail_open: %s", getattr(core_args, "mahalanobis_fail_open", "N/A"))
                logger.info("  mahalanobis_weight: %s", getattr(core_args, "mahalanobis_weight", "N/A"))
            logger.info("==================================================")
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate tracker '{tracker_type}': {e}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Executes detector inference and routes state updates.
        """
        profile_timing = getattr(self, "profile_timing", False)
        process_start = time.perf_counter() if profile_timing else None

        detector_start = time.perf_counter() if profile_timing else None
        jde_result = self.detector.predict(frame)
        detector_total_ms = ((time.perf_counter() - detector_start) * 1000.0) if profile_timing else 0.0
        self.last_jde_metadata = jde_result.metadata or {}

        dets = jde_result.detections
        if dets is None or dets.shape[0] == 0:
            dets = np.empty((0, 6), dtype=np.float32)

        if hasattr(self.tracker, "set_frame_aux"):
            self.tracker.set_frame_aux(
                relaxed_detections=jde_result.relaxed_detections,
                relaxed_embeddings=jde_result.relaxed_embeddings,
                metadata=jde_result.metadata,
                appearance_mode=self.appearance_mode,
                allow_zero_embs=self.allow_zero_embs,
            )

        embs = jde_result.embeddings
        if self.appearance_mode in ("external", "none"):
            embs = None

        # Dynamic Routing: 
        # Appearance routing is explicit. TrackTrack uses JDE embeddings by default.
        # External ReID fallback is only used when appearance_mode requests it and a ReID backend is available.
        tracker_start = time.perf_counter() if profile_timing else None
        if embs is not None:
            tracker_outputs = self.tracker.update(dets, frame, embs=embs)
        else:
            tracker_outputs = self.tracker.update(dets, frame)
        tracker_update_ms = ((time.perf_counter() - tracker_start) * 1000.0) if profile_timing else 0.0
        
        if hasattr(self.tracker, "last_route_metadata"):
            self.last_route_metadata = self.tracker.last_route_metadata

        if profile_timing:
            detector_timing = getattr(self.detector, "last_timing", {}) or {}
            self.last_timing_metadata = {
                "detector_total_ms": detector_total_ms,
                "detector_normal_ms": float(detector_timing.get("detector_normal_ms", 0.0) or 0.0),
                "detector_relaxed_ms": float(detector_timing.get("detector_relaxed_ms", 0.0) or 0.0),
                "detector_superset_ms": float(detector_timing.get("detector_superset_ms", 0.0) or 0.0),
                "software_split_nms_ms": float(detector_timing.get("software_split_nms_ms", 0.0) or 0.0),
                "tracker_update_ms": tracker_update_ms,
                "process_frame_ms": (time.perf_counter() - process_start) * 1000.0,
            }
        
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
