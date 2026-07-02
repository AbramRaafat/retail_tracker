import logging
from typing import Optional
import numpy as np
from types import SimpleNamespace
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.tracktrack.core.tracker import Tracker as CoreTrackTrack

logger = logging.getLogger(__name__)

class TrackTrack(BaseTracker):
    def __init__(
        self,
        det_thresh: float = 0.4,
        max_age: int = 60,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        match_thr: float = 0.8,
        penalty_p: float = 0.1,
        penalty_q: float = 0.2,
        reduce_step: float = 0.05,
        tai_thr: float = 0.4,
        init_thr: float = 0.6,
        cost_mode: str = "static",
        relaxed_association_mode: str = "recovery_only",
        relaxed_recovery_enabled: bool = True,
        relaxed_recovery_for_lost: bool = True,
        relaxed_recovery_for_unmatched_tracked: bool = False,
        relaxed_recovery_match_thr: float = 0.55,
        relaxed_recovery_penalty: float = 0.40,
        relaxed_recovery_freeze_feature_update: bool = True,
        **kwargs,
    ):
        self.reid_model = kwargs.get("reid_model")
        self._warned_missing_embs = False
        self._frame_aux = {}
        self.last_route_metadata = {}
        super().__init__(
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            **kwargs,
        )

        tracktrack_args = SimpleNamespace(
            det_thr=det_thresh,
            max_time_lost=max_age,
            match_thr=match_thr,
            penalty_p=penalty_p,
            penalty_q=penalty_q,
            reduce_step=reduce_step,
            tai_thr=tai_thr,
            init_thr=init_thr,
            min_len=min_hits,
            cost_mode=cost_mode,
            data_path="Live",
            relaxed_association_mode=relaxed_association_mode,
            relaxed_recovery_enabled=relaxed_recovery_enabled,
            relaxed_recovery_for_lost=relaxed_recovery_for_lost,
            relaxed_recovery_for_unmatched_tracked=relaxed_recovery_for_unmatched_tracked,
            relaxed_recovery_match_thr=relaxed_recovery_match_thr,
            relaxed_recovery_penalty=relaxed_recovery_penalty,
            relaxed_recovery_freeze_feature_update=relaxed_recovery_freeze_feature_update,
        )
        self.tracker = CoreTrackTrack(tracktrack_args)

    def set_frame_aux(
        self,
        relaxed_detections: np.ndarray = None,
        relaxed_embeddings: np.ndarray = None,
        metadata: dict = None,
        appearance_mode: str = "auto",
        allow_zero_embs: bool = False,
    ) -> None:
        self._frame_aux = {
            "relaxed_detections": relaxed_detections,
            "relaxed_embeddings": relaxed_embeddings,
            "metadata": metadata or {},
            "appearance_mode": appearance_mode,
            "allow_zero_embs": allow_zero_embs,
        }

    def set_audit_params(
        self,
        assoc_debug_csv: Optional[str] = None,
        assoc_debug_max_frames: Optional[int] = None,
        assoc_debug_summary: Optional[str] = None,
    ) -> None:
        if assoc_debug_csv:
            try:
                from retail_tracking.src.utils.audit_writer import AssociationAuditWriter
            except ImportError:
                from src.utils.audit_writer import AssociationAuditWriter
            
            self.audit_writer = AssociationAuditWriter(
                csv_path=assoc_debug_csv,
                max_frames=assoc_debug_max_frames,
                summary_path=assoc_debug_summary,
            )
            self.tracker.audit_writer = self.audit_writer
        else:
            self.audit_writer = None
            self.tracker.audit_writer = None

    def close_audit(self) -> None:
        if hasattr(self, "audit_writer") and self.audit_writer is not None:
            self.audit_writer.close()

    @staticmethod
    def _normalize_embeddings(embs: np.ndarray) -> np.ndarray:
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    def _extract_external_reid_embeddings(self, img: np.ndarray, dets: np.ndarray) -> Optional[np.ndarray]:
        if self.reid_model is None:
            return None
        
        if not hasattr(self.reid_model, "get_features"):
            raise NotImplementedError(
                "The loaded ReID backend model does not expose a 'get_features' method."
            )
            
        if len(dets) == 0:
            return np.empty((0, 128), dtype=np.float32)

        h, w = img.shape[:2]
        clipped_xyxys = []
        for det in dets:
            x1, y1, x2, y2 = det[:4]
            # Clip bounds
            cx1 = max(0.0, min(float(x1), float(w)))
            cy1 = max(0.0, min(float(y1), float(h)))
            cx2 = max(0.0, min(float(x2), float(w)))
            cy2 = max(0.0, min(float(y2), float(h)))
            
            # If invalid (e.g., zero or negative width/height), replace with a safe minimal 1x1 box at bounds
            if cx2 <= cx1 or cy2 <= cy1:
                cx1, cy1, cx2, cy2 = 0.0, 0.0, 1.0, 1.0
                
            clipped_xyxys.append([cx1, cy1, cx2, cy2])
            
        clipped_xyxys = np.array(clipped_xyxys, dtype=np.float32)
        
        # Extract features using get_features
        features = self.reid_model.get_features(clipped_xyxys, img)
        return features

    def resolve_embeddings(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: Optional[np.ndarray],
        appearance_mode: str,
        allow_zero_embs: bool,
    ) -> np.ndarray:
        if len(dets) == 0:
            return np.empty((0, 128), dtype=np.float32)

        resolved_embs = None
        embedding_route = "none"
        used_zero_embs = False

        if appearance_mode == "none":
            emb_dim = embs.shape[1] if embs is not None else 128
            resolved_embs = np.zeros((len(dets), emb_dim), dtype=np.float32)
            embedding_route = "none"
            used_zero_embs = True
            if not getattr(self, "_warned_appearance_disabled", False):
                logger.warning("WARNING: appearance matching disabled; TrackTrack will use zero embeddings. This is for ablation only.")
                self._warned_appearance_disabled = True

        elif appearance_mode == "jde":
            if embs is not None:
                resolved_embs = embs
                embedding_route = "jde"
            else:
                if allow_zero_embs:
                    resolved_embs = np.zeros((len(dets), 128), dtype=np.float32)
                    embedding_route = "none"
                    used_zero_embs = True
                else:
                    raise RuntimeError("JDE embeddings are required in 'jde' mode but were not provided by the detector.")

        elif appearance_mode == "auto":
            if embs is not None:
                resolved_embs = embs
                embedding_route = "jde"
            elif self.reid_model is not None:
                resolved_embs = self._extract_external_reid_embeddings(img, dets)
                if resolved_embs is not None and resolved_embs.size > 0:
                    embedding_route = "external_reid"
                else:
                    if allow_zero_embs:
                        resolved_embs = np.zeros((len(dets), 128), dtype=np.float32)
                        embedding_route = "none"
                        used_zero_embs = True
                    else:
                        raise RuntimeError(
                            "JDE embeddings were not returned, and external ReID extraction failed to return embeddings.\n"
                            "Check the ReID backend interface, provide valid --reid weights, or use --appearance-mode none only for ablation."
                        )
            else:
                if allow_zero_embs:
                    resolved_embs = np.zeros((len(dets), 128), dtype=np.float32)
                    embedding_route = "none"
                    used_zero_embs = True
                else:
                    raise RuntimeError(
                        "JDE embeddings were not returned by the detector and no external ReID model was provided.\n"
                        "Use --appearance-mode none only for ablation, or provide --reid, or fix JDE embedding extraction."
                    )

        elif appearance_mode == "external":
            if self.reid_model is not None:
                resolved_embs = self._extract_external_reid_embeddings(img, dets)
                if resolved_embs is not None and resolved_embs.size > 0:
                    embedding_route = "external_reid"
                else:
                    if allow_zero_embs:
                        resolved_embs = np.zeros((len(dets), 128), dtype=np.float32)
                        embedding_route = "none"
                        used_zero_embs = True
                    else:
                        raise RuntimeError("External ReID extraction failed to return embeddings in 'external' mode.")
            else:
                if allow_zero_embs:
                    resolved_embs = np.zeros((len(dets), 128), dtype=np.float32)
                    embedding_route = "none"
                    used_zero_embs = True
                else:
                    raise RuntimeError("--appearance-mode external requires --reid.")
        else:
            raise ValueError(f"Unknown appearance_mode: {appearance_mode}")

        normalized_embs = self._normalize_embeddings(resolved_embs)

        self.last_route_metadata = {
            "appearance_mode": appearance_mode,
            "embedding_route": embedding_route,
            "embedding_shape": tuple(normalized_embs.shape),
            "used_zero_embeddings": used_zero_embs,
            "num_detections": int(len(dets)),
            "num_relaxed_detections": 0,
            "has_relaxed_embeddings": False,
            "relaxed_embedding_shape": None,
        }

        return normalized_embs

    def _format_relaxed_detections(self, img: np.ndarray, emb_dim: int) -> np.ndarray:
        relaxed_dets = self._frame_aux.get("relaxed_detections")
        if relaxed_dets is None or len(relaxed_dets) == 0:
            return np.empty((0, 6 + emb_dim), dtype=np.float32)

        relaxed_dets = np.asarray(relaxed_dets, dtype=np.float32)
        relaxed_embs = self._frame_aux.get("relaxed_embeddings")
        
        appearance_mode = self._frame_aux.get("appearance_mode", "auto")
        allow_zero_embs = self._frame_aux.get("allow_zero_embs", False)
        
        normal_route = self.last_route_metadata.get("embedding_route", "none")
        
        resolved_relaxed_embs = None
        has_relaxed_embs = False
        
        if appearance_mode == "none":
            resolved_relaxed_embs = np.zeros((len(relaxed_dets), emb_dim), dtype=np.float32)
        elif appearance_mode == "jde":
            if relaxed_embs is not None:
                resolved_relaxed_embs = relaxed_embs
                has_relaxed_embs = True
            else:
                if allow_zero_embs:
                    resolved_relaxed_embs = np.zeros((len(relaxed_dets), emb_dim), dtype=np.float32)
                else:
                    raise RuntimeError("Relaxed detections exist but relaxed JDE embeddings are missing in 'jde' mode.")
        elif appearance_mode == "auto":
            if normal_route == "jde":
                if relaxed_embs is not None:
                    resolved_relaxed_embs = relaxed_embs
                    has_relaxed_embs = True
                else:
                    if allow_zero_embs:
                        resolved_relaxed_embs = np.zeros((len(relaxed_dets), emb_dim), dtype=np.float32)
                    else:
                        raise RuntimeError("Relaxed detections exist but relaxed JDE embeddings are missing in 'auto' mode when JDE is active.")
            elif normal_route == "external_reid":
                if self.reid_model is not None:
                    resolved_relaxed_embs = self._extract_external_reid_embeddings(img, relaxed_dets)
                    if resolved_relaxed_embs is not None and resolved_relaxed_embs.size > 0:
                        has_relaxed_embs = True
                    else:
                        if allow_zero_embs:
                            resolved_relaxed_embs = np.zeros((len(relaxed_dets), emb_dim), dtype=np.float32)
                        else:
                            raise NotImplementedError("External ReID extraction failed or is not implemented for relaxed detections in 'auto' mode.")
                else:
                    if allow_zero_embs:
                        resolved_relaxed_embs = np.zeros((len(relaxed_dets), emb_dim), dtype=np.float32)
                    else:
                        raise NotImplementedError("External ReID model is missing for relaxed detections in 'auto' mode.")
            else:
                if allow_zero_embs:
                    resolved_relaxed_embs = np.zeros((len(relaxed_dets), emb_dim), dtype=np.float32)
                else:
                    raise RuntimeError("Normal route had no embeddings, and allow_zero_embs is False.")
        elif appearance_mode == "external":
            if self.reid_model is not None:
                resolved_relaxed_embs = self._extract_external_reid_embeddings(img, relaxed_dets)
                if resolved_relaxed_embs is not None and resolved_relaxed_embs.size > 0:
                    has_relaxed_embs = True
                else:
                    if allow_zero_embs:
                        resolved_relaxed_embs = np.zeros((len(relaxed_dets), emb_dim), dtype=np.float32)
                    else:
                        raise NotImplementedError("External ReID extraction failed or is not implemented for relaxed detections in 'external' mode.")
            else:
                if allow_zero_embs:
                    resolved_relaxed_embs = np.zeros((len(relaxed_dets), emb_dim), dtype=np.float32)
                else:
                    raise RuntimeError("--appearance-mode external requires --reid.")
        else:
            raise ValueError(f"Unknown appearance_mode: {appearance_mode}")

        resolved_relaxed_embs = self._normalize_embeddings(resolved_relaxed_embs)
        
        if resolved_relaxed_embs.shape[0] != len(relaxed_dets):
            raise ValueError("Relaxed embedding count does not match relaxed detection count.")

        self.last_route_metadata["has_relaxed_embeddings"] = has_relaxed_embs
        self.last_route_metadata["relaxed_embedding_shape"] = tuple(resolved_relaxed_embs.shape) if has_relaxed_embs else None

        return np.concatenate((relaxed_dets, resolved_relaxed_embs), axis=1)

    def _update_impl(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        appearance_mode = self._frame_aux.get("appearance_mode", "auto")
        allow_zero_embs = self._frame_aux.get("allow_zero_embs", False)

        if len(dets) == 0:
            self.tracker.update_without_detections()
            self.last_route_metadata = {
                "appearance_mode": appearance_mode,
                "embedding_route": "none",
                "embedding_shape": (0, 128),
                "used_zero_embeddings": False,
                "num_detections": 0,
                "num_relaxed_detections": 0,
                "has_relaxed_embeddings": False,
                "relaxed_embedding_shape": None,
            }
            self._frame_aux = {}
            return self.empty_output()

        resolved_embs = self.resolve_embeddings(
            dets=dets,
            img=img,
            embs=embs,
            appearance_mode=appearance_mode,
            allow_zero_embs=allow_zero_embs,
        )

        # Initialize relaxed metadata fields before formatting
        self.last_route_metadata["num_relaxed_detections"] = 0
        self.last_route_metadata["has_relaxed_embeddings"] = False
        self.last_route_metadata["relaxed_embedding_shape"] = None

        # Format: [x1, y1, x2, y2, conf, cls, emb[0], emb[1]...]
        formatted_dets = np.concatenate((dets, resolved_embs), axis=1)
        formatted_dets_95 = self._format_relaxed_detections(img, formatted_dets.shape[1] - 6)
        
        self.last_route_metadata["num_relaxed_detections"] = int(len(formatted_dets_95))

        self._frame_aux = {}

        active_tracks = self.tracker.update(formatted_dets, formatted_dets_95)

        outputs = []
        for track in active_tracks:
            x1, y1, x2, y2 = track.x1y1x2y2
            outputs.append([x1, y1, x2, y2, track.track_id, track.score, track.cls, -1])

        return np.array(outputs) if len(outputs) > 0 else self.empty_output()
