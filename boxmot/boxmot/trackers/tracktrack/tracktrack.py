import logging
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
        **kwargs,
    ):
        self.reid_model = kwargs.get("reid_model")
        self._warned_missing_embs = False
        self._frame_aux = {}
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
            data_path="Live"
        )
        self.tracker = CoreTrackTrack(tracktrack_args)

    def set_frame_aux(
        self,
        relaxed_detections: np.ndarray = None,
        relaxed_embeddings: np.ndarray = None,
        metadata: dict = None,
    ) -> None:
        self._frame_aux = {
            "relaxed_detections": relaxed_detections,
            "relaxed_embeddings": relaxed_embeddings,
            "metadata": metadata or {},
        }

    @staticmethod
    def _normalize_embeddings(embs: np.ndarray) -> np.ndarray:
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    def _prepare_embeddings(self, dets: np.ndarray, embs: np.ndarray) -> np.ndarray:
        if embs is None:
            if not self._warned_missing_embs:
                logger.warning(
                    "TrackTrack received detections without embeddings; using zero vectors, "
                    "so appearance matching is disabled for those detections. TODO: route "
                    "reid_model extraction here when JDE embeddings are unavailable."
                )
                self._warned_missing_embs = True
            return np.zeros((len(dets), 128), dtype=np.float32)

        embs = self._normalize_embeddings(embs)
        if embs.shape[0] != len(dets):
            raise ValueError(
                f"Embedding count ({embs.shape[0]}) does not match detection count ({len(dets)})."
            )
        return embs

    def _format_detections(self, dets: np.ndarray, embs: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.float32)
        embs = self._prepare_embeddings(dets, embs)
        return np.concatenate((dets, embs), axis=1)

    def _format_relaxed_detections(self, emb_dim: int) -> np.ndarray:
        relaxed_dets = self._frame_aux.get("relaxed_detections")
        if relaxed_dets is None or len(relaxed_dets) == 0:
            return np.empty((0, 6 + emb_dim), dtype=np.float32)

        relaxed_dets = np.asarray(relaxed_dets, dtype=np.float32)
        relaxed_embs = self._frame_aux.get("relaxed_embeddings")
        if relaxed_embs is None:
            relaxed_embs = np.zeros((len(relaxed_dets), emb_dim), dtype=np.float32)
        else:
            relaxed_embs = self._normalize_embeddings(relaxed_embs)
            if relaxed_embs.shape[0] != len(relaxed_dets):
                raise ValueError(
                    "Relaxed embedding count does not match relaxed detection count."
                )

        return np.concatenate((relaxed_dets, relaxed_embs), axis=1)

    def _update_impl(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        if len(dets) == 0:
            self.tracker.update_without_detections()
            self._frame_aux = {}
            return self.empty_output()

        # Format: [x1, y1, x2, y2, conf, cls, emb[0], emb[1]...]
        formatted_dets = self._format_detections(dets, embs)
        formatted_dets_95 = self._format_relaxed_detections(formatted_dets.shape[1] - 6)
        self._frame_aux = {}

        active_tracks = self.tracker.update(formatted_dets, formatted_dets_95)

        outputs = []
        for track in active_tracks:
            x1, y1, x2, y2 = track.x1y1x2y2
            outputs.append([x1, y1, x2, y2, track.track_id, track.score, track.cls, -1])

        return np.array(outputs) if len(outputs) > 0 else self.empty_output()
