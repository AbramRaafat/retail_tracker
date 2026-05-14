import numpy as np
from types import SimpleNamespace
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.tracktrack.core.tracker import Tracker as CoreTrackTrack

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
        **kwargs,
    ):
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
            data_path="Live"
        )
        self.tracker = CoreTrackTrack(tracktrack_args)

    def _update_impl(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        if len(dets) == 0:
            self.tracker.update_without_detections()
            return self.empty_output()

        if embs is None:
            embs = np.zeros((len(dets), 128))

        # Format: [x1, y1, x2, y2, conf, cls, emb[0], emb[1]...]
        formatted_dets = np.concatenate((dets, embs), axis=1)
        formatted_dets_95 = np.empty((0, formatted_dets.shape[1]))

        active_tracks = self.tracker.update(formatted_dets, formatted_dets_95)

        outputs = []
        for track in active_tracks:
            x1, y1, x2, y2 = track.x1y1x2y2
            outputs.append([x1, y1, x2, y2, track.track_id, track.score, dets[0, 5], -1])

        return np.array(outputs) if len(outputs) > 0 else self.empty_output()