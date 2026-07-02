import logging
import time
from contextlib import nullcontext
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
from ultralytics import YOLO

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency in smoke contexts
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class JDEResult:
    detections: np.ndarray
    embeddings: Optional[np.ndarray] = None
    relaxed_detections: Optional[np.ndarray] = None
    relaxed_embeddings: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)




class BaseJDEAdapter(ABC):
    @abstractmethod
    def predict(self, frame: np.ndarray) -> JDEResult:
        pass


class UltralyticsJDEAdapter(BaseJDEAdapter):
    """
    Adapter bridging Ultralytics models with the tracking pipeline.
    Seamlessly supports both specialized JDE architectures (extracting appearance tensors)
    and standard YOLO detection models (returning None for embeddings).
    """
    
    def __init__(
        self,
        weights_path: str,
        device: str = "cuda:0",
        half: bool = True,
        imgsz: int = 1280,
        conf_threshold: float = 0.35,
        classes: Optional[List[int]] = None,
        relaxed_enabled: bool = False,
        relaxed_conf_threshold: float = 0.03,
        relaxed_iou_threshold: float = 0.95,
        normal_iou_threshold: float = 0.70,
        relaxed_source: str = "two-pass",
        profile_timing: bool = False,
    ):
        if relaxed_source not in {"two-pass", "single-pass"}:
            raise ValueError("relaxed_source must be either 'two-pass' or 'single-pass'.")
        self.detector = YOLO(weights_path)
        self.device = device
        self.half = half
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.classes = classes if classes is not None else [0]
        self.relaxed_enabled = relaxed_enabled
        self.relaxed_conf_threshold = relaxed_conf_threshold
        self.relaxed_iou_threshold = relaxed_iou_threshold
        self.normal_iou_threshold = normal_iou_threshold
        self.relaxed_source = relaxed_source
        self.profile_timing = profile_timing
        self.last_timing = {}

    def warmup(self, frame_shape: tuple[int, int, int] | None = None) -> None:
        shape = frame_shape if frame_shape is not None else (640, 640, 3)
        dummy = np.zeros(shape, dtype=np.uint8)
        try:
            self.predict(dummy)
        except Exception as exc:
            logger.warning("Detector warmup failed; continuing without warmup: %s", exc)

    def _run_predict(self, frame: np.ndarray, conf: float, iou: float):
        context = torch.inference_mode() if torch is not None else nullcontext()
        with context:
            results = self.detector.predict(
                frame,
                classes=self.classes,
                conf=conf,
                iou=iou,
                device=self.device,
                half=self.half,
                imgsz=self.imgsz,
                verbose=False,
            )
        if not results or len(results[0].boxes) == 0:
            return None
        return results[0]

    def _sync_for_timing(self) -> None:
        if not self.profile_timing or torch is None or not self.device.lower().startswith("cuda"):
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _timed_run_predict(self, frame: np.ndarray, conf: float, iou: float):
        self._sync_for_timing()
        start = time.perf_counter()
        res = self._run_predict(frame, conf, iou)
        self._sync_for_timing()
        return res, (time.perf_counter() - start) * 1000.0

    def _extract_dets_embs(self, res) -> tuple[np.ndarray, Optional[np.ndarray], str]:
        if res is None or len(res.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32), None, "none"

        raw_dets = res.boxes.data.cpu().numpy()
        dets = np.asarray(raw_dets[:, :6], dtype=np.float32)
        embs = None
        embedding_source = "none"
        
        # Sequentially evaluate attributes for JDE embedding vectors.
        if hasattr(res, 'embeds') and res.embeds is not None:
            embs = res.embeds.data.cpu().numpy() if hasattr(res.embeds, 'data') else res.embeds.cpu().numpy()
            embedding_source = "res.embeds"
        elif raw_dets.shape[1] > 6:
            embs = raw_dets[:, 6:]
            embedding_source = "raw_dets_extra_columns"
        elif hasattr(res, 'embeddings') and res.embeddings is not None:
            embs = res.embeddings.data.cpu().numpy() if hasattr(res.embeddings, 'data') else res.embeddings.cpu().numpy()
            embedding_source = "res.embeddings"
        elif hasattr(res.boxes, 'embs') and res.boxes.embs is not None:
            embs = res.boxes.embs.data.cpu().numpy() if hasattr(res.boxes.embs, 'data') else res.boxes.embs.cpu().numpy()
            embedding_source = "res.boxes.embs"
            
        if embs is not None:
            embs = np.asarray(embs, dtype=np.float32)
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)

        # Validate spatial-appearance mapping constraint
        if embs is not None and embs.shape[0] != dets.shape[0]:
            logger.warning("Dimension mismatch: Detections (%d) vs Embeddings (%d). Dropping embeddings.", 
                           dets.shape[0], embs.shape[0])
            embs = None
            embedding_source = "none"

        return dets, embs, embedding_source

    @staticmethod
    def _nms_indices(dets: np.ndarray, iou_threshold: float) -> np.ndarray:
        if dets.shape[0] == 0:
            return np.empty((0,), dtype=np.int64)

        boxes = np.asarray(dets[:, :4], dtype=np.float32)
        scores = np.asarray(dets[:, 4], dtype=np.float32)
        order = np.argsort(-scores, kind="mergesort")
        keep = []

        while order.size > 0:
            idx = int(order[0])
            keep.append(idx)
            if order.size == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(boxes[idx, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[idx, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[idx, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[idx, 3], boxes[rest, 3])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h

            area_idx = np.maximum(0.0, boxes[idx, 2] - boxes[idx, 0]) * np.maximum(0.0, boxes[idx, 3] - boxes[idx, 1])
            area_rest = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(0.0, boxes[rest, 3] - boxes[rest, 1])
            union = area_idx + area_rest - inter
            ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            order = rest[ious <= iou_threshold]

        return np.asarray(keep, dtype=np.int64)

    @classmethod
    def _class_aware_nms_indices(cls, dets: np.ndarray, iou_threshold: float) -> np.ndarray:
        if dets.shape[0] == 0:
            return np.empty((0,), dtype=np.int64)

        keep_global = []
        classes = np.asarray(dets[:, 5], dtype=np.int64) if dets.shape[1] > 5 else np.zeros((dets.shape[0],), dtype=np.int64)
        for cls_id in np.unique(classes):
            cls_indices = np.flatnonzero(classes == cls_id)
            local_keep = cls._nms_indices(dets[cls_indices], iou_threshold)
            keep_global.extend(cls_indices[local_keep].tolist())

        keep_global = np.asarray(keep_global, dtype=np.int64)
        score_order = np.argsort(-dets[keep_global, 4], kind="mergesort")
        return keep_global[score_order]

    def _split_single_pass_relaxed(
        self,
        relaxed_dets: np.ndarray,
        relaxed_embs: Optional[np.ndarray],
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if relaxed_dets.shape[0] == 0:
            return np.empty((0, 6), dtype=np.float32), None

        normal_candidate_indices = np.flatnonzero(relaxed_dets[:, 4] >= self.conf_threshold)
        if normal_candidate_indices.size == 0:
            return np.empty((0, 6), dtype=np.float32), None

        normal_candidates = relaxed_dets[normal_candidate_indices]
        keep_local = self._class_aware_nms_indices(normal_candidates, self.normal_iou_threshold)
        keep_indices = normal_candidate_indices[keep_local]

        normal_dets = np.asarray(relaxed_dets[keep_indices], dtype=np.float32)
        normal_embs = None
        if relaxed_embs is not None:
            normal_embs = np.asarray(relaxed_embs[keep_indices], dtype=np.float32)
        return normal_dets, normal_embs

    def predict(self, frame: np.ndarray) -> JDEResult:
        timing = {
            "detector_normal_ms": 0.0,
            "detector_relaxed_ms": 0.0,
            "detector_superset_ms": 0.0,
            "software_split_nms_ms": 0.0,
        }
        relaxed_dets = None
        relaxed_embs = None
        relaxed_emb_src = "none"

        if self.relaxed_enabled and self.relaxed_source == "single-pass":
            # Production fast path: one relaxed superset forward, then recover the normal
            # detector set with software NMS while preserving JDE embedding row alignment.
            res_relaxed, timing["detector_superset_ms"] = self._timed_run_predict(frame, self.relaxed_conf_threshold, self.relaxed_iou_threshold)
            relaxed_dets, relaxed_embs, relaxed_emb_src = self._extract_dets_embs(res_relaxed)
            split_start = time.perf_counter()
            dets, embs = self._split_single_pass_relaxed(relaxed_dets, relaxed_embs)
            timing["software_split_nms_ms"] = (time.perf_counter() - split_start) * 1000.0
            emb_src = relaxed_emb_src if embs is not None else "none"
        else:
            # Reference path: normal pass plus optional relaxed pass.
            res_normal, timing["detector_normal_ms"] = self._timed_run_predict(frame, self.conf_threshold, self.normal_iou_threshold)
            dets, embs, emb_src = self._extract_dets_embs(res_normal)

            if self.relaxed_enabled:
                res_relaxed, timing["detector_relaxed_ms"] = self._timed_run_predict(frame, self.relaxed_conf_threshold, self.relaxed_iou_threshold)
                relaxed_dets, relaxed_embs, relaxed_emb_src = self._extract_dets_embs(res_relaxed)

        self.last_timing = timing
            
        # Assemble metadata
        metadata = {
            "num_detections": int(dets.shape[0]),
            "has_embeddings": embs is not None,
            "embedding_shape": tuple(embs.shape) if embs is not None else None,
            "embedding_source": emb_src,
            "embedding_dim": int(embs.shape[1]) if embs is not None else None,
            
            "relaxed_enabled": self.relaxed_enabled,
            "relaxed_source": self.relaxed_source,
            "num_relaxed_detections": int(relaxed_dets.shape[0]) if relaxed_dets is not None else 0,
            "has_relaxed_embeddings": relaxed_embs is not None,
            "relaxed_embedding_shape": tuple(relaxed_embs.shape) if relaxed_embs is not None else None,
            "relaxed_embedding_source": relaxed_emb_src,
            "relaxed_conf_threshold": self.relaxed_conf_threshold,
            "relaxed_iou_threshold": self.relaxed_iou_threshold,
            "normal_iou_threshold": self.normal_iou_threshold,
            "timing_ms": dict(timing),
        }
        
        if embs is not None and embs.size > 0:
            metadata["embedding_norm_mean"] = float(np.linalg.norm(embs, axis=1).mean())

        return JDEResult(
            detections=dets,
            embeddings=embs,
            relaxed_detections=relaxed_dets,
            relaxed_embeddings=relaxed_embs,
            metadata=metadata,
        )
