import logging
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
    ):
        self.detector = YOLO(weights_path)
        self.device = device
        self.half = half
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.classes = classes if classes is not None else [0]

    def warmup(self, frame_shape: tuple[int, int, int] | None = None) -> None:
        shape = frame_shape if frame_shape is not None else (640, 640, 3)
        dummy = np.zeros(shape, dtype=np.uint8)
        try:
            self.predict(dummy)
        except Exception as exc:
            logger.warning("Detector warmup failed; continuing without warmup: %s", exc)

    def predict(self, frame: np.ndarray) -> JDEResult:
        context = torch.inference_mode() if torch is not None else nullcontext()
        with context:
            results = self.detector.predict(
                frame,
                classes=self.classes,
                conf=self.conf_threshold,
                device=self.device,
                half=self.half,
                imgsz=self.imgsz,
                verbose=False,
            )
        
        if not results or len(results[0].boxes) == 0:
            return JDEResult(
                detections=np.empty((0, 6), dtype=np.float32),
                metadata={
                    "num_detections": 0,
                    "has_embeddings": False,
                    "embedding_shape": None,
                    "embedding_source": "none",
                    "embedding_dim": None,
                },
            )

        res = results[0]
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
        elif embs is not None:
            # Do not normalize here. Trackers own final embedding normalization because
            # different trackers/backends may have different appearance assumptions.
            pass

        metadata = {
            "num_detections": int(dets.shape[0]),
            "has_embeddings": embs is not None,
            "embedding_shape": tuple(embs.shape) if embs is not None else None,
            "embedding_source": embedding_source,
            "embedding_dim": int(embs.shape[1]) if embs is not None else None,
        }

        if embs is not None and embs.size > 0:
            metadata["embedding_norm_mean"] = float(np.linalg.norm(embs, axis=1).mean())

        return JDEResult(
            detections=dets,
            embeddings=embs,
            metadata=metadata,
        )
