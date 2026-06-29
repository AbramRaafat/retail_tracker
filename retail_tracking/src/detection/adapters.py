import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
from ultralytics import YOLO

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
    
    def __init__(self, weights_path: str, conf_threshold: float = 0.35, classes: Optional[List[int]] = None):
        self.detector = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.classes = classes if classes is not None else [0]
        
    def predict(self, frame: np.ndarray) -> JDEResult:
        results = self.detector.predict(
            frame, 
            classes=self.classes, 
            conf=self.conf_threshold, 
            verbose=False
        )
        
        if not results or len(results[0].boxes) == 0:
            return JDEResult(detections=np.empty((0, 6)))

        res = results[0]
        raw_dets = res.boxes.data.cpu().numpy()
        dets = raw_dets[:, :6]
        embs = None
        
        # Sequentially evaluate attributes for JDE embedding vectors.
        if hasattr(res, 'embeds') and res.embeds is not None:
            embs = res.embeds.data.cpu().numpy() if hasattr(res.embeds, 'data') else res.embeds.cpu().numpy()
        elif raw_dets.shape[1] > 6:
            embs = raw_dets[:, 6:]
        elif hasattr(res, 'embeddings') and res.embeddings is not None:
            embs = res.embeddings.data.cpu().numpy() if hasattr(res.embeddings, 'data') else res.embeddings.cpu().numpy()
        elif hasattr(res.boxes, 'embs') and res.boxes.embs is not None:
            embs = res.boxes.embs.data.cpu().numpy() if hasattr(res.boxes.embs, 'data') else res.boxes.embs.cpu().numpy()
            
        if embs is not None:
            embs = np.asarray(embs, dtype=np.float32)
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)

        # Validate spatial-appearance mapping constraint
        if embs is not None and embs.shape[0] != dets.shape[0]:
            logger.warning("Dimension mismatch: Detections (%d) vs Embeddings (%d). Dropping embeddings.", 
                           dets.shape[0], embs.shape[0])
            embs = None
        elif embs is not None:
            embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

        return JDEResult(detections=dets, embeddings=embs)
