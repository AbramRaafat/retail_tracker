"""Lightweight smoke checks for the retail tracking integration."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BOXMOT_ROOT = ROOT / "boxmot"
if str(BOXMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOXMOT_ROOT))

try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules["cv2"] = types.SimpleNamespace(
        FONT_HERSHEY_SIMPLEX=0,
        rectangle=lambda *args, **kwargs: None,
        getTextSize=lambda *args, **kwargs: ((0, 0), 0),
        putText=lambda *args, **kwargs: None,
    )

try:
    import ultralytics  # noqa: F401
except ImportError:
    sys.modules["ultralytics"] = types.SimpleNamespace(
        YOLO=lambda *args, **kwargs: None,
        settings=types.SimpleNamespace(update=lambda *args, **kwargs: None),
    )

from retail_tracking.src.core.tracker import RetailTracker
from retail_tracking.src.detection import adapters
from retail_tracking.src.detection.adapters import JDEResult

boxmot_pkg = types.ModuleType("boxmot")
boxmot_pkg.__path__ = [str(BOXMOT_ROOT / "boxmot")]
sys.modules.setdefault("boxmot", boxmot_pkg)
trackers_pkg = types.ModuleType("boxmot.trackers")
trackers_pkg.__path__ = [str(BOXMOT_ROOT / "boxmot" / "trackers")]
sys.modules.setdefault("boxmot.trackers", trackers_pkg)
utils_pkg = types.ModuleType("boxmot.utils")
utils_pkg.__path__ = [str(BOXMOT_ROOT / "boxmot" / "utils")]
utils_pkg.logger = types.SimpleNamespace(
    warning=lambda *args, **kwargs: None,
    debug=lambda *args, **kwargs: None,
    info=lambda *args, **kwargs: None,
)
sys.modules.setdefault("boxmot.utils", utils_pkg)

try:
    import lap  # noqa: F401
except ImportError:
    sys.modules["lap"] = types.SimpleNamespace()

try:
    import scipy.linalg  # noqa: F401
except ImportError:
    scipy_pkg = types.ModuleType("scipy")
    scipy_linalg_pkg = types.ModuleType("scipy.linalg")
    scipy_pkg.linalg = scipy_linalg_pkg
    sys.modules["scipy"] = scipy_pkg
    sys.modules["scipy.linalg"] = scipy_linalg_pkg

from boxmot.trackers.tracktrack.tracktrack import TrackTrack


class _ArrayProxy:
    def __init__(self, array: np.ndarray):
        self._array = array
        self.data = self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _FakeBoxes:
    def __init__(self, dets: np.ndarray):
        self.data = _ArrayProxy(dets)

    def __len__(self):
        return len(self.data._array)


class _FakeResult:
    def __init__(self, dets: np.ndarray, embs: np.ndarray):
        self.boxes = _FakeBoxes(dets)
        self.embeds = _ArrayProxy(embs)


class _FakeYOLO:
    last_predict_kwargs = None

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        _FakeYOLO.last_predict_kwargs = kwargs
        dets = np.array([[0, 0, 10, 10, 0.9, 0], [20, 20, 30, 30, 0.8, 0]], dtype=np.float32)
        embs = np.array([[3, 4], [0, 2]], dtype=np.float32)
        return [_FakeResult(dets, embs)]


class _EmptyDetector:
    def predict(self, frame):
        return JDEResult(detections=np.empty((0, 6), dtype=np.float32))


class _DummyTracker:
    def __init__(self):
        self.updates = []
        self.aux = None

    def set_frame_aux(self, **kwargs):
        self.aux = kwargs

    def update(self, dets, frame, embs=None):
        self.updates.append((dets.copy(), embs))
        return np.empty((0, 8), dtype=np.float32)


def check_jde_result_optional_fields():
    result = JDEResult(
        detections=np.empty((0, 6), dtype=np.float32),
        relaxed_detections=np.empty((0, 6), dtype=np.float32),
        relaxed_embeddings=np.empty((0, 2), dtype=np.float32),
        metadata={"source": "smoke"},
    )
    assert result.relaxed_detections.shape == (0, 6)
    assert result.metadata["source"] == "smoke"


def check_embedding_normalization():
    original_yolo = adapters.YOLO
    adapters.YOLO = _FakeYOLO
    try:
        adapter = adapters.UltralyticsJDEAdapter(
            "unused.pt",
            device="cpu",
            half=False,
            imgsz=640,
            conf_threshold=0.2,
            classes=[0],
        )
        result = adapter.predict(np.zeros((32, 32, 3), dtype=np.uint8))
    finally:
        adapters.YOLO = original_yolo

    assert adapter.device == "cpu"
    assert adapter.half is False
    assert adapter.imgsz == 640
    assert _FakeYOLO.last_predict_kwargs["device"] == "cpu"
    assert _FakeYOLO.last_predict_kwargs["half"] is False
    assert _FakeYOLO.last_predict_kwargs["imgsz"] == 640
    assert _FakeYOLO.last_predict_kwargs["conf"] == 0.2

    norms = np.linalg.norm(result.embeddings, axis=1)
    assert result.embeddings.dtype == np.float32
    assert result.metadata["num_detections"] == 2
    assert result.metadata["embedding_shape"] == (2, 2)
    assert result.metadata["has_embeddings"] is True
    # Verify raw embeddings are NOT normalized (norm of [3,4] is 5, norm of [0,2] is 2)
    assert np.allclose(norms, np.array([5.0, 2.0], dtype=np.float32))

    # Verify TrackTrack._normalize_embeddings returns unit-norm vectors
    fake_embs = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    normed_embs = TrackTrack._normalize_embeddings(fake_embs)
    normed_norms = np.linalg.norm(normed_embs, axis=1)
    assert np.allclose(normed_norms, np.ones_like(normed_norms))


def check_empty_frame_updates_tracker():
    retail_tracker = RetailTracker.__new__(RetailTracker)
    retail_tracker.detector = _EmptyDetector()
    retail_tracker.tracker = _DummyTracker()
    retail_tracker.last_jde_metadata = {}
    tracks = retail_tracker.process_frame(np.zeros((16, 16, 3), dtype=np.uint8))
    assert tracks.shape == (0, 8)
    assert len(retail_tracker.tracker.updates) == 1
    assert retail_tracker.tracker.updates[0][0].shape == (0, 6)
    assert retail_tracker.last_jde_metadata == {}


def check_tracktrack_empty_update():
    tracker = TrackTrack()
    output = tracker._update_impl(
        np.empty((0, 6), dtype=np.float32),
        np.zeros((16, 16, 3), dtype=np.uint8),
    )
    assert output.shape == (0, 8)


def main() -> None:
    check_jde_result_optional_fields()
    check_embedding_normalization()
    check_empty_frame_updates_tracker()
    check_tracktrack_empty_update()
    print("retail_tracking smoke checks passed")


if __name__ == "__main__":
    main()
