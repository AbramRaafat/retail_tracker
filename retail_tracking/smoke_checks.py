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


class _ConfigurableFakeYOLO:
    last_predict_kwargs = []
    predict_side_effects = []

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        _ConfigurableFakeYOLO.last_predict_kwargs.append(kwargs)
        if _ConfigurableFakeYOLO.predict_side_effects:
            return _ConfigurableFakeYOLO.predict_side_effects.pop(0)
        return []


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


class _CustomFakeDetector:
    def __init__(self, dets, embs, metadata=None):
        self.dets = dets
        self.embs = embs
        self.metadata = metadata or {}

    def predict(self, frame):
        return JDEResult(
            detections=self.dets,
            embeddings=self.embs,
            metadata=self.metadata,
        )


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
    retail_tracker.appearance_mode = "auto"
    retail_tracker.allow_zero_embs = False
    retail_tracker.last_jde_metadata = {}
    retail_tracker.last_route_metadata = {}
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


def check_appearance_routing_modes():
    # 1. JDE route success
    dets = np.array([[0, 0, 10, 10, 0.9, 0]], dtype=np.float32)
    embs = np.array([[3.0, 4.0]], dtype=np.float32)
    meta = {
        "num_detections": 1,
        "has_embeddings": True,
        "embedding_shape": (1, 2),
        "embedding_source": "res.embeds",
        "embedding_dim": 2,
    }
    detector = _CustomFakeDetector(dets, embs, meta)
    tracker_wrapper = TrackTrack()
    
    retail_tracker = RetailTracker.__new__(RetailTracker)
    retail_tracker.detector = detector
    retail_tracker.tracker = tracker_wrapper
    retail_tracker.appearance_mode = "jde"
    retail_tracker.allow_zero_embs = False
    retail_tracker.last_jde_metadata = {}
    retail_tracker.last_route_metadata = {}
    
    tracks = retail_tracker.process_frame(np.zeros((16, 16, 3), dtype=np.uint8))
    
    assert hasattr(tracker_wrapper, "last_route_metadata")
    assert tracker_wrapper.last_route_metadata["embedding_route"] == "jde"
    assert retail_tracker.last_route_metadata["embedding_route"] == "jde"
    assert retail_tracker.last_route_metadata["used_zero_embeddings"] is False
    assert retail_tracker.last_jde_metadata["embedding_source"] == "res.embeds"
    
    # 2. Missing embeddings in JDE mode
    detector_no_embs = _CustomFakeDetector(dets, None, {
        "num_detections": 1,
        "has_embeddings": False,
        "embedding_shape": None,
        "embedding_source": "none",
        "embedding_dim": None,
    })
    tracker_wrapper = TrackTrack()
    retail_tracker.tracker = tracker_wrapper
    retail_tracker.detector = detector_no_embs
    retail_tracker.appearance_mode = "jde"
    
    try:
        retail_tracker.process_frame(np.zeros((16, 16, 3), dtype=np.uint8))
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "JDE embeddings are required" in str(e)
        
    # 3. None mode
    tracker_wrapper = TrackTrack()
    retail_tracker.tracker = tracker_wrapper
    retail_tracker.appearance_mode = "none"
    tracks = retail_tracker.process_frame(np.zeros((16, 16, 3), dtype=np.uint8))
    assert retail_tracker.last_route_metadata["used_zero_embeddings"] is True
    assert retail_tracker.last_route_metadata["embedding_route"] == "none"

    # 4. Auto mode without JDE & ReID
    tracker_wrapper = TrackTrack()
    retail_tracker.tracker = tracker_wrapper
    retail_tracker.appearance_mode = "auto"
    try:
        retail_tracker.process_frame(np.zeros((16, 16, 3), dtype=np.uint8))
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "no external ReID model" in str(e)


def check_adapter_relaxed_routing():
    original_yolo = adapters.YOLO
    adapters.YOLO = _ConfigurableFakeYOLO
    try:
        # Clear side effects
        _ConfigurableFakeYOLO.predict_side_effects = []
        _ConfigurableFakeYOLO.last_predict_kwargs = []

        # Setup side effects:
        # Pass 1: normal result (1 det, 128-D embs)
        dets_normal = np.array([[0, 0, 10, 10, 0.9, 0]], dtype=np.float32)
        embs_normal = np.ones((1, 128), dtype=np.float32)
        
        # Pass 2: relaxed result (2 dets, 128-D embs)
        dets_relaxed = np.array([[0, 0, 10, 10, 0.9, 0], [20, 20, 30, 30, 0.8, 0]], dtype=np.float32)
        embs_relaxed = np.ones((2, 128), dtype=np.float32)
        
        _ConfigurableFakeYOLO.predict_side_effects = [
            [_FakeResult(dets_normal, embs_normal)],
            [_FakeResult(dets_relaxed, embs_relaxed)]
        ]

        adapter = adapters.UltralyticsJDEAdapter(
            "unused.pt",
            device="cpu",
            half=False,
            imgsz=640,
            conf_threshold=0.2,
            classes=[0],
            relaxed_enabled=True,
            relaxed_conf_threshold=0.03,
            relaxed_iou_threshold=0.95,
            normal_iou_threshold=0.70,
        )
        
        # Run predict
        result = adapter.predict(np.zeros((32, 32, 3), dtype=np.uint8))
        
        # Assertions for Test A
        assert result.detections.shape == (1, 6)
        assert result.embeddings.shape == (1, 128)
        assert result.relaxed_detections.shape == (2, 6)
        assert result.relaxed_embeddings.shape == (2, 128)
        assert result.metadata["relaxed_enabled"] is True
        assert result.metadata["num_relaxed_detections"] == 2
        assert result.metadata["has_relaxed_embeddings"] is True
        assert result.metadata["relaxed_embedding_shape"] == (2, 128)
        assert result.metadata["normal_iou_threshold"] == 0.70
        assert result.metadata["relaxed_conf_threshold"] == 0.03
        assert result.metadata["relaxed_iou_threshold"] == 0.95
        assert _ConfigurableFakeYOLO.last_predict_kwargs[0]["conf"] == 0.2
        assert _ConfigurableFakeYOLO.last_predict_kwargs[0]["iou"] == 0.70
        assert _ConfigurableFakeYOLO.last_predict_kwargs[1]["conf"] == 0.03
        assert _ConfigurableFakeYOLO.last_predict_kwargs[1]["iou"] == 0.95

        # Test D: Backward compatibility test (relaxed_enabled=False)
        _ConfigurableFakeYOLO.predict_side_effects = [
            [_FakeResult(dets_normal, embs_normal)]
        ]
        adapter_disabled = adapters.UltralyticsJDEAdapter(
            "unused.pt",
            device="cpu",
            half=False,
            imgsz=640,
            conf_threshold=0.2,
            classes=[0],
            relaxed_enabled=False,
        )
        result_disabled = adapter_disabled.predict(np.zeros((32, 32, 3), dtype=np.uint8))
        
        assert result_disabled.relaxed_detections is None
        assert result_disabled.relaxed_embeddings is None
        assert result_disabled.metadata["relaxed_enabled"] is False
        assert result_disabled.metadata["num_relaxed_detections"] == 0
        
    finally:
        adapters.YOLO = original_yolo


def check_tracktrack_strict_jde_relaxed_error():
    dets = np.array([[0, 0, 10, 10, 0.9, 0]], dtype=np.float32)
    embs = np.array([[1.0] * 128], dtype=np.float32)
    
    detector = _CustomFakeDetector(dets, embs, {
        "num_detections": 1,
        "has_embeddings": True,
        "embedding_shape": (1, 128),
        "embedding_source": "res.embeds",
        "embedding_dim": 128,
        "relaxed_enabled": True,
        "num_relaxed_detections": 1,
        "has_relaxed_embeddings": False,
        "relaxed_embedding_shape": None,
    })
    
    def predict_mock(frame):
        return JDEResult(
            detections=dets,
            embeddings=embs,
            relaxed_detections=dets,
            relaxed_embeddings=None,
            metadata=detector.metadata,
        )
    detector.predict = predict_mock
    
    tracker_wrapper = TrackTrack()
    retail_tracker = RetailTracker.__new__(RetailTracker)
    retail_tracker.detector = detector
    retail_tracker.tracker = tracker_wrapper
    retail_tracker.appearance_mode = "jde"
    retail_tracker.allow_zero_embs = False
    retail_tracker.last_jde_metadata = {}
    retail_tracker.last_route_metadata = {}
    
    try:
        retail_tracker.process_frame(np.zeros((16, 16, 3), dtype=np.uint8))
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "relaxed JDE embeddings are missing" in str(e)


def check_tracktrack_strict_jde_relaxed_success():
    dets = np.array([[0, 0, 10, 10, 0.9, 0]], dtype=np.float32)
    embs = np.array([[1.0] * 128], dtype=np.float32)
    
    detector = _CustomFakeDetector(dets, embs, {
        "num_detections": 1,
        "has_embeddings": True,
        "embedding_shape": (1, 128),
        "embedding_source": "res.embeds",
        "embedding_dim": 128,
        "relaxed_enabled": True,
        "num_relaxed_detections": 1,
        "has_relaxed_embeddings": True,
        "relaxed_embedding_shape": (1, 128),
    })
    
    def predict_mock(frame):
        return JDEResult(
            detections=dets,
            embeddings=embs,
            relaxed_detections=dets,
            relaxed_embeddings=embs,
            metadata=detector.metadata,
        )
    detector.predict = predict_mock
    
    tracker_wrapper = TrackTrack()
    retail_tracker = RetailTracker.__new__(RetailTracker)
    retail_tracker.detector = detector
    retail_tracker.tracker = tracker_wrapper
    retail_tracker.appearance_mode = "jde"
    retail_tracker.allow_zero_embs = False
    retail_tracker.last_jde_metadata = {}
    retail_tracker.last_route_metadata = {}
    
    tracks = retail_tracker.process_frame(np.zeros((16, 16, 3), dtype=np.uint8))
    
    assert retail_tracker.last_route_metadata["num_relaxed_detections"] == 1
    assert retail_tracker.last_route_metadata["has_relaxed_embeddings"] is True
    assert retail_tracker.last_route_metadata["relaxed_embedding_shape"] == (1, 128)


def main() -> None:
    check_jde_result_optional_fields()
    check_embedding_normalization()
    check_empty_frame_updates_tracker()
    check_tracktrack_empty_update()
    check_appearance_routing_modes()
    check_adapter_relaxed_routing()
    check_tracktrack_strict_jde_relaxed_error()
    check_tracktrack_strict_jde_relaxed_success()
    print("retail_tracking smoke checks passed")


if __name__ == "__main__":
    main()
