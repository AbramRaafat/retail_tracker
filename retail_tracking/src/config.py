from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import argparse
from ultralytics import settings

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Force Ultralytics to utilize the localized weights directory
settings.update({'weights_dir': str(WEIGHTS_DIR)})


@dataclass
class PipelineConfig:
    """
    Immutable configuration container orchestrating I/O and model routing.
    Algorithmic hyperparameters are delegated to BoxMOT YAML configurations.
    """
    input_video: str
    output_video: str
    yolo_model: str
    reid_model: Optional[str]
    tracker_config: Optional[str]
    target_fps: int
    tracker_type: str
    verbose: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'PipelineConfig':
        """Constructs configuration from CLI, resolving absolute paths."""
        input_path = Path(args.input).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        yolo_path = Path(args.model) if Path(args.model).is_absolute() else WEIGHTS_DIR / args.model
        if not yolo_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {yolo_path}")

        reid_path = None
        if args.reid:
            reid_path = Path(args.reid) if Path(args.reid).is_absolute() else WEIGHTS_DIR / args.reid
            if not reid_path.exists():
                raise FileNotFoundError(f"ReID model not found: {reid_path}")

        tracker_cfg_path = None
        if args.config:
            tracker_cfg_path = Path(args.config).resolve()
            if not tracker_cfg_path.exists():
                raise FileNotFoundError(f"Tracker config not found: {tracker_cfg_path}")

        return cls(
            input_video=str(input_path),
            output_video=str(output_path),
            yolo_model=str(yolo_path),
            reid_model=str(reid_path) if reid_path else None,
            tracker_config=str(tracker_cfg_path) if tracker_cfg_path else None,
            target_fps=args.fps,
            tracker_type=args.tracker,
            verbose=args.verbose
        )