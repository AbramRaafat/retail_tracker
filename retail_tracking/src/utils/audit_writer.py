import csv
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

class AssociationAuditWriter:
    def __init__(self, csv_path: str, max_frames: Optional[int] = None, summary_path: Optional[str] = None):
        self.csv_path = csv_path
        self.max_frames = max_frames
        self.summary_path = summary_path
        
        # Create parent directories
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if summary_path:
            Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
            
        # Open CSV file
        self.file = open(csv_path, mode='w', newline='', encoding='utf-8')
        
        # Headers
        self.headers = [
            "row_type", "frame_id", "association_stage", "track_index", "track_id", "track_state",
            "track_history_len", "frames_since_update", "num_dets_high", "num_dets_low",
            "num_dets_deleted_high",
            "det_index", "det_tier", "det_score", "det_box", "track_predicted_box",
            "iou_sim", "iou_dist", "cosine_distance", "confidence_distance", "angle_distance",
            "static_weighted_cost_before_penalty", "low_or_deleted_penalty",
            "cost_after_penalty", "blocked_by_iou_gate", "final_cost_after_gate_clip",
            "track_matched",
            "best_det_index", "best_det_tier", "best_det_score",
            "best_final_cost", "second_best_final_cost", "best_second_margin",
            "best_iou_sim", "best_iou_dist", "best_cos_dist", "best_conf_dist",
            "best_angle_dist", "best_blocked_by_iou_gate", "matched", "matched_det_index",
            "matched_det_tier", "matched_det_score", "matched_final_cost", "matched_iou_sim",
            "matched_cos_dist", "matched_conf_dist", "matched_angle_dist",
            "best_normal_det_index", "best_normal_det_tier", "best_normal_final_cost",
            "best_static_candidate_index", "best_static_candidate_tier",
            "recovered_by_deleted_high", "feature_update_frozen",
            "mahalanobis_squared", "mahalanobis_gate_threshold", "mahalanobis_passed_gate",
            "mahalanobis_normalized_cost", "mahalanobis_applicable", "mahalanobis_failed",
            "matched_candidate_passed_mahalanobis_gate",
            "best_static_candidate_passed_mahalanobis_gate",
            "unmatched_track_had_mahalanobis_valid_candidate",
            "mahalanobis_would_reject_matched_candidate",
            "mahalanobis_would_allow_iou_blocked_candidate"
        ]
        
        self.writer = csv.DictWriter(self.file, fieldnames=self.headers)
        self.writer.writeheader()
        
        # Flush counter
        self.rows_written = 0
        
        # Store frame summaries in memory to dump to JSON at the end
        self.frame_summaries: List[Dict[str, Any]] = []

    def write_row(self, row_dict: Dict[str, Any]) -> None:
        if self.max_frames is not None and row_dict["frame_id"] > self.max_frames:
            return
        self.writer.writerow({header: row_dict.get(header, "") for header in self.headers})
        self.rows_written += 1
        if self.rows_written % 100 == 0:
            self.file.flush()

    def add_frame_summary(self, summary_dict: Dict[str, Any]) -> None:
        if self.max_frames is not None and summary_dict["frame_id"] > self.max_frames:
            return
        self.frame_summaries.append(summary_dict)

    def close(self) -> None:
        # Flush and close CSV
        self.file.flush()
        self.file.close()
        
        # If summary path provided, write frame summaries as JSON
        if self.summary_path:
            with open(self.summary_path, 'w', encoding='utf-8') as f:
                json.dump(self.frame_summaries, f, indent=2)
