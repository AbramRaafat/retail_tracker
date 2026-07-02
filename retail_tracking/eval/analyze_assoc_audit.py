import argparse
import csv
import json
import os
import numpy as np
from pathlib import Path

def analyze_audit_csv(csv_path: str, out_json: str, out_md: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Audit CSV file not found: {csv_path}")

    # Read rows
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            
    # Process rows
    total_association_rows = len(rows)
    
    # Get total frames
    frame_ids = {int(row["frame_id"]) for row in rows}
    total_audited_frames = len(frame_ids)
    
    # Match counts by tier
    match_counts = {"high": 0, "low": 0, "deleted_high": 0, "none": 0}
    for row in rows:
        matched = row["matched"].lower() == "true"
        if matched:
            tier = row["matched_det_tier"]
            if tier in match_counts:
                match_counts[tier] += 1
        else:
            match_counts["none"] += 1
            
    total_matches = match_counts["high"] + match_counts["low"] + match_counts["deleted_high"]
    deleted_high_match_rate = (match_counts["deleted_high"] / total_matches) if total_matches > 0 else 0.0
    
    # deleted_high-to-lost vs deleted_high-to-tracked counts
    deleted_high_to_lost = 0
    deleted_high_to_tracked = 0
    possible_stealing_cases = 0
    close_margin_stealing_cases = 0
    
    for row in rows:
        matched = row["matched"].lower() == "true"
        matched_tier = row["matched_det_tier"]
        track_state = row["track_state"]
        
        if matched and matched_tier == "deleted_high":
            if track_state == "Lost":
                deleted_high_to_lost += 1
            elif track_state == "Tracked":
                deleted_high_to_tracked += 1
                
            # Possible stealing: matched deleted_high but normal candidate was also available
            best_normal_tier = row.get("best_normal_det_tier", "none")
            best_normal_cost_str = row.get("best_normal_final_cost", "1.0")
            best_normal_cost = float(best_normal_cost_str) if best_normal_cost_str else 1.0
            
            if best_normal_tier in ("high", "low") and best_normal_cost < 1.0:
                possible_stealing_cases += 1
                
                # Check margin
                matched_cost_str = row["matched_final_cost"]
                matched_cost = float(matched_cost_str) if matched_cost_str else 1.0
                margin = best_normal_cost - matched_cost
                if margin <= 0.15:
                    close_margin_stealing_cases += 1
                    
    possible_stealing_percentage = (possible_stealing_cases / match_counts["deleted_high"] * 100.0) if match_counts["deleted_high"] > 0 else 0.0
    close_margin_stealing_percentage = (close_margin_stealing_cases / match_counts["deleted_high"] * 100.0) if match_counts["deleted_high"] > 0 else 0.0

    # median/mean metrics by matched tier
    margins_by_tier = {"high": [], "low": [], "deleted_high": []}
    ious_by_tier = {"high": [], "low": [], "deleted_high": []}
    cos_dists_by_tier = {"high": [], "low": [], "deleted_high": []}
    
    tracks_blocked_by_iou_gate = 0
    unmatched_count = 0
    blocked_unmatched_count = 0
    unmatched_cos_dists = []
    
    for row in rows:
        matched = row["matched"].lower() == "true"
        best_blocked = row["best_blocked_by_iou_gate"].lower() == "true"
        
        if best_blocked:
            tracks_blocked_by_iou_gate += 1
            
        if matched:
            tier = row["matched_det_tier"]
            if tier in margins_by_tier:
                margin_val = float(row["best_second_margin"]) if row["best_second_margin"] else 0.0
                iou_val = float(row["matched_iou_sim"]) if row["matched_iou_sim"] else 0.0
                cos_val = float(row["matched_cos_dist"]) if row["matched_cos_dist"] else 1.0
                
                margins_by_tier[tier].append(margin_val)
                ious_by_tier[tier].append(iou_val)
                cos_dists_by_tier[tier].append(cos_val)
        else:
            unmatched_count += 1
            if best_blocked:
                blocked_unmatched_count += 1
            cos_val = float(row["best_cos_dist"]) if row["best_cos_dist"] else 1.0
            unmatched_cos_dists.append(cos_val)
            
    # Calculate stats
    mean_margin = {}
    median_margin = {}
    mean_iou = {}
    median_iou = {}
    mean_cos = {}
    median_cos = {}
    
    for tier in ("high", "low", "deleted_high"):
        arr_m = margins_by_tier[tier]
        arr_i = ious_by_tier[tier]
        arr_c = cos_dists_by_tier[tier]
        
        mean_margin[tier] = float(np.mean(arr_m)) if arr_m else 0.0
        median_margin[tier] = float(np.median(arr_m)) if arr_m else 0.0
        
        mean_iou[tier] = float(np.mean(arr_i)) if arr_i else 0.0
        median_iou[tier] = float(np.median(arr_i)) if arr_i else 0.0
        
        mean_cos[tier] = float(np.mean(arr_c)) if arr_c else 0.0
        median_cos[tier] = float(np.median(arr_c)) if arr_c else 0.0

    # Heuristic Conclusions
    conclusions = []
    
    # Heuristic A: useful for lost recovery
    if deleted_high_to_lost > 0 and (possible_stealing_cases / max(1, match_counts["deleted_high"])) < 0.3:
        conclusions.append("relaxed detections look useful for lost recovery")
        
    # Heuristic B: stealing matches / too broad
    if match_counts["deleted_high"] > 0 and (possible_stealing_cases / match_counts["deleted_high"]) >= 0.3:
        conclusions.append("relaxed detections look too broad / stealing matches")
        
    # Heuristic C: failures IoU gate related
    if unmatched_count > 0 and (blocked_unmatched_count / unmatched_count) > 0.4:
        conclusions.append("failures mostly look IoU-gate related")
        
    # Heuristic D: failures appearance ambiguity related
    if unmatched_count > 0 and (blocked_unmatched_count / unmatched_count) <= 0.4:
        avg_unmatched_cos = np.mean(unmatched_cos_dists) if unmatched_cos_dists else 0.0
        if avg_unmatched_cos > 0.4:
            conclusions.append("failures mostly look appearance-ambiguity related")

    # JSON output dict
    json_data = {
        "total_audited_frames": total_audited_frames,
        "total_association_rows": total_association_rows,
        "match_counts": match_counts,
        "deleted_high_match_rate": deleted_high_match_rate,
        "deleted_high_to_lost": deleted_high_to_lost,
        "deleted_high_to_tracked": deleted_high_to_tracked,
        "possible_stealing_cases": possible_stealing_cases,
        "possible_stealing_percentage": possible_stealing_percentage,
        "close_margin_stealing_cases": close_margin_stealing_cases,
        "close_margin_stealing_percentage": close_margin_stealing_percentage,
        "mean_best_second_margin": mean_margin,
        "median_best_second_margin": median_margin,
        "mean_iou_by_tier": mean_iou,
        "median_iou_by_tier": median_iou,
        "mean_cos_dist_by_tier": mean_cos,
        "median_cos_dist_by_tier": median_cos,
        "tracks_blocked_by_iou_gate": tracks_blocked_by_iou_gate,
        "unmatched_tracks": unmatched_count,
        "unmatched_blocked_by_iou_gate": blocked_unmatched_count,
        "conclusions": conclusions
    }
    
    # Write JSON
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
        
    # Write Markdown
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# Association Audit Summary Report\n\n")
        f.write("## Overview\n")
        f.write(f"- **Total Audited Frames**: {total_audited_frames}\n")
        f.write(f"- **Total Association Decision Rows**: {total_association_rows}\n\n")
        
        f.write("## Match Counts by Tier\n")
        f.write(f"- **High-confidence Detections**: {match_counts['high']}\n")
        f.write(f"- **Low-confidence Detections**: {match_counts['low']}\n")
        f.write(f"- **Deleted High-confidence (Relaxed) Detections**: {match_counts['deleted_high']}\n")
        f.write(f"- **Unmatched (None)**: {match_counts['none']}\n\n")
        
        f.write("## Relaxed (Deleted High) Match Details\n")
        f.write(f"- **Deleted High Match Rate**: {deleted_high_match_rate * 100.0:.2f}%\n")
        f.write(f"- **Deleted High to Lost Tracks**: {deleted_high_to_lost}\n")
        f.write(f"- **Deleted High to Tracked Tracks**: {deleted_high_to_tracked}\n")
        f.write(f"- **Possible Stealing Cases**: {possible_stealing_cases} ({possible_stealing_percentage:.2f}% of deleted high matches)\n")
        f.write(f"- **Close Margin Stealing Cases (<= 0.15)**: {close_margin_stealing_cases} ({close_margin_stealing_percentage:.2f}% of deleted high matches)\n\n")
        
        f.write("## Metrics by Match Tier\n")
        f.write("| Matched Tier | Mean IoU Sim | Median IoU Sim | Mean Cosine Dist | Median Cosine Dist | Mean Margin | Median Margin |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for tier in ("high", "low", "deleted_high"):
            f.write(f"| {tier.capitalize()} | {mean_iou[tier]:.4f} | {median_iou[tier]:.4f} | {mean_cos[tier]:.4f} | {median_cos[tier]:.4f} | {mean_margin[tier]:.4f} | {median_margin[tier]:.4f} |\n")
        f.write("\n")
        
        f.write("## Failure Analysis\n")
        f.write(f"- **Tracks Blocked by Hard IoU Gate**: {tracks_blocked_by_iou_gate}\n")
        f.write(f"- **Unmatched Tracks**: {unmatched_count} (Blocked: {blocked_unmatched_count})\n\n")
        
        f.write("## Conclusions\n")
        if conclusions:
            for conclusion in conclusions:
                if conclusion == "relaxed detections look useful for lost recovery":
                    desc = "Deleted high matches primarily went to recovering lost tracks with low stealing rates."
                elif conclusion == "relaxed detections look too broad / stealing matches":
                    desc = "Deleted high matches frequently matched while normal candidates existed, suggesting potential stealing."
                elif conclusion == "failures mostly look IoU-gate related":
                    desc = "A large fraction of unmatched tracks had candidates blocked by the IoU gate threshold."
                elif conclusion == "failures mostly look appearance-ambiguity related":
                    desc = "Unmatched tracks had candidates within the IoU gate but suffered from large appearance cosine distance."
                else:
                    desc = ""
                f.write(f"- **{conclusion}**: {desc}\n")
        else:
            f.write("- **No strong conclusions**: Insufficient or balanced metric statistics.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TrackTrack Association Audit CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--out-json", type=str, required=True, help="Path to output JSON summary file")
    parser.add_argument("--out-md", type=str, required=True, help="Path to output Markdown report file")
    args = parser.parse_args()
    
    analyze_audit_csv(args.csv, args.out_json, args.out_md)
    print(f"Analysis completed. MD: {args.out_md}, JSON: {args.out_json}")
