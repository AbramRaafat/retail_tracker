import argparse
import csv
import json
import os
import numpy as np
from pathlib import Path


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _as_float(value, default=0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _median(values):
    return float(np.median(values)) if values else 0.0


def _valid_mahalanobis(row) -> bool:
    return _as_float(row.get("mahalanobis_squared"), -1.0) >= 0.0


def _track_key(row):
    return (
        row.get("frame_id", ""),
        row.get("association_stage", ""),
        row.get("track_id", ""),
        row.get("track_index", ""),
    )


def _best_cost_row(rows):
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: _as_float(
            row.get("final_cost_after_gate_clip", row.get("best_final_cost", 1.0)),
            1.0,
        ),
    )


def _summarize_rows(rows):
    maha_rows = [row for row in rows if _valid_mahalanobis(row)]
    matched_rows = [row for row in rows if _as_bool(row.get("matched"))]
    return {
        "rows": len(rows),
        "matched_rows": len(matched_rows),
        "mean_mahalanobis_squared": _mean([_as_float(row.get("mahalanobis_squared")) for row in maha_rows]),
        "median_mahalanobis_squared": _median([_as_float(row.get("mahalanobis_squared")) for row in maha_rows]),
        "mahalanobis_gate_pass_rate": (
            sum(1 for row in maha_rows if _as_bool(row.get("mahalanobis_passed_gate"))) / len(maha_rows)
            if maha_rows else 0.0
        ),
        "matched_rejected_by_mahalanobis": sum(
            1 for row in matched_rows if _as_bool(row.get("mahalanobis_would_reject_matched_candidate"))
        ),
    }


def analyze_audit_csv(csv_path: str, out_json: str, out_md: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Audit CSV file not found: {csv_path}")

    # Read rows
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            
    total_association_rows = len(rows)

    frame_ids = {int(row["frame_id"]) for row in rows if row.get("frame_id")}
    total_audited_frames = len(frame_ids)

    grouped = {}
    for row in rows:
        grouped.setdefault(_track_key(row), []).append(row)
    track_groups = list(grouped.values())
    matched_rows = [row for row in rows if _as_bool(row.get("matched"))]
    unmatched_groups = [group for group in track_groups if not any(_as_bool(row.get("matched")) for row in group)]

    match_counts = {"high": 0, "low": 0, "deleted_high": 0, "none": 0}
    for row in matched_rows:
        tier = row.get("matched_det_tier") or row.get("det_tier")
        if tier in match_counts:
            match_counts[tier] += 1
    match_counts["none"] = len(unmatched_groups)

    total_matches = match_counts["high"] + match_counts["low"] + match_counts["deleted_high"]
    deleted_high_match_rate = (match_counts["deleted_high"] / total_matches) if total_matches > 0 else 0.0

    deleted_high_to_lost = 0
    deleted_high_to_tracked = 0
    deleted_high_with_valid_normal_candidate = 0
    close_margin_stealing_cases = 0
    gaps = []
    better_count = 0
    within_005_count = 0
    
    for row in matched_rows:
        matched_tier = row.get("matched_det_tier", "none")
        track_state = row.get("track_state", "Unknown")

        if matched_tier == "deleted_high":
            if track_state == "Lost":
                deleted_high_to_lost += 1
            elif track_state == "Tracked":
                deleted_high_to_tracked += 1

            best_normal_tier = row.get("best_normal_det_tier", "none")
            best_normal_cost = _as_float(row.get("best_normal_final_cost"), 1.0)

            if best_normal_tier in ("high", "low") and best_normal_cost < 1.0:
                deleted_high_with_valid_normal_candidate += 1
                matched_cost = _as_float(row.get("matched_final_cost"), 1.0)
                gap = best_normal_cost - matched_cost
                gaps.append(gap)
                if gap > 0.0:
                    better_count += 1
                if abs(gap) <= 0.05:
                    within_005_count += 1
                if gap <= 0.15:
                    close_margin_stealing_cases += 1

    deleted_high_with_valid_normal_candidate_percentage = (deleted_high_with_valid_normal_candidate / match_counts["deleted_high"] * 100.0) if match_counts["deleted_high"] > 0 else 0.0
    close_margin_stealing_percentage = (close_margin_stealing_cases / match_counts["deleted_high"] * 100.0) if match_counts["deleted_high"] > 0 else 0.0

    deleted_high_normal_cost_gap_mean = _mean(gaps)
    deleted_high_normal_cost_gap_median = _median(gaps)
    deleted_high_normal_better_count = better_count
    deleted_high_normal_within_005_count = within_005_count

    deleted_high_to_lost_share = (deleted_high_to_lost / match_counts["deleted_high"]) if match_counts["deleted_high"] > 0 else 0.0
    deleted_high_to_tracked_share = (deleted_high_to_tracked / match_counts["deleted_high"]) if match_counts["deleted_high"] > 0 else 0.0

    margins_by_tier = {"high": [], "low": [], "deleted_high": []}
    ious_by_tier = {"high": [], "low": [], "deleted_high": []}
    cos_dists_by_tier = {"high": [], "low": [], "deleted_high": []}

    tracks_blocked_by_iou_gate = 0
    blocked_unmatched_count = 0
    unmatched_cos_dists = []

    for group in track_groups:
        best_row = _best_cost_row(group)
        if best_row and _as_bool(best_row.get("best_blocked_by_iou_gate")):
            tracks_blocked_by_iou_gate += 1
        matched_row = next((row for row in group if _as_bool(row.get("matched"))), None)
        if matched_row:
            tier = matched_row.get("matched_det_tier", "none")
            if tier in margins_by_tier:
                margins_by_tier[tier].append(_as_float(matched_row.get("best_second_margin"), 0.0))
                ious_by_tier[tier].append(_as_float(matched_row.get("matched_iou_sim"), 0.0))
                cos_dists_by_tier[tier].append(_as_float(matched_row.get("matched_cos_dist"), 1.0))
        elif best_row is not None:
            if _as_bool(best_row.get("best_blocked_by_iou_gate")):
                blocked_unmatched_count += 1
            unmatched_cos_dists.append(_as_float(best_row.get("best_cos_dist"), 1.0))

    unmatched_count = len(unmatched_groups)

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
        
        mean_margin[tier] = _mean(arr_m)
        median_margin[tier] = _median(arr_m)

        mean_iou[tier] = _mean(arr_i)
        median_iou[tier] = _median(arr_i)

        mean_cos[tier] = _mean(arr_c)
        median_cos[tier] = _median(arr_c)

    matched_maha = [_as_float(row.get("mahalanobis_squared")) for row in matched_rows if _valid_mahalanobis(row)]
    unmatched_best_rows = [_best_cost_row(group) for group in unmatched_groups]
    unmatched_best_rows = [row for row in unmatched_best_rows if row is not None]
    unmatched_best_maha = [_as_float(row.get("mahalanobis_squared")) for row in unmatched_best_rows if _valid_mahalanobis(row)]

    matched_maha_rows = [row for row in matched_rows if _valid_mahalanobis(row)]
    unmatched_best_maha_rows = [row for row in unmatched_best_rows if _valid_mahalanobis(row)]
    matched_gate_pass_rate = (
        sum(1 for row in matched_maha_rows if _as_bool(row.get("mahalanobis_passed_gate"))) / len(matched_maha_rows)
        if matched_maha_rows else 0.0
    )
    unmatched_best_gate_pass_rate = (
        sum(1 for row in unmatched_best_maha_rows if _as_bool(row.get("mahalanobis_passed_gate"))) / len(unmatched_best_maha_rows)
        if unmatched_best_maha_rows else 0.0
    )
    matched_rejected_by_maha = sum(1 for row in matched_maha_rows if _as_bool(row.get("mahalanobis_would_reject_matched_candidate")))
    matched_rejected_by_maha_share = matched_rejected_by_maha / len(matched_maha_rows) if matched_maha_rows else 0.0
    unmatched_with_any_maha_valid = sum(
        1 for group in unmatched_groups
        if any(_as_bool(row.get("mahalanobis_passed_gate")) for row in group if _valid_mahalanobis(row))
    )
    unmatched_with_any_maha_valid_share = unmatched_with_any_maha_valid / len(unmatched_groups) if unmatched_groups else 0.0
    iou_blocked_rows = [row for row in rows if _as_bool(row.get("blocked_by_iou_gate"))]
    iou_blocked_maha_valid = [
        row for row in iou_blocked_rows
        if _as_bool(row.get("mahalanobis_would_allow_iou_blocked_candidate"))
    ]
    iou_blocked_maha_valid_share = len(iou_blocked_maha_valid) / len(iou_blocked_rows) if iou_blocked_rows else 0.0

    by_track_state = {
        state: _summarize_rows([row for row in rows if row.get("track_state") == state])
        for state in ("New", "Tracked", "Lost")
    }
    by_detection_tier = {
        tier: _summarize_rows([row for row in rows if (row.get("det_tier") or row.get("matched_det_tier")) == tier])
        for tier in ("high", "low", "deleted_high")
    }
    by_outcome = {
        "matched": _summarize_rows(matched_rows),
        "unmatched": _summarize_rows(unmatched_best_rows),
        "high_matched": _summarize_rows([row for row in matched_rows if row.get("matched_det_tier") == "high"]),
        "low_matched": _summarize_rows([row for row in matched_rows if row.get("matched_det_tier") == "low"]),
        "deleted_high_matched": _summarize_rows([row for row in matched_rows if row.get("matched_det_tier") == "deleted_high"]),
    }

    conclusions = []

    if deleted_high_to_lost > 0 and (deleted_high_with_valid_normal_candidate / max(1, match_counts["deleted_high"])) < 0.3:
        conclusions.append("relaxed detections look useful for lost recovery")

    if match_counts["deleted_high"] > 0 and (deleted_high_with_valid_normal_candidate / match_counts["deleted_high"]) >= 0.3:
        conclusions.append("relaxed detections look too broad / stealing matches")

    if unmatched_count > 0 and (blocked_unmatched_count / unmatched_count) > 0.4:
        conclusions.append("failures mostly look IoU-gate related")

    if unmatched_count > 0 and (blocked_unmatched_count / unmatched_count) <= 0.4:
        avg_unmatched_cos = _mean(unmatched_cos_dists)
        if avg_unmatched_cos > 0.4:
            conclusions.append("failures mostly look appearance-ambiguity related")

    if unmatched_with_any_maha_valid_share >= 0.20 or iou_blocked_maha_valid_share >= 0.10:
        conclusions.append("Mahalanobis looks promising for missed associations")
    if matched_rejected_by_maha_share >= 0.05:
        conclusions.append("Mahalanobis gate looks risky for successful matches")
    tracked_rejections = by_track_state["Tracked"]["matched_rejected_by_mahalanobis"]
    tracked_matches = by_track_state["Tracked"]["matched_rows"]
    if tracked_matches and tracked_rejections / tracked_matches >= 0.03:
        conclusions.append("Mahalanobis gate looks risky for matched Tracked tracks")
    lost_valid = by_track_state["Lost"]["mahalanobis_gate_pass_rate"]
    tracked_valid = by_track_state["Tracked"]["mahalanobis_gate_pass_rate"]
    if lost_valid > tracked_valid + 0.10:
        conclusions.append("Mahalanobis may be more useful for Lost tracks than active Tracked tracks")

    json_data = {
        "total_audited_frames": total_audited_frames,
        "total_association_rows": total_association_rows,
        "total_track_decisions": len(track_groups),
        "match_counts": match_counts,
        "deleted_high_match_rate": deleted_high_match_rate,
        "deleted_high_to_lost": deleted_high_to_lost,
        "deleted_high_to_tracked": deleted_high_to_tracked,
        "deleted_high_with_valid_normal_candidate": deleted_high_with_valid_normal_candidate,
        "deleted_high_with_valid_normal_candidate_percentage": deleted_high_with_valid_normal_candidate_percentage,
        "close_margin_stealing_cases": close_margin_stealing_cases,
        "close_margin_stealing_percentage": close_margin_stealing_percentage,
        "deleted_high_normal_cost_gap_mean": deleted_high_normal_cost_gap_mean,
        "deleted_high_normal_cost_gap_median": deleted_high_normal_cost_gap_median,
        "deleted_high_normal_better_count": deleted_high_normal_better_count,
        "deleted_high_normal_within_005_count": deleted_high_normal_within_005_count,
        "deleted_high_to_lost_share": deleted_high_to_lost_share,
        "deleted_high_to_tracked_share": deleted_high_to_tracked_share,
        "mean_best_second_margin": mean_margin,
        "median_best_second_margin": median_margin,
        "mean_iou_by_tier": mean_iou,
        "median_iou_by_tier": median_iou,
        "mean_cos_dist_by_tier": mean_cos,
        "median_cos_dist_by_tier": median_cos,
        "tracks_blocked_by_iou_gate": tracks_blocked_by_iou_gate,
        "unmatched_tracks": unmatched_count,
        "unmatched_blocked_by_iou_gate": blocked_unmatched_count,
        "mahalanobis": {
            "mean_matched_mahalanobis_squared": _mean(matched_maha),
            "median_matched_mahalanobis_squared": _median(matched_maha),
            "mean_unmatched_best_mahalanobis_squared": _mean(unmatched_best_maha),
            "median_unmatched_best_mahalanobis_squared": _median(unmatched_best_maha),
            "matched_gate_pass_rate": matched_gate_pass_rate,
            "unmatched_best_gate_pass_rate": unmatched_best_gate_pass_rate,
            "matched_rejected_by_mahalanobis": matched_rejected_by_maha,
            "matched_rejected_by_mahalanobis_share": matched_rejected_by_maha_share,
            "unmatched_tracks_with_any_mahalanobis_valid_candidate": unmatched_with_any_maha_valid,
            "unmatched_tracks_with_any_mahalanobis_valid_candidate_share": unmatched_with_any_maha_valid_share,
            "iou_blocked_but_mahalanobis_valid_candidates": len(iou_blocked_maha_valid),
            "iou_blocked_but_mahalanobis_valid_candidates_share": iou_blocked_maha_valid_share,
            "by_track_state": by_track_state,
            "by_detection_tier": by_detection_tier,
            "by_outcome": by_outcome,
        },
        "conclusions": conclusions
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)

    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# Association Audit Summary Report\n\n")
        f.write("## Overview\n")
        f.write(f"- **Total Audited Frames**: {total_audited_frames}\n")
        f.write(f"- **Total Association Pair Rows**: {total_association_rows}\n")
        f.write(f"- **Total Track Decisions**: {len(track_groups)}\n\n")

        f.write("## Match Counts by Tier\n")
        f.write(f"- **High-confidence Detections**: {match_counts['high']}\n")
        f.write(f"- **Low-confidence Detections**: {match_counts['low']}\n")
        f.write(f"- **Deleted High-confidence (Relaxed) Detections**: {match_counts['deleted_high']}\n")
        f.write(f"- **Unmatched (None)**: {match_counts['none']}\n\n")

        f.write("## Mahalanobis Summary\n")
        f.write(f"- **Matched Mean / Median Mahalanobis^2**: {_mean(matched_maha):.4f} / {_median(matched_maha):.4f}\n")
        f.write(f"- **Unmatched Best Mean / Median Mahalanobis^2**: {_mean(unmatched_best_maha):.4f} / {_median(unmatched_best_maha):.4f}\n")
        f.write(f"- **Matched Gate Pass Rate**: {matched_gate_pass_rate * 100.0:.2f}%\n")
        f.write(f"- **Unmatched Best Gate Pass Rate**: {unmatched_best_gate_pass_rate * 100.0:.2f}%\n")
        f.write(f"- **Matched Pairs Rejected by Mahalanobis**: {matched_rejected_by_maha} ({matched_rejected_by_maha_share * 100.0:.2f}%)\n")
        f.write(f"- **Unmatched Tracks With Any Mahalanobis-valid Candidate**: {unmatched_with_any_maha_valid} ({unmatched_with_any_maha_valid_share * 100.0:.2f}%)\n")
        f.write(f"- **IoU-blocked but Mahalanobis-valid Candidates**: {len(iou_blocked_maha_valid)} ({iou_blocked_maha_valid_share * 100.0:.2f}% of IoU-blocked candidates)\n\n")

        f.write("## Mahalanobis by Track State\n")
        f.write("| State | Rows | Matched Rows | Mean M^2 | Median M^2 | Gate Pass Rate | Matched Rejected |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for state, stats in by_track_state.items():
            f.write(f"| {state} | {stats['rows']} | {stats['matched_rows']} | {stats['mean_mahalanobis_squared']:.4f} | {stats['median_mahalanobis_squared']:.4f} | {stats['mahalanobis_gate_pass_rate'] * 100.0:.2f}% | {stats['matched_rejected_by_mahalanobis']} |\n")
        f.write("\n")

        f.write("## Mahalanobis by Detection Tier\n")
        f.write("| Tier | Rows | Matched Rows | Mean M^2 | Median M^2 | Gate Pass Rate | Matched Rejected |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for tier, stats in by_detection_tier.items():
            f.write(f"| {tier} | {stats['rows']} | {stats['matched_rows']} | {stats['mean_mahalanobis_squared']:.4f} | {stats['median_mahalanobis_squared']:.4f} | {stats['mahalanobis_gate_pass_rate'] * 100.0:.2f}% | {stats['matched_rejected_by_mahalanobis']} |\n")
        f.write("\n")

        f.write("## Relaxed (Deleted High) Match Details\n")
        f.write(f"- **Deleted High Match Rate**: {deleted_high_match_rate * 100.0:.2f}%\n")
        f.write(f"- **Deleted High to Lost Tracks**: {deleted_high_to_lost} (Share: {deleted_high_to_lost_share * 100.0:.2f}%)\n")
        f.write(f"- **Deleted High to Tracked Tracks**: {deleted_high_to_tracked} (Share: {deleted_high_to_tracked_share * 100.0:.2f}%)\n")
        f.write(f"- **Deleted High With Valid Normal Candidate**: {deleted_high_with_valid_normal_candidate} ({deleted_high_with_valid_normal_candidate_percentage:.2f}% of deleted high matches)\n")
        f.write(f"- **Close Margin Stealing Cases (<= 0.15)**: {close_margin_stealing_cases} ({close_margin_stealing_percentage:.2f}% of deleted high matches)\n")
        f.write(f"- **Cost Gap (Normal - Deleted High) Mean**: {deleted_high_normal_cost_gap_mean:.4f}\n")
        f.write(f"- **Cost Gap (Normal - Deleted High) Median**: {deleted_high_normal_cost_gap_median:.4f}\n")
        f.write(f"- **Deleted High Cost Better than Normal Count**: {deleted_high_normal_better_count}\n")
        f.write(f"- **Deleted High Cost Within 0.05 of Normal Count**: {deleted_high_normal_within_005_count}\n\n")

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
