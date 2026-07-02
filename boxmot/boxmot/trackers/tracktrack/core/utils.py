import lap
import logging
import numpy as np

logger = logging.getLogger(__name__)

def bbox_overlaps(a_x1y1x2y2, b_x1y1x2y2):
    num_a, num_b = a_x1y1x2y2.shape[0], b_x1y1x2y2.shape[0]
    overlaps = np.zeros((num_a, num_b))
    for n_b in range(num_b):
        box_area = (b_x1y1x2y2[n_b, 2] - b_x1y1x2y2[n_b, 0] + 1) * (b_x1y1x2y2[n_b, 3] - b_x1y1x2y2[n_b, 1] + 1)
        for n_a in range(num_a):
            iw = min(a_x1y1x2y2[n_a, 2], b_x1y1x2y2[n_b, 2]) - max(a_x1y1x2y2[n_a, 0], b_x1y1x2y2[n_b, 0]) + 1
            if iw > 0:
                ih = min(a_x1y1x2y2[n_a, 3], b_x1y1x2y2[n_b, 3]) - max(a_x1y1x2y2[n_a, 1], b_x1y1x2y2[n_b, 1]) + 1
                if ih > 0:
                    ua = (a_x1y1x2y2[n_a, 2] - a_x1y1x2y2[n_a, 0] + 1) * (a_x1y1x2y2[n_a, 3] - a_x1y1x2y2[n_a, 1] + 1) + box_area - iw * ih
                    overlaps[n_a, n_b] = iw * ih / ua
    return overlaps

def find_deleted_detections(dets, dets_95):
    if len(dets_95) == 0:
        return np.empty((0, dets.shape[1]))
    a_x1y1x2y2 = np.ascontiguousarray(dets[:, :4], dtype=np.float64)
    b_x1y1x2y2 = np.ascontiguousarray(dets_95[:, :4], dtype=np.float64)
    ious = bbox_overlaps(a_x1y1x2y2, b_x1y1x2y2)
    return dets_95[np.max(ious, axis=0) < 0.97] if len(ious) > 0 else dets_95

def iou_distance(a_tracks, b_tracks):
    a_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in a_tracks], dtype=np.float64)
    b_boxes = np.ascontiguousarray([t.x1y1x2y2 for t in b_tracks], dtype=np.float64)
    if len(a_boxes) == 0 or len(b_boxes) == 0:
        return np.zeros((len(a_boxes), len(b_boxes)), dtype=np.float64), np.ones((len(a_boxes), len(b_boxes)), dtype=np.float64)
    h_iou = (np.minimum(a_boxes[:, 3:4], b_boxes[:, 3:4].T) - np.maximum(a_boxes[:, 1:2], b_boxes[:, 1:2].T)) / \
            (np.maximum(a_boxes[:, 3:4], b_boxes[:, 3:4].T) - np.minimum(a_boxes[:, 1:2], b_boxes[:, 1:2].T) + 1e-6)
    iou_sim = bbox_overlaps(a_boxes, b_boxes)
    return iou_sim, 1 - h_iou * iou_sim

def cos_distance(tracks, dets):
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)
    t_feat = np.concatenate([t.feat for t in tracks], axis=0)
    d_feat = np.concatenate([d.feat for d in dets], axis=0)
    return np.clip(1 - np.dot(t_feat, d_feat.T), 0., 1.)

def conf_distance(tracks, dets):
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)
    t_score_prev = []
    for t in tracks:
        frame_ids = sorted(list(t.history.keys()), reverse=True)
        t_score_prev.append(t.history[frame_ids[min(1, len(frame_ids) - 1)]][1])
    t_score = np.array([t.score for t in tracks]) + (np.array([t.score for t in tracks]) - np.array(t_score_prev))
    return np.abs(t_score[:, None] - np.array([d.score for d in dets])[None, :])

def get_prev_box(history, frame_id, dt):
    return history[frame_id - dt][0] if frame_id - dt in history else history[max(history.keys())][0]

def get_vel_t_d(b_1, b_2):
    b_1, b_2 = b_1[:, np.newaxis, :], b_2[np.newaxis, :, :]
    deltas = b_2 - b_1
    norm_lt = np.sqrt(deltas[:, :, 0:1]**2 + deltas[:, :, 1:2]**2) + 1e-5
    norm_lb = np.sqrt(deltas[:, :, 0:1]**2 + deltas[:, :, 3:4]**2) + 1e-5
    norm_rt = np.sqrt(deltas[:, :, 2:3]**2 + deltas[:, :, 1:2]**2) + 1e-5
    norm_rb = np.sqrt(deltas[:, :, 2:3]**2 + deltas[:, :, 3:4]**2) + 1e-5
    vel_lt = np.stack([deltas[:, :, 0], deltas[:, :, 1]], axis=-1) / norm_lt
    vel_lb = np.stack([deltas[:, :, 0], deltas[:, :, 3]], axis=-1) / norm_lb
    vel_rt = np.stack([deltas[:, :, 2], deltas[:, :, 1]], axis=-1) / norm_rt
    vel_rb = np.stack([deltas[:, :, 2], deltas[:, :, 3]], axis=-1) / norm_rb
    return np.stack([vel_lt, vel_lb, vel_rt, vel_rb], axis=2)

def calc_angle(vel_t, vel_t_d):
    angle_ = 0
    for vdx in range(vel_t.shape[2]):
        vel_t_x = np.repeat(vel_t[:, :, vdx, 0], vel_t_d.shape[1], axis=1)
        vel_t_y = np.repeat(vel_t[:, :, vdx, 1], vel_t_d.shape[1], axis=1)
        angle = np.abs(np.arccos(np.clip(vel_t_x * vel_t_d[:, :, vdx, 0] + vel_t_y * vel_t_d[:, :, vdx, 1], -1, 1))) / np.pi
        angle_ += angle / 4
    return angle_

def angle_distance(tracks, dets, frame_id, d_t=3):
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)
    track_boxes = np.stack([get_prev_box(t.history, frame_id, d_t) for t in tracks], axis=0)
    angle_dist = calc_angle(np.stack([t.velocity for t in tracks], axis=0)[:, np.newaxis], get_vel_t_d(track_boxes, np.stack([d.x1y1x2y2 for d in dets], axis=0)))
    return angle_dist * np.array([d.score for d in dets])[np.newaxis, :]

def associate(cost, match_thr):
    matches = []
    if cost.shape[0] > 0 and cost.shape[1] > 0:
        min_ddx, min_tdx = np.argmin(cost, axis=1), np.argmin(cost, axis=0)
        for tdx, ddx in enumerate(min_ddx):
            if min_tdx[ddx] == tdx and cost[tdx, ddx] < match_thr:
                matches.append([tdx, ddx])
    return matches

def compute_association_cost(tracks, dets, iou_sim, iou_dist, frame_id, d_t=3, cost_mode="static", context=None):
    if cost_mode != "static":
        logger.warning(
            "Unsupported TrackTrack cost_mode '%s'; falling back to static association cost. "
            "TODO: add adaptive association here.",
            cost_mode,
        )

    return (
        0.50 * iou_dist
        + 0.50 * cos_distance(tracks, dets)
        + 0.10 * conf_distance(tracks, dets)
        + 0.05 * angle_distance(tracks, dets, frame_id, d_t)
    )

def iterative_assignment(tracks, dets_high, dets_low, dets_del_high, match_thr, penalty_p, penalty_q, reduce_step, frame_id, d_t=3, cost_mode="static", context=None, association_stage="unknown"):
    matches, dets = [], dets_high + dets_low + dets_del_high
    iou_sim, iou_dist = iou_distance(tracks, dets)
    cost = compute_association_cost(
        tracks,
        dets,
        iou_sim,
        iou_dist,
        frame_id,
        d_t=d_t,
        cost_mode=cost_mode,
        context=context,
    )
    cost[:, len(dets_high):len(dets_high + dets_low)] += penalty_p
    cost[:, len(dets_high + dets_low):] += penalty_q
    
    # Store blocked status
    blocked_by_iou_gate = (iou_sim <= 0.10)
    
    cost[blocked_by_iou_gate] = 1.
    cost = np.clip(cost, 0, 1)

    # Copy of cost matrix after gating/clipping
    original_cost = cost.copy()

    # Precompute distance matrices only if audit is enabled
    audit_writer = getattr(context, "audit_writer", None) if context is not None else None
    if audit_writer is not None:
        cos_dist = cos_distance(tracks, dets)
        conf_dist = conf_distance(tracks, dets)
        angle_dist = angle_distance(tracks, dets, frame_id, d_t)
    else:
        cos_dist = None
        conf_dist = None
        angle_dist = None

    while True:
        matches_ = associate(cost, match_thr)
        match_thr -= reduce_step
        if len(matches_) == 0: break
        matches += matches_
        for t, d in matches_:
            cost[t, :] = cost[:, d] = 1.

    m_tracks, m_dets = [t for t, _ in matches], [d for _, d in matches]
    
    # Perform auditing
    if audit_writer is not None:
        state_map = {0: "New", 1: "Tracked", 2: "Lost", 3: "Removed"}
        matched_d_map = {t_idx: d_idx for t_idx, d_idx in matches}
        
        for t in range(len(tracks)):
            track = tracks[t]
            track_state = state_map.get(track.state, "Unknown")
            track_history_len = len(track.history)
            frames_since_update = frame_id - track.end_frame_id
            
            best_det_index = -1
            best_det_tier = "none"
            best_det_score = 0.0
            best_final_cost = 1.0
            second_best_final_cost = 1.0
            best_second_margin = 0.0
            best_iou_sim = 0.0
            best_iou_dist = 1.0
            best_cos_dist = 1.0
            best_conf_dist = 1.0
            best_angle_dist = 1.0
            best_blocked_by_iou_gate = False
            
            best_normal_det_index = -1
            best_normal_det_tier = "none"
            best_normal_final_cost = 1.0

            if len(dets) > 0:
                track_costs = original_cost[t, :]
                sorted_indices = np.argsort(track_costs)
                best_d = int(sorted_indices[0])
                best_det_index = best_d
                best_final_cost = float(track_costs[best_d])
                
                if association_stage == "relaxed_recovery":
                    best_det_tier = "deleted_high"
                elif best_d < len(dets_high):
                    best_det_tier = "high"
                elif best_d < len(dets_high) + len(dets_low):
                    best_det_tier = "low"
                else:
                    best_det_tier = "deleted_high"
                
                best_det_score = float(dets[best_d].score)
                best_iou_sim = float(iou_sim[t, best_d])
                best_iou_dist = float(iou_dist[t, best_d])
                best_cos_dist = float(cos_dist[t, best_d])
                best_conf_dist = float(conf_dist[t, best_d])
                best_angle_dist = float(angle_dist[t, best_d])
                best_blocked_by_iou_gate = bool(blocked_by_iou_gate[t, best_d])
                
                if len(dets) > 1:
                    second_d = int(sorted_indices[1])
                    second_best_final_cost = float(track_costs[second_d])
                else:
                    second_best_final_cost = 1.0
                best_second_margin = second_best_final_cost - best_final_cost

                # Find best normal candidate (high or low)
                normal_indices = [i for i in range(len(dets)) if i < len(dets_high) + len(dets_low)]
                if len(normal_indices) > 0:
                    normal_costs = original_cost[t, normal_indices]
                    best_normal_idx_in_subset = np.argmin(normal_costs)
                    best_normal_d = int(normal_indices[best_normal_idx_in_subset])
                    
                    best_normal_det_index = best_normal_d
                    best_normal_final_cost = float(normal_costs[best_normal_idx_in_subset])
                    if best_normal_d < len(dets_high):
                        best_normal_det_tier = "high"
                    else:
                        best_normal_det_tier = "low"

            matched = t in matched_d_map
            if matched:
                matched_d = int(matched_d_map[t])
                matched_det_index = matched_d
                if association_stage == "relaxed_recovery":
                    matched_det_tier = "deleted_high"
                elif matched_d < len(dets_high):
                    matched_det_tier = "high"
                elif matched_d < len(dets_high) + len(dets_low):
                    matched_det_tier = "low"
                else:
                    matched_det_tier = "deleted_high"
                
                matched_det_score = float(dets[matched_d].score)
                matched_final_cost = float(original_cost[t, matched_d])
                matched_iou_sim = float(iou_sim[t, matched_d])
                matched_cos_dist = float(cos_dist[t, matched_d])
                matched_conf_dist = float(conf_dist[t, matched_d])
                matched_angle_dist = float(angle_dist[t, matched_d])
            else:
                matched_det_index = -1
                matched_det_tier = "none"
                matched_det_score = 0.0
                matched_final_cost = 1.0
                matched_iou_sim = 0.0
                matched_cos_dist = 1.0
                matched_conf_dist = 1.0
                matched_angle_dist = 1.0

            recovered_by_deleted_high = False
            feature_update_frozen = False
            
            if matched and (matched_det_tier == "deleted_high" or association_stage == "relaxed_recovery"):
                recovered_by_deleted_high = True
                mode = getattr(context.args, "relaxed_association_mode", "recovery_only")
                if mode == "recovery_only":
                    if getattr(context.args, "relaxed_recovery_freeze_feature_update", True):
                        feature_update_frozen = True

            if association_stage == "relaxed_recovery":
                stage_val = "relaxed_recovery_lost" if track.state == 2 else "relaxed_recovery_unmatched_tracked"
            else:
                stage_val = association_stage

            row = {
                "frame_id": int(frame_id),
                "association_stage": stage_val,
                "track_index": int(t),
                "track_id": int(track.track_id),
                "track_state": track_state,
                "track_history_len": int(track_history_len),
                "frames_since_update": int(frames_since_update),
                "num_dets_high": int(len(dets_high)),
                "num_dets_low": int(len(dets_low)),
                "num_dets_deleted_high": int(len(dets_del_high)),
                "best_det_index": int(best_det_index),
                "best_det_tier": best_det_tier,
                "best_det_score": float(best_det_score),
                "best_final_cost": float(best_final_cost),
                "second_best_final_cost": float(second_best_final_cost),
                "best_second_margin": float(best_second_margin),
                "best_iou_sim": float(best_iou_sim),
                "best_iou_dist": float(best_iou_dist),
                "best_cos_dist": float(best_cos_dist),
                "best_conf_dist": float(best_conf_dist),
                "best_angle_dist": float(best_angle_dist),
                "best_blocked_by_iou_gate": bool(best_blocked_by_iou_gate),
                "matched": bool(matched),
                "matched_det_index": int(matched_det_index),
                "matched_det_tier": matched_det_tier,
                "matched_det_score": float(matched_det_score),
                "matched_final_cost": float(matched_final_cost),
                "matched_iou_sim": float(matched_iou_sim),
                "matched_cos_dist": float(matched_cos_dist),
                "matched_conf_dist": float(matched_conf_dist),
                "matched_angle_dist": float(matched_angle_dist),
                "best_normal_det_index": int(best_normal_det_index),
                "best_normal_det_tier": best_normal_det_tier,
                "best_normal_final_cost": float(best_normal_final_cost),
                "recovered_by_deleted_high": bool(recovered_by_deleted_high),
                "feature_update_frozen": bool(feature_update_frozen)
            }
            audit_writer.write_row(row)

    return matches, [t for t in range(len(tracks)) if t not in m_tracks], [d for d in range(len(dets)) if d not in m_dets]

def track_aware_nms(pair_sims, scores, num_tracks, nms_thresh, score_thresh):
    num_dets = len(pair_sims) - num_tracks
    allow_indices = np.ones(num_dets) * (scores > score_thresh)
    for idx in range(num_dets):
        if allow_indices[idx] == 0: continue
        if num_tracks > 0 and np.max(pair_sims[num_tracks + idx, :num_tracks]) > nms_thresh:
            allow_indices[idx] = 0; continue
        for jdx in range(num_dets):
            if idx != jdx and allow_indices[jdx] == 1 and scores[idx] > scores[jdx]:
                if pair_sims[num_tracks + idx, num_tracks + jdx] > nms_thresh:
                    allow_indices[jdx] = 0
    return allow_indices == 1
