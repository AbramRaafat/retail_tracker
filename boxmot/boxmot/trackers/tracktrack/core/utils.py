import lap
import logging
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_COST_MODES = {
    "static",
    "static_shadow_mahalanobis",
    "static_mahalanobis_gate",
    "static_mahalanobis_blend",
}

MAHALANOBIS_THRESHOLDS = {
    2: {0.95: 5.9915, 0.99: 9.2103, 0.999: 13.8155},
    4: {0.95: 9.4877, 0.99: 13.2767, 0.999: 18.4668},
}

STATE_NAMES = {0: "New", 1: "Tracked", 2: "Lost", 3: "Removed"}


def bbox_overlaps(a_x1y1x2y2, b_x1y1x2y2):
    a = np.asarray(a_x1y1x2y2, dtype=np.float64)
    b = np.asarray(b_x1y1x2y2, dtype=np.float64)
    num_a, num_b = a.shape[0], b.shape[0]
    if num_a == 0 or num_b == 0:
        return np.zeros((num_a, num_b), dtype=np.float64)

    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + 1.0
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + 1.0
    valid = (iw > 0.0) & (ih > 0.0)
    inter = np.where(valid, iw * ih, 0.0)

    area_a = (a[:, 2] - a[:, 0] + 1.0) * (a[:, 3] - a[:, 1] + 1.0)
    area_b = (b[:, 2] - b[:, 0] + 1.0) * (b[:, 3] - b[:, 1] + 1.0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)

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

def compute_association_components(tracks, dets, iou_dist, frame_id, d_t=3):
    cos_dist = cos_distance(tracks, dets)
    conf_dist = conf_distance(tracks, dets)
    angle_dist = angle_distance(tracks, dets, frame_id, d_t)
    static_cost = (
        0.50 * iou_dist
        + 0.50 * cos_dist
        + 0.10 * conf_dist
        + 0.05 * angle_dist
    )
    return static_cost, cos_dist, conf_dist, angle_dist


def compute_association_cost(tracks, dets, iou_sim, iou_dist, frame_id, d_t=3, cost_mode="static", context=None):
    if cost_mode not in SUPPORTED_COST_MODES and not getattr(context, "_warned_unknown_cost_mode", False):
        logger.warning(
            "Unsupported TrackTrack cost_mode '%s'; falling back to static association cost. ",
            cost_mode,
        )
        if context is not None:
            context._warned_unknown_cost_mode = True

    return compute_association_components(tracks, dets, iou_dist, frame_id, d_t)[0]


def _mahalanobis_threshold(args):
    dim = int(getattr(args, "mahalanobis_gate_dim", 4) or 4)
    if dim not in (2, 4):
        dim = 4
    override = getattr(args, "mahalanobis_gate_threshold", None)
    if override is not None:
        try:
            override = float(override)
            if override > 0:
                return override
        except (TypeError, ValueError):
            pass

    confidence = float(getattr(args, "mahalanobis_gate_confidence", 0.99) or 0.99)
    confidence = min(MAHALANOBIS_THRESHOLDS[dim], key=lambda value: abs(value - confidence))
    return MAHALANOBIS_THRESHOLDS[dim][confidence]


def _mahalanobis_solve_squared(track, det, gate_dim=4, jitter=1e-6):
    if track.kalman_filter is None or track.mean is None or track.covariance is None:
        return np.inf, True

    projected_mean, projected_cov = track.kalman_filter.project(track.mean, track.covariance, det.score)
    measurement = det.cxcywh.copy()
    if int(gate_dim) == 2:
        projected_mean = projected_mean[:2]
        projected_cov = projected_cov[:2, :2]
        measurement = measurement[:2]

    innovation = measurement - projected_mean
    eye = np.eye(projected_cov.shape[0], dtype=np.float64)
    cov = np.asarray(projected_cov, dtype=np.float64)

    for scale in (0.0, float(jitter), float(jitter) * 10.0, float(jitter) * 100.0):
        try:
            chol = np.linalg.cholesky(cov + scale * eye)
            y = np.linalg.solve(chol, innovation)
            return float(np.dot(y, y)), False
        except np.linalg.LinAlgError:
            continue

    return np.inf, True


def mahalanobis_distance_matrix(tracks, dets, args):
    num_tracks, num_dets = len(tracks), len(dets)
    distances = np.full((num_tracks, num_dets), np.inf, dtype=np.float64)
    failures = np.zeros((num_tracks, num_dets), dtype=bool)
    if num_tracks == 0 or num_dets == 0:
        return distances, failures

    gate_dim = int(getattr(args, "mahalanobis_gate_dim", 4) or 4)
    if gate_dim not in (2, 4):
        gate_dim = 4
    jitter = float(getattr(args, "mahalanobis_jitter", 1e-6) or 1e-6)

    for t_idx, track in enumerate(tracks):
        for d_idx, det in enumerate(dets):
            distances[t_idx, d_idx], failures[t_idx, d_idx] = _mahalanobis_solve_squared(
                track,
                det,
                gate_dim=gate_dim,
                jitter=jitter,
            )
    return distances, failures


def _state_applies(track_state, apply_to):
    state_name = STATE_NAMES.get(track_state, "Unknown")
    if apply_to == "all":
        return True
    if apply_to == "lost_only":
        return state_name == "Lost"
    if apply_to == "tracked_only":
        return state_name == "Tracked"
    return False


def _tier_applies(tier, apply_to):
    if apply_to == "all":
        return True
    if apply_to == "low_only":
        return tier == "low"
    if apply_to == "high_low":
        return tier in ("high", "low")
    if apply_to == "deleted_only":
        return tier == "deleted_high"
    return False


def _mahalanobis_apply_mask(tracks, det_tiers, args, mode):
    num_tracks, num_dets = len(tracks), len(det_tiers)
    mask = np.zeros((num_tracks, num_dets), dtype=bool)
    if num_tracks == 0 or num_dets == 0:
        return mask

    if mode == "static_mahalanobis_blend":
        apply_states = getattr(args, "mahalanobis_blend_apply_to_states", getattr(args, "mahalanobis_apply_to_states", "lost_only"))
        apply_tiers = getattr(args, "mahalanobis_blend_apply_to_tiers", getattr(args, "mahalanobis_apply_to_tiers", "all"))
    else:
        apply_states = getattr(args, "mahalanobis_apply_to_states", "lost_only")
        apply_tiers = getattr(args, "mahalanobis_apply_to_tiers", "all")

    for t_idx, track in enumerate(tracks):
        if not _state_applies(track.state, apply_states):
            continue
        for d_idx, tier in enumerate(det_tiers):
            if _tier_applies(tier, apply_tiers):
                mask[t_idx, d_idx] = True
    return mask


def _det_tiers(dets_high, dets_low, dets_del_high):
    return (
        ["high"] * len(dets_high)
        + ["low"] * len(dets_low)
        + ["deleted_high"] * len(dets_del_high)
    )


def _format_box(box):
    return " ".join(f"{float(v):.2f}" for v in box)

def iterative_assignment(tracks, dets_high, dets_low, dets_del_high, match_thr, penalty_p, penalty_q, reduce_step, frame_id, d_t=3, cost_mode="static", context=None, association_stage="unknown"):
    matches, dets = [], dets_high + dets_low + dets_del_high
    args = getattr(context, "args", None)
    audit_writer = getattr(context, "audit_writer", None) if context is not None else None
    mode = cost_mode if cost_mode in SUPPORTED_COST_MODES else "static"

    iou_sim, iou_dist = iou_distance(tracks, dets)
    static_weighted_cost, cos_dist, conf_dist, angle_dist = compute_association_components(
        tracks,
        dets,
        iou_dist,
        frame_id,
        d_t=d_t,
    )

    penalties = np.zeros_like(static_weighted_cost, dtype=np.float64)
    low_start = len(dets_high)
    deleted_start = len(dets_high) + len(dets_low)
    if penalties.size:
        penalties[:, low_start:deleted_start] = penalty_p
        penalties[:, deleted_start:] = penalty_q

    cost_after_penalty = static_weighted_cost + penalties
    blocked_by_iou_gate = iou_sim <= 0.10

    static_final_cost = cost_after_penalty.copy()
    static_final_cost[blocked_by_iou_gate] = 1.0
    static_final_cost = np.clip(static_final_cost, 0.0, 1.0)

    det_tiers = _det_tiers(dets_high, dets_low, dets_del_high)
    mahalanobis_enabled = bool(getattr(args, "mahalanobis_enabled", mode != "static"))
    need_mahalanobis = mahalanobis_enabled and (
        mode in ("static_mahalanobis_gate", "static_mahalanobis_blend")
        or audit_writer is not None
    )
    mahalanobis_threshold = _mahalanobis_threshold(args)
    mahalanobis_squared = np.full_like(static_weighted_cost, np.inf, dtype=np.float64)
    mahalanobis_failed = np.zeros_like(static_weighted_cost, dtype=bool)
    mahalanobis_passed = np.zeros_like(static_weighted_cost, dtype=bool)
    mahalanobis_normalized = np.ones_like(static_weighted_cost, dtype=np.float64)
    mahalanobis_applicable = np.zeros_like(static_weighted_cost, dtype=bool)

    if need_mahalanobis:
        mahalanobis_squared, mahalanobis_failed = mahalanobis_distance_matrix(tracks, dets, args)
        fail_open = bool(getattr(args, "mahalanobis_fail_open", True))
        mahalanobis_passed = mahalanobis_squared <= mahalanobis_threshold
        if fail_open:
            mahalanobis_passed = mahalanobis_passed | mahalanobis_failed
        finite_dist = np.where(np.isfinite(mahalanobis_squared), mahalanobis_squared, mahalanobis_threshold)
        mahalanobis_normalized = np.clip(finite_dist / mahalanobis_threshold, 0.0, 1.0)
        mahalanobis_normalized[mahalanobis_failed & fail_open] = 0.0
        mahalanobis_applicable = _mahalanobis_apply_mask(tracks, det_tiers, args, mode)

        failures = int(np.sum(mahalanobis_failed))
        if failures and context is not None:
            context.mahalanobis_failure_count = getattr(context, "mahalanobis_failure_count", 0) + failures
            if not getattr(context, "_warned_mahalanobis_failure", False):
                logger.warning(
                    "TrackTrack Mahalanobis projection failed for %d pair(s); fail_open=%s.",
                    failures,
                    fail_open,
                )
                context._warned_mahalanobis_failure = True

    cost = cost_after_penalty.copy()
    if mode == "static_mahalanobis_gate" and need_mahalanobis:
        cost[mahalanobis_applicable & ~mahalanobis_passed] = 1.0
    elif mode == "static_mahalanobis_blend" and need_mahalanobis:
        weight = float(getattr(args, "mahalanobis_weight", 0.05) or 0.05)
        cost[mahalanobis_applicable] += weight * mahalanobis_normalized[mahalanobis_applicable]

    cost[blocked_by_iou_gate] = 1.0
    cost = np.clip(cost, 0.0, 1.0)
    original_cost = cost.copy()

    while True:
        matches_ = associate(cost, match_thr)
        match_thr -= reduce_step
        if len(matches_) == 0:
            break
        matches += matches_
        for t, d in matches_:
            cost[t, :] = cost[:, d] = 1.0

    m_tracks, m_dets = [t for t, _ in matches], [d for _, d in matches]

    if audit_writer is not None:
        matched_d_map = {t_idx: d_idx for t_idx, d_idx in matches}
        stage_val_by_track = {}
        for t_idx, track in enumerate(tracks):
            if association_stage == "relaxed_recovery":
                stage_val_by_track[t_idx] = "relaxed_recovery_lost" if track.state == 2 else "relaxed_recovery_unmatched_tracked"
            else:
                stage_val_by_track[t_idx] = association_stage

        for t_idx, track in enumerate(tracks):
            track_state = STATE_NAMES.get(track.state, "Unknown")
            track_history_len = len(track.history)
            frames_since_update = frame_id - track.end_frame_id
            track_matched = t_idx in matched_d_map
            matched_d = int(matched_d_map[t_idx]) if track_matched else -1

            best_det_index = -1
            best_det_tier = "none"
            best_det_score = 0.0
            best_final_cost = 1.0
            second_best_final_cost = 1.0
            best_second_margin = 0.0
            best_normal_det_index = -1
            best_normal_det_tier = "none"
            best_normal_final_cost = 1.0
            best_static_candidate_index = -1
            best_static_candidate_tier = "none"
            best_static_candidate_passed_gate = False
            unmatched_track_had_mahalanobis_valid_candidate = False
            matched_candidate_passed_gate = False
            mahalanobis_would_reject_matched_candidate = False

            if len(dets) > 0:
                sorted_indices = np.argsort(original_cost[t_idx, :])
                best_det_index = int(sorted_indices[0])
                best_det_tier = det_tiers[best_det_index]
                best_det_score = float(dets[best_det_index].score)
                best_final_cost = float(original_cost[t_idx, best_det_index])
                if len(dets) > 1:
                    second_best_final_cost = float(original_cost[t_idx, int(sorted_indices[1])])
                best_second_margin = second_best_final_cost - best_final_cost

                static_sorted = np.argsort(static_final_cost[t_idx, :])
                best_static_candidate_index = int(static_sorted[0])
                best_static_candidate_tier = det_tiers[best_static_candidate_index]
                if need_mahalanobis:
                    best_static_candidate_passed_gate = bool(mahalanobis_passed[t_idx, best_static_candidate_index])

                normal_indices = [i for i in range(len(dets)) if i < deleted_start]
                if normal_indices:
                    normal_costs = original_cost[t_idx, normal_indices]
                    best_normal_d = int(normal_indices[int(np.argmin(normal_costs))])
                    best_normal_det_index = best_normal_d
                    best_normal_det_tier = det_tiers[best_normal_d]
                    best_normal_final_cost = float(original_cost[t_idx, best_normal_d])

                if need_mahalanobis and not track_matched:
                    unmatched_track_had_mahalanobis_valid_candidate = bool(np.any(mahalanobis_passed[t_idx, :]))

            if need_mahalanobis and track_matched and matched_d >= 0:
                matched_candidate_passed_gate = bool(mahalanobis_passed[t_idx, matched_d])
                mahalanobis_would_reject_matched_candidate = bool(
                    mahalanobis_applicable[t_idx, matched_d] and not mahalanobis_passed[t_idx, matched_d]
                )

            recovered_by_deleted_high = False
            feature_update_frozen = False
            if track_matched and matched_d >= 0:
                matched_tier = det_tiers[matched_d]
                recovered_by_deleted_high = matched_tier == "deleted_high" or association_stage == "relaxed_recovery"
                if recovered_by_deleted_high and getattr(args, "relaxed_association_mode", "recovery_only") == "recovery_only":
                    feature_update_frozen = bool(getattr(args, "relaxed_recovery_freeze_feature_update", True))

            for d_idx, det in enumerate(dets):
                pair_matched = track_matched and d_idx == matched_d
                tier = det_tiers[d_idx]
                maha_value = float(mahalanobis_squared[t_idx, d_idx]) if need_mahalanobis and np.isfinite(mahalanobis_squared[t_idx, d_idx]) else -1.0
                maha_pass = bool(mahalanobis_passed[t_idx, d_idx]) if need_mahalanobis else False
                row = {
                    "row_type": "pair",
                    "frame_id": int(frame_id),
                    "association_stage": stage_val_by_track[t_idx],
                    "track_index": int(t_idx),
                    "track_id": int(track.track_id),
                    "track_state": track_state,
                    "track_history_len": int(track_history_len),
                    "frames_since_update": int(frames_since_update),
                    "num_dets_high": int(len(dets_high)),
                    "num_dets_low": int(len(dets_low)),
                    "num_dets_deleted_high": int(len(dets_del_high)),
                    "det_index": int(d_idx),
                    "det_tier": tier,
                    "det_score": float(det.score),
                    "det_box": _format_box(det.x1y1x2y2),
                    "track_predicted_box": _format_box(track.x1y1x2y2),
                    "iou_sim": float(iou_sim[t_idx, d_idx]),
                    "iou_dist": float(iou_dist[t_idx, d_idx]),
                    "cosine_distance": float(cos_dist[t_idx, d_idx]),
                    "confidence_distance": float(conf_dist[t_idx, d_idx]),
                    "angle_distance": float(angle_dist[t_idx, d_idx]),
                    "static_weighted_cost_before_penalty": float(static_weighted_cost[t_idx, d_idx]),
                    "low_or_deleted_penalty": float(penalties[t_idx, d_idx]),
                    "cost_after_penalty": float(cost_after_penalty[t_idx, d_idx]),
                    "blocked_by_iou_gate": bool(blocked_by_iou_gate[t_idx, d_idx]),
                    "final_cost_after_gate_clip": float(original_cost[t_idx, d_idx]),
                    "track_matched": bool(track_matched),
                    "matched": bool(pair_matched),
                    "best_det_index": int(best_det_index),
                    "best_det_tier": best_det_tier,
                    "best_det_score": float(best_det_score),
                    "best_final_cost": float(best_final_cost),
                    "second_best_final_cost": float(second_best_final_cost),
                    "best_second_margin": float(best_second_margin),
                    "best_iou_sim": float(iou_sim[t_idx, best_det_index]) if best_det_index >= 0 else 0.0,
                    "best_iou_dist": float(iou_dist[t_idx, best_det_index]) if best_det_index >= 0 else 1.0,
                    "best_cos_dist": float(cos_dist[t_idx, best_det_index]) if best_det_index >= 0 else 1.0,
                    "best_conf_dist": float(conf_dist[t_idx, best_det_index]) if best_det_index >= 0 else 1.0,
                    "best_angle_dist": float(angle_dist[t_idx, best_det_index]) if best_det_index >= 0 else 1.0,
                    "best_blocked_by_iou_gate": bool(blocked_by_iou_gate[t_idx, best_det_index]) if best_det_index >= 0 else False,
                    "matched_det_index": int(d_idx) if pair_matched else -1,
                    "matched_det_tier": tier if pair_matched else "none",
                    "matched_det_score": float(det.score) if pair_matched else 0.0,
                    "matched_final_cost": float(original_cost[t_idx, d_idx]) if pair_matched else 1.0,
                    "matched_iou_sim": float(iou_sim[t_idx, d_idx]) if pair_matched else 0.0,
                    "matched_cos_dist": float(cos_dist[t_idx, d_idx]) if pair_matched else 1.0,
                    "matched_conf_dist": float(conf_dist[t_idx, d_idx]) if pair_matched else 1.0,
                    "matched_angle_dist": float(angle_dist[t_idx, d_idx]) if pair_matched else 1.0,
                    "best_normal_det_index": int(best_normal_det_index),
                    "best_normal_det_tier": best_normal_det_tier,
                    "best_normal_final_cost": float(best_normal_final_cost),
                    "best_static_candidate_index": int(best_static_candidate_index),
                    "best_static_candidate_tier": best_static_candidate_tier,
                    "recovered_by_deleted_high": bool(pair_matched and recovered_by_deleted_high),
                    "feature_update_frozen": bool(pair_matched and feature_update_frozen),
                    "mahalanobis_squared": maha_value,
                    "mahalanobis_gate_threshold": float(mahalanobis_threshold) if need_mahalanobis else -1.0,
                    "mahalanobis_passed_gate": maha_pass,
                    "mahalanobis_normalized_cost": float(mahalanobis_normalized[t_idx, d_idx]) if need_mahalanobis else -1.0,
                    "mahalanobis_applicable": bool(mahalanobis_applicable[t_idx, d_idx]) if need_mahalanobis else False,
                    "mahalanobis_failed": bool(mahalanobis_failed[t_idx, d_idx]) if need_mahalanobis else False,
                    "matched_candidate_passed_mahalanobis_gate": bool(matched_candidate_passed_gate),
                    "best_static_candidate_passed_mahalanobis_gate": bool(best_static_candidate_passed_gate),
                    "unmatched_track_had_mahalanobis_valid_candidate": bool(unmatched_track_had_mahalanobis_valid_candidate),
                    "mahalanobis_would_reject_matched_candidate": bool(mahalanobis_would_reject_matched_candidate),
                    "mahalanobis_would_allow_iou_blocked_candidate": bool(blocked_by_iou_gate[t_idx, d_idx] and maha_pass),
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
