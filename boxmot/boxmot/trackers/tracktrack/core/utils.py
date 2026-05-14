import lap
import numpy as np

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

def iterative_assignment(tracks, dets_high, dets_low, dets_del_high, match_thr, penalty_p, penalty_q, reduce_step, frame_id, d_t=3):
    matches, dets = [], dets_high + dets_low + dets_del_high
    iou_sim, iou_dist = iou_distance(tracks, dets)
    cost = 0.50 * iou_dist + 0.50 * cos_distance(tracks, dets) + 0.10 * conf_distance(tracks, dets) + 0.05 * angle_distance(tracks, dets, frame_id, d_t)
    cost[:, len(dets_high):len(dets_high + dets_low)] += penalty_p
    cost[:, len(dets_high + dets_low):] += penalty_q
    cost[iou_sim <= 0.10] = 1.
    cost = np.clip(cost, 0, 1)

    while True:
        matches_ = associate(cost, match_thr)
        match_thr -= reduce_step
        if len(matches_) == 0: break
        matches += matches_
        for t, d in matches_:
            cost[t, :] = cost[:, d] = 1.

    m_tracks, m_dets = [t for t, _ in matches], [d for _, d in matches]
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