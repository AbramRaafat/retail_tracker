from .utils import *
from .track import *

class Tracker(object):
    def __init__(self, args):
        self.args = args
        self.max_time_lost = args.max_time_lost
        self.tracks = []
        self.frame_id = 0
        self.counter = TrackCounter()

    def init_tracks(self, dets):
        tracks = [t for t in self.tracks if t.state == TrackState.Tracked or t.state == TrackState.New]
        iou_sim = iou_distance(tracks + dets, tracks + dets)[0]
        scores = np.array([d.score for d in dets])
        allow_indices = track_aware_nms(iou_sim, scores, len(tracks), self.args.tai_thr, self.args.init_thr)
        for idx, flag in enumerate(allow_indices):
            if flag:
                dets[idx].initiate(self.frame_id, self.counter)
                self.tracks.append(dets[idx])

    def update(self, dets, dets_95):
        self.frame_id += 1
        
        # Capture track states before update
        track_states_before = {t.track_id: t.state for t in self.tracks}
        
        dets_del = find_deleted_detections(dets, dets_95)
        dets = [Track(self.args, d) for d in dets]
        dets_del_tracks = [Track(self.args, d) for d in dets_del]

        dets_high = [d for d in dets if d.score > self.args.det_thr]
        dets_low = [d for d in dets if d.score <= self.args.det_thr]
        dets_del_high = [d for d in dets_del_tracks if d.score > self.args.det_thr]

        tracked_lost = [t for t in self.tracks if t.state == TrackState.Tracked or t.state == TrackState.Lost]
        new = [t for t in self.tracks if t.state == TrackState.New]

        [t.predict() for t in tracked_lost]
        [t.predict() for t in new]

        dets_all = dets_high + dets_low + dets_del_high
        
        # Determine association mode
        mode = getattr(self.args, "relaxed_association_mode", "recovery_only")
        
        if mode == "recovery_only":
            # Stage 1: Tracked/Lost vs dets_high + dets_low only
            matches1, u_tracks1, u_dets1 = iterative_assignment(
                tracked_lost, dets_high, dets_low, [],
                self.args.match_thr, self.args.penalty_p, 0.0,
                self.args.reduce_step, self.frame_id,
                cost_mode=self.args.cost_mode,
                context=self,
                association_stage="tracked_lost_vs_normal"
            )
            
            # Save Stage 1 matches and mark unmatched tracks as lost initially
            for t, d in matches1:
                tracked_lost[t].update(self.frame_id, dets_all[d])
            for t in u_tracks1:
                tracked_lost[t].mark_lost()

            # Relaxed recovery stage
            matches_rec = []
            u_tracks_rec = []
            u_dets_rec = []
            recovery_matched_ids = set()
            recovery_candidates = []
            
            if getattr(self.args, "relaxed_recovery_enabled", True) and len(dets_del_high) > 0:
                recovery_candidate_original_indices = []
                for t_idx in u_tracks1:
                    track = tracked_lost[t_idx]
                    prev_state = track_states_before.get(track.track_id)
                    if prev_state == TrackState.Lost and getattr(self.args, "relaxed_recovery_for_lost", True):
                        recovery_candidates.append(track)
                        recovery_candidate_original_indices.append(t_idx)
                    elif prev_state == TrackState.Tracked and getattr(self.args, "relaxed_recovery_for_unmatched_tracked", False):
                        recovery_candidates.append(track)
                        recovery_candidate_original_indices.append(t_idx)
                        
                if len(recovery_candidates) > 0:
                    matches_rec, u_tracks_rec, u_dets_rec = iterative_assignment(
                        recovery_candidates, dets_del_high, [], [],
                        self.args.relaxed_recovery_match_thr, self.args.relaxed_recovery_penalty, 0.0,
                        self.args.reduce_step, self.frame_id,
                        cost_mode=self.args.cost_mode,
                        context=self,
                        association_stage="relaxed_recovery"
                    )
                    
                    # Apply updates for recovery matches with optional feature freezing
                    for t_rec, d_rec in matches_rec:
                        track = recovery_candidates[t_rec]
                        det = dets_del_high[d_rec]
                        freeze = getattr(self.args, "relaxed_recovery_freeze_feature_update", True)
                        track.update(self.frame_id, det, freeze_feature_update=freeze)
                        recovery_matched_ids.add(track.track_id)

            # Stage 2: New tracks vs high normal detections left unmatched
            matched_dets_high_indices = {d for _, d in matches1 if d < len(dets_high)}
            dets_high_left = [dets_high[i] for i in range(len(dets_high)) if i not in matched_dets_high_indices]
            
            matches_new, u_tracks_new, u_dets_new = iterative_assignment(
                new, dets_high_left, [], [],
                self.args.match_thr, self.args.penalty_p, 0.0,
                self.args.reduce_step, self.frame_id,
                cost_mode=self.args.cost_mode,
                context=self,
                association_stage="new_vs_high_left"
            )
            
            matches2 = list(matches_new)
            u_tracks2 = list(u_tracks_new)
            
            for t, d in matches_new:
                new[t].update(self.frame_id, dets_high_left[d])
            for t in u_tracks_new:
                new[t].mark_removed()
                
        else: # original_pool mode
            # Stage 1: Tracked/Lost vs dets_high + dets_low + dets_del_high
            matches1_all, u_tracks1_all, u_dets1_all = iterative_assignment(
                tracked_lost, dets_high, dets_low, dets_del_high,
                self.args.match_thr, self.args.penalty_p, self.args.penalty_q,
                self.args.reduce_step, self.frame_id,
                cost_mode=self.args.cost_mode,
                context=self,
                association_stage="tracked_lost_vs_all"
            )
            
            matches1 = list(matches1_all)
            u_tracks1 = list(u_tracks1_all)
            
            for t, d in matches1_all:
                tracked_lost[t].update(self.frame_id, dets_all[d])
            for t in u_tracks1_all:
                tracked_lost[t].mark_lost()

            dets_high_left = [dets_all[i] for i in u_dets1_all if i < len(dets_high)]
            
            matches_new, u_tracks_new, u_dets_new = iterative_assignment(
                new, dets_high_left, [], [],
                self.args.match_thr, self.args.penalty_p, 0.0,
                self.args.reduce_step, self.frame_id,
                cost_mode=self.args.cost_mode,
                context=self,
                association_stage="new_vs_high_left"
            )
            
            matches2 = list(matches_new)
            u_tracks2 = list(u_tracks_new)
            
            for t, d in matches_new:
                new[t].update(self.frame_id, dets_high_left[d])
            for t in u_tracks_new:
                new[t].mark_removed()
                
            matches_rec = []
            recovery_matched_ids = set()

        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        # Audit Frame Summary Logging
        audit_writer = getattr(self, "audit_writer", None)
        if audit_writer is not None:
            if mode == "recovery_only":
                matches_total = len(matches1) + len(matches_rec) + len(matches2)
                matches_high = sum(1 for _, d in matches1 if d < len(dets_high)) + len(matches2)
                matches_low = sum(1 for _, d in matches1 if len(dets_high) <= d < len(dets_high) + len(dets_low))
                matches_deleted_high = len(matches_rec)
                
                matches_deleted_high_to_tracked = sum(
                    1 for t, _ in matches_rec 
                    if track_states_before.get(recovery_candidates[t].track_id) == TrackState.Tracked
                )
                matches_deleted_high_to_lost = sum(
                    1 for t, _ in matches_rec 
                    if track_states_before.get(recovery_candidates[t].track_id) == TrackState.Lost
                )
                
                unmatched_tracked = sum(
                    1 for t in u_tracks1 
                    if track_states_before.get(tracked_lost[t].track_id) == TrackState.Tracked and tracked_lost[t].track_id not in recovery_matched_ids
                )
                unmatched_lost = sum(
                    1 for t in u_tracks1 
                    if track_states_before.get(tracked_lost[t].track_id) == TrackState.Lost and tracked_lost[t].track_id not in recovery_matched_ids
                )
                unmatched_new = len(u_tracks2)
            else:
                matches_total = len(matches1) + len(matches2)
                matches_high = sum(1 for _, d in matches1 if d < len(dets_high)) + len(matches2)
                matches_low = sum(1 for _, d in matches1 if len(dets_high) <= d < len(dets_high) + len(dets_low))
                matches_deleted_high = sum(1 for _, d in matches1 if d >= len(dets_high) + len(dets_low))
                
                matches_deleted_high_to_tracked = sum(
                    1 for t, d in matches1 
                    if d >= len(dets_high) + len(dets_low) and track_states_before.get(tracked_lost[t].track_id) == TrackState.Tracked
                )
                matches_deleted_high_to_lost = sum(
                    1 for t, d in matches1 
                    if d >= len(dets_high) + len(dets_low) and track_states_before.get(tracked_lost[t].track_id) == TrackState.Lost
                )
                
                unmatched_tracked = sum(
                    1 for t in u_tracks1 
                    if track_states_before.get(tracked_lost[t].track_id) == TrackState.Tracked
                )
                unmatched_lost = sum(
                    1 for t in u_tracks1 
                    if track_states_before.get(tracked_lost[t].track_id) == TrackState.Lost
                )
                unmatched_new = len(u_tracks2)

            summary = {
                "frame_id": self.frame_id,
                "num_tracks_total": len(self.tracks),
                "num_tracks_tracked": sum(1 for t in self.tracks if track_states_before.get(t.track_id) == TrackState.Tracked),
                "num_tracks_lost": sum(1 for t in self.tracks if track_states_before.get(t.track_id) == TrackState.Lost),
                "num_tracks_new": sum(1 for t in self.tracks if track_states_before.get(t.track_id) == TrackState.New),
                "num_dets_high": len(dets_high),
                "num_dets_low": len(dets_low),
                "num_dets_deleted_high": len(dets_del_high),
                "matches_total": matches_total,
                "matches_high": matches_high,
                "matches_low": matches_low,
                "matches_deleted_high": matches_deleted_high,
                "matches_deleted_high_to_tracked": matches_deleted_high_to_tracked,
                "matches_deleted_high_to_lost": matches_deleted_high_to_lost,
                "unmatched_tracked": unmatched_tracked,
                "unmatched_lost": unmatched_lost,
                "unmatched_new": unmatched_new,
            }
            audit_writer.add_frame_summary(summary)

        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]
        self.init_tracks([dets_high_left[udx] for udx in u_dets_new])

        return [t for t in self.tracks if t.state == TrackState.Tracked]

    def update_without_detections(self):
        self.frame_id += 1
        self.tracks = [t for t in self.tracks if t.state != TrackState.New]
        [t.predict() for t in self.tracks]
        for t in self.tracks: t.mark_lost()
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]
        return []
