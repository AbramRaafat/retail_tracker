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
        
        # Pass self as context and specify stage
        matches, u_tracks, u_dets = iterative_assignment(
            tracked_lost, dets_high, dets_low, dets_del_high,
            self.args.match_thr, self.args.penalty_p, self.args.penalty_q,
            self.args.reduce_step, self.frame_id,
            cost_mode=self.args.cost_mode,
            context=self,
            association_stage="tracked_lost_vs_all"
        )
        
        matches1 = list(matches)
        u_tracks1 = list(u_tracks)

        for t, d in matches: tracked_lost[t].update(self.frame_id, dets_all[d])
        for t in u_tracks: tracked_lost[t].mark_lost()

        dets_high_left = [dets_all[i] for i in u_dets if i < len(dets_high)]
        
        matches_new, u_tracks_new, u_dets_new = iterative_assignment(
            new, dets_high_left, [], [],
            self.args.match_thr, self.args.penalty_p, self.args.penalty_q,
            self.args.reduce_step, self.frame_id,
            cost_mode=self.args.cost_mode,
            context=self,
            association_stage="new_vs_high_left"
        )
        
        matches2 = list(matches_new)
        u_tracks2 = list(u_tracks_new)

        for t, d in matches_new: new[t].update(self.frame_id, dets_high_left[d])
        for t in u_tracks_new: new[t].mark_removed()

        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        # Audit Frame Summary Logging
        audit_writer = getattr(self, "audit_writer", None)
        if audit_writer is not None:
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
