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
        matches, u_tracks, u_dets = iterative_assignment(tracked_lost, dets_high, dets_low, dets_del_high,
                                                         self.args.match_thr, self.args.penalty_p, self.args.penalty_q,
                                                         self.args.reduce_step, self.frame_id)

        for t, d in matches: tracked_lost[t].update(self.frame_id, dets_all[d])
        for t in u_tracks: tracked_lost[t].mark_lost()

        dets_high_left = [dets_all[i] for i in u_dets if i < len(dets_high)]
        matches, u_tracks, u_dets = iterative_assignment(new, dets_high_left, [], [], self.args.match_thr,
                                                         self.args.penalty_p, self.args.penalty_q,
                                                         self.args.reduce_step, self.frame_id)

        for t, d in matches: new[t].update(self.frame_id, dets_high_left[d])
        for t in u_tracks: new[t].mark_removed()

        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]
        self.init_tracks([dets_high_left[udx] for udx in u_dets])

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