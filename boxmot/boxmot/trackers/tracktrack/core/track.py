import numpy as np
from .utils import get_prev_box
from .kalman_filter import KalmanFilter

def get_vel(b_1, b_2):
    deltas = b_2 - b_1
    norm_lt = np.sqrt(deltas[0]**2 + deltas[1]**2) + 1e-5
    norm_lb = np.sqrt(deltas[0]**2 + deltas[3]**2) + 1e-5
    norm_rt = np.sqrt(deltas[2]**2 + deltas[1]**2) + 1e-5
    norm_rb = np.sqrt(deltas[2]**2 + deltas[3]**2) + 1e-5
    return np.stack([
        np.array([deltas[0], deltas[1]]) / norm_lt,
        np.array([deltas[0], deltas[3]]) / norm_lb,
        np.array([deltas[2], deltas[1]]) / norm_rt,
        np.array([deltas[2], deltas[3]]) / norm_rb
    ], axis=0)

class TrackState(object):
    New = 0; Tracked = 1; Lost = 2; Removed = 3

class TrackCounter(object):
    track_count = 0
    def get_track_id(self):
        self.track_count += 1
        return self.track_count

class Track(object):
    def __init__(self, args, detection):
        self.args = args
        self.box = detection[:4]
        self.score = detection[4]
        self.cls = detection[5] if len(detection) >= 6 else 0
        self.feat = detection[6:][np.newaxis, :].copy()
        
        self.delta_t = 3
        self.history = {}
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.velocity = np.zeros((4, 2))
        self.alpha = 0.95
        
        self.track_id = 0
        self.end_frame_id = 0
        self.state = TrackState.New

    def mark_lost(self): self.state = TrackState.Lost
    def mark_removed(self): self.state = TrackState.Removed

    def update_features(self, feat, score):
        beta = self.alpha + (1 - self.alpha) * (1 - score)
        self.feat = beta * self.feat + (1 - beta) * feat
        self.feat /= np.linalg.norm(self.feat) + 1e-12

    def initiate(self, frame_id, counter):
        self.track_id = counter.get_track_id()
        self.kalman_filter = KalmanFilter()
        self.mean, self.covariance = self.kalman_filter.initiate(self.cxcywh.copy())
        self.history[frame_id] = [self.box.copy(), self.score.copy(), self.mean.copy(), self.covariance.copy(), self.feat.copy()]
        self.end_frame_id = frame_id
        self.state = TrackState.New

    def predict(self):
        if self.state != TrackState.Tracked and 'Dance' in self.args.data_path:
            self.mean[6] = self.mean[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    def update(self, frame_id, detection, freeze_feature_update=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, detection.cxcywh.copy(), detection.score)
        if not freeze_feature_update:
            self.update_features(detection.feat.copy(), detection.score)
        self.history[frame_id] = [detection.box.copy(), detection.score, self.mean.copy(), self.covariance.copy(), self.feat.copy()]
        self.velocity = np.zeros((4, 2))
        for d_t in range(1, self.delta_t + 1):
            self.velocity += get_vel(get_prev_box(self.history, frame_id, d_t).copy(), detection.x1y1x2y2) / d_t
        self.velocity /= self.delta_t
        self.box = detection.box.copy()
        self.score = detection.score
        self.cls = detection.cls
        self.end_frame_id = frame_id
        self.state = TrackState.Tracked if len(self.history.keys()) >= self.args.min_len else TrackState.New

    @property
    def cxcywh(self):
        if self.mean is None: return np.array([(self.box[0]+self.box[2])/2, (self.box[1]+self.box[3])/2, self.box[2]-self.box[0], self.box[3]-self.box[1]])
        return np.array([self.mean[0], self.mean[1], self.mean[2], self.mean[3]])

    @property
    def x1y1x2y2(self):
        if self.mean is None: return np.array([self.box[0], self.box[1], self.box[2], self.box[3]])
        return np.array([self.mean[0]-self.mean[2]/2, self.mean[1]-self.mean[3]/2, self.mean[0]+self.mean[2]/2, self.mean[1]+self.mean[3]/2])
