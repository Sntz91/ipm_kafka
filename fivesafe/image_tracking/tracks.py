import numpy as np
from . import Track

class Tracks(list):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        out = ''
        for track in self:
            out += f'{track.id}: {track}\n'
        return out

    def to_numpy(self) -> np.ndarray:
        """ measurements to numpy: [x,y,x,y,score,id,label_id] """
        out = []
        for measurement in self:
            x1, y1, x2, y2 = measurement.xyxy()
            out.append([
                x1, y1, x2, y2, 
                measurement.score,
                int(measurement.id),
                measurement.label_id
            ])
        if len(out) == 0:
            return np.empty((0, 7))
        return np.array(out)

    def numpy_to_tracks(self, track_list: np.ndarray, img_w, img_h):
        for track_candidate in track_list:
            x, y, w, h, track_id, detection_id, score, label_id = track_candidate
            track_candidate = Track(
                xywh=(x, y, w, h), 
                label_id=int(label_id),
                score=score, 
                id=int(track_id), 
                detection_id=int(detection_id),
                angle = 0.0 # TODO: Important
            )
            if track_candidate.is_collision_between_bbox_and_img_border(img_w, img_h):
                continue
            self.append_measurement(track_candidate)
        return self

    def append_measurement(self, measurement) -> None:
        self.append(measurement)

    def get_world_positions(self):
        positions = []
        for track in self:
            positions.append(track.world_position)
        return positions
