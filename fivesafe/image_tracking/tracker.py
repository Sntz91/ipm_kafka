from .sort import Sort
from . import Tracks

class Tracker:
    def __init__(self, img_width, img_height):
        self.mot_tracker = Sort()
        self.img_width = img_width
        self.img_height = img_height

    def track(self, detections): # TODO NUMPY AND detection.to_numpy() in main!
        #xyxy, score, id, label -> xyxy, old_id, new_id, score, label
        tracks = Tracks()
        print('detections:')
        print(detections.to_numpy())
        print()
        trackers = self.mot_tracker.update(detections.to_numpy()) 
        tracks = tracks.numpy_to_tracks(trackers, self.img_width, self.img_height)
        return tracks
