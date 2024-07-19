from .world_tracker import WorldSort
from .tracks import Tracks
import numpy as np

class Tracker:
    def __init__(self):
        self.world_tracker_vehicles = WorldSort()
        self.world_tracker_vrus = WorldSort()

    def track(self, tracks): 
        """ returns world_tracks_vehicles and world_tracks_vrus """
        result_tracks = Tracks()

        dets_world_vehicles = np.empty((0, 4))
        dets_world_vrus = np.empty((0, 4))
        for track in tracks: 
            if track.label_id in [2, 3, 5, 7]: 
                dets_world_vehicles = np.append(
                        dets_world_vehicles, 
                        np.array([[track.xy[0], track.xy[1], track.label_id, track.detection_id]]),
                        axis=0
                )
            else:
                dets_world_vrus = np.append(
                        dets_world_vrus, 
                        np.array([[track.xy[0], track.xy[1], track.label_id, track.detection_id]]), 
                        axis=0
                )
        trjs_vehicles = self.world_tracker_vehicles.update(dets_world_vehicles)
        trjs_vrus = self.world_tracker_vrus.update(dets_world_vrus)
        
        # If one list is empty, return the other one without concatenating
        # TODO: Better solution
        if trjs_vehicles.shape[0] == 0:
            return result_tracks.numpy_to_tracks(trjs_vrus)
        if trjs_vrus.shape[0] == 0:
            return result_tracks.numpy_to_tracks(trjs_vehicles)
        trjs_overall = np.concatenate((trjs_vehicles, trjs_vrus), axis=0)

        result_tracks = result_tracks.numpy_to_tracks(trjs_overall)
        return result_tracks
