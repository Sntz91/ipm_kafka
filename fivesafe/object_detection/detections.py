import numpy as np

class Detections(dict):
    def __init__(self) -> None:
        super().__init__()

    # def __setitem__(self, item, value: Detection) -> None:
        # """ assert that only Detection class valid """
        # assert type(value) == Detection
        # super().__setitem__(item, value)

    def append_measurement(self, measurement) -> None:
        id_ = len(self)+1
        measurement.id = id_
        self[id_] = measurement

    def to_numpy(self) -> np.ndarray:
        """ measurements to numpy: [x,y,x,y,score,id,label_id] """
        out = []
        for _, measurement in self.items():
            x1, y1, x2, y2 = measurement.xyxy
            out.append([
                x1, y1, x2, y2,
                measurement.score,
                int(measurement.id),
                measurement.label_id
            ])
        if len(out) == 0:
            return np.empty((0, 7))
        return np.array(out)
