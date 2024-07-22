import numpy as np

class Detections(list):
    def __init__(self) -> None:
        super().__init__()

    # def __setitem__(self, item, value: Detection) -> None:
        # """ assert that only Detection class valid """
        # assert type(value) == Detection
        # super().__setitem__(item, value)

    def append_measurement(self, measurement) -> None:
        measurement.id = len(self)+1
        self.append(measurement)

    def to_numpy(self) -> np.ndarray:
        """ measurements to numpy: [x,y,x,y,score,id,label_id] """
        out = []
        for measurement in self:
            x, y, w, h = measurement.xywh
            out.append([
                x, y, w, h,
                measurement.score,
                int(measurement.id),
                measurement.label_id
            ])
        if len(out) == 0:
            return np.empty((0, 7))
        return np.array(out)
