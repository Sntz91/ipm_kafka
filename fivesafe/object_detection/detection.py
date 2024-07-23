from fivesafe.labels import LABELS
import numpy as np

class Detection():
    def __init__(self, 
        xyxy: list[float], 
        xyxy_rotated: np.ndarray, 
        label_id: int, 
        score: int
    ):
        self.id = None
        self.xyxy_rotated = xyxy_rotated
        self.xyxy = xyxy
        self.label_id = label_id
        self.score = score

    def xywh(self) -> list[int]:
        return [
            int(self.xyxy[0]),
            int(self.xyxy[1]),
            int(self.xyxy[2]) - int(self.xyxy[0]),
            int(self.xyxy[3]) - int(self.xyxy[1])
        ]

    def label(self) -> str:
        return LABELS[self.label_id]

    def is_from_interest(self):
        if self.label() == 'rider':
            return False
        return True

    def __repr__(self) -> str:
        return f'Detection id: {self.id}, class: {self.label()}, \
            score: {self.score}, box: {self.xyxy}'
