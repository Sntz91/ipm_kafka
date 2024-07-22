from fivesafe.labels import LABELS

class Detection():
    def __init__(self, xywh: list[float], label_id: int, score: int, angle: float):
        self.id = None
        self.xywh = xywh
        self.rotation_angle = angle
        self.label_id = label_id
        self.score = score

    def xyxy(self) -> list[int]:
        x, y, w, h = self.xywh
        return [
            int(x),
            int(y),
            int(x + w),
            int(x + h),
        ]

    def label(self) -> str:
        return LABELS[self.label_id]

    def is_from_interest(self):
        if self.label() == 'rider':
            return False
        return True

    def __repr__(self) -> str:
        return f'Detection id: {self.id}, class: {self.label()}, \
            score: {self.score}, box: {self.xywh}'
