from fivesafe.labels import LABELS

class Track():
    def __init__(
        self, 
        xywh: tuple, 
        label_id: int, 
        score: int, 
        detection_id: int, 
        id: int, 
        angle: float,
        threshold: int = 10
    ) -> None:
        self.id = id
        self.xywh = xywh
        self.rotation_angle = angle
        self.label_id = label_id
        self.score = score
        self.detection_id = detection_id
        self.threshold = threshold

    def __repr__(self) -> str:
        return f'Track id: {self.id}, class: {self.label()}, \
            score: {self.score:.2f}, box: {self.xywh}, \
            detection_id: {self.detection_id}'

    def label(self) -> str:
        return LABELS[self.label_id]

    def xyxy(self) -> list[int]:
        x, y, w, h = self.xywh
        return [
            int(x),
            int(y),
            int(x + w),
            int(x + h),
        ]
    
    def is_collision_between_bbox_and_img_border(self, img_width, img_height):
        x1, y1, x2, y2 = self.xyxy()
        if self.check_collision(x1, img_width, self.threshold) \
            or self.check_collision(y1, img_height, self.threshold) \
            or self.check_collision(x2, img_width, self.threshold) \
            or self.check_collision(y2, img_height, self.threshold):
            return True
        return False 

    @staticmethod
    def check_collision(bbox_coord, img_parameter, threshold):
        if(
            bbox_coord > (0 + threshold) \
            and bbox_coord < (img_parameter - threshold)
        ):
            return False
        return True
