from fivesafe.labels import LABELS

class Track():
    def __init__(
        self, 
        xyxy: tuple, 
        label_id: int, 
        score: int, 
        detection_id: int, 
        id: int, 
        threshold: int = 10
    ) -> None:
        self.id = id # TODO: No overflow plz
        self.xyxy = xyxy
        self.label_id = label_id
        self.score = score
        self.detection_id = detection_id
        self.threshold = threshold

    def __repr__(self) -> str:
        return f'Track id: {self.id}, class: {self.label()}, \
            score: {self.score:.2f}, box: {self.xyxy}, \
            detection_id: {self.detection_id}'

    def label(self) -> str:
        return LABELS[self.label_id]

    def xywh(self) -> list[int]:
        return [
            int(self.xyxy[0]),
            int(self.xyxy[1]),
            int(self.xyxy[2]) - int(self.xyxy[0]),
            int(self.xyxy[3]) - int(self.xyxy[1])
        ]
    
    def is_collision_between_bbox_and_img_border(self, img_width, img_height):
        x1, y1, x2, y2 = self.xyxy
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
