def draw_detections(frame, detections, mask=False):
    for detection in detections:
        frame = draw_detection(frame, detection, mask)
    return frame

def draw_detection(frame, detection, mask=False, draw_id=True, draw_score=True):
    if mask:
        frame = detection.draw_mask(
            frame,
            thickness=2
        )
    else:
        frame = detection.draw_rectangle(
            frame, 
            thickness=2
        )
    frame = detection.draw_label(frame)
    if draw_id:
        frame = detection.draw_id(frame)
    if draw_score:
        frame = detection.draw_score(frame)
    return frame

def draw_detection_offset(frame, detection):
    frame = detection.draw_rectangle(
        frame, 
        offset=(5, 5), 
        color=(0, 0, 0)
    )
    frame = detection.draw_id(
        frame, 
        offset=(0.5*detection.xywh()[2], 0.5*detection.xywh()[3]), 
        color=(0, 0, 0)
    )
    return frame
