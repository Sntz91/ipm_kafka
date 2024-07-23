from fivesafe.object_detection import Detection, Detections
from fivesafe.image_tracking import Tracker as ImageTracker
from fivesafe.world_tracking import Tracker as WorldTracker
from fivesafe.bev import PositionEstimation
from fivesafe.labels import LABELS
from kafka import KafkaConsumer, KafkaProducer
import json 
import cv2
import numpy as np

INPUT_TOPIC = 'detected_objects_pp1_1'
OUTPUT_TOPIC = 'world_tracks_pp1_1'
SERVER = '17.11.1.21:19095'
HOMOGRAPHY_FILE = './conf/homography.json'
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

def parse_detection(detection):
    bbox = detection['boundingBox'] 
    x, y, w, h, angle = (
        bbox['x']*IMG_WIDTH, 
        bbox['y']*IMG_HEIGHT, 
        bbox['width']*IMG_WIDTH,
        bbox['height']*IMG_HEIGHT,
        bbox['rotationAngleDegree']
    )
    obj_class = detection['detectedClass']
    confidence = detection['confidence']
    return x, y, w, h, angle, confidence, LABELS.index(obj_class)

def main():
    consumer = KafkaConsumer(INPUT_TOPIC, bootstrap_servers=SERVER) 
    producer = KafkaProducer(bootstrap_servers=SERVER)
    image_tracker = ImageTracker(IMG_WIDTH, IMG_HEIGHT)
    world_tracker = WorldTracker()
    position_estimator = PositionEstimation(
        HOMOGRAPHY_FILE,
        16
    )
    print('Starting Application')
    # Thing is. First I have to get xyxy from rotated bbox. This will make 
    # the box quiet big. However, this is already implemented in the 
    # tracker.
    # Better solution (for second step) would be to make tracking with 
    # rotated bboxes.

    for msg in consumer:
        detections = Detections()
        msg_json = json.loads(msg.value.decode())
        # Detections 
        detected_objects = msg_json['detectedObjects']
        for detection in detected_objects:
            x, y, w, h, angle, confidence, label_id = parse_detection(detection)
            # rotated_rect = ((x, y), (w, h), angle)
            # box_points = cv2.boxPoints(rotated_rect)
            # print(box_points)
            print(detection)
            detection_candidate = Detection(
                xywh = [x, y, w, h],
                score = confidence,
                label_id = label_id,
                angle = angle,
            )
            if not detection_candidate.is_from_interest:
                continue
            detections.append_measurement(detection_candidate)
        image_tracks = image_tracker.track(detections)
        image_tracks_transformed = position_estimator.transform(
            image_tracks
        )
        world_tracks = world_tracker.track(
            image_tracks_transformed
        )
        producer.send(OUTPUT_TOPIC, json.dumps(world_tracks.to_json()).encode('utf-8'))

    consumer.close()


def main_debug():
    consumer = KafkaConsumer(INPUT_TOPIC, bootstrap_servers=SERVER) 
    producer = KafkaProducer(bootstrap_servers=SERVER)
    image_tracker = ImageTracker(IMG_WIDTH, IMG_HEIGHT)
    world_tracker = WorldTracker()
    position_estimator = PositionEstimation(
        HOMOGRAPHY_FILE,
        16
    )
    pv_org = cv2.imread("perspective_view.jpg")
    # cv2.namedWindow("pv_rotated_bboxes", cv2.WINDOW_NORMAL)
    cv2.namedWindow("pv_tracking", cv2.WINDOW_NORMAL)

    print("Starting Application")
    for msg in consumer:
        pv = pv_org.copy()
        detections = Detections()
        msg_json = json.loads(msg.value.decode())
        # Detections 
        detected_objects = msg_json['detectedObjects']
        for detection in detected_objects:
            x, y, w, h, angle, confidence, label_id = parse_detection(detection)
            # This is the thing i get and is rotated
            rotated_rect = ((x, y), (w, h), angle)
            box_points = cv2.boxPoints(rotated_rect)
            box_points = np.int0(box_points)
            # This is now the non-rotated
            xx1 = int(np.min(box_points[:,0]))
            yy1 = int(np.min(box_points[:,1]))
            xx2 = int(np.max(box_points[:,0]))
            yy2 = int(np.max(box_points[:,1]))
            ww = xx2-xx1
            hh = yy2-yy1

            detection_candidate = Detection(
                xywh = [xx1, yy1, ww, hh],
                score = confidence,
                label_id = label_id,
                angle = angle,
            )
            if not detection_candidate.is_from_interest:
                continue
            detections.append_measurement(detection_candidate)
            # cv2.drawContours(pv, [box_points], 0, (0, 255, 0), 2)
            # cv2.rectangle(pv, (xx1, yy1), (xx1+ww, yy1+hh), (255, 0, 0), 2)
        # cv2.imshow('pv_rotated_bboxes', pv)
        if cv2.waitKey(1) == ord('q'):
            break
        image_tracks = image_tracker.track(detections)
        print(image_tracks)
        # TODO: plot function for image tracks. No need to have it in class. utils.
        image_tracks_transformed = position_estimator.transform(
            image_tracks
        )
        world_tracks = world_tracker.track(
            image_tracks_transformed
        )
        producer.send(OUTPUT_TOPIC, json.dumps(world_tracks.to_json()).encode('utf-8'))
    consumer.close()

if __name__ == '__main__': 
    main_debug()
