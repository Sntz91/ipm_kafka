from fivesafe.object_detection import Detection, Detections
from fivesafe.image_tracking import Tracker as ImageTracker
from fivesafe.world_tracking import Tracker as WorldTracker
from fivesafe.bev import PositionEstimation
from fivesafe.measurements.labels import LABELS
from kafka import KafkaConsumer, KafkaProducer
import json 

INPUT_TOPIC = 'detected_objects_pp1_2'
OUTPUT_TOPIC = 'world_tracks_pp1_2'
SERVER = '17.11.1.21:19095'
HOMOGRAPHY_FILE = './conf/homography.json'
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

def parse_detection(detection):
    bbox = detection['boundingBox'] 
    x, y, w, h = (
        int(bbox['x']*IMG_WIDTH), 
        int(bbox['y']*IMG_HEIGHT), 
        int(bbox['width']*IMG_WIDTH), 
        int(bbox['height']*IMG_HEIGHT)
    )
    obj_class = detection['detectedClass']
    confidence = detection['confidence']
    return x, y, w, h, confidence, LABELS.index(obj_class)

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

    for msg in consumer:
        detections = Detections()
        msg_json = json.loads(msg.value.decode())
        # Detections 
        detected_objects = msg_json['detectedObjects']
        for detection in detected_objects:
            x, y, w, h, confidence, label_id = parse_detection(detection)
            detection_candidate = Detection(
                xyxy = [x, y, x+w, y+h],
                score = confidence,
                label_id = label_id
            )
            # TODO: Function is_from_interest()
            if detection_candidate.label() == 'rider':
                continue
            detections.append_measurement(detection_candidate)
        image_tracks = image_tracker.track(detections)
        image_tracks_transformed = position_estimator.transform(
            image_tracks,
            detections
        )
        world_tracks = world_tracker.track(
            image_tracks_transformed
        )
        producer.send(OUTPUT_TOPIC, json.dumps(world_tracks.to_json()).encode('utf-8'))

    consumer.close()

if __name__ == '__main__': 
    main()
