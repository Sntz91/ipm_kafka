from kafka import KafkaConsumer, KafkaProducer
from collections import Counter
import json
import datetime

INPUT_TOPIC = 'detected_objects_pp1_1'
OUTPUT_TOPIC = 'crowded_pp1_1'
SERVER = '17.11.1.21:19095'
CROWDED_THRESHOLD = 3

def parse_detection(detection):
    label = detection['detectedClass']
    return label

def is_crowded(object_count):
    if object_count['pedestrian'] > CROWDED_THRESHOLD:
        return True
    return False

if __name__ == '__main__':
    LAST_CROWDED_TS = datetime.datetime.now()
    consumer = KafkaConsumer(INPUT_TOPIC, bootstrap_servers=SERVER)
    producer = KafkaProducer(bootstrap_servers=SERVER)
    last_crowded_signal = datetime.datetime.now()
    crowded_state = False
    print('Starting Application')
    
    for msg in consumer:
        prev_state = crowded_state
        prev_crowded_state = crowded_state

        msg_json = json.loads(msg.value.decode())
        detections = msg_json['detectedObjects']
        detected_classes = [parse_detection(detection) for detection in detections]
        object_count = Counter(detected_classes)

        # TODO ENTPRELLEN
        if is_crowded(object_count):
            crowded_state = True
            print('Crowded, nr of peds: ', object_count['pedestrian'])
            msg = json.dumps({'Crowded': True}).encode('utf-8')
        else:
            crowded_state = False
            msg = json.dumps({'Crowded': False}).encode('utf-8')

        if crowded_state != prev_crowded_state:
            producer.send(OUTPUT_TOPIC, msg)
        print('Detected Objects:', object_count)

