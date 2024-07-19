from kafka import KafkaConsumer, KafkaProducer
from collections import Counter
import json

INPUT_TOPIC = 'crowded_pp1_1'
SERVER = '17.11.1.21:19095'

if __name__ == '__main__':
    consumer = KafkaConsumer(INPUT_TOPIC, bootstrap_servers=SERVER)
    
    for msg in consumer:
        msg_json = json.loads(msg.value.decode())
        print(msg_json)
