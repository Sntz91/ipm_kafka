from kafka import KafkaConsumer
import json
import cv2

INPUT_TOPIC = 'world_tracks_pp1_2'
SERVER = '17.11.1.21:19095'

if __name__ == '__main__':
    consumer = KafkaConsumer(INPUT_TOPIC, bootstrap_servers=SERVER)
    tv_org = cv2.imread("topview.jpg")
    print('Starting Application')
    
    for msg in consumer:
        print('msg received.')
        tv = tv_org.copy()
        msg_json = json.loads(msg.value.decode())
        print(msg_json)
        for k, track in msg_json.items():
            print(track['xy'], track['object_class'])
            if track['object_class'] == 'pedestrian':
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            gcp = (int(track['xy'][0]), int(track['xy'][1]))
            cv2.circle(tv, gcp, 10, color, -1)

        cv2.imshow('tv', tv)
        if cv2.waitKey(1) == ord('q'):
            break
