import base64
import time

import pandas as pd
import pprint
import argparse
import requests

DETECTION_URL = "http://115.178.87.240:5090/menesdemo"
# DETECTION_URL = "http://192.168.219.203:5090/menesdemo"


def authenticity(image):
    DETECTION_URL = "http://115.178.87.240:5090/menesdemo"
    image_data = open(image, "rb").read()

    # 신분증 진위확인 요청
    response = requests.post(f'{DETECTION_URL}/predict', files={"file": image_data}).json()
    pprint.pprint(response)

    # return 예시
    # {
    #     "success": true,
    #     "prediction": {
    #         {
    #             "label": "copy",
    #             "probability": 0.5123
    #         }
    #     }
    # }
    return response


def change_threshold(threshold):
    DETECTION_URL = "http://115.178.87.240:5090/menesdemo"
    response = requests.post(f'{DETECTION_URL}/predict/threshold?value={threshold}').json()
    pprint.pprint(response)

    # return 예시
    # {
    #     "success": true,
    #     “message”:”메시지”
    # }


if __name__ == "__main__":
    t1 = time.time()
    authenticity(r"C:\Users\home\Desktop\idscan\id_test/test (01).JPG")
    print(time.time() - t1)
    # change_threshold(0.6)