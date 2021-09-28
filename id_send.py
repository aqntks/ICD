import pandas as pd
import pprint
import argparse
import requests


DETECTION_URL = "http://0.0.0.0:5090/aa"


def authenticity(image):
    image_data = open(image, "rb").read()

    # 신분증 진위확인 요청
    response = requests.post(f'{DETECTION_URL}/predict', files={"file": image_data, "user_agent": 'code1system'}).json()
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


def change_threshold(threshold):
    response = requests.post(f'{DETECTION_URL}/predict/threshold?value={threshold}').json()
    pprint.pprint(response)

    # return 예시
    # {
    #     "success": true,
    #     “message”:”메시지”
    # }


if __name__ == "__main__":
    authenticity('data/d240.jpg')
    # change_threshold(0.6)