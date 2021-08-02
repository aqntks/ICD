"""Perform test request"""
import pprint
import argparse
import requests


def send(arg):
    DETECTION_URL = "http://localhost:5000/id-scan"
    TEST_IMAGE = arg.img

    image_data = open(TEST_IMAGE, "rb").read()

    response = requests.post(DETECTION_URL, files={"image": image_data}).json()

    pprint.pprint(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str)
    args = parser.parse_args()
    send(args)