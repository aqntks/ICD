
import io
import cv2
import pandas as pd
import yaml
import torch
import argparse
import numpy as np
import json

from collections import OrderedDict

from core.id_scan import pt_detect
from models.experimental import attempt_load
from code1ocr.code1ocr import Code1OCR

import pprint
from PIL import Image
from flask import Flask, request
from waitress import serve

app = Flask(__name__)

DETECTION_URL = "/aa"


@app.route(f'{DETECTION_URL}/bb', methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("file"):
        image_file = request.files["file"]
        image_bytes = image_file.read()
        agent = request.files["user_agent"].read().decode('utf-8')
        print('agent:', agent)

        result = OrderedDict()
        result['success'] = True
        result['prediction'] = {"label": "copy", "probability": 0.5123}

        result_json = json.dumps(result, ensure_ascii=False)
        pprint.pprint(result_json)

        return result_json


@app.route(f'{DETECTION_URL}/cc/dd', methods=["POST"])
def threshold():
    if not request.method == "POST":
        return

    result = OrderedDict()
    result['success'] = True
    result['message'] = '메시지'

    result_json = json.dumps(result, ensure_ascii=False)
    pprint.pprint(result_json)

    return result_json


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5090)
    # serve(app, host='0.0.0.0', port=5090)
