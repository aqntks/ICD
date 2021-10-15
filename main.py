import os
import time
import yaml
import torch
import argparse
import pandas as pd
from id_send import authenticity

from code1ocr.code1ocr import Code1OCR
from easys.easyocr import Reader

from core.util import watchDir
from core.id_card import *
from core.id_scan import pt_detect
from models.experimental import attempt_load


def main(arg):
    gpu, gray, ciou, auth_on = arg.gpu, arg.gray, arg.ciou, arg.auth
    img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo', 'gif']

    # 디바이스 세팅
    if gpu == -1:
        dev = 'cpu'
    else:
        dev = f'cuda:{gpu}'
    device = torch.device(dev)

    # config 로드

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    img_path, cls_weight, jumin_weight, driver_weight, passport_weight, welfare_weight, alien_weight, hangul_weight, encnum_weight = \
        config['images'], config['cls-weights'], config['jumin-weights'], config['driver-weights'], \
        config['passport-weights'], config['welfare-weights'], config['alien-weights'], config['hangul-weights'], config['encnum-weights']
    f.close()

    # 모델 세팅
    cls_model = attempt_load(cls_weight, map_location=device)
    jumin_model = attempt_load(jumin_weight, map_location=device)
    driver_model = attempt_load(driver_weight, map_location=device)
    passport_model = attempt_load(passport_weight, map_location=device)
    welfare_model = attempt_load(welfare_weight, map_location=device)
    alien_model = attempt_load(alien_weight, map_location=device)
    hangul_model = attempt_load(hangul_weight, map_location=device)
    encnum_model = attempt_load(encnum_weight, map_location=device)
    models = (cls_model, jumin_model, driver_model, passport_model, welfare_model, alien_model, hangul_model, encnum_model)

    code1ocr_dg = Reader(['en'], only_dg=True)
    # code1ocr_en = Reader(['en'], only_dg=False)
    code1ocr_en_ko = Reader(['en', 'ko'], only_dg=False)
    print('----- 모델 로드 완료 -----')

    jumin_result_csv, driver_result_csv, welfare_result_csv, alien_result_csv, passport_result_csv = \
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ####### tests
    result_csv = pd.DataFrame()

    fileList = watchDir(img_path)
    if fileList:
        images = [x for x in fileList if x.split('.')[-1].lower() in img_formats]
        for img in images:

            # pytorch 검출
            result = pt_detect(img, device, models, ciou, code1ocr_dg, code1ocr_en_ko, gray=gray, byteMode=False, perspect=False)
            print(img)
            if result is None:
                print('검출 실패')
            else:
                # 진위 여부
                if auth_on:
                    auth_result = authenticity('temp/auth_temp.jpg')
                    if auth_result['success'] == 'true':
                        probability = auth_result['prediction']['probability']
                        label = auth_result['prediction']['label']
                        result.set_auth(probability, label)

                result.resultPrint()

            if type(result) is Jumin:
                df = result.mkDataFrame(img)
                jumin_result_csv = pd.concat([jumin_result_csv, df])
            if type(result) is Driver:
                df = result.mkDataFrame(img)
                driver_result_csv = pd.concat([driver_result_csv, df])
            if type(result) is Welfare:
                df = result.mkDataFrame(img)
                welfare_result_csv = pd.concat([welfare_result_csv, df])
            if type(result) is Alien:
                df = result.mkDataFrame(img)
                alien_result_csv = pd.concat([alien_result_csv, df])
            if type(result) is Passport:
                df = result.mkDataFrame(img)
                passport_result_csv = pd.concat([passport_result_csv, df])

            # ####### test$$$4
            if type(result) is Jumin:
                if len(result.name) > 3:
                    result.name = result.name[0:3]

            if type(result) is Jumin:
                df = pd.DataFrame({"jumin": [result.regnum], "name": [result.name],
                                   "license": ['-'], "encnum": ['-'], "issue": [result.issueDate]})
                result_csv = pd.concat([result_csv, df])
            if type(result) is Driver:
                if len(result.local):
                    result.licensenum = result.local + '-' + result.licensenum
                df = pd.DataFrame({"jumin": [result.regnum], "name": [result.name],
                                   "license": [result.licensenum], "encnum": [result.encnum], "issue": [result.issueDate]})
                result_csv = pd.concat([result_csv, df])

        images.clear()

    ###3 테스트 용
    result_csv.to_csv('csv/result.csv', index=False, encoding='utf-8-sig')

    print(jumin_result_csv)
    jumin_result_csv.to_csv('csv/jumin_result.csv', index=False, encoding='utf-8-sig')
    print(driver_result_csv)
    driver_result_csv.to_csv('csv/driver_result.csv', index=False, encoding='utf-8-sig')
    print(welfare_result_csv)
    welfare_result_csv.to_csv('csv/welfare_result.csv', index=False, encoding='utf-8-sig')
    print(alien_result_csv)
    alien_result_csv.to_csv('csv/alien_result.csv', index=False, encoding='utf-8-sig')
    print(passport_result_csv)
    passport_result_csv.to_csv('csv/passport_result.csv', index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--ciou', type=float, default=25)
    parser.add_argument('--gray', type=bool, default=False)
    parser.add_argument('--auth', type=bool, default=False)
    opt = parser.parse_args()
    main(opt)
