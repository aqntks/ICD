import os
import time
import yaml
import torch
import argparse

from core.util import watchDir
from core.id_scan import pt_detect
from models.experimental import attempt_load


def main(arg):
    gpu = arg.gpu
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
    img_path, cls_weight, jumin_weight, driver_weight, passport_weight, welfare_weight, alien_weight, hangul_weight = \
        config['images'], config['cls-weights'], config['jumin-weights'], config['driver-weights'], \
        config['passport-weights'], config['welfare-weights'], config['alien-weights'], config['hangul-weights']
    f.close()

    # 모델 세팅
    cls_model = attempt_load(cls_weight, map_location=device)
    jumin_model = attempt_load(jumin_weight, map_location=device)
    driver_model = attempt_load(driver_weight, map_location=device)
    passport_model = attempt_load(passport_weight, map_location=device)
    welfare_model = attempt_load(welfare_weight, map_location=device)
    alien_model = attempt_load(alien_weight, map_location=device)
    hangul_model = attempt_load(hangul_weight, map_location=device)
    models = (cls_model, jumin_model, driver_model, passport_model, welfare_model, alien_model, hangul_model)

    print('----- 모델 로드 완료 -----')

    # while True:
    #     fileList = watchDir(img_path)
    #     if fileList:
    #         images = [x for x in fileList if x.split('.')[-1].lower() in img_formats]
    #         for img in images:
    #
    #             # pytorch 검출
    #             result = pt_detect(img, device, models)
    #             print(img)
    #             if result is None:
    #                 print('검출 실패')
    #             else:
    #                 result.resultPrint()
    #
    #         images.clear()
    #     else:
    #         time.sleep(1)

    fileList = watchDir(img_path)
    if fileList:
        images = [x for x in fileList if x.split('.')[-1].lower() in img_formats]
        for img in images:

            # pytorch 검출
            result = pt_detect(img, device, models)
            print(img)
            if result is None:
                print('검출 실패')
            else:
                result.resultPrint()

        images.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    opt = parser.parse_args()
    main(opt)