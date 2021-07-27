import re
import yaml

from core.util import *
from core.general import *
from core.image_handler import ImagePack


# pt 모델 설정 세팅
def model_setting(model, half, imgz):
    if half:
        model.half()
    stride = int(model.stride.max())
    img_size = check_img_size(imgz, s=stride)

    names = model.module.names if hasattr(model, 'module') else model.names

    return model, stride, img_size, names


# pt 검출
def detecting(model, img, im0s, device, img_size, half, option):
    confidence, iou = option
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))

    # 이미지 정규화
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 추론 & NMS 적용
    prediction = model(img, augment=False)[0]
    prediction = non_max_suppression(prediction, confidence, iou, classes=None, agnostic=False)

    detect = None
    for _, det in enumerate(prediction):
        obj, det[:, :4] = {}, scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
        detect = det

    return detect


# 지문 검출
def fingerScan(det, names):
    result, back, finger = [], [], []
    for *rect, conf, cls in det:
        if names[int(cls)] == 'finger':
            finger.append((rect, conf, cls))
        if names[int(cls)] == 'jumin-fingerback':
            back.append((rect, conf, cls))

    # 주민 뒷면과 지문이 전부 검출된 경우
    if len(back) and len(finger):
        for b in back:
            for f in finger:
                # 주민 뒷면 안에 지문이 있는 경우
                bX1, bY1, bX2, bY2 = int(b[0][0]), int(b[0][1]), int(b[0][2]), int(b[0][3])
                fX1, fY1, fX2, fY2 = int(f[0][0]), int(f[0][1]), int(f[0][2]), int(f[0][3])
                if bX1 < fX1 and bY1 < fY1 and bX2 > fX2 and bY2 > fY2:
                    back_area = (bY2 - bY1) * (bX2 - bX1)
                    finger_area = (fY2 - fY1) * (fX2 - fX1)
                    bCenterX, bCenterY = center_point((bX1, bY1, bX2, bY2))

                    if fX1 < bCenterX < fX2 and fY1 < bCenterY < fY2: continue

                    if finger_area / back_area < 0.25:  # 중복 영역 25% 이하인 경우
                        result.append({'x1': fX1, 'y1': fY1, 'x2': fX2, 'y2': fY2})

    return result


# 인감 검출
def ingamScan(det, names):
    det_copy = det.cpu().detach().numpy()
    result, ingam_title, jumin_num, numbers = [], [], [], []

    for x1, y1, x2, y2, cf, cls in det_copy:
        if names[int(cls)] == 'ingam_title':
            ingam_title.append([int(x1), int(y1), int(x2), int(y2), 'ingam_title'])
        elif names[int(cls)] == 'jumin_num':
            jumin_num.append([int(x1), int(y1), int(x2), int(y2), 'jumin_num'])
        elif str(names[int(cls)]).isdigit():
            numbers.append([int(x1), int(y1), int(x2), int(y2), names[int(cls)]])

    if jumin_num:
        if len(numbers) >= 0:
            numbers.sort(key=lambda n0: n0[1])

            # 생년월일, 주민번호 뒷자리 split
            yymmdd, privacynum = [], []

            for i, num in enumerate(numbers):
                if numbers[i][3] < numbers[-1][1]:
                    yymmdd.append(num)
                else:
                    privacynum.append(num)
            # 재정렬
            yymmdd.sort(key=lambda n0: n0[0])
            privacynum.sort(key=lambda n0: n0[0])

            prv_x_minus = []
            if len(privacynum) > 1:
                for pn in range(len(privacynum)):
                    prv_x_minus.append(abs(privacynum[pn][2] - privacynum[pn][0]))

                for pn in range(len(privacynum)):
                    if max(prv_x_minus) > 1.3 * min(prv_x_minus):
                        del privacynum[prv_x_minus.index(max(prv_x_minus))]
                mask_ingam_pt1 = (privacynum[0][0], privacynum[0][1])
                mask_ingam_pt2 = (privacynum[-1][2], privacynum[-1][3])

                result.append(
                    {'x1': mask_ingam_pt1[0], 'y1': mask_ingam_pt1[1], 'x2': mask_ingam_pt2[0],
                     'y2': mask_ingam_pt2[1]})

    return result


# 신분증 검출
def idScan(det, names):
    detect_mrz, detect_kor, isId = False, False, False
    hyphen, regnum, result = [], [], []

    for *rect, conf, cls in det:
        if names[int(cls)] == '-': hyphen.append(rect)
        if names[int(cls)] == 'regnum': regnum.append(rect)
        if names[int(cls)] == 'mrz': detect_mrz = True
        if names[int(cls)] == 'kor': detect_kor = True
        if names[int(cls)] == 'kor' or names[int(cls)] == 'title_jumin' or \
                names[int(cls)] == 'title_driver' or names[int(cls)] == 'title_welfare' or \
                names[int(cls)] == 'period' or names[int(cls)] == 'visatype':
            isId = True

    if isId:
        for r in regnum:
            count = 0
            for *rect, conf, cls in det:
                if r[0] <= rect[0] and r[1] <= rect[1] and r[2] >= rect[2] and r[3] >= rect[
                    3]: count += 1

            if count < 10: break

            x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            x1 = int(((r[2] - r[0]) / 7 * 4) + r[0])
            result.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    return result, detect_mrz, detect_kor


def easyOCR(reader, image_pack, kind):
    passport, alien = kind
    _, img = image_pack.getImg()
    masking_result, result = [], []
    det = reader.readtext(img)  # easyOCR 검출
    merge_text, passport_pId, detect_log = '', '', ''  # 전체 검출 텍스트 merge # 여권에서 검출 된 주민번호 뒷자리 저장 # 검출 로그

    # 외국인 등록증이 포함된 경우 - (Id_Scan 에서 kor 검출)
    if alien:
        jumin_number = re.compile(
            '([(]*\d{2}([0][1-9]|[1][0-2])([0][1-9]|[1-2]\d|[3][0-1])[-]*[1-2]\d{4,6}[)]*)|([(]*([0-1]\d|[2][0-1])([0]\d|[1][0-2])([0][1-9]|[1-2]\d|[3][0-1])[-]*[3-4]\d{4,6}[)]*)|([(]*\d{2}([0][1-9]|[1][0-2])([0][1-9]|[1-2]\d|[3][0-1])[-]*[5-8]\d{4,6}[)]*)')
    else:
        jumin_number = re.compile(
            '([(]*\d{2}([0][1-9]|[1][0-2])([0][1-9]|[1-2]\d|[3][0-1])[-]*[1-2]\d{4,6}[)]*)|([(]*([0-1]\d|[2][0-1])([0]\d|[1][0-2])([0][1-9]|[1-2]\d|[3][0-1])[-]*[3-4]\d{4,6}[)]*)')

    # 전체 검출 텍스트 merge 및 OCR 전체 결과 저장
    for pt, text, _ in det:
        x1, y1, x2, y2 = int(pt[0][0]), int(pt[0][1]), int(pt[2][0]), int(pt[2][1])
        merge_text += text
        # OCR 전체 결과 저장
        result.append({'text': text, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    # 등본인 지 여부 확인
    db_cb = True if merge_text.replace(' ', '').replace('\n', '').replace('\t', '').find('등본') != -1 else False

    # 마스킹 위치 검출
    for pt, original_text, cf in det:
        text = original_text
        pt[0][0], pt[0][1], pt[2][0], pt[2][1] = int(pt[0][0]), int(pt[0][1]), int(pt[2][0]), int(pt[2][1])
        # 여권인 경우 - (Id_Scan 에서 mrz 검출)
        if passport:
            passport_regex = re.compile('[1-4]\d{4,6}[)]*')
            if passport_regex.match(text) and len(passport_pId) == 0:  # 주민번호 뒷자리 매칭
                if len(text) < 8 and re.compile('[A-Za-z]').match(text) is None:  # 여권 내 주민번호 항목 마스킹
                    masking_result.append({'x1': pt[0][0], 'y1': pt[0][1], 'x2': pt[2][0], 'y2': pt[2][1]})
                    passport_pId = text
                    detect_log = detect_log + '[p-1] ' \
                                 + text + ' - ' + str(masking_result[len(masking_result) - 1]) + '\n'
            if passport_regex.search(text) and len(passport_pId) > 5:  # mrz 라인 중 주민번호를 포함한 라인
                id_number_in_mrz = text.find(passport_pId[0:6])
                if len(text) > 40:  # if - mrz 값이 한번에 검출
                    if id_number_in_mrz == -1:  # mrz 라인에서 주민번호 검출 안됨 - > 예측 위치로 마스킹
                        x1_ratio = 28 / len(text)
                        x2_ratio = 35 / len(text)
                        x_len = pt[2][0] - pt[0][0]
                        masking_result.append(
                            {'x1': int(pt[0][0] + x_len * x1_ratio), 'y1': pt[0][1],
                             'x2': int(pt[0][0] + x_len * x2_ratio),
                             'y2': pt[2][1]})
                        detect_log = detect_log + '[p-2] ' \
                                     + text[28:35] + ' - ' + str(
                            masking_result[len(masking_result) - 1]) + '\n'

                    else:  # mrz 라인에서 주민번호 검출 됨 - > 주민번호 위치 마스킹
                        x1_ratio = (text.find(passport_pId[0:6])) / len(text)
                        x2_ratio = (text.find(passport_pId[0:6]) + 7) / len(text)
                        x_len = pt[2][0] - pt[0][0]
                        masking_result.append(
                            {'x1': int(pt[0][0] + x_len * x1_ratio), 'y1': pt[0][1],
                             'x2': int(pt[0][0] + x_len * x2_ratio),
                             'y2': pt[2][1]})
                        detect_log = detect_log + '[p-3] ' \
                                     + text[text.find(passport_pId[0:6]):text.find(
                            passport_pId[0:6]) + 7] + ' - ' + str(
                            masking_result[len(masking_result) - 1]) + '\n'

                else:  # if - mrz 값이 떨어져서 검출
                    if id_number_in_mrz != -1:  # mrz 라인에서 주민번호 검출  - > 주민번호 위치 마스킹
                        x1_ratio = (text.find(passport_pId[0:6])) / len(text)
                        x2_ratio = (text.find(passport_pId[0:6]) + 7) / len(text)
                        x_len = pt[2][0] - pt[0][0]
                        masking_result.append(
                            {'x1': int(pt[0][0] + x_len * x1_ratio), 'y1': pt[0][1],
                             'x2': int(pt[0][0] + x_len * x2_ratio),
                             'y2': pt[2][1]})
                        detect_log = detect_log + '[p-4] ' \
                                     + text[text.find(passport_pId[0:6]):text.find(
                            passport_pId[0:6]) + 7] + ' - ' + \
                                     str(masking_result[len(masking_result) - 1]) + '\n'
        # 등본인 경우 보정
        if db_cb:
            text = re.sub('[A-Za-z]', '1', text).replace('(', '1').replace(')', '1').replace('[', '1').replace(']', '1').replace('{', '1').replace('}', '1')

        # 여권 제외 주민번호 검출
        if passport is False:
            # 주민번호 정확히 일치 (match) --- 예) 901103-2718310
            if jumin_number.match(text) and len(text) <= 20:  # 최대 20개 까지만
                juminRatio = 8 / 14
                masking_result.append(
                    {'x1': pt[0][0] + int((pt[2][0] - pt[0][0]) * juminRatio), 'y1': pt[0][1],
                     'x2': pt[2][0], 'y2': pt[2][1]})
                detect_log = detect_log + '[j-1] ' \
                             + text + ' - ' + str(masking_result[len(masking_result) - 1]) + '\n'

            # 검출 내역 중에 주민번호 (search) --- 예) 주민번호:901103-2718310입니다
            elif jumin_number.search(text) and len(text) <= 30:  # 최대 30개 까지만
                if text not in '-': pass  # print('하이픈 미검출')

                textTemp = text.replace(' ', '')
                hypen_index = textTemp.index('-')

                if 5 < hypen_index and len(textTemp) > hypen_index + 7:
                    checksum = textTemp[hypen_index - 6:hypen_index + 8]
                    if jumin_number.match(checksum):
                        hypen_index = text.index('-')
                        hypen_position = (hypen_index + 1) / len(text)
                        masking_range = 7 / len(text)
                        x1_pos = pt[0][0] + int((pt[2][0] - pt[0][0]) * hypen_position)
                        x2_pos = x1_pos + int((pt[2][0] - pt[0][0]) * masking_range)
                        masking_result.append(
                            {'x1': x1_pos, 'y1': pt[0][1], 'x2': x2_pos, 'y2': pt[2][1]})
                        detect_log = detect_log + '[j-2] ' \
                                     + checksum + ' - ' + str(
                            masking_result[len(masking_result) - 1]) + '\n'
            else:
                pass

    # 주민번호가 여러개의 상자로 나뉘어서 검출 된 경우 처리
    separated = jumin_number.search(merge_text)
    if separated:
        jumin_index = separated.start()
        jumin_check_index = 0
        for pt, text, cf in det:
            pt[0][0], pt[0][1], pt[2][0], pt[2][1] = int(pt[0][0]), int(pt[0][1]), int(pt[2][0]), int(
                pt[2][1])
            # index를 주민번호 검출 위치로 이동 , 6자리 이상만 검출하여 주민번호 앞자리와 '-' 은 pass
            if jumin_index <= jumin_check_index < jumin_index + 13 and len(text) > 6:
                if len(text) > 10:  # 주민번호 13자리 검출 된 경우
                    juminRatio = 8 / 14
                else:  # 주민번호 뒷자리만 검출 된 경우
                    juminRatio = 1 / 7
                masking_result.append(
                    {'x1': pt[0][0] + int((pt[2][0] - pt[0][0]) * juminRatio), 'y1': pt[0][1],
                     'x2': pt[2][0],
                     'y2': pt[2][1]})
                detect_log = detect_log + '[j-3] ' \
                             + text + ' - ' + str(masking_result[len(masking_result) - 1]) + '\n'
                break
            jumin_check_index += len(text)  # 다음 검출 상자로 이동

    # 등본일 경우 ( 위아래 검출해서 추가 마스킹 )
    mask_temp = masking_result.copy()
    if db_cb:
        for mask in mask_temp:
            mX1, mY1, mX2, mY2 = mask['x1'], mask['y1'], mask['x2'], mask['y2']
            mHeight = mY2 - mY1
            for pt, text, cf in det:
                ptX1, ptY1, ptX2, ptY2 = int(pt[0][0]), int(pt[0][1]), int(pt[2][0]), int(pt[2][1])
                if ptX1 < mX2 and (mY1 - mHeight * 4 < ptY2 < mY1 or mY2 < ptY1 < mY2 + mHeight * 4):
                    if len(text) > 9 and '-' in text:
                        hypen_index = text.index('-')
                        ratio = (hypen_index + 2) / len(text)
                        dbX = ptX1 + int((ptX2 - ptX1) * ratio)
                        masking_result.append({'x1': dbX, 'y1': ptY1, 'x2': ptX2, 'y2': ptY2})

    return masking_result, result


def pt_detect(path, device, models):
    finger_weights, ingam_weights, id_weights = models

    half = device.type != 'cpu'
    # config 로드
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    img_size, confidence, iou = config['finger-img_size'], config['finger-confidence'], config['finger-iou']
    finger_option = (img_size, confidence, iou)
    img_size, confidence, iou = config['ingam-img_size'], config['ingam-confidence'], config['ingam-iou']
    ingam_option = (img_size, confidence, iou)
    img_size, confidence, iou = config['id-img_size'], config['id-confidence'], config['id-iou']
    id_option = (img_size, confidence, iou)
    f.close()

    # 지문 스캔
    model, stride, img_size, names = model_setting(finger_weights, half, finger_option[0])
    image_pack = ImagePack(path, img_size, stride)
    img, im0s = image_pack.getImg()
    det = detecting(model, img, im0s, device, img_size, half, finger_option[1:])
    finger_result = fingerScan(det, names)
    finger_result = remove_intersect_box(finger_result)
    finger_count = len(finger_result)
    masking = dict_masking(finger_result, im0s)
    image_pack.setImg(masking)

    # 인감 스캔
    model, stride, img_size, names = model_setting(ingam_weights, half, ingam_option[0])
    image_pack.reset(img_size, stride)
    img, im0s = image_pack.getImg()
    det = detecting(model, img, im0s, device, img_size, half, ingam_option[1:])
    ingam_result = ingamScan(det, names)
    masking = dict_masking(ingam_result, im0s)
    image_pack.setImg(masking)

    # 신분증 스캔
    model, stride, img_size, names = model_setting(id_weights, half, id_option[0])
    image_pack.reset(img_size, stride)
    img, im0s = image_pack.getImg()
    det = detecting(model, img, im0s, device, img_size, half, id_option[1:])
    id_result, detect_mrz, detect_kor = idScan(det, names)
    kind = (detect_mrz, detect_kor)
    masking = dict_masking(id_result, im0s)
    image_pack.setImg(masking)

    # 결과 merge
    result = finger_result + ingam_result + id_result

    return image_pack, result, kind, finger_count



