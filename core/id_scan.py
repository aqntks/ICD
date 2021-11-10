import re
import yaml
import ctypes as c
from core.util import *
from core.id_card import *
from core.general import *
from core.correction import *
from core.image_handler import ImagePack
from core.perspective_transform import perspective_transform


# pt 모델 설정 세팅
def model_setting(model, half, imgz):
    if half:
        model.half()
    stride = int(model.stride.max())
    img_size = check_img_size(imgz, s=stride)

    names = model.module.names if hasattr(model, 'module') else model.names

    return model, stride, img_size, names


# pt 검출
def detecting(model, img, im0s, device, img_size, half, option, ciou=20):
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
        detect = det.cpu()

    # 중복 상자 제거
    detList = []
    for *rect, conf, cls in detect:
        detList.append((rect, conf, cls))

    detect = unsorted_remove_intersect_box_det(detList, ciou)

    return detect


# ID 카드 분류
def id_classification(det, names):
    bestConf, bestPlateConf = 0, 0
    bestId, plateArea, bestIdRect = None, None, None
    id_list = ['jumin', 'driver', 'welfare', 'alien', '-', 't_jumin']
    for *rect, conf, cls in det:
        if names[int(cls)] == 'plate':
            if conf > bestPlateConf:
                bestPlateConf = conf
                plateArea = rect
        else:
            if conf > bestConf:
                bestConf = conf
                bestId = cls
                bestIdRect = rect

    if bestId is None:
        return None, plateArea, bestIdRect
    else:
        return id_list[int(bestId)], plateArea, bestIdRect


# 여권 검출 여부 체크
def passport_classification(det, names):
    for *rect, conf, cls in det:
        if names[int(cls)] == 'mrz':
            return 'passport', (rect[0][0][0], rect[0][0][1], rect[0][0][2], rect[0][0][3])
    return None, None


# 주민등록증 검출
def juminScan(det, names):
    name_conf, regnum_conf, issueDate_conf, bracket_conf = 0, 0, 0, 0
    nameRect, regnumRect, issueDateRect, bracketRect = None, None, None, None
    regnum, issueDate = "", ""
    expatriate = False

    for *rect, conf, cls in det:
        # 재외국민 여부
        if names[int(cls)] == 'expatriate':
            expatriate = True
        if names[int(cls)] == 'name':
            if conf > name_conf:
                name_conf = conf
                nameRect = rect
        if names[int(cls)] == 'regnum':
            if conf > regnum_conf:
                regnum_conf = conf
                regnumRect = rect
        if names[int(cls)] == 'issuedate':
            if conf > issueDate_conf:
                issueDate_conf = conf
                issueDateRect = rect

    # 상하좌우 10% 추가
    if regnumRect is not None:
        regnumRectY = regnumRect[0][0][3] - regnumRect[0][0][1]
        regnumRect[0][0][0] = int(regnumRect[0][0][0] - regnumRectY * 0.3)
        regnumRect[0][0][1] = int(regnumRect[0][0][1] - regnumRectY * 0.3)
        regnumRect[0][0][2] = int(regnumRect[0][0][2] + regnumRectY * 0.3)
        regnumRect[0][0][3] = int(regnumRect[0][0][3] + regnumRectY * 0.3)
        regnum = rect_in_value(det, regnumRect, names)
    if issueDateRect is not None:
        issueDateRectY = issueDateRect[0][0][3] - issueDateRect[0][0][1]
        issueDateRect[0][0][0] = int(issueDateRect[0][0][0] - issueDateRectY * 0.1)
        issueDateRect[0][0][1] = int(issueDateRect[0][0][1] - issueDateRectY * 0.2)
        issueDateRect[0][0][2] = int(issueDateRect[0][0][2] + issueDateRectY * 0.1)
        issueDateRect[0][0][3] = int(issueDateRect[0][0][3] + issueDateRectY * 0.3)
        issueDate = rect_in_value(det, issueDateRect, names)

    if nameRect:
        nameRectY = nameRect[0][0][3] - nameRect[0][0][1]
        nameRect[0][0][0] = int(nameRect[0][0][0] - nameRectY * 0.1)
        nameRect[0][0][1] = int(nameRect[0][0][1] - nameRectY * 0.1)
        nameRect[0][0][2] = int(nameRect[0][0][2] + nameRectY * 0.1)
        nameRect[0][0][3] = int(nameRect[0][0][3] + nameRectY * 0.1)
        for *rect, conf, cls in det:
            if rect[0][0][0] > nameRect[0][0][0] and rect[0][0][1] > nameRect[0][0][1] - nameRectY * 0.3 \
                    and rect[0][0][2] < nameRect[0][0][2] and rect[0][0][3] < nameRect[0][0][3] + nameRectY * 0.3:
                if names[int(cls)] == '(':
                    if conf > bracket_conf:
                        bracket_conf = conf
                        bracketRect = rect

    if bracketRect:
        nameRect[0][0][2] = bracketRect[0][0][0]

    return Jumin(nameRect, regnum, issueDate, issueDateRect, expatriate, regnumRect)


# 임시 주민등록증 검출
def temp_juminScan(det, names):
    name_conf, regnum_conf, issue1_conf, issue2_conf, expire_conf, check_yes_conf, check_mid_conf = 0, 0, 0, 0, 0, 0, 0
    nameRect, regnumRect, issue1Rect, issue2Rect, expireRect, check_yesRect, check_midRect = None, None, None, None, None, None, None
    regnum, issue1, issue2, expire, check = "", "", "", "", ""

    for *rect, conf, cls in det:
        if names[int(cls)] == 't_jumin_name':
            if conf > name_conf:
                name_conf = conf
                nameRect = rect
        if names[int(cls)] == 't_jumin_regnum':
            if conf > regnum_conf:
                regnum_conf = conf
                regnumRect = rect
        if names[int(cls)] == 't_jumin_issue1':
            if conf > issue1_conf:
                issue1_conf = conf
                issue1Rect = rect
        if names[int(cls)] == 't_jumin_issue2':
            if conf > issue2_conf:
                issue2_conf = conf
                issue2Rect = rect
        if names[int(cls)] == 't_jumin_expire':
            if conf > expire_conf:
                expire_conf = conf
                expireRect = rect
        if names[int(cls)] == 'check_yes':
            if conf > check_yes_conf:
                check_yes_conf = conf
                check_yesRect = rect
        if names[int(cls)] == 'check_mid':
            if conf > check_mid_conf:
                check_mid_conf = conf
                check_midRect = rect

    if check_yesRect and check_midRect:
        if check_yesRect[0][0][0] < check_midRect[0][0][0]:
            check = '거주자'
        else:
            check = '재외국민'

    # 상하좌우 10% 추가
    if regnumRect is not None:
        regnumRectY = regnumRect[0][0][3] - regnumRect[0][0][1]
        regnumRect[0][0][0] = int(regnumRect[0][0][0] - regnumRectY * 0.1)
        regnumRect[0][0][1] = int(regnumRect[0][0][1] - regnumRectY * 0.1)
        regnumRect[0][0][2] = int(regnumRect[0][0][2] + regnumRectY * 0.1)
        regnumRect[0][0][3] = int(regnumRect[0][0][3] + regnumRectY * 0.1)
        regnum = rect_in_value(det, regnumRect, names)
    if issue1Rect is not None:
        issue1RectY = issue1Rect[0][0][3] - issue1Rect[0][0][1]
        issue1Rect[0][0][0] = int(issue1Rect[0][0][0] - issue1RectY * 0.1)
        issue1Rect[0][0][1] = int(issue1Rect[0][0][1] - issue1RectY * 0.1)
        issue1Rect[0][0][2] = int(issue1Rect[0][0][2] + issue1RectY * 0.1)
        issue1Rect[0][0][3] = int(issue1Rect[0][0][3] + issue1RectY * 0.1)
        issue1 = rect_in_value(det, issue1Rect, names)
    if issue2Rect is not None:
        issue2RectY = issue2Rect[0][0][3] - issue2Rect[0][0][1]
        issue2Rect[0][0][0] = int(issue2Rect[0][0][0] - issue2RectY * 0.1)
        issue2Rect[0][0][1] = int(issue2Rect[0][0][1] - issue2RectY * 0.1)
        issue2Rect[0][0][2] = int(issue2Rect[0][0][2] + issue2RectY * 0.1)
        issue2Rect[0][0][3] = int(issue2Rect[0][0][3] + issue2RectY * 0.1)
        issue2 = rect_in_value(det, issue2Rect, names)
    if expireRect is not None:
        expireRectY = expireRect[0][0][3] - expireRect[0][0][1]
        expireRect[0][0][0] = int(expireRect[0][0][0] - expireRectY * 0.1)
        expireRect[0][0][1] = int(expireRect[0][0][1] - expireRectY * 0.1)
        expireRect[0][0][2] = int(expireRect[0][0][2] + expireRectY * 0.1)
        expireRect[0][0][3] = int(expireRect[0][0][3] + expireRectY * 0.1)
        expire = rect_in_value(det, expireRect, names)

    if nameRect:
        nameRectY = nameRect[0][0][3] - nameRect[0][0][1]
        nameRect[0][0][0] = int(nameRect[0][0][0] - nameRectY * 0.1)
        nameRect[0][0][1] = int(nameRect[0][0][1] - nameRectY * 0.1)
        nameRect[0][0][2] = int(nameRect[0][0][2] + nameRectY * 0.1)
        nameRect[0][0][3] = int(nameRect[0][0][3] + nameRectY * 0.1)

    if len(issue1) == 8:
        issue1 = issue1[0:4] + '.' + issue1[4:6] + '.' + issue1[6:8]

    if len(expire) == 8:
        expire = expire[0:4] + '.' + expire[4:6] + '.' + expire[6:8]

    if len(regnum) == 13:
        regnum = regnum[0:6] + '-' + regnum[6:13]

    return JuminTemp(nameRect, regnum, issue1, issue1Rect, expire, check, regnumRect)


# 운전면허증 검출
def driverScan(det, names):
    name_conf, regnum_conf, issueDate_conf, local_conf, licensenum_conf, encnum_conf = 0, 0, 0, 0, 0, 0
    nameRect, regnumRect, issueDateRect, licensenumRect, encnumRect = None, None, None, None, None
    regnum, issueDate, licensenum, encnum, local = "", "", "", "", ""

    for *rect, conf, cls in det:
        if names[int(cls)] == 'name':
            if conf > name_conf:
                name_conf = conf
                nameRect = rect
        if names[int(cls)] == 'regnum':
            if conf > regnum_conf:
                regnum_conf = conf
                regnumRect = rect
        if names[int(cls)] == 'issuedate':
            if conf > issueDate_conf:
                issueDate_conf = conf
                issueDateRect = rect
        if names[int(cls)] == 'licensenum':
            if conf > licensenum_conf:
                licensenum_conf = conf
                licensenumRect = rect
        if names[int(cls)] == 'encnum':
            if conf > encnum_conf:
                encnum_conf = conf
                encnumRect = rect
        if names[int(cls)].split('_')[0] == 'local':
            if conf > local_conf:
                local_conf = conf
                local = names[int(cls)]

    # 상하좌우 10% 추가
    if regnumRect is not None:
        regnumRectY = regnumRect[0][0][3] - regnumRect[0][0][1]
        regnumRect[0][0][0] = int(regnumRect[0][0][0] - regnumRectY * 0.1)
        regnumRect[0][0][1] = int(regnumRect[0][0][1] - regnumRectY * 0.1)
        regnumRect[0][0][2] = int(regnumRect[0][0][2] + regnumRectY * 0.1)
        regnumRect[0][0][3] = int(regnumRect[0][0][3] + regnumRectY * 0.1)
        regnum = rect_in_value(det, regnumRect, names)
    if issueDateRect is not None:
        issueDateRectY = issueDateRect[0][0][3] - issueDateRect[0][0][1]
        issueDateRect[0][0][0] = int(issueDateRect[0][0][0] - issueDateRectY * 0.1)
        issueDateRect[0][0][1] = int(issueDateRect[0][0][1] - issueDateRectY * 0.1)
        issueDateRect[0][0][2] = int(issueDateRect[0][0][2] + issueDateRectY * 0.1)
        issueDateRect[0][0][3] = int(issueDateRect[0][0][3] + issueDateRectY * 0.1)
        issueDate = rect_in_value(det, issueDateRect, names)
    if licensenumRect is not None:
        licensenumRectY = licensenumRect[0][0][3] - licensenumRect[0][0][1]
        licensenumRect[0][0][0] = int(licensenumRect[0][0][0] - licensenumRectY * 0.1)
        licensenumRect[0][0][1] = int(licensenumRect[0][0][1] - licensenumRectY * 0.1)
        licensenumRect[0][0][2] = int(licensenumRect[0][0][2] + licensenumRectY * 0.1)
        licensenumRect[0][0][3] = int(licensenumRect[0][0][3] + licensenumRectY * 0.1)
        licensenum = rect_in_value(det, licensenumRect, names)
    if encnumRect is not None:
        encnumRectY = encnumRect[0][0][3] - encnumRect[0][0][1]
        encnumRect[0][0][0] = int(encnumRect[0][0][0] - encnumRectY * 0.1)
        encnumRect[0][0][1] = int(encnumRect[0][0][1] - encnumRectY * 0.1)
        encnumRect[0][0][2] = int(encnumRect[0][0][2] + encnumRectY * 0.1)
        encnumRect[0][0][3] = int(encnumRect[0][0][3] + encnumRectY * 0.1)
        encnum = rect_in_value(det, encnumRect, names)

    if nameRect:
        nameRectY = nameRect[0][0][3] - nameRect[0][0][1]
        nameRect[0][0][0] = int(nameRect[0][0][0] - nameRectY * 0.1)
        nameRect[0][0][1] = int(nameRect[0][0][1] - nameRectY * 0.1)
        nameRect[0][0][2] = int(nameRect[0][0][2] + nameRectY * 0.1)
        nameRect[0][0][3] = int(nameRect[0][0][3] + nameRectY * 0.1)

    return Driver(nameRect, regnum, issueDate, local, licensenum, encnum, encnumRect, issueDateRect, regnumRect)


# 복지카드 검출
def welfareScan(det, names):
    name_conf, regnum_conf, issueDate_conf, gradeType_conf, expire_conf = 0, 0, 0, 0, 0
    nameRect, regnumRect, issueDateRect, gradeTypeRect, expireRect = None, None, None, None, None
    regnum, issueDate, gradeType, expire = "", "", "", ""

    for *rect, conf, cls in det:
        if names[int(cls)] == 'name':
            if conf > name_conf:
                name_conf = conf
                nameRect = rect
        if names[int(cls)] == 'regnum':
            if conf > regnum_conf:
                regnum_conf = conf
                regnumRect = rect
        if names[int(cls)] == 'issuedate':
            if conf > issueDate_conf:
                issueDate_conf = conf
                issueDateRect = rect
        if names[int(cls)] == 'gradetype':
            if conf > gradeType_conf:
                gradeType_conf = conf
                gradeTypeRect = rect
        if names[int(cls)] == 'expire':
            if conf > expire_conf:
                expire_conf = conf
                expireRect = rect

    if regnumRect is not None:
        regnum = rect_in_value(det, regnumRect, names)
    if issueDateRect is not None:
        issueDate = rect_in_value(det, issueDateRect, names)
    if gradeTypeRect is not None:
        gradeType = rect_in_value(det, gradeTypeRect, names)
    if expireRect is not None:
        expire = rect_in_value(det, expireRect, names)

    return Welfare(nameRect, regnum, issueDate, gradeType, expire)


# 외국인등록증 검출
def alienScan(det, names):
    name_conf, regnum_conf, issueDate_conf, nationality_conf, visaType_conf, sex_conf = 0, 0, 0, 0, 0, 0
    nameRect, regnumRect, issueDateRect, nationalityRect, visaTypeRect, sexRect = None, None, None, None, None, None
    name, regnum, issueDate, nationality, visaType, sex = "", "", "", "", "", ""

    for *rect, conf, cls in det:
        if names[int(cls)] == 'name':
            if conf > name_conf:
                name_conf = conf
                nameRect = rect
        if names[int(cls)] == 'regnum':
            if conf > regnum_conf:
                regnum_conf = conf
                regnumRect = rect
        if names[int(cls)] == 'issuedate':
            if conf > issueDate_conf:
                issueDate_conf = conf
                issueDateRect = rect
        if names[int(cls)] == 'nationality':
            if conf > nationality_conf:
                nationality_conf = conf
                nationalityRect = rect
        if names[int(cls)] == 'visatype':
            if conf > visaType_conf:
                visaType_conf = conf
                visaTypeRect = rect
        if names[int(cls)] == 'sex':
            if conf > sex_conf:
                sex_conf = conf
                sexRect = rect

    if nameRect is not None:
        name = rect_in_value(det, nameRect, names)
    if regnumRect is not None:
        regnumRectX = regnumRect[0][0][2] - regnumRect[0][0][0]
        regnumRectY = regnumRect[0][0][3] - regnumRect[0][0][1]
        regnumRect[0][0][0] = int(regnumRect[0][0][0] + regnumRectX * 0.26)
        regnumRect[0][0][1] = int(regnumRect[0][0][1] - regnumRectY * 0.1)
        regnumRect[0][0][3] = int(regnumRect[0][0][3] - regnumRectY * 0.1)
        regnum = rect_in_value(det, regnumRect, names)
    if issueDateRect is not None:
        issueDateRectX = issueDateRect[0][0][2] - issueDateRect[0][0][0]
        issueDateRect[0][0][0] = int(issueDateRect[0][0][0] + issueDateRectX * 0.34)
        issueDate = rect_in_value(det, issueDateRect, names)
    if nationalityRect is not None:
        # nationalityRectX = nationalityRect[0][0][2] - nationalityRect[0][0][0]
        # nationalityRect[0][0][0] = int(nationalityRect[0][0][0] + nationalityRectX * 0.30)
        nationality = rect_in_value(det, nationalityRect, names)
    if visaTypeRect is not None:
        # visaTypeRectX = visaTypeRect[0][0][2] - visaTypeRect[0][0][0]
        # visaTypeRect[0][0][0] = int(visaTypeRect[0][0][0] + visaTypeRectX * 0.34)
        visaType = rect_in_value(det, visaTypeRect, names)
    if sexRect is not None:
        sex = rect_in_value(det, sexRect, names)

    return Alien(name, regnum, issueDate, nationality, visaType, sex, issueDateRect, regnumRect, nationalityRect, visaTypeRect)


# 여권 검출
def passportScan(det, names):
    # 검출 값 처리
    # mrz 검출
    mrz_rect = None
    for *rect, conf, cls in det:
        if names[int(cls)] == 'mrz':
            mrz_rect = rect
            break

    # mrz 정렬
    result, mrzStr, = '', []
    for *rect, conf, cls in det:
        if (rect[0][0][0] > mrz_rect[0][0][0]) and (rect[0][0][1] > mrz_rect[0][0][1]) and (rect[0][0][2] < mrz_rect[0][0][2]) and (
                rect[0][0][3] < mrz_rect[0][0][3]):
            cls_name = names[int(cls)] if names[int(cls)] != 'sign' else '<'
            mrzStr.append((rect, cls_name, conf))

    mrzStr.sort(key=lambda x: x[0][0][0][1])

    # 라인단위 정렬 v2
    # mrzFirst, mrzSecond = sort_v2(mrzStr)

    # 라인 단위 정렬
    mrzFirst, mrzSecond = line_by_line_sort(mrzStr)

    # 한번에 정렬
    # mrzFirst, mrzSecond = all_sort(mrzStr)

    # 중복 상자 제거
    # mrzFirst, mrzSecond = remove_intersect_box(mrzFirst), remove_intersect_box(mrzSecond)

    # 결과 저장
    firstLine, secondLine = "", ""
    for rect, mrz_cls, conf in mrzFirst:
        firstLine += mrz_cls
    for rect, mrz_cls, conf in mrzSecond:
        secondLine += mrz_cls

    if len(firstLine) < 44:
        for i in range(len(firstLine), 44):
            firstLine += '<'

    if len(secondLine) < 44:
        for i in range(len(secondLine), 44):
            secondLine += '<'

    surName, givenNames = spiltName(firstLine[5:44])
    passportType = typeCorrection(mrzCorrection(firstLine[0:2].replace('<', ''), 'dg2en'))
    issuingCounty = nationCorrection(mrzCorrection(firstLine[2:5], 'dg2en'))
    if issuingCounty == 'D<<':
        issuingCounty = 'DEU'
    sur = mrzCorrection(surName.replace('<', ' ').strip(), 'dg2en')
    given = mrzCorrection(givenNames.replace('<', ' ').strip(), 'dg2en')

    passportNo = secondLine[0:9].replace('<', '')
    nationality = nationCorrection(mrzCorrection(secondLine[10:13], 'dg2en'))
    if nationality == 'D<<':
        nationality = 'DEU'
    birth = mrzCorrection(secondLine[13:19].replace('<', ''), 'en2dg')
    sex = sexCorrection(mrzCorrection(secondLine[20].replace('<', ''), 'dg2en'))
    expiry = mrzCorrection(secondLine[21:27].replace('<', ''), 'en2dg')
    personalNo = mrzCorrection(secondLine[28:35].replace('<', ''), 'en2dg')

    return Passport(passportType, issuingCounty, sur, given, passportNo, nationality, birth, sex, expiry,
                        personalNo)


# 한글 검출
def hangulScan(det, names, y, temp_jumin):
    obj, name = [], ''
    for *rect, conf, cls in det:
        if rect[0][0][1] < y/2 < rect[0][0][3]:
            obj.append((rect, conf, names[int(cls)]))
            obj.sort(key=lambda x: x[0][0])
    for s, conf, cls in obj:
        name += cls

    if temp_jumin:
        if len(name) > 3 and '한글' in name:
            name = name.replace('한글', '')
    else:
        if len(name) > 3 and '성명' in name:
            name = name.replace('성명', '')

    return name


# 암호 일련번호 검출
def encnumScan(det, names):
    obj, name = [], ''
    for *rect, conf, cls in det:
        obj.append((rect, conf, names[int(cls)]))
        obj.sort(key=lambda x: x[0][0])
    for s, conf, cls in obj:
        name += cls

    return name


# issue 검출
def code1_issue(value):
    results = value[0][1].replace(' ', '').replace('..', '.').replace('...', '.')

    result_list = results.split('.')

    if len(result_list[0]) > 4 and len(result_list) > 2:
        result_list[0] = result_list[0][len(result_list[0]) - 4:len(result_list[0])]
        results = result_list[0] + '.' + result_list[1] + '.' + result_list[2] + '.'
    if len(result_list) == 4:
        results = result_list[0] + '.' + result_list[1] + '.' + result_list[2] + '.'

    return results


def code1_regnum(value):
    results = value[0][1].replace(' ', '').replace('.', '').replace('--', '-').replace('---', '-')

    split = results.split('-')

    if len(split) == 2:
        if len(split[0]) > 6:
            split[0] = split[0][len(split[0]) - 6:len(split[0])]
        if len(split[1]) > 7:
            split[1] = split[1][len(split[1]) - 7:len(split[1])]

        results = f'{split[0]}-{split[1]}'

    return results


def code1_nationality(value):
    value = value[0][1]
    results = ''
    en_count = 0

    for c in value:
        if c in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y', 'Z']:
            en_count += 1

    if en_count == 0 or value[0:2] == '국적':
        results = value[2:]
    else:
        for c in value:
            if c in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
                results += c

    return results


def code1_visatype(value):
    results = value[0][1]

    if len(results) > 4:
        results = results[4:]

    return results


def pt_detect(path, device, models, ciou, code1ocr_dg, code1ocr_en_ko, gray=False, byteMode=False, perspect=False):
    id_cls_weights, jumin_weights, driver_weights, passport_weights, welfare_weights, alien_weights, hangul_weights, encnum_weights = models

    half = device.type != 'cpu'

    # config 로드 --------------------------------------------------------------------------------------------------
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    img_size, confidence, iou = config['cls-img_size'], config['cls-confidence'], config['cls-iou']
    id_cls_option = (img_size, confidence, iou)
    img_size, confidence, iou = config['jumin-img_size'], config['jumin-confidence'], config['jumin-iou']
    jumin_option = (img_size, confidence, iou)
    img_size, confidence, iou = config['driver-img_size'], config['driver-confidence'], config['driver-iou']
    driver_option = (img_size, confidence, iou)
    img_size, confidence, iou = config['passport-img_size'], config['passport-confidence'], config['passport-iou']
    passport_option = (img_size, confidence, iou)
    img_size, confidence, iou = config['welfare-img_size'], config['welfare-confidence'], config['welfare-iou']
    welfare_option = (img_size, confidence, iou)
    img_size, confidence, iou = config['alien-img_size'], config['alien-confidence'], config['alien-iou']
    alien_option = (img_size, confidence, iou)
    img_size, confidence, iou = config['hangul-img_size'], config['hangul-confidence'], config['hangul-iou']
    hangul_option = (img_size, confidence, iou)
    img_size, confidence, iou = config['encnum-img_size'], config['encnum-confidence'], config['encnum-iou']
    encnum_option = (img_size, confidence, iou)
    f.close()

    # 분류 --------------------------------------------------------------------------------------------------------
    c1 = time.time()
    model, stride, img_size, names = model_setting(id_cls_weights, half, id_cls_option[0])
    image_pack = ImagePack(path, img_size, stride, byteMode=byteMode, gray=gray, perspect=perspect)
    img, im0s = image_pack.getImg()
    det = detecting(model, img, im0s, device, img_size, half, id_cls_option[1:], ciou)
    cla, plateArea, idRect = id_classification(det, names)
    if cla is None:
        model, stride, img_size, names = model_setting(passport_weights, half, passport_option[0])
        image_pack.reset(img_size, stride)
        img, im0s = image_pack.getImg()
        det = detecting(model, img, im0s, device, img_size, half, passport_option[1:], ciou)
        cla, mrzRect = passport_classification(det, names)
    if cla is None:
        model, stride, img_size, names = model_setting(driver_weights, half, driver_option[0])
        image_pack.reset(img_size, stride)
        img, im0s = image_pack.getImg()
        det = detecting(model, img, im0s, device, img_size, half, driver_option[1:], ciou)
        regnumB, nameB, issueB = False, False, False
        for *rect, conf, cls in det:
            if (names[int(cls)] == 'encnum' or names[int(cls)] == 'period') and conf > 0.9:
                cla = 'driver'
                break
            if names[int(cls)] == 'regnum':
                regnumB = True
            if names[int(cls)] == 'name':
                nameB = True
            if names[int(cls)] == 'issuedate':
                issueB = True
        if regnumB and nameB and issueB:
            cla = 'jumin'

    if cla is None:
        return None
    print('분류', time.time() - c1)

    # plate crop --------------------------------------------------------------------------------------------------
    # if plateArea is not None:
    #     _, im0s = image_pack.setCrop(plateArea)

    # perspective transform ---------------------------------------------------------------------------------------
    perspect_on = False
    p1 = time.time()
    perspect_img, point_rect = perspective_transform(im0s)
    print('perspective', time.time() - p1)

    if point_rect is not False:
        pMinX, pMinY, pMaxX, pMaxY = im0s.shape[1], im0s.shape[0], 0, 0
        for point in point_rect:
            if pMinX > point[0]:
                pMinX = point[0]
            if pMaxX < point[0]:
                pMaxX = point[0]
            if pMinY > point[1]:
                pMinY = point[1]
            if pMaxY < point[1]:
                pMaxY = point[1]

        pointCropImage = crop((pMinX, pMinY, pMaxX, pMaxY), im0s)
        cv2.imshow("pointCropImage", pointCropImage)
        cv2.waitKey(0)

    if perspect_on:
        image_pack.o_img = perspect_img
        image_pack.setImg(perspect_img)
        _, im0s = image_pack.getImg()
    # img_name = path.split('/')[-1].split('.')[0]
    # cv2.imwrite(f'data/{img_name}.jpg', im0s)

    # 신분증 검출 ----------------------------------------------------------------------------------------------------
    i1 = time.time()
    if cla == 'jumin':
        model, stride, img_size, names = model_setting(jumin_weights, half, jumin_option[0])
        image_pack.reset(img_size, stride)
        img, im0s = image_pack.getImg()
        det = detecting(model, img, im0s, device, img_size, half, jumin_option[1:], ciou)
        result = juminScan(det, names)
    elif cla == 't_jumin':
        model, stride, img_size, names = model_setting(jumin_weights, half, jumin_option[0])
        image_pack.reset(img_size, stride)
        img, im0s = image_pack.getImg()
        det = detecting(model, img, im0s, device, img_size, half, jumin_option[1:], ciou)
        result = temp_juminScan(det, names)
    elif cla == 'driver':
        model, stride, img_size, names = model_setting(driver_weights, half, driver_option[0])
        image_pack.reset(img_size, stride)
        img, im0s = image_pack.getImg()
        det = detecting(model, img, im0s, device, img_size, half, driver_option[1:], ciou)
        result = driverScan(det, names)
    elif cla == 'welfare':
        model, stride, img_size, names = model_setting(welfare_weights, half, welfare_option[0])
        image_pack.reset(img_size, stride)
        img, im0s = image_pack.getImg()
        det = detecting(model, img, im0s, device, img_size, half, welfare_option[1:], ciou)
        result = welfareScan(det, names)
    elif cla == 'alien':
        model, stride, img_size, names = model_setting(alien_weights, half, alien_option[0])
        image_pack.reset(img_size, stride)
        img, im0s = image_pack.getImg()
        det = detecting(model, img, im0s, device, img_size, half, alien_option[1:], ciou)
        result = alienScan(det, names)
    elif cla == 'passport':
        model, stride, img_size, names = model_setting(passport_weights, half, passport_option[0])
        image_pack.reset(img_size, stride)
        img, im0s = image_pack.passportCrop(mrzRect)
        det = detecting(model, img, im0s, device, img_size, half, passport_option[1:], ciou)
        result = passportScan(det, names)
    else:
        result = None
    print('신분증', time.time() - i1)

    # 이름 검출 ----------------------------------------------------------------------------------------------------
    h1 = time.time()
    if (cla == 'jumin' or cla == 'driver' or cla == 'welfare' or cla == 't_jumin') and result is not None:
        temp_jumin = True if cla == 't_jumin' else False
        
        if result.nameRect is None:
            result.setName('')
        else:
            model, stride, img_size, names = model_setting(hangul_weights, half, hangul_option[0])
            image_pack.reset(img_size, stride)

            _, im0s = image_pack.setSizeCrop(result.nameRect, 480)
            _, _ = image_pack.resize_ratio(im0s, 640)
            # _, im0s = image_pack.setCrop(result.nameRect)
            image_pack.setGray()
            img, im0s = image_pack.getImg()

            img_name = path.split('/')[-1].split('.')[0]
            cv2.imwrite(f'crop/name_{img_name}.jpg', im0s)

            det = detecting(model, img, im0s, device, img_size, half, hangul_option[1:], ciou)
            name = hangulScan(det, names, im0s.shape[0], temp_jumin)
            result.setName(name)
    print('이름검출', time.time() - h1)

    # 암호 일련번호 검출 --------------------------------------------------------------------------------------------
    enc1 = time.time()
    if (cla == 'driver') and result is not None:
        if result.encnumRect is None:
            result.setEncnum('')
        else:
            model, stride, img_size, names = model_setting(encnum_weights, half, encnum_option[0])
            oImg = image_pack.getOImg()
            image_pack.setImg(oImg)
            image_pack.reset(img_size, stride)

            y1, y2 = result.encnumRect[0][0][1], result.encnumRect[0][0][3]
            result.encnumRect[0][0][3] = result.encnumRect[0][0][3] - int((y2 - y1) * 0.3)
            _, im0s = image_pack.setCrop(result.encnumRect)
            # img, im0s = image_pack.setSizeCrop(result.encnumRect, 480)

            # _, _ = image_pack.resize_ratio(im0s, 640)
            image_pack.setGray()
            img, im0s = image_pack.getImg()

            img_name = path.split('/')[-1].split('.')[0]
            cv2.imwrite(f'crop/enc_{img_name}.jpg', im0s)
            # cv2.imshow('46436', im0s)
            # cv2.waitKey(0)

            det = detecting(model, img, im0s, device, img_size, half, encnum_option[1:], ciou)
            encnum = encnumScan(det, names)
            result.setEncnum(encnum)
    print('암호검출', time.time() - enc1)

    # code1ocr - issue date--------------------------------------------------------------------------------------
    if (cla == 'driver') or (cla == 'jumin') or (cla == 'alien'):
        if result.issueDateRect is None:
            return result

        easyList = []

        issue_crop_img = crop(result.issueDateRect[0][0], image_pack.getOImg())
        easyList.append([0, int(issue_crop_img.shape[1]), 0, int(issue_crop_img.shape[0])])

        results = code1ocr_dg.recogss(issue_crop_img, easyList)
        issue_value = code1_issue(results)

        result.issueDate = issue_value

        img_name = path.split('/')[-1].split('.')[0]
        cv2.imwrite(f'crop/issue_{img_name}.jpg', issue_crop_img)

    # code1ocr - regnum --------------------------------------------------------------------------------------
    if cla == 'alien':
        if result.regnumRect is None:
            return result

        easyList = []

        regnum_crop_img = crop(result.regnumRect[0][0], image_pack.getOImg())
        # regnum_crop_img = remove_background(regnum_crop_img)
        easyList.append([0, int(regnum_crop_img.shape[1]), 0, int(regnum_crop_img.shape[0])])

        results = code1ocr_dg.recogss(regnum_crop_img, easyList)
        regnum_value = code1_regnum(results)

        result.regnum = regnum_value

        img_name = path.split('/')[-1].split('.')[0]
        cv2.imwrite(f'crop/regnum_{img_name}.jpg', regnum_crop_img)

    # code1ocr - nationality --------------------------------------------------------------------------------------
    if cla == 'alien':
        if result.nationalityRect is None:
            return result

        easyList = []

        nationality_crop_img = crop(result.nationalityRect[0][0], image_pack.getOImg())
        easyList.append([0, int(nationality_crop_img.shape[1]), 0, int(nationality_crop_img.shape[0])])

        results = code1ocr_en_ko.recogss(nationality_crop_img, easyList)
        nationality_value = code1_nationality(results)

        result.nationality = nationality_value

        img_name = path.split('/')[-1].split('.')[0]
        cv2.imwrite(f'crop/nationality_{img_name}.jpg', nationality_crop_img)

    # code1ocr - visatype --------------------------------------------------------------------------------------
    if cla == 'alien':
        if result.visatypeRect is None:
            return result

        easyList = []

        visatype_crop_img = crop(result.visatypeRect[0][0], image_pack.getOImg())
        easyList.append([0, int(visatype_crop_img.shape[1]), 0, int(visatype_crop_img.shape[0])])

        results = code1ocr_en_ko.recogss(visatype_crop_img, easyList)
        visatype_value = code1_visatype(results)

        result.visatype = visatype_value

        img_name = path.split('/')[-1].split('.')[0]
        cv2.imwrite(f'crop/visatype_{img_name}.jpg', visatype_crop_img)

    return result



