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

    if nameRect:
        nameRectY = nameRect[0][0][3] - nameRect[0][0][1]
        nameRect[0][0][0] = int(nameRect[0][0][0] - nameRectY * 0.1)
        nameRect[0][0][1] = int(nameRect[0][0][1] - nameRectY * 0.1)
        nameRect[0][0][2] = int(nameRect[0][0][2] + nameRectY * 0.1)
        nameRect[0][0][3] = int(nameRect[0][0][3] + nameRectY * 0.1)
        for *rect, conf, cls in det:
            if rect[0][0][0] > nameRect[0][0][0] and rect[0][0][1] > nameRect[0][0][1] \
                    and rect[0][0][2] < nameRect[0][0][2] and rect[0][0][3] < nameRect[0][0][3]:
                if names[int(cls)] == '(':
                    if conf > bracket_conf:
                        bracket_conf = conf
                        bracketRect = rect

    if bracketRect:
        nameRect[0][0][2] = bracketRect[0][0][0]

    return Jumin(nameRect, regnum, issueDate, issueDateRect, expatriate)


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

    return JuminTemp(nameRect, regnum, issue1, issue1Rect, expire, check)


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

    return Driver(nameRect, regnum, issueDate, local, licensenum, encnum, encnumRect, issueDateRect)


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
        regnum = rect_in_value(det, regnumRect, names)
    if issueDateRect is not None:
        issueDate = rect_in_value(det, issueDateRect, names)
    if nationalityRect is not None:
        nationality = rect_in_value(det, nationalityRect, names)
    if visaTypeRect is not None:
        visaType = rect_in_value(det, visaTypeRect, names)
    if sexRect is not None:
        sex = rect_in_value(det, sexRect, names)

    return Alien(name, regnum, issueDate, nationality, visaType, sex)


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
    sur = mrzCorrection(surName.replace('<', ' ').strip(), 'dg2en')
    given = mrzCorrection(givenNames.replace('<', ' ').strip(), 'dg2en')

    passportNo = secondLine[0:9].replace('<', '')
    nationality = nationCorrection(mrzCorrection(secondLine[10:13], 'dg2en'))
    birth = mrzCorrection(secondLine[13:19].replace('<', ''), 'en2dg')
    sex = sexCorrection(mrzCorrection(secondLine[20].replace('<', ''), 'dg2en'))
    expiry = mrzCorrection(secondLine[21:27].replace('<', ''), 'en2dg')
    personalNo = mrzCorrection(secondLine[28:35].replace('<', ''), 'en2dg')

    return Passport(passportType, issuingCounty, sur, given, passportNo, nationality, birth, sex, expiry,
                        personalNo)


# 한글 검출
def hangulScan(det, names):
    obj, name = [], ''
    for *rect, conf, cls in det:
        obj.append((rect, conf, names[int(cls)]))
        obj.sort(key=lambda x: x[0][0])
    for s, conf, cls in obj:
        name += cls

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


# code1ocr issueDate & encnum 검출
def code1ocr_issue_encnum(results):
    result_line = []

    for r in results:
        line = r[1].replace(' ', '')
        result_line.append(line)

    if len(result_line) == 2:
        issue = result_line[0]
        encnum = result_line[1]
    else:
        encnum = '~'
        issue = result_line[0]

    if encnum != '-':
        en_dg_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        encnum = encnum.replace('-', '').replace('.', '').replace('(', '')
        for e in encnum:
            if (e in en_dg_list) is False:
                encnum = encnum.replace(e, '')

    issue = issue.replace('-', '').replace('(', '').replace('L', '1').replace('O', '0').replace('Q', '0') \
        .replace('U', '0').replace('D', '0').replace('I', '1').replace('Z', '2').replace('B', '3') \
        .replace('A', '4').replace('S', '5').replace('T', '1')

    return issue, encnum


def pt_detect(path, device, models, ciou, code1ocr, gray=False, byteMode=False, perspect=False):
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
    det = detecting(model, img, im0s, device, img_size, half, id_cls_option[1:])
    cla, plateArea, idRect = id_classification(det, names)
    if cla is None:
        model, stride, img_size, names = model_setting(passport_weights, half, passport_option[0])
        image_pack.reset(img_size, stride)
        img, im0s = image_pack.getImg()
        det = detecting(model, img, im0s, device, img_size, half, passport_option[1:])
        cla, mrzRect = passport_classification(det, names)
    if cla is None:
        model, stride, img_size, names = model_setting(driver_weights, half, driver_option[0])
        image_pack.reset(img_size, stride)
        img, im0s = image_pack.getImg()
        det = detecting(model, img, im0s, device, img_size, half, driver_option[1:])
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
    # p1 = time.time()
    # perspect_img = perspective_transform(im0s)
    # print('perspective', time.time() - p1)
    # image_pack.o_img = perspect_img
    # image_pack.setImg(perspect_img)
    #
    # _, im0s = image_pack.getImg()
    # img_name = path.split('/')[-1].split('.')[0]
    # cv2.imwrite(f'data/{img_name}.jpg', im0s)

    # 신분증 검출 ----------------------------------------------------------------------------------------------------
    i1 = time.time()
    if cla == 'jumin':
        model, stride, img_size, names = model_setting(jumin_weights, half, jumin_option[0])
        image_pack.reset(img_size, stride)
        img, im0s = image_pack.getImg()
        det = detecting(model, img, im0s, device, img_size, half, jumin_option[1:])
        result = juminScan(det, names)
    elif cla == 't_jumin':
        model, stride, img_size, names = model_setting(jumin_weights, half, jumin_option[0])
        image_pack.reset(img_size, stride)
        img, im0s = image_pack.getImg()
        det = detecting(model, img, im0s, device, img_size, half, jumin_option[1:])
        result = temp_juminScan(det, names)
    elif cla == 'driver':
        model, stride, img_size, names = model_setting(driver_weights, half, driver_option[0])
        image_pack.reset(img_size, stride)
        img, im0s = image_pack.getImg()
        det = detecting(model, img, im0s, device, img_size, half, driver_option[1:])
        result = driverScan(det, names)
    elif cla == 'welfare':
        model, stride, img_size, names = model_setting(welfare_weights, half, welfare_option[0])
        image_pack.reset(img_size, stride)
        img, im0s = image_pack.getImg()
        det = detecting(model, img, im0s, device, img_size, half, welfare_option[1:])
        result = welfareScan(det, names)
    elif cla == 'alien':
        model, stride, img_size, names = model_setting(alien_weights, half, alien_option[0])
        image_pack.reset(img_size, stride)
        img, im0s = image_pack.getImg()
        det = detecting(model, img, im0s, device, img_size, half, alien_option[1:])
        result = alienScan(det, names)
    elif cla == 'passport':
        model, stride, img_size, names = model_setting(passport_weights, half, passport_option[0])
        image_pack.reset(img_size, stride)
        img, im0s = image_pack.passportCrop(mrzRect)
        det = detecting(model, img, im0s, device, img_size, half, passport_option[1:])
        result = passportScan(det, names)
    else:
        result = None
    print('신분증', time.time() - i1)

    # 이름 검출 ----------------------------------------------------------------------------------------------------
    h1 = time.time()
    if (cla == 'jumin' or cla == 'driver' or cla == 'welfare' or cla == 't_jumin') and result is not None:
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
            name = hangulScan(det, names)
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

    # code1ocr --------------------------------------------------------------------------------------------------
    if result.issueDateRect is None:
        return result

    # 테스트용
    a = crop(result.issueDateRect[0][0], image_pack.getOImg())
    img_name = path.split('/')[-1].split('.')[0]
    cv2.imwrite(f'crop/issue_{img_name}.jpg', a)
    
    easyT = time.time()
    easyList = []
    # issueDate 위치 저장
    x1, y1, x2, y2 = result.issueDateRect[0][0][0], result.issueDateRect[0][0][1], result.issueDateRect[0][0][2], result.issueDateRect[0][0][3]
    easyList.append([int(x1), int(x2), int(y1), int(y2)])

    # 암호 일련번호 위치 저장
    if cla == 'driver':
        if result.encnumRect is None:
            return result
        
        # 암호일련번호 위치 강제 크롭 (임시)
        # y1, y2 = result.encnumRect[0][0][1], result.encnumRect[0][0][3]
        # result.encnumRect[0][0][3] = result.encnumRect[0][0][3] - int((y2 - y1) * 0.3)

        x1, y1, x2, y2 = result.encnumRect[0][0][0], result.encnumRect[0][0][1], result.encnumRect[0][0][2], result.encnumRect[0][0][3]
        easyList.append([int(x1), int(x2), int(y1), int(y2)])

    results = code1ocr.code1ocr(image_pack.getOImg(), easyList)
    issue, encnum = code1ocr_issue_encnum(results)

    # if encnum != '~':
    #     result.encnum = encnum
    result.issueDate = issue
    print("code1ocr", time.time() - easyT)

    return result



