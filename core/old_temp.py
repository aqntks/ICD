from core.util import *


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



 # code1ocr --------------------------------------------------------------------------------------------------
    # if (cla == 'driver') or (cla == 'jumin'):
    #     if result.issueDateRect is None:
    #         return result
    #
    #     # 테스트용
    #     a = crop(result.issueDateRect[0][0], image_pack.getOImg())
    #     img_name = path.split('/')[-1].split('.')[0]
    #     cv2.imwrite(f'crop/issue_{img_name}.jpg', a)
    #
    #     easyT = time.time()
    #     easyList = []
    #     # issueDate 위치 저장
    #     x1, y1, x2, y2 = result.issueDateRect[0][0][0], result.issueDateRect[0][0][1], result.issueDateRect[0][0][2], result.issueDateRect[0][0][3]
    #     easyList.append([int(x1), int(x2), int(y1), int(y2)])
    #
    #     # 암호 일련번호 위치 저장
    #     if cla == 'driver':
    #         if result.encnumRect is None:
    #             return result
    #
    #         # 암호일련번호 위치 강제 크롭 (임시)
    #         # y1, y2 = result.encnumRect[0][0][1], result.encnumRect[0][0][3]
    #         # result.encnumRect[0][0][3] = result.encnumRect[0][0][3] - int((y2 - y1) * 0.3)
    #
    #         x1, y1, x2, y2 = result.encnumRect[0][0][0], result.encnumRect[0][0][1], result.encnumRect[0][0][2], result.encnumRect[0][0][3]
    #         easyList.append([int(x1), int(x2), int(y1), int(y2)])
    #
    #     results = code1ocr.code1ocr(image_pack.getOImg(), easyList)
    #     issue, encnum = code1ocr_issue_encnum(results)
    #
    #     # if encnum != '~':
    #     #     result.encnum = encnum
    #     result.issueDate = issue