import pandas as pd


# Jumin, Driver, Welfare, Alien -> Id 상속
class Id:
    def __init__(self, nameRect, regnum, issueDate):
        if isinstance(nameRect, str):
            self.name = nameRect
        else:
            self.nameRect = nameRect
        self.regnum = regnum
        self.issueDate = issueDate

    def setName(self, name):
        self.name = name

    def resultPrint(self):
        print("---------------------------------------")
        print('name: ' + self.name)
        print('regum: ' + self.regnum)
        print('issuedate: ' + self.issueDate)


class Jumin(Id):
    def __init__(self, nameRect, regnum, issueDate):
        super().__init__(nameRect, regnum, issueDate)

    def mkDataFrame(self):
        series = pd.Series({"name": self.name, "regnum": self.regnum, "issuedate": self.issueDate})
        return pd.DataFrame(series)


class Driver(Id):
    def __init__(self, nameRect, regnum, issueDate, local, licensenum, encnum):
        super().__init__(nameRect, regnum, issueDate)
        self.local = self.localRename(local)
        self.licensenum = licensenum
        self.encnum = encnum

    def resultPrint(self):
        super().resultPrint()
        if len(self.local):
            print('local: ' + self.local)
        print('licensenum: ' + self.licensenum)
        print('encnum: ' + self.encnum)
        print("---------------------------------------\n")

    def localRename(self, local):
        if local == 'local_busan': return '부산'
        if local == 'local_cb': return '충북'
        if local == 'local_cn': return '충남'
        if local == 'local_daegu': return '대구'
        if local == 'local_daejeon': return '대전'
        if local == 'local_incheon': return '인천'
        if local == 'local_jb': return '전북'
        if local == 'local_jeju': return '제주'
        if local == 'local_jn': return '전남'
        if local == 'local_kangwon': return '강원'
        if local == 'local_kb': return '경북'
        if local == 'local_kn': return '경남'
        if local == 'local_kyounggi': return '경기'
        if local == 'local_seoul': return '서울'
        if local == 'local_ulsan': return '울산'
        return ''

    def mkDataFrame(self):
        series = \
            pd.Series({"name": self.name, "regnum": self.regnum, "issuedate": self.issueDate,
                       "local": self.local, "licensenum": self.licensenum, "encnum": self.encnum})
        return pd.DataFrame(series)


class Welfare(Id):
    def __init__(self, nameRect, regnum, issueDate, gradetype, expire):
        super().__init__(nameRect, regnum, issueDate)
        self.gradetype = gradetype
        self.expire = expire

    def resultPrint(self):
        super().resultPrint()
        print('gradetype: ' + self.gradetype)
        if len(self.expire):
            print('expire: ' + self.expire)
        print("---------------------------------------\n")

    def mkDataFrame(self):
        series = \
            pd.Series({"name": self.name, "regnum": self.regnum, "issuedate": self.issueDate,
                       "gradetype": self.gradetype, "expire": self.expire})
        return pd.DataFrame(series)


class Alien(Id):
    def __init__(self, name, regnum, issueDate, nationality, visatype, sex):
        super().__init__(name, regnum, issueDate)
        self.nationality = nationality
        self.visatype = visatype
        self.sex = sex

    def resultPrint(self):
        super().resultPrint()
        print('nationality: ' + self.nationality)
        print('visatype: ' + self.visatype)
        print('sex: ' + self.sex)
        print("---------------------------------------\n")

    def mkDataFrame(self):
        series = \
            pd.Series({"name": self.name, "regnum": self.regnum, "issuedate": self.issueDate,
                       "nationality": self.nationality, "visatype": self.visatype, "sex": self.sex})
        return pd.DataFrame(series)


class Passport:
    def __init__(self, passportType, issuingCounty, sur, given, passportNo, nationality, birth, sex, expiry, personalNo):
        self.passportType = passportType
        self.issuingCounty = issuingCounty
        self.sur = sur
        self.given = given
        self.passportNo = passportNo
        self.nationality = nationality
        self.birth = birth
        self.sex = sex
        self.expiry = expiry
        self.personalNo = personalNo

    def all(self):
        return (self.passportType, self.issuingCounty, self.sur, self.given, self.passportNo,
                self.nationality, self.birth, self.sex, self.expiry, self.personalNo)

    def resultPrint(self):
        print(f"\n\n----- Passport Scan Result -----")
        print('Type            :', self.passportType)
        print('Issuing county  :', self.issuingCounty)
        print('Passport No.    :', self.passportNo)
        print('Surname         :', self.sur)
        print('Given names     :', self.given)
        print('Nationality     :', self.nationality)
        # print('Personal No.    :', self.personalNo)
        print('Date of birth   :', self.birth)
        print('Sex             :', self.sex)
        print('Date of expiry  :', self.expiry)
        print("---------------------------------------\n")

    def mkDataFrame(self):
        series = \
            pd.Series({"passportType": self.passportType, "issuingCounty": self.issuingCounty, "passportNo": self.passportNo,
                       "sur": self.sur, "given": self.given, "nationality": self.nationality,
                       "birth": self.birth, "sex": self.sex, "expiry": self.expiry})

        return pd.DataFrame(series)
