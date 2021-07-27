
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