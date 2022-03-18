import cv2
import numpy as np

class ExtractObjects:
    def __init__(self, imgpath):
        self.path = imgpath

    def getMask(self, img):
        lower = np.array([0])
        higher = np.array([120])
        mask = cv2.inRange(img, lower, higher)
        return mask

    def getDigits(self):
        img = cv2.imread(self.path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = self.getMask(img)
        conts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cntr_index = np.argsort([cv2.boundingRect(i)[0] for i in conts])
        objects = []
        for cnt in cntr_index:
            x, y, w, h = cv2.boundingRect(conts[cnt])
            if h > 10 and w > 20:
                if h/w < 0.5:
                    h = w//2
                cx = x+w//2
                cy = y+h//2
                cr = max(w, h)//2
                r = cr + 2 * 10
                objects.append(cv2.cvtColor(img[cy-r:cy + r, cx-r:cx+r], cv2.COLOR_GRAY2RGB))

        return objects


# extractor = ExtractObjects("minus.jpg")
# digits = extractor.getDigits()
# print(len(digits))
# i=0
# for digit in digits:
#     cv2.imwrite(f"object{i}.jpg", cv2.resize(digit, (45, 45)))
#     i = i+1
#
