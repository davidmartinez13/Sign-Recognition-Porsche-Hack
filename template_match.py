import cv2
import numpy as np
from matplotlib import pyplot as plt
class TemplateMatcher:
    def __init__(self):

        self.speed_sign = cv2.imread('images/speedlim.png',0)
        self.stop_sign = cv2.imread('images/stop.png',0)
        self.road_repair_sign = cv2.imread('images/roadwork.png',0)
        self.road_block_sign = cv2.imread('images/roadblock.png',0)
    def temp_select (self, temp_key):
        if temp_key == 'slow':
            template = self.speed_sign
        if temp_key == 'stop':
            template = self.stop_sign
        if temp_key == 'repair':
            template = self.road_repair_sign
        if temp_key == 'block':
            template = self.road_block_sign

        return template

    def contour_detection(self, img ):
        mask = np.zeros_like(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
        
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([165,100,20])
        upper2 = np.array([179,255,255])
        
        lower_mask = cv2.inRange(hsv, lower1, upper1)
        upper_mask = cv2.inRange(hsv, lower2, upper2)
        
        mask = lower_mask + upper_mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x,y,w,h = cv2.boundingRect(mask)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
            # if area > 25000 and area < 110000:
                # cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                rect = cv2.minAreaRect(cnt)
                
                (x, y), (w, h), angle = rect
                # cx , cy = x+xmin, y+ymin
                # centroid = (int(cx), int(cy))
                # box = cv2.boxPoints(((cx,cy),(w, h), angle))
                # box = np.int0(box)
            img = cv2.drawContours(img, contours, -1, (0,255,0), 3,lineType = cv2.LINE_AA)
        return img
        
    def template_conv(self, image, template,threshold = 0.3):
        img = image.copy()
        # img = cv2.resize(image.copy(), (600,600))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= threshold)
        detections = []
        for pt in zip(*loc[::-1]):
            detections.append([pt, (pt[0] + w, pt[1] + h)])
            img = cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (255,0,0), 2)
        return img, detections

    def template_single_conv(self, image, template,threshold = 0.3):

        img = image.copy()
        # img = cv2.resize(image.copy(), (600,600))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        img = self.contour_detection(img)
        img = cv2.rectangle(img,top_left, bottom_right, 255, 2)
        detections = []
        print(bottom_right[1], top_left[1], top_left[0], bottom_right[0])
        return img, detections

    def offline_test(self, img_rgb, temp_key):
        # img_rgb = cv2.imread('images/speedlim_car_view.png')
        # img_rgb = cv2.resize(img_rgb, (600,600))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template = self.temp_select(temp_key)

        # template = cv2.resize(template, (100,200))
        img_detect, detections = self.template_single_conv(img_rgb, template, threshold = 1)
        while True:
            cv2.imshow('winname', img_detect)
            cv2.imshow('temp', template)
            key = cv2.waitKey(1)
            if key == 27 :
                cv2.destroyAllWindows()
                break
        
    def capture_processing(self, temp_key):
        template = self.temp_select(temp_key)
        # template = cv2.resize(template, (200,200))
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,600)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,600)
        while True:
            _, img_rgb = cap.read()
            img_detect, detections = self.template_single_conv(img_rgb, template, threshold= 0.9)
            cv2.imshow('winname', img_detect)
            cv2.imshow('temp', template)
            key = cv2.waitKey(1)
            if key == 27 :
                cv2.destroyAllWindows()
                break
if __name__ == "__main__":

    tm = TemplateMatcher()
    img_rgb = cv2.imread('images/speedlim_car_view.png')
    tm.offline_test(img_rgb, 'slow')
    # tm.capture_processing('slow')