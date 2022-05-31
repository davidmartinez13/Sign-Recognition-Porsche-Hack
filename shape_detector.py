import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
while True:
    _, img = cap.read()
    original = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])

    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])

    lower_mask = cv2.inRange(hsv, lower1, upper1)
    upper_mask = cv2.inRange(hsv, lower2, upper2)

    # mask = lower_mask
    mask = lower_mask + upper_mask
    rows, cols = mask.shape
    mask[:,int(cols/2):] = 0
    ROI_number = 0

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for cnt in cnts:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

        if len(approx)==3:
            print("Triangle")
            cv2.drawContours(img,[cnt],0,(0,255,0),-1)
        elif len(approx) == 6:
            print("Stop")
            cv2.drawContours(img,[cnt],0,(0,255,255),-1)
        elif len(approx) > 12:
            print("Slow Down")
            cv2.drawContours(img,[cnt],0,(0,255,255),-1)

    cv2.imshow('image', img)
    cv2.imshow('Binary',mask)    

    key = cv2.waitKey(1)
    if key == 27 :
        cv2.destroyAllWindows()
        break
