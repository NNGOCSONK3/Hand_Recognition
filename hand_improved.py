import cv2
import mediapipe as mp
import time
import math

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20] # ID đầu các ngón tay

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        self.bbox = []
        
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                
                # Lấy thông tin Tay Trái hay Phải
                if self.results.multi_handedness:
                    self.label = self.results.multi_handedness[handNo].classification[0].label
                else:
                    self.label = "Unknown"

                h, w, c = img.shape
                xList = []
                yList = []
                
                for id, lm in enumerate(myHand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                self.bbox = xmin, ymin, xmax, ymax

                if draw:
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                                  (0, 255, 0), 2)
        
        return self.lmList, self.bbox

    def fingersUp(self):
        fingers = []
        if len(self.lmList) == 0:
            return []

        # Logic Ngón Cái (Thumb) - Đã fix ngược hướng
        is_thumb_open = False
        if self.label == "Right": # Tay phải trên màn hình
             if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
                 is_thumb_open = True
        else: 
             if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                 is_thumb_open = True
        
        fingers.append(1 if is_thumb_open else 0)

        # 4 Ngón còn lại
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers