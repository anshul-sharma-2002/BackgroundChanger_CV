import cv2
import cvzone
from cvzone. SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

segmentor = SelfiSegmentation()

fpsReader = cvzone.FPS()

listImg = os.listdir("Img")
print(listImg)
imgList = []
imgIndex = 0

for imgPath in listImg:
    img = cv2.imread(f"Img/{imgPath}")
    imgList.append(img)


while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgList[imgIndex], threshold=0.6)

    imgStack = cvzone.stackImages([img, imgOut], 2, 1)
    fps, imgStack = fpsReader.update(imgStack)
    cv2.imshow("Live Capture", imgStack)
    # cv2.imshow("Background Changer", imgOut)
    key = cv2.waitKey(1)
    if key == ord("a"):
        if imgIndex > 0:
            imgIndex -= 1
    elif key == ord("d"):
        if imgIndex < len(imgList) - 1:
            imgIndex += 1
    elif key == ord("q"):
        break
