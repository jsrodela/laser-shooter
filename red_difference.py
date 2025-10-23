import cv2
import numpy as np
import os


update = lambda name, default: (
    lambda x: int(x) if x.isdigit() else default
)(input(f"{name} (default: {default}): "))
    
WHITE_THRESHOLD = update("White threshhold(0-255)", 160)

vid = 0

print("Select Video input\n\t(0) Webcam (1) Videos\n")
ans = update("Video input", 0)

if ans == 0:
    vid = update("Video Input id", 0)
if ans == 1:
    videos = os.listdir("videos")
    for idx, file in enumerate(videos):
        print(f'\t({idx}) {file}')
    print()
    vid = os.path.join('videos', videos[int(input("Video id: "))])

print(vid)
cap = cv2.VideoCapture(vid)

colored = False

while True:
    # 1. 이미지 로드
    ret, img = cap.read()
    if not ret:
        break

    hh, ww = img.shape[:2]

    # 2. BGR → YCrCb 변환 (YCbCr과 동일)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # 3. 채널 분리
    Y, Cr, Cb = cv2.split(ycrcb)

    # 4. 색상 시각화를 위한 중간 회색 채널 생성
    gray = np.full((hh, ww), 128, dtype=np.uint8)

    # 역색상(invert)을 이용해 Cb/Cr 대비를 강화
    Cr_inv = cv2.bitwise_not(Cr)
    Cb_inv = cv2.bitwise_not(Cb)

    # 5. 색 강조형 출력: 중간 회색 + 해당 채널
    # Cr → 붉은색 계열, Cb → 푸른색 계열, Y → 밝기 흑백
    Cr_colored = cv2.merge([gray, Cr_inv, Cr])
    Cb_colored = cv2.merge([Cb, Cb_inv, gray])
    Y_colored = cv2.merge([Y, Y, Y])

    # cv2.imshow('Y (Luma)', Y_colored)
    # cv2.imshow('Cb (Chroma Blue)', Cb_colored)

    target = Cr_colored if colored else Cr
    # adjust = cv2.convertScaleAbs(target, alpha=1.5, beta=-1)
    # adjust = cv2.equalizeHist(target)
    _, adjust = cv2.threshold(target, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
    cv2.imshow("Cr (Chroma Red)", adjust)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        colored ^= 1

cap.release()
cv2.destroyAllWindows()
