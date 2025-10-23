import cv2
import numpy as np
from typing import Callable
import os
import time

update = lambda name, default: (
    lambda x: int(x) if x.isdigit() else default
    )(input(f"{name} (default: {default}): "))

# --------------------------
# 설정: 아루코 마커, 과녁 크기, 점수링
# --------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

print("> Aruco Detector initialized.")

# A4 과녁 크기 (픽셀 기준)
TARGET_WIDTH: float = 595
TARGET_HEIGHT: float = 842
WHITE_THRESHOLD: float = 160

SCORES = [10, 8, 6, 4, 2]
print("> Target size set (595x842 pixels).")
print(f"> Scores: {SCORES}")

# 사용자 지정: 10점 원과 외곽 원
INNER_RADIUS: float = 28
OUTER_RADIUS: float = 293
print(f"> Inner Radius: {INNER_RADIUS}, Outer Radius: {OUTER_RADIUS}")


# 자동으로 같은 간격 링 생성
RING_RADIUS = np.linspace(INNER_RADIUS, OUTER_RADIUS, len(SCORES)).astype(int).tolist()
print("> Auto Generated RING_RADIUS:", RING_RADIUS)

# 최소 거리 차이 (같은 레이저를 중복 감지 방지)
MIN_DIST = 0
# 최소 시간 간격 (초 단위)
MIN_INTERVAL = 0.3

MAX_SHOTS = 10

SHOW_ORIGINAL = False

print(f"> DPS init.")
print("> System initialized.")


# --------------------------
# 유틸리티 함수
# --------------------------
def calculate_score(x, y):
    """과녁 중심 기준 점수 계산"""
    cx, cy = TARGET_WIDTH // 2, TARGET_HEIGHT // 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    for radius, score in zip(RING_RADIUS, SCORES):
        if dist <= radius:
            return score
    return 0


def is_new_shot(new_pt, hits, min_interval=MIN_INTERVAL, min_dist=MIN_DIST):
    """최근 점들과 비교해서 150ms 내 근처 점이면 False, 아니면 True"""
    current_time = time.time()
    for hx, hy, t in hits:
        if ((new_pt[0] - hx) ** 2 + (new_pt[1] - hy) ** 2) ** 0.5 < min_dist or (
            current_time - t
        ) < min_interval:
            return False
    return True


def reset():
    for _ in range(10):
        cap.read()
    total_score = sum([s for _, _, s in shots])
    print(f"=== 10 Shots Completed ===")
    print(f"Total Score: {name} - {total_score}")
    shots.clear()  # 초기화
    hits.clear()  # 초기화
    input("Press Enter to reset shots and hits for next round...\n")
    print(f"==========================\n")
    print("Shots and hits have been reset for next round.\n")
    name_input()


def name_input():
    """사용자 이름 입력"""
    global name
    name = input("Enter your name(settings|debug|*): ")


    if name == "settings":
        print("Settings mode activated. Please configure your settings.")
        # 여기에 설정 관련 코드를 추가할 수 있습니다.
        TARGET_WIDTH = update("Target Width", 595)
        TARGET_HEIGHT = update("Target Height", 842)

        print(
            f"\tSettings updated: TARGET_WIDTH={TARGET_WIDTH}, TARGET_HEIGHT={TARGET_HEIGHT}"
        )
        INNER_RADIUS = update("Inner Radius", 28)
        OUTER_RADIUS = update("Outer Radius", 293)
        print(
            f"\tSettings updated: INNER_RADIUS={INNER_RADIUS}, OUTER_RADIUS={OUTER_RADIUS}"
        )
        RING_RADIUS = (
            np.linspace(INNER_RADIUS, OUTER_RADIUS, len(SCORES)).astype(int).tolist()
        )
        print(f"\tAuto Generated RING_RADIUS: {RING_RADIUS}")

        WHITE_THRESHOLD = update("White Threshol(0-255)", 160)
        print(f"\tSettings updated: WHITE_THRESHOLD={WHITE_THRESHOLD}")
        
        print()
        name_input()  # 재귀 호출로 이름 입력 반복
    print(f"Welcome {name}!\n")


print("""

==============================

== RS Target System Ver.4.0 ==
- Developed By H1xaru of RDL -
- Enhanced By  ywbird of RDL -

==============================

Ver.4.0 Changes
    - Changed red point detection method from HSV to YCbCr.
    - Added 0.3 seconds delay after round to prevent cause error in next game.
    - Added demo video testing.

""")

name_input()

# --------------------------
# 웹캠 열기
# --------------------------

update: Callable[[str, float], float] = lambda name, default: (
        lambda x: int(x) if x.isdigit() else default
    )(input(f"{name} (default: {default}): "))

vid = 0

print("Select Video input\n\t(0) Webcam (1) Videos\n")
ans = update("Video input", 0)

if ans == 0:
    vid = update("Webcam Input id", 0)
if ans == 1:
    videos = os.listdir("videos")
    for idx, file in enumerate(videos):
        print(f'\t({idx}) {file}')
    print()
    vid = os.path.join('videos', videos[int(input("Video id: "))])

print(vid)
cap = cv2.VideoCapture(vid)

shots = []  # 기록: (shot_number, (x,y), score)
hits = []  # 기록: (x,y,timestamp)

# while len(shots) + 1 <= 10:
while True:
    ret, frame = cap.read()
    if not ret:
        break
    hh, ww = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 아루코 마커 검출
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and len(ids) >= 4:
        if all(i in ids for i in range(4)):
            # 마커 좌표 정리
            marker_corners = {i: corners[np.where(ids == i)[0][0]][0] for i in range(4)}
            src_pts = np.array(
                [
                    marker_corners[0][0],
                    marker_corners[1][1],
                    marker_corners[2][2],
                    marker_corners[3][3],
                ],
                dtype=np.float32,
            )
            dst_pts = np.array(
                [
                    [0, 0],
                    [TARGET_WIDTH, 0],
                    [TARGET_WIDTH, TARGET_HEIGHT],
                    [0, TARGET_HEIGHT],
                ],
                dtype=np.float32,
            )

            # 투시 변환
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(frame, M, (TARGET_WIDTH, TARGET_HEIGHT))

            # 레이저 포인트 검출 (빨강 + 밝은 점)

            # BGR → YCrCb 변환 (YCbCr과 동일)
            ycrcb = cv2.cvtColor(warped, cv2.COLOR_BGR2YCrCb)

            # 채널 분리
            Y, Cr, Cb = cv2.split(ycrcb)

            # 이진 이미지
            _, binary = cv2.threshold(Cr, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

            # hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
            #
            # lower_red1 = np.array([0, 150, 150])
            # upper_red1 = np.array([10, 255, 255])
            # lower_red2 = np.array([160, 150, 150])
            # upper_red2 = np.array([179, 255, 255])
            # mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            # mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            #
            # lower_bright = np.array([0, 0, 200])
            # upper_bright = np.array([179, 50, 255])
            # mask3 = cv2.inRange(hsv, lower_bright, upper_bright)
            #
            # mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)
            # mask = cv2.medianBlur(mask, 5)

            # 점 추출
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                if cv2.contourArea(cnt) < 5:
                    continue
                M_cnt = cv2.moments(cnt)
                if M_cnt["m00"] == 0:
                    continue
                cx = int(M_cnt["m10"] / M_cnt["m00"])
                cy = int(M_cnt["m01"] / M_cnt["m00"])

                if is_new_shot((cx, cy), hits):
                    score = calculate_score(cx, cy)
                    shot_number = len(shots) + 1
                    shots.append((shot_number, (cx, cy), score))
                    hits.append((cx, cy, time.time()))
                    print(f"Shot {shot_number}: Score={score}, Position=({cx},{cy})")

                if len(shots) >= MAX_SHOTS:
                    reset()

            # 과녁 원 & 점수 표시
            center = (TARGET_WIDTH // 2, TARGET_HEIGHT // 2)
            for radius, score in zip(RING_RADIUS, SCORES):
                cv2.circle(warped, center, radius, (255, 255, 255), 2)
                cv2.putText(
                    warped,
                    str(score),
                    (center[0] + radius - 30, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            # 맞은 자리 표시
            for i, (_, (sx, sy), _) in enumerate(shots):
                if i == len(shots) - 1:
                    cv2.circle(warped, (sx, sy), 7, (0, 255, 0), -1)  # 최근: 초록
                else:
                    cv2.circle(warped, (sx, sy), 5, (0, 0, 255), -1)  # 이전: 빨강

            # 최근 점수 표시
            if shots:
                cv2.putText(
                    warped,
                    f"{name} - Last Score: {shots[-1][2]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Warped Target", warped)

    if name == "debug" or SHOW_ORIGINAL == True:
        cv2.imshow("Original", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    if key == ord("h"):
        print("hits: ", hits)
        print("shots: ", shots)

cap.release()
cv2.destroyAllWindows()
