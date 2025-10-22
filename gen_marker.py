import cv2
import os

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

for idx, key in enumerate(ARUCO_DICT.keys()):
    print("[{}]: {}".format(idx, key))

Type = list(ARUCO_DICT.keys())[int(input(f"pick aruco type 0-{len(ARUCO_DICT)-1}: "))]
id = int(input('pick aruco id: '))

# load the ArUCo dictionary
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[Type])

# tag on the output image
print("[INFO] generating ArUCo tag type '{}' with ID '{}'".format(Type, id))
img_size = 500

marker_img = cv2.aruco.generateImageMarker(arucoDict, id, img_size)

os.makedirs("markers", exist_ok=True)
cv2.imwrite("./markers/" + Type + "_" + "id_" + str(id) + ".jpg", marker_img)

cv2.imshow(f"Marker {Type} {id}", marker_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
