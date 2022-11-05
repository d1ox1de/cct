import numpy as np
import cv2
import yaml

# LABELS =  {
#     0: "no_object",
#     1: "label1",
#     2: "label2",
#     3: "label3",
#     4: "label4",
# }

ROTATE_POLICY = {
    1: cv2.ROTATE_90_COUNTERCLOCKWISE,
    2: cv2.ROTATE_180,
    3: cv2.ROTATE_90_CLOCKWISE
}


def augment_hsv(img, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5) -> None:
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
    return None


def img_rotate_angle(img: np.ndarray, angle_max: float) -> np.ndarray:
    wh = np.array(img.shape)[1::-1]
    wh_img_center = tuple(wh / 2)
    angle = np.random.uniform(low=-1., high=1.) * angle_max
    M = cv2.getRotationMatrix2D(wh_img_center, angle, scale=1)
    img_rotated = cv2.warpAffine(img, M, tuple(wh))
    return img_rotated


def img_rotate_multipleof90(img: np.ndarray, rotate_code: int = 0) -> np.ndarray:
    if rotate_code == 0:
        return img
    else:
        return cv2.rotate(img, ROTATE_POLICY[rotate_code])