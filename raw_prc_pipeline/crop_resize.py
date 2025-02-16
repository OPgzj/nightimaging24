import cv2
import numpy as np

'''
根据huawei bounds进行裁剪
'''
def crop_by_bounds(image: np.ndarray, bounds_coords: tuple) -> np.ndarray:
    h_start, h_end, w_start, w_end = bounds_coords
    return image[h_start:h_end, w_start:w_end]


'''
垂直裁剪[200:2200],水平居中裁剪
'''

def upper_crop(img, upper_bound = 200):
    mid = img.shape[1] // 2
    return img[upper_bound:upper_bound + 2000, mid-1000:mid+1000]


def crop_resize(undistorted_img: np.ndarray, bounds: tuple[int, int, int, int]):
    """
    Crops and resizes undistorted image to 2000x2000.

    Parameters:
        undistorted_img (np.ndarray): Undistorted image, original spatial size.
    Returns:
        np.ndarray: Cropped and resized image.
    """
    resized = cv2.resize(undistorted_img, dsize=(2654, 3538)) # 压缩图像
    cropped = upper_crop(crop_by_bounds(resized, bounds))
    return cropped