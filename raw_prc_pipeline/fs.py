import cv2
import numpy as np

'''
作用：
该函数用于模拟“闪光”效果，增强图像的亮度和对比度。
参数：
source：输入图像。
a：控制增强强度的参数，默认为5。
target：目标亮度，默认为-1，表示自动调整。
perform_gamma_correction：是否进行伽马校正，默认为True。
处理流程：
计算输入图像的亮度通道（取RGB三通道的最大值）。
防止亮度为0的情况，将0值替换为极小值（1e-9）。
根据亮度值和参数a计算调整后的像素值。
如果启用了伽马校正，对结果进行伽马校正。
根据目标亮度target调整结果图像的亮度，如果没有指定目标亮度，则将最大亮度调整为255。
'''
def perform_flash(source, a=5, target=-1, perform_gamma_correction=True):
    rows, cols, _ = source.shape

    v = np.max(source, axis=2)
    vd = np.copy(v)
    vd[vd == 0] = 1e-9
    result = source / (a * np.exp(np.mean(np.log(vd))) + np.tile(np.expand_dims(vd, axis=2), (1, 1, 3)))

    if perform_gamma_correction:
        result **= 1.0 / 2.2

    if target >= 0:
        result *= target / np.mean((0.299 * result[:, :, 2] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 0]))
    else:
        result *= 255.0 / np.max(result)

    return result


'''
作用：
该函数用于实现“风暴”效果，通过多尺度亮度调整增强图像的细节和对比度。
参数：
source：输入图像。
a：控制增强强度的参数，默认为5。
target：目标亮度，默认为-1，表示自动调整。
kernels：一组用于多尺度亮度调整的核大小。
perform_gamma_correction：是否进行伽马校正，默认为True。
处理流程：
计算输入图像的亮度通道（取RGB三通道的最大值）。
防止亮度为0的情况，将0值替换为极小值（1e-9）。
对亮度值进行对数变换。
使用不同大小的核进行盒式滤波，模拟多尺度亮度调整。
根据滤波后的亮度值和参数a计算调整后的像素值。
如果启用了伽马校正，对结果进行伽马校正。
根据目标亮度target调整结果图像的亮度，如果没有指定目标亮度，则将最大亮度调整为255。
'''
def perform_storm(source, a=5, target=-1, kernels=(1, 4, 16, 64, 256), perform_gamma_correction=True):
    rows, cols, _ = source.shape

    v = np.max(source, axis=2)
    vd = np.copy(v)
    vd[vd == 0] = 1e-9
    lv = np.log(vd)
    result = sum([source / np.tile(
        np.expand_dims(a * np.exp(cv2.boxFilter(lv, -1, (int(min(rows // kernel, cols // kernel)),) * 2)) + vd, axis=2),
        (1, 1, 3)) for kernel in kernels])

    if perform_gamma_correction:
        result **= 1.0 / 2.2

    if target >= 0:
        result *= target / np.mean((0.299 * result[:, :, 2] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 0]))
    else:
        result *= 255.0 / np.max(result)

    return result
