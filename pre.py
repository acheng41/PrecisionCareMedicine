import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff, fs, order=4):
    # 归一化截止频率 (Nyquist 频率为 fs/2)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # 获取滤波器系数
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # 应用滤波器
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def resample_angle(data):
    n = data.shape[0]
    index = np.linspace(0, 100, n)
    target = np.linspace(0, 100, 100)
    resampled_data = np.zeros((100, 3))
    for i in range(3):
        interp_func = interp1d(index, data[:, i])
        resampled_data[:, i] = interp_func(target)

    return resampled_data

