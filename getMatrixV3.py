from itertools import chain

import pandas as pd
import numpy as np
from pathlib import Path
import os
import scipy.io as sio
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt

from pre import resample_angle
from torch.utils.data import Dataset, DataLoader



def detect_outliers(values):
    Q1, Q3 = np.percentile(values, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def segment_insole_data(insole_data):
    ffs = insole_data[:, :21, :]
    mfs = insole_data[:, 21:42, :]
    rfs = insole_data[:, 42:, :]
    return ffs, mfs, rfs

def butter_lowpass_filter(data, cutoff=8, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def reshape_insole_grid(data):
    num_frames = data.shape[0]
    reshaped_data = np.empty((num_frames, 64, 16), dtype=data.dtype)
    for i in range(num_frames):
        frame = data[i].reshape(16, 64).T.copy()
        frame[:32, :] = np.flipud(frame[:32, :])
        reshaped_data[i] = frame

    return reshaped_data

def gait_segmentation(insole, h_th, t_th):

    insole = reshape_insole_grid(insole)
    
    p = np.mean(insole, axis=(1, 2))
    

    p_filtered = butter_lowpass_filter(p, 8, 100)
   
    p_derivative = np.gradient(p_filtered)

    hc_indices, _ = find_peaks(p_derivative, height=h_th, distance=10)
    to_indices, _ = find_peaks(-p_derivative, height=t_th, distance=10)

    return hc_indices, to_indices



def resample_data(data):
    n = data.shape[0]
    index = np.linspace(0, 100, n)
    target = np.linspace(0, 100, 100)
    resampled_data = np.zeros((100, 3))
    for i in range(3):
        interp_func = interp1d(index, data[:, i])
        resampled_data[:, i] = interp_func(target)

    return resampled_data


def get_gait_parameters_insole3(insole_r, insole_l, t_r, t_l, thresholds):
    """
    Get gait parameters from insole data
    """

    [h_th_r, t_th_r, h_th_l, t_th_l, strike_th_l, strike_th_r] = thresholds

    gait = {'foot_trace_r': np.zeros(len(t_r)), 'foot_trace_l': np.zeros(len(t_l)),
            'dim': [int(np.sqrt(insole_r.shape[1]) * 2), int(np.sqrt(insole_r.shape[1]) / 2)],
            'cop_x_r': np.zeros(len(t_r)),
            'cop_y_r': np.zeros(len(t_r)),
            'cont_area_r': np.zeros(len(t_r)),
            'cop_x_l': np.zeros(len(t_l)),
            'cop_y_l': np.zeros(len(t_l)),
            'cont_area_l': np.zeros(len(t_l)),
            'step_r':[],
            'step_l':[]
            }

    for i in range(len(t_r)):
        frame = insole_r[i, :].reshape(gait['dim'][0], gait['dim'][1], order='F')
        frame[:gait['dim'][0] // 2, :] = np.flipud(frame[:gait['dim'][0] // 2, :])
        gait['foot_trace_r'][i] = np.mean(frame)
        x, y = np.where(frame > 0)

        # COP
        sum_frame_r = np.sum(frame[x, y])
        if sum_frame_r > 0:
            gait['cop_x_r'][i] = np.sum(x * frame[x, y]) / sum_frame_r
            gait['cop_y_r'][i] = np.sum(y * frame[x, y]) / sum_frame_r
        else:
            gait['cop_x_r'][i] = np.nan
            gait['cop_y_r'][i] = np.nan

        gait['cont_area_r'][i] = len(x)


    for i in range(len(t_l)):
        frame = insole_l[i, :].reshape(gait['dim'][0], gait['dim'][1], order='F')
        frame[:gait['dim'][0] // 2, :] = np.flipud(frame[:gait['dim'][0] // 2, :])
        gait['foot_trace_l'][i] = np.mean(frame)
        x, y = np.where(frame > 0)

        # COP
        sum_frame_l = np.sum(frame[x, y])
        if sum_frame_l > 0:
            gait['cop_x_l'][i] = np.sum(x * frame[x, y]) / sum_frame_l
            gait['cop_y_l'][i] = np.sum(y * frame[x, y]) / sum_frame_l
        else:
            gait['cop_x_l'][i] = np.nan
            gait['cop_y_l'][i] = np.nan

        gait['cont_area_l'][i] = len(x)


    # Gait events
    hc_indices, to_indices = gait_segmentation(insole_r, h_th_r, t_th_r)
    strike_r = hc_indices
    off_r = to_indices

    hc_indices, to_indices = gait_segmentation(insole_l, h_th_l, t_th_l)
    strike_l = hc_indices
    off_l = to_indices

    # Isolate complete gait cycles
    gait['step_r'] = []
    gait['step_l'] = []
    for i in range(len(strike_r) - 1):
        start, end = strike_r[i], strike_r[i + 1]
        if t_r[end] - t_r[start] > strike_th_r:
            continue
        step_off = [o for o in off_r if start <= o <= end]
        gait['step_r'].append({'strike': [start, end], 'off': step_off})
    
    off_indices = list(chain.from_iterable(step["off"] for step in gait['step_r']))
    strike_indices = [step["strike"][0] for step in gait['step_r'] if len(step["strike"]) > 0]

    off_values = np.array(gait['foot_trace_r'])[off_indices]
    strike_values = np.array(gait['foot_trace_r'])[strike_indices]

    lower_bound_off, upper_bound_off= detect_outliers(off_values)
    lower_bound_strike, upper_bound_strike = detect_outliers(strike_values)

    abnormal_off = set(off_indices[i] for i in range(len(off_values))
                       if off_values[i] < lower_bound_off or off_values[i] > upper_bound_off)

    abnormal_strike = set(strike_indices[i] for i in range(len(strike_values))
                          if strike_values[i] < lower_bound_strike or strike_values[i] > upper_bound_strike)

    filtered_gait_step_r = [step for step in gait['step_r']
                            if not (set(step["off"]) & abnormal_off or set(step["strike"]) & abnormal_strike)]

    gait['step_r'] = filtered_gait_step_r
    
    for i in range(len(strike_l) - 1):
        start, end = strike_l[i], strike_l[i + 1]
        if t_l[end] - t_l[start] > strike_th_l:
            continue
        step_off = [o for o in off_l if start <= o <= end]
        gait['step_l'].append({'strike': [start, end], 'off': step_off})
    
    off_indices = list(chain.from_iterable(step["off"] for step in gait['step_l']))
    strike_indices = [step["strike"][0] for step in gait['step_l'] if len(step["strike"]) > 0]

    off_values = np.array(gait['foot_trace_l'])[off_indices]
    strike_values = np.array(gait['foot_trace_l'])[strike_indices]

    lower_bound_off, upper_bound_off = detect_outliers(off_values)
    lower_bound_strike, upper_bound_strike = detect_outliers(strike_values)

    abnormal_off = set(off_indices[i] for i in range(len(off_values))
                       if off_values[i] < lower_bound_off or off_values[i] > upper_bound_off)

    abnormal_strike = set(strike_indices[i] for i in range(len(strike_values))
                          if strike_values[i] < lower_bound_strike or strike_values[i] > upper_bound_strike)

    filtered_gait_step_l = [step for step in gait['step_l']
                            if not (set(step["off"]) & abnormal_off or set(step["strike"]) & abnormal_strike)]

    gait['step_l'] = filtered_gait_step_l
   
    # Temporal parameters
    gait['cycle_dur_l'] = np.zeros((len(gait['step_l'])))
    gait['swing_dur_l']= np.zeros((len(gait['step_l'])))
    gait['stance_dur_l'] = np.zeros((len(gait['step_l'])))
    gait['cadence_l'] = np.zeros(len(gait['cycle_dur_l']))
    gait['cycle_dur_r'] = np.zeros((len(gait['step_r'])))
    gait['swing_dur_r']= np.zeros((len(gait['step_r'])))
    gait['stance_dur_r'] = np.zeros((len(gait['step_r'])))
    gait['cadence_r'] = np.zeros(len(gait['cycle_dur_r']))

    for i in range((len(gait['step_l']))):
        start, end = gait['step_l'][i]['strike']
        off = gait['step_l'][i]['off']
        gait['cycle_dur_l'][i] = t_l[end] - t_l[start]
        gait['stance_dur_l'][i] = t_l[off] - t_l[start]
    gait['swing_dur_l'] = gait['cycle_dur_l'] - gait['stance_dur_l']
    gait['stance_phase_l'] = gait['stance_dur_l']/gait['cycle_dur_l']
    gait['swing_phase_l'] = gait['swing_dur_l'] / gait['cycle_dur_l']
    gait['cadence_l'] = 60/gait['cycle_dur_l']

    for i in range((len(gait['step_r']))):
        start, end = gait['step_r'][i]['strike']
        off = gait['step_r'][i]['off']
        gait['cycle_dur_r'][i] = t_r[end] - t_r[start]
        gait['stance_dur_r'][i] = t_r[off] - t_r[start]
    gait['swing_dur_r'] = gait['cycle_dur_r'] - gait['stance_dur_r']
    gait['stance_phase_r'] = gait['stance_dur_r']/gait['cycle_dur_r']
    gait['swing_phase_r'] = gait['swing_dur_r'] / gait['cycle_dur_r']
    gait['cadence_r'] = 60/gait['cycle_dur_r']


    gait['swing_asym'] = np.abs((np.mean(gait['swing_phase_l']) - np.mean(gait['swing_phase_r']))) / (
            0.5 * (np.mean(gait['swing_phase_l']) + np.mean(gait['swing_phase_r'])))

    gait['cadence_asym'] = np.abs((np.mean(gait['cadence_l']) - np.mean(gait['cadence_r']))) / (
            0.5 * (np.mean(gait['cadence_l']) + np.mean(gait['cadence_r'])))

    for i in range(len(gait['step_l'])):
        step = gait['step_l'][i]
        strike_start, strike_end = step['strike']
        off_frame = step['off'][0]

        # **Strike post 5 frames**
        strike_end_idx = min(strike_start + 5, len(gait['cop_x_l']))  #
        avg_cop_y_strike = np.mean(gait['cop_x_l'][strike_start:strike_end_idx])

        # **Off previous 5 frames**
        off_start_idx = max(0, off_frame - 5)
        avg_cop_y_off = np.mean(gait['cop_x_l'][off_start_idx:off_frame])

        # ** strike region**
        if avg_cop_y_strike >= 40:
            gait['step_l'][i]['region_strike'] = 'heel_strike'
        elif avg_cop_y_strike >= 20:
            gait['step_l'][i]['region_strike'] = 'midfoot_strike'
        else:
            gait['step_l'][i]['region_strike'] = 'forefoot_strike'

        # ** off region **
        if avg_cop_y_off >= 40:
            gait['step_l'][i]['region_off'] = 'heel_off'
        elif avg_cop_y_off >= 20:
            gait['step_l'][i]['region_off'] = 'midfoot_off'
        else:
            gait['step_l'][i]['region_off'] = 'toe_off'

    for i in range(len(gait['step_r'])):
        step = gait['step_r'][i]
        strike_start, strike_end = step['strike']
        off_frame = step['off'][0]

        # **Strike post 5 frame**
        strike_end_idx = min(strike_start + 5, len(gait['cop_x_r']))
        avg_cop_y_strike = np.mean(gait['cop_x_r'][strike_start:strike_end_idx])

        # ** The average COP of the 5 frames before the off event**
        off_start_idx = max(0, off_frame - 5)
        avg_cop_y_off = np.mean(gait['cop_x_r'][off_start_idx:off_frame])

        # **Determine the strike region**
        if avg_cop_y_strike >= 32:
            gait['step_r'][i]['region_strike'] = 'heel_strike'
        elif avg_cop_y_strike >= 13:
            gait['step_r'][i]['region_strike'] = 'midfoot_strike'
        else:
            gait['step_r'][i]['region_strike'] = 'forefoot_strike'

        # **Determine the off region**
        if avg_cop_y_off >= 32:
            gait['step_r'][i]['region_off'] = 'heel_off'
        elif avg_cop_y_off >= 13:
            gait['step_r'][i]['region_off'] = 'midfoot_off'
        else:
            gait['step_r'][i]['region_off'] = 'toe_off'

    cop_y_list_r, cop_x_list_r = [], []
    cop_y_r = gait['cop_x_r']
    cop_x_r = gait['cop_y_r']
    for step in gait['step_r']:
        strike_start, _ = step['strike']
        off = step['off'][0]
        cop_y_list_r.append(cop_y_r[strike_start:off])
        cop_x_list_r.append(cop_x_r[strike_start:off])

    cop_y_list_l, cop_x_list_l = [], []
    cop_y_l = gait['cop_x_l']
    cop_x_l = gait['cop_y_l']
    for step in gait['step_l']:
        strike_start, _ = step['strike']
        off = step['off'][0]
        cop_y_list_l.append(cop_y_l[strike_start:off])
        cop_x_list_l.append(cop_x_l[strike_start:off])

    # normalized to 100
    interp = lambda lst: [np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(x)), x) for x in lst]

    # mean and std
    cop_x_r_avg = np.mean(interp(cop_x_list_r), axis=0)
    cop_y_r_avg = np.mean(interp(cop_y_list_r), axis=0)
    cop_x_r_std = np.std(interp(cop_x_list_r), axis=0)
    cop_y_r_std = np.std(interp(cop_y_list_r), axis=0)

    cop_x_l_avg = np.mean(interp(cop_x_list_l), axis=0)
    cop_y_l_avg = np.mean(interp(cop_y_list_l), axis=0)
    cop_x_l_std = np.std(interp(cop_x_list_l), axis=0)
    cop_y_l_std = np.std(interp(cop_y_list_l), axis=0)

    gait['cop_x_avg_r'] = cop_x_r_avg
    gait['cop_y_avg_r'] = cop_y_r_avg
    gait['cop_x_avg_l'] = cop_x_l_avg
    gait['cop_y_avg_l'] = cop_y_l_avg
    gait['cop_x_std_r'] = cop_x_r_std
    gait['cop_y_std_r'] = cop_y_r_std
    gait['cop_x_std_l'] = cop_x_l_std
    gait['cop_y_std_l'] = cop_y_l_std

    cop_motion_rangex_r = []
    cop_motion_rangey_r = []
    for x, y in zip(cop_x_list_r, cop_y_list_r):
        motion_range_x = np.max(x) - np.min(x)
        motion_range_y = np.max(y) - np.min(y)
        cop_motion_rangex_r.append(motion_range_x*2)
        cop_motion_rangey_r.append(motion_range_y*2)

    cop_motion_rangex_l = []
    cop_motion_rangey_l = []
    for x, y in zip(cop_x_list_l, cop_y_list_l):
        motion_range_x = np.max(x) - np.min(x)
        motion_range_y = np.max(y) - np.min(y)
        cop_motion_rangex_l.append(motion_range_x*2)
        cop_motion_rangey_l.append(motion_range_y*2)

    gait['cop_range_x_r'] = cop_motion_rangex_r
    gait['cop_range_y_r'] = cop_motion_rangey_r
    gait['cop_range_x_l'] = cop_motion_rangex_l
    gait['cop_range_y_l'] = cop_motion_rangey_l

    vx_list_l, vy_list_l = [], []
    cop_length_list_l = []

    for step in gait['step_l']:
        strike_start, strike_end = step['strike']
        off = step['off'][0]

        x = gait['cop_y_l'][strike_start:off]
        y = gait['cop_x_l'][strike_start:off]
        t = t_l[strike_start:off].flatten()

        dx = np.diff(x) * 2
        dy = np.diff(y) * 2
        dt = np.diff(t)
        dt[dt == 0] = 1e-6

        vx = dx / dt
        vy = dy / dt
        vx_list_l.append(vx)
        vy_list_l.append(vy)
        length = np.sum(np.sqrt(dx ** 2 + dy ** 2))
        cop_length_list_l.append(length)
    interp_vx_l = interp(vx_list_l)
    interp_vy_l = interp(vy_list_l)
    gait['cop_length_l']=cop_length_list_l
    gait['cop_vx_l']=interp_vx_l
    gait['cop_vy_l']=interp_vy_l

    vx_list_r, vy_list_r = [], []
    cop_length_list_r = []

    for step in gait['step_r']:
        strike_start, _ = step['strike']
        off = step['off'][0]

        x = gait['cop_y_r'][strike_start:off] *2
        y = gait['cop_x_r'][strike_start:off] *2
        t = t_r[strike_start:off].flatten()

        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)
        dt[dt == 0] = 1e-6

        vx = dx / dt
        vy = dy / dt
        vx_list_r.append(vx)
        vy_list_r.append(vy)
        length = np.sum(np.sqrt(dx ** 2 + dy ** 2))
        cop_length_list_r.append(length)
    interp_vx_r = interp(vx_list_r)
    interp_vy_r = interp(vy_list_r)
    gait['cop_length_r']=cop_length_list_r
    gait['cop_vx_r']=interp_vx_r
    gait['cop_vy_r']=interp_vy_r

    # mean and std
    cop_vx_r_avg = np.mean(interp_vx_r, axis=0)
    cop_vy_r_avg = np.mean(interp_vy_r, axis=0)
    cop_vx_r_std = np.std(interp_vx_r, axis=0)
    cop_vy_r_std = np.std(interp_vy_r, axis=0)

    cop_vx_l_avg = np.mean(interp_vx_l, axis=0)
    cop_vy_l_avg = np.mean(interp_vy_l, axis=0)
    cop_vx_l_std = np.std(interp_vx_l, axis=0)
    cop_vy_l_std = np.std(interp_vy_l, axis=0)

    gait['cop_vx_avg_r'] = cop_vx_r_avg
    gait['cop_vy_avg_r'] = cop_vy_r_avg
    gait['cop_vx_avg_l'] = cop_vx_l_avg
    gait['cop_vy_avg_l'] = cop_vy_l_avg
    gait['cop_vx_std_r'] = cop_vx_r_std
    gait['cop_vy_std_r'] = cop_vy_r_std
    gait['cop_vx_std_l'] = cop_vx_l_std
    gait['cop_vy_std_l'] = cop_vy_l_std

    return gait
