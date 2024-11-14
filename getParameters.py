import pandas as pd
import numpy as np
from pathlib import Path
import os
import scipy.io as sio
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import torch
from pre import resample_angle
from torch.utils.data import Dataset, DataLoader


def get_gait_parameters_insole(insole_r, insole_l, t_r, t_l):
    """
    Get gait parameters from insole data
    """
    gait = {'t_r': t_r, 'insole_r': insole_r, 't_l': t_l, 'insole_l': insole_l, 'area': 0.002 ** 2,
            'dim': [int(np.sqrt(insole_r.shape[1]) * 2), int(np.sqrt(insole_r.shape[1]) / 2)],
            'foot_trace_r': np.zeros(len(t_r)), 'foot_trace_l': np.zeros(len(t_l)),
            'cop_x_r': np.zeros(len(t_r)), 'cop_y_r': np.zeros(len(t_r)),
            'cop_x_l': np.zeros(len(t_l)), 'cop_y_l': np.zeros(len(t_l)),
            'cont_area_r': np.zeros(len(t_r)), 'cont_area_l': np.zeros(len(t_l)),
            'pp_r': np.max(insole_r, axis=1), 'pp_l': np.max(insole_l, axis=1),
            'pp_x_r': np.zeros_like(t_r), 'pp_y_r': np.zeros_like(t_r),
            'pp_x_l': np.zeros_like(t_l), 'pp_y_l': np.zeros_like(t_l),
            }

    # Center of Pressure, Gait Trajectory, Contact Area and Trace
    for i in range(len(t_r)):
        frame = insole_r[i, :].reshape(gait['dim'][0], gait['dim'][1], order='F')
        # frame = np.fliplr(frame)
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

        # Instance peak pressure
        x, y = np.nonzero(frame == gait['pp_r'][i])
        if len(x) > 1:
            gait['pp_x_r'][i] = np.mean(x)
            gait['pp_y_r'][i] = np.mean(y)
        else:
            gait['pp_x_r'][i] = x
            gait['pp_y_r'][i] = y

    for i in range(len(t_l)):
        frame = insole_l[i, :].reshape(gait['dim'][0], gait['dim'][1], order='F')
        # frame[:gait['dim'][0] // 2, :] = np.flipud(frame[:gait['dim'][0] // 2, :])
        gait['foot_trace_l'][i] = np.mean(frame)
        x, y = np.where(frame > 0)
        sum_frame_l = np.sum(frame[x, y])
        if sum_frame_l > 0:
            gait['cop_x_l'][i] = np.sum(x * frame[x, y]) / sum_frame_l
            gait['cop_y_l'][i] = np.sum(y * frame[x, y]) / sum_frame_l
        else:
            gait['cop_x_l'][i] = np.nan
            gait['cop_y_l'][i] = np.nan

        gait['cont_area_l'][i] = len(x)

        # Instance peak pressure
        x, y = np.nonzero(frame == gait['pp_l'][i])
        if len(x) > 1:
            gait['pp_x_l'][i] = np.mean(x)
            gait['pp_y_l'][i] = np.mean(y)
        else:
            gait['pp_x_l'][i] = x
            gait['pp_y_l'][i] = y

    ## Heel Strike Toe Off and Related Parameters
    thresh_r = np.min(gait['foot_trace_r']) + 0.1 * np.ptp(gait['foot_trace_r'])
    thresh_l = np.min(gait['foot_trace_l']) + 0.1 * np.ptp(gait['foot_trace_l'])
    gait['strike_r'] = []
    gait['off_r'] = []
    gait['strike_l'] = []
    gait['off_l'] = []

    # Right foot strikes and offs
    for i in range(1, len(t_r) - 1):
        if gait['foot_trace_r'][i] >= thresh_r > gait['foot_trace_r'][i - 1]:
            gait['strike_r'].append(i)
        if gait['foot_trace_r'][i] >= thresh_r > gait['foot_trace_r'][i + 1]:
            gait['off_r'].append(i + 1)

    # Left foot strikes and offs
    for i in range(1, len(t_l) - 1):
        if gait['foot_trace_l'][i] >= thresh_l > gait['foot_trace_l'][i - 1]:
            gait['strike_l'].append(i)
        if gait['foot_trace_l'][i] >= thresh_l > gait['foot_trace_l'][i + 1]:
            gait['off_l'].append(i + 1)

    # Isolate complete gait cycles
    gait['off_r'] = [off for off in gait['off_r'] if gait['strike_r'][0] < off <= gait['strike_r'][-1]]
    gait['off_l'] = [off for off in gait['off_l'] if gait['strike_l'][0] < off <= gait['strike_l'][-1]]

    # Cycle duration
    gait['cycle_dur_r'] = np.diff(t_r[gait['strike_r']].flatten())
    gait['cycle_dur_l'] = np.diff(t_l[gait['strike_l']].flatten())

    # Cycle duration variability
    gait['cycle_var_r'] = np.std(gait['cycle_dur_r']) / np.mean(gait['cycle_dur_r']) * 100
    gait['cycle_var_l'] = np.std(gait['cycle_dur_l']) / np.mean(gait['cycle_dur_l']) * 100

    # Cadence
    gait['cadence'] = min(len(gait['cycle_dur_r']), len(gait['cycle_dur_l'])) / min(np.sum(gait['cycle_dur_r']),
                                                                                    np.sum(gait['cycle_dur_l'])) * 60
    # Stance and swing phases
    gait['stance_r'] = (t_r[gait['off_r']] - t_r[gait['strike_r'][:-1]]) / gait['cycle_dur_r'] * 100
    gait['stance_l'] = (t_l[gait['off_l']] - t_l[gait['strike_l'][:-1]]) / gait['cycle_dur_l'] * 100

    gait['swing_r'] = 100 - gait['stance_r']
    gait['swing_l'] = 100 - gait['stance_l']

    # Asymmetry (swing)
    gait['asym'] = (np.mean(gait['swing_l']) - np.mean(gait['swing_r'])) / (
                0.5 * (np.mean(gait['swing_l']) + np.mean(gait['swing_r']))) * 100


    return gait



def gait_aligned_jnt(gait, jnt_angles_l, jnt_angles_r, jnt_pos_l, jnt_pos_r, t_trackers):
    """
    Align joint angles with gait cycle phases (heel strikes and toe offs).
    """

    joint = {'jnt_angles_all_l': np.zeros((np.size(jnt_angles_l),3,3)),
             'jnt_angles_all_r': np.zeros((np.size(jnt_angles_r), 3,3)),
             'jnt_pos_all_l': np.zeros((np.size(jnt_pos_l),4,3)),
             'jnt_pos_all_r': np.zeros((np.size(jnt_pos_r), 4,3)),
             'angles_l':{'ankle':[],'knee':[],'hip':[]},
             'angles_r':{'ankle':[],'knee':[],'hip':[]},
             'resampled_angles_l':{'ankle':[],'knee':[],'hip':[]},
             'resampled_angles_r':{'ankle':[],'knee':[],'hip':[]},
             't_trackers': t_trackers,
             'strike_r':[], 'strike_l':[], 'off_r':[], 'off_l':[]}

    ankle = []
    knee = []
    hip = []
    for i in range(np.size(jnt_angles_l)):
        ankle.append(jnt_angles_l[i][0][2, :])
        knee.append(jnt_angles_l[i][0][1, :])
        hip.append(jnt_angles_l[i][0][0, :])

    joint['angles_l']['ankle'] = np.vstack(ankle)
    joint['angles_l']['knee'] = np.vstack(knee)
    joint['angles_l']['hip'] = np.vstack(hip)

    ankle = []
    knee = []
    hip = []
    for i in range(np.size(jnt_angles_r)):
        ankle.append(jnt_angles_r[i][0][2, :])
        knee.append(jnt_angles_r[i][0][1, :])
        hip.append(jnt_angles_r[i][0][0, :])

    joint['angles_r']['ankle'] = np.vstack(ankle)
    joint['angles_r']['knee'] = np.vstack(knee)
    joint['angles_r']['hip'] = np.vstack(hip)

    # align with pressure
    for strike_time_r in gait['strike_r']:
        strike_idx_r = np.argmin(np.abs(t_trackers - gait['t_r'][strike_time_r]))
        joint['strike_r'].append(strike_idx_r)

    for off_time_r in gait['off_r']:
        off_idx_r = np.argmin(np.abs(t_trackers - gait['t_r'][off_time_r]))
        joint['off_r'].append(off_idx_r)

    for strike_time_l in gait['strike_l']:
        strike_idx_l = np.argmin(np.abs(t_trackers - gait['t_l'][strike_time_l]))
        joint['strike_l'].append(strike_idx_l)

    for off_time_l in gait['off_l']:
        off_idx_l = np.argmin(np.abs(t_trackers - gait['t_l'][off_time_l]))
        joint['off_l'].append(off_idx_l)


    # resample to 100 points for each step
    for i in range(len(joint['strike_l']) - 1):
        start_idx, end_idx = joint['strike_l'][i], joint['strike_l'][i + 1]
        joint['resampled_angles_l']['ankle'].append(resample_angle(joint['angles_l']['ankle'][start_idx:end_idx,:]))
        joint['resampled_angles_l']['knee'].append(resample_angle(joint['angles_l']['knee'][start_idx:end_idx,:]))
        joint['resampled_angles_l']['hip'].append(resample_angle(joint['angles_l']['hip'][start_idx:end_idx, :]))
    joint['resampled_angles_l']['ankle'] = np.vstack(joint['resampled_angles_l']['ankle'])
    joint['resampled_angles_l']['knee'] = np.vstack(joint['resampled_angles_l']['knee'])
    joint['resampled_angles_l']['hip'] = np.vstack(joint['resampled_angles_l']['hip'])

    # resample to 100 points for each step
    for i in range(len(joint['strike_r']) - 1):
        start_idx, end_idx = joint['strike_r'][i], joint['strike_r'][i + 1]
        joint['resampled_angles_r']['ankle'].append(resample_angle(joint['angles_r']['ankle'][start_idx:end_idx,:]))
        joint['resampled_angles_r']['knee'].append(resample_angle(joint['angles_r']['knee'][start_idx:end_idx,:]))
        joint['resampled_angles_r']['hip'].append(resample_angle(joint['angles_r']['hip'][start_idx:end_idx, :]))
    joint['resampled_angles_r']['ankle'] = np.vstack(joint['resampled_angles_r']['ankle'])
    joint['resampled_angles_r']['knee'] = np.vstack(joint['resampled_angles_r']['knee'])
    joint['resampled_angles_r']['hip'] = np.vstack(joint['resampled_angles_r']['hip'])

    return joint



