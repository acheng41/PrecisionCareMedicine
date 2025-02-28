import pandas as pd
import numpy as np
from pathlib import Path
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from skimage.draw import ellipse
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage import measure
from DataManager import DataManager
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy.signal import find_peaks
from scipy import stats


def plot_insole_grid(grid):
    fig, ax = plt.subplots(figsize=(4, 8))
    im = ax.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Pixel intensity')  # Attach colorbar to the axis
    ax.set_title('64x16 Array Pixel Representation')
    plt.close()  # Prevents duplicate display in some environments
    return fig, ax

def plot_region_on_fig(fig, ax, center_x, center_y, width, height, angle, color='r'):
    assert len(center_x) == len(center_y) == len(width) == len(height), "Input lists must have the same length."

    for x, y, w, h, a in zip(center_x, center_y, width, height, angle):
        ellipse = Ellipse((x, y), w, h, angle=a, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(ellipse)

    return fig, ax

def reshape_insole_grid(data):
    data = data.reshape(16,64).T.copy()
    data[:32,] = np.flipud(data[:32,])
    return data

def pixels_within_ellipses(center_x, center_y, width_list, height_list, angle_list):
    all_pixels = []

    for cx, cy, width, height, angle in zip(center_x, center_y, width_list, height_list, angle_list):
        # Convert angle to radians
        theta = np.radians(angle)

        # Define the bounding box of the ellipse
        x_min = int(cx - width / 2)
        x_max = int(cx + width / 2)
        y_min = int(cy - height / 2)
        y_max = int(cy + height / 2)

        # Generate all pixel coordinates in the bounding box
        x_range = np.arange(x_min, x_max + 1)
        y_range = np.arange(y_min, y_max + 1)
        xx, yy = np.meshgrid(x_range, y_range)
        coordinates = np.column_stack((xx.ravel(), yy.ravel()))

        # Rotate and normalize coordinates to ellipse's local coordinate system
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_prime = (coordinates[:, 0] - cx) * cos_theta + (coordinates[:, 1] - cy) * sin_theta
        y_prime = -(coordinates[:, 0] - cx) * sin_theta + (coordinates[:, 1] - cy) * cos_theta

        # Check if points are within the ellipse
        ellipse_mask = ((x_prime**2) / (width / 2)**2 + (y_prime**2) / (height / 2)**2) <= 1

        # Get valid pixels within the ellipse
        valid_pixels = coordinates[ellipse_mask]
        all_pixels.append(valid_pixels.tolist())

    return all_pixels

def visualize_pixels(pixels):
    img_shape = (64,16)
    plt.figure(figsize=(6, 10))
    canvas = np.zeros(img_shape)

    # Plot each ellipse region
    for i, region in enumerate(pixels):
        for x, y in region:
            if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                canvas[y, x] = i + 1  # Assign a unique value to each ellipse

    plt.imshow(canvas, cmap="tab10", origin="lower")
    plt.colorbar(label="Ellipse Region")
    plt.title("Pixels within Ellipse Regions")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()
    plt.show()

def segment_insole_data(arr, height_seg_cords, width_seg_cords):
    arr = np.array([reshape_insole_grid(arr[i]) for i in range(arr.shape[0])])
    segmented_dict = {}
    
    height_seg_cords = [0] + height_seg_cords + [arr.shape[1]]  # Add start and end
    
    for i in range(len(height_seg_cords) - 1):
        
        sub_width_seg_cords = [0] + width_seg_cords[i] + [arr.shape[2]]  # Add start and end
        for j in range(len(sub_width_seg_cords) - 1):
            h_start, h_end = height_seg_cords[i], height_seg_cords[i + 1]
            w_start, w_end = sub_width_seg_cords[j], sub_width_seg_cords[j + 1]
            
            segmented_dict[f'({h_start}-{h_end},{w_start}-{w_end})'] = arr[:, h_start:h_end, w_start:w_end]
    
    return segmented_dict

def calculate_peak_impulse(section_data, threshold_ratio = None):
    if threshold_ratio is None:
        threshold_ratio = 0.7

    sum_array = section_data.sum(axis = 0)
    activation_threshold = 0.05 * np.max(sum_array)
    activation_masks = sum_array >= activation_threshold
    section_data = section_data[:,activation_masks]
    
    total_impulse = np.sum(section_data, axis = 1)
    peaks_index,_ = find_peaks(total_impulse)
    peaks_value = total_impulse[peaks_index]
    max_peaks_value = np.max(peaks_value)
    threshold = max_peaks_value*threshold_ratio
    peaks_value = peaks_value[peaks_value >= threshold]
    max_peak = np.mean(peaks_value)/activation_masks.sum()
    return max_peak

def calculate_region_impulse(section_data, sample_frequency):
    sum_array = section_data.sum(axis = 0)
    activation_threshold = 0.05 * np.max(sum_array)
    activation_masks = sum_array >= activation_threshold
    section_data = section_data[:,activation_masks]

    total_impulse = np.sum(section_data, axis = 1)/activation_masks.sum()
    time_length = np.arange(total_impulse.shape[0])/sample_frequency
    region_impulse = np.trapz(total_impulse, time_length)
    return region_impulse

def calculate_relative_region_impulse(segmented_dict):
    RI_list = []
    name_list = []
    for region in segmented_dict:
        section_data = segmented_dict[region]
        region_impulse = calculate_region_impulse(section_data, sample_frequency = 100)
        RI_list.append(region_impulse)
        name_list.append(region)
    df = pd.DataFrame({'Region':name_list, 'Regional Impulse': RI_list})
    df['Relative Regional Impulse'] = df['Regional Impulse']/df['Regional Impulse'].sum()
    return df

def anova_test(df, cat = None, num = None):
    categories = df[cat].unique()
    groups = [df[df[cat]==name][num] for name in categories]
    f_stat, p_value = stats.f_oneway(*groups)
    return pd.DataFrame({'F-statistics': f_stat,'p-value':p_value}, index = [0])