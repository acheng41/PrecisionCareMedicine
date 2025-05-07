Markdown
# anatomical_foot_segmentation.ipynb

## Overview

This Python script is designed for processing and analyzing insole pressure data collected during walking. It loads data from `.mat` files, performs gait event detection (heel-strikes and toe-offs), segments the foot into anatomical regions, visualizes pressure patterns, and exports segmented data to CSV files.

## Features

* **Data Loading & Preprocessing**: Imports insole pressure data and timestamps for left and right feet from Matlab `.mat` files.
* **Gait Event Detection**: Identifies heel-strike (HC) and toe-off (TO) events using pressure signal analysis.
* **Foot Mask Generation**: Creates an aggregated binary mask representing the foot's outline during each step.
* **Multi-Region Foot Segmentation**:
    * Initially divides the foot into 3 regions: forefoot, midfoot, and hindfoot.
    * Subsequently, refines segmentation into 6 regions: fore-lateral, fore-medial, mid-lateral, mid-medial, hind-lateral, and hind-medial.
* **Data Visualization**:
    * Displays average pressure distribution across insoles.
    * Animates raw pressure data for selected steps.
    * Overlays aggregated foot outlines and region-dividing lines on average pressure images.
    * Generates color-coded images showing pressure distribution in the segmented foot regions.
* **Data Export**: Saves segmented pressure data (timestamp, sensor location, pressure value) into CSV files for detailed offline analysis.

## Input Data

### 1. Matlab Data File (`.mat`)

The script primarily requires a Matlab data file (`.mat`) containing insole pressure recordings.
* **File Path (Example)**: The script currently uses a hardcoded path like `./data/021925/gait_recording_021925_walk11.mat`. This will need to be adjusted based on your file location.
* **Required Variables within the `.mat` file**:
    * `insoleAll_l`: A 2D NumPy array `(num_frames_l, num_sensors)` for left insole pressure. `num_sensors` is typically 1024 (for a 64x16 sensor grid).
    * `insoleAll_r`: A 2D NumPy array `(num_frames_r, num_sensors)` for right insole pressure.
    * `t_insole_l`: A 2D NumPy array `(num_frames_l, 1)` containing timestamps for each left insole frame.
    * `t_insole_r`: A 2D NumPy array `(num_frames_r, 1)` containing timestamps for each right insole frame.

### 2. (Optional) CSV files for visualization

For some later visualization steps (e.g., `build_first_step_color_image` functions), the script reads CSV files that it generated in prior steps. If running these parts in isolation, these CSVs become direct inputs.

## Outputs

### 1. Visualizations (Displayed via Matplotlib)

* **Average Pressure Images**: Static plots of time-averaged pressure for left and right insoles.
* **Step Animation**: Animation of pressure changes during the first valid detected step for each foot.
* **Aggregated Foot Outlines**: Average pressure images overlaid with the computed foot outline for the first step.
* **3-Region Dividing Lines**: Visualization of lines used to segment the foot into forefoot, midfoot, and hindfoot.
* **Color-Coded 3-Region Images**: Images of the first step where forefoot (Red), midfoot (Green), and hindfoot (Blue) are color-coded, with brightness indicating pressure.
* **6-Region Dividing Lines**: Visualization of lines (including a dynamic medial-lateral line) used for 6-region segmentation.
* **Color-Coded 6-Region Images**: Images of the first step with six distinct colors for fore-lateral, fore-medial, mid-lateral, mid-medial, hind-lateral, and hind-medial regions. Brightness indicates pressure. The top half of these images is flipped vertically, and the right foot image is also flipped horizontally for consistent anatomical display.

### 2. CSV Data Files

Segmented pressure data is saved for each foot region. Each CSV file includes columns: `timestamp`, `row`, `col`, `pressure`.

* **Example Output Paths for 3-Region Segmentation**:
    * `data_by_section/GAIT080624-01/walk_4/joint/forefoot_left.csv`
    * `data_by_section/GAIT080624-01/walk_4/joint/midfoot_left.csv`
    * `data_by_section/GAIT080624-01/walk_4/joint/hindfoot_left.csv`
    * *(and corresponding files for the right foot)*

* **Example Output Paths for 6-Region Segmentation**:
    * `data_by_section/021925/walk_11/six/forelat_left.csv`
    * `data_by_section/021925/walk_11/six/foremed_left.csv`
    * `data_by_section/021925/walk_11/six/midlat_left.csv`
    * `data_by_section/021925/walk_11/six/midmed_left.csv`
    * `data_by_section/021925/walk_11/six/hindlat_left.csv`
    * `data_by_section/021925/walk_11/six/hindmed_left.csv`
    * *(and corresponding files for the right foot)*

    *Note: These output paths are hardcoded in the script and may need to be modified.*

## Workflow / How it Works

1.  **Initialization**: Loads data from the specified `.mat` file. Computes and displays initial average pressure images.
2.  **Gait Event Segmentation**:
    * Extracts pressure data specifically from heel and toe/forefoot regions.
    * Applies a Butterworth low-pass filter to these regional pressure signals.
    * Calculates the time derivatives of the filtered signals.
    * Identifies heel-strikes as significant peaks in the heel pressure derivative.
    * Identifies toe-offs as significant peaks in the negative derivative of the toe/forefoot pressure.
3.  **Foot Mask Aggregation**: For each identified step, the script generates foot masks for frames around the step's midpoint. These individual masks are combined (logical OR) to form a robust, aggregated outline of the foot for that step.
4.  **Region Definition & Pressure Data Assignment**:
    * **3-Region Segmentation**: The foot is divided into forefoot, midfoot, and hindfoot using two lines. One line is angled, separating forefoot from midfoot. The other is horizontal, separating midfoot from hindfoot. These lines are positioned based on percentages of the total foot length.
    * **6-Region Segmentation**: The initial 3 regions are further divided into medial and lateral sub-regions. This uses a dynamically calculated medial-lateral dividing line, determined row-by-row based on the center of the aggregated foot mask.
    * For every frame within a detected step, pressure data from each sensor falling within the aggregated foot mask is assigned to one of these defined regions.
5.  **Visualization & Data Export**: The script produces various plots and animations to visualize the raw data, intermediate processing steps, and final segmented regions. The detailed, segmented pressure data for each region is then exported to CSV files.

## Tunable Parameters

Several parameters controlling the analysis are defined in the `tunable_params` dictionary at the beginning of the script:

* `'insole_dims'`: `(rows, cols)` tuple defining the sensor grid dimensions (e.g., `(64, 16)`).
* **Foot Mask Extraction Parameters**:
    * `'pad_width'`: Width of padding applied to data edges before filtering/masking.
    * `'sigma'`: Standard deviation for the Gaussian filter (if `apply_gaussian` is `True`).
    * `'morph_size'`: Size of the structuring element for morphological opening/closing operations on the binary foot mask.
    * `'foot_mask_threshold_l'`: Pressure threshold for creating the left foot's binary mask.
    * `'foot_mask_threshold_r'`: Pressure threshold for creating the right foot's binary mask.
    * `'apply_gaussian'`: Boolean; if `True`, Gaussian smoothing is applied before thresholding for mask creation.
* `'aggregation_window'`: Number of frames before and after a step's midpoint used to create the aggregated foot mask.
* **Butterworth Low-pass Filter Parameters (for gait event detection)**:
    * `'lowpass_cutoff'`: Cutoff frequency in Hz.
    * `'fs'`: Sampling frequency of the insole data in Hz.
    * `'filter_order'`: Order of the filter.
* **Gait Event Detection Thresholds**:
    * `'h_th_r'`: Peak height threshold for right foot heel strike detection.
    * `'t_th_r'`: Peak height threshold for right foot toe-off detection (from negative derivative).
    * `'h_th_l'`: Peak height threshold for left foot heel strike detection.
    * `'t_th_l'`: Peak height threshold for left foot toe-off detection.
    * `'strike_th_l'`: Maximum duration (derived from timestamps) to filter out erroneously long "steps" for the left foot.
    * `'strike_th_r'`: Similar to `strike_th_l` for the right foot.
* **Animation Parameters**:
    * `'animation_interval'`: Delay in milliseconds between frames in the step animation.

## Region Division Parameters (Hardcoded Logic)

The following parameters for foot region division are currently hardcoded within the script logic, not in the `tunable_params` dictionary:

* **3-Region Division (Anterior-Posterior)**:
    * The dividing line between forefoot and midfoot is positioned `54%` down from the top (anterior edge) of the foot outline.
    * The dividing line between midfoot and hindfoot is positioned `29%` up from the bottom (posterior edge) of the foot outline.
    * The forefoot/midfoot dividing line is angled at `+15` degrees for the left foot and `-15` degrees for the right foot (relative to the horizontal axis of the image).
* **6-Region Division (Medial-Lateral)**:
    * Uses the same `54%` and `29%` proportional lines for anterior-posterior division.
    * The medial-lateral division is achieved using a dynamic midline calculated for each row of the aggregated foot mask, representing the center of the foot's width at that specific row.
