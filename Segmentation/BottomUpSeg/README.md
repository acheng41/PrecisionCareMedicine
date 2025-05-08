# Andrew

Main file is clustering.py which invokes functions from clustering_utils.py to segment the insole heatmap using the bottom-up approach. We take the mean across columns and pick out the valleys to separate the toe, ball of foot, and heel regions. The Savitzky-Golay filter and median filter are used to smooth and remove outliers from the signal. The rowwise mean and valleys are used for further segmentation along the medial-lateral axis. Note that I usually run this file line by line rather than in one shot.

visualization.py provides functions for plotting heatmaps of the insole and animations of walking.

roemmich.py is an early attempt to organize some of the data processed by Dr. Roemmich's lab for us.
convert_tables_to_struct.m is part of the preprocessing of Dr. Roemmich's data, since scipy can't process MATLAB tables. These files weren't used in the final analyses.