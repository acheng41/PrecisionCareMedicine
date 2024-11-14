# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:50:57 2024

@author: jackz
"""

import os
from pathlib import Path
import pandas as pd

class DataManager:
    def __init__(self):
        
        # Define Data path
        self.data_path = Path('data/')
        
        # Load meta data
        self.ID_list = self.list_gait_ID()
        self.meta_data_dict = self.load_meta_data()
        


    def list_gait_ID(self):
        """
        Lists all folders in the specified data folder that start with 'GAIT'.
    
        Parameters:
        data_folder (str): Path to the directory containing folders to be filtered.
    
        Returns:
        list: A list of folder names that start with 'GAIT'.
        """
        data_folder = self.data_path
        
        gait_folders = [
            folder_name for folder_name in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, folder_name)) and folder_name.startswith("GAIT")
        ]
        return gait_folders
    
    def load_meta_data(self):
        meta_data_dict = {}
        
        for ID in self.ID_list:
            meta_df = pd.read_excel(self.data_path.joinpath(f'{ID}/meta_data.xlsx'))
            meta_data_dict[ID] = meta_df
            
        return meta_data_dict
    
    def get_valid_data_file_dict(self, ID_list = None):
        if ID_list is None:
            ID_list = self.ID_list
        

        valid_data_info_list = [] # A list of dictionary

        for ID in ID_list:

            meta_data = self.meta_data_dict[ID]
            
            data_files = os.listdir(self.data_path.joinpath(ID))

            
            for file in data_files:
                # Skip Meta Data
                if 'meta_data' in file:
                    continue
                
                data_dict = {}
                
                # First check meta data if this is a valid dataset for use
                trial_num = file.split('walk')[1].split('.')[0]
                data_dict['meta_data'] = meta_data[meta_data['Trial'] == int(trial_num)].reset_index(drop = True).to_dict()
                
                if data_dict['meta_data']['Notes'][0] == 'JUNK DATA. DO NOT USE!':
                    continue
                

                data_dict['path'] = self.data_path.joinpath(f'{ID}/{file}')
                data_dict['ID'] = ID
                
                
                valid_data_info_list.append(data_dict)
            
        return valid_data_info_list
