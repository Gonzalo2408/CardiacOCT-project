import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import os
import sys
sys.path.append("..") 


counts_dir = r'Z:\grodriguez\CardiacOCT\info-files\counts'

counts_files = os.listdir(counts_dir)
counts_files = [file for file in counts_files if file.endswith('.xlsx')]

lipid_dict = {}
calcium_dict = {}

for file in counts_files:

    model = file.split('.')[0].split('_')[2]

    df = pd.read_excel(os.path.join(counts_dir, file))
    df = df.sort_values(['pullback', 'frame'], ascending=[True, True])

    if len(lipid_dict) == 0:
        lipid_dict['pullback'] = df['pullback']
        lipid_dict['frame'] = df['frame']

    
    if len(calcium_dict) == 0:
        calcium_dict['pullback'] = df['pullback']
        calcium_dict['frame'] = df['frame']


    lipid_dict['FCT {}'.format(model)] = df['cap_thickness']
    lipid_dict['Lipid arc {}'.format(model)] = df['lipid arc']
    calcium_dict['Depth {}'.format(model)] = df['calcium_depth']
    calcium_dict['Arc {}'.format(model)] = df['calcium_arc']
    calcium_dict['Thickness {}'.format(model)] = df['calcium_thickness']
    

lipid_measures_df = pd.DataFrame(data=lipid_dict)
lipid_measures_df = lipid_measures_df.reindex(sorted(lipid_measures_df.columns), axis=1)
lipid_measures_df.to_excel(r'Z:\grodriguez\CardiacOCT\info-files\statistics\second_split\model_rgb_script_measures_with_cal.xlsx', sheet_name='Lipid')


calcium_measures_df = pd.DataFrame(data=calcium_dict)
calcium_measures_df = calcium_measures_df.reindex(sorted(calcium_measures_df.columns), axis=1)

with pd.ExcelWriter(r'Z:\grodriguez\CardiacOCT\info-files\statistics\second_split\model_rgb_script_measures_with_cal.xlsx', engine='openpyxl', mode='a') as writer:  
    calcium_measures_df.to_excel(writer, sheet_name='Calcium')





