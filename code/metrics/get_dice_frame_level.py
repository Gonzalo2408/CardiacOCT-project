import pandas as pd
import json
import os
import sys
sys.path.append("..") 
from utils.metrics_utils import mean_metrics


json_file_name = "model_rgb_2d"
folder = "model_rgb_2d_preds"
annots = pd.read_excel('Z:/grodriguez/CardiacOCT/info-files/train_test_split_final_v2.xlsx')
test_folder = "Z:/grodriguez/CardiacOCT/preds_second_split/{}".format(folder)


json_results_file = os.path.join(test_folder, 'summary.json')

#Load raw
with open(json_results_file) as f:
    summary = json.load(f)

final_dict = {}

for file in os.listdir(test_folder):

    if file.endswith('.nii.gz'):

        list_dicts_per_frame = []

        #Get patient name
        patient_name_raw = file.split('_')[0]
        first_part = patient_name_raw[:3]
        second_part = patient_name_raw[3:-4]
        third_part = patient_name_raw[-4:]
        patient_name = '{}-{}-{}'.format(first_part, second_part, third_part)

        #Get pullback_name
        n_pullback = file.split('_')[1]
        pullback_name = annots[(annots['Nº pullback'] == int(n_pullback)) & (annots['Patient'] == patient_name)]['Pullback'].values[0]

        #Get frame
        frame = file.split('_')[2][5:]

        #Get DICE score from frame
        for sub_dict in summary['results']['all']:
            
            if sub_dict['test'] == '/mnt/netcache/diag/grodriguez/CardiacOCT/preds_second_split/{}/{}'.format(folder, file):
                list_dicts_per_frame.append({k: v for i, (k, v) in enumerate(sub_dict.items()) if i < len(sub_dict) - 2})
                break
            else:
                continue

        #Include frame
        mean_result = mean_metrics(list_dicts_per_frame)
        mean_result['frame'] = frame
        mean_result['pullback'] = pullback_name

        final_dict[file] = mean_result

#Write final dict in a json file
with open("Z:/grodriguez/CardiacOCT/info-files/metrics/second_split/{}.json".format(json_file_name), 'w') as f:
    json.dump(final_dict, f, indent=4)