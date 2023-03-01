import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

path_silvan = 'Z:/grodriguez/CardiacOCT/metrics_build/summary_silvan.json'
path_gonzalo = 'Z:/grodriguez/CardiacOCT/predicted_results_model2_2d/summary.json' 
annots = pd.read_excel('Z:/grodriguez/CardiacOCT/data-original/train_test_split_dataset2.xlsx')

with open(path_silvan, 'r') as f1:
    silvan_json = json.load(f1)

with open(path_gonzalo, 'r') as f2:
    gonzalo_json = json.load(f2)


final_dict = {}

def get_dice(dict_frame):

    result = {}
    for label, metrics in dict_frame.items():
        result[label] = metrics['Dice']

    return result

for sub_dict_1 in gonzalo_json['results']['all']:

    file = sub_dict_1['test'].split('/')[-1]

    #Take patient name
    patient_id_raw = file.split('_')[0]
    first_part = patient_id_raw[:3]
    second_part = patient_id_raw[3:-4]
    third_part = patient_id_raw[-4:]  
    patient_name = '{}-{}-{}'.format(first_part, second_part, third_part)

    n_pullback = file.split('_')[1]
    pullback_name = annots[(annots['Nº pullback'] == int(n_pullback)) & (annots['Patient'] == patient_name)]['Pullback'].values[0]

    n_frame = int(file.split('_')[2][5:])+1

    fullname = '{}_frame_{}_0.nii.gz'.format(pullback_name, str(n_frame))

    for sub_dict_2 in silvan_json['results']['all']:

        if fullname == sub_dict_2['test'].split('/')[-1]:
   
            dices_gonzalo = get_dice({k: v for i, (k, v) in enumerate(sub_dict_1.items()) if i < len(sub_dict_1) - 2})
            dices_silvan = get_dice({k: v for i, (k, v) in enumerate(sub_dict_2.items()) if i < len(sub_dict_2) - 2})

            final_dict[fullname] = [dices_gonzalo]
            final_dict[fullname].append(dices_silvan)
        else:
            continue

with open('./comparisons.json', 'w') as f3:
    json.dump(final_dict, f3, indent=4)