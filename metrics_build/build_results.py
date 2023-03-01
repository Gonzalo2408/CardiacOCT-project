#!/usr/local/bin/python

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

def merge_frames_into_pullbacks(path_predicted):

    pullbacks_origs = os.listdir(path_predicted)
    pullbacks_origs_set = []
    pullbacks_dict = {}

    #Save into a list the patiend id + n_pullback substring
    for i in range(len(pullbacks_origs)):
        if pullbacks_origs[i].split('_frame')[0] not in pullbacks_origs_set:
            pullbacks_origs_set.append(pullbacks_origs[i].split('_frame')[0])

        else:
            continue

    #Create dict with patient_id as key and list of belonging frames as values
    for i in range(len(pullbacks_origs_set)):
        frames_from_pullback = [frame for frame in pullbacks_origs if pullbacks_origs_set[i] in frame]
        pullbacks_dict[pullbacks_origs_set[i]] = frames_from_pullback

    #Remove last 3 key-value pairs (they are not frames)
    keys = list(pullbacks_dict.keys())[-3:]
    for key in keys:
        pullbacks_dict[key].pop()
        if not pullbacks_dict[key]:
            pullbacks_dict.pop(key)

    return pullbacks_dict

def mean_metrics(list_dicts):

    result = {}
    for d in list_dicts:
        for label, metrics in d.items():
            if label not in result:
                result[label] = [metrics['Dice']]

            else:
                result[label].append(metrics['Dice'])

    for label, dices in result.items():
        result[label] = np.nanmean(dices)
            
    return result


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--new_json_file', type=str)
    args, unknown = parser.parse_known_args(argv)

    annots = pd.read_excel('Z:/grodriguez/CardiacOCT/data-original/train_test_split_dataset2.xlsx')
    
    json_results_file = args.results_folder+'/summary.json'

    #Load raw 
    with open(json_results_file) as f:
        summary = json.load(f)

    if args.mode == 'pullback':

        #Obtain list of frames for each pullback
        pullbacks_dict = merge_frames_into_pullbacks(args.results_folder)

        final_dict = {}

        #Inside each pullback, there is a list of frames, and each frame is a dictionary with the metrics
        for pullback in pullbacks_dict:

            list_dicts_pullback = []

            for frame in pullbacks_dict[pullback]:

                for sub_dict in summary['results']['all']:

                    #Select between these according to the folder you are in!!
                    #if sub_dict['test'] == '/mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/results/nnUNet/2d/Task501_CardiacOCT/nnUNetTrainerV2__nnUNetPlansv2.1/cv_niftis_postprocessed/{}'.format(frame):
                    if sub_dict['test'] == '/mnt/netcache/diag/grodriguez/CardiacOCT/data-2d/results/nnUNet/2d/Task502_CardiacOCT/nnUNetTrainerV2__nnUNetPlansv2.1/cv_niftis_postprocessed/{}'.format(frame):
                    #if sub_dict['test'] == '/mnt/netcache/diag/grodriguez/CardiacOCT/predicted_results_model2_2d/{}'.format(frame):
                    #if sub_dict['test'] == '/mnt/netcache/diag/grodriguez/CardiacOCT/predicted_results/{}'.format(frame):
                        list_dicts_pullback.append({k: v for i, (k, v) in enumerate(sub_dict.items()) if i < len(sub_dict) - 2})

                    else:
                        continue
            
            mean_result = mean_metrics(list_dicts_pullback)
            final_dict[pullback] = mean_result

        #Change name of keys (pullack names) so it matches with the Excel
        for key in final_dict.copy().keys():

            #Take patient name
            key_patient = key.split('_')[0]
            first_part = key_patient[:3]
            second_part = key_patient[3:-4]
            third_part = key_patient[-4:]  
            patient_name = '{}-{}-{}'.format(first_part, second_part, third_part)

            #Take pullback name
            n_pullback = key.split('_')[1]
            pullback_name = annots[(annots['Nº pullback'] == int(n_pullback)) & (annots['Patient'] == patient_name)]['Pullback'].values[0]
            
            final_dict[pullback_name] = final_dict.pop(key)

        #Write final dict in a json file
        with open(args.new_json_file, 'w') as f:
            json.dump(final_dict, f, indent=4)

    elif args.mode == 'frame':

        final_dict = {}

        for file in os.listdir(args.results_folder):

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

                    if sub_dict['test'] == '/mnt/netcache/diag/grodriguez/CardiacOCT/predicted_results/{}'.format(file):
                        list_dicts_per_frame.append({k: v for i, (k, v) in enumerate(sub_dict.items()) if i < len(sub_dict) - 2})
                        break
                    else:
                        continue

                #Include frame
                mean_result = mean_metrics(list_dicts_per_frame)
                mean_result['frame'] = frame

                final_dict[pullback_name] = mean_result


        #Write final dict in a json file
        with open(args.new_json_file, 'w') as f:
            json.dump(final_dict, f, indent=4)

    else:
        raise ValueError('Invalid mode. Please, select a valid one')

if __name__ == '__main__':
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)


    